import time
import logging
from collections.abc import Mapping
from typing import Optional, Any, Dict, Set, Tuple, Union, List

from hass_api import InfoForRecognition
from models import Area, Floor
from wyoming.error import Error
from wyoming.event import Event
from wyoming.server import AsyncEventHandler
from wyoming.info import (
    Attribution,
    Describe,
    Info,
    IntentProgram,
    IntentModel,
)
from wyoming.asr import Transcript
from wyoming.intent import Intent, Entity, NotRecognized, IntentsStart, IntentsStop
from jinja2 import BaseLoader
from jinja2.nativetypes import NativeEnvironment
from rapidfuzz import process, fuzz

from const import AppState, Tool, ToolIntent
from gemma4_recognizer import Gemma4Recognizer
from lang_intents import LanguageIntents
from name_resolver import NameResolver
from util import normalize_name

_LOGGER = logging.getLogger(__name__)

TOOL_ARGS = Dict[str, Any]
TOOL_CALL = Tuple[str, TOOL_ARGS]

_FUZZY_SCORE_CUTOFF = 95
_ENV = NativeEnvironment(loader=BaseLoader())


class UnresolvedNameError(Exception):
    pass


class Gemma4EventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        state: AppState,
        recognizer: Gemma4Recognizer,
        lang_intents: LanguageIntents,
        name_resolver: NameResolver,
        *args,
        **kwargs,
    ) -> None:
        """Initialize event handler."""
        super().__init__(*args, **kwargs)

        self.client_id = str(time.monotonic_ns())
        self.state = state
        self.recognizer = recognizer
        self.lang_intents = lang_intents
        self.name_resolver = name_resolver

        self._info_event: Optional[Event] = None

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming event."""
        try:
            return await self._handle_event(event)
        except Exception:
            _LOGGER.exception("Error handling event")

        return True

    async def _handle_event(self, event: Event) -> bool:
        """Handle Wyoming event."""
        if Describe.is_type(event.type):
            await self._write_info()
            return True

        if Transcript.is_type(event.type):
            transcript = Transcript.from_event(event)
            _LOGGER.debug("Handling: %s", transcript)

            device_id: Optional[str] = None
            satellite_id: Optional[str] = None
            if transcript.context:
                device_id = transcript.context.get("device_id")
                satellite_id = transcript.context.get("satellite_id")

            try:
                hass_info = await self.state.hass.get_info(device_id, satellite_id)
                await self._handle_transcript(transcript, hass_info)
            except Exception:
                _LOGGER.exception("Unexpected error during handling")
                await self.write_event(
                    Error(
                        text="Unexpected error during handling", code="handle-error"
                    ).event()
                )

            return True

        return True

    async def _handle_transcript(
        self, transcript: Transcript, hass_info: InfoForRecognition
    ) -> None:
        tools_list = [t.tool for t in self.state.tools.values()]
        tool_calls = self.recognizer.get_tool_calls(transcript.text, tools_list)
        if not tool_calls:
            await self.write_event(
                NotRecognized(
                    text=self.lang_intents.get_error_response(
                        language=transcript.language or "en", key="no_intent"
                    ),
                    context=transcript.context,
                ).event()
            )
            return

        intent_events: List[Intent] = []

        for tool_name, tool_args in tool_calls:
            tool = self.state.tools[tool_name]

            try:
                self._resolve_names(tool, tool_args, hass_info)
            except UnresolvedNameError:
                _LOGGER.exception(
                    "Failed to resolve names: tool_name=%s, tool_args=%s",
                    tool_name,
                    tool_args,
                )
                # Fail
                intent_events.clear()
                break

            if tool.intent:
                intent_name, intent_slots = tool_call_to_intent(tool.intent, tool_args)
            else:
                intent_name, intent_slots = tool_name, tool_args

            _LOGGER.debug("Intent: name=%s, slots=%s", intent_name, intent_slots)
            intent_events.append(
                Intent(
                    name=intent_name,
                    entities=[
                        Entity(name=name, value=value)
                        for name, value in intent_slots.items()
                    ],
                    text=self.lang_intents.get_intent_response(
                        language=transcript.language or "en",
                        intent_name=intent_name,
                    ),
                    context=transcript.context,
                )
            )

        if intent_events:
            await self.write_event(IntentsStart(context=transcript.context).event())
            for intent in intent_events:
                await self.write_event(intent.event())
            await self.write_event(IntentsStop(context=transcript.context).event())
        else:
            # TODO: give a better error message for unresolved names
            await self.write_event(
                NotRecognized(
                    text=self.lang_intents.get_error_response(
                        language=transcript.language or "en", key="handle_error"
                    ),
                    context=transcript.context,
                ).event()
            )

    def _resolve_names(
        self, tool: Tool, tool_args: TOOL_ARGS, hass_info: InfoForRecognition
    ) -> None:
        location: Optional[str] = tool_args.get("location")
        if location:
            self._resolve_location(location, tool_args, hass_info)

        entity_name: Optional[str] = tool_args.get("device_name") or tool_args.get(
            "list_name"
        )
        if entity_name:
            self._resolve_entity(entity_name, tool.name_domains, tool_args, hass_info)

    def _resolve_location(
        self, location: str, tool_args: TOOL_ARGS, hass_info: InfoForRecognition
    ) -> None:
        location_norm = normalize_name(location)
        location_names: Dict[str, str] = {}
        location_names_norm: Dict[str, str] = {}

        for area in hass_info.areas.values():
            # Ensure area/floor ids are deconflicted
            mapped_id = f"area_{area.area_id}"
            for area_name in area.names:
                location_names[area_name] = mapped_id
            for area_name_norm in area.names_norm:
                location_names_norm[area_name_norm] = mapped_id
        for floor in hass_info.floors.values():
            # Ensure area/floor ids are deconflicted
            mapped_id = f"floor_{floor.floor_id}"
            location_names[floor.name] = mapped_id
            location_names_norm[floor.name_norm] = mapped_id

        best_id: Optional[str] = location_names.get(location)
        if not best_id:
            best_id = location_names_norm.get(location_norm)

        if not best_id:
            best_name = self.name_resolver.best_candidate(
                location, list(location_names)
            )
            if best_name:
                best_id = location_names[best_name]

        if not best_id:
            # Fuzzy match
            result = process.extractOne(
                location_norm,
                list(location_names_norm),
                scorer=fuzz.ratio,
                score_cutoff=_FUZZY_SCORE_CUTOFF,
            )
            if result:
                # Map back to original name
                best_name_norm = result[0]
                best_id = location_names_norm[best_name_norm]

        if best_id:
            location_type, location_id = best_id.split("_", maxsplit=1)
            if location_type == "area":
                best_area = hass_info.areas[location_id]
                _LOGGER.debug("Resolved %s to %s", location, best_area)
                tool_args["area"] = best_area
            elif location_type == "floor":
                best_floor = hass_info.floors[location_id]
                _LOGGER.debug("Resolved %s to %s", location, best_floor)
                tool_args["floor"] = best_floor
        else:
            raise UnresolvedNameError(f"Unable to resolve location: {location}")

    def _resolve_entity(
        self,
        entity_name: str,
        name_domains: Optional[Set[str]],
        tool_args: TOOL_ARGS,
        hass_info: InfoForRecognition,
    ) -> None:
        entity_norm = normalize_name(entity_name)
        entity_names: Dict[str, str] = {}
        entity_names_norm: Dict[str, str] = {}

        entities = hass_info.entities.values()
        if name_domains:
            entities = [e for e in entities if e.domain in name_domains]

        for entity in entities:
            for name in entity.names:
                entity_names[name] = entity.entity_id
            for entity_name_norm in entity.names_norm:
                entity_names_norm[entity_name_norm] = entity.entity_id

        best_id: Optional[str] = entity_names.get(entity_name)
        if not best_id:
            best_id = entity_names_norm.get(entity_norm)

        if not best_id:
            best_name = self.name_resolver.best_candidate(
                entity_name, list(entity_names)
            )
            if best_name:
                best_id = entity_names[best_name]

        if not best_id:
            # Fuzzy match
            result = process.extractOne(
                entity_norm,
                list(entity_names_norm),
                scorer=fuzz.ratio,
                score_cutoff=_FUZZY_SCORE_CUTOFF,
            )
            if result:
                # Map back to original name
                best_name_norm = result[0]
                best_id = entity_names_norm[best_name_norm]
                _LOGGER.debug("Fuzzy match: %s -> %s", entity_name, best_name_norm)

        if best_id:
            best_entity = hass_info.entities[best_id]
            _LOGGER.debug("Resolved %s to %s", entity_name, best_entity)
            tool_args["entity"] = best_entity
        else:
            raise UnresolvedNameError(f"Unable to resolve entity: {entity_name}")

    async def _write_info(self) -> None:
        if self._info_event is not None:
            await self.write_event(self._info_event)
            return

        info = Info(
            intent=[
                IntentProgram(
                    name="intent-gemma4",
                    attribution=Attribution(
                        "Open Home Foundation Voice", "https://github.com/OHF-Voice"
                    ),
                    installed=True,
                    description="Gemma 4 Intent Recognizer",
                    version="0.0.1",
                    models=[
                        IntentModel(
                            name="gemma4",
                            attribution=Attribution(
                                "Google DeepMind",
                                "https://deepmind.google/models/gemma/gemma-4/",
                            ),
                            installed=True,
                            description="gemma4",
                            version="",
                            languages=[],  # all languages
                        )
                    ],
                )
            ],
        )

        self._info_event = info.event()
        await self.write_event(self._info_event)


# -----------------------------------------------------------------------------


def tool_call_to_intent(tool_intent: ToolIntent, tool_args: TOOL_ARGS) -> TOOL_CALL:
    intent_name = tool_intent.name
    if is_template_string(intent_name):
        intent_name = render_template(intent_name, tool_args) or ""

    intent_name = intent_name.strip()
    if not intent_name:
        raise ValueError("No intent name")

    intent_slots: TOOL_ARGS = {}
    if tool_intent.slots:
        intent_slots = render_templates_recursive(tool_intent.slots, tool_args) or {}

    # Remove empty values
    if intent_slots:
        intent_slots = {
            key: value for key, value in intent_slots.items() if value is not None
        }

    return (intent_name, intent_slots)


def render_templates_recursive(data: Any, variables: Mapping[str, Any]) -> Any:
    # Template string handling
    if isinstance(data, str) and is_template_string(data):
        return render_template(data, variables)

    # Mapping (dict-like)
    if isinstance(data, Mapping):
        return {k: render_templates_recursive(v, variables) for k, v in data.items()}

    # Sequence (but not str/bytes)
    if isinstance(data, (list, tuple)):
        rendered = [render_templates_recursive(v, variables) for v in data]
        return rendered if isinstance(data, list) else tuple(rendered)

    return data


def render_template(data: str, variables: Mapping[str, Any]) -> Any:
    return _ENV.from_string(data).render(variables)


def is_template_string(maybe_template: str) -> bool:
    """Check if the input is a Jinja2 template."""
    return "{" in maybe_template and (
        "{%" in maybe_template or "{{" in maybe_template or "{#" in maybe_template
    )


# def tool_call_to_intent(tool_name: str, tool_args: TOOL_ARGS) -> TOOL_CALL:
#     if tool_name == "start_timer":
#         return _start_timer(tool_args)

#     if tool_name == "control_timer":
#         return _control_timer(tool_args)

#     if tool_name == "lights_on_off":
#         return _lights_on_off(tool_args)

# if tool_name == "control_media":
#     media_action = tool_args.get("action")
#     if media_action == "pause":
#         return ("HassMediaPause", {})

#     if media_action == "resume":
#         return ("HassMediaUnpause", {})

#     if media_action == "next":
#         return ("HassMediaNext", {})

#     raise ValueError(f"Unexpected action for control_media: {media_action}")

# return (tool_name, tool_args)


def _start_timer(tool_args: TOOL_ARGS) -> TOOL_CALL:
    total_seconds = tool_args.get("total_seconds", 0)
    timer_args: Dict[str, Any] = {}
    hours = total_seconds // 3600
    if hours > 0:
        timer_args["hours"] = hours
    minutes = (total_seconds % 3600) // 60
    if minutes > 0:
        timer_args["minutes"] = minutes
    seconds = total_seconds % 60
    if seconds > 0:
        timer_args["seconds"] = seconds

    timer_name = tool_args.get("name")
    if timer_name:
        timer_args["name"] = timer_name

    return ("HassStartTimer", timer_args)


def _control_timer(tool_args: TOOL_ARGS) -> TOOL_CALL:
    action = tool_args.get("action")
    if action == "cancel_all":
        return ("HassCancelAllTimers", {})

    timer_args: Dict[str, Any] = {}
    total_seconds = tool_args.get("total_seconds")

    if total_seconds is not None:
        hours = total_seconds // 3600
        if hours > 0:
            timer_args["start_hours"] = hours
        minutes = (total_seconds % 3600) // 60
        if minutes > 0:
            timer_args["start_minutes"] = minutes
        seconds = total_seconds % 60
        if seconds > 0:
            timer_args["start_seconds"] = seconds

    timer_name = tool_args.get("name")
    if timer_name:
        timer_args["name"] = timer_name

    if action == "pause":
        return ("HassPauseTimer", timer_args)

    if action == "resume":
        return ("HassUnpauseTimer", timer_args)

    if action == "cancel":
        return ("HassCancelTimer", timer_args)

    raise ValueError(f"Unexpected action for control_timer: {action}")


def _lights_on_off(tool_args: TOOL_ARGS) -> TOOL_CALL:
    action = tool_args.get("action")
    light_args = {"domain": "light"}

    if action == "on":
        return ("HassTurnOn", light_args)

    if action == "off":
        return ("HassTurnOff", light_args)

    raise ValueError(f"Unexpected action for lights_on_off: {action}")
