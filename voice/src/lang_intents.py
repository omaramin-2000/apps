import re
from typing import Any, Dict, Optional

from home_assistant_intents import get_intents


class LanguageIntents:
    def __init__(self) -> None:
        self.lang_intents: Dict[str, Any] = {}

    def get_lang_intents(self, language: str) -> Dict[str, Any]:
        if language not in self.lang_intents:
            original_language = language
            lang_intents = get_intents(language)
            if not lang_intents:
                # Normalize and try again
                lang_parts = re.split(r"[-_]", language, maxsplit=1)
                lang_family: Optional[str] = None
                if len(lang_parts) == 1:
                    language = language.lower()
                else:
                    lang_family = lang_parts[0]
                    assert lang_family
                    language = lang_family.lower() + "-" + language[0].upper()

                lang_intents = get_intents(language)
                if (not lang_intents) and lang_family:
                    lang_intents = get_intents(lang_family)

            if lang_intents:
                self.lang_intents[language] = lang_intents
                self.lang_intents[original_language] = lang_intents

        return self.lang_intents.get(language, {})

    def get_intent_response(
        self,
        language: str,
        intent_name: str,
        intent_slots: Optional[Dict[str, Any]],
        default_key: str = "default",
    ) -> Optional[str]:
        responses = (
            self.get_lang_intents(language)
            .get("responses", {})
            .get("intents", {})
            .get(intent_name, {})
        )
        if len(responses) == 1:
            return next(iter(responses.values()))

        key: Optional[str] = None
        if intent_slots is None:
            intent_slots = {}

        domain = intent_slots.get("domain")

        if intent_name in ("HassTurnOn", "HassTurnOff"):
            if domain in ("cover", "value"):
                key = domain
            elif "area" in intent_slots:
                if domain == "light":
                    key = "lights_area"
            elif "floor" in intent_slots:
                if domain == "light":
                    key = "lights_floor"
        elif intent_name == "HassLightSet":
            if "brightness" in intent_slots:
                key = "brightness"
            elif "color" in intent_slots:
                key = "color"

        key = key or default_key
        return responses.get(key)

    def get_error_response(self, language: str, key: str) -> Optional[str]:
        return (
            self.get_lang_intents(language)
            .get("responses", {})
            .get("errors", {})
            .get(key)
        )
