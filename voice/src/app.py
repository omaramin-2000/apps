#!/usr/bin/env python3

import argparse
import asyncio
import logging
from functools import partial
from typing import Dict, List

from ruamel.yaml import YAML
from wyoming.server import AsyncServer

from const import BASE_DIR, AppState, FuzzyCommand, Tool, ToolIntent
from gemma4_recognizer import Gemma4Recognizer
from hass_api import HomeAssistant
from intent_server import Gemma4EventHandler
from lang_intents import LanguageIntents
from name_resolver import NameResolver
from fuzzy_matcher import FuzzyMatcher

_LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------


async def main() -> None:
    """Run app."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    #
    parser.add_argument("--http-host")
    parser.add_argument("--http-port", type=int, default=5000)
    #
    parser.add_argument("--hass-token", required=True)
    parser.add_argument("--hass-api", default="http://homeassistant.local:8123")
    parser.add_argument(
        "--default-area-id", help="Area id to use if no context area is available"
    )
    #
    parser.add_argument("--tools", required=True, help="Path to tools YAML file")
    parser.add_argument(
        "--llama-state", required=True, help="Path to save llama.cpp state"
    )
    #
    parser.add_argument(
        "--fuzzy-commands", required=True, help="Path to fuzzy commands YAML file"
    )
    #
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    # Load tools
    yaml = YAML(typ="safe")
    with open(args.tools, "r", encoding="utf-8") as tools_file:
        yaml_tools = yaml.load(tools_file)

    tools: Dict[str, Tool] = {}
    for tool_dict in yaml_tools:
        tool_name = tool_dict["tool"]["function"]["name"]
        tool = Tool(
            tool=tool_dict["tool"], context_area=tool_dict.get("context_area", False)
        )
        tool_intent_dict = tool_dict.get("intent")
        if tool_intent_dict:
            tool.intent = ToolIntent(
                name=tool_intent_dict["name"], slots=tool_intent_dict.get("slots")
            )
        tool_name_domains = tool_dict.get("name_domains")
        if tool_name_domains:
            tool.name_domains = set(tool_name_domains)
        tools[tool_name] = tool

    _LOGGER.debug("Loaded %s tool(s)", len(tools))

    # Load fuzzy commands
    with open(args.fuzzy_commands, "r", encoding="utf-8") as fuzzy_commands_file:
        yaml_commands = yaml.load(fuzzy_commands_file)

    fuzzy_commands: List[FuzzyCommand] = []
    for command_dict in yaml_commands:
        fuzzy_commands.append(
            FuzzyCommand(
                intent_name=command_dict["intent"]["name"],
                sentences=command_dict["sentences"],
                intent_slots=command_dict["intent"].get("slots"),
                context_area=command_dict.get("context_area"),
                duration=command_dict.get("duration"),
                number=command_dict.get("number"),
            )
        )

    _LOGGER.debug("Loaded %s fuzzy command(s)", len(fuzzy_commands))

    state = AppState(
        hass=HomeAssistant(token=args.hass_token, api_url=args.hass_api),
        http_host=args.http_host,
        http_port=args.http_port,
        tools=tools,
        fuzzy_commands=fuzzy_commands,
        fuzzy_candidates=[
            (s, i) for i, cmd in enumerate(fuzzy_commands) for s in cmd.sentences
        ],
        default_area_id=args.default_area_id,
    )

    recognizer = Gemma4Recognizer(state_path=args.llama_state)
    recognizer.load([t.tool for t in tools.values()])

    lang_intents = LanguageIntents()

    name_resolver = NameResolver()
    name_resolver.load()

    fuzzy_matcher = FuzzyMatcher()
    fuzzy_matcher.model = name_resolver.model
    # fuzzy_matcher.load()
    fuzzy_matcher.train(s for s, _ in state.fuzzy_candidates)

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")

    try:
        await server.run(
            partial(
                Gemma4EventHandler,
                state,
                recognizer,
                lang_intents,
                name_resolver,
                fuzzy_matcher,
            )
        )
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
