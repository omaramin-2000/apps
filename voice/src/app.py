#!/usr/bin/env python3

import argparse
import asyncio
import logging
from functools import partial
from typing import Dict

from ruamel.yaml import YAML
from wyoming.server import AsyncServer

from const import BASE_DIR, AppState, Tool, ToolIntent
from gemma4_recognizer import Gemma4Recognizer
from hass_api import HomeAssistant
from intent_server import Gemma4EventHandler
from lang_intents import LanguageIntents
from name_resolver import NameResolver

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
    #
    parser.add_argument(
        "--llama-state", required=True, help="Path to save llama.cpp state"
    )
    #
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    # Load tools
    yaml = YAML()
    tools_path = BASE_DIR / "tools.yaml"
    with open(tools_path, "r", encoding="utf-8") as tools_file:
        yaml_tools = yaml.load(tools_file)

    tools: Dict[str, Tool] = {}
    for tool_dict in yaml_tools:
        tool_name = tool_dict["tool"]["function"]["name"]
        tool = Tool(tool_dict["tool"])
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

    state = AppState(
        hass=HomeAssistant(token=args.hass_token, api_url=args.hass_api),
        http_host=args.http_host,
        http_port=args.http_port,
        tools=tools,
    )

    recognizer = Gemma4Recognizer(state_path=args.llama_state)
    recognizer.load([t.tool for t in tools.values()])

    lang_intents = LanguageIntents()

    name_resolver = NameResolver()
    name_resolver.load()

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")

    try:
        await server.run(
            partial(Gemma4EventHandler, state, recognizer, lang_intents, name_resolver)
        )
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
