from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set

from hass_api import HomeAssistant

BASE_DIR = Path(__file__).parent


@dataclass
class ToolIntent:
    name: str
    slots: Optional[Dict[str, Any]] = None


@dataclass
class Tool:
    tool: Dict[str, Any]
    intent: Optional[ToolIntent] = None
    name_domains: Optional[Set[str]] = None


@dataclass
class AppState:
    hass: HomeAssistant
    http_host: str
    http_port: int
    tools: Dict[str, Tool]
