from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set, List, Tuple

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
    context_area: bool = False


@dataclass
class FuzzyCommand:
    intent_name: str
    sentences: List[str]
    intent_slots: Optional[Dict[str, Any]] = None
    context_area: bool = False
    duration: bool = False
    number: bool = False


@dataclass
class AppState:
    hass: HomeAssistant
    http_host: str
    http_port: int
    tools: Dict[str, Tool]
    fuzzy_commands: List[FuzzyCommand]
    fuzzy_candidates: List[Tuple[str, int]]
    default_area_id: Optional[str] = None
