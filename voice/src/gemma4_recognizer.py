import logging
import time
from typing import TYPE_CHECKING, Optional
import json
import re
from typing import Any, Dict, List, Tuple

if TYPE_CHECKING:
    from llama_cpp import Llama


TOOL_CALL_RE = re.compile(
    r"<\|tool_call>call:([a-zA-Z0-9_]+)\{(.*?)\}<tool_call\|>",
    re.DOTALL,
)


DEFAULT_REPO = "ggml-org/gemma-4-E2B-it-GGUF"
DEFAULT_FILENAME = "gemma-4-E2B-it-Q8_0.gguf"
DEFAULT_PROMPT = 'Call tools for this sentence: "{text}"'

_LOGGER = logging.getLogger(__name__)


class Gemma4Recognizer:
    def __init__(
        self,
        repo_id: str = DEFAULT_REPO,
        filename: str = DEFAULT_FILENAME,
        prompt: str = DEFAULT_PROMPT,
        debug: bool = False,
    ) -> None:
        self.llm: Optional[Llama] = None
        self.repo_id = repo_id
        self.filename = filename
        self.prompt = prompt
        self.n_ctx = 2048
        self.max_tokens = 64
        self.temperature = 0.0
        self.top_p = 1.0
        self.enable_thinking = False
        self.tool_choice: str = "auto"
        self.debug = debug

    def load(self) -> None:
        if self.llm:
            # Already loaded
            return

        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
        )
        _LOGGER.debug("Loading gemma4: %s", model_path)
        self.llm = Llama(
            model_path=model_path,
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
            n_ctx=self.n_ctx,
            verbose=self.debug,
        )

    def get_tool_calls(
        self, text: str, tools: List[Dict[str, Any]]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        assert self.llm, "Not loaded"

        start_time = time.monotonic()
        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": self.prompt.format(text=text),
                }
            ],
            tools=tools,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            tool_choice=self.tool_choice,
        )
        end_time = time.monotonic()
        _LOGGER.debug("Response in %s second(s): %s", end_time - start_time, response)

        content = response["choices"][0]["message"]["content"]
        return _parse_tool_calls(content)

    @staticmethod
    def is_available() -> bool:
        try:
            from llama_cpp import Llama
            from huggingface_hub import hf_hub_download

            return True

        except ImportError:
            return False


# -----------------------------------------------------------------------------


def _parse_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    text = _normalize_gemma_tool_text(text)
    calls = []

    for match in TOOL_CALL_RE.finditer(text):
        name = match.group(1)
        raw_args = match.group(2).strip()
        args = {}

        if raw_args:
            for part in _split_args(raw_args):
                key, value = part.split(":", 1)
                args[key.strip()] = _parse_value(value)

        calls.append((name, args))

    return calls


def _normalize_gemma_tool_text(text: str) -> str:
    return text.replace('<|"|>', '"').replace("<|'|>", "'")


def _parse_value(value: str):
    value = _normalize_gemma_tool_text(value.strip())

    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]

    if re.fullmatch(r"-?\d+", value):
        return int(value)

    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    return value


def _split_args(raw_args: str) -> List[str]:
    parts = []
    buf = []
    in_string = False
    escape = False

    for ch in raw_args:
        if escape:
            buf.append(ch)
            escape = False
            continue

        if ch == "\\":
            buf.append(ch)
            escape = True
            continue

        if ch == '"':
            buf.append(ch)
            in_string = not in_string
            continue

        if ch == "," and not in_string:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf.clear()
            continue

        buf.append(ch)

    part = "".join(buf).strip()
    if part:
        parts.append(part)

    return parts
