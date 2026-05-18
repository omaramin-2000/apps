import hashlib
import json
import logging
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

TOOL_CALL_RE = re.compile(
    r"<\|tool_call>call:([a-zA-Z0-9_]+)\{(.*?)\}<tool_call\|>",
    re.DOTALL,
)


DEFAULT_REPO = "ggml-org/gemma-4-E2B-it-GGUF"
DEFAULT_FILENAME = "gemma-4-E2B-it-Q8_0.gguf"
DEFAULT_SYSTEM_PROMPT = "Call tools for the following sentence."
DEFAULT_USER_PROMPT = 'Sentence: "{text}"'

_LOGGER = logging.getLogger(__name__)


class Gemma4Recognizer:
    def __init__(
        self,
        state_path: Union[str, Path],
        repo_id: str = DEFAULT_REPO,
        filename: str = DEFAULT_FILENAME,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str = DEFAULT_USER_PROMPT,
        debug: bool = False,
    ) -> None:
        self.llm: Optional[Llama] = None
        self.state_path = Path(state_path)
        self.repo_id = repo_id
        self.filename = filename
        self.user_prompt = user_prompt
        self.n_ctx = 2048
        self.max_tokens = 64
        self.temperature = 0.0
        self.top_p = 1.0
        self.enable_thinking = False
        self.tool_choice: str = "auto"
        self.tools: Optional[List[Dict[str, Any]]] = None
        self.debug = debug

        self.system_message = {
            "role": "system",
            "content": system_prompt,
        }

    def load(self, tools: List[Dict[str, Any]]) -> None:
        self.tools = tools
        if self.llm is None:
            try:
                model_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    local_files_only=True,
                )
            except OSError:
                model_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    local_files_only=False,
                )
            _LOGGER.debug("Loading gemma4: %s", model_path)
            self.llm = Llama(
                model_path=model_path,
                chat_template_kwargs={"enable_thinking": self.enable_thinking},
                n_ctx=self.n_ctx,
                verbose=self.debug,
            )

        actual_tools_hash = _get_tools_hash(tools)
        state_metadata_path = self.state_path.with_suffix(".sha256")
        rebuild_state = True
        if state_metadata_path.exists() and self.state_path.exists():
            expected_tools_hash = state_metadata_path.read_text(
                encoding="utf-8"
            ).strip()
            if expected_tools_hash == actual_tools_hash:
                _LOGGER.debug("Cache hit. Loading state: %s", self.state_path)
                with open(self.state_path, "rb") as state_file:
                    state = pickle.load(state_file)

                self.llm.load_state(state)
                rebuild_state = False

        if rebuild_state:
            _LOGGER.debug("Cache miss. Rebuilding state")
            self.llm.create_chat_completion(
                messages=[self.system_message],  # type: ignore
                tools=self.tools,  # type: ignore
                max_tokens=0,
            )
            _LOGGER.debug("Saving state: %s", self.state_path)
            with open(self.state_path, "wb") as state_file:
                state = self.llm.save_state()
                pickle.dump(state, state_file)

            state_metadata_path.write_text(actual_tools_hash, encoding="utf-8")

    def get_tool_calls(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        assert self.llm, "Not loaded"

        start_time = time.monotonic()
        response = cast(
            Dict[str, Any],
            self.llm.create_chat_completion(
                messages=[
                    self.system_message,  # type: ignore
                    {
                        "role": "user",
                        "content": self.user_prompt.format(text=text),
                    },
                ],
                tools=self.tools,  # type: ignore
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                tool_choice=self.tool_choice,  # type: ignore
            ),
        )
        end_time = time.monotonic()
        _LOGGER.debug("Response in %s second(s): %s", end_time - start_time, response)

        content = response["choices"][0]["message"]["content"]
        return _parse_tool_calls(content)


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


def _get_tools_hash(tools: List[Dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(tools).encode()).hexdigest()
