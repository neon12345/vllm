# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tool call parser for Google Gemma4 models.

Gemma4 uses a custom serialization format (not JSON) for tool calls::

    <|tool_call>call:func_name{key:<|"|>value<|"|>,num:42}<tool_call|>

Strings are delimited by ``<|"|>`` (token 52), keys are unquoted, and
multiple tool calls are concatenated without separators.

Used when ``--enable-auto-tool-choice --tool-call-parser gemma4`` are set.

For offline inference tool call parsing (direct ``tokenizer.decode()`` output),
see ``vllm.tool_parsers.gemma4_utils.parse_tool_calls``.
"""

import json
from collections.abc import Sequence

from collections import deque
import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
from vllm.tool_parsers.utils import find_common_prefix

logger = init_logger(__name__)

# Gemma4 special tokens for tool calls
TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"
STRING_DELIM = '<|"|>'
CHANNEL_CLOSE = "<channel|>"  # channel transition token, never content


# ---------------------------------------------------------------------------
# Gemma4 argument parser (used by both streaming and non-streaming paths)
# ---------------------------------------------------------------------------


def _parse_gemma4_value(value_str: str) -> object:
    """Parse a single Gemma4 value (after key:) into a Python object."""
    value_str = value_str.strip()
    if not value_str:
        return value_str

    # Boolean
    if value_str == "true":
        return True
    if value_str == "false":
        return False

    # Null
    if value_str.lower() in ("null", "none", "nil"):
        return None

    # Number (int or float)
    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass

    # Bare string (no <|"|> delimiters — shouldn't happen but be safe)
    return value_str


def _parse_gemma4_args(args_str: str, *, partial: bool = False) -> dict:
    """Parse Gemma4's custom key:value format into a Python dict.

    Format examples::

        location:<|"|>Tokyo<|"|>
        location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>
        count:42,flag:true
        nested:{inner_key:<|"|>val<|"|>}
        items:[<|"|>a<|"|>,<|"|>b<|"|>]

    Args:
        args_str: The raw Gemma4 argument string.
        partial: When True (streaming), bare values at end of string are
            omitted because they may be incomplete and type-unstable
            (e.g. partial boolean parsed as bare string).

    Returns a dict ready for ``json.dumps()``.
    """
    if not args_str or not args_str.strip():
        return {}

    result: dict = {}
    i = 0
    n = len(args_str)

    while i < n:
        # Skip whitespace and commas
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        # Parse key (unquoted, ends at ':')
        key_start = i
        while i < n and args_str[i] != ":":
            i += 1
        if i >= n:
            break
        key = args_str[key_start:i].strip()
        i += 1  # skip ':'

        # Parse value
        if i >= n:
            if not partial:
                result[key] = ""
            break

        # Skip whitespace after ':'
        while i < n and args_str[i] in (" ", "\n", "\t"):
            i += 1
        if i >= n:
            if not partial:
                result[key] = ""
            break

        # String value: <|"|>...<|"|>
        if args_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            val_start = i
            end_pos = args_str.find(STRING_DELIM, i)
            if end_pos == -1:
                # Unterminated string — take rest
                result[key] = args_str[val_start:]
                break
            result[key] = args_str[val_start:end_pos]
            i = end_pos + len(STRING_DELIM)

        # Nested object: {...}
        elif args_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    # Skip over string contents to avoid counting { inside strings
                    i += len(STRING_DELIM)
                    next_delim = args_str.find(STRING_DELIM, i)
                    i = n if next_delim == -1 else next_delim + len(STRING_DELIM)
                    continue
                if args_str[i] == "{":
                    depth += 1
                elif args_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                # Incomplete nested object — use i (not i-1) to avoid
                # dropping the last char, and recurse as partial.
                result[key] = _parse_gemma4_args(args_str[obj_start:i], partial=True)
            else:
                result[key] = _parse_gemma4_args(args_str[obj_start : i - 1])

        # Array: [...]
        elif args_str[i] == "[":
            depth = 1
            arr_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    next_delim = args_str.find(STRING_DELIM, i)
                    i = n if next_delim == -1 else next_delim + len(STRING_DELIM)
                    continue
                if args_str[i] == "[":
                    depth += 1
                elif args_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                result[key] = _parse_gemma4_array(args_str[arr_start:i], partial=True)
            else:
                result[key] = _parse_gemma4_array(args_str[arr_start : i - 1])

        # Bare value (number, boolean, etc.)
        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                i += 1
            if partial and i >= n:
                # Value may be incomplete (e.g. partial boolean) —
                # withhold to avoid type instability during streaming.
                break
            result[key] = _parse_gemma4_value(args_str[val_start:i])

    return result


def _parse_gemma4_array(arr_str: str, *, partial: bool = False) -> list:
    """Parse a Gemma4 array content string into a Python list."""
    items: list = []
    i = 0
    n = len(arr_str)

    while i < n:
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        # String element
        if arr_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            end_pos = arr_str.find(STRING_DELIM, i)
            if end_pos == -1:
                items.append(arr_str[i:])
                break
            items.append(arr_str[i:end_pos])
            i = end_pos + len(STRING_DELIM)

        # Nested object
        elif arr_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    nd = arr_str.find(STRING_DELIM, i)
                    i = nd + len(STRING_DELIM) if nd != -1 else n
                    continue
                if arr_str[i] == "{":
                    depth += 1
                elif arr_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_gemma4_args(arr_str[obj_start:i], partial=True))
            else:
                items.append(_parse_gemma4_args(arr_str[obj_start : i - 1]))

        # Nested array
        elif arr_str[i] == "[":
            depth = 1
            sub_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i] == "[":
                    depth += 1
                elif arr_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_gemma4_array(arr_str[sub_start:i], partial=True))
            else:
                items.append(_parse_gemma4_array(arr_str[sub_start : i - 1]))

        # Bare value
        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                i += 1
            if partial and i >= n:
                break
            items.append(_parse_gemma4_value(arr_str[val_start:i]))

    return items


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class Gemma4ToolParser(ToolParser):
    """
    Tool call parser for Google Gemma4 models.

    Handles the Gemma4 function call format::

        <|tool_call>call:func_name{key:<|"|>value<|"|>}<tool_call|>

    Used when ``--enable-auto-tool-choice --tool-call-parser gemma4``
    are set.

    Streaming strategy: **accumulate-then-parse-then-diff**

    Instead of trying to convert Gemma4's custom format to JSON
    token-by-token (which fails because Gemma4 uses bare keys, custom
    delimiters, and structural braces that differ from JSON), this parser:

    1. Accumulates the raw Gemma4 argument string during streaming
    2. Parses it with ``_parse_gemma4_args()`` into a Python dict
    3. Converts to JSON with ``json.dumps()``
    4. Diffs against the previously-streamed JSON string
    5. Emits only the new JSON fragment as the delta

    This follows the same pattern used by FunctionGemma, Hermes, and Llama
    tool parsers.
    """

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        # Token strings
        self.tool_call_start_token = TOOL_CALL_START
        self.tool_call_end_token = TOOL_CALL_END
        self.string_delim_token = STRING_DELIM
        self.channel_close_token = CHANNEL_CLOSE
        # Token IDs
        self.tool_call_start_token_id = self.vocab.get(TOOL_CALL_START)
        self.tool_call_end_token_id = self.vocab.get(TOOL_CALL_END)
        self.string_delim_token_id = self.vocab.get(STRING_DELIM)
        self.channel_close_token_id = self.vocab.get(CHANNEL_CLOSE)
        checks = {
            self.tool_call_start_token: self.tool_call_start_token_id,
            self.tool_call_end_token: self.tool_call_end_token_id,
            self.string_delim_token: self.string_delim_token_id,
            self.channel_close_token: self.channel_close_token_id,
        }
        self.special_tokens = set(checks.values())
        self.special_token_items = checks.items()

        for token, token_id in self.special_token_items:
            if token_id is None:
                raise RuntimeError(
                    f"Gemma4 ToolParser could not locate the {token} token in the tokenizer!"
                )

        # Regex for non-streaming: extract complete tool calls.
        # Supports function names with letters, digits, underscores,
        # hyphens, and dots (e.g. "get-weather", "module.func").
        self.tool_call_regex = re.compile(
            r"<\|tool_call>call:([\w\-\.]+)\{(.*?)\}<tool_call\|>",
            re.DOTALL,
        )

        # Queue and mapping for special token detection
        self.token_queue = deque()

        # Streaming state — reset per-request via _reset_streaming_state()
        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state for a new request."""
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []
        self.tool_calls = []
        self.raw_buffer = "" # reset here, not only in __init__
        self.tool_buffer = "" # reset here, not only in __init__
        self.content_buffer = "" # reset here, not only in __init__
        self.in_tool_call = False

        self.token_queue.clear()

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Don't skip special tokens — <|tool_call> etc. are needed for
            # the parser to detect tool calls. Apply to BOTH
            # ChatCompletionRequest and ResponsesRequest (the previous
            # isinstance(ChatCompletionRequest) guard caused tool-call
            # delimiters to be stripped on /v1/responses, leaking raw
            # `call:fn{...}` text via output_text.delta).
            request.skip_special_tokens = False
        return request

    # ------------------------------------------------------------------
    # Non-streaming extraction
    # ------------------------------------------------------------------

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            matches = self.tool_call_regex.findall(model_output)
            if not matches:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tool_calls: list[ToolCall] = []
            for func_name, args_str in matches:
                arguments = _parse_gemma4_args(args_str)
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=func_name,
                            arguments=json.dumps(arguments, ensure_ascii=False),
                        ),
                    )
                )

            # Content = text before first tool call (if any)
            content_end = model_output.find(self.tool_call_start_token)
            content = model_output[:content_end].strip() if content_end > 0 else None

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error extracting tool calls from Gemma4 response")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    # ------------------------------------------------------------------
    # Delta buffering for multi-token special sequences
    # ------------------------------------------------------------------

    def _update_buffer(self, delta_text: str = ""):
        if self.token_queue:    # we are expectiong a token
            self.raw_buffer += delta_text
        else:
            combined = self.raw_buffer + delta_text
            self.raw_buffer = ""
            if self.in_tool_call: # we are in a tool call
                self.tool_buffer += combined
            else:                   # raw content
                self.content_buffer += combined

    def _scan_tokens(self, delta_token_ids):
        for tid in delta_token_ids:
            if tid in self.special_tokens:
                self.token_queue.append(tid)

    def _process_queue(self):
        """
        Converts token_queue + raw_buffer state into events.
        Does NOT mutate tool state directly.
        """

        def has(token: str) -> bool:
            return token in self.raw_buffer

        def add(token: str):
            if self.in_tool_call:
                self.tool_buffer += token
            else:
                self.content_buffer += token

        if self.token_queue:
            for token, token_id in self.special_token_items:
                if self.token_queue[0] == token_id and has(token):
                    self.token_queue.popleft()
                    before, _, after = self.raw_buffer.partition(token)
                    self.raw_buffer = after
                    add(before)

                    # TOOL START EVENT
                    if token_id == self.tool_call_start_token_id:
                        self._on_tool_start()

                    # TOOL END EVENT
                    if token_id == self.tool_call_end_token_id:
                        self._on_tool_end()

                    if token_id == self.string_delim_token_id:
                        add(token)


                    return False
        return True

    def _on_tool_start(self):
        self.in_tool_call = True

        self.current_tool_id += 1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool.append("")
        self.prev_tool_call_arr.append({})
        logger.debug("Starting new tool call %d", self.current_tool_id)

        self.tool_buffer = ""

    def _on_tool_end(self):

        self._handle_tool_call_end()
        self.in_tool_call = False

    # ------------------------------------------------------------------
    # Streaming extraction — accumulate-then-parse-then-diff
    # ------------------------------------------------------------------

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if not previous_token_ids:
            self._reset_streaming_state()

        if not delta_token_ids:
            self.raw_buffer += delta_text
            return None

        def result() -> DeltaMessage | None:
            content = self.content_buffer
            if not content:
                content = None
                if not self.tool_calls:
                    return None
            out = DeltaMessage(content=content, tool_calls=self.tool_calls)
            self.content_buffer = ""
            self.tool_calls = []
            return out

        self._scan_tokens(delta_token_ids)
        self._update_buffer(delta_text)

        while True:

            if self._process_queue():
                break

            self._update_buffer()

            if self.in_tool_call:
                self._handle_tool_call_middle()

        return result()

    def _extract_partial_call(self) -> tuple[str | None, str]:
        """Extract function name and raw argument string from partial text.

        Returns (func_name, raw_args_str) or (None, "") if not parseable yet.
        """
        # Get the text after the last <|tool_call> token
        if not self.tool_buffer:
            return None, ""

        partial_call = self.tool_buffer

        # Expect "call:name{args...}" or "call:name{args...}"
        if not partial_call.startswith("call:"):
            return None, ""

        func_part = partial_call[5:]  # skip "call:"

        if "{" not in func_part:
            # Still accumulating function name, not ready yet
            return None, ""

        func_name, _, args_part = func_part.partition("{")
        func_name = func_name.strip()

        # Strip trailing '}' if present (Gemma4 structural brace)
        if args_part.endswith("}"):
            args_part = args_part[:-1]

        return func_name, args_part

    def _handle_tool_call_middle(self):
        """Handle streaming when we're inside an active tool call.

        Accumulates the raw Gemma4 arguments, parses them into JSON, and
        diffs against the previously-streamed JSON to emit only the new
        fragment.
        """
        func_name, args_part = self._extract_partial_call()

        if func_name is None:
            return

        # Step 1: Send function name (once)
        if not self.current_tool_name_sent and func_name:
            self.current_tool_name_sent = True
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": func_name,
                "arguments": {},
            }
            self.tool_calls.append(
                    DeltaToolCall(
                        index=self.current_tool_id,
                        type="function",
                        id=make_tool_call_id(),
                        function=DeltaFunctionCall(
                            name=func_name,
                            arguments="",
                        ).model_dump(exclude_none=True),
                    )
            )

        # Step 2: Parse and diff arguments
        if self.current_tool_name_sent and args_part:
            self._emit_argument_diff(args_part)

    def _handle_tool_call_end(self):
        """Handle streaming when a tool call has just completed.

        Performs a final parse of the complete tool call and flushes
        any remaining un-streamed argument fragments.
        """
        if self.current_tool_id < 0 or self.current_tool_id >= len(
            self.prev_tool_call_arr
        ):
            logger.debug(
                "Tool call end detected but no active tool call (current_tool_id=%d)",
                self.current_tool_id,
            )
            return

        _, args_str = self._extract_partial_call()
        if not args_str:
            return  # No arguments to parse

        # Parse the complete tool call
        final_args = _parse_gemma4_args(args_str)
        final_args_json = json.dumps(final_args, ensure_ascii=False)

        prev_streamed = self.streamed_args_for_tool[self.current_tool_id]
        if len(final_args_json) > len(prev_streamed):
            diff = final_args_json[len(prev_streamed) :]
            self.streamed_args_for_tool[self.current_tool_id] = final_args_json
            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = final_args
            self.tool_calls.append(
                DeltaToolCall(
                    index=self.current_tool_id,
                    function=DeltaFunctionCall(arguments=diff).model_dump(
                        exclude_none=True
                    ),
                )
            )

    def _emit_argument_diff(self, raw_args_str: str):
        """Parse raw Gemma4 arguments, convert to JSON, diff, and emit.

        This is the core of the accumulate-then-parse-then-diff strategy:
        1. Parse ``raw_args_str`` with ``_parse_gemma4_args()``
        2. Convert to JSON string with ``json.dumps()``
        3. Withhold trailing closing characters (``"}``) that may move
           as more tokens arrive
        4. Diff against previously streamed JSON and emit only new chars

        **Why withholding is necessary:**

        Gemma4's custom format produces *structurally incomplete* JSON
        during streaming. For example, when ``<|"|>Paris`` arrives
        without a closing delimiter, ``_parse_gemma4_args`` treats it
        as a complete value and produces ``{"location": "Paris"}``. But
        when ``, France<|"|>`` arrives next, the JSON becomes
        ``{"location": "Paris, France"}``. If we had sent the closing
        ``"}`` from the first parse, the concatenated client output
        would be ``{"location": "Paris"}France"}``, which is garbage.

        The solution: **never send trailing closing chars during
        streaming**. They get flushed by ``_handle_tool_call_end()``
        when the ``<tool_call|>`` end marker arrives.

        Args:
            raw_args_str: The raw Gemma4 argument text accumulated so far
                (without the surrounding ``{`` ``}``).

        Returns:
            DeltaMessage with the argument diff, or None if no new content.
        """
        try:
            current_args = _parse_gemma4_args(raw_args_str, partial=True)
        except Exception:
            logger.debug(
                "Could not parse partial Gemma4 args yet: %s",
                raw_args_str[:100],
            )
            return

        if not current_args:
            return

        current_args_json = json.dumps(current_args, ensure_ascii=False)

        # Withhold trailing closing characters that may shift as more
        # tokens arrive. Strip trailing '}', '"', ']' and partial
        # STRING_DELIM fragments ('<', '|', '\\', '>') to get the
        # "safe prefix".
        safe_json = current_args_json
        while safe_json and safe_json[-1] in ("}", '"', "]", "<", "|", "\\", ">"):
            safe_json = safe_json[:-1]

        prev_streamed = self.streamed_args_for_tool[self.current_tool_id]

        if not safe_json or safe_json == prev_streamed:
            return

        # Use find_common_prefix to handle cases where the value changed
        # structurally (e.g., a string grew).
        if prev_streamed:
            prefix = find_common_prefix(prev_streamed, safe_json)
            sent_len = len(prev_streamed)
            prefix_len = len(prefix)

            if prefix_len < sent_len:
                # Structure changed — we sent too much. Truncate our
                # tracking to the common prefix and wait for the final
                # flush in _handle_tool_call_end.
                self.streamed_args_for_tool[self.current_tool_id] = prefix
                return

            # Stream the new stable portion
            diff = safe_json[sent_len:]
        else:
            # First emission
            diff = safe_json

        if diff:
            self.streamed_args_for_tool[self.current_tool_id] = safe_json
            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = current_args

            self.tool_calls.append(
                    DeltaToolCall(
                        index=self.current_tool_id,
                        function=DeltaFunctionCall(arguments=diff).model_dump(
                            exclude_none=True
                        ),
                    )
            )
