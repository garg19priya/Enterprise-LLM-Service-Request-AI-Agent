"""Model providers behind one small interface, so the agent loop never changes.

    open           open model served by Ollama in the Colab runtime, called with the
                   OpenAI Python library (the same code would call OpenAI's hosted models
                   by changing base_url, api_key and model)
    gemini         Gemini through Google's google-genai library (native function calling)
    gemini_openai  Gemini through the OpenAI library (fallback when no GPU is available)
    mock           rule-based stand-in, no model at all (offline testing)

Each session keeps its own conversation history in its provider's native format.
"""
import json
import time
import uuid
from dataclasses import dataclass, field

from src import config
from src.tool_schemas import TOOL_NAMES, gemini_function_declarations, openai_tools

PROVIDERS = ["open", "gemini", "gemini_openai", "mock"]


class ProviderError(RuntimeError):
    """The model could not be reached or returned something unusable."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict = None        # parsed arguments, when the provider gives a dict
    arguments_raw: str = None     # raw JSON text, when the provider gives a string
    extra: dict = None            # provider data that must be sent back unchanged (Gemini thought signature)


@dataclass
class ModelTurn:
    text: str = ""
    tool_calls: list = field(default_factory=list)
    latency_s: float = 0.0
    note: str = None              # e.g. "tool call recovered from text"


class ChatSession:
    provider = "base"
    model = ""

    def add_user(self, text: str) -> None:
        raise NotImplementedError

    def next_turn(self) -> ModelTurn:
        raise NotImplementedError

    def add_tool_results(self, results: list) -> None:
        """results: [(ToolCall, result_dict), ...] for the calls of the last turn."""
        raise NotImplementedError


# ----------------------------------------------------------------------------
# OpenAI-compatible: the open model in Ollama, or Gemini's OpenAI endpoint
# ----------------------------------------------------------------------------
def _recover_tool_call(text: str):
    """Small models sometimes write a tool call as plain JSON text. Recover it if so."""
    cleaned = text.strip().strip("`")
    if cleaned.startswith("json"):
        cleaned = cleaned[4:]
    try:
        obj = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(obj, dict) and obj.get("name") in TOOL_NAMES:
        args = obj.get("arguments", obj.get("parameters", {}))
        return ToolCall(id=f"recovered-{uuid.uuid4().hex[:8]}", name=obj["name"],
                        arguments=args if isinstance(args, dict) else None,
                        arguments_raw=args if isinstance(args, str) else None)
    return None


class OpenAICompatSession(ChatSession):
    def __init__(self, system_prompt, use_tools=True, *, base_url, api_key, model, provider, min_interval=0.0):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=config.REQUEST_TIMEOUT_SECONDS, max_retries=1)
        self.model, self.provider = model, provider
        self.tools = openai_tools() if use_tools else None
        self.messages = [{"role": "system", "content": system_prompt}]
        self.min_interval, self._last_call = min_interval, 0.0

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})

    def next_turn(self):
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        kwargs = {"model": self.model, "messages": self.messages, "temperature": config.TEMPERATURE}
        if self.tools:
            kwargs["tools"] = self.tools
        start = time.time()
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # connection refused, 404 model, 429 rate limit, ...
            raise ProviderError(f"{self.provider} call failed: {type(exc).__name__}: {exc}") from exc
        finally:
            self._last_call = time.time()
        latency = time.time() - start
        if not response.choices:
            raise ProviderError(f"{self.provider} returned no choices")
        msg = response.choices[0].message

        calls = []
        for tc in msg.tool_calls or []:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            extra = getattr(tc, "extra_content", None) or (getattr(tc, "model_extra", None) or {}).get("extra_content")
            calls.append(ToolCall(id=tc.id or f"call-{uuid.uuid4().hex[:8]}", name=fn.name,
                                  arguments_raw=fn.arguments, extra=extra))
        text, note = msg.content or "", None
        if not calls and text:
            recovered = _recover_tool_call(text)
            if recovered:
                calls, text, note = [recovered], "", "tool call written as text; recovered by the agent"

        assistant = {"role": "assistant", "content": text}
        if calls:
            assistant["tool_calls"] = [{
                "id": c.id, "type": "function",
                "function": {"name": c.name,
                             "arguments": c.arguments_raw if c.arguments_raw is not None else json.dumps(c.arguments or {})},
                **({"extra_content": c.extra} if c.extra else {})}
                for c in calls]
        self.messages.append(assistant)
        return ModelTurn(text=text, tool_calls=calls, latency_s=latency, note=note)

    def add_tool_results(self, results):
        for call, result in results:
            self.messages.append({"role": "tool", "tool_call_id": call.id,
                                  "content": json.dumps(result, default=str)})


# ----------------------------------------------------------------------------
# Gemini through google-genai (native function calling)
# ----------------------------------------------------------------------------
class GeminiSession(ChatSession):
    provider = "gemini"

    def __init__(self, system_prompt, use_tools=True, *, api_key, model):
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.contents = []
        self.config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=config.TEMPERATURE,
            tools=[types.Tool(function_declarations=gemini_function_declarations())] if use_tools else None,
            # The agent loop runs the tools itself, so the SDK must not call them automatically.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        self._last_call = 0.0

    def add_user(self, text):
        self.contents.append(self.types.Content(role="user", parts=[self.types.Part(text=text)]))

    def _generate(self):
        wait = config.GEMINI_MIN_SECONDS_BETWEEN_CALLS - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            return self.client.models.generate_content(model=self.model, contents=self.contents, config=self.config)
        finally:
            self._last_call = time.time()

    def next_turn(self):
        start = time.time()
        try:
            try:
                response = self._generate()
            except Exception as exc:
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    print("Gemini free-tier rate limit reached; waiting 30 seconds and retrying once...")
                    time.sleep(30)
                    response = self._generate()
                else:
                    raise
        except Exception as exc:
            raise ProviderError(f"gemini call failed: {type(exc).__name__}: {exc}") from exc
        latency = time.time() - start

        candidate = response.candidates[0] if response.candidates else None
        if candidate is None or candidate.content is None:
            reason = getattr(candidate, "finish_reason", None) if candidate else "no candidates"
            raise ProviderError(f"gemini returned an empty response (finish reason: {reason})")
        # Keep the model's turn exactly as returned: it may carry thought signatures
        # that Gemini needs to see again on the next call.
        self.contents.append(candidate.content)

        calls, texts = [], []
        for part in candidate.content.parts or []:
            if part.function_call:
                fc = part.function_call
                calls.append(ToolCall(id=fc.id or f"gemini-{uuid.uuid4().hex[:8]}", name=fc.name,
                                      arguments=dict(fc.args or {})))
            elif part.text and not getattr(part, "thought", False):
                texts.append(part.text)
        return ModelTurn(text="".join(texts), tool_calls=calls, latency_s=latency)

    def add_tool_results(self, results):
        t = self.types
        parts = []
        for call, result in results:
            response = result if isinstance(result, dict) else {"result": result}
            fid = call.id if not call.id.startswith("gemini-") else None
            parts.append(t.Part(function_response=t.FunctionResponse(id=fid, name=call.name, response=response)))
        self.contents.append(t.Content(role="user", parts=parts))


# ----------------------------------------------------------------------------
# Mock (no model)
# ----------------------------------------------------------------------------
class MockSession(ChatSession):
    provider, model = "mock", "rule-based mock (not an LLM)"

    def __init__(self, system_prompt, use_tools=True):
        self.use_tools = use_tools
        self.messages = [{"role": "system", "content": system_prompt}]

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})

    def next_turn(self):
        from src.mock_model import mock_reply
        start = time.time()
        reply = mock_reply(self.messages, tools_enabled=self.use_tools)
        calls = [ToolCall(id=c["id"], name=c["name"], arguments_raw=json.dumps(c["arguments"]))
                 for c in reply["tool_calls"]]
        assistant = {"role": "assistant", "content": reply["content"] or ""}
        if calls:
            assistant["tool_calls"] = [{"id": c.id, "type": "function",
                                        "function": {"name": c.name, "arguments": c.arguments_raw}} for c in calls]
        self.messages.append(assistant)
        return ModelTurn(text=reply["content"] or "", tool_calls=calls, latency_s=time.time() - start)

    def add_tool_results(self, results):
        for call, result in results:
            self.messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)})


# ----------------------------------------------------------------------------
def _gemini_key():
    key = config.get_secret(config.GEMINI_KEY_NAME)
    if not key:
        raise ProviderError("No Gemini API key found. In Colab, open the key icon (Secrets), add "
                            f"{config.GEMINI_KEY_NAME} and switch on notebook access.")
    return key


def make_session(provider: str, system_prompt: str, use_tools: bool = True) -> ChatSession:
    provider = provider.lower()
    if provider == "open":
        return OpenAICompatSession(system_prompt, use_tools, base_url=config.OPEN_MODEL_BASE_URL,
                                   api_key="ollama", model=config.OPEN_MODEL, provider="open")
    if provider == "gemini":
        return GeminiSession(system_prompt, use_tools, api_key=_gemini_key(), model=config.GEMINI_MODEL)
    if provider == "gemini_openai":
        return OpenAICompatSession(system_prompt, use_tools, base_url=config.GEMINI_OPENAI_BASE_URL,
                                   api_key=_gemini_key(), model=config.GEMINI_MODEL, provider="gemini_openai",
                                   min_interval=config.GEMINI_MIN_SECONDS_BETWEEN_CALLS)
    if provider == "mock":
        return MockSession(system_prompt, use_tools)
    raise ValueError(f"Unknown provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}")


def model_name(provider: str) -> str:
    return {"open": config.OPEN_MODEL, "gemini": config.GEMINI_MODEL, "gemini_openai": config.GEMINI_MODEL,
            "mock": MockSession.model}.get(provider, provider)
