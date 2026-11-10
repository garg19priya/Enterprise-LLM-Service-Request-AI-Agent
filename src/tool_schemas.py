"""Tool contracts: what each tool accepts, as JSON Schema.

The same contracts are:
  * sent to the model (OpenAI format for the open model, function declarations for Gemini);
  * used by the agent to validate every tool argument BEFORE the tool runs.

Change a contract here and both the model and the validator see the change.
"""
from jsonschema import Draft202012Validator

from src import config
from src.config import CATEGORIES, URGENCIES
from src.extension import EXTENSION_SPEC

TOOL_SPECS = [
    {
        "name": "read_status",
        "description": (
            "Read the current status of one IT or facilities service, including its state "
            "(operational, degraded, outage, maintenance or unknown), last incident and sensor summary. "
            "Pass the service name exactly as the user wrote it; the tool understands common names and "
            "aliases itself. Never replace a name you do not recognise with a different service: if the "
            "tool does not know the name, it returns an unknown_service error, and that is the correct result."),
        "parameters": {
            "type": "object",
            "properties": {
                "service": {"type": "string",
                            "description": "The service name as the user wrote it, e.g. 'outlook' or 'shared drive'."},
            },
            "required": ["service"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_knowledge",
        "description": ("Search the knowledge base of how-to guides and policies. Returns matching note ids, "
                        "titles and short snippets. Use for how-to, policy and escalation questions."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A few keywords, e.g. 'reset vpn token'."},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 5,
                          "description": "Number of notes to return (1 to 5). Default 3."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "record_proposed_action",
        "description": ("Record an operational action (for example a restart, an escalation to a team or a new incident) "
                        "in the action log. A human must approve it first; if approval is denied, do not try again. "
                        "Do not use it for how-to, policy or information questions: answer those from the "
                        "knowledge base and record nothing."),
        "parameters": {
            "type": "object",
            "properties": {
                "request_id": {"type": "string", "description": "The id of the request, e.g. 'R02'."},
                "action": {"type": "string", "description": "The action, in a short imperative sentence."},
                "reason": {"type": "string", "description": "Why the action is needed, citing the evidence."},
            },
            "required": ["request_id", "action", "reason"],
            "additionalProperties": False,
        },
    },
]

TOOL_NAMES = [spec["name"] for spec in TOOL_SPECS] + [EXTENSION_SPEC["name"]]


def active_specs() -> list:
    """The contracts offered to the model: the three core tools, plus sensor_trend
    when config.EXTENSION_TOOL_ON is True."""
    return TOOL_SPECS + ([EXTENSION_SPEC] if config.EXTENSION_TOOL_ON else [])

# The structured final answer every run must end with.
FINAL_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "request_id": {"type": "string"},
        "category": {"type": "string", "enum": CATEGORIES},
        "urgency": {"type": "string", "enum": URGENCIES},
        "summary": {"type": "string", "minLength": 1},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "next_step": {"type": "string", "minLength": 1},
        "action_recorded": {"type": "boolean"},
        "needs_human": {"type": "boolean"},
    },
    "required": ["request_id", "category", "urgency", "summary", "evidence",
                 "next_step", "action_recorded", "needs_human"],
}

_ARG_VALIDATORS = {spec["name"]: Draft202012Validator(spec["parameters"]) for spec in TOOL_SPECS + [EXTENSION_SPEC]}
_FINAL_VALIDATOR = Draft202012Validator(FINAL_ANSWER_SCHEMA)


def _messages(validator, instance):
    problems = []
    for err in sorted(validator.iter_errors(instance), key=lambda e: list(e.path)):
        where = ".".join(str(p) for p in err.path) or "(top level)"
        problems.append(f"{where}: {err.message}")
    return problems


def validate_arguments(tool_name: str, arguments) -> list:
    """Return a list of problems with the arguments; an empty list means valid."""
    if tool_name not in _ARG_VALIDATORS:
        return [f"unknown tool '{tool_name}'. Valid tools: {', '.join(TOOL_NAMES)}"]
    if not isinstance(arguments, dict):
        return ["arguments must be a JSON object"]
    return _messages(_ARG_VALIDATORS[tool_name], arguments)


def validate_final_answer(answer) -> list:
    if not isinstance(answer, dict):
        return ["final answer must be a JSON object"]
    return _messages(_FINAL_VALIDATOR, answer)


def _model_schema(parameters: dict) -> dict:
    """The schema sent to the model. 'additionalProperties' is left out because not every
    endpoint accepts it; the agent still enforces it locally with validate_arguments."""
    return {k: v for k, v in parameters.items() if k != "additionalProperties"}


def openai_tools() -> list:
    """Contracts in the OpenAI Chat Completions 'tools' format."""
    return [{"type": "function",
             "function": {"name": s["name"], "description": s["description"],
                          "parameters": _model_schema(s["parameters"])}}
            for s in active_specs()]


def gemini_function_declarations():
    """Contracts as Gemini function declarations (google-genai SDK)."""
    from google.genai import types
    return [types.FunctionDeclaration(name=s["name"], description=s["description"],
                                      parameters_json_schema=_model_schema(s["parameters"])) for s in active_specs()]
