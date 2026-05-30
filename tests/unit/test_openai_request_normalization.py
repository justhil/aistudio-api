from aistudio_api.api.schemas import ChatRequest
from aistudio_api.application.chat_service import normalize_chat_request, normalize_openai_tools


def _normalize(payload: dict):
    req = ChatRequest(**payload)
    return normalize_chat_request(req.messages, req.model)


def test_assistant_tool_calls_with_null_content_converts_to_function_call_part():
    norm = _normalize(
        {
            "model": "gemini-3.5-flash",
            "messages": [
                {"role": "user", "content": "weather in SF?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
                        }
                    ],
                },
            ],
        }
    )

    model_content = norm["contents"][1]
    assert model_content.role == "model"
    part = model_content.parts[0]
    assert part.function_call == ("get_weather", {"city": "SF"}, "call_1")


def test_tool_role_message_converts_to_function_response_part():
    norm = _normalize(
        {
            "model": "gemini-3.5-flash",
            "messages": [
                {"role": "user", "content": "weather in SF?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 18}'},
            ],
        }
    )

    tool_content = norm["contents"][2]
    assert tool_content.role == "user"
    part = tool_content.parts[0]
    assert part.function_response == ("get_weather", {"temp": 18}, "call_1")


def test_tool_role_name_resolved_from_prior_assistant_tool_calls():
    norm = _normalize(
        {
            "model": "gemini-3.5-flash",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c9", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c9", "content": "plain text result"},
            ],
        }
    )

    part = norm["contents"][1].parts[0]
    assert part.function_response[0] == "lookup"
    assert part.function_response[1] == {"result": "plain text result"}


def test_consecutive_tool_results_merge_into_single_content():
    norm = _normalize(
        {
            "model": "gemini-3.5-flash",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "a", "type": "function", "function": {"name": "f1", "arguments": "{}"}},
                        {"id": "b", "type": "function", "function": {"name": "f2", "arguments": "{}"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "a", "content": "1"},
                {"role": "tool", "tool_call_id": "b", "content": "2"},
            ],
        }
    )

    tool_content = norm["contents"][1]
    assert len(tool_content.parts) == 2
    assert [p.function_response[0] for p in tool_content.parts] == ["f1", "f2"]


def test_assistant_text_and_tool_call_coexist():
    norm = _normalize(
        {
            "model": "gemini-3.5-flash",
            "messages": [
                {
                    "role": "assistant",
                    "content": "let me check",
                    "tool_calls": [
                        {"id": "x", "type": "function", "function": {"name": "f", "arguments": '{"k":1}'}}
                    ],
                },
            ],
        }
    )

    parts = norm["contents"][0].parts
    assert parts[0].text == "let me check"
    assert parts[1].function_call == ("f", {"k": 1}, "x")


def test_invalid_tool_call_arguments_fall_back_to_empty_dict():
    norm = _normalize(
        {
            "model": "gemini-3.5-flash",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "x", "type": "function", "function": {"name": "f", "arguments": "not-json"}}
                    ],
                },
            ],
        }
    )

    assert norm["contents"][0].parts[0].function_call == ("f", {}, "x")


def test_openai_tools_with_ref_and_nullable_union_do_not_crash():
    req = ChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "complex",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "filter": {"$ref": "#/$defs/F"},
                        },
                        "$defs": {"F": {"type": "object", "properties": {"q": {"type": "string"}}}},
                    },
                },
            }
        ],
    )

    wire = normalize_openai_tools(req.tools)
    assert wire is not None
    schema = wire[0][1][0][2]
    properties = {name: value for name, value in schema[6]}
    assert properties["city"] == [1]
    assert properties["filter"][0] == 6
