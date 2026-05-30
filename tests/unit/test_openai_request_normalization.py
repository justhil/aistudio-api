from aistudio_api.api.schemas import ChatRequest
from aistudio_api.application.chat_service import (
    TOOL_HISTORY_END,
    TOOL_HISTORY_START,
    normalize_chat_request,
    normalize_openai_tools,
)


def _normalize(payload: dict):
    req = ChatRequest(**payload)
    return normalize_chat_request(req.messages, req.model)


def _no_native_function_part(part) -> bool:
    return part.function_call is None and part.function_response is None


def test_assistant_tool_calls_with_null_content_are_dropped_from_model_role():
    # An assistant message that is only a tool call (content empty) carries no
    # model-role text, so it must not appear as a model turn (and must not leak
    # the transcript marker into the model's own output pattern).
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

    assert len(norm["contents"]) == 1
    assert norm["contents"][0].role == "user"
    for content in norm["contents"]:
        assert content.role != "model"


def test_tool_result_transcript_includes_call_args_in_user_role():
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

    tool_content = norm["contents"][-1]
    assert tool_content.role == "user"
    part = tool_content.parts[0]
    assert _no_native_function_part(part)
    assert TOOL_HISTORY_START in part.text
    assert "tool: get_weather" in part.text
    assert '"city": "SF"' in part.text  # call args folded into the result transcript
    assert '{"temp": 18}' in part.text
    assert TOOL_HISTORY_END in part.text


def test_tool_result_name_resolved_from_prior_assistant_tool_calls():
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

    part = norm["contents"][-1].parts[0]
    assert "tool: lookup" in part.text
    assert "plain text result" in part.text


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

    tool_content = norm["contents"][-1]
    assert len(tool_content.parts) == 2
    assert "tool: f1" in tool_content.parts[0].text
    assert "tool: f2" in tool_content.parts[1].text


def test_assistant_text_kept_but_tool_call_not_in_model_role():
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

    model_content = norm["contents"][0]
    assert model_content.role == "model"
    assert len(model_content.parts) == 1
    assert model_content.parts[0].text == "let me check"
    assert TOOL_HISTORY_START not in model_content.parts[0].text


def test_no_native_function_parts_or_model_role_markers_in_tool_round_trip():
    norm = _normalize(
        {
            "model": "gemini-3.5-flash",
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "x", "content": "ok"},
            ],
        }
    )

    for content in norm["contents"]:
        for part in content.parts:
            assert _no_native_function_part(part)
            if content.role == "model":
                assert TOOL_HISTORY_START not in (part.text or "")


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


def test_resolve_openai_thinking_config_supports_multiple_param_styles():
    from aistudio_api.application.chat_service import resolve_openai_thinking_config

    def cfg(**kw):
        req = ChatRequest(model="gemini-3.5-flash", messages=[{"role": "user", "content": "hi"}], **kw)
        return resolve_openai_thinking_config(req)

    # wire = [mode, None, None, level]; ThinkingLevel LOW=1 MEDIUM=2 HIGH=3 MINIMAL=4
    assert cfg(reasoning_effort="high") == [1, None, None, 3]
    assert cfg(reasoning_effort="low") == [1, None, None, 1]
    assert cfg(reasoning_effort="minimal") == [1, None, None, 4]
    assert cfg(thinking="medium") == [1, None, None, 2]
    assert cfg(thinking={"thinkingLevel": "high"}) == [1, None, None, 3]
    assert cfg(thinking={"type": "enabled", "budget_tokens": 20000}) == [1, None, None, 3]
    assert cfg(thinking={"type": "enabled", "budget_tokens": 1000}) == [1, None, None, 1]
    assert cfg(reasoning={"effort": "medium"}) == [1, None, None, 2]
    assert cfg() is None
    assert cfg(reasoning_effort="bogus") is None


def test_openai_responses_emit_reasoning_content_field():
    from aistudio_api.api.responses import chat_completion_response, sse_chunk
    import json

    resp = chat_completion_response(model="m", content="answer", thinking="reasoned")
    msg = resp.choices[0].message.model_dump(exclude_none=True)
    assert msg["reasoning_content"] == "reasoned"
    assert "thinking" not in msg

    delta = json.loads(sse_chunk("id", "m", "", thinking="t").split("data: ", 1)[1])["choices"][0]["delta"]
    assert delta["reasoning_content"] == "t"
    assert "thinking" not in delta
