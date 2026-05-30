from aistudio_api.infrastructure.gateway.session import BrowserSession, _snapshot_content_hash
from aistudio_api.infrastructure.gateway.wire_types import AistudioContent, AistudioPart


def test_resolve_captured_info_preserves_partial_overrides():
    session = BrowserSession.__new__(BrowserSession)
    session._templates = {
        "model": {
            "url": "https://fallback.example/rpc",
            "headers": {
                "Authorization": "Bearer token",
                "Host": "ignored.example",
                "Content-Length": "123",
            },
        }
    }

    url, headers = session._resolve_captured_info("https://override.example/rpc", None)
    assert url == "https://override.example/rpc"
    assert headers == {"Authorization": "Bearer token"}

    url, headers = session._resolve_captured_info(None, {"X-Test": "1"})
    assert url == "https://fallback.example/rpc"
    assert headers == {"X-Test": "1"}


def test_snapshot_content_hash_includes_thought_signature():
    base = AistudioContent(
        role="model",
        parts=[AistudioPart(function_call=("Read", {"file_path": "a.py"}, "call_1"))],
    )
    signed = AistudioContent(
        role="model",
        parts=[AistudioPart(function_call=("Read", {"file_path": "a.py"}, "call_1"), thought_signature="sig")],
    )

    assert _snapshot_content_hash([base]) != _snapshot_content_hash([signed])


def test_is_aistudio_url_ignores_signin_continue_param():
    from aistudio_api.infrastructure.gateway.session import _is_aistudio_url, _is_google_signin_url

    signin = (
        "https://accounts.google.com/v3/signin/identifier?"
        "continue=https%3A%2F%2Faistudio.google.com%2Fapp%2Fprompts%2Fnew_chat&flowName=GlifWebSignIn"
    )
    studio = "https://aistudio.google.com/prompts/new_chat?model=gemma-4-31b-it"

    # 登录页 continue= 参数里含 aistudio.google.com，但 host 是 accounts.google.com
    assert _is_aistudio_url(signin) is False
    assert _is_google_signin_url(signin) is True
    assert _is_aistudio_url(studio) is True
    assert _is_google_signin_url(studio) is False
