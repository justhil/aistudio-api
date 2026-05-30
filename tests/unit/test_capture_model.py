import asyncio
import json

from aistudio_api.infrastructure.gateway.capture import RequestCaptureService


class _FakeSession:
    def __init__(self, template_body: str):
        self._template_body = template_body

    async def capture_template(self, model: str):
        return {
            "url": "https://aistudio.example/rpc",
            "headers": {"authorization": "Bearer x"},
            "body": self._template_body,
        }

    async def generate_snapshot(self, contents):
        return "!snapshot"


# Bootstrap template is captured from the default model page (gemma-4-31b-it)
# and reused for every model.
_TEMPLATE_BODY = '["models/gemma-4-31b-it",[[[[null,"old"]],"user"]],null,[null,null,null,128,0.5,0.8,16],"!snap",null,null]'


def test_capture_uses_requested_model_not_template_model():
    service = RequestCaptureService(_FakeSession(_TEMPLATE_BODY))
    captured = asyncio.run(service.capture(prompt="hi", model="gemini-3.5-flash"))
    assert captured is not None
    assert captured.model == "models/gemini-3.5-flash"
    assert json.loads(captured.body)[0] == "models/gemini-3.5-flash"


def test_capture_preserves_existing_models_prefix():
    service = RequestCaptureService(_FakeSession(_TEMPLATE_BODY))
    captured = asyncio.run(service.capture(prompt="hi", model="models/gemini-3.1-pro-preview"))
    assert captured is not None
    assert captured.model == "models/gemini-3.1-pro-preview"
