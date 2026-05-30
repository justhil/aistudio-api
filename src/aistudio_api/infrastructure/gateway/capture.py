"""Hook-first request capture workflow."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from aistudio_api.config import DEFAULT_TEXT_MODEL
from aistudio_api.infrastructure.gateway.request_rewriter import modify_body
from aistudio_api.infrastructure.gateway.session import BrowserSession
from aistudio_api.infrastructure.gateway.wire_types import AistudioContent, AistudioPart

logger = logging.getLogger("aistudio")


@dataclass
class CapturedRequest:
    url: str
    headers: dict[str, str]
    body: str
    model: str = ""
    snapshot: str = ""

    def __post_init__(self):
        parsed = json.loads(self.body)
        self.model = parsed[0] if parsed else ""
        self.snapshot = parsed[4] if len(parsed) > 4 and isinstance(parsed[4], str) else ""


class RequestCaptureService:
    """Single-page hook flow modeled after camoufox-api."""

    def __init__(self, session: BrowserSession):
        self._session = session
        self._templates: dict[str, CapturedRequest] = {}

    async def capture(
        self,
        prompt: str,
        model: str = DEFAULT_TEXT_MODEL,
        images: list[str] | None = None,
        contents: list[AistudioContent] | None = None,
        force_refresh: bool = False,
    ) -> CapturedRequest | None:
        template = await self._ensure_template(model)
        # 先只走 inlineData 路径，避免 fileData/Drive 上传链路干扰主流程。
        rewritten_contents = contents
        snapshot_contents = rewritten_contents or [self._build_capture_content(prompt=prompt, images=images)]
        snapshot = await self._session.generate_snapshot(snapshot_contents)
        # 模板是从默认模型页（gemma-4-31b-it）抓取并被所有模型复用的，其 body 里
        # 烘焙的 model 不能作为实际请求模型，必须用调用方请求的 model（补全 models/ 前缀）。
        wire_model = model if model.startswith("models/") else f"models/{model}"
        body = modify_body(
            template.body,
            model=wire_model,
            prompt=prompt,
            contents=rewritten_contents,
            snapshot=snapshot,
        )
        captured = CapturedRequest(url=template.url, headers=template.headers, body=body)
        logger.info(
            "Hook 拦截成功: model=%s, snapshot=%s chars, body=%s chars",
            captured.model,
            len(captured.snapshot),
            len(captured.body),
        )
        return captured

    async def _ensure_template(self, model: str) -> CapturedRequest:
        if model in self._templates:
            return self._templates[model]

        captured = await self._session.capture_template(model)
        template = CapturedRequest(**captured)
        self._templates[model] = template
        logger.info("Hook 模板已就绪: model=%s", model)
        return template

    def _build_capture_content(self, prompt: str, images: list[str] | None) -> AistudioContent:
        parts = [AistudioPart(text=prompt)]
        return AistudioContent(role="user", parts=parts)
