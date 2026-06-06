"""Async client for ComfyUI's HTTP + websocket API.

Kept free of any pydantic-ai / agent imports so it can be reused standalone.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import httpx
from websockets.asyncio.client import connect as ws_connect

# (value, maximum) sampling step counts.
ProgressCallback = Callable[[int, int], None]


class ComfyUIError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OutputImage:
    filename: str
    subfolder: str
    type: str


@dataclass(frozen=True, slots=True)
class RenderedImage:
    output: OutputImage
    data: bytes


class ComfyUIClient:
    """Submit API-format workflows and retrieve their output images."""

    def __init__(self, base_url: str, *, request_timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=request_timeout)
        self._client_id = uuid.uuid4().hex

    async def __aenter__(self) -> "ComfyUIClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def upload_image(self, data: bytes, filename: str) -> str:
        """Upload an input image; return the name a LoadImage node references it by."""
        resp = await self._client.post(
            "/upload/image",
            files={"image": (filename, data, "image/png")},
            data={"overwrite": "true"},
        )
        if resp.status_code != httpx.codes.OK:
            raise ComfyUIError(f"upload failed ({resp.status_code}): {resp.text}")
        body = resp.json()
        subfolder = body.get("subfolder") or ""
        name = body["name"]
        return f"{subfolder}/{name}" if subfolder else name

    async def submit(self, workflow: dict) -> str:
        resp = await self._client.post(
            "/prompt", json={"prompt": workflow, "client_id": self._client_id}
        )
        if resp.status_code != httpx.codes.OK:
            raise ComfyUIError(f"workflow rejected ({resp.status_code}): {resp.text}")
        return resp.json()["prompt_id"]

    async def run(
        self,
        workflow: dict,
        *,
        progress: ProgressCallback | None = None,
        timeout: float = 300.0,
    ) -> list[RenderedImage]:
        # Open the websocket before submitting so no early events are missed.
        async with ws_connect(self._ws_url(), max_size=None) as ws:
            prompt_id = await self.submit(workflow)
            await self._await_completion(ws, prompt_id, progress, timeout)
        outputs = await self._history_outputs(prompt_id)
        return [RenderedImage(o, await self.fetch_image(o)) for o in outputs]

    async def fetch_image(self, image: OutputImage) -> bytes:
        resp = await self._client.get(
            "/view",
            params={
                "filename": image.filename,
                "subfolder": image.subfolder,
                "type": image.type,
            },
        )
        resp.raise_for_status()
        return resp.content

    async def history(self, max_items: int = 20) -> dict:
        """Most recent prompt executions, keyed by prompt_id (server-wide, in-memory)."""
        resp = await self._client.get("/history", params={"max_items": max_items})
        resp.raise_for_status()
        return resp.json()

    def _ws_url(self) -> str:
        parts = urlsplit(self._base_url)
        scheme = "wss" if parts.scheme == "https" else "ws"
        return urlunsplit((scheme, parts.netloc, "/ws", f"clientId={self._client_id}", ""))

    async def _await_completion(
        self, ws, prompt_id: str, progress: ProgressCallback | None, timeout: float
    ) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise ComfyUIError(f"timed out after {timeout:.0f}s waiting for {prompt_id}")
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except TimeoutError:
                raise ComfyUIError(
                    f"timed out after {timeout:.0f}s waiting for {prompt_id}"
                ) from None
            if not isinstance(message, str):
                continue  # binary preview frame
            event = json.loads(message)
            data = event.get("data", {})
            event_prompt = data.get("prompt_id")
            if event_prompt is not None and event_prompt != prompt_id:
                continue  # another client's prompt
            match event.get("type"):
                case "progress" if progress is not None:
                    progress(data.get("value", 0), data.get("max", 0))
                case "execution_error":
                    raise ComfyUIError(_format_ws_error(data))
                case "execution_success":
                    return
                case "executing" if data.get("node") is None and event_prompt == prompt_id:
                    return

    async def _history_outputs(self, prompt_id: str) -> list[OutputImage]:
        # The websocket completion signal can land a beat before /history is written.
        for _ in range(5):
            resp = await self._client.get(f"/history/{prompt_id}")
            resp.raise_for_status()
            entry = resp.json().get(prompt_id)
            if entry is not None:
                return _collect_outputs(entry.get("outputs", {}))
            await asyncio.sleep(0.2)
        raise ComfyUIError(f"no history for {prompt_id} after completion")


def _collect_outputs(outputs: dict) -> list[OutputImage]:
    images: list[OutputImage] = []
    for node_output in outputs.values():
        for image in node_output.get("images", []):
            if image.get("type") == "temp":
                continue
            images.append(
                OutputImage(
                    filename=image["filename"],
                    subfolder=image.get("subfolder", ""),
                    type=image.get("type", "output"),
                )
            )
    return images


def _format_ws_error(data: dict) -> str:
    node = data.get("node_type") or data.get("node_id") or "?"
    return f"ComfyUI execution error in {node}: {data.get('exception_message', 'unknown')}"
