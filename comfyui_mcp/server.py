"""ComfyUI image generation/editing exposed to LibreChat as MCP tools.

Stateless by design: an image *id* encodes the ComfyUI reference it points at (``{type}:{subfolder/}{filename}``, e.g. ``o:comfyui-mcp_00042_.png``), so there is no database — ComfyUI keeps the bytes on disk (durable across both ComfyUI and MCP restarts) and the agent remembers the ids it cares about.

A tool-result image only enters the model's context on the turn it's produced, so viewing is its own tool (view_image).
"""

from __future__ import annotations

import asyncio
import base64
import io
import random
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from PIL import Image as PILImage

from . import models, workflows
from .comfyui_client import ComfyUIClient, OutputImage
from .config import Config

# Ceiling on the longest edge of an image handed to the model — only to bound the base64 payload under
# LibreChat's MCP_IMAGE_DATA_MAX_BYTES (10 MB) and vision-token cost; a 1024² render passes untouched. Set
# above the default 1024² gen so fine detail (thin spires, small objects) survives.
MODEL_IMAGE_MAX_EDGE = 1536

MAX_DOWNLOAD_BYTES = 30 * 1024 * 1024

_TYPE_FROM_CODE = {"o": "output", "i": "input", "t": "temp"}
_CODE_FROM_TYPE = {"output": "o", "input": "i", "temp": "t"}

_EXT_FROM_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

# TODO(retention): nothing here ever deletes images — every generate/edit/compose/download accumulates a
# file in ComfyUI's /output or /input forever. ComfyUI has no delete API, so real cleanup needs filesystem
# access to /s/comfyui (a separate decision); until then it grows unbounded and is pruned manually.
config = Config.from_env()
mcp = FastMCP("comfyui")

MODELS_DIR = Path(config.models_dir)


@dataclass(frozen=True, slots=True)
class Profile:
    """A named model/workflow choice for an operation. `loras` auto-download on first use."""

    description: str
    loras: tuple[tuple[str, float], ...] = ()  # (registered model name, strength)
    engine: str = "qwen"  # edit only: "qwen" | "kontext"
    trigger: str = ""  # generate only: prepended to the prompt to fire a trigger-word LoRA


# Per-operation profiles. The default is our latest / most-capable; add LoRAs or tune strengths here to
# experiment (a new profile, or a new entry in an existing stack, auto-provisions on next use).
GENERATE_PROFILES = {
    "flux": Profile("Flux.1 dev, uncensored", loras=(("flux_aidma_nsfw", 0.8),), trigger="aidmaNSFWunlock"),
    "flux-base": Profile("Flux.1 dev, base (SFW)"),
}
GENERATE_DEFAULT = "flux"

EDIT_PROFILES = {
    "qwen": Profile("Qwen-Image-Edit — general edits, uncensored (clothing, pose, body, situation)", loras=(("qwen_mcnl_v1", 0.9),)),
    "qwen-max": Profile("Qwen-Image-Edit — broadest uncensored stack (MCNL + Unchained + SNOFS)", loras=(("qwen_mcnl_v1", 0.8), ("qwen_unchained_xxx", 0.9), ("qwen_snofs_v1_3", 0.8))),
    "qwen-uncensor": Profile("Qwen-Image-Edit — uncensor specialist", loras=(("qwen_edit_uncensor_nudify_v3", 0.9),)),
    "kontext": Profile("Flux Kontext — tight, identity-preserving edits (SFW)", engine="kontext"),
}
EDIT_DEFAULT = "qwen"

COMPOSE_PROFILES = {
    "qwen": Profile("Qwen-Image-Edit — combine images, uncensored", loras=(("qwen_mcnl_v1", 0.9),)),
    "qwen-base": Profile("Qwen-Image-Edit — combine images, base (SFW)"),
}
COMPOSE_DEFAULT = "qwen"

POSE_PROFILES = {
    "anypose": Profile(
        "Qwen-Image-Edit + AnyPose — copy a pose from a reference onto a character, preserving identity",
        loras=(("qwen_anypose_base", 0.7), ("qwen_anypose_helper", 0.7)),
    ),
}
POSE_DEFAULT = "anypose"

# AnyPose's recommended instruction — image1 is the character, image2 the pose reference (qwen_compose wires
# them in that order). Extra user notes (clothing/background) get appended. From the model card.
POSE_TRANSFER_INSTRUCTION = (
    "Make the person in image 1 do the exact same pose of the person in image 2. Changing the style and "
    "background of the image of the person in image 1 is undesirable, so don't do it. The new pose should be "
    "pixel accurate to the pose we are trying to copy. The position of the arms and head and legs should be "
    "the same as the pose we are trying to copy. Change the field of view and angle to match exactly image 2. "
    "Head tilt and eye gaze pose should match the person in image 2."
)


@dataclass(frozen=True, slots=True)
class VisionProfile:
    """A named vision model for ask_image."""

    description: str
    model: str  # NanoGPT callable id


VISION_PROFILES = {
    "qwen3-vl": VisionProfile("Qwen3-VL 235B — highest quality", "Qwen/Qwen3-VL-235B-A22B-Instruct"),
    "qwen2.5-vl": VisionProfile("Qwen2.5-VL 72B — smaller, more widely available", "qwen25-vl-72b-instruct"),
}
VISION_DEFAULT = "qwen3-vl"

# ask_image retries transient NanoGPT overload (429 / 5xx) across these inter-attempt waits — ~30s total.
VISION_RETRY_DELAYS = (10.0, 10.0, 10.0)


def _resolve_profile[T](profiles: dict[str, T], name: str, default: str) -> T:
    key = name or default
    if key not in profiles:
        raise ValueError(f"Unknown profile {name!r}. Available: {', '.join(profiles)}.")
    return profiles[key]


def _encode_id(image: OutputImage) -> str:
    """ComfyUI image reference → opaque-ish id the agent passes around."""
    code = _CODE_FROM_TYPE.get(image.type, "o")
    name = f"{image.subfolder}/{image.filename}" if image.subfolder else image.filename
    return f"{code}:{name}"


def _decode_id(image_id: str) -> OutputImage:
    code, sep, rest = image_id.partition(":")
    image_type = _TYPE_FROM_CODE.get(code)
    if not sep or not image_type or not rest:
        raise ValueError(f"Invalid image id {image_id!r} (expected e.g. 'o:comfyui-mcp_00042_.png').")
    subfolder, slash, filename = rest.rpartition("/")
    return OutputImage(filename=filename, subfolder=subfolder if slash else "", type=image_type)


def _thumbnail_png(data: bytes, max_edge: int = MODEL_IMAGE_MAX_EDGE) -> bytes:
    with PILImage.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        image.thumbnail((max_edge, max_edge))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def _new_filename_prefix() -> str:
    """A random SaveImage prefix per render.

    ComfyUI names outputs ``{prefix}_NNNNN_.png``; a fixed prefix yields a predictable incrementing sequence, tempting the model to *construct* the "next" id instead of carrying back the exact id a tool returned (→ 404s on nonexistent files). A random prefix removes any pattern to guess.
    """
    return f"mcp-{uuid.uuid4().hex}"


async def _stage_as_input(client: ComfyUIClient, image_id: str) -> str:
    """Resolve an id to its bytes and (re)upload them to ComfyUI's input dir, returning the LoadImage name."""
    data = await client.fetch_image(_decode_id(image_id))
    return await client.upload_image(data, f"mcp-in-{uuid.uuid4().hex[:12]}.png")


# output_schema=None on every tool: they return content/text, not a structured object. Without it FastMCP infers an output schema from the return annotation and the client rejects the unstructured result.
@mcp.tool(output_schema=None)
async def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    profile: str = "",
) -> str:
    """Generate an image from a text prompt. Returns an image id, NOT the image (call view_image to see it).

    profile (default 'flux'):
      flux — Flux.1 dev, uncensored
      flux-base — Flux.1 dev, base (SFW)

    prompt: subject, style, composition, lighting, mood; translate loose requests into a strong concrete prompt yourself. seed: omit for a random one.
    """
    chosen = _resolve_profile(GENERATE_PROFILES, profile, GENERATE_DEFAULT)
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    await models.ensure_models([*workflows.FLUX_TXT2IMG_MODELS, *(name for name, _ in chosen.loras)], MODELS_DIR, token=config.hf_token)
    extra_loras = [(models.model_filename(name), strength) for name, strength in chosen.loras]
    if chosen.trigger:
        prompt = f"{chosen.trigger}, {prompt}"
    workflow = workflows.flux_txt2img(
        prompt=prompt, seed=seed, width=width, height=height,
        filename_prefix=_new_filename_prefix(), extra_loras=extra_loras,
    )
    async with ComfyUIClient(config.comfyui_url) as client:
        rendered = await client.run(workflow)
    if not rendered:
        raise RuntimeError("ComfyUI produced no image")
    image_id = _encode_id(rendered[0].output)
    return f"Generated image {image_id} ({width}x{height}, profile '{profile or GENERATE_DEFAULT}', seed {seed}). Call view_image('{image_id}') to see it."


@mcp.tool(output_schema=None)
async def download_image(url: str) -> str:
    """Fetch an external image by URL into the ComfyUI store. Returns an image id, NOT the image itself.

    Use this to bring an image the user references (or you found) into the workspace so you can view it or use it as input to edit_image / compose_images. url must be a direct link to a png/jpeg/webp/gif.
    """
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as http:
        resp = await http.get(url)
        resp.raise_for_status()
    content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
    if not content_type.startswith("image/"):
        raise ValueError(f"URL did not return an image (content-type: {content_type or 'unknown'}).")
    if len(resp.content) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"Image too large ({len(resp.content)} bytes; max {MAX_DOWNLOAD_BYTES}).")
    ext = _EXT_FROM_MIME.get(content_type, ".png")
    async with ComfyUIClient(config.comfyui_url) as client:
        ref = await client.upload_image(resp.content, f"mcp-dl-{uuid.uuid4().hex[:12]}{ext}")
    subfolder, slash, filename = ref.rpartition("/")
    image_id = _encode_id(OutputImage(filename=filename, subfolder=subfolder if slash else "", type="input"))
    return f"Downloaded image {image_id} from {url}. Call view_image('{image_id}') to see it."


@mcp.tool(output_schema=None)
async def edit_image(image_id: str, instruction: str, seed: int | None = None, profile: str = "") -> str:
    """Edit an image with an instruction. Returns a new image id (call view_image to see it).

    profile (default 'qwen'):
      qwen — Qwen-Image-Edit, general edits, uncensored (clothing, pose, body, situation)
      qwen-max — broadest uncensored stack
      qwen-uncensor — uncensor specialist
      kontext — Flux Kontext, tight identity-preserving edits (SFW)

    instruction: name specifics for reliability. seed: omit for a random seed.
    """
    chosen = _resolve_profile(EDIT_PROFILES, profile, EDIT_DEFAULT)
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    base_models = workflows.FLUX_KONTEXT_MODELS if chosen.engine == "kontext" else workflows.QWEN_EDIT_MODELS
    await models.ensure_models([*base_models, *(name for name, _ in chosen.loras)], MODELS_DIR, token=config.hf_token)
    lora_stack = [(models.model_filename(name), strength) for name, strength in chosen.loras]
    async with ComfyUIClient(config.comfyui_url) as client:
        input_name = await _stage_as_input(client, image_id)
        if chosen.engine == "kontext":
            workflow = workflows.flux_kontext_edit(
                instruction=instruction, input_image=input_name, seed=seed, filename_prefix=_new_filename_prefix()
            )
        else:
            workflow = workflows.qwen_compose(
                instruction=instruction, input_images=[input_name], seed=seed,
                filename_prefix=_new_filename_prefix(), extra_loras=lora_stack,
            )
        rendered = await client.run(workflow)
    if not rendered:
        raise RuntimeError("ComfyUI produced no image")
    new_id = _encode_id(rendered[0].output)
    return f"Edited {image_id} into {new_id} (profile '{profile or EDIT_DEFAULT}', seed {seed}). Call view_image('{new_id}') to see it."


@mcp.tool(output_schema=None)
async def compose_images(
    image_ids: list[str], instruction: str, seed: int | None = None, profile: str = ""
) -> str:
    """Combine 1-3 existing images into a new one per an instruction. Returns a new image id.

    Use for "take the character from <id1> and the style of <id2> and …". The first id sets the output size.

    profile (default 'qwen'):
      qwen — combine images, uncensored
      qwen-base — base (SFW)

    seed: omit for a random seed.
    """
    if not 1 <= len(image_ids) <= 3:
        raise ValueError("compose_images takes 1 to 3 image ids.")
    chosen = _resolve_profile(COMPOSE_PROFILES, profile, COMPOSE_DEFAULT)
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    await models.ensure_models([*workflows.QWEN_EDIT_MODELS, *(name for name, _ in chosen.loras)], MODELS_DIR, token=config.hf_token)
    lora_stack = [(models.model_filename(name), strength) for name, strength in chosen.loras]
    async with ComfyUIClient(config.comfyui_url) as client:
        input_names = [await _stage_as_input(client, image_id) for image_id in image_ids]
        workflow = workflows.qwen_compose(
            instruction=instruction, input_images=input_names, seed=seed,
            filename_prefix=_new_filename_prefix(), extra_loras=lora_stack,
        )
        rendered = await client.run(workflow)
    if not rendered:
        raise RuntimeError("ComfyUI produced no image")
    new_id = _encode_id(rendered[0].output)
    joined = ", ".join(image_ids)
    return f"Composed [{joined}] into {new_id} (profile '{profile or COMPOSE_DEFAULT}', seed {seed}). Call view_image('{new_id}') to see it."


@mcp.tool(output_schema=None)
async def pose_transfer(
    character_id: str, pose_ref_id: str, extra: str = "", seed: int | None = None, profile: str = ""
) -> str:
    """Repose a character to match a reference pose, preserving their identity. Returns a new image id.

    Use this when compose_images/edit_image only weakly follow a target pose (the usual "gave them props and
    called it done" failure). It copies the pose of the person in pose_ref onto the character, keeping the
    character's face, body, and clothing.

    One person per image only: if the character image or the pose reference has multiple people, AnyPose gets
    confused. To pose two characters into one scene, run pose_transfer per character (each against the same
    pose reference, cropped to that person), then compose_images the results together.

    profile (default 'anypose'):
      anypose — Qwen-Image-Edit + AnyPose, identity-preserving pose copy

    character_id: the person the output keeps. pose_ref_id: image whose pose to copy. extra: optional instruction appended (e.g. clothing or background changes). seed: omit for a random seed.
    """
    chosen = _resolve_profile(POSE_PROFILES, profile, POSE_DEFAULT)
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    await models.ensure_models([*workflows.QWEN_EDIT_MODELS, *(name for name, _ in chosen.loras)], MODELS_DIR, token=config.hf_token)
    lora_stack = [(models.model_filename(name), strength) for name, strength in chosen.loras]
    instruction = f"{POSE_TRANSFER_INSTRUCTION} {extra}".strip() if extra else POSE_TRANSFER_INSTRUCTION
    async with ComfyUIClient(config.comfyui_url) as client:
        character_name = await _stage_as_input(client, character_id)
        pose_name = await _stage_as_input(client, pose_ref_id)
        workflow = workflows.qwen_compose(
            instruction=instruction, input_images=[character_name, pose_name], seed=seed,
            filename_prefix=_new_filename_prefix(), extra_loras=lora_stack,
        )
        rendered = await client.run(workflow)
    if not rendered:
        raise RuntimeError("ComfyUI produced no image")
    new_id = _encode_id(rendered[0].output)
    return f"Reposed {character_id} to match {pose_ref_id} → {new_id} (profile '{profile or POSE_DEFAULT}', seed {seed}). Call view_image('{new_id}') to see it."


@mcp.tool(output_schema=None)
async def view_image(image_id: str) -> list:
    """Return an image by id so you can see it.

    An image is only visible to you on the turn you view it — call this whenever you need to look at or re-examine one to describe it, critique it, or answer a question about its content.
    """
    async with ComfyUIClient(config.comfyui_url) as client:
        data = await client.fetch_image(_decode_id(image_id))
    return [f"Image {image_id}:", Image(data=_thumbnail_png(data), format="png")]


@mcp.tool(output_schema=None)
async def ask_image(image_id: str, question: str, profile: str = "") -> str:
    """Ask a vision model a question about an image and get a text answer.

    This is how you perceive an image's content: you don't see images directly — call this with a focused question (e.g. "how many people are on the bench? is the lighting consistent? does it match <X>?") and reason over the answer. Use it to inspect a result before deciding whether to edit it further.

    profile (default 'qwen3-vl'):
      qwen3-vl — Qwen3-VL 235B, highest quality
      qwen2.5-vl — Qwen2.5-VL 72B, smaller and more widely available
    """
    if not config.vision_api_key:
        raise RuntimeError("Vision API key not configured (VISION_API_KEY).")
    chosen = _resolve_profile(VISION_PROFILES, profile, VISION_DEFAULT)
    async with ComfyUIClient(config.comfyui_url) as client:
        data = await client.fetch_image(_decode_id(image_id))
    image_url = f"data:image/png;base64,{base64.b64encode(_thumbnail_png(data)).decode('ascii')}"
    payload = {
        "model": chosen.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }
    async with httpx.AsyncClient(timeout=120.0) as http:
        for delay in (0.0, *VISION_RETRY_DELAYS):
            if delay:
                await asyncio.sleep(delay)
            resp = await http.post(
                f"{config.vision_base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {config.vision_api_key}"},
            )
            if resp.status_code != 429 and resp.status_code < 500:
                break
    if resp.is_error:
        raise RuntimeError(f"Vision API {resp.status_code} for model {chosen.model!r}: {resp.text}")
    return resp.json()["choices"][0]["message"]["content"]


@mcp.tool(output_schema=None)
async def list_images(limit: int = 20) -> str:
    """List recent images known to ComfyUI, newest first, with their ids and prompts.

    This is server-wide recent activity and is not retained across a ComfyUI restart; the ids you were given by other tools remain valid regardless (they resolve directly to stored files).
    """
    async with ComfyUIClient(config.comfyui_url) as client:
        history = await client.history(max_items=limit)
    lines: list[str] = []
    for entry in reversed(list(history.values())):
        prompt_text = _extract_prompt_text(entry.get("prompt"))
        for node_output in entry.get("outputs", {}).values():
            for image in node_output.get("images", []):
                if image.get("type") == "temp":
                    continue
                image_id = _encode_id(
                    OutputImage(
                        filename=image["filename"],
                        subfolder=image.get("subfolder", ""),
                        type=image.get("type", "output"),
                    )
                )
                lines.append(f"{image_id} — {prompt_text}" if prompt_text else image_id)
    return "\n".join(lines) if lines else "No recent images."


def _extract_prompt_text(prompt: object) -> str:
    """Best-effort positive-prompt text from a /history entry's workflow graph."""
    if not isinstance(prompt, list) or len(prompt) < 3 or not isinstance(prompt[2], dict):
        return ""
    for node in prompt[2].values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") in {"CLIPTextEncode", "TextEncodeQwenImageEditPlus"}:
            inputs = node.get("inputs", {})
            text = inputs.get("text") or inputs.get("prompt")
            if isinstance(text, str) and text.strip():
                return text.strip()
    return ""


def main() -> None:
    mcp.run(transport="http", host=config.host, port=config.port)


if __name__ == "__main__":
    main()
