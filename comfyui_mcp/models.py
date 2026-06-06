"""Idempotent model-file provisioning for ComfyUI workflows.

ComfyUI picks up newly written files without a restart — its filename cache invalidates on the model directory's mtime.

The idempotency check is existence + size; the full sha256 is verified once, at download time.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass(frozen=True, slots=True)
class ModelFile:
    folder: str  # ComfyUI model subdir: loras, diffusion_models, vae, text_encoders, ...
    filename: str  # name on disk — must match the workflow's *_name input
    url: str
    sha256: str = ""  # lowercase hex; verified on download when present
    size: int = 0  # expected byte size; the idempotency check


MODELS: dict[str, ModelFile] = {
    # Base models the workflows load (checkpoints, text encoders, VAEs). flux1-dev and the Flux VAE (ae.safetensors) are gated on black-forest-labs/FLUX.1-dev; the downloader sends the HF_TOKEN bearer to fetch them.
    "flux_dev": ModelFile(
        "diffusion_models", "flux1-dev.safetensors",
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors",
        "4610115bb0c89560703c892c59ac2742fa821e60ef5871b33493ba544683abd7", 23802932552,
    ),
    "flux_kontext": ModelFile(
        "diffusion_models", "flux1-dev-kontext_fp8_scaled.safetensors",
        "https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors",
        "630ba795ec64283b4230ea23cf79406c2c68b7c578229ed139f30043eadb30a2", 11904640136,
    ),
    "flux_clip_l": ModelFile(
        "text_encoders", "clip_l.safetensors",
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
        "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd", 246144152,
    ),
    "flux_t5": ModelFile(
        "text_encoders", "t5xxl_fp8_e4m3fn_scaled.safetensors",
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors",
        "a498f0485dc9536735258018417c3fd7758dc3bccc0a645feaa472b34955557a", 5157348688,
    ),
    "flux_vae": ModelFile(
        "vae", "ae.safetensors",
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors",
        "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38", 335304388,
    ),
    "qwen_edit_2511": ModelFile(
        "diffusion_models", "qwen_image_edit_2511_bf16.safetensors",
        "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors",
        "ae42d927b5fac4f278b9a894554c727e619727a63622976f2d95625be4bce08c", 40861031560,
    ),
    "qwen_clip": ModelFile(
        "text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4", 9384670680,
    ),
    "qwen_vae": ModelFile(
        "vae", "qwen_image_vae.safetensors",
        "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
        "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f", 253806246,
    ),
    "qwen_lightning": ModelFile(
        "loras", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "22226e8d05d354bb356627d428809f5afd7819399b077238a2b70a82883a904f", 849608296,
    ),
    # Z-Image Turbo. UNET folder is unet/ (not diffusion_models/) to match the deployed layout; ComfyUI's UNETLoader reads both.
    "z_image": ModelFile(
        "unet", "z_image_turbo_bf16.safetensors",
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors",
        "2407613050b809ffdff18a4ac99af83ea6b95443ecebdf80e064a79c825574a6", 12309866400,
    ),
    "z_image_clip": ModelFile(
        "text_encoders", "qwen_3_4b.safetensors",
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors",
        "6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a", 8044982048,
    ),
    # General uncensored *edit* LoRAs for Qwen-Image-Edit-2511 (LoraLoaderModelOnly on the transformer). Trained on
    # the 2509/Qwen-Image-Edit transformer; load cleanly on 2511 (same shape) — nudge strength down if needed.
    "qwen_mcnl_v1": ModelFile(
        "loras", "qwen_MCNL_v1.0.safetensors",
        "https://huggingface.co/spectra7/qwen-mcnl-lora/resolve/main/qwen_MCNL_v1.0.safetensors",
        "16c4841028615bb82c38e79756c0abad42494d85bca0daebc2939384a74d86bb", 590058864,
    ),
    "qwen_unchained_xxx": ModelFile(
        "loras", "Unchained-XXX.safetensors",
        "https://huggingface.co/urogin/QWEN-Edit-2509-Unchained-XXX/resolve/main/Unchained-XXX.safetensors",
        "ff403675ab7613655b1292f3ee92b8fae8665ad857427709acc814f5634b55f5", 295146176,
    ),
    "qwen_snofs_v1_3": ModelFile(
        "loras", "Qwen_Snofs_1_3.safetensors",
        "https://huggingface.co/rkppvc/qwen_snofs_v13/resolve/main/Qwen_Snofs_1_3.safetensors",
        "4a34f57bdb5333b884db997a9b74b2ba740b5a8344b750c6b8cb75a35088cee7", 613841472,
    ),
    "qwen_edit_uncensor_nudify_v3": ModelFile(
        "loras", "qwen_edit_uncensor_nudify_v3.safetensors",
        "https://huggingface.co/amethyst9/2066568/resolve/main/2066568_qwen_edit_uncensor+nudify_lora_v3.safetensors",
        "ff813118e5eddd38ea0c83eeef4c777c9947c388a5e839d99ab75ffc98aaac73", 590154560,
    ),
    "qwen_edit_f2p": ModelFile(
        "loras", "qwen_edit_f2p_0928_step40000.safetensors",
        "https://huggingface.co/DiffSynth-Studio/Qwen-Image-Edit-F2P/resolve/main/edit_0928_lora_step40000.safetensors",
        "2766a84f5d01b14b172c0307ef69c077fe9c399d304359eadb5fa5819594a175", 472047184,
    ),
    # AnyPose pose-transfer LoRA pair for Qwen-Image-Edit-2511 (base + helper, both loaded at ~0.7). Copies
    # the pose of person-in-image-2 onto person-in-image-1 while preserving identity — no OpenPose/skeleton
    # needed. V1 trips on more than one person per input, so pose one character at a time.
    "qwen_anypose_base": ModelFile(
        "loras", "2511-AnyPose-base-000006250.safetensors",
        "https://huggingface.co/lilylilith/AnyPose/resolve/main/2511-AnyPose-base-000006250.safetensors",
        "0396314e43b94f7520cf5242e6aa6e8e9095d784f7e87ee45980c2e541ee8c4d", 295146208,
    ),
    "qwen_anypose_helper": ModelFile(
        "loras", "2511-AnyPose-helper-00006000.safetensors",
        "https://huggingface.co/lilylilith/AnyPose/resolve/main/2511-AnyPose-helper-00006000.safetensors",
        "674340c62fff3cf981d0cfc55daf2f0e6f51c56b50159e241d931748200cf1eb", 295146216,
    ),
    # Uncensored *generation* LoRA for Flux.1 dev. Trigger word: "aidmaNSFWunlock".
    "flux_aidma_nsfw": ModelFile(
        "loras", "aidmaNSFWunlock-FLUX-V0.2.safetensors",
        "https://huggingface.co/shahtab/FLUXNSFWunlock/resolve/main/aidmaNSFWunlock-FLUX-V0.2.safetensors",
        "7d3408f4a7b7890f470c1cce7f7255b6d8e03ea770a2829066c1ce41145562ec", 19268648,
    ),
}

# One lock per target path so concurrent calls don't double-fetch the same file while distinct files still download in parallel. Lazy population is safe: dict access never yields in single-threaded asyncio.
_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def model_filename(name: str) -> str:
    return MODELS[name].filename


async def ensure_models(names: list[str], models_dir: Path, *, token: str = "") -> None:
    for name in names:
        await _ensure_one(MODELS[name], models_dir, token)


async def _ensure_one(model: ModelFile, models_dir: Path, token: str) -> None:
    target = models_dir / model.folder / model.filename
    if _present(target, model.size):
        return
    async with _locks[str(target)]:
        if _present(target, model.size):  # another caller may have fetched it while we waited
            return
        await _download(model, target, token)


def _present(target: Path, expected_size: int) -> bool:
    try:
        return target.is_file() and (expected_size == 0 or target.stat().st_size == expected_size)
    except OSError:
        return False


def _auth_headers(url: str, token: str) -> dict[str, str]:
    # httpx drops the Authorization header on cross-host redirects, so it reaches huggingface.co (gating check) but not the presigned CDN target.
    if token and httpx.URL(url).host == "huggingface.co":
        return {"Authorization": f"Bearer {token}"}
    return {}


async def _download(model: ModelFile, target: Path, token: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    hasher = hashlib.sha256()
    timeout = httpx.Timeout(connect=30.0, read=300.0, write=300.0, pool=30.0)
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=timeout, headers=_auth_headers(model.url, token)
        ) as client:
            async with client.stream("GET", model.url) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as handle:
                    async for chunk in resp.aiter_bytes(chunk_size=1 << 20):
                        handle.write(chunk)
                        hasher.update(chunk)
        if model.sha256 and hasher.hexdigest() != model.sha256:
            raise RuntimeError(
                f"sha256 mismatch for {model.filename}: got {hasher.hexdigest()}, expected {model.sha256}"
            )
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
