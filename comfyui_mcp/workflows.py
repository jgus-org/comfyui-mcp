"""Builders for ComfyUI API-format workflow graphs.

A workflow is a ``{node_id: {"class_type", "inputs"}}`` dict where an input wired from another node is ``[node_id, output_index]``.
"""

from __future__ import annotations

from . import models

# Filenames each workflow wires into the graph, resolved from the model registry (the single source of truth for identity + provisioning).
FLUX_DEV_UNET = models.model_filename("flux_dev")
FLUX_KONTEXT_UNET = models.model_filename("flux_kontext")
FLUX_CLIP_L = models.model_filename("flux_clip_l")
FLUX_T5 = models.model_filename("flux_t5")
FLUX_VAE = models.model_filename("flux_vae")
QWEN_EDIT_UNET = models.model_filename("qwen_edit_2511")
QWEN_CLIP = models.model_filename("qwen_clip")
QWEN_VAE = models.model_filename("qwen_vae")
QWEN_LIGHTNING_LORA = models.model_filename("qwen_lightning")
ZIMAGE_UNET = models.model_filename("z_image")
ZIMAGE_CLIP = models.model_filename("z_image_clip")

# Base model registry keys each workflow loads; pass to models.ensure_models before running, alongside any profile LoRAs.
ZIMAGE_MODELS = ("z_image", "z_image_clip", "flux_vae")
FLUX_TXT2IMG_MODELS = ("flux_dev", "flux_clip_l", "flux_t5", "flux_vae")
FLUX_KONTEXT_MODELS = ("flux_kontext", "flux_clip_l", "flux_t5", "flux_vae")
QWEN_EDIT_MODELS = ("qwen_edit_2511", "qwen_clip", "qwen_vae", "qwen_lightning")

# Flux.1 dev is bf16 (~24 GB); cast to fp8 at load so model plus text encoders fit the 24 GB GPU.
FLUX_DEV_WEIGHT_DTYPE = "fp8_e4m3fn"

# Qwen-Image-Edit is 20B (bf16 ~40 GB); cast to fp8 at load to fit the 24 GB GPU.
QWEN_EDIT_WEIGHT_DTYPE = "fp8_e4m3fn"
QWEN_EDIT_SHIFT = 3.1


def zimage_txt2img(
    *,
    prompt: str,
    seed: int,
    width: int = 1024,
    height: int = 1024,
    steps: int = 8,
    filename_prefix: str = "comfyui-agent",
) -> dict:
    """Text-to-image with Z-Image Turbo. Mirrors ComfyUI's stock ``image_z_image_turbo`` template."""
    return {
        "28": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": ZIMAGE_UNET, "weight_dtype": "default"},
        },
        "11": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["28", 0], "shift": 3.0},
        },
        "30": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": ZIMAGE_CLIP, "type": "lumina2", "device": "default"},
        },
        "27": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["30", 0], "text": prompt},
        },
        "33": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["27", 0]},
        },
        "29": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": FLUX_VAE},
        },
        "13": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "res_multistep",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["11", 0],
                "positive": ["27", 0],
                "negative": ["33", 0],
                "latent_image": ["13", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["29", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": filename_prefix, "images": ["8", 0]},
        },
    }


def flux_txt2img(
    *,
    prompt: str,
    seed: int,
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    filename_prefix: str = "comfyui-agent",
    extra_loras: list[tuple[str, float]] = (),
) -> dict:
    """Text-to-image with Flux.1 [dev]. Mirrors ComfyUI's stock ``flux_dev_full_text_to_image`` template.

    No explicit FluxGuidance; if output looks flat, a ``FluxGuidance`` node at ~3.5 on the positive conditioning is the usual fix.
    """
    workflow = {
        "46": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": FLUX_DEV_UNET, "weight_dtype": FLUX_DEV_WEIGHT_DTYPE},
        },
        "47": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": FLUX_CLIP_L,
                "clip_name2": FLUX_T5,
                "type": "flux",
                "device": "default",
            },
        },
        "45": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}},
        "49": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["47", 0], "text": prompt}},
        "52": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["49", 0]}},
        "48": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "50": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["46", 0],
                "positive": ["49", 0],
                "negative": ["52", 0],
                "latent_image": ["48", 0],
            },
        },
        "51": {"class_type": "VAEDecode", "inputs": {"samples": ["50", 0], "vae": ["45", 0]}},
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": filename_prefix, "images": ["51", 0]},
        },
    }
    _chain_loras(workflow, extra_loras, after="46", into=("50", "model"))
    return workflow


def flux_kontext_edit(
    *,
    instruction: str,
    input_image: str,
    seed: int,
    guidance: float = 2.5,
    steps: int = 20,
    filename_prefix: str = "comfyui-agent",
) -> dict:
    """Instruction-based single-image edit with Flux.1 Kontext [dev]. Mirrors the stock ``flux_kontext_dev_basic`` template (minus the two-image ImageStitch).

    Denoise stays 1.0; edit strength is governed by ``guidance``, not an img2img denoise fraction.
    """
    return {
        "37": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": FLUX_KONTEXT_UNET, "weight_dtype": "default"},
        },
        "38": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": FLUX_CLIP_L,
                "clip_name2": FLUX_T5,
                "type": "flux",
                "device": "default",
            },
        },
        "39": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}},
        "10": {"class_type": "LoadImage", "inputs": {"image": input_image}},
        "42": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["10", 0]}},
        "124": {"class_type": "VAEEncode", "inputs": {"pixels": ["42", 0], "vae": ["39", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["38", 0], "text": instruction}},
        "177": {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": ["6", 0], "latent": ["124", 0]},
        },
        "35": {
            "class_type": "FluxGuidance",
            "inputs": {"conditioning": ["177", 0], "guidance": guidance},
        },
        "135": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
        "31": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["37", 0],
                "positive": ["35", 0],
                "negative": ["135", 0],
                "latent_image": ["124", 0],
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["31", 0], "vae": ["39", 0]}},
        "136": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": filename_prefix, "images": ["8", 0]},
        },
    }


def qwen_compose(
    *,
    instruction: str,
    input_images: list[str],
    seed: int,
    steps: int = 4,
    filename_prefix: str = "comfyui-agent",
    extra_loras: list[tuple[str, float]] = (),
) -> dict:
    """Multi-image composition / editing with Qwen-Image-Edit-2511. Mirrors the stock ``image_qwen_image_edit_2511`` template, fixed to the 4-step Lightning LoRA path.

    Takes 1-3 reference images; the first sets the output canvas.
    """
    if not 1 <= len(input_images) <= 3:
        raise ValueError("qwen_compose needs 1 to 3 input images")

    def encode(prompt: str) -> dict:
        inputs = {"clip": ["5", 0], "prompt": prompt, "vae": ["6", 0], "image1": ["8", 0]}
        if len(input_images) >= 2:
            inputs["image2"] = ["9", 0]
        if len(input_images) >= 3:
            inputs["image3"] = ["10", 0]
        return {"class_type": "TextEncodeQwenImageEditPlus", "inputs": inputs}

    workflow = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": QWEN_EDIT_UNET, "weight_dtype": QWEN_EDIT_WEIGHT_DTYPE},
        },
        "2": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": QWEN_EDIT_SHIFT}},
        "3": {"class_type": "CFGNorm", "inputs": {"model": ["2", 0], "strength": 1.0}},
        "4": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["3", 0], "lora_name": QWEN_LIGHTNING_LORA, "strength_model": 1.0},
        },
        "5": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": QWEN_CLIP, "type": "qwen_image", "device": "default"},
        },
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
        "7": {"class_type": "LoadImage", "inputs": {"image": input_images[0]}},
        "8": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["7", 0]}},
        "11": encode(instruction),
        "12": encode(""),
        "13": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {"conditioning": ["11", 0], "reference_latents_method": "index_timestep_zero"},
        },
        "14": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {"conditioning": ["12", 0], "reference_latents_method": "index_timestep_zero"},
        },
        "15": {"class_type": "VAEEncode", "inputs": {"pixels": ["8", 0], "vae": ["6", 0]}},
        "16": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["13", 0],
                "negative": ["14", 0],
                "latent_image": ["15", 0],
            },
        },
        "17": {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["6", 0]}},
        "18": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": filename_prefix, "images": ["17", 0]},
        },
    }
    if len(input_images) >= 2:
        workflow["9"] = {"class_type": "LoadImage", "inputs": {"image": input_images[1]}}
    if len(input_images) >= 3:
        workflow["10"] = {"class_type": "LoadImage", "inputs": {"image": input_images[2]}}
    _chain_loras(workflow, extra_loras, after="4", into=("16", "model"))
    return workflow


def _chain_loras(
    workflow: dict,
    extra_loras: list[tuple[str, float]],
    *,
    after: str,
    into: tuple[str, str],
) -> None:
    """Insert LoraLoaderModelOnly nodes between node ``after`` and the model input ``into``."""
    prev = after
    for index, (lora_name, strength) in enumerate(extra_loras):
        node_id = f"lora_{index}"
        workflow[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": [prev, 0], "lora_name": lora_name, "strength_model": strength},
        }
        prev = node_id
    if extra_loras:
        node, key = into
        workflow[node]["inputs"][key] = [prev, 0]
