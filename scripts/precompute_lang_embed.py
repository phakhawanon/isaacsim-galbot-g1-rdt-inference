"""Precompute the T5 language embedding for the task instruction used by main.py.

Run this once, standalone (plain python, NOT through isaaclab.sh), before launching
main.py. Loading T5-xxl (~22GB) inside the Isaac Sim process alongside RDT + SigLip is
what caused CUDA OOM during fine-tuning (see rdt-1b-galbot/data/NOTES.md) -- the fix
there was the same as here: precompute once, then only load the small .pt file at
inference time.

The task instruction, T5 checkpoint, and rdt-1b-galbot checkout location are read
from config/precompute_lang_embed.yaml (repo root), not hardcoded here -- see that
file, or README.md's "Selecting the RDT-1B model" section, to change the task
instruction. Override with PRECOMPUTE_LANG_EMBED_CONFIG=/path/to/other.yaml to avoid
editing the default config in place.

Usage:
    cd rdt-1b-galbot && python ../inference_galbot_golf/scripts/precompute_lang_embed.py
"""

import os
import sys

import torch
import yaml

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LANG_EMBED_CONFIG_PATH = os.environ.get(
    "PRECOMPUTE_LANG_EMBED_CONFIG", os.path.join(REPO_ROOT, "config", "precompute_lang_embed.yaml")
)

with open(LANG_EMBED_CONFIG_PATH, "r") as fp:
    LANG_EMBED_CONFIG = yaml.safe_load(fp)

RDT_ROOT = os.path.normpath(os.path.join(REPO_ROOT, LANG_EMBED_CONFIG["rdt_root"]))
sys.path.insert(0, RDT_ROOT)

from models.multimodal_encoder.t5_encoder import T5Embedder  # noqa: E402

GPU = LANG_EMBED_CONFIG["gpu"]
MODEL_PATH = os.path.join(RDT_ROOT, LANG_EMBED_CONFIG["t5_model"])
CONFIG_PATH = os.path.join(RDT_ROOT, "configs", "base.yaml")
# rdt_server.py loads lang_embed.pt from its own directory by default -- keep save_path
# pointed there unless you also change where rdt_server.py looks.
SAVE_PATH = os.path.join(REPO_ROOT, LANG_EMBED_CONFIG["save_path"])

TASK_NAME = LANG_EMBED_CONFIG["task_name"]
INSTRUCTION = LANG_EMBED_CONFIG["instruction"]

# If GPU VRAM is less than 24GB, set this to an existing directory to enable offloading.
OFFLOAD_DIR = LANG_EMBED_CONFIG["offload_dir"]


def main():
    if not INSTRUCTION:
        raise ValueError(
            "Set 'instruction' in config/precompute_lang_embed.yaml to the actual task "
            "instruction before running."
        )

    with open(CONFIG_PATH, "r") as fp:
        config = yaml.safe_load(fp)

    device = torch.device(f"cuda:{GPU}")

    # T5Embedder.available_models only lists the HF hub id ("google/t5-v1_1-xxl") and
    # asserts against it; register our local checkpoint path too so we don't trigger
    # the AssertionError documented in rdt-1b-galbot/data/NOTES.md Session 3.
    if MODEL_PATH not in T5Embedder.available_models:
        T5Embedder.available_models.append(MODEL_PATH)

    text_embedder = T5Embedder(
        from_pretrained=MODEL_PATH,
        model_max_length=config["dataset"]["tokenizer_max_length"],
        device=device,
        use_offload_folder=OFFLOAD_DIR,
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    tokens = tokenizer(
        INSTRUCTION, return_tensors="pt",
        padding="longest",
        truncation=True,
    )["input_ids"].to(device)

    tokens = tokens.view(1, -1)
    with torch.no_grad():
        pred = text_encoder(tokens).last_hidden_state.detach().cpu()

    torch.save({
        "name": TASK_NAME,
        "instruction": INSTRUCTION,
        "embeddings": pred,
    }, SAVE_PATH)

    print(f'"{INSTRUCTION}" encoded into shape {pred.shape} and saved to "{SAVE_PATH}"')


if __name__ == "__main__":
    main()
