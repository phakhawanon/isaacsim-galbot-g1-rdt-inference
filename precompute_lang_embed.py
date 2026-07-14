"""Precompute the T5 language embedding for the task instruction used by main.py.

Run this once, standalone (plain python, NOT through isaaclab.sh), before launching
main.py. Loading T5-xxl (~22GB) inside the Isaac Sim process alongside RDT + SigLip is
what caused CUDA OOM during fine-tuning (see rdt-1b-galbot/data/NOTES.md) -- the fix
there was the same as here: precompute once, then only load the small .pt file at
inference time.

Usage:
    cd rdt-1b-galbot && python ../inference_galbot_golf/precompute_lang_embed.py
"""

import os
import sys

import torch
import yaml

RDT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rdt-1b-galbot")
RDT_ROOT = os.path.normpath(RDT_ROOT)
sys.path.insert(0, RDT_ROOT)

from models.multimodal_encoder.t5_encoder import T5Embedder  # noqa: E402

GPU = 0
MODEL_PATH = os.path.join(RDT_ROOT, "t5-v1_1-xxl")
CONFIG_PATH = os.path.join(RDT_ROOT, "configs", "base.yaml")
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lang_embed.pt")

# Fill this in with the actual instruction for the golf task before running.
TASK_NAME = "galbot_golf"
INSTRUCTION = "Move forward"

# If GPU VRAM is less than 24GB, set this to an existing directory to enable offloading.
OFFLOAD_DIR = None


def main():
    if INSTRUCTION.startswith("<fill in"):
        raise ValueError("Set INSTRUCTION to the actual task instruction before running.")

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
