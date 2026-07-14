"""Persistent RDT-1B inference server for the Galbot golf task.

Run this standalone, in the `rdt` conda env (matching rdt-1b-galbot/README.md's
environment) -- NOT through isaaclab.sh, and NOT in the `isaaclab` conda env:

    conda activate rdt
    ./scripts/start_rdt_server.sh

(uses a relative import for its rdt_ipc sibling, so it must be launched as a module --
the wrapper script above handles that; don't run this file directly with `python
rdt_server.py`.)

main.py (running in the isaaclab env) connects to this server over TCP and never
imports any RDT-specific package directly, so the two conda environments'
conflicting torch/CUDA versions never need to coexist in one process. Because the
transport is a plain TCP socket (not a Unix domain socket), any other client on the
network -- e.g. a real robot's control process, later -- can talk to this same
server by pointing at its host:port.

Serves one client connection at a time; a client disconnecting (or being restarted)
does not require restarting this server.
"""

import os
import socket
import sys
import time

import torch
import yaml

from .rdt_ipc import (  # noqa: E402
    CHUNK_SIZE,
    DEFAULT_BIND_HOST,
    DEFAULT_PORT,
    decode_image,
    parse_addr,
    recv_msg,
    send_msg,
)

RDT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "rdt-1b-galbot")
)
sys.path.insert(0, RDT_ROOT)

from scripts.galbot_model import create_model  # noqa: E402



# Order the 6 image slots are decoded in -- must match the key names main.py sends.
IMAGE_KEYS = [
    "head_prev", "right_wrist_prev", "left_wrist_prev",
    "head_cur", "right_wrist_cur", "left_wrist_cur",
]


def load_policy():
    with open(os.path.join(RDT_ROOT, "configs", "base.yaml"), "r") as fp:
        rdt_config = yaml.safe_load(fp)

    assert rdt_config["common"]["action_chunk_size"] == CHUNK_SIZE, (
        "configs/base.yaml's action_chunk_size no longer matches rdt_ipc.CHUNK_SIZE -- "
        "update rdt_ipc.CHUNK_SIZE to match."
    )

    policy = create_model(
        args=rdt_config,
        dtype=torch.bfloat16,
        pretrained=os.path.join(RDT_ROOT, "checkpoints", "rdt-1b-finetune", "checkpoint-70000"),
        pretrained_vision_encoder_name_or_path="google/siglip-so400m-patch14-384",
        control_frequency=30,
    )

    lang_embed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lang_embed.pt")
    lang_embeds = torch.load(lang_embed_path, map_location="cpu")
    # precompute_lang_embed.py saves {"embeddings": (1, seq_len, 4096)}; encode_lang_hdf5.py
    # (per-datapoint HDF5 preprocessing) saves the raw (seq_len, 4096) tensor with no batch
    # dim and no dict wrapper -- accept either.
    if isinstance(lang_embeds, dict):
        lang_embeds = lang_embeds["embeddings"]
    if lang_embeds.dim() == 2:
        lang_embeds = lang_embeds.unsqueeze(0)

    return policy, lang_embeds


def serve_client(conn, policy, lang_embeds):
    while True:
        req = recv_msg(conn)  # raises ConnectionError when the client disconnects

        proprio = torch.tensor(req["proprio"], dtype=torch.float32).unsqueeze(0)
        images = [decode_image(req["images"][key]) for key in IMAGE_KEYS]

        t0 = time.time()
        action_chunk = policy.step(
            proprio=proprio, images=images, text_embeds=lang_embeds,
        ).squeeze(0).cpu().tolist()
        print(f"[RDT Server] inference took {time.time() - t0:.3f}s")

        send_msg(conn, {"action_chunk": action_chunk})


def main():
    print("[RDT Server] loading policy...")
    policy, lang_embeds = load_policy()
    print("[RDT Server] policy loaded.")

    host, port = parse_addr(os.environ.get("RDT_SERVER_ADDR"), f"{DEFAULT_BIND_HOST}:{DEFAULT_PORT}")

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(1)
    print(f"[RDT Server] ready, listening on {host}:{port}")

    try:
        while True:
            conn, addr = server_sock.accept()
            print(f"[RDT Server] client connected from {addr}")
            try:
                serve_client(conn, policy, lang_embeds)
            except ConnectionError:
                print(f"[RDT Server] client {addr} disconnected")
            finally:
                conn.close()
    finally:
        server_sock.close()


if __name__ == "__main__":
    main()
