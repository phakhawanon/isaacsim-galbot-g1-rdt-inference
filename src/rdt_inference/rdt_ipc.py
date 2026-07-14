"""Wire protocol shared between main.py (isaaclab env, client) and rdt_server.py
(rdt env, server) -- and, eventually, a real-robot client running some other env
entirely. Deliberately pure-stdlib + PIL only, so it imports unmodified regardless
of which conda env or which torch/numpy build the caller has installed.
"""

import io
import pickle
import struct

from PIL import Image

# Server default: bind on all interfaces so other machines (e.g. a real robot's
# control box) can reach it, not just localhost.
DEFAULT_BIND_HOST = "0.0.0.0"
# Client default: only localhost unless RDT_SERVER_ADDR points elsewhere.
DEFAULT_CONNECT_HOST = "localhost"
DEFAULT_PORT = 8765

CHUNK_SIZE = 64
RPC_TIMEOUT_S = 120.0

_PICKLE_PROTOCOL = 5  # pinned explicitly so differing Python patch versions never disagree


def parse_addr(env_value: str | None, default: str) -> tuple[str, int]:
    """Parses a "host:port" string (e.g. from an env var) into (host, port)."""
    host, _, port = (env_value or default).rpartition(":")
    return host, int(port)


def send_msg(sock, obj) -> None:
    payload = pickle.dumps(obj, protocol=_PICKLE_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


def _recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed while receiving")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock):
    (length,) = struct.unpack("!Q", _recv_exact(sock, 8))
    return pickle.loads(_recv_exact(sock, length))


def encode_image(image: Image.Image | None, quality: int = 90) -> bytes | None:
    """Encodes a PIL Image as JPEG bytes for cheap transport over a real network link."""
    if image is None:
        return None
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def decode_image(data: bytes | None) -> Image.Image | None:
    if data is None:
        return None
    return Image.open(io.BytesIO(data)).convert("RGB")
