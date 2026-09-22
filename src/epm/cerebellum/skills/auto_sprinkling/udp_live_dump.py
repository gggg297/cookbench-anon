"""
Live UDP dump for CS_CamDump Alt+J interaction stream.

Purpose: verify whether packets are received on 127.0.0.1:52525 and whether they
contain a non-empty `PourAmount:` field.

Usage:
  python udp_live_dump.py
  python udp_live_dump.py --port 52525 --bind 127.0.0.1
"""

from __future__ import annotations

import argparse
import socket
import sys
import time


def _parse_kv_block(payload: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in payload.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bind", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=52525, help="UDP port (default: 52525)")
    p.add_argument("--timeout", type=float, default=1.0, help="Socket timeout seconds (default: 1.0)")
    p.add_argument("--once", action="store_true", help="Exit after first packet")
    args = p.parse_args(argv)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((args.bind, args.port))
    except OSError as e:
        print(f"[udp] bind failed: {args.bind}:{args.port} -> {e}")
        return 2

    sock.settimeout(float(args.timeout))
    print(f"[udp] listening on {args.bind}:{args.port} (timeout={args.timeout}s)")
    print("[udp] tip: in-game press `Alt+J` and trigger a Pouring popup")

    received = 0
    while True:
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[udp] recv error: {e}")
            continue

        received += 1
        try:
            payload = data.decode("utf-8", errors="replace")
        except Exception:
            payload = repr(data)

        kv = _parse_kv_block(payload)
        ts = kv.get("Timestamp", "")
        item = kv.get("ItemName", "")
        action = kv.get("Action", "")
        pour = kv.get("PourAmount", "")
        weight = kv.get("Weight", "")
        container = kv.get("ContainerName", "")

        print(
            f"[udp#{received}] from {addr[0]}:{addr[1]} ts='{ts}' item='{item}' action='{action}' "
            f"pour='{pour}' weight='{weight}' container='{container}'"
        )

        if args.once:
            break

    print(f"[udp] done (received={received})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

