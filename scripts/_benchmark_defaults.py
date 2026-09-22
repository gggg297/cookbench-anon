from __future__ import annotations

import re


CURRENT_BENCHMARK_LABEL = "benchmark_133"
CURRENT_BENCHMARK_RANGE_SPEC = "1-131,137,138"
CURRENT_BENCHMARK_EXCLUDED_IDS = (132, 133, 134, 135, 136)


def parse_dish_range(spec: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    text = str(spec or "").strip()
    if not text:
        return out
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            start = int(m.group(1))
            end = int(m.group(2))
            step = 1 if end >= start else -1
            for dish_id in range(start, end + step, step):
                if dish_id not in seen:
                    seen.add(dish_id)
                    out.append(dish_id)
            continue
        if re.fullmatch(r"\d+", part):
            dish_id = int(part)
            if dish_id not in seen:
                seen.add(dish_id)
                out.append(dish_id)
            continue
        raise ValueError(f"invalid dish range segment: {part!r}")
    return out


def format_dish_range(dish_ids: list[int]) -> str:
    if not dish_ids:
        return ""
    if len(dish_ids) == 1:
        return str(dish_ids[0])
    if any(dish_ids[i] >= dish_ids[i + 1] for i in range(len(dish_ids) - 1)):
        return ",".join(str(x) for x in dish_ids)
    parts: list[str] = []
    start = dish_ids[0]
    prev = dish_ids[0]
    for dish_id in dish_ids[1:]:
        if dish_id == prev + 1:
            prev = dish_id
            continue
        parts.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = dish_id
    parts.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(parts)


def current_benchmark_dish_ids() -> list[int]:
    return parse_dish_range(CURRENT_BENCHMARK_RANGE_SPEC)


def current_benchmark_size() -> int:
    return len(current_benchmark_dish_ids())

