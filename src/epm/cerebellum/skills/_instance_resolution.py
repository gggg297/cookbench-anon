from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from epm.cerebellum.cookbench_api import ActionResult
from epm.cerebellum.realtime_products import extract_items, item_display_name, read_realtime_products


def _norm(s: Any) -> str:
    return (str(s or "")).strip().lower()


def _safe_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


@dataclass(frozen=True)
class MappingEntry:
    object_id: int
    name_cn: str
    name_en: str
    category: str


def _repo_root() -> Path:
    """Locate the repository root, tolerating either checkout layout."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data").is_dir() and (parent / "src").is_dir():
            return parent
    # <repo>/src/epm/cerebellum/skills/_instance_resolution.py -> parents[4] == <repo>
    return here.parents[4]


def _mapping_path() -> Path:
    return _repo_root() / "data" / "object_en_ch_mapping.txt"


def _norm_name_for_match(s: Any) -> str:
    # conservative normalization: keep spacing for substring matching, lower-case, collapse whitespace.
    raw = (str(s or "")).strip().lower()
    if not raw:
        return ""
    raw = raw.replace("_", " ")
    raw = " ".join(raw.split())
    return raw


def _norm_en_for_distance(s: Any) -> str:
    # distance normalization: compare on alnum only to tolerate minor punctuation differences.
    raw = _norm_name_for_match(s)
    return "".join(ch for ch in raw if ch.isalnum())


def _edit_distance_leq1(a: str, b: str) -> bool:
    """
    Check Levenshtein distance <= 1 with early exits (fast for our use-case).
    """
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    # Ensure a is the shorter/equal string.
    if la > lb:
        a, b = b, a
        la, lb = lb, la

    # Same length: at most 1 substitution.
    if la == lb:
        diff = 0
        for ca, cb in zip(a, b):
            if ca != cb:
                diff += 1
                if diff > 1:
                    return False
        return True

    # Length differs by 1: at most 1 insertion/deletion.
    i = j = 0
    diff = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        diff += 1
        if diff > 1:
            return False
        j += 1  # skip one char in longer string
    return True


@lru_cache(maxsize=1)
def _load_object_mapping() -> tuple[list[MappingEntry], Optional[str]]:
    path = _mapping_path()
    if not path.exists():
        return [], f"missing:{path}"
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except Exception as e:
        return [], f"unreadable:{e}"

    out: list[MappingEntry] = []
    for line in lines:
        s = (line or "").strip()
        if not s or s.startswith("#"):
            continue
        parts = [p.strip() for p in s.split(",")]
        if len(parts) < 4:
            continue
        try:
            object_id = int(parts[0])
        except Exception:
            continue
        name_cn = parts[1]
        # English name may contain commas; it is the part between the 2nd comma and the last comma.
        name_en = ",".join(parts[2:-1]).strip()
        category = parts[-1].strip().lower()
        out.append(MappingEntry(object_id=object_id, name_cn=name_cn, name_en=name_en, category=category))
    return out, None


def _mapping_candidates(name: str) -> tuple[Optional[MappingEntry], list[MappingEntry], Optional[str]]:
    """
    Return (best_match, suggestions, mapping_error).

    Matching policy (English):
      - exact (case-insensitive)
      - plural heuristic (trail 's'/'es')
      - edit distance <= 1 on alnum-normalized
      - substring contains

    Also tries exact match on Chinese name.
    """
    entries, mapping_err = _load_object_mapping()
    want = _norm_name_for_match(name)
    if not want:
        return None, [], mapping_err

    def _maybe_singular(s: str) -> str:
        if s.endswith("es") and len(s) > 2:
            return s[:-2]
        if s.endswith("s") and len(s) > 1:
            return s[:-1]
        return s

    # Heuristic singular/plural tolerance for English inputs.
    want_singular = _maybe_singular(want)

    want_dist = _norm_en_for_distance(want)
    want_singular_dist = _norm_en_for_distance(want_singular)

    # Pass 1: exact CN / exact EN (with plural tolerance).
    exact: list[MappingEntry] = []
    for e in entries:
        if _norm_name_for_match(e.name_cn) == want:
            exact.append(e)
            continue
        en = _norm_name_for_match(e.name_en)
        if en == want or en == want_singular or _maybe_singular(en) == want or _maybe_singular(en) == want_singular:
            exact.append(e)
            continue
    if len(exact) == 1:
        return exact[0], [], mapping_err
    if len(exact) > 1:
        return None, exact[:10], mapping_err

    # Pass 2: plural/singular exact on EN
    sing_exact: list[MappingEntry] = []
    if want_singular != want:
        for e in entries:
            en = _norm_name_for_match(e.name_en)
            if en == want_singular or _maybe_singular(en) == want_singular:
                sing_exact.append(e)
    if len(sing_exact) == 1:
        return sing_exact[0], [], mapping_err
    if len(sing_exact) > 1:
        return None, sing_exact[:10], mapping_err

    # Pass 3: edit distance <=1 on EN (alnum-normalized)
    near: list[MappingEntry] = []
    if want_dist:
        for e in entries:
            en_d = _norm_en_for_distance(e.name_en)
            if not en_d:
                continue
            if _edit_distance_leq1(en_d, want_dist) or (want_singular_dist and _edit_distance_leq1(en_d, want_singular_dist)):
                near.append(e)
    if len(near) == 1:
        return near[0], [], mapping_err
    if len(near) > 1:
        return None, near[:10], mapping_err

    # Pass 4: substring suggestions (EN)
    subs: list[MappingEntry] = []
    for e in entries:
        en = _norm_name_for_match(e.name_en)
        if not en:
            continue
        if want in en or en in want:
            subs.append(e)
            if len(subs) >= 10:
                break
    return None, subs, mapping_err


def _all_mapping_english_names_lower() -> tuple[list[str], Optional[str]]:
    entries, mapping_err = _load_object_mapping()
    names: list[str] = []
    for e in entries:
        en = _norm_name_for_match(e.name_en)
        if en:
            names.append(en)
    # De-dup while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out, mapping_err


def _candidates_by_name(items: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    want = _norm(name)
    if not want:
        return []

    def _maybe_singular(s: str) -> str:
        if s.endswith("es") and len(s) > 2:
            return s[:-2]
        if s.endswith("s") and len(s) > 1:
            return s[:-1]
        return s

    want_singular = _maybe_singular(want)
    keys = ("name_en", "name_cn", "name", "label", "game_object")

    def _exact(it: dict[str, Any]) -> bool:
        disp = _norm(item_display_name(it))
        if disp == want or disp == want_singular or _maybe_singular(disp) == want or _maybe_singular(disp) == want_singular:
            return True
        for k in keys:
            v = _norm(it.get(k))
            if v == want or v == want_singular or _maybe_singular(v) == want or _maybe_singular(v) == want_singular:
                return True
        return False

    exact = [it for it in items if isinstance(it, dict) and _exact(it)]
    if exact:
        return exact

    def _sub(it: dict[str, Any]) -> bool:
        disp = _norm(item_display_name(it))
        if want in disp or (want_singular and want_singular in disp):
            return True
        for k in keys:
            v = _norm(it.get(k))
            if want in v or (want_singular and want_singular in v):
                return True
        return False

    return [it for it in items if isinstance(it, dict) and _sub(it)]


def _instance_choice_result(
    *,
    action: str,
    name_arg: str,
    instance_arg: str,
    name_value: str,
    candidates: list[dict[str, Any]],
    error: str,
    provided_instance_id: Optional[int] = None,
    kind_required: Optional[str] = None,
) -> ActionResult:
    kind_note = f" (restricted to kind={kind_required!r})" if kind_required else ""
    if error == "instance_id_not_found" and provided_instance_id is not None:
        selection_prompt = (
            f"Invalid instance selection: provided {instance_arg}={int(provided_instance_id)} does not match any "
            f"item with name={name_value!r}{kind_note} in realtime_products.json. This usually means the instance_id is wrong "
            f"or the name/instance_id pair is inconsistent. Choose exactly one `instance_id` from `candidates`, then "
            f"retry with `{name_arg}={name_value!r}` and `{instance_arg}=<chosen_instance_id>`."
        )
    else:
        selection_prompt = (
            f"Ambiguous target: multiple items match name={name_value!r}{kind_note} in realtime_products.json. "
            f"You must choose exactly one `instance_id` from `candidates`, then call the skill again with "
            f"`{name_arg}={name_value!r}` and `{instance_arg}=<chosen_instance_id>`."
        )

    raw: dict[str, Any] = {
        "action": str(action),
        "name_arg": str(name_arg),
        "instance_arg": str(instance_arg),
        "name": str(name_value),
        "reason": str(error),
        "kind_required": (str(kind_required) if kind_required else None),
        # Keep candidates as the raw dicts from realtime_products.json (schemas vary by kind).
        # De-dup by instance_id when possible (realtime scans can contain repeats).
        "candidates": _dedup_candidates([it for it in candidates if isinstance(it, dict)]),
        "selection_prompt": selection_prompt,
    }
    if provided_instance_id is not None:
        raw["provided_instance_id"] = int(provided_instance_id)

    return ActionResult(False, raw=raw, error=str(error))


@dataclass(frozen=True)
class ResolvedInstance:
    name: str
    instance_id: Optional[int]
    candidates: list[dict[str, Any]]


def _dedup_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    De-duplicate candidates while keeping the original dict payload.

    Primary key: instance_id (when present/int-convertible).
    Fallback: stable JSON dump (sorted keys) for entries without instance_id.
    """
    out: list[dict[str, Any]] = []
    seen_iid: set[int] = set()
    seen_dump: set[str] = set()
    for it in candidates:
        iid = _safe_int(it.get("instance_id"))
        if iid is not None:
            if iid in seen_iid:
                continue
            seen_iid.add(iid)
            out.append(it)
            continue
        try:
            s = json.dumps(it, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            s = repr(sorted(it.items(), key=lambda kv: str(kv[0])))
        if s in seen_dump:
            continue
        seen_dump.add(s)
        out.append(it)
    return out


def resolve_instance_id(
    *,
    realtime_products_path: Path,
    name: str,
    instance_id: Optional[int],
    action: str,
    name_arg: str,
    instance_arg: str,
    kind_required: Optional[str] = None,
) -> tuple[Optional[ResolvedInstance], Optional[ActionResult]]:
    """
    Resolve an entity by (name, instance_id) using realtime_products.json.

    Rules:
    - If no candidates -> error target_not_found.
    - If one candidate and instance_id is None -> auto-fill instance_id when available.
    - If multiple candidates and instance_id is None -> error instance_id_required with candidate list.
    - If instance_id is provided -> it must exist among candidates; otherwise instance_id_not_found.
    """
    try:
        data = read_realtime_products(realtime_products_path)
        items = extract_items(data)
    except Exception as e:
        return None, ActionResult(False, raw={"path": str(realtime_products_path)}, error=f"realtime_products_unreadable:{e}")

    candidates = _dedup_candidates(_candidates_by_name(items, name))
    if candidates and kind_required:
        want_kind = _norm(kind_required)
        filtered = [it for it in candidates if _norm(it.get("kind")) == want_kind]
        if not filtered:
            kinds = sorted({_norm(it.get("kind")) for it in candidates if _norm(it.get("kind"))})
            return None, ActionResult(
                False,
                raw={
                    "name": str(name),
                    "action": str(action),
                    "kind_required": str(kind_required),
                    "matched_kinds": kinds,
                    # include a few examples so the model can see what it matched instead
                    "matched_examples": [it for it in candidates[:10] if isinstance(it, dict)],
                },
                error="target_kind_mismatch",
            )
        candidates = _dedup_candidates(filtered)
    if not candidates:
        # Provide semantic feedback: distinguish "name invalid" vs "name valid but not in scene".
        best, suggestions, mapping_err = _mapping_candidates(str(name))

        # If the name looks valid per mapping, try resolving again using canonical EN/CN from mapping.
        if best is not None:
            candidates2: list[dict[str, Any]] = []
            if best.name_en and _norm(best.name_en) != _norm(name):
                candidates2 = _candidates_by_name(items, best.name_en)
            if not candidates2 and best.name_cn and _norm(best.name_cn) != _norm(name):
                candidates2 = _candidates_by_name(items, best.name_cn)
            if candidates2:
                candidates = candidates2
            else:
                return None, ActionResult(
                    False,
                    raw={
                        "name": str(name),
                        "action": str(action),
                        "scene_missing": True,
                        "mapping_path": str(_mapping_path()),
                        "mapping_match": {
                            "object_id": int(best.object_id),
                            "name_cn": best.name_cn,
                            "name_en": best.name_en,
                            "category": best.category,
                        },
                        "mapping_error": mapping_err,
                    },
                    error="target_not_found",
                )
        else:
            # No mapping match: treat as invalid name (or a too-ambiguous mapping lookup).
            all_en, all_en_err = _all_mapping_english_names_lower()
            raw: dict[str, Any] = {
                "name": str(name),
                "action": str(action),
                "scene_missing": True,
                "mapping_path": str(_mapping_path()),
                "mapping_error": mapping_err or all_en_err,
            }
            # User preference: when name doesn't match, return ALL english names for selection (lowercased).
            raw["mapping_suggestions"] = all_en
            # Also include top nearby candidates (if any) for convenience.
            if suggestions:
                raw["mapping_nearby"] = [
                    {"object_id": int(s.object_id), "name_cn": s.name_cn, "name_en": s.name_en, "category": s.category}
                    for s in suggestions
                ]
            return None, ActionResult(False, raw=raw, error="invalid_target_name")

    if instance_id is None:
        if len(candidates) > 1:
            return None, _instance_choice_result(
                action=action,
                name_arg=name_arg,
                instance_arg=instance_arg,
                name_value=str(name),
                candidates=candidates,
                error="instance_id_required",
                kind_required=kind_required,
            )
        # Single candidate: auto-fill instance_id when present; otherwise leave None.
        picked = candidates[0]
        try:
            iid = int(picked.get("instance_id")) if picked.get("instance_id") is not None else None
        except Exception:
            iid = None
        return ResolvedInstance(name=str(name), instance_id=iid, candidates=candidates), None

    # instance_id provided: must match a candidate (and therefore also matches name).
    match = None
    for it in candidates:
        try:
            if int(it.get("instance_id")) == int(instance_id):  # type: ignore[arg-type]
                match = it
                break
        except Exception:
            continue
    if match is None:
        return None, _instance_choice_result(
            action=action,
            name_arg=name_arg,
            instance_arg=instance_arg,
            name_value=str(name),
            candidates=candidates,
            error="instance_id_not_found",
            provided_instance_id=int(instance_id),
            kind_required=kind_required,
        )

    return ResolvedInstance(name=str(name), instance_id=int(instance_id), candidates=candidates), None
