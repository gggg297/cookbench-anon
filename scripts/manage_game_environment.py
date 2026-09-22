#!/usr/bin/env python3
"""Inspect, deploy, and verify the CookBench Cooking Simulator environment."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


APP_ID = "641320"
DEPOT_ID = "641321"
MANIFEST_ID = "8794508976528179697"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOCK = SCRIPT_DIR.parent / "configs" / "cookbench_game_lock.json"
STEAMCMD_DIR = Path(os.environ.get("COOKBENCH_STEAMCMD_DIR", SCRIPT_DIR.parent / ".steamcmd"))
TARGET_ENV_DIR = Path(os.environ.get("COOKBENCH_GAME_DIR", SCRIPT_DIR.parent / ".game_env"))

STEAMCMD_URLS = {
    "Windows": "https://steamcdn-a.akamaihd.net/client/installer/steamcmd.zip",
    "Linux": "https://steamcdn-a.akamaihd.net/client/installer/steamcmd_linux.tar.gz",
}
PROMPT_PATTERNS = (
    re.compile(r"steam\s*guard", re.I),
    re.compile(r"two[- ]factor.*code", re.I),
    re.compile(r"auth(?:entication)?\s*code", re.I),
)
PASSWORD_PATTERNS = (
    re.compile(r"password\s*[:>]", re.I),
    re.compile(r"enter.*password", re.I),
)
FAILURE_PATTERNS = (
    "FAILED (Invalid Password)",
    "Invalid Password",
    "Access Denied",
    "No subscription",
    "ERROR! Download item",
)


class DeploymentError(RuntimeError):
    pass


def load_lock(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentError(f"Cannot read lock file {path}: {exc}") from exc
    for key in ("app_id", "build_id", "required_depots"):
        if not data.get(key):
            raise DeploymentError(f"Lock file is missing {key!r}")
    return data


def steamcmd_executable(root: Path) -> Path:
    return root / ("steamcmd.exe" if os.name == "nt" else "steamcmd.sh")


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()

    def validate_paths(names: list[str]) -> None:
        for name in names:
            target = (destination / name).resolve()
            if destination != target and destination not in target.parents:
                raise DeploymentError(f"Unsafe archive path: {name}")

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            names = [item.filename for item in members]
            for item in members:
                if item.is_dir():
                    continue
                mode = item.external_attr >> 16
                if mode and (mode & 0o170000) == 0o120000:
                    raise DeploymentError("SteamCMD archive contains a symbolic link")
            validate_paths(names)
            bundle.extractall(destination)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            names = [item.name for item in members]
            if any(not (item.isfile() or item.isdir()) for item in members):
                raise DeploymentError("SteamCMD archive contains a link or special file")
            validate_paths(names)
            bundle.extractall(destination, members=members)
    else:
        raise DeploymentError("Downloaded SteamCMD archive has an unknown format")


def ensure_steamcmd(root: Path) -> Path:
    executable = steamcmd_executable(root)
    if executable.is_file():
        return executable
    system = platform.system()
    url = STEAMCMD_URLS.get(system)
    if not url:
        raise DeploymentError(f"SteamCMD automatic setup is unsupported on {system}")
    root.mkdir(parents=True, exist_ok=True)
    suffix = ".zip" if system == "Windows" else ".tar.gz"
    archive = root / f"steamcmd{suffix}"
    print(f"Downloading SteamCMD from Valve: {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        safe_extract(archive, root)
    except (OSError, urllib.error.URLError, zipfile.BadZipFile, tarfile.TarError) as exc:
        raise DeploymentError(f"SteamCMD setup failed: {exc}") from exc
    finally:
        archive.unlink(missing_ok=True)
    if not executable.is_file():
        raise DeploymentError(f"SteamCMD executable was not created at {executable}")
    if os.name != "nt":
        executable.chmod(executable.stat().st_mode | 0o111)
    return executable


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_reference_files(root: Path, expected: dict[str, str], label: str) -> list[str]:
    failures = []
    for relative, expected_hash in expected.items():
        path = root / Path(relative)
        if not path.is_file():
            failures.append(f"{label}: missing {relative}")
            continue
        actual = hash_file(path)
        if actual.lower() != expected_hash.lower():
            failures.append(f"{label}: hash mismatch for {relative}: {actual}")
    return failures


def write_inventory(root: Path, output: Path, lock: dict) -> None:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        files.append({"path": relative, "size": path.stat().st_size, "sha256": hash_file(path)})
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "app_id": lock["app_id"],
        "build_id": lock["build_id"],
        "depots": lock["required_depots"],
        "files": files,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def disk_anchor(path: Path) -> Path:
    anchor = path
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    return anchor


def check_deployment_space(download_root: Path, target: Path, depot_bytes: int) -> None:
    download_anchor = disk_anchor(download_root)
    target_anchor = disk_anchor(target)
    same_volume = download_anchor.stat().st_dev == target_anchor.stat().st_dev
    download_required = int(depot_bytes * 1.2)
    target_required = int(depot_bytes * 1.2)
    if same_volume:
        required = download_required + target_required
        free = shutil.disk_usage(target_anchor).free
        if free < required:
            raise DeploymentError(
                f"Insufficient disk space: download and deployment share a volume; "
                f"need about {required:,} bytes, have {free:,}"
            )
        return
    download_free = shutil.disk_usage(download_anchor).free
    target_free = shutil.disk_usage(target_anchor).free
    if download_free < download_required:
        raise DeploymentError(
            f"Insufficient SteamCMD disk space: need about {download_required:,} bytes, "
            f"have {download_free:,}"
        )
    if target_free < target_required:
        raise DeploymentError(
            f"Insufficient target disk space: need about {target_required:,} bytes, "
            f"have {target_free:,}"
        )


def _stream_output(process: subprocess.Popen[str], output_queue: queue.Queue[str]) -> None:
    assert process.stdout is not None
    while True:
        char = process.stdout.read(1)
        if not char:
            break
        sys.stdout.write(char)
        sys.stdout.flush()
        output_queue.put(char)


def run_steamcmd(executable: Path, username: str, password: str, depots: dict) -> None:
    process = subprocess.Popen(
        [str(executable)],
        cwd=executable.parent,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=0,
    )
    assert process.stdin is not None
    output_queue: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(target=_stream_output, args=(process, output_queue), daemon=True)
    reader.start()

    def send(command: str) -> None:
        if process.poll() is not None:
            raise DeploymentError(f"SteamCMD exited unexpectedly with code {process.returncode}")
        process.stdin.write(command + "\n")
        process.stdin.flush()

    if '"' in username or "\n" in username or "\r" in username:
        raise DeploymentError("Steam username contains unsupported characters")
    send(f'login "{username}"')
    password_sent = False
    commands_sent = False
    guard_requested = False
    buffer = ""
    completed: set[str] = set()
    deadline = time.monotonic() + 24 * 60 * 60
    try:
        while process.poll() is None:
            if time.monotonic() > deadline:
                raise DeploymentError("SteamCMD timed out after 24 hours")
            try:
                buffer = (buffer + output_queue.get(timeout=0.2))[-8192:]
            except queue.Empty:
                continue
            if not password_sent and any(pattern.search(buffer) for pattern in PASSWORD_PATTERNS):
                send(password)
                password = ""
                password_sent = True
                buffer = ""
                continue
            if not guard_requested and any(pattern.search(buffer) for pattern in PROMPT_PATTERNS):
                guard_requested = True
                send(getpass.getpass("Steam Guard code: "))
                buffer = ""
                continue
            if not commands_sent and ("Logged in OK" in buffer or "Waiting for user info...OK" in buffer):
                for depot_id, record in depots.items():
                    send(f'download_depot {APP_ID} {depot_id} {record["manifest_id"]}')
                commands_sent = True
                buffer = ""
                continue
            match = re.search(r"Depot download complete.*depot_(\d+)", buffer, re.I)
            if match:
                completed.add(match.group(1))
                buffer = ""
                if completed == set(depots):
                    send("quit")
            if any(marker in buffer for marker in FAILURE_PATTERNS):
                raise DeploymentError("SteamCMD reported an authentication or download failure")
        reader.join(timeout=5)
    finally:
        if process.poll() is None:
            try:
                send("quit")
                process.wait(timeout=10)
            except Exception:
                process.terminate()
    if process.returncode not in (0, None):
        raise DeploymentError(f"SteamCMD exited with code {process.returncode}")
    if completed != set(depots):
        missing = sorted(set(depots) - completed)
        raise DeploymentError(f"SteamCMD did not confirm completion for depots: {missing}")


def deploy(args: argparse.Namespace) -> None:
    lock = load_lock(args.lock)
    depots = dict(lock["required_depots"])
    if args.include_observed_dlc:
        depots.update(lock.get("observed_optional_depots", {}))
    check_deployment_space(
        args.steamcmd_dir,
        args.target,
        sum(int(item["size"]) for item in depots.values()),
    )
    executable = ensure_steamcmd(args.steamcmd_dir)
    username = args.username or input("Steam username: ").strip()
    if not username:
        raise DeploymentError("Steam username is required")
    password = getpass.getpass("Steam password: ")
    run_steamcmd(executable, username, password, depots)

    stage = args.target.with_name(args.target.name + ".staging")
    if stage.exists():
        raise DeploymentError(f"Staging directory already exists: {stage}")
    stage.mkdir(parents=True)
    try:
        for depot_id in depots:
            source = args.steamcmd_dir / "steamapps" / "content" / f"app_{APP_ID}" / f"depot_{depot_id}"
            if not source.is_dir():
                raise DeploymentError(f"Downloaded depot directory is missing: {source}")
            shutil.copytree(source, stage, dirs_exist_ok=True)
        failures = verify_reference_files(stage, lock.get("reference_files", {}), "base game")
        if failures:
            raise DeploymentError("Downloaded files do not match the lock:\n" + "\n".join(failures))
        write_inventory(stage, stage / "cookbench-deployment.json", lock)
        if args.target.exists():
            raise DeploymentError(f"Target already exists; refusing to overwrite: {args.target}")
        stage.replace(args.target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"CookBench game environment deployed to {args.target}")


def verify(args: argparse.Namespace) -> None:
    lock = load_lock(args.lock)
    failures = verify_reference_files(args.target, lock.get("reference_files", {}), "base game")
    if args.with_mod:
        failures.extend(verify_reference_files(args.target, lock.get("mod_reference_files", {}), "mod layer"))
    if failures:
        raise DeploymentError("Verification failed:\n" + "\n".join(failures))
    inventory = args.target / "cookbench-deployment.json"
    if inventory.is_file():
        for record in json.loads(inventory.read_text(encoding="utf-8"))["files"]:
            path = (args.target / record["path"]).resolve()
            if args.target.resolve() not in path.parents:
                raise DeploymentError("Unsafe inventory path")
            if not path.is_file() or hash_file(path) != record["sha256"]:
                raise DeploymentError(f"Inventory mismatch: {record['path']}")
        print("Deployment inventory hashes verified.")
    print(f"Reference hashes match lock for build {lock['build_id']} at {args.target}")


def depot_snapshot(source: Path) -> dict:
    result = {}
    if source.exists():
        for path in source.rglob("*"):
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                raise DeploymentError(f"Unexpected link in depot: {path}")
            try:
                if path.is_file():
                    stat = path.stat()
                    result[path.relative_to(source).as_posix()] = (stat.st_size, stat.st_mtime_ns)
            except FileNotFoundError:
                continue
    return result


def watch_download(args: argparse.Namespace) -> None:
    if args.interval < 1:
        raise DeploymentError("Interval must be at least one second")
    previous = None
    print(f"Watching {args.source}\nLogical file sizes only; preallocation is possible. Ctrl+C stops monitoring, not Steam.")
    while True:
        snapshot = depot_snapshot(args.source)
        total = sum(size for size, _ in snapshot.values())
        changed = 0 if previous is None else sum(previous.get(k) != v for k, v in snapshot.items())
        print(f"{datetime.now():%H:%M:%S} files={len(snapshot)} logical_bytes={total:,} changed_files={changed}; completion: check Steam console", flush=True)
        if args.once:
            return
        previous = snapshot
        time.sleep(args.interval)


def import_depot(args: argparse.Namespace) -> None:
    lock = load_lock(args.lock)
    if not args.download_complete:
        raise DeploymentError("Wait for Steam's 'Depot download complete', then pass --download-complete")
    if set(lock["required_depots"]) != {DEPOT_ID}:
        raise DeploymentError("Client import currently supports the single base depot only")
    source, target = args.source.resolve(), args.target.resolve()
    if target == source or source in target.parents or target in source.parents:
        raise DeploymentError("Source and target must be separate, non-nested directories")
    stage = target.with_name(target.name + ".staging")
    if target.exists() or stage.exists():
        raise DeploymentError("Target or staging directory already exists; refusing to overwrite")
    before = depot_snapshot(source)
    total = sum(size for size, _ in before.values())
    expected = int(lock["required_depots"][DEPOT_ID]["size"])
    if total != expected:
        raise DeploymentError(f"Depot size mismatch: expected {expected:,}, observed {total:,}; download may be incomplete or contain stale files")
    failures = verify_reference_files(source, lock.get("reference_files", {}), "source")
    if failures:
        raise DeploymentError("\n".join(failures))
    if shutil.disk_usage(disk_anchor(target)).free < int(total * 1.2):
        raise DeploymentError("Insufficient target disk space")
    stage.mkdir(parents=True)
    try:
        records = []
        for index, relative in enumerate(sorted(before), 1):
            src, dst = source / relative, stage / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            digest = hash_file(src)
            shutil.copy2(src, dst)
            if hash_file(dst) != digest:
                raise DeploymentError(f"Copy verification failed: {relative}")
            records.append({"path": relative, "size": before[relative][0], "sha256": digest})
            if index % 100 == 0 or index == len(before):
                print(f"Copied and verified {index}/{len(before)} files", flush=True)
        if depot_snapshot(source) != before:
            raise DeploymentError("Source changed during import; wait for Steam to finish")
        inventory = {"schema_version": 1, "app_id": lock["app_id"], "build_id": lock["build_id"],
                     "depots": lock["required_depots"], "source": "steam-client-user-confirmed",
                     "created_at": datetime.now(timezone.utc).isoformat(), "files": records}
        (stage / "cookbench-deployment.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
        stage.rename(target)
    except BaseException:
        print(f"Import incomplete. Staging retained for inspection: {stage}", file=sys.stderr)
        raise
    print(f"Base game imported and copy verified: {target}. Mod installation and launch acceptance remain separate.")


def inspect_local(args: argparse.Namespace) -> None:
    candidates = []
    if args.steam_root:
        candidates.append(args.steam_root)
    if os.name == "nt":
        candidates.extend((Path("C:/Program Files (x86)/Steam"), Path("C:/Program Files/Steam")))
        extra = os.environ.get("STEAM_LIBRARY_ROOT", "").strip()
        if extra:
            candidates.append(Path(extra))
    seen = set()
    for root in candidates:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        manifest = root / "steamapps" / f"appmanifest_{APP_ID}.acf"
        if manifest.is_file():
            text = manifest.read_text(encoding="utf-8", errors="replace")
            field = lambda name: (re.search(rf'"{re.escape(name)}"\s+"([^"]+)"', text) or [None, None])[1]
            print(json.dumps({
                "manifest": str(manifest),
                "build_id": field("buildid"),
                "install_dir": str(root / "steamapps" / "common" / (field("installdir") or "")),
                "locked_manifest_id": MANIFEST_ID,
            }, indent=2))
            return
    raise DeploymentError("Cooking Simulator appmanifest_641320.acf was not found")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(lock=DEFAULT_LOCK, steamcmd_dir=STEAMCMD_DIR, target=TARGET_ENV_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_source = Path(
        os.environ.get(
            "STEAM_DEPOT_SOURCE",
            "C:/Program Files (x86)/Steam/steamapps/content/app_641320/depot_641321",
        )
    )
    watch_parser = subparsers.add_parser("watch-download", help="observe Steam client depot file activity")
    watch_parser.add_argument("--source", type=Path, default=default_source)
    watch_parser.add_argument("--interval", type=float, default=5)
    watch_parser.add_argument("--once", action="store_true")
    watch_parser.set_defaults(handler=watch_download)
    import_parser = subparsers.add_parser("import-depot", help="import a completed Steam client base depot")
    import_parser.add_argument("--source", type=Path, default=default_source)
    import_parser.add_argument("--target", type=Path, required=True)
    import_parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    import_parser.add_argument("--download-complete", action="store_true")
    import_parser.set_defaults(handler=import_depot)
    inspect_parser = subparsers.add_parser("inspect", help="read the locally installed Steam build")
    inspect_parser.add_argument("--steam-root", type=Path)
    inspect_parser.set_defaults(handler=inspect_local)
    deploy_parser = subparsers.add_parser("deploy", help="download and assemble the locked build")
    deploy_parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    deploy_parser.add_argument("--steamcmd-dir", type=Path, default=STEAMCMD_DIR)
    deploy_parser.add_argument("--target", type=Path, default=TARGET_ENV_DIR)
    deploy_parser.add_argument("--username")
    deploy_parser.add_argument("--include-observed-dlc", action="store_true")
    deploy_parser.set_defaults(handler=deploy)
    verify_parser = subparsers.add_parser("verify", help="verify locked reference files")
    verify_parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    verify_parser.add_argument("--target", type=Path, default=TARGET_ENV_DIR)
    verify_parser.add_argument("--with-mod", action="store_true")
    verify_parser.set_defaults(handler=verify)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.handler(args)
        return 0
    except KeyboardInterrupt:
        print("Stopped. Steam client downloads are not cancelled.")
        return 130
    except (DeploymentError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
