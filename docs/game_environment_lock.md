# CookBench Game Environment Version Lock

## Current locked baseline

The information below was read on 2026-09-18 directly from Steam's
`<steam-library>/steamapps/appmanifest_641320.acf`:

| Item | Locked value |
| --- | --- |
| Game | Cooking Simulator |
| App ID | `641320` |
| Steam Build ID | `24711773` |
| Main depot | `641321` |
| Main depot manifest | `8794508976528179697` |
| Main depot size | `5,400,417,433` bytes |
| Unity | `2022.3.62f3` |
| Install directory | `<steam-library>/steamapps/common/CookingSimulator` |

Six DLC depots are also installed on the reference machine. They and their
manifests are recorded under `observed_optional_depots` in
`configs/cookbench_game_lock.json`, but are not deployed by default. Only when a
benchmark task explicitly depends on a DLC should the corresponding entry be
moved into `required_depots`; `--include-observed-dlc` exists solely to
replicate the reference machine's full DLC set.

The current install directory contains MelonLoader 0.5.7 and
`Mods/CS_CamDump.dll`, so it is not a pristine Steam directory. The environment
is therefore split into three layers:

1. Stock Steam layer: downloaded from a pinned depot manifest and checked with
   the hashes of key stock files.
2. CookBench mod layer: MelonLoader, `version.dll`, and `Mods` are copied or
   installed separately; the lock file records key mod hashes.
3. Runtime data layer: `UserData`, logs, screenshots, and experiment output;
   never part of the stock snapshot.

## Commands

### Recommended: use a logged-in Steam client

1. Press Win+R, enter `steam://open/console`, then in the Steam Console run:

```text
download_depot 641320 641321 8794508976528179697
```

2. In a separate PowerShell window, from the repository root, run:

```powershell
python scripts/manage_game_environment.py watch-download
```

Every 5 seconds it prints the file count, logical byte count, and how many
files changed in size or mtime since the last poll. `--interval 10` and
`--once` are supported. By default it watches
`<steam-library>/steamapps/content/app_641320/depot_641321`; use `--source` to
point at another path. Ctrl+C only stops the watcher; it does not cancel the
Steam download. Files may be preallocated, so logical byte count is not actual
network progress, and no change does not mean the download is stuck. The
Console's 4037 MB and the lock's ~5.4 GB file size may use different
accounting, so a percentage cannot be computed from them. Wait until Steam
prints `Depot download complete`.

3. Once complete, import (add `--source` if Steam wrote to a different
   directory):

```powershell
python scripts/manage_game_environment.py import-depot --download-complete --target D:/CookBenchRuntime/CookingSimulator-24711773
python scripts/manage_game_environment.py verify --target D:/CookBenchRuntime/CookingSimulator-24711773
```

Import supports only the single main depot in the lock; it checks total size
and key file hashes, copies file by file with SHA-256 verification, confirms
source file sizes and mtimes did not change, generates a full manifest, and
then publishes the directory. If the target or staging directory already
exists, it refuses to overwrite. On failure the staging directory is kept for
inspection. If the total size does not match, do not skip the check — first
verify the download actually finished and that no stale files are mixed in.
The manifest is integrity evidence for this copy, not an independent proof of
every file in the Steam manifest. When a deployment manifest exists, `verify`
checks every file listed in it; extra mod and runtime files are allowed and
are not audited.

4. Deploy the complete locked MelonLoader and semantic mod (copying just two
   DLLs is not enough), then run `verify --with-mod` below. Currently this
   option only checks two mod reference hashes; it does not prove the whole
   loader is intact. Point EPM's `paths.game_userdata_root` at the `UserData`
   directory under the new location, and only use the environment for
   experiments after an actual launch, mod load, and telemetry update have
   been accepted. Do not assume `steam://rungameid/641320` starts the new
   copy; the new copy's DRM/Steam launch behaviour has not been tested yet.

### Fallback: SteamCMD (login and full download not yet verified end to end)

Read the current Steam build without modifying anything:

```powershell
python scripts/manage_game_environment.py inspect --steam-root <steam-library>
```

Download and deploy the locked main depot:

```powershell
python scripts/manage_game_environment.py deploy `
  --steamcmd-dir D:/CookBenchRuntime/steamcmd `
  --target D:/CookBenchRuntime/CookingSimulator-24711773
```

The script interactively reads the Steam username and a hidden password, and
reads the Steam Guard code when SteamCMD asks for it. Credentials are never
written to the lock file or the command line. The target directory must not
exist, to avoid clobbering a working environment.

After deployment, verify the key stock files:

```powershell
python scripts/manage_game_environment.py verify `
  --target D:/CookBenchRuntime/CookingSimulator-24711773
```

After installing the CookBench mod, also verify the key mod files:

```powershell
python scripts/manage_game_environment.py verify `
  --target D:/CookBenchRuntime/CookingSimulator-24711773 `
  --with-mod
```

## Full procedure

1. Run `inspect` and confirm the machine's Build ID is still `24711773`. If
   not, do not auto-update the lock file; run mod compatibility tests first.
2. `deploy` installs SteamCMD automatically, checks for roughly 20% free space
   for both the download source and the final copy, logs in, and downloads the
   pinned manifest.
3. Each depot first lands in SteamCMD's own content directory and is then
   merged into a temporary staging directory.
4. The SHA-256 of `CookingSim.exe` and `Assembly-CSharp.dll` is verified; only
   on success is the staging directory atomically renamed to the target.
5. The script writes `cookbench-deployment.json` in the target directory,
   containing a full per-file SHA-256 manifest.
6. Overlay the version-controlled CookBench mod onto the target directory,
   then run `verify --with-mod`.
7. Point EPM's `paths.game_userdata_root` at the `UserData` directory of that
   target.
8. Run `verify --with-mod` before every launch. Do not use an unverified
   current Steam install directory for formal experiments.

## Boundaries

- The Steam account must own the game and any DLC being downloaded; the script
  does not circumvent ownership or DRM.
- Steam servers may stop serving a historical manifest in the future. After the
  first successful deployment, keep an offline backup with its full hash
  manifest, as permitted by the license.
- `steam://rungameid/641320` launches whatever directory Steam currently has
  registered, which proves nothing about the locked environment. The formal
  runner still needs controlled launching against the target directory with a
  version check before start.
- A file version lock only addresses game/mod binary drift. GPU drivers, OS,
  resolution, language, save files, random seeds, and input configuration must
  still be recorded in each experiment's metadata.
