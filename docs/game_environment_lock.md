# CookBench 游戏环境版本锁定

## 当前锁定基线

以下信息于 2026-09-18 从 Steam 的
`<steam-library>/steamapps/appmanifest_641320.acf` 直接读取：

| 项目 | 锁定值 |
| --- | --- |
| 游戏 | Cooking Simulator |
| App ID | `641320` |
| Steam Build ID | `24711773` |
| 主 Depot | `641321` |
| 主 Depot Manifest | `8794508976528179697` |
| 主 Depot 大小 | `5,400,417,433` bytes |
| Unity | `2022.3.62f3` |
| 安装目录 | `<steam-library>/steamapps/common/CookingSimulator` |

本机还安装了 6 个 DLC Depot。它们及各自 Manifest 已记录在
`configs/cookbench_game_lock.json` 的 `observed_optional_depots` 中，但默认不部署。
只有基准任务明确依赖 DLC 时，才应把对应项移入 `required_depots`；
`--include-observed-dlc` 仅用于复刻当前机器的完整 DLC 集合。

当前安装目录包含 MelonLoader 0.5.7 和 `Mods/CS_CamDump.dll`，不是纯净的 Steam
原版目录。因此环境被划分为三层：

1. Steam 原版层：由固定 Depot Manifest 下载，并用原版关键文件哈希校验。
2. CookBench Mod 层：单独复制或安装 MelonLoader、`version.dll` 和 `Mods`；锁文件记录关键 Mod 哈希。
3. 运行数据层：`UserData`、日志、截图和实验输出，不进入原版快照。

## 命令

### 推荐：使用已登录的 Steam 客户端

1. Win+R 输入 `steam://open/console`，在 Steam Console 执行：

```text
download_depot 641320 641321 8794508976528179697
```

2. 另开 PowerShell，在仓库根目录运行：

```powershell
python scripts/manage_game_environment.py watch-download
```

每 5 秒显示文件数量、逻辑字节数、相较上次发生大小或修改时间变化的文件数。
支持 `--interval 10` 和 `--once`。默认监测
`<steam-library>/steamapps/content/app_641320/depot_641321`，可用 `--source` 指定其他路径。
Ctrl+C 只停止观察，不取消 Steam 下载。文件可能预分配，逻辑字节数不是实际网络进度；
没有变化也不代表卡死。Console 的 4037 MB 与锁中约 5.4 GB 文件大小可能使用不同口径，
不能据此计算百分比。必须等待 Steam 输出 `Depot download complete`。

3. 确认完成后导入（若 Steam 输出目录不同，增加 `--source`）：

```powershell
python scripts/manage_game_environment.py import-depot --download-complete --target D:/CookBenchRuntime/CookingSimulator-24711773
python scripts/manage_game_environment.py verify --target D:/CookBenchRuntime/CookingSimulator-24711773
```

导入只支持锁中的单个主 Depot；检查总大小、关键文件哈希，逐文件复制并校验 SHA-256，
确认源文件大小和修改时间未变化，生成完整清单后发布目录。目标或 staging 已存在会拒绝覆盖。
失败会保留 staging 供检查。总大小不符时不要跳过检查，先检查是否未下载完成或混有旧文件。
清单是这次复制的完整性证据，不是独立验证 Steam Manifest 中每个文件的证明。
`verify` 在存在部署清单时检查清单内全部文件；允许额外 Mod 和运行时文件，不检测额外文件。

4. 部署完整的已锁定 MelonLoader 和语义 Mod（不能只复制两个 DLL），再执行下方
`verify --with-mod`。当前该选项只检查两个 Mod 参考哈希，不证明整个加载器完整。
将 EPM 的 `paths.game_userdata_root` 指向新目录下的 `UserData`，完成实际启动、
Mod 加载、遥测更新验收后才能用于实验。不要用 `steam://rungameid/641320` 假定启动的是新副本；
新副本的 DRM/Steam 启动行为尚待实测。

### 备用：SteamCMD（登录和完整下载尚未端到端验证）

读取当前 Steam build，不做修改：

```powershell
python scripts/manage_game_environment.py inspect --steam-root <steam-library>
```

下载并部署锁定的主 Depot：

```powershell
python scripts/manage_game_environment.py deploy `
  --steamcmd-dir D:/CookBenchRuntime/steamcmd `
  --target D:/CookBenchRuntime/CookingSimulator-24711773
```

脚本会交互读取 Steam 用户名、隐藏密码，并在 SteamCMD 请求时读取 Steam Guard
验证码。凭据不会写入锁文件或命令行。目标目录必须不存在，避免覆盖一个可工作的环境。

部署后验证原版关键文件：

```powershell
python scripts/manage_game_environment.py verify `
  --target D:/CookBenchRuntime/CookingSimulator-24711773
```

安装 CookBench Mod 后同时验证 Mod 关键文件：

```powershell
python scripts/manage_game_environment.py verify `
  --target D:/CookBenchRuntime/CookingSimulator-24711773 `
  --with-mod
```

## 完整流程

1. 运行 `inspect`，确认本机 Build ID 仍是 `24711773`。若不同，不自动更新锁文件，先进行 Mod 兼容性测试。
2. `deploy` 自动安装 SteamCMD、为下载源和最终副本各检查约 20% 空间余量、登录并下载固定 Manifest。
3. 每个 Depot 先进入 SteamCMD 自己的 content 目录，再合并至临时 staging 目录。
4. 验证 `CookingSim.exe` 与 `Assembly-CSharp.dll` 的 SHA-256；验证通过后才原子改名为目标目录。
5. 脚本在目标目录生成 `cookbench-deployment.json`，保存完整逐文件 SHA-256 清单。
6. 将经版本管理的 CookBench Mod 覆盖到目标目录，再运行 `verify --with-mod`。
7. EPM 的 `paths.game_userdata_root` 指向该目标目录的 `UserData`。
8. 启动时先运行 `verify --with-mod`。不要使用未经校验的 Steam 当前安装目录执行正式实验。

## 边界

- Steam 账号必须拥有游戏和所下载的 DLC；脚本不会规避所有权或 DRM。
- Steam 服务器未来可能停止提供某个历史 Manifest。首次成功部署后，应按许可证允许的方式保留离线备份及其完整哈希清单。
- `steam://rungameid/641320` 启动的是 Steam 当前注册目录，不能证明它是锁定环境。正式运行器还需改为针对目标目录的受控启动，并在启动前校验版本。
- 文件版本锁只解决游戏/Mod 二进制漂移。显卡驱动、操作系统、分辨率、语言、存档、随机种子和输入配置仍需写入每次实验元数据。
