# WSL 环境技能指南

## 基本信息

| 项目 | 值 |
|------|------|
| **发行版** | Ubuntu-22.04-F (Ubuntu 22.04.5 LTS, Jammy Jellyfish) |
| **WSL 版本** | WSL 2 |
| **WSL 引擎版本** | 2.6.3.0 |
| **内核版本** | 6.6.87.2-microsoft-standard-WSL2 |
| **主机名** | DESKTOP-J7F034P |
| **用户名** | a1439775520 |
| **用户密码** | a1439775520 |
| **Home 目录** | /home/a1439775520 |
| **默认 Shell** | /bin/bash |
| **IP 地址** | 172.31.93.54 (动态分配，每次启动可能变化) |
| **Windows 版本** | 10.0.26200.8514 |

## GPU 与 CUDA

| 项目 | 值 |
|------|------|
| **GPU** | NVIDIA GeForce RTX 4060 |
| **显存** | 8188 MiB (8GB) |
| **驱动版本** | 595.97 |
| **CUDA Toolkit** | 未安装（计划通过 `wsl-setup/04-cuda-toolkit.sh` 安装） |

## 软件环境

| 项目 | 状态 |
|------|------|
| **Python** | 3.10.12 (系统自带) |
| **Miniconda** | 未安装（计划通过 `wsl-setup/05-miniconda.sh` 安装） |
| **ROS2 Humble** | 未安装（计划通过 `wsl-setup/03-ros2-humble.sh` 安装） |
| **代理** | ✅ 已配置，工作正常（端口 7897） |

## 磁盘挂载

Windows 磁盘在 WSL 中的挂载路径:

| Windows 盘符 | WSL 挂载点 |
|-------------|-----------|
| C:\ | /mnt/c |
| D:\ | /mnt/d |
| F:\ | /mnt/f |
| G:\ | /mnt/g |

**本项目在 WSL 中的路径**: `/mnt/f/Code/GitHub/CasuaLNN`

## 网络与代理配置

### .wslconfig（Windows 侧，`C:\Users\14397\.wslconfig`）

```ini
[wsl2]
networkingMode=mirrored
autoProxy=true
dnsTunneling=true
firewall=true
```

- **mirrored 模式**: WSL 与 Windows 共享网络栈，WSL 可直接通过 `127.0.0.1` 访问 Windows 上的服务。
- **autoProxy=true**: Windows 侧的 Clash Verge 代理设置会自动注入到 WSL 环境变量中，无需手动配置。

### 当前生效的代理（由 autoProxy 自动注入）

```bash
# 代理端口: 7897 (Clash Verge)
http_proxy=http://127.0.0.1:7897
https_proxy=http://127.0.0.1:7897
HTTP_PROXY=http://127.0.0.1:7897
HTTPS_PROXY=http://127.0.0.1:7897

# no_proxy 包含完整的内网网段
no_proxy=172.31.*,172.30.*,...,172.16.*,10.*,192.168.*,127.*,localhost,<local>
```

### 连通性验证

| 目标 | 状态 |
|------|------|
| GitHub (https://github.com) | ✅ HTTP 200 |
| Google (https://google.com) | ✅ HTTP 301 |

> **注意**: `~/.bashrc` 中有旧的 `PROXY_PORT=7890` 配置（由 `01-proxy-and-locale.sh` 写入），但已被 `autoProxy` 自动注入的 7897 端口覆盖。建议后续清理 bashrc 中的旧配置或将其更新为 7897。

### WSL 内的 wsl.conf（`/etc/wsl.conf`）

```ini
[user]
default=a1439775520
```

## 常用 WSL 命令

### 从 Windows 操作 WSL

```powershell
# 启动 WSL
wsl

# 在 WSL 中执行命令
wsl -- bash -c "命令内容"

# 以 root 身份执行
wsl -u root -- bash -c "命令内容"

# 查看 WSL 状态
wsl --list --verbose

# 关闭 WSL
wsl --shutdown

# 重启 WSL 发行版
wsl --terminate Ubuntu-22.04-F
```

### WSL 内常用操作

```bash
# 使用 sudo（密码: a1439775520）
sudo apt update && sudo apt upgrade -y

# 访问 Windows 文件
cd /mnt/f/Code/GitHub/CasuaLNN

# 从 WSL 打开 Windows 资源管理器
explorer.exe .

# 从 WSL 调用 Windows 程序
/mnt/c/Windows/System32/cmd.exe /c "echo hello"
```

## 注意事项

1. **Mirrored 网络模式**: WSL 使用 mirrored 网络，与 Windows 共享 IP，可直接通过 `127.0.0.1` 互通。
2. **autoProxy**: 代理由 Windows 的 Clash Verge（端口 7897）自动注入，无需在 WSL 内手动配置。
3. **文件权限**: 在 `/mnt/` 下操作 Windows 文件时注意权限问题，建议在 WSL 原生文件系统中进行开发。
4. **GPU 直通**: RTX 4060 通过 WSL2 GPU 直通可用，但需要安装 CUDA Toolkit。
5. **磁盘空间**: WSL 根分区有 ~954GB 可用空间。
6. **DNS 隧道**: `dnsTunneling=true` 确保 WSL 的 DNS 请求通过 Windows DNS 隧道解析，提高稳定性。
