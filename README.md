# UAV Mission Planning

无 GPS 室内无人机任务规划、被动双目建图与 PX4 闭环控制的研究代码。采用 ROS 2 Jazzy、Gazebo Harmonic、OpenVINS 和定制 Cartographer，围绕七英寸近似机体完成走廊、迷宫、隧道、穿窗或 Gate B、平台降落与停桨。

当前开发入口是 **`cartographer/v4/`**。这个仓库保存现有源码和可复核的配置，不是已经完成真机认证的飞控发行版。

## 实现

- 被动双目左右一致性与光度检查，共用校正后相机标定，按曝光时刻插值姿态。
- VIO、单个下视 ToF 与 Cartographer 局部观测融合；闭环地图跳变不直接进入 PX4 控制坐标。
- 连续路径跟踪、净空平滑、曲率限速、制动距离与前视可见性约束。
- 基于实际双目观测的结构点筛选，稀疏扫描外推修复、IMU/VIO 联合图优化和错误回环检查。
- 有界感知工作线程、解锁前建图检查、完整飞行与 SLAM 同屏录像、独立事后裁判和几何评估。

导航可以使用公开 CAD 先验与实测红线/窗框配准。SLAM 重建使用实际传感器观测；仿真真值仅用于独立评估。默认验证入口是 `source stereo`，没有添加环境扫描激光雷达或主动 IR 纹理。

## 已有验证

| 轮次 | 首次运动至停桨 | 碰撞 | 路线与状态 |
|---|---:|---:|---|
| v3 / slamfix-05 | 132.072 s | 0 | 穿窗，全程完成 |
| v4 / fast-09 | 61.868 s | 0 | 穿窗，全程完成，较早的控制配置 |
| v4 / fast-11 | 71.596 s | 0 | 穿窗，全程完成，后续可见性约束及高度恢复配置 |

这些轮次配置不同，不能当作同配置连续成功。**尚未验证稳定的 60 秒内零碰撞通关。** 上述时间从首次运动计时；fast-09 从确认解锁到停桨为 64.940 s。完整录屏包含起飞前准备和停桨后片段，不加速。

地图仍为稀疏观测图，低纹理墙面和未知空白不能当作已重建或已确认自由空间。优化节点误差也不等于全程在线定位误差。评估口径及局限见 [验证说明](docs/VALIDATION.md) 和 [指标 JSON](docs/episode-metrics.json)。

## 目录

```text
cartographer/
  scripts/, config/       早期基线与赛道工具
  results/cad-grid.*      从原始 CAD 生成的规划资源，非历史飞行地图
  v2/                    七英寸机体、任务定位与早期完整流程
  v3/                    Cartographer 漂移/回环修复版本
  v4/                    当前规划、感知、融合和评估代码
patches/                  已验证上游修订的 C++ / OpenVINS 差异
scripts/, config/         原始 PX4/ROS/Gazebo 环境工具
course_source/            用户提供的初步 CAD 资源
docker/                   已有运行环境的 Dockerfile
 docs/                     复现、指标、来源与迁移边界
```

## 运行与测试

完整 SITL 需要 Linux、ROS 2 Jazzy、Gazebo Harmonic、匹配的 PX4/ROS 消息、上游构建产物及图形环境。源码仍使用容器内 `/work` 布局。参见 [运行说明](docs/RUNNING.md) 与 [上游版本/补丁](patches/README.md)。

不需要 ROS 或仿真的核心回归测试：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m unittest discover -s cartographer/v4/scripts -p 'test_*.py' -v
```

这些测试验证规划、限速、感知调度、深度检查、高度恢复和合成裁判行为，不替代实际飞行验证。

## 真机迁移

标定、曝光/IMU 时钟、机体包络、制动能力、振动、暗光、ToF 反射率和 Orin 负载都需要实测。`HARDWARE_MODE=1` 要求明确提供实际校正后相机标定。任务执行器保留 SITL 解锁保护，仿真中的旋翼/推力参数不能直接用于实机。

详见 [迁移说明](cartographer/v4/HARDWARE_PORTABILITY.txt)。

录像、MCAP、ULog、构建目录和主机临时文件没有提交到 Git。第三方组件与 CAD 的权利说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。
