# 现场测量表 + 真机上手清单（2026-11-07）

量完的数填进 `mission/course_prior.venue.json`（格式照 `mission/course_prior.example.json`），代码不用改。
精度：卷尺量，±0.2 m 就很好，最多 ±0.5 m。全部按"起点坐标系"记：

- 飞机放在起飞垫上，**机头朝向 = 前（forward）**，机头左边 = 左（left），向上 = 上（up），单位米。
- 角度从"前"开始逆时针算，90° = 正左，-90° = 正右，180° = 正后。

## A. 起点和围网（最重要）
| 项 | 怎么量 | 填到 |
|---|---|---|
| 起飞垫中心到前、后、左、右四面网的距离 | 卷尺量水平距离 | `fence.forward_max / forward_min / left_max / left_min`（后、右写负数） |
| 网高（约 3.05 m） | 量或估 | `fence.up_max`（取网高减 0.2） |
| 飞机放在垫上时机头朝哪 | 拍照记下 | 决定"前"的方向，起飞时必须一致 |

## B. 各关位置（每关量一个代表点，都是相对起飞垫中心）
| 关 | 量哪个点 | 填到 |
|---|---|---|
| 走廊 | 走廊中线上 2–3 个点（入口、中间、出口），记飞行高度 | `corridor[]`：forward / left / up |
| 隧道 | 第一个铜环中心（含离地高度）；穿越方向；第一环到最后一环的长度 | `tunnel.entry`、`axis_deg`、`length_m` |
| 隧道（帮检测） | 铜环半径、环间距、环最低点离地高度 | 告诉我，填进 `TunnelParams.radius_m / hoop_spacing_m / bottom_gap_m` |
| 窗口 | 窗洞中心（含离地高度）；穿越方向 | `window.center`、`through_deg` |
| 窗口（帮检测） | 窗洞宽 × 高，板宽 × 高，洞下沿离地高度 | 告诉我，填进 `known_size_m` |
| 白石 | 顶面中心；顶面离地高度；长 × 宽 | `landing_target.center`、`top_up` |
| 墙 | 墙两侧各取一个地面点（白石那侧一个、起点那侧一个），连线要穿过墙；墙顶离地高度 | `wall.cross_from`、`cross_to`、`top_up` |

再拍一张俯视草图，标出每关的大致位置和朝向，并标出真石头的位置（只要大概）。

## C. 真机检查（第一次通电，先不飞）
1. `ros2 topic list` / `hz`，确认有这些话题：
   - `/stereo/left/image`、`/stereo/right/image`、`/camera/depth/image_raw`：频率约 15 Hz
   - `/fmu/out/*_v1`
   - `/fmu/out/distance_sensor`
   - `/estimator/healthy`
2. 把这些存成文件发我：
   - `/stereo/left/camera_info`
   - `/camera/depth/camera_info`
   - TF（`ros2 run tf2_ros tf2_echo base_link oak_left`）
   - PX4 参数：`param show LNDMC*`、`MPC*`、`EKF2*`
3. 手持或推着飞机慢慢走一遍全程，录一个 bag（`flight_run.py` 带图像录制，或者直接 `ros2 bag record`）。用它离线验证 VIO 和三个检测器，**这一步比飞更重要**。
4. 第一次飞：只做悬停加降落（hover prior），确认高度、漂移、落地判定。
5. 再分段飞：先只飞走廊；再飞到隧道前悬停、只检测不穿；确认稳定后再穿。

## D. 今晚的数字哪些靠不住（明天必须用实测替换）
- 复刻场地所有尺寸都是从 7 张照片目测的；各关在场内的相对位置是按关卡顺序自己排的，没有依据。
- 白石的尺寸和位置纯属估计。
- 墙顶 1.70 m、窗洞 0.75 × 0.6 m、隧道环离地 0.08 m 都是照片比例推的。
- 机体和相机参数取自官方仿真里的 Sky Beta + OAK-D S2，要对照真机核对。
