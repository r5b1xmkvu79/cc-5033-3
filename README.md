# 电视档案去交错工具（Archive Deinterlacer）

面向自制隔行电视档案的**场序感知**去交错工作台。它把隔行帧拆成两个采样时刻
不同的场，按显式声明的场序重建逐行视频，并输出与所选模式严格一致的帧率、帧数
和时间戳。Web 界面只负责编排内核，所有算法判定都来自 `archive_deinterlacer.core`。

## 1. 关键概念（上场先后 ≠ 像素行位置）

这两件事在代码里是独立的，文档与界面均分开解释：

| 概念 | 含义 | 代码位置 |
| --- | --- | --- |
| 场的**像素行位置** | 上场/顶场（top field）占偶数行 `0,2,4…`；下场/底场（bottom field）占奇数行 `1,3,5…` | `core/fields.py: TOP/BOTTOM` |
| 场的**采样先后** | TFF（top-field-first）先采样偶数行；BFF（bottom-field-first）先采样奇数行 | `core/fields.py: TFF/BFF` |

N 个隔行帧、帧率 F 对应 2N 个场，场速率 2F。时间戳只由采集顺序推导
（`field_timestamps`），绝不根据行位置猜测。

## 2. 三种可解释重建策略（静止区/运动区区别对待）

- **bob**：每个场输出一帧，输出帧率 `2F`、帧数 `2N`。纯空间插值，不合并其它
  时刻的像素，因此运动平滑且不可能产生梳齿，垂直细节略软。
- **weave**：把同一时刻配对的两场重新交错，输出帧率 `F`、帧数 `N`。静止内容
  完全无损；运动区域会保留真实梳齿（这是物理事实，不做模糊掩盖）。
- **adaptive（默认）**：输出帧率 `2F`、帧数 `2N`。用**同位置相邻场**之差计算
  运动掩膜（`motion_mask`，含邻域判决抑制噪点）：
  - 静止像素：与时间最近的异位置场 weave，保留全部垂直细节；
  - 运动像素：只在当前场内空间插值，杜绝异时刻像素形成梳齿。
  - 掩膜随每帧暴露给界面，可叠加显示“运动边缘”，解释每个像素的处理依据。

明确**不采用**“两场直接合并再垂直模糊”的做法；该策略仅作为
`blur_merge_baseline` 保留在基准里，用于量化展示它如何用细节损失掩盖梳齿。

## 3. 元信息不可靠时：诊断，不猜测

`core/diagnostics.py` 输出结构化诊断，而不是悄悄替用户决定：

- 梳齿能量（经典 comb 检测器）；
- **同位置场时间残差**（主判据）：逐行素材两场描绘同一瞬间，残差低；真实隔行
  运动残差高。这能抵御 H.264 4:2:0 色度子采样在锐边上制造的“伪梳齿”；
- TFF/BFF 两种读法的时序平滑度，给出带置信度的场序建议；
- 逐行素材识别、奇数高度检查、元数据与内容冲突告警。

当元数据缺失且内容分析无法高置信定序，或输入本身就是逐行素材时，内核**阻断**
处理并回传原因；场序必须由操作员显式选择，CLI 和 Web 都会拒绝无场序请求。

## 4. 时间一致性

每种模式都返回自己的输出时间戳，编码时用匹配的帧率调用 ffmpeg。清单
（manifest）内含 `duration_check`，断言 `帧数/帧率 == 总时长`，杜绝“改了帧数却
保留错误时长”。导出的 MP4 以 `-field_order progressive` 标记，播放器不会再触发
自带去交错——交付的是算法真实导出结果，而非播放器显示效果。

## 5. 安装

需要 Python 3.11+。ffmpeg 无需系统安装：`imageio-ffmpeg` 自带静态二进制；
若 PATH 上有 ffmpeg 则优先使用，也可用环境变量 `FFMPEG=/path/to/ffmpeg` 指定。

```bash
python3 -m pip install -r requirements.txt
```

## 6. 如何启动界面（档案去交错的图形入口）

档案去交错以 **Web 影像校正界面为主入口**，命令行只用于批处理与基准。在仓库
根目录（`/workspace`）执行：

```bash
python3 -m archive_deinterlacer serve --host 127.0.0.1 --port 8000 --workdir output/web
```

然后浏览器打开 <http://127.0.0.1:8000>。操作流程：

1. 上传隔行视频，自动解码并给出诊断（容器声明 vs 内容分析）；
2. 在界面上**实际选择场序**（TFF/BFF）与重建策略，可调运动阈值；内核不接受
   无场序请求，缺场序会在状态区直接提示拒绝原因；
3. 运行后逐帧对照三栏：**原始隔行帧 / 单场视图 / 最终重建帧**，重建帧可勾选
   “叠加运动边缘”；切换 TFF/BFF 时单场视图复用已分离的场行，不重新复制像素；
4. 时间信息栏显示每帧时间戳、来源场及其位置与采样时刻、帧率/帧数/时长一致性；
5. 下载真实导出的逐行 MP4 与时间戳清单 JSON。

运动边缘与重建帧均来自内核的实际运算结果，界面只做编排与渲染；内核拒绝
（无场序、逐行输入、奇数高度等）会作为结构化异常直接显示在页面状态区。

## 7. 命令行用法（仅辅助图形界面）

```bash
# 生成合成参考素材（横向移动条纹、细字、斜线、静止图案），含逐行真值与 TFF/BFF 隔行片
python3 -m archive_deinterlacer generate --out output/synthetic

# 只诊断、不改片；--strict 在无法定序时以退出码 2 失败
python3 -m archive_deinterlacer diagnose input.mp4 --strict

# 显式场序去交错并导出 MP4 + manifest
python3 -m archive_deinterlacer deinterlace input.mp4 --field-order tff --mode adaptive --out output/jobs

# 对三种模式 + 模糊基线做量化基准（运动平滑/梳齿残留/细节损失）
python3 -m archive_deinterlacer benchmark --out output/benchmark

# 场处理加速 A/B：分阶段成本分解、梳齿误差、每帧成本与逐像素一致性校验
python3 -m archive_deinterlacer perf --out output/perf
```

## 8. 基准指标如何对应需求

报告 `output/benchmark/benchmark_report.json`：

- `motion_smoothness_mad`：运动带相邻帧平均亮度差，越低运动越平滑；
- `comb_residue`：重建帧残留梳齿能量，0 表示无梳齿；
- `detail_loss_db`：静止图案高频能量相对真值的损失（dB），0 为无损，负值为损失。

实测（合成素材，TFF/BFF 结论一致）：weave 细节 0 dB 但运动区梳齿最高；adaptive
细节约 -0.07 dB 且梳齿远低于 weave；模糊基线以约 -3.2 dB 的细节损失换取低梳齿。

## 9. 场处理加速：计算顺序、成本分解与回退准则

加速只改**成本**，不改判定与像素。变更前的完整行为冻结在
`core/legacy.py`，性能集与测试对两条路径做逐帧逐像素、运动掩膜、时间戳比对。

### 9.1 消除的重复计算（按执行顺序）

1. **场分离一次、行信息复用**：TFF/BFF 评估原本各自把每场行复制一遍。现在
   `fields.split_fields_reusable` 每个隔行帧只分离一次，BFF 用同一份行数组
   重标采样序号；诊断与 Web 切换场序时不再二次复制。场序（采样先后）与像素
   行位置仍是两个独立概念，输出场序与时间戳语义不变。
2. **运动判断留在 uint8 域**：同位置场差分原先把两场各转一次 int16 再转
   float；现仅保留一次 int16 减法，绝对差（必在 0..255）收进 uint8，亮度均值
   用 uint16 累加。两个阈值（像素差分 + 3×3 邻域）与输入完全相同，运动掩膜
   逐位等价，细线与运动边缘仍按原准则校验。
3. **重建每个输出行只写一次**：旧实现先各建一整帧空间插值与时间 weave，再用
   `np.where` 合并（每行被写两遍）。现在场自身的行直接落位，缺失行仅按掩膜在
   “空间均值 / 异位置场 weave”间二选一，不再生成两个完整临时帧。不跳过任何
   场、不降低分辨率——任何跳帧或缩帧都会被一致性断言拦下。

### 9.2 成本分解与性能集（`core/perf.py`，CLI `perf`）

- 负载一 `thin_line_motion`：1 px 细竖线与高频棋盘格逐场移动，最大化细线运动
  与运动边缘判决压力；
- 负载二 `field_order_switch`：同一真值分别按 TFF/BFF 采集，加压场序变化与
  行复用路径。
- 每个负载在**相同输入**上跑冻结旧路径与加速路径，按 `split / motion /
  reconstruct` 分阶段计时，报告每帧秒数、总加速比与重建帧**梳齿误差**；
  计时采用 best-of-N 墙钟。
- 先断言后计时：帧数、分辨率、逐像素、掩膜、时间戳任一不一致即抛错失败，
  梳齿误差若高于旧路径同样失败。指标不允许用默认值遮错——零耗时会直接使
  断言失败，报告里的每个数字都对应一次真实执行。

实测（288×384、60 隔行帧 → 120 重建帧，`output/perf/field_processing_perf.json`）：
三条负载均 `parity=exact`、梳齿误差新旧完全相同，总耗时约 1.15–1.18×。

### 9.3 回退准则

若发现梳齿误差或每帧成本之外的任何行为退化（像素、掩膜、帧数、帧率、时长、
场序语义），撤回对应优化：`pipeline.run(..., accelerated=False)` 即切回
`core.legacy` 的冻结构建器，无需改调用方。

## 10. 目录结构

```
archive_deinterlacer/
├── core/
│   ├── ffmpeg_io.py    # ffmpeg 进程、元数据解析、流式编解码
│   ├── fields.py       # 场分离/采样时刻/bob/weave/运动掩膜/自适应重建 + 加速原语
│   ├── diagnostics.py  # 梳齿、场序推断、逐行识别、阻断式诊断（单行副本双场序复用）
│   ├── pipeline.py     # 模式编排、时间戳一致性、导出清单（accelerated 开关）
│   ├── legacy.py       # 冻结的加速前自适应构建器：性能基准与回退参照
│   ├── perf.py         # 场处理加速性能集：成本分解/梳齿误差/每帧成本/一致性闸门
│   ├── synthetic.py    # 四类图案的真值与隔行采集模拟
│   └── benchmark.py    # 三指标量化对比 + 模糊基线
├── web/                # Flask 编排层（无算法），含模板与原生 JS 界面
├── cli.py              # generate/diagnose/deinterlace/benchmark/perf/serve
└── tests/              # 内核语义、加速一致性、ffmpeg/HTTP 端到端测试
```

## 11. 测试

```bash
python3 -m unittest discover -s archive_deinterlacer/tests
```

覆盖场序与行位置语义、场时间戳、三种模式的帧数/帧率/时长不变量、奇数高度与无
场序拒绝、梳齿检测、合成片往返、逐行片拒绝、完整 Web API 工作流，以及加速
专项：可复用分离等价、uint8 运动掩膜逐位等价、新旧管线精确一致、性能集注入
退化（改像素/丢帧）时闸门必须失败。
