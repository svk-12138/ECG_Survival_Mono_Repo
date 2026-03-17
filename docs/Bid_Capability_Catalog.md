
---
## 架构总览（实现导向复杂描述）
> 目标：以可落地组件视角呈现整体复杂度；所有条目均与当前保留能力匹配（无虚构模块）。

### A. 逻辑分层与主责
| 层级 | 作用 | 关键子组件 | 主要技术/实现 |
|------|------|------------|--------------|
| Ingest & 标准化 | 各模态原始数据进入与规范化 | 影像 Loader, 临床字段校验, 组学正态化 | dcm2niix, nibabel, OpenSlide, Python 校验脚本 |
| Token 化表示 | 统一多模态结构 | Token Builder, 缺失掩码生成器 | PyTorch 张量管线 |
| 表征骨干 | 特征抽取 | ResNet50, Swin-UNETR, ViT-B/16 | PyTorch + 预训练权重 |
| 融合与预后 | 多事件风险计算 / 治疗效应 | CrossFormer 扩展, 多事件头 | 自研模块 (PyTorch) |
| 知识增强 | 报告结构化 & 证据检索 | 抽取器, RAG 检索器 | 指令微调 LLM, BM25+向量检索 |
| 性能与资源 | 运行效率控制 | 动态批次, Mixed Precision, Token 剪裁 | AMP, 自定义显存探测 |
| 交付与运行 | 可复现部署 | Docker 多阶段, 镜像扫描 | Docker, Trivy, SBOM |

### B. 运行时与部署视角
| 运行域 | 形态 | 关键容器/进程 | 扩缩策略 | 监控指标 |
|--------|------|--------------|----------|----------|
| 预处理批处理 | Cron / 手动 | image-normalizer, clinical-validator | 任务队列长度 | 吞吐, 错误率 |
| 训练集群 | GPU Pod | trainer (FSDP/AMP), hpo-worker | 剪枝后自动收缩 | GPU 利用率, trial 成功率 |
| 推理服务 | 在线 API | inference-gateway, explanation, retriever | QPS 自适应副本 | P95 延迟, 错误率 |
| 构建发布 | CI Pipeline | docker-builder, trivy-scan | 按提交触发 | 失败率, 构建时长 |
| 治理后台 | 轻量服务 | lineage-writer, metrics-exporter | 固定 | 写入滞后, 丢包率 |

### C. 治理与可靠性控制点
| 控制点 | 监控信号 | 触发动作 | 记录位置 |
|--------|----------|----------|----------|
| 数据异常 | 缺失率/体素差异阈值 | 阻断入库并生成异常报告 | validation_logs/ |
| 训练劣化 | 验证指标下降 ≥N 轮 | 早停 + 标记 trial | mlflow runs |
| 显存压力 | 使用率>阈值 + OOM 捕获 | 动态减 batch | resource_logs/ |
| 推理漂移 | 特征分布 PSI>阈值 | 触发重训练评审 | drift_monitor/ |
| 安全漏洞 | Trivy 高危发现 | 构建失败并告警 | ci_security/ |
| 校准偏移 | Brier>阈值 | 重新校准并版本化 | calibration/ |

---
## 1. 数据与标准化能力
| 能力模块 | 覆盖 | 描述 | 交付产出 | 验证方式 |
|----------|------|------|----------|----------|
| 多格式影像规范化 | DICOM / NIfTI / WSI / 超声序列 / 旧格式桥接 | 统一坐标/体素/方向并生成索引，剔除异常切片 | 统一体素/方向 NIfTI；WSI 多倍率 Patch 索引；关键帧提取结果 | 样例前后体素参数对照表 |
| 临床结构化整合 | 枚举字段+缺失掩码 | 字段标准命名 + 缺失标注 + 数值归一规则 | 标准字段字典 + 缺失统计 JSON | 字段覆盖率报告 |
| 稀疏组学对接 | Index+Value 嵌入投影 | 稀疏基因列表映射至稠密向量并校正批次效应 | Gene Index Map + 正态化矩阵快照 | 随机抽样重现脚本 |
| Token 统一表示 | 模态 type id + mask + value proj | 多模态特征打平为统一 (B,T,D) 结构以便后续融合 | Token Schema 文档 + 样例张量 (.pt) | Schema 校验日志 |

---
## 2. 表征与核心模型基座
| 类别 | 提供骨干 | 描述 | 交付物 | 校验 |
|------|----------|------|--------|------|
| 2D/3D 影像 | ResNet50 / Swin-UNETR / ViT-B/16 | 分层特征抽取（局部+全局注意力） | 预训练 / 微调权重 (含 meta JSON) | 验证集指标表 |
| 分割 | nnU-Net / U-Net / TransUNet | 病灶/器官像素级掩码生成 | Dice 指标 + 推理脚本 | 随机病例可复现 |
| 检测 | YOLOv8 / Mask R-CNN | 病灶候选框/实例掩码定位 | 推理 Notebook + 置信度阈值建议 | 置信度-F1 曲线 |
| 预后/生存 | CrossFormer 多事件扩展 | 多事件风险曲线与治疗差异估计 | 风险曲线 JSON + C-index / IBS | 基线对比报告 |
| LLM 增强 | 指令微调医学 LLM + RAG 检索器 | 报告结构抽取 + 证据支撑解释 | 报告→结构化示例 + Evidence 列表 | 字段抽取精度表 |

---
## 3. 自动化与训练编排
| 能力 | 实现策略（抽象） | 描述 | 交付物 | 验证 |
|------|------------------|------|--------|------|
| 超参搜索 | Optuna TPE + Early Pruning | 自动探索关键超参以提升指标并减少无效轮次 | 最优 trial config + 搜索轨迹 | 学习曲线对比 |
| 配置管理 | Hydra 层级 + 冻结快照 | 多环境覆盖与参数覆写统一化可重现 | 最终 config YAML + Hash | Hash 可复现校验脚本 |
| 实验追踪 | MLflow | 训练指标/模型/工件集中记录与检索 | metrics.json / artifact 归档 | 指标面板截图 |
| 版本/谱系 | Git + DVC + 元数据绑定 | 数据→代码→权重全链条追踪 | lineage 报告 PDF | 链路完整性检查 |
| 资源感知 | 动态 batch / mixed precision | 根据显存波动自适应保持吞吐 | 显存利用率曲线 | GPU 利用率日志 |
| 容器镜像发布 | Docker 多阶段构建 + 安全扫描 (Trivy) | 生成可复现最小镜像并含 SBOM 及入仓自动签名 | 镜像 tar / digest 列表 + SBOM | 扫描报告 + 重现拉起脚本 |

---
## 4. 性能与资源优化
| 能力 | 交付产出 | 效果指标 | 说明 |
|------|----------|----------|------|
| 动态批次回退 | 批次自调策略说明 | OOM 发生率下降数据 | 提高稳定性 |
| Checkpoint 精简 | Top-K + BF16 压缩 | 存储占用下降比 | 快速回滚 |
| 推理加速 | TensorRT / ONNX 双路径 | QPS / Latency 对比 | SLA 自适应 |
| Token 剪裁 | 注意力 Top-K + 蒸馏 | 减少 Token 比例 vs 性能 | 降推理成本 |

