

## 架构总览

A. 逻辑分层与主责
1. Ingest & 标准化：多模态原始数据流入口（影像 / 临床结构化 / 组学与仿真输出）执行一致性与完整性校验；包含：
	- 影像 Loader：基于 dcm2niix + nibabel 解析 DICOM→NIfTI，执行方向矩阵 (ImageOrientationPatient) 与像素间距 (PixelSpacing) 归一。
	输出：标准化张量 + 结构化 Parquet。
2. 仿真特征生成：参数化分子/材料/量子级批处理（LAMMPS / GROMACS / Quantum ESPRESSO）→ 解析中间日志 → 收敛与异常判定 → 描述符向量缓存。关键机制：
	- LAMMPS：能量与温度序列滑动窗口 ΔE < 1e-5 视为收敛；脚本式后处理提取 RDF(g(r))、MSD(t) 斜率（扩散系数近似）、势能方差。
	- GROMACS：gmx 构建氢键网络（阈：距离<0.35nm，角>150°），计算 RMSD/RMSF 及二级结构 DSSP 分布；时间对齐使用线性插值补齐缺失帧。
	- Quantum ESPRESSO：电子结构计算解析 band.x / dos.x 输出，提取带隙(Eg)、费米能级(Ef)、积分态密度（窗口内数值积分）；收敛判据基于 scf Δρ 与总能量差。
	输出：多域结构/能量/动力学特征矩阵。
3. Token 化表示：将多源模态归约为 (B,T,D,....)；
	- 缺失掩码：uint8 bitmask（节省显存）+ attention mask 动态裁剪；对超长序列应用分段窗口合并策略降低 T。 
	- 内存策略：channels_last + pinned memory 预取；批内排序按估计算子 FLOPs 自适应重排提升 GPU 利用率。
4. 表征骨干（多选）：ResNet50（卷积局部模式捕获）、Swin-UNETR（层次 Transformer + Encoder-Decoder）、ViT-B/16（全局注意力）、DeiT等可选择。
	- 双后端：PyTorch（AMP + gradient checkpointing）主路径；TensorFlow 路径经统一权重转换（shape / dtype map + 层名对齐）+ optional XLA 编译。
	- 训练策略：梯度缩放、Lookahead / AdamW 组合、余弦退火 + warmup 调度；混合精度下保持稳定的 loss scaling 自适应。
	- 注意力机制扩展：
		• 分层多头自注意力：Swin 窗口内 MSA + Shifted Window 融合形成局部→全局递进；ViT 全局 MSA 提供长程依赖；可选启用 FlashAttention (块状流式 softmax) 降低 O(L^2) 显存常数因子。
		• 交叉注意力桥接：影像 Token ↔ 仿真描述符 ↔ 临床结构化嵌入采用 Cross-Attention Block（Q=主模态, K/V=辅模态），先进行尺度对齐（Linear Projection + LayerNorm），再通过门控残差 (gate * CrossAttn + (1-gate) * Identity)。
		• 多查询/共享键值 (MQA/GQA)：对大头数配置将多头 Query 独立、K/V 分组共享（减少参数与缓存占用）。
		• 相对位置编码：3D 体素 Token 使用相对坐标偏置表（Δx,Δy,Δz → bias bucket）并缓存以加速；窗口注意力内部采用预计算 index map。
		• 头重要性裁剪：基于 attention entropy / gradient-based saliency 统计低贡献头，执行 Head Pruning（保存裁剪映射以便回滚）。
		• 多尺度融合注意力：将不同分辨率特征序列拼接前插入尺度标签 embedding；可启用分辨率加权 CrossAttn（权重来自参数化 softmax over scales）。
		• 低秩适配 (LoRA) 到注意力投影矩阵（Wq/Wk/Wv/Wo）用于快速领域微调，rank r 可调并支持合并权重或按需解耦。
		• 正则与稳定性：对注意力 logits 施加温度 τ 与 dropout、并监控 head-wise KL 距离防止塌缩；Cross-Attn 输出前加 LayerScale (γ 可学习) 改善收敛微调精度。
5. 性能与资源：运行时效率治理；
	- 动态批次控制：闭环反馈（监控当前显存占用与安全余量 headroom，PID-like 调节 BatchSize）；
	- 混合精度：自动黑名单层（如 LayerNorm 精度敏感）回退 FP32；
	- Token 剪裁：注意力熵 / 平均注意力权重阈值裁剪 + Top-K 保留，统计裁剪前后覆盖率。
6. 训练劣化监控：滑动窗口验证指标（如 C-index / Dice / F1）下降 N 次触发 EarlyStop；结合 Optuna pruning（Median / SuccessiveHalving）快速淘汰；mlflow runs 记录 trial_rank、pruned_flag。
7. 显存压力调节：捕获 OOM（RuntimeError hook）或阈值（占用/总显存 > p%）触发 batch 缩放（乘以 α<1）与精度降级（FP32→AMP 或 AMP→BF16）；记录调整轨迹（before/after、step、loss 漂移）于 resource_logs/ 供回放。

---
## 1. 数据与标准化能力
- 多格式影像规范化：DICOM（重采样 / 方向矩阵统一 / 体素正交化 / 异常切片剔除（空白 & 低信息熵））；超声序列关键帧抽取（基于帧间 SSIM 差分 + 峰值检测）；输出含：标准 NIfTI、Patch 索引（JSONL）、关键帧列表。
## 2. 表征与核心模型基座
- 2D/3D 影像表征：ResNet50（分层卷积局部纹理）+ Swin-UNETR（Swin 层次窗口注意力 + U-Net 解码器）+ ViT-B/16（全局自注意）互补；支持 FP16 + 梯度检查点以控显存；输出 meta 包含层级特征维度 / patch stride / 均值方差统计；验证：标准验证集多指标表（AUC / Dice / F1）。
- 分割能力：nnU-Net（自适应配置）+ U-Net + TransUNet（CNN+Transformer）生成像素/体素掩码；支持测试时增强（TTA: 翻转/尺度）与滑动窗口重叠融合（加权平均）；
- 检测 / 分类 / 回归评估能力：
	• 目标检测：YOLOv8 + Mask R-CNN；训练含多尺度与基本增强；指标：mAP@0.5、mAP@0.5:0.95、Precision、Recall、F1、PR 曲线。
	• 目标分割：nnU-Net / U-Net / TransUNet；指标：Dice、IoU(Jaccard)、Hausdorff95。
	• 分类任务：Top-1 Accuracy、Macro F1、AUROC、AUPRC、Confusion Matrix。
	• 连续回归：MAE、RMSE、R²。
	• 阈值分析：提供阈值 vs Precision/Recall/F1 扫描与最佳 F1 / Youden 点。
	• 结果可追溯：评估元数据写入 metrics_manifest.json；保留 PR / ROC / Dice 关键曲线 SVG。
- LLM 增强：指令微调医学 LLM（LoRA/QLoRA）；Prompt 动态 few-shot（相似案例嵌入 Top-K）；输出：结构化字段 JSON、证据 spans、置信度；评估：F1 / Exact Match / 证据命中率。

## 3. 自动化与训练编排
（框架支持：默认 PyTorch；TensorFlow 通过抽象 DataLoader / Callback / Metric Adapter 统一接口；条件参数（模型ViT 则开启 layer_decay；模型nnUNet 则开启 deep_supervision等。）
- 超参搜索：Optuna TPE + Median/Successive Halving Pruner；搜索维度示例：lr（log uniform）、batch_size（分层列表）、optimizer（AdamW/ Lion）、weight_decay、augment_prob、drop_path；自动早停 trial 降低资源浪费；交付：最优配置、全 trial 曲线、pruned 统计。
- 资源感知：、记录 GPU 利用率、吞吐 (samples/s)；输出：网络曲线与调整事件序列。

## 4. 性能与资源优化
- 动态批次回退：显存实时监测（NVML 池化 + EMA 平滑）→ 预测下步峰值；若预测>阈值触发 batch 乘 α（如0.85）并记录回退层级；指标：OOM 次数下降率、吞吐波动幅度。
- Checkpoint 精简：策略=（Top-K 最优指标 + 末尾最新）+ BF16 压缩；提供 sha256 校验与快速恢复脚本；统计存储占用下降%、恢复耗时、指标回滚偏差。
- 推理加速：ONNX Graph Simplify + TensorRT FP16/Fused Kernel；自动基准（不同 batch / 序列长度）生成延迟-吞吐曲线。
- Token 剪裁：注意力权重均值 + 熵双阈混合策略；蒸馏保留核心语义 Token；报告：指标（AUC/F1/Dice）曲线与拐点建议。

