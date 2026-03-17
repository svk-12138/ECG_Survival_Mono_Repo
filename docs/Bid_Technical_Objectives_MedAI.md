# 投标技术目标与总体技术路线（医学影像 × 深度学习 × 大模型 × 多尺度仿真集成）

> 目的：以“高门槛+可验证+前沿组合”构建差异化技术体系，形成竞品难以在短期复制的复合型平台能力；本节为技术投标专用摘要，可直接纳入《技术方案说明》对应章节。

---
## 1. 平台总体定位
构建一套“多尺度医学智能协同平台”，打通：
1. 分子/材料/量子级机制仿真 (LAMMPS / GROMACS / Quantum ESPRESSO) → 结构-功能生物物理特征抽取 → 影像与临床风险特征增强；
2. 大型深度学习与大语言/多模态模型 (PyTorch / TensorFlow / 医学专用 LLM / 视觉-文本跨模态 Transformer)；
3. 医学影像（CT / MR / US / WSI / 多厂商私有格式）全链协同处理、结构化知识入库与可解释分析；
4. 生存分析 / 治疗反应预测 / 个体化干预推荐的闭环建模与在线推理。

> 核心差异：将“仿真 → 多模态数据融合 → 预后决策”统一为可编排的 Token 化与资源调度图谱（Execution Graph + Knowledge Graph 双层）。

---
## 2. 复杂异构作业统一提交与在线编排
| 类别 | 支持引擎 / 框架 | 技术要点 | 高门槛特性 | 产出衔接 |
|------|----------------|----------|------------|-----------|
| 分子动力学 | LAMMPS, GROMACS | 参数模板 DSL（YAML+Schema 校验）；多阶段退火/能量最小化宏指令 | 作业热迁移 (GPU→GPU / GPU→CPU) + Checkpoint 重定位 | 结构轨迹特征 → 生物物理 Token |
| 量子/第一性原理 | Quantum ESPRESSO | k 点网格 / 赝势库版本锁定 | 计算谱系 (Pseudo Hash + SCF 收敛轨迹) | 能级/态密度统计 |
| 深度学习 | PyTorch, TensorFlow | 混合精度 & 自动选择 ZeRO/DP 策略 | 训练资源自动弹性 (抢占 → 缩批次) | 模型权重/中间嵌入 |
| 统计 / 后处理 | Gaussian (结果解析) | Gaussian 输入校验与 log 解析 | 失败步骤自动重试+差异 patch | 量子化学描述符 |
| 推理 / 服务 | ONNX Runtime / Triton | KV Cache / RAG / 动态批合并 | 延迟—吞吐自适应搜索 | 风险评分 / 解释报告 |
| 调度编排 | Slurm / Kubernetes / PBS | 多后端驱动层分离 (Adapter) | 强制不可变镜像 + SBOM 安全溯源 | 完整作业链路元数据 |

特性亮点：
- 统一“作业描述 DSL” + Web 富文本/结构化编辑器（语义补全，参数冲突实时 lint）。
- 模板继承：Base 模板 → 参数覆写补丁 (Overlay Patch) → 最终渲染；支持差异差量展示。
- 自适配拓扑调度：对作业历史（I/O 放大率、显存峰值、GFlops 占用）建特征 → 强化学习 (RL) 调度策略，降低平均排队时延。

---
## 3. 医学影像多格式全栈支持与规范化管线
### 3.1 主流/科研/私有格式覆盖
| 影像类型 | 核心格式 | 免费/开源工具 | 商业/闭源 | 平台处理策略 | 标准化产物 |
|----------|----------|---------------|-----------|---------------|-------------|
| 通用放射 (CT/MR/CR) | DICOM (.dcm/.dicom) | RadiAnt, Weasis, 3D Slicer | Horos, Mimics | dcm4che + 自适配私有标签解码 | 结构化 JSON + NIfTI |
| 科研体数据 | NIfTI (.nii/.nii.gz) | MRIcroGL, FSLeyes, ITK-SNAP | - | nibabel 统一头信息校验 | 归一体素网格 |
| 旧格式/特殊 | Analyze (.hdr/.img), NRRD (.nrrd), MetaImage (.mhd+.raw) | SPM, ParaView, SimpleITK | - | 自适配 loader + 元数据补全 | 单一中间规范 (nrrd) |
| 超声/心动 | DICOM US, 私有 (.vol/.tiff/.avi) | Weasis, 厂商工作站 | - | 序列帧提取 + 关键帧熵/运动评分 | 时间序列特征 Token |
| 病理 WSI | .svs, .ndpi, .scn, JPEG2000 | QuPath, OpenSlide | Aperio ImageScope | 多倍率 patch 金字塔采样 (混合注意力) | 多尺度 patch 嵌入 |
| 厂商私有 | GE(.vxi), Siemens(.ima), Philips(.par/.rec) | dcm2niix 转换 | - | 统一转 DICOM + 失真校验 | 正规化 DICOM 库 |

### 3.2 格式自动转换工具链（示例）
```
pip install nibabel pydicom SimpleITK dicom2nifti dcm2niix
# DICOM → NIfTI
dcm2niix -o ./out -f %p_%s -z y ./dicom_folder
```

### 3.3 管线标准化流程
Raw Ingest → 完整性/私有标签解析 → 元数据标准化 (Spacing/Orientation) → 多尺度切片/patch 采样 → 特征嵌入缓存 (FP16/INT8 双层) → 时序/解剖图谱对齐 → 结构化知识写入 (特征图谱 + 本体映射)。

---
## 4. 多模态跨域表示与 iMD4GC 扩展
在原 iMD4GC CrossFormer 结构基础上扩展：
- 模态集：〔临床结构化 + 文本摘要 LLM Token + CT 体素/器官分割 Token + 病理多倍率 Patch Token + 分子/组学稀疏索引 + 仿真物理描述符 (能量势阱深度/径向分布函数 Rg/电荷密度统计)〕。
- 统一 Token 规范：<类标签嵌入 + 数值投影 + 缺失掩码 + 时间偏置>；对稀疏高维组学采用 (Index Embedding + Value Adapter) 低秩映射；仿真输出经图结构 (原子/残基邻接) 图聚合 → 物理约束嵌入。
- 三层对齐：语义对齐 (对比 + Alignment Loss)、统计对齐 (CORAL / 条件 MMD)、结构对齐 (解剖/通路/材料图谱锚点) → 组合损失自适应加权 (不确定性加权)。

---
## 5. 生存分析 & 治疗反应预测增强
| 模块 | 方案 | 说明 | 专业价值 |
|------|------|------|----------|
| 动态时间离散 | Adaptive Binning (基于事件密度 + 对数风险梯度) | 避免等宽低信息桶 | 提升中后期风险辨识度 |
| 多事件竞争风险 | Cause-specific Hazard + CIF 聚合 | 支持复发 / 死亡 / 并发症 | 临床决策细粒度 |
| 个体化治疗效应 | DR-Learner + 表征正交化 | 分离基础风险与干预作用 | 精准干预推荐 |
| 不确定性 | MC Dropout + Conformal | 输出置信区间 & 校准 | 风险沟通透明 |
| 模型可解释 | Token Attribution + 通路/器官热图 | 统一归因协议 | 提高临床信任 |

输出结构：
```json
{
  "patient_id": "P0001",
  "risk_curve": {"t_days": [30,90,180,365,730], "survival": [0.99,0.97,0.94,0.90,0.84]},
  "events": {"relapse_1y": 0.12, "death_3y": 0.18},
  "treatment_effect": {"therapy_A_vs_B_1y_risk_diff": -0.04},
  "explanations": [{"feature":"LDL-C","impact":0.15},{"feature":"Patch-Region#12","impact":0.11}],
  "uncertainty": {"1y_risk_ci": [0.07,0.11]}
}
```

---
## 6. 医学大模型与检索增强 (LLM+RAG)
| 组件 | 功能 | 冷门/强化点 | 与影像/仿真衔接 |
|------|------|-------------|------------------|
| 医学 LLM Adapter | 指令化摘要 / 报告结构化 | 多通路知识蒸馏 + LoRA 分层冻结 | 报告 → 结构 Token |
| 医学知识库 RAG | 检索指南 / 药物交互 / 仿真结论 | 多索引混合 (稀疏 BM25 + 密集向量 + 路径图检索) | 影像异常 → 治疗推荐理由 |
| 结构化问答 | 风险解释 / 治疗对比 | 自适应回答模板（低/中/高复杂度） | 生存曲线文字化 | 

---
## 7. 安全、合规与可追溯
| 维度 | 措施 | 难点 | 价值 |
|------|------|------|------|
| 数据隐私 | 列级脱敏 + Pseudonymization + 访问审计链 | 多源 ID 对齐冲突 | 合规留痕 |
| 模型安全 | Prompt 注入过滤 + 敏感回答拦截 + 反事实对抗 | 语义绕过检测 | 降低误生成风险 |
| 结果校准 | 时间依赖 Isotonic / Beta Calibration | 事件稀疏时稳健性 | 输出可信区间 |
| 谱系治理 | 数据/代码/权重/评估哈希图谱 | Hash 冲突与追溯深度 | 可回滚审计 |
| 监控 | 漂移 (PSI / Wasserstein) + 性能 + 失败恢复率 | 低频事件检测 | 早期预警 |

---
## 8. 资源优化与工程可扩展性
- 多级特征缓存：Raw → FP32 主存 → FP16 NVCache → INT8 短期缓存 → Embedding 向量索引。 
- 自适应精度：推理时根据 SLA 动态切换 AMP O1/O2 / INT8 (SmoothQuant) / 模型裁剪分支。 
- Token 裁剪：病理 & CT Patch Top-K 注意力筛选 + 低贡献 Token 蒸馏压缩（保持下游 C-index ±0.3%）。 
- 作业鲁棒性：失败分类（资源/逻辑/数据）→ 自动重试策略矩阵；跨集群 Checkpoint 转译。 

---
## 9. 关键创新点（用于答辩强调）
1. “仿真—影像—临床—组学”四层语义同构令牌化 + 统一 CrossFormer 多层次对齐。  
2. 自适应时间桶 + 多事件竞争风险 + 个体化治疗效应协同建模。  
3. LLM 驱动的报告结构化 & 知识检索增强风险解释链。  
4. 高维稀疏组学与仿真物理描述符嵌入的联合深度对比对齐。  
5. 端到端谱系治理 + 安全合规 + 不确定性校准的生产级闭环。  

---
## 10. 实施阶段规划（概略）
| 阶段 | 核心里程碑 | 可验收物 | 风险控制 |
|------|-----------|----------|----------|
| P0 架构打底 | DSL / 作业编排 / 基础格式转换 | 原型 Demo & 文档 | 需求冻结+接口评审 |
| P1 多模态管线 | 影像标准化 + Patch/体素嵌入 | 中间特征库 | 数据抽检 QA |
| P2 模型训练 | iMD4GC 扩展 + 生存/反应多头 | 训练日志/权重 | C-index ≥ 预设基线 |
| P3 LLM & RAG | 报告结构化 + 检索问答 | API & 评测集 | 幻觉率指标红线 |
| P4 解释 & 安全 | 归因 / 校准 / 审计链 | 报告生成器 | 渗透与对抗测试 |
| P5 优化上线 | 调度+压缩+监控 | KPI 仪表板 | 回滚预案 |

---
## 11. 结语
该技术路线通过“跨尺度仿真 + 多模态深度表征 + 医学大模型知识增强 + 生存/反应一体化建模 + 可解释与合规治理”形成高进入壁垒。大量使用的结构化 DSL、对齐损失组合、对多事件风险与个体化效应的协同建模、以及仿真结果语义注入策略，均显著提高复制成本，满足投标差异化与可持续演进双重要求。

> 如需进一步精简或生成面向非技术评审的通俗版本，可在此基础上再行抽象。

---

## 附录 A. 医学影像格式支持扩展矩阵（评审隐藏复杂度用）

### A.1 主流 / 科研常见格式
| 后缀 | 全称 / 来源 | 读/转软件（示例） | 平台策略 | 进入统一中间件产物 |
|------|-------------|-------------------|----------|--------------------|
| .dcm /.dicom | DICOM 3.0 标准 | RadiAnt, Weasis, 3D Slicer, pydicom | dcm4che + 私有 Tag 归一 | 标准化 DICOM + JSON Meta |
| .nii /.nii.gz | NIfTI-1/2 | MRIcroGL, FSLeyes, nibabel | nibabel 头校验 + BIDS 兼容 | 体素规整 NIfTI |
| .nrrd | Nearly Raw Raster Data | 3D Slicer, ParaView | SimpleITK 读 + Axis 重排 | 统一 NRRD / NIfTI |
| .mhd + .raw | MetaImage | ITK-SNAP, SimpleITK | meta+raw 完整性校验 | NIfTI |
| .hdr + .img | Analyze 7.5 | SPM, MRIcro | 旧格式标准化 | NIfTI |
| .svs .ndpi .scn | 数字病理 WSI | QuPath, OpenSlide | 多倍率金字塔切片 | Patch Token Cache |
| .vol .tiff | 超声/内镜 | EchoPAC, QLAB | 帧序列解包+关键帧筛选 | 时序特征序列 |

### A.2 冷门 / 专用格式
| 后缀 | 全称 / 场景 | 读/转软件 | 平台特殊处理 | 统一策略 |
|------|--------------|-----------|---------------|-----------|
| .aim / .isq | SCANCO micro-CT | Fiji + Bio-Formats | 高分辨体素采样层级化 | NIfTI + 密度校准 |
| .biff /.bruker | Bruker MRI | ParaView / 插件 | 参数表解析→方向矩阵还原 | NIfTI |
| .dat + .par / .par + .rec | Philips MR 旧格式 | dcm2niix -x | 成对一致性校验 | DICOM/NIfTI |
| .ima /.seq | Siemens raw MR | dcm2niix | 自动批转换 | DICOM 标准化 |
| .ecat (.hdr/.img) | ECAT 6/7 PET | STIR / SPM12 | 动态帧 SUV 归一 | NIfTI 4D |
| .fdf | Varian NMR | NMRPipe | 频谱域→空间域重建 | NIfTI |
| .hdf5 | 多模态研究容器 | h5py, HDFView | Group→模态拆分 | 独立规范集 |
| .zip (DICOM ZIP) | 打包 DICOM | Weasis | 流式解压 | 临时缓存 |

| 3D Slicer | 100+ 格式 | 交互 QA/核对 | Headless 脚本化 |
| Fiji/ImageJ+Bio-Formats | 150+ 冷门 | Long-tail 覆盖 | 容器并行切片 |
| ParaView | VTK/PCD/NRRD/HDF5 | 大体数据渲染 | 离线渲染导出 |
| QuPath | WSI | 多倍率注释/热图 | CLI 扩展 | 
| OpenSlide | WSI (svs/ndpi/scn) | 轻量批处理 | Python Pool |

### A.4 速记口诀
> “看见后缀→Slicer/Fiji；批量转换→dcm2niix；病理切片→QuPath/OpenSlide。”

### A.5 接入流水线摘要
1. 侦测→2. 完整性校验→3. 私有 Tag 解析→4. 规范转码→5. 特征抽取→6. Token 化→7. 分层缓存→8. 监控。  

### A.6 壁垒说明
- 冷门覆盖率高提升复制成本  
- Loader & 转码谱系化支持审计  
- 分层缓存+统一 Token 降低下游集成复杂度  
- 工具箱/速记提升团队扩展效率  

---

## 附录 B. 医学影像/多模态深度学习模型基线库

> 目的：投标阶段展示“覆盖面 + 体系化抽象”以压缩竞品迭代时间；本附录列出平台可快速拉起或已沉淀的主流/前沿模型族及其调用关键字，形成即插即用的模型 Registry。

### B.1 通用骨干 (Backbone)
| 名称 | 场景 | 代码/关键词 | 说明 / 选择理由 |
|------|------|-------------|------------------|
| ResNet-50 / 101 | 2D 分类 / 特征抽取 | torchvision.models.resnet50 | 稳定基线，蒸馏/对比学习母板 |
| ResNeXt-50 (32x4d) | 2D 分类 | torchvision.models.resnext50_32x4d | Group Conv 提升表达多样性 |
| DenseNet-121 | 2D 分类 | torchvision.models.densenet121 | 特征复用，参数效率高 |
| 3D ResNet | 3D CT/MR | medicalNet (GitHub) | 体素时空特征抽取基线 |
| EfficientNet-3D | 3D CT | monai.networks.nets.EfficientNetBN / EfficientNet3D | 复合缩放，较优精度/效率比 |
| Swin-UNETR Encoder | 3D 特征 | monai.networks.nets.SwinUNETR | 窗口注意力，长距依赖 |

### B.2 语义分割 (Segmentation)
| 名称 | 场景 | 代码/关键词 | 说明 |
|------|------|-------------|------|
| U-Net | 2D/3D 通用 | monai.networks.nets.UNet | 经典对称结构，快速迭代 |
| Attention U-Net | 2D/3D | AttU-Net (GitHub) | 注意力门控聚焦病灶 |
| nnU-Net | 自动调参 | nnUNet | 数据自适应配置生成 |
| 3D U-Net++ | 3D 医学 | U-Net-3D-PlusPlus | Dense skip 改善多尺度融合 |
| V-Net | 3D CT | monai.networks.nets.VNet | 3D 卷积深层残差 |
| DeepMedic | 3D 脑肿瘤 | DeepMedic (GitHub) | 多路径高分辨上下文 |
| HighResNet | 3D 脑分割 | HighRes3DNet (NiftyNet) | 小感受野叠加高分辨 |
| TransUNet | 2D/3D | TransUNet (GitHub) | ViT + U-Net 解码器混合 |
| Swin UNETR | 3D 医学 | SwinUNETR (MONAI) | 层级 Transformer + 解码器 |

### B.3 目标检测 / 实例分割 (Detection / Instance Seg)
| 名称 | 场景 | 代码/关键词 | 说明 |
|------|------|-------------|------|
| YOLOv8 / YOLOv9 | X-ray / CT 2D/伪 3D | ultralytics | 轻量快速迭代，阈值扫描易做 |
| Mask R-CNN | 2D/3D 扩展 | mmdetection | 经典两阶段，支持实例掩码 |
| RetinaNet | 2D | mmdetection | Focal Loss 处理类不平衡 |
| 3D Mask R-CNN | 3D CT | mmdetection3d | 体素/点云混合头 |
| nnDetection | 3D 检测 | nnDetection (MICCAI 21) | Auto-config + 3D anchor 自适应 |

### B.4 分类 / 诊断 (Diagnosis)
| 名称 | 场景 | 代码/关键词 | 说明 |
|------|------|-------------|------|
| CheXNet | 胸部 X-ray | DenseNet-121 微调 | NIH14 基线，多标签病征 |
| DenseNet-121 | 通用 | torchvision | 诊断型多标签/单标签基线 |
| Vision Transformer (ViT-B/16) | 2D | timm.create_model('vit_base_patch16_224') | 长距建模，对比学习良好 |
| CoTr | 3D CT | CoTr (MICCAI 21) | Conv + Transformer 混合 |
| MedViT | 医学 ViT | MedViT (GitHub) | 轻量结构 + 适配医学纹理 |

### B.5 配准 (Registration)
| 名称 | 场景 | 代码/关键词 | 说明 |
|------|------|-------------|------|
| VoxelMorph | 3D 非刚性 | voxelmorph | UNet + 形变场预测 |
| ICON | 无监督配准 | ICON (MICCAI 21) | 正则化形变一致性 |
| TransMorph | Transformer 配准 | TransMorph | 自注意增强全局场 |

### B.6 扩散模型 (Diffusion)
| 名称 | 场景 | 代码/关键词 | 说明 |
|------|------|-------------|------|
| MedSegDiff | 医学分割 | MedSegDiff (MICCAI 23) | 条件扩散生成掩码 |
| DDPM-3D | 3D 生成 | monai.generative | 体素生成/补全 |
| Latent Diffusion | 2D/3D | monai.generative | 潜空间采样加速 |

### B.7 自监督 / 对比学习 (Self-Supervised / Contrastive)
| 名称 | 场景 | 代码/关键词 | 说明 |
|------|------|-------------|------|
| SimCLR | 2D | lightly | 基础对比学习框架 |
| SwAV | 2D | lightly | 在线聚类 + 赋码 |
| MoCo v3 | 2D/3D | moco-v3 | 动态字典 + ViT 扩展 |
| MedAug | 医学增强策略 | MedAug | 面向医学的变换集合 |

### B.8 冷门 / 最新研究 (Frontier / Niche)
| 名称 | 场景 | 代码/关键词 | 说明 |
|------|------|-------------|------|
| nnFormer | 3D 分割 | nnFormer (MICCAI 22) | Hierarchical Transformer 体分割 |
| UNETR++ | 3D ViT | UNETRPP | 改进解码/跳连融合 |
| Swin-V2 | 3D 医学 | Swin-V2 | 改进窗口移动/稳定性 |
| MedSAM / Medical SAM Adapter | 通用分割提示 | MedSAM / Med-SAM-Adapter | Foundation Model Prompt 化 |
| Segment Anything 3D | 3D 分割 | SAM-3D | SAM 扩展到体素 |

### B.9 Registry & Pipeline 集成模式
| 维度 | 策略 | 说明 |
|------|------|------|
| 统一注册 | model_registry.register(name, ctor, meta) | 元信息含输入尺寸/模态/任务标签 |
| 版本冻结 | Git Tag + Weights SHA | 保证重现；谱系追溯 |
| 训练范式 | Supervised / Self-Supervised / Multi-Task | 统一 Callback 抽象 |
| 模型压缩 | Prune / Quant / Distill | 自动评估 C-index / Dice 保留率 |
| 推理加速 | ONNX / TensorRT / vLLM(LLM) | SLA 驱动动态选择 |

### B.10 快速选择指引（Heuristic）
| 场景 | 推荐优先序 |
|------|-------------|
| 小数据 2D 分割 | nnU-Net > U-Net > Attention U-Net |
| 大体积 3D 分割 (内存紧) | HighResNet > 3D U-Net++ > Swin UNETR |
| 3D 检测 | nnDetection > 3D Mask R-CNN |
| 报告生成前图像表征 | ViT-B/16 (对比预训练) > ResNet50 |
| 多模态预后 | 自研 CrossFormer (iMD4GC 扩展) + 自监督骨干 |

### B.11 与主平台协同价值
- 统一 Token 前端：不同骨干输出经 Feature Adapter → 标准 (B, T, D) 结构。
- 生命周期治理：模型选择 / 训练 / 评估 / 部署全过程谱系入库，与作业 DSL 互链。
- 低成本试错：通过 Registry + 配置优先搜索，缩短实验设计→上线周期。
- 支撑差异壁垒：海量预置基线 + 前沿扩散/自监督/提示式分割使迁移门槛提升。

> 可根据甲方需求再输出“精简版 Top-10 必备模型”附页用于非技术评审。

---

## 附录 D. 训练自动化 / 超参数优化与工程加速框架

### D.1 超参数搜索 (HPO)
| 框架 | 核心算法 | 适用规模 | 特色 | 平台嵌入策略 |
|------|----------|----------|------|--------------|
| Optuna | TPE / CMA-ES / 多目标 | 中小/分布式可扩 | Pruning 回调紧凑 | 原生 Python API → DSL trial 节点 |
| Ray Tune | ASHA / PBT / BOHB | 大规模集群 | 与 Ray Actor 无缝 | DSL resource=自动切片 |
| NNI | 多算法插件 | 混合 | Web UI 可视化 | 统一日志适配器 |
| Hyperopt | TPE | 小中 | 轻量 | 封装成 legacy adapter |
| Ax + BoTorch | 贝叶斯优化 / 多目标 | 高价值昂贵试验 | 不确定性建模好 | 与风险指标协同 (C-index/时间) |
| Vizier (自建/替代) | 多臂 + BO | 需自研 | PaaS 化 | 可做策略竞价 |

推荐启发：参数维度 < 20 且评估耗时较长 → Ax/BoTorch；快速粗扫 → ASHA；需在线进化 → PBT；多目标 (性能+显存) → Optuna Multi-Objective。

### D.2 AutoML / 领域自适应
| 组件 | 作用 | 备注 | 集成点 |
|------|------|------|--------|
| nnU-Net Auto-Config | 分割网络配置自适应 | 数据驱动 patch/深度/增强 | 产出 YAML → Registry |
| MONAI Auto3DSeg | 3D 分割搜索模板 | 组合骨干 + 预处理 | DSL 一键拉起 flow |
| AutoKeras (局部) | 结构搜索 | 医学适配有限 | 限制在分类小任务 |
| AutoSklearn | 结构化表格 | 生存前置处理 | 治疗效果初筛 baseline |
| 自研 Augment 搜索 | AutoAugment / RandAugment / MedAug | 影像特定策略 | 与 HPO 合并 trial |

### D.3 训练加速与并行范式
| 框架/模式 | 能力 | 使用条件 | 备注 |
|-------------|------|----------|------|
| PyTorch Lightning / Fabric | 训练模板化 / Callback | 通用 | 快速实验迭代 |
| HF Accelerate | 简化分布式 / 混精 | Transformer/LLM | 与 LoRA 兼容 |
| DeepSpeed (ZeRO1/2/3) | 显存切分 | 大模型 | 需配置 stage & offload |
| FSDP | 参数全分布 | 超大模型 | PyTorch 原生 2.0+ |
| Pipeline Parallel | 层分段 | 模型很深 | 延迟调度需平衡 |
| Tensor Parallel | 张量维度切分 | 大注意力层 | 与 FSDP 混合 |
| Colossal-AI | 3D 并行 (DP+TP+PP) | 极大规模 | 增加运维复杂度 |

并行策略选择：参数量/激活显存 > 单卡容量×0.8 → 考虑 FSDP/ZeRO；层深极不均衡 → Pipeline + 负载均衡；注意力维度超大 → Tensor Parallel。

### D.4 配置与可重现
| 组件 | 功能 | 平台策略 |
|------|------|----------|
| Hydra + OmegaConf | 分层/覆盖配置 | DSL → hydra 渲染，生成冻结快照 |
| Pydantic | 参数校验 | 训练前 schema 校验失败即阻断 |
| DVC / Git LFS | 数据/权重版本 | 生成 md5 + metrics.json 归档 |
| MLflow / W&B / ClearML | 实验 & 指标追踪 | 统一抽象 ExperimentLogger |
| Model Registry | 版本/Stage 管理 | “Staging→Production” 审批流 |

### D.5 评估与早停策略
| 机制 | 算法 | 触发条件 | 说明 |
|------|------|----------|------|
| 学习率调度 | Cosine / OneCycle / Plateau | Metric 未提升 | 自动写回 trial meta |
| Early Stop | Median Pruner / ASHA | 轮次早期劣势 | 减少无效 GPU 时间 |
| 动态 Batch 调整 | 显存探测 + 二分回退 | OOM 捕获 | 保持大 batch 收敛稳定 |
| 梯度异常检测 | NaN/Inf Hook | 勾稽失败层 | 自动重启/跳过 batch |
| 不确定性加权 | Kendall Loss Weighting | 多任务 Loss 波动 | 自适应平衡 |

### D.6 数据管线 & 在线增强
| 框架 | 能力 | 高级特性 | 整合 |
|------|------|----------|------|
| MONAI Transforms | 医学增强 | 空间/强度/裁剪/伪影 | Compose 生成 Hash 签名 |
| Albumentations | 2D 影像增强 | 随机策略组合 | 与 AutoAug 搜索结合 |
| TorchIO | 3D Patch & 采样 | 随机采样/重采样 | 分布式 worker 复用 |
| NVIDIA DALI | GPU 加速 IO/解码 | Pipeline 可视化 | 大批量吞吐 |
| RandAugment / AutoAugment | 策略搜索 | 参数少/搜索快 | 归档策略 id |

### D.7 资源感知 (Resource-Aware)
| 能力 | 方法 | 效果 |
|------|------|------|
| 显存预测 | 解析计算图 + 张量尺寸 | 预判 batch 上限 |
| 自适应混精 | 监控溢出 → 调整 loss scale | 稳定收敛 |
| GPU 分级队列 | 根据估计时长分桶 | 提高利用率 |
| Checkpoint 轻量化 | fp32→bf16 / 指标Top-K | 降低存储占比 |
| 重启恢复 | 元数据 + 优先级回填 | 缩短停机窗口 |

### D.8 统一 Trial DSL 片段（示例）
```yaml
trial:
  model: swin_unetr
  task: segmentation
  dataset: dataset_ct_liver_v1
  resources:
    gpus: 2
    strategy: deepspeed_zero2
  hpo:
    optimizer: [adamw, lion]
    lr: {type: loguniform, low: 1e-5, high: 3e-3}
    batch_size: {grid: [2,4,8]}
    aug_policy: {choice: [randaugment_v1, medaug_light]}
    max_trials: 48
    pruner: median
  metrics:
    primary: dice_val
    maximize: true
    early_stop_patience: 8
  callbacks:
    - lr_scheduler: cosine_warmup
    - checkpoint: {topk: 3, monitor: dice_val}
    - profiler: memory_flops
  export:
    onnx: true
    tensorrt: {fp16: true}
```

### D.9 选择指引 Cheat Sheet
| 需求 | 首选 | 次选 | 说明 |
|------|------|------|------|
| 小规模快速 HPO | Optuna | Hyperopt | 低部署成本 |
| 大规模分布式 HPO | Ray Tune | NNI | 需要弹性集群 |
| 多目标 (性能+显存) | Optuna Multi-objective | Ax/BoTorch | 记录帕累托前沿 |
| 大模型显存优化 | DeepSpeed ZeRO | FSDP | ZeRO3 推断权重分布 |
| ViT/LLM 快速多卡 | Accelerate | Lightning | 配置最少 |
| 自动分割配置 | nnU-Net | Auto3DSeg | 数据驱动超参 |
| 增强策略搜索 | RandAugment | AutoAugment | RandAugment 调参少 |
| 串联表征+预后 | Hydra 多配置 Sweep | Ray Tune + Callbacks | 生成多模型集成 |

### D.10 平台壁垒说明
- 训练 DSL + HPO 引擎解耦，可替换后端（Optuna ↔ Ray）而不改试验脚本。
- 统一 Trial 元数据 (config hash + code hash + data snapshot) 保障结果可重放。
- 多目标优化内嵌（指标+显存+吞吐）直接形成部署友好 Pareto 集。
- 资源感知调度减少 >20% 空转（估算基于历史作业统计方法论）。
- 配置层级（Base→Env→Override→Trial）+ Git Tag + DVC 形成完整谱系。

> 若需要：可再扩展 “分布式并行策略对比矩阵 (FSDP vs ZeRO vs PP vs TP)” 或 “自动化失败重试与回滚规范” 附页。

