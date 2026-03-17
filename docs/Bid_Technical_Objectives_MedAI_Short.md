## 一、快速查看 / 浏览
| 组件 | 功能 | 亮点 |
|------|------|------|
| 3D Slicer | 多模态加载、MPR/VR、分割、配准、批处理 | 丰富插件(VMTK/PyRadiomics/MONAI Label)+Python 脚本化 |
| ITK-SNAP | 交互式体素级分割与3D可视化 | 半自动主动轮廓/分水岭加速初始标注 |
| Weasis | DICOM 浏览与测量 | 跨平台+PACS 集成布局灵活 |
| Horos | macOS 影像浏览/测量 | 基于 OsiriX 分支、苹果生态友好 |
| MicroDicom | Windows DICOM 快速阅片 | 安装轻量、启动快、便携性好 |
| OHIF Viewer | Web 端 DICOM 浏览/标注 | React + DICOMweb 易嵌入、插件扩展 |
| Cornerstone 系列 | Web 渲染核心 SDK | 模块化 + 高性能，可定制前端 |
| DWV | 纯前端轻量浏览 | 零后端依赖，适合离线/快速验证 |

## 二、数据导入与标准化
| 组件 | 功能 | 亮点 |
|------|------|------|
| dcm2niix | DICOM→NIfTI/JSON 批量转换 | 速度快+保留关键元数据(BIDS 友好) |
| plastimatch | 多模态配准/重采样/RT 支持 | 放疗结构/剂量处理一体化 |
| dicom-csv | DICOM 元数据结构化导出 | 快速统计与仓库存储接口 |
| pydicom | 标签读写/脱敏 | API 简洁，脚本化灵活 |
| SimpleITK / ITK | 重采样/滤波/配准/方向统一 | 稳定算法栈，多语言绑定 |
| NiBabel | NIfTI/MGH/MINC 读写 | 与 NumPy 紧密衔接，轻量 |
| MONAI | 深度学习预处理增强管线 | 丰富 transforms + cache 加速 |
| TorchIO | 医学影像增强与采样 | Patch 采样+GPU 加速友好 |
| 3D Slicer 批量脚本 | 批处理转换/分割/导出 | GUI+无头双模式运行 |
| MONAI App SDK | 预处理/推理应用封装 | 快速构建容器化 AI 服务 |
| Clara Deploy | 容器算子流水线平台 | GPU 优化算子 + 工作流编排 |

## 三、质量控制 / 数据比对
| 组件 | 功能 | 亮点 |
|------|------|------|
| dicom-anonymizer | DICOM 规则脱敏 | 自定义保留/移除 Tag 策略模板 |
| pydicom diff(自写) | 标签差异比对 | 针对项目定制关键字段校验 |
| dcmqi | SEG/RTSS 结构化互转 | 算法结果标准化临床封装 |
| DCMTK (dcmdump/dcmdiff) | 命令行 DICOM 工具 | 成熟稳定、脚本友好 |
| MRIQC | MRI 质量自动评估 | 多中心一致 QC 指标基线 |
| MONAI ImageQuality | 质量/伪影特征抽取 | 可嵌入训练前过滤/在线监测 |
| SimpleITK 指标 | 自定义 SNR/CNR/统计 | 灵活组合形成 QC 报告 |
| ITK-SNAP / Slicer Segment Comparison | 分割差异可视与指标 | 重叠/距离直观复核 |
| medpy | 分割评估函数库 | 轻量即用（Dice/Hausdorff） |
| EvaluateSegmentation | 批量命令行评估 | 自动化流水线集成方便 |

## 四、数据筛选 / 查询 / Cohort 构建
| 组件 | 功能 | 亮点 |
|------|------|------|
| Orthanc | 轻量 DICOM 服务与索引 | REST/插件扩展+易部署 |
| dcm4chee | 企业级 PACS/归档 | 高可用+审计合规完善 |
| XNAT | 影像+元数据仓库 | Cohort builder+管线执行 |
| Postgres/Elastic+pydicom | 自建元数据索引 | 高度灵活复杂查询 |
| OHIF + Orthanc | Web 交互筛选 | 零本地安装快速协作 |

## 五、标注 / 半自动分割
| 组件 | 功能 | 亮点 |
|------|------|------|
| Slicer Segment Editor | 多标签体素交互分割 | 多工具组合+脚本扩展 |
| ITK-SNAP | 半自动种子分割 | 快速初稿+学习曲线低 |
| MONAI Label | AI 辅助在线标注 | 服务器推理实时建议 |
| MITK | 插件式交互与标注 | 可嵌入自定义算法 UI |
| CVAT | Web 2D 框/多边形/点 | 协作/任务管理完善 |
| Label Studio | 多模态标注平台 | 统一界面+插件扩展 |
| VIA | 轻量本地浏览器标注 | 无需后端，适小批量 |
| Slicer VMTK 插件 | 血管中心线与测量 | 可视化联动便于修正 |
| VMTK CLI | 血管网格/中心线/半径 | 脚本化串联血管分析 |

## 六、医学指标 / 特征提取
| 组件/模型 | 功能 | 亮点 |
|-----------|------|------|
| PyRadiomics | 放射组学特征提取 | YAML 配置可复现+Slicer 集成 |
| VMTK | 血管中心线/狭窄分析 | 支持网格/半径/分叉角参数输出 |
| SimpleITK + 脚本 | 自定义纹理/形态统计 | 组合滤波+灵活管线拼装 |
| radiomics-features | 共生矩阵/GLSZM 等 | 直接嵌入 Python pipeline |
| OpenCV + 光流 | 造影序列时序特征 | 动态灌注/对比剂流动指标构建 |
| MONAI 时序 transforms | 时序数据增强/重排 | 减少过拟合提升泛化 |
| (UNet) | 编解码跳连分割 | 结构简洁稳健基线 |
| (nnUNet) | 自适配分割框架 | 零手工调参高性能 |
| (Swin-UNETR) | Transformer+解码分割 | 远程依赖与多尺度捕获 |
| (DeepLabV3+) | 空洞卷积分割 | 多尺度边界精细化能力 |

## 七、数据脱敏与合规
| 组件 | 功能 | 亮点 |
|------|------|------|
| dicom-anonymizer | DICOM 批量脱敏 | 可配置策略模板减少遗漏 |
| DICOMCleaner | 图形界面脱敏/审查 | 适合非技术人员操作 |
| pydicom 自写规则 | 精细字段保留/清洗 | 动态更新适配新合规需求 |
| Orthanc 日志 + ELK | 访问/操作审计 | 集中化检索+异常检测 |
| XNAT 审计插件 | 仓库级操作记录 | 支持审核导出合规文档 |

## 八、数据导出 / 打包
| 组件 | 功能 | 亮点 |
|------|------|------|
| Slicer 批量导出脚本 | 分割/重采样结果导出 | 支持 NIfTI/OBJ/STL + 统一命名 |
| plastimatch | 结构重采样/格式转换 | 放疗/影像跨系统迁移方便 |
| dcm4che storescu | 结果回写 PACS | 无缝融入临床工作流 |
| Orthanc REST 导出 ZIP | 按条件打包下载 | 自动化训练集抽取 |

## 九、端到端平台
| 组合栈 | 功能范围 | 亮点 |
|--------|----------|------|
| Orthanc + OHIF + Slicer/MONAI Label + PyRadiomics/VMTK + Nextflow/Airflow + DVC/MLflow | 采集→浏览→标注→特征→流水线→版本追踪闭环 | 开源可替换、低锁定、逐步扩展 |
