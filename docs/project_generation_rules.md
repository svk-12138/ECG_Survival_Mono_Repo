# AI 项目生成指导规则

本规则用于提醒 AI 助手在创建新项目或脚手架时，应包含的文件、流程、要素、关键点和注意点。默认情况下所有版本都应满足“通用规则”，在此基础上根据项目类型选择相应章节（大模型训练 / 微调 / 蒸馏 / 强化学习）。测试与验证属于所有版本的共性要求。

---

## 1. 通用规则（适用于所有项目）
1. **目录结构**  
   - `README.md`：简介、功能、依赖、快速上手流程、常见问题。  
   - `requirements.txt` 或 `environment.yml`：列出依赖及版本。  
   - `data/`：用于存放原始与处理后的数据；需包含 `README`（或 `data/README.md`）说明数据格式、命名规则、下载/脱敏流程。对文本语料建议默认采用 `json/jsonl`（每行一个对象，字段含 `id`, `text`, `meta` 等），输入模型前如需拼接为 txt 或 token 列表，应提供相应的转换脚本与示例命令。  
   - `configs/`：集中存放 YAML/JSON 配置，便于修改。  
   - `scripts/`：功能脚本（数据处理、训练、评估、推理等）。  
   - `docs/`：额外说明（流程图、调参计划、部署说明等）。  
   - `logs/`、`outputs/`、`checkpoints/`（可延迟创建，需在 README 说明用途）。

2. **流程描述**  
   - 在 README 中给出“数据准备 → 配置 → 运行脚本 → 评估/部署”的完整步骤。  
   - 每个脚本需注明输入/输出、主要参数及示例命令；若需要终端命令，需分别给出 **Windows CMD、PowerShell 7、Linux/macOS** 三种写法（含续行符用法），确保不同平台都能直接复制运行。

3. **关键要素**  
   - 统一的配置加载方式（优先配置文件，其次命令行覆盖）。  
   - 日志记录：训练/推理过程写入 `logs/`；关键指标、异常需记录。  
   - 数据格式标准：  
     - 文本语料默认以 `json/jsonl` 保存，字段包含 `prompt/input`, `response/label`, `metadata`，如需兼容 txt，请在 `scripts/convert_*.py` 中支持。  
     - 图像/音频等二进制数据使用 `images/`, `audio/` 子目录，配 `manifest.csv/jsonl` 存储索引。  
     - 标注文件尽量采用可版本化格式（JSON、CSV、Parquet），并在 README 指出列名、单位、编码。  
   - 命名约定：统一大小写、用下划线，必要时在配置文件中以 `data.train_file` 等键暴露路径。

4. **注意点**  
   - 明确硬件/环境要求（Python 版本、GPU、驱动）。  
   - 给出安全提醒（API Key、隐私数据）与调参建议。  
   - 运行环境建议采用 Conda 虚拟环境（或同等可复现方式）：  
     - 在 README 中列出创建环境的完整步骤，例如 `conda create -n project python=3.10 -y`、`conda activate project`、`pip install -r requirements.txt`。  
     - 若有 CUDA/驱动依赖，需注明版本匹配关系（如 CUDA 11.8 + PyTorch 2.1）。  
     - 如果同时支持 venv/poetry 等方式，也要写明如何切换。  
   - 若依赖外部服务，写明配置方法和速率限制；涉及敏感凭证（API Key 等）时，需提供 `.env.example` 并在 README 中指明复制为 `.env` 后填入真实值，脚本端默认通过 `python-dotenv` 自动加载，避免明文放在命令或代码中。  
   - **代码注释规范**：  
     - 每个脚本 / 模块文件开头必须使用中文注释，说明“功能、主要依赖、输入输出”。  
     - 文件内的每个函数 / 方法需要撰写中文 docstring 或注释，描述参数用途和返回值。  
     - 对关键逻辑、接口调用、异常处理等重要位置添加简洁中文注释，帮助团队快速理解和排障。*** End Patch

5. **测试（共性功能）**  
   - 至少提供一个 `tests/` 或 `scripts/test_*.py` 例程，涵盖关键模块（如数据加载、模型前向）。  
   - README 说明如何运行测试；确保持续集成（若有）可接入。  
   - **一站式自检脚本**：额外提供 `scripts/run_all_tests.py`（或类似命名），用于执行以下流程：  
     - 检查关键文件/目录是否存在（数据、模板、输出目录、配置等）；  
     - 读取示例 JSONL/CSV，验证格式与字段；  
     - 可选地运行单元测试命令（如 `python -m unittest …`）；  
     - 可选地进行模型连通性测试（如调用教师 API 生成一条示例输出）。  
     README 中需给出示例命令和参数说明（例如 `--run-unit-tests`, `--test-teacher`, `--api-key`），确保一条命令即可完成项目跑通前的健康检查。

---

## 2. 大模型训练版本（从零开始）
1. **额外目录/文件**  
   - `configs/training.yaml`（包含模型结构、优化器、学习率计划）。  
   - `scripts/train.py`：支持断点续训、混合精度、分布式。  
   - `scripts/data_preprocess.py`：原始数据→训练样本 pipeline。

2. **流程与要素**  
   - 说明数据集规模、分布、切分策略，并在 `data/README.md` 中提供示例（如 JSON 字段说明、对齐方式、cls 标签列表）。  
   - 提供模型架构说明（网络模块、参数量、激活函数）。  
   - 训练脚本需支持：多 GPU/多节点配置、梯度累积、自动保存 checkpoint、早停策略。  
   - 加入监控 hooks（tensorboard/wandb 可选）。

3. **注意点**  
   - 强调算力需求、显存规划、分布式通信设置。  
   - 自检表：梯度是否爆炸、loss 是否 NaN、checkpoint 兼容性。

---

## 3. 大模型微调版本（基座 + 任务微调）
1. **目录/文件**  
   - `configs/finetune.yaml`：基座路径、LoRA/QLoRA/全参方式、任务数据路径。  
   - `scripts/finetune.py`：支持加载基座、可配置微调策略（LoRA、全参、Adapter）。  
   - `scripts/infer.py`：推理示例（单条输入、批量评估）。

2. **流程**  
   - 步骤：准备任务数据→生成/验证标签→切分 train/val/test→运行微调→评估/推理。  
   - README 加入“如何更新 prompt 模板、如何合并 LoRA”等指南。

3. **要素**  
   - 统一提示词模板，确保教师/学生/推理一致，模板文件内需标明输出 JSON schema 以及必填字段。  
   - 日志记录微调超参、训练集大小、效果指标，必要时输出每批次样例以便溯源。  
   - 数据规范：任务语料推荐以 `jsonl` 存储（字段：`instruction`, `input`, `output`, `meta`），若模型仅接受 txt，可通过 `scripts/build_txt_corpus.py` 自动拼接（如 `[INST] ... [/INST]`）。  
   - 提供模型权重保存位置及加载方式（尤其是 LoRA adapter）；若需要将 LoRA 融合成全参模型，应提供 `scripts/merge_lora.py`。

4. **注意点**  
   - 明确显存占用、batch size 与梯度累积关系。  
   - 若涉及医疗/隐私数据，强调脱敏和访问控制。  
   - 建议附加调参指南（如学习率、LoRA r/alpha、max_length 等）。

---

## 4. 大模型蒸馏版本（教师→学生）
1. **目录/文件**  
   - `configs/distill.yaml`：教师 API/模型配置、学生模型路径、数据路径。  
   - `scripts/generate_teacher_labels.py`、`scripts/prepare_dataset.py`、`scripts/train_student.py`、`scripts/evaluate_student.py`。  
   - `docs/distillation_plan.md`：说明为何选择该教师、指标、调参策略。

2. **流程**  
   - 提示词模板统一管理（教师与学生一致）。  
   - 标注生成 → 数据清洗 → 划分 train/val/test → 学生训练 → 评估（含教师 vs 学生对比）→ 可选的可视化/审查。  
   - 训练日志应记录教师响应摘要、学生损失、评估差值（如 JSON 解析成功率）。

3. **关键要素**  
   - 教师 API 速率限制与重试机制，需在脚本中实现指数回退/异常记录。  
   - 数据格式（JSONL 等）需在 README 中定义字段，并标注 `teacher_response`、`structured_result`、`audit_status` 等字段含义；如需导出 txt 批量训练，应提供转换脚本并记录最大长度、分隔符。  
   - 若学生使用 LoRA，保存 adapter、tokenizer，并说明如何与基座合并。

4. **注意点**  
   - 强调标签质量控制（人工抽查、脚本化校验）。  
   - 在日志中记录蒸馏前后指标差值，避免“训练成功但效果变差”。  
   - 若跨多数据集，需要单独预处理并在最终脚本统一汇总。

---

## 5. 大模型强化学习版本（RLHF / PPO / DPO 等）
1. **目录/文件**  
   - `configs/rl.yaml`：偏好数据、奖励模型、PPO/DPO 超参。  
   - `scripts/train_reward_model.py`、`scripts/rl_trainer.py`、`scripts/sample_generation.py`。  
   - `docs/rl_pipeline.md`：说明 SFT → RM → RL 的顺序与评估方法。

2. **流程与要素**  
   - **数据链**：偏好样本（正负对）、奖励模型训练、策略训练、对齐评估。  
   - **记录**：每次更新的 KL 值、reward、policy loss、采样样本。  
   - **安全检查**：输出过滤、对抗提示、拒答策略。

3. **注意点**  
   - 说明经验池/回放策略（若有）。  
   - 给出如何降期望（防止奖励黑客）和如何验证鲁棒性。  
   - 若资源有限，建议离线 RL（DPO/IPO）并说明优缺点。

---

## 6. 测试与验证（所有版本必须具备）
1. **单元/集成测试**  
   - 针对核心模块（数据处理、模型前向、解析器），提供可运行的测试脚本。  
   - 保证默认数据或示例数据即可运行，通过测试验证依赖是否正确。

2. **评估脚本**  
   - 无论训练/微调/蒸馏/RL，均需提供 `scripts/evaluate_*.py`，输出关键指标（精度、F1、合法 JSON 比例等），并支持从 `json/jsonl` 输入读取，必要时提供 `--from-txt` 选项以兼容 txt。  
   - 评估日志写入 `logs/`，包含配置、数据版本、指标值。

3. **可视化/审查**  
   - 若任务含结构化输出，提供部分可视化（如 waveform、样例对比图）或审查脚本，辅助人工核验。  
   - README 说明如何查看这些结果。

---

### 使用方式
- 在生成新项目时，先应用“通用规则”，再根据任务类型叠加对应章节。  
- 检查列表：目录是否齐全、README 是否涵盖流程、脚本是否具备日志/测试、是否有调参/安全提示。  
- 该规则可扩展到其他变体（如多模态、检索增强），原则是保持“结构明确、流程闭环、日志详尽、易于调参和复现”。
