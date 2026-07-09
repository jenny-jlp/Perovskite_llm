# Perovskite_LLM — Project Overview for AI Agents

## 项目目标（Research Goal）

本项目旨在将大型语言模型（LLM）技术迁移并应用于钙钛矿太阳能电池（Perovskite Solar Cells, PSC）的科学研究中。

**核心方向：** 基于真实实验数据库，通过指令微调（Instruction Fine-Tuning）和 QLoRA 参数高效适配，将开源 LLM 转化为专精于钙钛矿器件配方理解、光电性能预测与逆向配方设计的领域专家模型。

**重点研究体系：** 三阳离子混卤素钙钛矿 `Cs_x(FA_y MA_{1-y})_{1-x} Pb(I_z Br_{1-z})_3`（CsFAMA Pb IBr），该体系在钙钛矿太阳能电池领域是目前性能最优、稳定性最好的主流实用体系之一。

---

## 参考文献（Base Paper）

本项目的技术灵感来源于以下论文：

> **"Crystal structure generation with autoregressive large language modeling"**
> Antunes et al., *Nature Communications*, 2024
> 文件路径：`Antunes 等 - 2024 - Crystal structure generation with autoregressive large language modeling.pdf`

原论文使用自回归 LLM（CrystaLLM）将 CIF 格式的晶体结构文件视为文本序列，通过生成式语言建模来预测和生成新的无机晶体结构。

**本项目的迁移与创新：**
- 脱离原论文的三维 CIF 结构生成方向
- 转向面向真实实验数据的表格式器件配方预测任务
- 覆盖了原论文难以处理的**有机-无机杂化体系（CsFAMA）**
- 研究范式从"晶体学生成"切换到"实验配方理解 + 性能预测 + 逆向设计"

---

## 技术架构（Technical Architecture）

```
数据来源 ──────────────────────────────────────────────────────┐
│  The Perovskite Database (Zenodo Record 5837035)            │
│  42,443 个真实 PSC 器件实验记录 / 410 个特征维度             │
│  来源：perovskitedatabase.com / NOMAD 实验数据库             │
└──────────────────────────────────────────────────────────────┘
          │ prepare_llm_dataset.py
          ▼
数据处理 ──────────────────────────────────────────────────────┐
│  • 清洗缺失值，保留 41,519 条有效 PCE 记录                  │
│  • 分离 CsFAMA 体系（3,833 条）与其他钙钛矿（37,686 条）     │
│  • 转换为指令微调格式（Instruction-Output 问答对）           │
│    - Input: 器件配方文本（组分、传输层、工艺参数等）         │
│    - Output: 光电性能指标（PCE, Voc, Jsc, FF）              │
│  • 训练集（38,867 条）：perovskite_llm_train.jsonl           │
│  • 测试集（2,652 条）：perovskite_llm_test.jsonl             │
└──────────────────────────────────────────────────────────────┘
          │ train_lora.py
          ▼
模型微调 ──────────────────────────────────────────────────────┐
│  基础模型：Qwen/Qwen2.5-3B-Instruct（30 亿参数）            │
│  微调方式：QLoRA (4-bit NF4 量化 + LoRA r=16, alpha=32)     │
│  可训练参数：29,933,568 / 3,115,872,256 (0.96%)             │
│  硬件：NVIDIA GeForce RTX 5060 Ti (16GB VRAM)               │
│  CUDA 版本：13.0 (PyTorch cu130 专用构建)                    │
│  训练配置：3 Epochs, lr=2e-4, batch_size=4, grad_accum=4    │
│  Loss 监控：每 10 步记录，每 100 步在 CsFAMA 测试集上评估    │
└──────────────────────────────────────────────────────────────┘
          │ inference.py
          ▼
模型推理 ──────────────────────────────────────────────────────┐
│  • 加载 LoRA 适配器权重 + 基础模型                           │
│  • 支持"正向预测"：配方文本 → 预测 PCE/Voc/Jsc/FF           │
│  • 支持"交互式问答"：终端输入自定义配方，实时输出预测        │
└──────────────────────────────────────────────────────────────┘
```

---

## 项目文件结构（File Structure）

```
e:/Antigravity_storage/Perovskite_LLM/
├── AGENTS.md                          # 本文档（AI 代理阅读入口）
│
├── Antunes 等 - 2024 - Crystal structure generation ... .pdf
│                                      # 参考文献原文 PDF
│
├── download_perovskite_db.py          # 从 Zenodo 下载钙钛矿数据库
├── retrieve_data.py                   # （备用）从 COD 下载无机 CIF 结构（已停用）
├── explore_data.py                    # 数据库统计分析脚本
├── prepare_llm_dataset.py             # 数据清洗 + 指令微调格式化 + 训练集/测试集切分
├── train_lora.py                      # QLoRA 微调训练主脚本
├── inference.py                       # 模型推理与交互预测脚本
│
└── data/
    ├── Jesperkemist-perovskitedatabase_data-9b6ed4c/
    │   └── data/
    │       └── Perovskite_database_content_all_data.csv  # 原始数据库（84.7 MB）
    └── processed/
        ├── perovskite_llm_train.jsonl  # 微调训练集（38,867 条问答对）
        └── perovskite_llm_test.jsonl   # 独立测试集（2,652 条，含 CsFAMA 标签）

（微调训练完成后会新增）
└── output/
    └── perovskite_qwen_lora/
        ├── adapter_model.safetensors   # 训练好的 LoRA 权重
        ├── adapter_config.json
        └── tokenizer_config.json 等
```

---

## 数据集说明（Dataset Summary）

| 项目 | 内容 |
| :--- | :--- |
| **数据库名称** | The Perovskite Database (v1.0.1 Archive) |
| **来源** | Zenodo Record 5837035 (CC BY 4.0 License) |
| **总样本数** | 42,443 个器件记录（过滤后有效：41,519 条） |
| **特征维度数** | 410 个字段（组分、工艺、传输层、性能、稳定性等） |
| **CsFAMA 样本数** | **3,833 条**（占 9.2%）|
| **PCE 范围** | 0.00% – 36.20%（平均 12.03%）|
| **最常用 ETL** | TiO2-c \| TiO2-mp (27.5%), TiO2-c (16.2%), SnO2-np (3.9%) |
| **最常用 HTL** | Spiro-MeOTAD (49.2%), PEDOT:PSS (15.5%), PTAA (4.4%) |

**关键数据列（与本项目最相关）：**
- `Perovskite_composition_short_form`：化学式简写（如 `CsFAMAPbBrI`）
- `Perovskite_composition_long_form`：精确化学式（含系数）
- `Perovskite_composition_a/b/c_ions` + `_coefficients`：A/B/X 位离子及配比
- `ETL_stack_sequence` / `HTL_stack_sequence`：传输层材料
- `Perovskite_deposition_solvents`：沉积溶剂
- `Perovskite_deposition_thermal_annealing_temperature/time`：退火工艺
- `JV_default_PCE` / `JV_default_Voc` / `JV_default_Jsc` / `JV_default_FF`：光电性能

---

## 环境配置（Environment Setup）

```bash
# 创建 Conda 环境
conda create -n perovskite_llm python=3.10 -y
conda activate perovskite_llm

# 安装 CUDA 13.0 专用版 PyTorch（针对 RTX 50 系列显卡）
pip install torch --index-url https://download.pytorch.org/whl/cu130

# 安装 Hugging Face 微调工具链
pip install transformers accelerate peft trl bitsandbytes datasets

# 安装数据处理依赖
pip install requests numpy pandas scipy
```

**硬件要求：**
- GPU：NVIDIA RTX 5060 Ti 16GB（CUDA 13.0）
- RAM：建议 ≥ 16 GB 系统内存
- 磁盘：建议 ≥ 20 GB 可用空间（含模型缓存）

---

## 运行流程（Workflow）

```bash
# 第一步：下载原始数据库（已完成）
python download_perovskite_db.py

# 第二步：格式化训练数据（已完成）
python prepare_llm_dataset.py

# 第三步：启动 QLoRA 微调训练（训练中）
python train_lora.py

# 第四步：使用训练好的模型进行预测
python inference.py
```

---

## 研究创新点（Research Novelty）

1.  **解决数据稀疏性（Sparsity）问题**：相比传统机器学习（XGBoost/RF）要求完整数值矩阵，LLM 对含有大量 `Unknown`/`NaN` 字段的高稀疏性实验数据具有天然的鲁棒性。
2.  **双向建模能力**：支持正向预测（配方 → 性能）与逆向设计（目标性能 → 生成配方），这是传统回归模型无法实现的。
3.  **领域专化（CsFAMA 专家化）**：通过对 CsFAMA 体系数据的过采样与独立评估，使模型成为该先进钙钛矿体系的专精预测器。
4.  **可扩展性**：数据集、模型规模（3B→7B）、额外数据源（最新文献）均可灵活扩展以持续提升性能。

---

## 下一步计划（Roadmap）

- [ ] 完成 3 个 Epoch 的 QLoRA 微调训练
- [ ] 运行 `inference.py`，评估模型对 CsFAMA 体系的预测精度
- [ ] 从近期文献（2024-2026）中补充最新高效率 CsFAMA 器件数据（目标：+500 条高 PCE 样本）
- [ ] 升级至 `Qwen2.5-7B-Instruct` 基础模型以提升推理能力
- [ ] 引入逆向设计任务：输入目标 PCE 区间，生成最优 CsFAMA 配方
- [ ] 撰写学术论文并开源数据集与模型权重
