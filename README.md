# NLP Math Reasoning Project

**CS6493: Natural Language Processing - Group Project**  
**Topic**: Mathematical Reasoning Ability of Large Language Models

---

## 📋 项目概述

### 任务目标
评估大语言模型（Qwen2.5-Math-1.5B 和 DeepSeek-R1-Qwen-1.5B）在数学推理任务上的表现，系统对比三种提示工程方法的效果。

### 实验配置（18组实验）

| 模型 | 数据集 | 方法 |
|------|--------|------|
| Qwen2.5-Math-1.5B | GSM8K, MATH-500, AIME-2024 | CoT, Self-Refine, Self-Consistency |
| DeepSeek-R1-Qwen-1.5B | GSM8K, MATH-500, AIME-2024 | CoT, Self-Refine, Self-Consistency |

> 说明：命令行中的模型键使用 `deepseek-r1-qwen-1.5b`，其对应的 Hugging Face 模型为 `deepseek-ai/DeepSeek-R1-Qwen-1.5B`。

### 评估指标
- **Accuracy**: 准确率（模型答案与标准答案匹配率）
- **Response Length**: 平均响应长度（用于分析推理深度）

### 时间节点
- **3月25日**: 进度报告
- **4月14日**: 课堂展示（15分钟）
- **4月15日**: 最终提交（报告+代码+幻灯片）

---

## 🗂️ 项目结构

```
.
├── data/                            # 数据集
│   ├── loader.py                   # 数据加载器（统一入口）
│   ├── MATH-500/
│   │   └── test.json              # 500道测试题
│   ├── GSM8K/
│   │   └── test.json              # ~1319道测试题（需运行脚本下载）
│   └── AIME-2024/
│       └── aime2024.json          # 30道竞赛题
├── models/                          # 模型加载
│   └── loader.py                   # Qwen/DeepSeek 加载器
├── prompts/                         # 提示工程实现
│   ├── cot.py                     # Chain of Thought
│   ├── self_refine.py             # Self-Refine 迭代改进
│   └── self_consistency.py        # Self-Consistency 多数投票
├── evaluation/                      # 评估
│   └── metrics.py                 # 准确率 / 答案提取 / 规范化
├── experiments/                     # 实验运行
│   └── runner.py                  # 单实验运行器（支持 --limit 测试）
├── scripts/                         # 工具脚本
│   └── download_data.py           # 数据集下载脚本
├── results/                        # 实验结果（自动生成）
├── app.py                          # Streamlit 可视化界面
├── run_batch.py                    # 批量实验脚本 ⭐
├── requirements.txt                 # Python 依赖
└── README.md
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 推荐使用 conda 或 venv 创建独立环境
conda create -n nlp_math python=3.10
conda activate nlp_math

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载数据集（如需）

```bash
# 下载 GSM8K 和 MATH-500（如已手动放置可跳过）
python scripts/download_data.py
```

> 当前实现中，`scripts/download_data.py` 会从公开切分 `HuggingFaceH4/MATH-500` 的 `test` split 下载并写入 `data/MATH-500/test.json`。

> AIME-2024 使用公开数据集 `HuggingFaceH4/aime_2024`（`train` split，30题），并由 `scripts/download_data.py` 自动写入 `data/AIME-2024/aime2024.json`。

### 3. 设置模型缓存（重要！）

模型权重会缓存到 Hugging Face 默认目录。如需指定目录：

```bash
# Windows
set HF_HOME=E:\huggingface_cache

# Linux / Mac
export HF_HOME=~/huggingface_cache
```

首次运行时会自动从 Hugging Face Hub 下载模型（需要网络连接）。

### 4. 运行实验

#### 批量运行（推荐）

```bash
# 运行所有18组实验（约5-6小时，串行）
python run_batch.py --all

# 只运行 Qwen 的9组实验
python run_batch.py --model qwen2.5-math-1.5b

# 只运行 MATH-500 的6组实验
python run_batch.py --dataset math500

# 只运行单个实验
python run_batch.py --id 4
```

#### 手动运行单个实验

```bash
# 格式: python experiments/runner.py --model <模型> --dataset <数据集> --method <方法>

# 示例: Qwen + MATH-500 + CoT
python experiments/runner.py --model qwen2.5-math-1.5b --dataset math500 --method cot

# 示例: DeepSeek + GSM8K + Self-Consistency
python experiments/runner.py --model deepseek-r1-qwen-1.5b --dataset gsm8k --method self_consistency
```

#### 快速测试（少量样本）

```bash
# 用前5条数据快速验证流程，不生成正式结果文件
python experiments/runner.py --model qwen2.5-math-1.5b --dataset math500 --method cot --limit 5
```

#### 查看结果

```bash
# 启动可视化界面
python -m streamlit run app.py
```

浏览器打开 `http://localhost:8501`，支持：单题测试 / 查看已有结果 / 多实验对比。

---

## 📊 18组实验清单

| ID | 模型 | 数据集 | 方法 | 预计时间 | 状态 |
|----|------|--------|------|---------|------|
| 1 | Qwen2.5-Math-1.5B | GSM8K | CoT | ~40min | ⏳ |
| 2 | Qwen2.5-Math-1.5B | GSM8K | Self-Refine | ~80min | ⏳ |
| 3 | Qwen2.5-Math-1.5B | GSM8K | Self-Consistency | ~200min | ⏳ |
| 4 | Qwen2.5-Math-1.5B | MATH-500 | CoT | ~15min | ✅ |
| 5 | Qwen2.5-Math-1.5B | MATH-500 | Self-Refine | ~30min | ⏳ |
| 6 | Qwen2.5-Math-1.5B | MATH-500 | Self-Consistency | ~75min | ⏳ |
| 7 | Qwen2.5-Math-1.5B | AIME-2024 | CoT | ~1min | ⏳ |
| 8 | Qwen2.5-Math-1.5B | AIME-2024 | Self-Refine | ~2min | ⏳ |
| 9 | Qwen2.5-Math-1.5B | AIME-2024 | Self-Consistency | ~5min | ⏳ |
| 10 | DeepSeek-R1-Qwen-1.5B | GSM8K | CoT | ~40min | ⏳ |
| 11 | DeepSeek-R1-Qwen-1.5B | GSM8K | Self-Refine | ~80min | ⏳ |
| 12 | DeepSeek-R1-Qwen-1.5B | GSM8K | Self-Consistency | ~200min | ⏳ |
| 13 | DeepSeek-R1-Qwen-1.5B | MATH-500 | CoT | ~15min | ⏳ |
| 14 | DeepSeek-R1-Qwen-1.5B | MATH-500 | Self-Refine | ~30min | ⏳ |
| 15 | DeepSeek-R1-Qwen-1.5B | MATH-500 | Self-Consistency | ~75min | ⏳ |
| 16 | DeepSeek-R1-Qwen-1.5B | AIME-2024 | CoT | ~1min | ⏳ |
| 17 | DeepSeek-R1-Qwen-1.5B | AIME-2024 | Self-Refine | ~2min | ⏳ |
| 18 | DeepSeek-R1-Qwen-1.5B | AIME-2024 | Self-Consistency | ~5min | ⏳ |

> **总计**: ~5.5小时（串行）。建议晚上挂机运行，或两台机器分别跑 Qwen / DeepSeek。
>
> 实验中断后重新运行，已完成的结果会自动跳过（无需担心重复运行）。

---

## 👥 小组分工建议

### 方案A：按模型分工（推荐）
- **成员A**: 负责 Qwen 的9组实验（ID 1-9）
- **成员B**: 负责 DeepSeek 的9组实验（ID 10-18）

### 方案B：按数据集分工
- **成员A**: GSM8K 的6组（ID 1-3, 10-12）
- **成员B**: MATH-500 的6组（ID 4-6, 13-15）
- **成员C**: AIME-2024 的6组（ID 7-9, 16-18）+ 报告撰写

### 方案C：混合分工
- **成员A**: 运行所有实验
- **成员B**: 数据分析与可视化
- **成员C**: 报告 + 幻灯片

---

## 🔧 常见问题

### Q1: 模型下载很慢 / 失败？
确保网络可以访问 Hugging Face Hub。设置镜像站点（可选）：
```bash
# 使用 HF Mirror（国内加速）
set HF_ENDPOINT=https://hf-mirror.com
```

### Q2: 显存不足（OOM）？
两个模型**不要同时加载**，实验脚本每次只加载一个模型。如仍 OOM，减少 `max_new_tokens` 参数。

### Q3: 实验中断怎么办？
直接重新运行。脚本默认会跳过已有结果文件（`--skip-existing`），不会重复运行。

### Q4: 如何强制重新运行某个实验？
```bash
python experiments/runner.py --model qwen2.5-math-1.5b --dataset math500 --method cot --no-skip
```

### Q5: GSM8K 数据报错？
需要先运行数据下载脚本：
```bash
python scripts/download_data.py
```

### Q6: 如何修改模型生成参数？
编辑 `models/loader.py` 中的 `generate_response` 函数，调整 `temperature`、`top_p`、`max_new_tokens` 等参数。

---

## 📝 进度报告模板

见 `report/progress_report.md`

建议结构：Introduction / Related Work / Methodology / Preliminary Results / Challenges / Next Steps

---

## 📚 参考文献

1. Wei J et al. Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS*, 2022.
2. Madaan A et al. Self-refine: Iterative refinement with self-feedback. *NeurIPS*, 2024.
3. Wang X et al. Self-consistency improves chain of thought reasoning in language models. *arXiv:2203.11171*, 2022.

---

**最后更新**: 2026-03-25
