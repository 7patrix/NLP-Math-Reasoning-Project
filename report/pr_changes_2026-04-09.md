# PR 变更说明（2026-04-09）

## 范围
本次 PR 主要对齐以下三项决策：
1. 使用 `DeepSeek-R1-Qwen-1.5B`。
2. `MATH-500` 使用公开切分。
3. `AIME-2024` 切换为官方公开数据集 `HuggingFaceH4/aime_2024`。

## 变更文件
- `data/loader.py`
- `scripts/download_data.py`
- `data/AIME-2024/aime2024.json`
- `README.md`
- `report/progress_report.md`

## 变更内容与原因

### 1）模型对齐：DeepSeek-R1-Qwen-1.5B
- 文档中模型名称由 `DeepSeek-R1-Distill-Qwen-1.5B` 统一改为 `DeepSeek-R1-Qwen-1.5B`。
- 文档中的模型映射说明改为 `deepseek-ai/DeepSeek-R1-Qwen-1.5B`。

原因：
- 保持实验文档与当前代码配置一致，避免复现实验时模型版本混淆。

### 2）公开切分对齐：MATH-500
- `data/loader.py`：将 HF 回退数据源从 `hendrycks/competition_math` 的前 500 条，改为公开切分 `HuggingFaceH4/MATH-500`（`test`）。
- `scripts/download_data.py`：下载源改为 `HuggingFaceH4/MATH-500`（`test`），移除“截取前 500 条”的逻辑。
- 增加字段兼容映射（`problem/question`、`answer/final_answer/expected_answer`），降低字段差异导致的数据读取失败风险。

原因：
- 提高 benchmark 结果的可复现性，避免自定义子集带来的偏差。

### 3）AIME-2024 数据源对齐
- `scripts/download_data.py`：AIME 从“手动下载”改为自动从 `HuggingFaceH4/aime_2024`（`train`）下载，并保存到 `data/AIME-2024/aime2024.json`。
- `data/loader.py`：当本地文件不存在时，增加从 `HuggingFaceH4/aime_2024` 的自动回退加载。
- `data/AIME-2024/aime2024.json`：已用 `HuggingFaceH4/aime_2024` 的数据覆盖本地内容。
- `data/loader.py`：保留严格校验逻辑，若样本显式 `year` 且不为 `2024` 则报错。

原因：
- 明确 AIME 数据来源，确保来源可追溯、流程可复现，并与官方公开切分一致。

## 验证说明
- 静态代码与文档修改已完成。
- 本 PR 不包含完整实验重跑结果，仅包含数据与流程对齐改动。

## 数据来源
- AIME 数据集来源：`https://huggingface.co/datasets/HuggingFaceH4/aime_2024`
- 本 PR 已将代码逻辑与本地数据同步到该来源。
