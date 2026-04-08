# PR2 变更说明（2026-04-09）

## 范围
本次 PR 聚焦“实验严谨性与可复现性”改进，主要解决以下问题：
1. Self-Consistency 的多数投票结果与最终评估答案可能不一致。
2. CoT / Self-Refine 采样默认随机，导致同配置多次运行波动。
3. 答案抽取规则过于脆弱，容易误抽中间过程数字。
4. 结果文件缺少关键运行元数据，不利于复现实验与解释差异。
5. 报告中“已完成实验”与仓库结果文件状态不一致。

## 变更文件
- `models/loader.py`
- `prompts/cot.py`
- `prompts/self_refine.py`
- `prompts/self_consistency.py`
- `evaluation/metrics.py`
- `experiments/runner.py`
- `README.md`
- `report/progress_report.md`

## 变更内容与原因

### 1）Self-Consistency 与评估口径对齐
- `prompts/self_consistency.py`
  - 增加 `dataset_type` 参数，按数据集提取答案。
  - 保留多数投票机制后，返回文本末尾显式追加：
    - `Final voted answer: ...`
    - `The answer is ...`

原因：
- 保证“多数投票答案”与后续评估抽取的一致性，避免票选答案与最终计分答案不一致。

### 2）可复现性增强（seed + 解码策略）
- `models/loader.py`
  - `generate_response` 新增 `do_sample` 参数；仅在采样时使用 `temperature/top_p`。
- `prompts/cot.py`、`prompts/self_refine.py`
  - 默认改为 `do_sample=False`，确保同输入稳定输出。
- `experiments/runner.py`
  - 新增 `--seed` 参数（默认 42）。
  - 在实验启动时统一设置 `random / numpy / torch` 随机种子。

原因：
- 降低 CoT / Self-Refine 的随机波动，支持可重复实验与稳定对比。

### 3）答案抽取规则增强
- `evaluation/metrics.py`
  - 增加 dataset_type 归一化分支（GSM8K/AIME）。
  - 优先识别最后一次显式答案声明（`final answer` / `the answer is` / `answer:`）。
  - 等号规则改为“取最后一个等式结果”而非第一个。

原因：
- 减少误抽中间计算值，提升自动评估准确性。

### 4）结果元数据补充
- `experiments/runner.py`
  - 结果 JSON 新增 `run_metadata`，包含：
    - `seed`
    - `started_at`
    - `duration_seconds`
    - `device`
    - `decode_config`
    - `max_new_tokens`

原因：
- 增强结果可解释性与可追溯性，便于复盘配置差异。

### 5）文档与报告对齐
- `README.md`
  - 单实验与快速测试命令示例加入 `--seed 42`。
- `report/progress_report.md`
  - 移除“已完成 ID 4”表述，改为“仓库尚未提交最终结果文件，实验已可运行”。

原因：
- 保持仓库事实与文本描述一致，避免评审误解。

## 验证说明
- 相关 Python 文件已通过静态错误检查（无语法错误）。
- 本 PR 为方法与评估行为改动，尚未在本 PR 内附带完整 18 组重跑结果。

## 兼容性说明
- 旧命令仍可用；不传 `--seed` 时默认使用 `42`。
- 结果文件格式向前兼容：新增 `run_metadata` 字段，不破坏原有字段读取。
