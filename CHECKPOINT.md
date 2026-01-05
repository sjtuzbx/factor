Checkpoint 使用心得

适用脚本：
- `fill_balance_fina_daily.py`
- `run_fill_balance_fina_batches.sh`

机制说明：
- checkpoint 文件记录“已处理到的 symbol 索引”，用于断点续跑。
- 默认路径：`fill_balance_fina_daily.progress`

常见坑与建议：
- 旧 checkpoint 会覆盖 `--start-index`，导致从较后位置开始。
- 要全量重跑，用 `--reset-checkpoint` 或换一个 checkpoint 路径。
- 大文件写入容易超时，建议分段跑（如每 500 个 symbols）。
- 避免并发写 `daily.hdf`，否则容易损坏。

常用指令：
```bash
# 全量重跑（仅日期字段），清空旧进度
python fill_balance_fina_daily.py --only-meta --reset-checkpoint

# 分段跑（示例：0-500）
python fill_balance_fina_daily.py --only-meta --start-index 0 --end-index 500 \
  --checkpoint /tmp/fill_balance_fina_daily.meta.progress
```
