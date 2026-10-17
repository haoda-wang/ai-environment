# V1 · 传统机器学习闭环

## 目标（验收线）

> 一行命令跑通「数据 → 清洗/特征 → EDA → 3 基线模型 → 评估对比」，产出：
> 1. 可复现的代码管线
> 2. 4 张 EDA 图
> 3. 3 个基线模型的指标对比表（RMSE / MAE / MAPE / R²）
> 4. 最优模型的预测曲线图

**达到这条线即算 V1 完成，不追加。**

## 设定（默认值，可改）

- 区域：沈阳 / 辽宁
- 预测目标：工业用电量
- 切分：末段 20% 作测试集（时序必须按时间顺序切分，不能随机切）

统一在 [config.py](config.py) 里改。

## 目录

- `src/make_demo_data.py` — 生成演示数据集（合成数据，仅用于跑通管线）
- `src/preprocess.py` — 清洗 + 特征工程（时间特征 / 滞后 / 滚动统计）
- `src/eda.py` — 探索性分析出图
- `src/train.py` — 3 个基线模型 + 网格搜索 + 评估 + 对比图
- `run_all.py` — 一键跑通全流程

## 怎么跑

```bash
python v1/run_all.py
```

输出：

- `data/processed/features.csv` — 特征表
- `outputs/metrics.csv` / `metrics.json` — 指标对比
- `outputs/figures/` — 6 张图（4 张 EDA + RMSE 对比 + 预测曲线）
- `outputs/models/` — 训练好的模型（.pkl）

## 换成真实数据（下一步）

1. 把真实数据放到 `data/raw/`，需包含 `date` 列和目标列（默认 `electricity`）
2. 改 [config.py](config.py) 的 `DATA_FILE`、`REGION`、`TARGET`
3. 重跑 `python v1/run_all.py`

**数据源待验证**：沈阳市 / 辽宁省统计年鉴（工业用电量）、中国碳核算数据库（CEADs）、国家统计局。

> ⚠️ 当前 `make_demo_data.py` 生成的是**合成数据**，只用于验证管线是否打通，
> 任何指标结论在换成真实数据前都不成立。

## 待办

- [ ] 验证真实数据源（频率 / 粒度 / 可获取性）
- [ ] 用真实数据替换演示数据，重跑并记录指标
- [ ] 补多步预测（1 / 6 / 24 步）与数据字典
- [ ] 整理 V1 技术报告
