import numpy as np
import pandas as pd
"""
碳排放 = 活动数据 × 排放因子
活动数据 = 你用了多少电、多少煤、多少气 ← 你的模型预测的就是这个
排放因子 = 每单位活动数据对应多少碳（电网官方发布，量级约 0.55 kgCO₂/kWh）
"""
#1,创建rng
rng = np.random.default_rng(42)

#2,生成日期
# freq(频率)  为步长   有M, D, W, H, T, S, MS等，分别表示月，天，周，时，分，秒，毫秒等
# 从2024-01-01开始，90天
dates = pd.date_range("2024-01-01", periods=90, freq="D")

#3,生成values
# dayofweek 属性返回一个 Series，表示日期中的星期几，其中星期一为0，星期日为6
# astype(int)表示将布尔值转换为整数
weekend = (dates.dayofweek >= 5).astype(int)               # 周六日=1，工作日=0
#造用电量：平时约 100，周末低 8，再叠加均值 0、标准差 3 的随机噪声—— 这就是"数据"的样子：规律 + 噪声
values = 100 - 8 * weekend + rng.normal(0, 3, 90)          # 基准100，周末降8，再加随机波动

#4,创建DataFrame
df = pd.DataFrame({"date": dates, "electricity": np.round(values, 2)})

print(df.head())        # 表的前 5 行
print(df.shape)         # 表是几行几列

#5,保存DataFrame
df.to_csv("data/demo_tiny.csv", index=False)        # 存成 csv 文件
print("已保存 data/demo_tiny.csv")