import pandas as pd


# todo parse_datas用于在读取 CSV 文件时自动将指定列解析为日期时间类型（datetime），而不是默认的字符串类型。
#df=pd.read_csv('data/demo_tiny.csv')
#print(df.dtypes)
"""
date               str                        
electricity    float64
dtype: object
"""
df=pd.read_csv('data/demo_tiny.csv',parse_dates=['date'])
#print(df.dtypes)
"""
date           datetime64[us]
electricity           float64
dtype: object
"""
df.describe()      # 统计描述(count / mean / std / min / 25% / 50% / 75% / max)
df.info()          # 每列的类型 / 非空数量 / 内存占用
print("--"*32)
print(df.isna().sum())  # 每列缺失值统计
"""
diff()相邻两行的差（后一行减前一行）。第一行没有前一行 → 是 NaN
dropna()删除包含 NaN 的行
unique()返回唯一值
"""
gap=df["date"].diff().dropna().unique()
print ( "相邻日期间隔：" , gap)