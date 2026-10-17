#滞后特征——用昨天预测今天
import pandas as pd

df = pd.read_csv( 'data/demo_tiny.csv' , parse_dates=[ 'date' ])
# shift(1)：整列往下挪 1 格 → 今天这行看到的"lag_1"就是昨天的值
df['lag_1'] = df['electricity'].shift(1)
# print(df.head(5))
df['lag_7'] = df['electricity'].shift(7)
# print(df.head(8))


#todo 滚动均值：过去 7 天的平均水平（注意先 shift 再 rolling，防未来泄漏）
#"上帝视角"这个说法就是业内说的 data leakage（信息泄漏） ，
#shift(1) 的作用是将数据向上移动一行，即今天这行看到的"lag_1"就是昨天的值,"避免未来信息泄露"
#rolling(7) 的作用是创建一个窗口大小为 7 的滑动窗口（Moving Window）
df['roll_mean_7'] = df['electricity'].shift( 1 ).rolling( 7 ).mean()
print(df.head(12))
"""
      date  electricity   lag_1   lag_7  roll_mean_7
0  2024-01-01       100.91     NaN     NaN          NaN
1  2024-01-02        96.88  100.91     NaN          NaN
2  2024-01-03       102.25   96.88     NaN          NaN
3  2024-01-04       102.82  102.25     NaN          NaN
4  2024-01-05        94.15  102.82     NaN          NaN
5  2024-01-06        88.09   94.15     NaN          NaN
6  2024-01-07        92.38   88.09     NaN          NaN
7  2024-01-08        99.05   92.38  100.91    (96.782857)   <--- (100.91+96.88+102.25+102.82+94.15+88.09+92.38)/7
8  2024-01-09        99.95   99.05   96.88    96.517143
9  2024-01-10        97.44   99.95  102.25    96.955714
10 2024-01-11       102.64   97.44  102.82    96.268571
11 2024-01-12       102.33  102.64   94.15    96.242857
"""

# 前 7 行因为"没有更早的数据"产生 NaN，必须丢掉
#reset_index()重置 DataFrame 的行索引，使其从 0 开始连续递增，并丢弃旧的索引。
# print("未重置index的值:",df['roll_mean_7'][0])  #non
df_model = df.dropna().reset_index(drop= True )
# print(df['roll_mean_7'][7])
#重置后的df_model，roll_7[0]的值=未重置的df['roll_mean_7'][7]
print("重置后的df_model，roll_7[0]的值:",df_model['roll_mean_7'][0])
print ( "可用建模样本：" , len (df_model), "行" )#90-7= 83
df_model.to_csv('data/features.csv', index=False)
