import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams[ 'font.sans-serif' ] = [ 'SimHei' ]
plt.rcParams[ 'axes.unicode_minus' ] = False

df=pd.read_csv('data/demo_tiny.csv',parse_dates=['date'])

df_dirty=df.copy()
df_dirty.loc[[22,34,54,66,79],'electricity']=None

print ( "缺失数量：" , df_dirty[ 'electricity' ].isna().sum ())
df_clean=df_dirty.copy()
#方式1:   删除包含缺失值的行
df_1=df_clean.dropna()  # 删除包含缺失值的行
#方式2:   填充
df_2=df_clean.fillna(df_clean['electricity'].mean())  #用均值填充

#对比图
plt.figure(figsize=( 20, 8 ),dpi=80)
plt.plot(df_dirty[ 'date' ], df_dirty[ 'electricity' ], label='随机NON数据')
plt.plot(df_1[ 'date' ], df_1[ 'electricity' ], label='删除缺失值处理')
plt.plot(df_2[ 'date' ], df_2[ 'electricity' ], label='填充缺失值处理')
plt.title('数据处理对比图')
plt.xlabel('日期')
plt.ylabel('用电量')
plt.legend()
plt.tight_layout()
plt.savefig('fig/data_processing_comparison.png')
plt.show()

