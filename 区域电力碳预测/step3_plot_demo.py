import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']    # 如果是Mac本，不支持SimHei的时候，可以修改为 'Microsoft YaHei' 或者 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False


#1, 读取数据
df=pd.read_csv('data/demo_tiny.csv',parse_dates=['date'])
#2, 绘制折线图
plt.figure(figsize=(10,4),dpi=80)
plt.plot(df['date'], df['electricity'], label='Electricity')
plt.title('90天用电量趋势图')
plt.xlabel('日期')
plt.ylabel('用电量')
plt.xticks(rotation= 30 ) # 日期斜着放，否则挤成一团
plt.tight_layout()        # 调整子图参数，使之填充整个图像区域
#3, 保存图像
plt.savefig('fig/electricity_trend.png')
plt.show()
print("图像保存完成")
