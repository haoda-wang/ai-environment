"""
mse: 均方误差                 误差² 的平均
rmse: 均方根误差              误差² 的平均的平方根
mae: 平均绝对误差             误差绝对值平均
mape: 平均绝对百分比误差       误差/真值 的百分比

MSE 永远 ≥ MAE 的平方量级、RMSE 永远 ≥ MAE （平方会放大大误差）
mae≈0.45*rmse   #todo不会差太多
"""

import numpy as np
import pandas as pd

x_test = pd.read_csv( 'data/X_test.csv' )
#单列 DataFrame 压成 Series，不然后面减法形状对不上
# squeeze: DataFrame → Series
'''
test_y = pd.read_csv( 'data/y_test.csv' )
print(test_y.shape)      #(17, 1)
'''
y_test = pd.read_csv( 'data/y_test.csv' ).squeeze()    # squeeze: DataFrame → Series
#print(test_y.shape)      #(17,)

# 最笨基线：预测明天 = 今天（lag_1 就是"昨天的值"，即预测日的"今天"）
y_pred = x_test[ 'lag_1' ]

# todo sklearn 里都有，但 先手写一遍，你才真的知道它是什么 ；微步 8 换库函数，你对比结果会发现一模一样

def mse( y , p):return np.mean( (y - p) ** 2 )
def rmse( y , p):return np.sqrt( mse( y , p) )
def mae( y , p):return np.mean( np.abs( y - p) )
def mape( y , p):return np.mean( np.abs( y - p) / y )*100
if __name__ == '__main__':
      print(f"基线（明天=今天）  "
            f"MSE={mse(y_test, y_pred):.3f}  "
            f"RMSE={rmse(y_test, y_pred):.3f}  "
            f"MAE={mae(y_test, y_pred):.3f} "
            f"MAPE={mape(y_test, y_pred):.2f}%")
#基线（明天=今天）  MSE=22.194  RMSE=4.711  MAE=3.896  MAPE=4.02%
"""
真正的噪声信号在这里： MAE=3.90 ≈ 噪声标准差 3 。
意思是——基线的误差主要由噪声构成 （这是谁都预测不了的部分，叫 不可约误差 ）
剩下的差距（3.9 - 3 ≈ 0.9）才是基线没学到的东西： 周末 −8 的规律 。
"""