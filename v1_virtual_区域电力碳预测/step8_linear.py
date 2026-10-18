import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from step7_baseline import mse, rmse, mae, mape         #import 一个 py 文件时，会从头到尾执行它一遍
#1,读训练集 测试集
x_train = pd.read_csv( 'data/x_train.csv' )
y_train = pd.read_csv( 'data/y_train.csv' ).squeeze()
x_test = pd.read_csv( 'data/x_test.csv' )
y_test = pd.read_csv( 'data/y_test.csv' ).squeeze()

#2,构建线性模型
e=LinearRegression()                    #线性回归 = 学一条`y = w₁·lag_1 + w₂·lag_7 + w₃·roll_mean_7 + b` 的直线
#3,训练模型
e.fit(x_train,y_train)                  #它内部做的事：找到一组 w、b，让训练集上 MSE 最小
#4,预测
y_pred=e.predict(x_test)
#5,评估模型
if __name__ == '__main__':
    print ( f"线性回归  MSE= {mse(y_test, y_pred):.3f}"
        f" RMSE= {rmse(y_test, y_pred):.3f} "
        f"MAE= {mae(y_test, y_pred):.3f} "
        f"MAPE= {mape(y_test, y_pred):.2f} %" )

#最震撼的一个数：MAE=2.148

#6, 打印每个特征的权重w + 截距b
#model.coef_->w
#model.intercept_->b
if __name__ == '__main__':
    print(e.coef_)
    print(e.intercept_)

#用for循环打印每个特征的权重w和截距b
if __name__ == '__main__':
    for name, w in zip (x_train.columns, e.coef_):
        print ( f" {name:>12} : {w:+.3f} " )
        print ( f" { '截距':>12} : {e.intercept_:+.3f} " )
#3个指标->3个w     3个b但是相同


