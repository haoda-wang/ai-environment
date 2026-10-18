import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from step7_baseline import rmse, mae, mape,mse
from step8_linear import y_pred
from sklearn.model_selection import GridSearchCV

plt.rcParams[ 'font.sans-serif' ] = [ 'SimHei' ]
plt.rcParams[ 'axes.unicode_minus' ] = False

x_train = pd.read_csv( 'data/x_train.csv' )
y_train = pd.read_csv( 'data/y_train.csv' ).squeeze()
x_test = pd.read_csv( 'data/x_test.csv' )
y_test = pd.read_csv( 'data/y_test.csv' ).squeeze()

#线性模型
#调用线性模型 step8_linear.py
#打印线性模型的预测结果
if __name__ == '__main__':
    print ( f"线性模型  MSE= {mse(y_test, y_pred):.3f}"
            f" RMSE= {rmse(y_test, y_pred):.3f} "
            f"MAE= {mae(y_test, y_pred):.3f} "
            f"MAPE= {mape(y_test, y_pred):.2f} %" )

# 默认的随机森林模型
e1=RandomForestRegressor(random_state=42)   #默认  n_estimators=100, max_depth=None
e1.fit(x_train,y_train)
y_pred1=e1.predict(x_test)
if __name__ == '__main__':
    print ( f"默认随机森林  MSE= {mse(y_test, y_pred1):.3f}"
            f" RMSE= {rmse(y_test, y_pred1):.3f} "
            f"MAE= {mae(y_test, y_pred1):.3f} "
            f"MAPE= {mape(y_test, y_pred1):.2f} %" )
    # for name, imp in zip (x_train.columns, e1.feature_importances_): print ( f"{name:>12}:{imp:.3f}" )
    """
    lag_1:         0.110
    lag_7:         0.738
    roll_mean_7:   0.151
    """
#todo 随机森林加(网格搜索,交叉验证)

"""
交叉验证,网格搜索 
1,创建空模型   e=
2,准备超参    param_grid =
3,网格搜索    grid =
4,训练模型    grid.fit(x_train, y_train)
5,调用最佳模型  best_e=grid.best_estimator_
6,预测    
"""

#准备参数
e2=RandomForestRegressor(random_state=42)
num_e=[30,70,110,150]
max_d=[5,10]
param_grid = {
    'n_estimators': num_e,
    'max_depth': max_d
}
#网格搜索
grid = GridSearchCV(e2, param_grid=param_grid, cv=4)
grid.fit(x_train, y_train)
if __name__ == '__main__':
    print("最佳参数为:", grid.best_params_)
# 最佳模型
best_e=grid.best_estimator_
y_pred2=best_e.predict(x_test)
if __name__ == '__main__':
    print ( f"网格搜索随机森林  MSE= {mse(y_test, y_pred2):.3f}"
            f" RMSE= {rmse(y_test, y_pred2):.3f} "
            f"MAE= {mae(y_test, y_pred2):.3f} "
            f"MAPE= {mape(y_test, y_pred2):.2f} %" )

    #可视化
    # datas=pd.read_csv('data/features.csv',parse_dates=['date'])
    # datas=datas['date'].iloc[-len(y_test):]     #从features.csv中取长度为len(y_test)行作为x轴(日期),因为测试在最后,所以有 '-'
    # print(f"日期的长度:{len(datas)},测试集长度:{len(y_test)}")  # 打印日期的长度和y_test的长度
    # plt.figure(figsize=(10,8),dpi=80)
    # plt.plot(datas, y_test, label='真实值')
    # plt.plot(datas, y_pred, label='线性回归预测值')
    # plt.plot(datas, y_pred1, label='随机森林预测值')
    # plt.plot(datas, y_pred2, label='随机森林+网格搜索预测值')
    # plt.title( 'linear vs rf vs (rf+grid)' )
    # plt.xlabel('时间')
    # plt.ylabel('电力需求')
    # plt.legend()
    # plt.xticks(rotation= 30 )
    # plt.tight_layout()
    # plt.savefig('fig/linear_vs_rf_vs_rf_grid.png',dpi=80)
    # plt.show()
    # print("可视化结果保存在 fig/linear_vs_rf_vs_rf_grid.png")
"""
RF 把 70.9% 的宝押在`lag_7`
线性回归的最大权重也是`lag_7`
两个完全不同原理的模型，独立达成了同一个判断：“今天最像上周同一天”
当一个结论被两种方法交叉验证，它就可信了——这叫稳健性
"""
