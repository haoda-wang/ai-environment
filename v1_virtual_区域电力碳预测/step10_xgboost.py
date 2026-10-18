"""
RF 和 XGBoost 都是“一群决策树投票”.
区别是:
RF 的树互相独立、并行投票 （bagging）；
XGBoost 的树排成一队，每棵新树专门纠正前面所有树留下的错误 （boosting）。
后者通常更强，也更容易过拟合。
"""
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import numpy as np
#作用等价于前面step7中的函数
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from step9_forest import y_pred1                      # e1 是前面 step9 中训练好的默认随机森林模型
from step8_linear import y_pred
x_train = pd.read_csv( 'data/x_train.csv' )
y_train = pd.read_csv( 'data/y_train.csv' ).squeeze()
x_test = pd.read_csv( 'data/x_test.csv' )
y_test = pd.read_csv( 'data/y_test.csv' ).squeeze()

"""
learning_rate 学习率 默认0.3,每棵树纠错时“改多少”。 小步慢走 （0.1）比大步猛改稳；这是 XGBoost 最重要的旋钮
max_depth 树深度 默认6，调参时可调小 3-4； 建议压低 ——数据就 90 天，树太深必过拟合
"""
e=xgb.XGBRegressor(n_estimators= 100 , max_depth= 3 , learning_rate= 0.5 ,random_state=42)

e.fit(x_train, y_train)
y_pred_xg=e.predict(x_test)
if __name__ == '__main__':
    print(f"XGBoost  学习率为{e.learning_rate} "
          f" MSE= {mse(y_test, y_pred_xg):.3f}"
          f" RMSE= {np.sqrt(mse(y_test, y_pred_xg)):.3f} "
          f"MAE= {mae(y_test, y_pred_xg):.3f} "
          f"MAPE= {mape(y_test, y_pred_xg)*100:.2f} %")  #sklearn 的 MAPE 返回小数需 ×100

if __name__ == '__main__':
    print ( f"默认随机森林  MSE= {mse(y_test, y_pred1):.3f}"
            f" RMSE= {np.sqrt(mse(y_test, y_pred1)):.3f} "
            f"MAE= {mae(y_test, y_pred1):.3f} "
            f"MAPE= {mape(y_test, y_pred1)*100:.2f} %" )  #sklearn 的 MAPE 返回小数需 ×100

#数据可视化
# datas = pd.read_csv('data/features.csv',parse_dates=['date'])
# datas=datas['date'].iloc[-len(y_test):]
# plt.figure(figsize=(10,8),dpi=80)
#
# plt.plot(datas,y_test, label='Actual',linestyle='--')
# plt.plot(datas,y_pred1, label='RF_P')
# plt.plot(datas,y_pred_xg, label='XGBoost_P')
# plt.plot(datas,y_pred, label='Linear_P')
# plt.legend()
# plt.xlabel('Date')
# plt.xticks(rotation=45)
# plt.ylabel('Power')
# plt.title('XGBoost vs RF vs Linear ')
# plt.grid()
# plt.savefig('fig/xgboost_vs_rf_vs_linear.png', dpi=80)
# plt.tight_layout()
# plt.show()
# print("Plot saved as 'fig/xgboost_vs_rf_vs_linear.png'")
"""
总学习量” ≈ 学习率 × 树的棵数
lr=0.01 × 100棵  小步×100 次，总步幅很小   学不够（欠拟合） 
lr=0.1 × 100棵   中速                   通常的甜点区 
lr=0.5 × 100棵   大步猛改                改过头（过拟合）


正确玩法是 lr 减半、树数翻倍  eg:（0.1 × 100）-> (0.05 × 200)
参数永远是 成组 调的
"""




























