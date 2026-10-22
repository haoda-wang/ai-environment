import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from step1_read_data import load_raw_data, clean_reshape
from step2_feature import make_features
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

def evaluate(y_true, y_pred, name='model'):
    """统一算 4 个指标：越高越差"""
    mse  = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)                                     # 注意：不是 mean_squared_error
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # ×100 才是百分比
    # print(f"{name:10s} RMSE {rmse:8.2f}  MAE {mae:8.2f}  MAPE {mape:6.2f}%")
    return {'name': name, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

def run_naive(x_test, y_test, target='total'):
    # 基线：下月 = 上月（lag_1 就在特征里，按 target 动态取列名）
    return evaluate(y_test, x_test[f'{target}_lag1'], '基线')

def run_linear ( x_train, y_train, x_test, y_test ):
    m = LinearRegression()
    m.fit(x_train, y_train)

    # print('线性回归 w:', np.round(m.coef_, 4), ' b:', round(m.intercept_, 2))
    """
    有其他特征时:
    线性回归的w: [ 504.93291757    0.88816731 -504.76615072 -504.69259847 -505.13830868-504.61950271 -505.1323349 ]
    线性回归的b: -30.909280414333807
    
    无其他特征时:
    线性回归 w: [0.1707 0.8528 6.4791]  b: 119.79
    """
    pred= m.predict(x_test)
    return evaluate(y_test, pred, '线性回归' ),pred

def run_rf ( x_train, y_train, x_test, y_test, tune= False ): 
    if tune:
        """
        交叉验证,网格搜索 
        1,创建空模型   e=
        2,准备超参    param_grid =
        3,网格搜索    grid =
        4,训练模型    grid.fit(x_train, y_train)
        5,调用最佳模型  best_e=grid.best_estimator_
        6,预测    
        """
        param = { 'n_estimators' : [100,200,300], 'max_depth' : [3,4,5]}
        e=RandomForestRegressor(random_state=42)
        grid = GridSearchCV(e,
                            param,
                            cv=TimeSeriesSplit(n_splits= 3),  #`TimeSeriesSplit` 强制“只用过去预测未来”
                            scoring= 'neg_root_mean_squared_error' )
        grid.fit(x_train, y_train)
        # print ( '最优参数:' , grid.best_params_)
        """
        有其他特征时:最优参数: {'max_depth': 5, 'n_estimators': 100}
        无其他特征时:最优参数: {'max_depth': 4, 'n_estimators': 100}
        """

        m = grid.best_estimator_
        pred= m.predict(x_test)
    else :
        m = RandomForestRegressor(random_state= 42 )
        m.fit(x_train, y_train)
        pred= m.predict(x_test)
    return evaluate(y_test, pred, '随机森林' ),pred

def run_xgb ( x_train, y_train, x_test, y_test ):
    m = XGBRegressor(n_estimators= 300 , learning_rate= 0.05 , random_state= 42 )
    m.fit(x_train, y_train) 
    pred= m.predict(x_test)
    return evaluate(y_test, pred, 'XGBoost' ),pred

def run_all(x_train, y_train, x_test, y_test, target='total'):
    """跑全部模型：返回 (指标表, 预测值字典)"""
    p0 = x_test[f'{target}_lag1']
    r0 = evaluate(y_test, p0, '基线')
    r1, p1 = run_linear(x_train, y_train, x_test, y_test)
    r2, p2 = run_rf(x_train, y_train, x_test, y_test, tune=True)
    r3, p3 = run_xgb(x_train, y_train, x_test, y_test)

    df = pd.DataFrame([r0, r1, r2, r3]).set_index('name')
    preds = {'基线': p0, '线性回归': p1, '随机森林': p2, 'XGBoost': p3}
    return df, preds

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = make_features(clean_reshape(load_raw_data('data/月度数据.csv')))
    # run_naive(y_train, y_test)
    df, preds = run_all(x_train, y_train, x_test, y_test)
    print(df)
    """
    有其他特征时:
    
        name       RMSE       MAE     MAPE
0       基线  703.620674  608.876675  7.509611
1     线性回归  239.350937  191.238311  2.435664
2     随机森林  383.227304  324.639725  3.88153
3  XGBoost  357.519951  285.496275  3.401669
    
    无其他特征时:
               RMSE         MAE      MAPE
name                                     
基线       703.620674  608.876675  7.509611
线性回归     234.075529  179.530322  2.240779
随机森林     311.963963  250.045675  2.976881
XGBoost  296.322292  254.315197  3.071143

    """