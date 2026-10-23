import numpy as np
import pandas as pd
from step1_read_data import load_raw_data, clean_reshape
from step2_feature import make_features
from step3_model import run_all
from step4_plot import plot_pred

FACTOR = 0.8333   # 火电排放因子 kgCO2/kWh

def calc_carbon(df, factor=FACTOR):
    """火电发电量(亿kWh) → 碳排放(万吨CO2)：1e8 kWh × kg/kWh ÷1e3 ÷1e4 = ×10"""
    df = df.copy()
    df['carbon'] = df['thermal'] * factor * 10
    return df

if __name__ == '__main__':
    df = clean_reshape(load_raw_data('data/月度数据.csv'))

    # 预测目标改为火电：预测火电 = 预测碳排（只差常数因子）
    x_train, x_test, y_train, y_test = make_features(df, target='thermal')
    metrics, preds = run_all(x_train, y_train, x_test, y_test, target='thermal')
    print(metrics)
    """  预测 thermal 的指标（参考）:
                  RMSE         MAE      MAPE
    基线       575.492857  476.786764  9.059187
    线性回归     276.082880  202.142222  3.918297
    随机森林     287.595547  214.964667  4.180310
    XGBoost  421.040854  316.871920  6.251165
    """

    # 图1：火电发电量预测（单位：亿千瓦时）
    plot_pred(y_test, preds)

    # 图2：碳排预测（火电预测值 × 因子 → 万吨CO2）
    y_test_c = y_test * FACTOR * 10
    preds_c = {k: np.asarray(v) * FACTOR * 10 for k, v in preds.items()}
    # 字典item:它返回 所有键值对
    #preds.items():遍历`{'基线': array, '线性回归': array, ...}` ，每次拿到一个模型名`k` 和它的预测数组`v`
    #{k: np.asarray(v) * FACTOR * 10 for k, v in preds.items()}: 火电预测值 × 因子 → 万吨CO2
    print(preds.items())
    plot_pred(y_test_c, preds_c, path='fig/carbon_pred.png',
              title='2025-01 ~ 2026-09 月度电力碳排放预测对比（万吨CO2）',
              ylabel='碳排放（万吨CO2）')

    # 历史碳核算（真值换算，不是预测）
    df = calc_carbon(df)
    # resample('YE'): 按年(Year End, 每年最后一天)分组降采样，求每年碳排总和
    # /1e4: 万吨 → 亿吨
    print(df['carbon'].resample('YE').sum() / 1e4)    # 亿吨/年
