import pandas as pd
from step1_read_data import load_raw_data, clean_reshape
from step2_feature import make_features
from step3_model import run_all
from step4_plot import plot_pred

def calc_carbon(df, factor=0.8333):
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
    plot_pred(y_test, preds)

    # 历史碳核算（真值换算，不是预测）
    df = calc_carbon(df)
    print(df['carbon'].resample('YE').sum() / 1e4)    # 亿吨/年