import pandas as pd
from step1_read_data import load_raw_data, clean_reshape

def fill_month(df):
    """补 1、2 月当期值：有2月当期的直接相减得真值，无2月当期的按天数比例拆分合计"""
    df=df.copy()
    keys=['total', 'thermal', 'hydro', 'nuclear', 'wind', 'solar']

    for k in keys:
        cum_col=f'{k}_cum'                         # 累计值列名

        for t in df.index[df.index.month==2]:      # t为2月的日期   遍历2月
            cum=df.loc[t,cum_col]                  # 获取2月对应的累计值   loc[ 日期, 列名]
            # print(cum)

            if pd.isna(cum):                       # solar 早年没有累计值，跳过
                continue

            else:
                feb=df.loc[t,k]                    #取k : 当期值
                jan=t.replace(month=1)
                # 获取1月的日期   利用replace(月=1) 方法
                # print(jan)             # 打印1月的用电量 ,全是0
                if pd.isna(feb):
                    # 两个月的当期值都没有：按天数比例拆分合计
                    # python没有这种写法: d = t.year.is_leap_year() ? 29:28
                    d = 29 if t.is_leap_year else 28  # 判断是否是闰年   is_leap_year()方法
                    total_d = 31 + d

                    df.loc[jan, k] = cum * 31 / total_d
                    df.loc[t, k] = cum * d / total_d
                    df.loc[jan, cum_col] = cum * 31 / total_d  #补1月的累计值

                else:
                    # 2月当期有值（只有2015年）：1月 = 合计 − 2月，精确还原
                    df.loc[jan, k] = cum - feb
                    df.loc[jan, cum_col] = cum - feb           #同理,补1月的累计值
    return df

def make_features(df, start_date='2017-01-01', target='total'):
    """构造 lag 特征 + 设定建模窗口 + 切分训练/测试"""
    #todo 可以不用copy()
    #1, 补月当期值
    df=fill_month(df)
    df = df.loc[start_date:]       # 设定建模窗口

    #2, 构造 lag 特征
    x=pd.DataFrame(index=df.index)
    x[f'{target}_lag1'] = df[target].shift(1)  # 上月
    x[ f'{target}_lag12' ] = df[target].shift( 12 ) # 去年同月（月频的关键特征）
    """
    # 火/水/风/核/光也 lag1，辅助特征
    for k in [ 'thermal' , 'hydro' , 'nuclear' , 'wind' , 'solar' ]:
        x[f'{k}_lag1'] = df[k].shift(1)
    """

    #构造 lag 特征（只用 total，避免与 5 个分项完全共线）
    x['month']=df.index.month           # 日历：预测未来时唯一 100% 已知的信息
    y =df[target]  # 目标变量：当月 total

    #3, 切分训练/测试
    valid = x.dropna(subset=[f'{target}_lag12']).index  # 去除缺失值
    # print(valid) # 打印有效数据的索引
    x,y=x.loc[valid],y.loc[valid]

    train=int(len(x)*0.8)
    #                  train_test_split(X, y, shuffle= False ) # 等价于 iloc 切
    x_train,x_test,y_train,y_test=x.iloc[:train],x.iloc[train:],y.iloc[:train],y.iloc[train:]
    return x_train,x_test,y_train,y_test

if __name__ == '__main__':
    df = clean_reshape(load_raw_data('data/月度数据.csv'))
    # print("补之前缺失：\n", df.isna().sum())
    # df=fill_month(df)
    #ps:一定用df=fill_month(df) 而不是fill_month(df)来更新df
    # print("补之后缺失：\n", df.isna().sum())
    # 打印2015年1月到3月的合计和累计值
    # print("\n2015-01 ~ 2015-03：\n", df.loc['2015-01': '2015-03', ['total', 'total_cum']])
    """
    2015-01 ~ 2015-03：
              total  total_cum
2015-01-01  4907.1     4907.1
2015-02-01  3653.8     8560.9
2015-03-01  4510.6    13102.7
    """


    # print("一月:4907.1","2月的cum-当期值为:",8560.9-3653.8)  #一月:4907.1 2月的cum-当期值为: 4907.099999999999
    x_train, x_test, y_train, y_test = make_features(df)
    print(f"训练: {x_train.shape} 测试: {x_test.shape} ")                 #训练: (84, 7) 测试: (21, 7)
    # print(f"训练窗口: {x_train.index[0].date()} ~ {x_train.index[-1].date()} ") #训练窗口: 2018-01-01 ~ 2024-12-01
    # print(f"测试窗口: {x_test.index[0].date()} ~ {x_test.index[-1].date()} ")   #测试窗口: 2025-01-01 ~ 2026-09-01
    print(f"特征列: {x_train.columns.tolist()} ")
# 特征列: ['total_lag1', 'total_lag12', 'thermal_lag1', 'hydro_lag1', 'nuclear_lag1', 'wind_lag1', 'solar_lag1']
# 特征列: ['total_lag1', 'total_lag12', 'month']