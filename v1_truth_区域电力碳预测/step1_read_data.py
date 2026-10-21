import pandas as pd
# 1,加载原始数据
def load_raw_data(path):
    return pd.read_csv(path,skiprows=2)


# 2,数据清洗和重塑
def clean_reshape(df_raw):
    # 不动传入的原始数据——原始数据只读原则
    df= df_raw.copy()

    df=df.iloc[:12]                                                   # 只留前12行指标，切掉尾部2行注释
    # 文件中存在没有列名的空列，Pandas会自动将其命名为Unnamed: 0、Unnamed: 1等
    """
    df.columns.str.contains('Unnamed'):  对每个列名检查是否包含字符串 'Unnamed'，返回一个布尔 Series（如 [False, True, False, ...]）
    
    ~df.columns.str.contains('Unnamed')
    ~为取反操作符，等价于 df.columns.str.contains('Unnamed') == False
    """

    df=df.loc[:,df.columns.str.contains('Unnamed') == False]         # 删掉全空的Unnamed列
    df.columns=df.columns.str.strip()                                # 删掉列名中的空格,\t
    # print(type(df))  # 确认它是 DataFrame 还是 Index/Series
    df=df.set_index('指标').T                                         # 将"指标"列变成行索引      columns->>index
    # '2015年2月' → 2015-02：中年换横杠、月字删掉，to_datetime 就认得了
    df.index=pd.to_datetime(df.index.str.replace('年','-').str.replace('月',''))
    df=df.sort_index()                   #月份从旧到新排好
    """
    df.apply 对 DataFrame 的每一列分别应用括号里的函数
    errors='coerce'	关键参数：遇到无法转换的值时，不报错，而是把它变成 NaN
    """

    df = df.apply(pd.to_numeric, errors='coerce')  # 带\t的字符串→数字，转不动的变NaN
    """
     total: 总发电量
     thermal: 热力发电量
     hydro: 水力发电量
     nuclear: 核能发电量
     wind: 风力发电量
     solar: 太阳能发电量
     _cum: _累计发电量
     """
    df.columns = [ 'total' , 'total_cum' , 'thermal' , 'thermal_cum' , 'hydro' , 'hydro_cum' , 'nuclear' , 'nuclear_cum' , 'wind' , 'wind_cum' , 'solar' , 'solar_cum' ]
    return df
#
def check_cum(df):
    """
    交叉验证：累计值逐月差值 应  ≈ 当期值
    df['total'] → 单月值（每月发电量）
    df['total_cum'] → 累计值（年初到当月的总和）
    理论上：累计值的相邻差 = 单月值
    """
    print("累计值逐月差值与当期值的偏差：")
    """
           df[f'{k}_cum']: 累计值
           df[f'{k}_cum'].diff(): 累计值的相邻差
           df[k]: 当期值
           (df[f'{k}_cum'].diff() - df[k]).abs(): 相邻差与当期值的绝对偏差
           """
    for k in ['total', 'thermal', 'hydro', 'nuclear', 'wind', 'solar']:
        gap = (df[f'{k}_cum'].diff() - df[k]).abs()
        print(f" {k:8s} 平均偏差 {gap.mean():7.2f} 最大 {gap.max():7.2f} ")






if __name__ == '__main__':
    df = load_raw_data('data/月度数据.csv')
    # print(data)
    """
                                                    指标\t  ... Unnamed: 142
0                                   发电量当期值 (亿千瓦时) \t  ...          NaN
1                                 火力发电量当期值 (亿千瓦时) \t  ...          NaN
2                                 水力发电量当期值 (亿千瓦时) \t  ...          NaN
3                                 核能发电量当期值 (亿千瓦时) \t  ...          NaN
4                                 风力发电量当期值 (亿千瓦时) \t  ...          NaN
5                                太阳能发电量当期值 (亿千瓦时) \t  ...          NaN
6  注：1. 主要能源产品产量月度统计范围为规模以上工业法人单位，即年主营业务收入2000万元及...  ...          NaN
7                                         数据来源：国家统计局  ...          NaN

    """
    # print(data.head())
    print("数据清洗前->",df.shape) #数据清洗前 (14, 143)
    # print(data.tail(2))
    # data.info()
    df=clean_reshape(df)
    print(df.head())
    print("数据清洗后->",df.shape) #数据清洗后 (141, 12)
    # print(df.index.is_monotonic_increasing)  # 时间必须递增 → True
    # print(df.dtypes)  # 6列应该全是 float64
    df.info()
    # df.to_csv('data/cleaned_data.csv')
    check_cum(df)
    gap = (df['total_cum'].diff() - df['total']).abs() # 计算累计值逐月差值与当期值的绝对偏差
    print("偏差最大的 5 个月: ",gap.nlargest(5))  # 偏差最大的 5 个月