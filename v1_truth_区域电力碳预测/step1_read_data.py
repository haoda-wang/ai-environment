import pandas as pd


# 1,加载原始数据
def load_raw_data(path):
    return pd.read_csv(path,skiprows=2)

if __name__ == '__main__':
    data = load_raw_data('data/月度数据.csv')
    print(data)
    print(data.head())
    print(data.shape)
    print(data.tail(2))




