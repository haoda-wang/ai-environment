# 按时间切训练/测试集
#一句话原则：时序数据，绝不随机切。

import pandas as pd
#1,读数据
df=pd.read_csv('data/features.csv',parse_dates=['date'])
print(df.head())
feature=['lag_1','lag_7','roll_mean_7'] # 就是列表存特征名
print(feature)
target='electricity'
#todo[]里放列表 = 一次取多列；放字符串 = 取单列。
# feature 是列表所以能取多列，target 只有一个就建议用字符串。
# 这样 train[target] 取出来的是 DataFrame（一张表），而不是 Series（一列）。
# 程序不会报错，to_csv 也能保存。但如果后面训练模型时 sklearn 期望 y 是一维 Series，可能会有麻烦。


#2,切训练/测试集
split=int(len(df)*0.8) # 训练集占80%
train=df.iloc[:split]      #`iloc` 按 位置 切片（用数字）。注意`df.loc` 是按 标签 （日期/索引名）
test=df.iloc[split:]

#x_train,x_test,y_train,y_test=train_test_split(feature,target,test_size=0.2,random_state=23)
x_train,x_test,y_train,y_test=train[feature],test[feature],train[target],test[target]

#3,打印训练/测试集信息
print ( "训练集：" , len (train), "行，" , train['date']. min ().date(), "~" , train[ 'date' ]. max ().date())
print ( "测试集：" , len (test), "行，" , test[ 'date' ]. min ().date(), "~" , test[ 'date' ]. max ().date())
"""
训练集： 66 行， 2024-01-08 ~ 2024-03-13
测试集： 17 行， 2024-03-14 ~ 2024-03-30
66+17=83
"""
#4,保存训练/测试集
x_train.to_csv( 'data/x_train.csv' , index= False )
y_train.to_csv( 'data/y_train.csv' , index= False )
x_test.to_csv( 'data/x_test.csv' , index= False )
y_test.to_csv( 'data/y_test.csv' , index= False )
