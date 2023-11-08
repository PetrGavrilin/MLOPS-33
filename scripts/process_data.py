#from tsai.all import *
import warnings
import pandas as pd
import datetime
import numpy as np

# функция сигментации данных
def to_segments(df, column, size = 24):  
    df.index.hour[0]
    start_idx = 24-df.index.hour[0]
    df = df.iloc[start_idx:]
    val = df[[column]].values
    return val[:size*(val.size//size)].reshape(-1,size)


df = pd.read_csv('/home/petr/project/datasets/data.csv', index_col='timestamp', parse_dates=True)

# Формируем датасет с почасовой статистикой
data = df.groupby(pd.Grouper(freq='1h')).sum()
data.drop(data.tail(1).index,inplace=True)
data.drop(data.head(1).index,inplace=True)

data['day_of_the_week'] = pd.to_datetime(data.index).weekday

working_week = to_segments(data[data['day_of_the_week'] <= 5], 'value', size = 24)
week_end = to_segments(data[data['day_of_the_week'] > 5], 'value', size = 24)

data_p = np.concatenate((
                    working_week, 
                    week_end
                   ))
                   
data_p = pd.DataFrame(data=data_p)

data_p['y'] = np.concatenate((
                    0*np.ones(working_week.shape[0]),
                    1*np.ones(week_end.shape[0]),
                   ))

data_p.to_csv('/home/petr/project/datasets/data_processed.csv') 
