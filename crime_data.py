from datetime import datetime
import pandas as pd
import numpy as np


def read_data():
    file_path = '/mnt/scratch_dump/samples/final.csv'
    cols_wanted = ['data_hora_inclusao', 'natureza_descricao', 'numero_latitude', 'numero_longitude']
    df = pd.read_csv(file_path, usecols=cols_wanted)
    df=df.dropna()
    return df


def string2binary(strings, T=24):
    '''
    strings: list of strings formatted like ['2018-11-01 00:00:00', '2018-11-01 01:00:00']
    return: list of strings formatted as binary
    '''
    binary_timestamps = []

    for string in strings:
        ts = datetime.strptime(string, '%Y-%m-%d %H:%M:%S')
        year, month, day = ts.year, ts.month, ts.day
        slot = int((ts.hour * 60 + ts.minute) / (60 * (24 / T))) + 1
        binary_timestamp = f'{year:04d}{month:02d}{day:02d}{slot:02d}'.encode('utf-8')
        binary_timestamps.append(binary_timestamp)

    return binary_timestamps

def remove_outliers(df):
    coluna_desejada = 'numero_latitude'

    limite_inferior = df[coluna_desejada].quantile(0.003)
    limite_superior = df[coluna_desejada].quantile(0.997)

    # Filtrar o dataframe para manter apenas os valores dentro do intervalo desejado
    df = df[(df[coluna_desejada] >= limite_inferior) & (df[coluna_desejada] <= limite_superior)]
    coluna_desejada = 'numero_longitude'

    limite_inferior = df[coluna_desejada].quantile(0.003)
    limite_superior = df[coluna_desejada].quantile(0.997)

    # Filtrar o dataframe para manter apenas os valores dentro do intervalo desejado
    df = df[(df[coluna_desejada] >= limite_inferior) & (df[coluna_desejada] <= limite_superior)]
    return df

def create_bins(df):
    bins = pd.cut(df['numero_latitude'], bins=16, include_lowest=True, labels=False)
    df['Latitude'] = bins
    
    bins_lon = pd.cut(df['numero_longitude'], bins=8, include_lowest=True, labels=False)
    df['Longitude'] = bins_lon

    return df


def section_time(df):
    df['data_hora_inclusao'] = pd.to_datetime(df['data_hora_inclusao'])

    # Agrupe os dados em intervalos de 30 minutos
    df['data_hora_inclusao'] = df['data_hora_inclusao'].dt.floor('60T')

    df['data_hora_inclusao'] = df['data_hora_inclusao'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df


def get_arrays_and_timestamps(df):
    grupos = df.groupby('data_hora_inclusao')
    group_identifiers = grupos.groups

    # Criar uma lista de dataframes ordenados pelos valores únicos
    lista_dataframes = [grupo.copy() for _, grupo in grupos]
    timestamps=string2binary(list(group_identifiers.keys()))
    timestamps=np.array(timestamps)

    crimes_by_timestamp=[]
    array_shape = (16, 8)

    # Create an array filled with zeros
    for temp_df in lista_dataframes:
        crime_array, x_edges, y_edges = np.histogram2d(
            temp_df['Latitude'], 
            temp_df['Longitude'], 
            bins=[array_shape[0], array_shape[1]], 
            range=[[0, 16], [0, 8]],
            density=False
        )
        crime_array = crime_array.astype(int)
        crimes_by_timestamp.append([crime_array, crime_array])
    data=np.array(crimes_by_timestamp)
    return data, timestamps


def load_crime():
    df=read_data()
    df=remove_outliers(df)
    df=create_bins(df)
    df=section_time(df)
    data, timestamps = get_arrays_and_timestamps(df)
    return data, timestamps