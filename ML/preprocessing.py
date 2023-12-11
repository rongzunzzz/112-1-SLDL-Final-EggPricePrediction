import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def readData(years, location):
    dfs = []
    for year in years:
        df = pd.read_csv(f'../concat-data/{location}/{year}.csv')
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True) # all data from 2018 to 2022

    return df


def clean(df, weekly=False):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    df = df.drop(columns=['StationName'])

    # 將所有資料轉成數字，不是數字的資料會變成 NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # 填補缺失值
    df['ImportChickenQuantity'].fillna(0, inplace=True)
    df['Precipitation'].fillna(0, inplace=True)

    # 移除含有缺失值的row
    df = df.dropna()

    if weekly:
        df = df.resample('W').mean()

    # rename columns
    new_columns = {
        'egg_TaijinPrice':"EggPrice", 
        'Duckegg_TNN_TaijinPrice': "DuckEggPrice", 
        '入中雞雛數':"NumRuChung",
        '產蛋隻數':"NumOfChickLayingEggs",
        '均  日     產蛋箱數':"BoxesOfEggsPerDay", 
        '淘汰隻數':"NumDisuse", 
        '目  前         換羽隻數':"NumMoulting", 
        'WhiteChickQuantity':"WhiteChickQuantity",
        'RedChickQuantity':"RedChickQuantity", 
        'ImportChickenQuantity':"ImportChickenQuantity", 
        ' 玉米粒':"Corn", 
        ' 黃豆粉 ':"SoyBeanFlour", 
        '玉米粉 ':"CornFlour",
        '脫殼豆粉':"DehulledBeanFlour", 
        '高蛋白豆粉':"HighProteinBeanFlour", 
        'AirTemperatureMean':"AirTemperatureMean", 
        'AirTemperatureMax':"AirTemperatureMax",
        'Precipitation':"Precipitation"
    }
    df = df.rename(columns=new_columns)

    return df


def split_and_scale(df, year_to_predict):
    from sklearn.preprocessing import StandardScaler

    # Splitting into two DataFrames, train and test
    # df_other_years is train and df_to_predict is test
    df_to_predict = df.loc[f'{year_to_predict}-01-01':f'{year_to_predict}-12-31']

    if year_to_predict == 2018:
        df_other_years = df.loc['2019-01-01':'2022-12-31']
    elif year_to_predict == 2022:
        df_other_years = df.loc['2018-01-01':'2021-12-31']
    else:
        df_other_years = pd.concat(
            [df.loc[f'2018-01-01':f'{year_to_predict-1}-12-31'], 
            df.loc[f'{year_to_predict+1}-01-01':f'2022-12-31']], axis=0)
        

    # df_other_years is train and df_to_predict is test
    X_train = df_other_years.drop('EggPrice', axis=1)
    y_train = df_other_years['EggPrice']
    X_test = df_to_predict.drop('EggPrice', axis=1)
    y_test = df_to_predict['EggPrice']

    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def feature_selection(df, type='corr'):
    if type == 'corr':
        featuresCorr = df.corr()
        plt.figure(figsize=(60,50))
        sns.heatmap(featuresCorr, annot=True, cmap=plt.cm.Blues, vmax=1, vmin=-1)

        targetCorr = featuresCorr['EggPrice']
        targetCorr = targetCorr.drop('EggPrice')

        targetCorr_abs = abs(targetCorr)
        selectedFeatures = targetCorr_abs[targetCorr_abs > 0.4]
        print(f"Number of selected features: {len(selectedFeatures)} \n\nHighly relative feature list:\n{selectedFeatures}")

        df = df[list(selectedFeatures.keys()) + ['EggPrice']]

        return df, selectedFeatures, targetCorr

    return df
