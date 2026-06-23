import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler # 改用 RobustScaler

def get_saif_time_features(df):
    df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
    
    # 提取小時與週幾
    hour = df['SETTLEMENTDATE'].dt.hour
    day = df['SETTLEMENTDATE'].dt.dayofweek
    
    # 正餘弦編碼：將週期性資訊轉為連續空間
    # 小時週期 (24小時)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    # 星期週期 (7天)
    day_sin = np.sin(2 * np.pi * day / 7)
    day_cos = np.cos(2 * np.pi * day / 7)
    
    # 合併特徵 (這裡會變成 4 個特徵，對應模型的 enc_in 也要記得修改)
    return np.stack([hour_sin, hour_cos, day_sin, day_cos], axis=1)

class SAIFScaler:
    def __init__(self):
        self.scaler = RobustScaler() 
        
    def fit_transform(self, df):
        # 使用 arcsinh 代替 log1p，徹底解決 nan 問題
        df['RRP_trans'] = np.arcsinh(df['RRP']) 
        
        # 修正 fillna 警告：使用 bfill() 代替 fillna(method='bfill')[cite: 15]
        df['RRP_lag24'] = df['RRP_trans'].shift(24).bfill()
        df['RRP_ma12'] = df['RRP_trans'].rolling(window=12).mean().bfill()
        
        # 需求量也建議用 arcsinh 處理[cite: 15]
        df['Demand_trans'] = np.arcsinh(df['TOTALDEMAND'])
        
        features = df[['RRP_trans', 'Demand_trans', 'RRP_lag24', 'RRP_ma12']].values
        return self.scaler.fit_transform(features)

    def inverse_rrp(self, scaled_rrp):
        dummy = np.zeros((len(scaled_rrp), 4))
        dummy[:, 0] = scaled_rrp
        inv = self.scaler.inverse_transform(dummy)[:, 0]
        # 使用 sinh 還原 arcsinh[cite: 15]
        return np.sinh(inv)