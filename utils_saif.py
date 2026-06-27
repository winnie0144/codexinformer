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
        df['RRP_trans'] = np.arcsinh(df['RRP']) 
        df['Demand_trans'] = np.arcsinh(df['TOTALDEMAND'])
        
        # 原有特徵
        df['RRP_lag24'] = df['RRP_trans'].shift(24).bfill()
        df['RRP_ma12'] = df['RRP_trans'].rolling(window=12).mean().bfill()
        
        # --- 【新增 1】RRP_lag168 (歷史同日同時段錨點，168小時 = 7天) ---
        df['RRP_lag168'] = df['RRP_trans'].shift(168).bfill()
        
        # --- 【新增 2】RRP_std24 (近期波動率指標，捕捉市場混亂度) ---
        df['RRP_std24'] = df['RRP_trans'].rolling(window=24).std().bfill()
        
        # 將新特徵加入 features 陣列 (現在總共 6 個特徵)
        features = df[['RRP_trans', 'Demand_trans', 'RRP_lag24', 'RRP_ma12', 'RRP_lag168', 'RRP_std24']].values
        return self.scaler.fit_transform(features)

    def inverse_rrp(self, scaled_rrp):
        # --- 【修改】因為特徵數從 4 變成 6，用來反轉縮放的空矩陣寬度要改成 6 ---
        dummy = np.zeros((len(scaled_rrp), 6))
        dummy[:, 0] = scaled_rrp
        inv = self.scaler.inverse_transform(dummy)[:, 0]
        return np.sinh(inv)