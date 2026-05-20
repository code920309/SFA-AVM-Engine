import pandas as pd
import numpy as np
import os
import glob
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# ==========================================
# 지표 보정 + 데이터 분석 + 모델 재학습 통합 파이프라인
# ==========================================

def run_advanced_pipeline():
    index_path = r'c:\Users\rkgka\OneDrive\바탕 화면\SFA-AVM-Engine\external\land_index_raw'
    data_path = 'national_single_house_cleaned.csv'
    
    # 1. 지수 데이터 통합 및 전처리
    print("Step 1: Processing Land Indices...")
    all_indices = []
    for file in glob.glob(os.path.join(index_path, "land_index_*.csv")):
        if 'utf8' in file: continue
        try:
            temp_idx = pd.read_csv(file, encoding='cp949')
            melted = temp_idx.melt(id_vars=['No', '지역', '지역.1'], var_name='ym', value_name='index_value')
            all_indices.append(melted)
        except: continue
        
    master_index = pd.concat(all_indices, ignore_index=True)
    
    def clean_ym(x):
        try:
            parts = x.split('년 ')
            year = parts[0]
            month = parts[1].replace('월', '').strip()
            return f"{year}{int(month):02d}"
        except: return x
    master_index['deal_ym'] = master_index['ym'].apply(clean_ym)
    
    # index_value 수치형 변환 (에러 발생 시 NaN 처리)
    master_index['index_value'] = pd.to_numeric(master_index['index_value'], errors='coerce')
    # 결측치(NaN) 제거
    master_index = master_index.dropna(subset=['index_value'])
    
    # 2. 거래 데이터에 매핑 및 시점 보정
    print("Step 2: Performing Time Correction...")
    df = pd.read_csv(data_path, encoding='cp949')
    df['deal_ym'] = df['dealYear'].astype(str) + df['dealMonth'].apply(lambda x: f"{x:02d}")
    
    sido_map = {
        '11': '서울', '26': '부산', '27': '대구', '28': '인천', '29': '광주',
        '30': '대전', '31': '울산', '36': '세종', '41': '경기', '42': '강원',
        '51': '강원', '43': '충북', '44': '충남', '45': '전북', '52': '전북',
        '46': '전남', '47': '경북', '48': '경남', '50': '제주'
    }
    df['sido_nm'] = df['sggCd'].astype(str).str[:2].map(sido_map)
    
    sido_index = master_index.groupby(['지역', 'deal_ym'])['index_value'].mean().reset_index()
    df = pd.merge(df, sido_index, left_on=['sido_nm', 'deal_ym'], right_on=['지역', 'deal_ym'], how='left')
    
    base_index = 100.0
    df['index_value'] = df['index_value'].fillna(base_index)
    df['adj_dealAmount'] = df['dealAmount'] * (base_index / df['index_value'])
    df['adj_price_per_m2'] = df['adj_dealAmount'] / df['plottageAr']
    df['target_price_log'] = np.log1p(df['adj_price_per_m2'])
    
    # 3. 상위 1% / 하위 1% 분석 (지수 보정 후 가격 기준)
    p01 = np.percentile(df['adj_price_per_m2'], 1)
    p99 = np.percentile(df['adj_price_per_m2'], 99)
    print("\n" + "="*50)
    print("--- [Price Distribution Analysis (Corrected)] ---")
    print(f"하위 1% 지점 가격: {p01:,.0f} 원/m2")
    print(f"상위 1% 지점 가격: {p99:,.0f} 원/m2")
    print(f"가격 격차 (P99/P01): {p99/p01:.2f}배")
    print("="*50 + "\n")
    
    # 4. 모델 재학습 (v2와 동일 피처셋 사용)
    print("Step 4: Re-training XGBoost with Corrected Data...")
    # 파생 피처 재생성 (Age, floor_area_ratio 등)
    df['Age'] = df['dealYear'] - df['buildYear']
    df['Age'] = df['Age'].apply(lambda x: max(x, 0))
    df['floor_area_ratio'] = df['totalFloorAr'] / (df['plottageAr'] + 1e-5)
    df['deal_month_sin'] = np.sin(2 * np.pi * df['dealMonth'] / 12)
    df['deal_month_cos'] = np.cos(2 * np.pi * df['dealMonth'] / 12)
    
    features = ['houseType', 'buildYear', 'plottageAr', 'totalFloorAr', 'dealYear', 
                'Age', 'floor_area_ratio', 'deal_month_sin', 'deal_month_cos', 'sggCd', 'umdNm']
    
    le = LabelEncoder()
    df['houseType'] = le.fit_transform(df['houseType'].astype(str))
    
    X = df[features]
    y = df['target_price_log']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Target Encoding
    for col in ['umdNm', 'sggCd']:
        mapping = y_train.groupby(X_train[col]).mean()
        global_mean = y_train.mean()
        X_train[col] = X_train[col].map(mapping).fillna(global_mean)
        X_val[col] = X_val[col].map(mapping).fillna(global_mean)

    model = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=8, 
        random_state=42, n_jobs=-1, objective='reg:squarederror',
        early_stopping_rounds=30
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=None)
    
    # 평가
    y_pred_log = model.predict(X_val)
    y_val_real = np.expm1(y_val)
    y_pred_real = np.expm1(y_pred_log)
    
    r2 = r2_score(y_val_real, y_pred_real)
    mae = mean_absolute_error(y_val_real, y_pred_real)
    
    print("="*50)
    print(f"--- [Final Adjusted Model Results] ---")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:,.2f} 원/m2")
    print("="*50)
    
    df.to_csv('national_single_house_time_corrected.csv', index=False, encoding='cp949')

if __name__ == "__main__":
    run_advanced_pipeline()
