import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import os

# ==========================================
# 순수 데이터 기반 고도화 모델 (노후 기조 테마 반영)
# ==========================================

def run_pure_data_pipeline(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # 1. 데이터 로드 (시점 보정이 완료된 데이터셋)
    df = pd.read_csv(file_path, encoding='cp949')
    print(f"Data Loaded: {len(df):,} rows")

    # [사전 단계] 아웃라이어 정제 (Trimming 1.5% ~ 98.5%)
    p_low = np.percentile(df['adj_price_per_m2'], 1.5)
    p_high = np.percentile(df['adj_price_per_m2'], 98.5)
    df = df[(df['adj_price_per_m2'] >= p_low) & (df['adj_price_per_m2'] <= p_high)].copy()
    print(f"- Trimming Complete: {len(df):,} rows remain.")

    # 2. 내부 데이터 기반 '노후 밀집도' 피처 엔지니어링
    # 건물 나이 계산 (기본 피처)
    df['Age'] = df['dealYear'] - df['buildYear']
    df['Age'] = df['Age'].apply(lambda x: max(x, 0))
    
    # 30년 이상 노후 주택 여부 (is_old_building)
    df['is_old_building'] = (df['Age'] >= 30).astype(int)
    
    # 법정동(umdNm)별 노후 주택 비율 (umd_old_building_ratio)
    umd_old_ratio = df.groupby('umdNm')['is_old_building'].mean().reset_index()
    umd_old_ratio.columns = ['umdNm', 'umd_old_building_ratio']
    df = pd.merge(df, umd_old_ratio, on='umdNm', how='left')
    
    # 미시 공간 블록(spatial_block_id)별 평균 건물 나이 (block_mean_age)
    # spatial_block_id 생성 (sggCd + umdNm + jibun)
    df['jibun'] = df['jibun'].fillna('').astype(str)
    df['spatial_block_id_str'] = (
        df['sggCd'].astype(str) + "_" + 
        df['umdNm'].astype(str) + "_" + 
        df['jibun']
    )
    
    block_age = df.groupby('spatial_block_id_str')['Age'].mean().reset_index()
    block_age.columns = ['spatial_block_id_str', 'block_mean_age']
    df = pd.merge(df, block_age, on='spatial_block_id_str', how='left')
    
    # Label Encoding for spatial_block_id
    df['spatial_block_id'] = LabelEncoder().fit_transform(df['spatial_block_id_str'])
    
    print("- Internal old-age features generated.")

    # 3. 모델 피처 구성 (최종 15개 변수 체제)
    features = [
        'houseType', 'buildYear', 'plottageAr', 'totalFloorAr', 'dealYear', 
        'Age', 'floor_area_ratio', 'deal_month_sin', 'deal_month_cos', 
        'sggCd', 'umdNm', 'spatial_block_id', 
        'is_old_building', 'umd_old_building_ratio', 'block_mean_age'
    ]
    target = 'target_price_log'
    
    # 범주형 인코딩 (houseType)
    if df['houseType'].dtype == object:
        df['houseType'] = LabelEncoder().fit_transform(df['houseType'].astype(str))
    
    X = df[features]
    y = df[target]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Target Encoding (umdNm, sggCd)
    for col in ['umdNm', 'sggCd']:
        if X_train[col].dtype == object:
            mapping = y_train.groupby(X_train[col]).mean()
            global_mean = y_train.mean()
            X_train[col] = X_train[col].map(mapping).fillna(global_mean)
            X_val[col] = X_val[col].map(mapping).fillna(global_mean)

    # 4. 최종 고도화 XGBoost 학습
    print("Training Final Pure-Data XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=12,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror',
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    # 5. 결과 리포트
    y_pred_log = model.predict(X_val)
    y_val_real = np.expm1(y_val)
    y_pred_real = np.expm1(y_pred_log)
    
    r2 = r2_score(y_val_real, y_pred_real)
    mae = mean_absolute_error(y_val_real, y_pred_real)
    mape = np.mean(np.abs((y_val_real - y_pred_real) / y_val_real)) * 100
    
    print("\n" + "="*50)
    print(f"--- [Final Pure-Data Model Performance] ---")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:,.2f} 원/m2")
    print(f"Final MAPE: {mape:.2f} %")
    print("="*50)

    # 피처 중요도 출력
    importances = pd.Series(model.feature_importances_, index=features)
    print("\n--- [Final Feature Importance (Top 10)] ---")
    print(importances.sort_values(ascending=False).head(10))

if __name__ == "__main__":
    target_data = "national_single_house_time_corrected.csv"
    run_pure_data_pipeline(target_data)
