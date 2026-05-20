import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import os

# ==========================================
# 고도화된 XGBoost 부동산 가치 산정 모델 (MAPE 최적화)
# ==========================================

def run_optimized_pipeline(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # 1. 데이터 로드
    df = pd.read_csv(file_path, encoding='cp949')
    initial_count = len(df)
    
    # 2. 아웃라이어 제거 (Trimming 1.5% ~ 98.5%)
    p_low = np.percentile(df['adj_price_per_m2'], 1.5)
    p_high = np.percentile(df['adj_price_per_m2'], 98.5)
    
    df_trimmed = df[(df['adj_price_per_m2'] >= p_low) & (df['adj_price_per_m2'] <= p_high)].copy()
    print(f"Trimming Complete: {initial_count:,} -> {len(df_trimmed):,} rows (Removed 3%)")
    
    # 3. 미시 공간 블록 피처 생성 (spatial_block_id)
    # 지번정보(jibun)가 결측치인 경우 빈 문자열로 처리
    df_trimmed['jibun'] = df_trimmed['jibun'].fillna('').astype(str)
    df_trimmed['spatial_block_id'] = (
        df_trimmed['sggCd'].astype(str) + "_" + 
        df_trimmed['umdNm'].astype(str) + "_" + 
        df_trimmed['jibun']
    )
    
    # Label Encoding
    le_block = LabelEncoder()
    df_trimmed['spatial_block_id'] = le_block.fit_transform(df_trimmed['spatial_block_id'])
    
    # 기존 범주형 전처리 (houseType)
    le_house = LabelEncoder()
    df_trimmed['houseType'] = le_house.fit_transform(df_trimmed['houseType'].astype(str))
    
    # 4. 파생 피처 (기존 유지)
    df_trimmed['Age'] = df_trimmed['dealYear'] - df_trimmed['buildYear']
    df_trimmed['Age'] = df_trimmed['Age'].apply(lambda x: max(x, 0))
    df_trimmed['floor_area_ratio'] = df_trimmed['totalFloorAr'] / (df_trimmed['plottageAr'] + 1e-5)
    df_trimmed['deal_month_sin'] = np.sin(2 * np.pi * df_trimmed['dealMonth'] / 12)
    df_trimmed['deal_month_cos'] = np.cos(2 * np.pi * df_trimmed['dealMonth'] / 12)

    # 5. 데이터 분할 및 타겟 인코딩
    features = [
        'houseType', 'buildYear', 'plottageAr', 'totalFloorAr', 'dealYear', 
        'Age', 'floor_area_ratio', 'deal_month_sin', 'deal_month_cos', 
        'sggCd', 'umdNm', 'spatial_block_id'
    ]
    target = 'target_price_log'
    
    X = df_trimmed[features]
    y = df_trimmed[target]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Target Encoding (umdNm, sggCd)
    for col in ['umdNm', 'sggCd']:
        mapping = y_train.groupby(X_train[col]).mean()
        global_mean = y_train.mean()
        X_train[col] = X_train[col].map(mapping).fillna(global_mean)
        X_val[col] = X_val[col].map(mapping).fillna(global_mean)

    # 6. 고도화된 XGBoost 모델 학습
    print("Training Advanced XGBoost Regressor (Depth 12)...")
    model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=12,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
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

    # 7. 성능 평가 및 비교
    y_pred_log = model.predict(X_val)
    y_val_real = np.expm1(y_val)
    y_pred_real = np.expm1(y_pred_log)
    
    # 지표 계산
    r2 = r2_score(y_val_real, y_pred_real)
    mae = mean_absolute_error(y_val_real, y_pred_real)
    mape = np.mean(np.abs((y_val_real - y_pred_real) / y_val_real)) * 100
    
    baseline_mape = 43.95
    improvement = baseline_mape - mape
    
    print("\n" + "="*50)
    print(f"--- [Optimized Model Final Results] ---")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:,.2f} 원/m2")
    print(f"Final MAPE: {mape:.2f} %")
    print(f"MAPE Improved: {improvement:.2f} %p (vs Baseline {baseline_mape}%)")
    print("="*50)

    # 저장
    df_trimmed.to_csv('national_single_house_optimized.csv', index=False, encoding='cp949')

if __name__ == "__main__":
    target_data = "national_single_house_time_corrected.csv"
    run_optimized_pipeline(target_data)
