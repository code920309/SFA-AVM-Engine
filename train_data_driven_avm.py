import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import os

# ==========================================
# 데이터 기반 인프라 대리 변수 생성 및 고도화 모델
# ==========================================

def run_data_driven_pipeline(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # 1. 데이터 로드 (정제 및 트리밍이 완료된 최신 데이터셋)
    df = pd.read_csv(file_path, encoding='cp949')
    total_rows = len(df)
    print(f"Data Loaded: {total_rows:,} rows")

    # 2. 동네 인프라 밀집도(infrastructure_density) 생성
    # 법정동(umdNm)별 고유 블록 수 카운트
    umd_infra = df.groupby('umdNm')['spatial_block_id'].nunique().reset_index()
    umd_infra.columns = ['umdNm', 'umd_block_count']
    
    # 상위 20% 임계값 계산
    threshold = umd_infra['umd_block_count'].quantile(0.8)
    umd_infra['is_urban_core'] = (umd_infra['umd_block_count'] >= threshold).astype(int)
    
    # 기존 데이터에 매핑
    df = pd.merge(df, umd_infra, on='umdNm', how='left')
    print(f"- Urban core threshold (Top 20%): {threshold} blocks")

    # 3. 블록별 거래 활성도(block_activity) 생성
    # 블록별 거래 건수 비율
    block_activity = df.groupby('spatial_block_id').size().reset_index()
    block_activity.columns = ['spatial_block_id', 'block_deal_count']
    block_activity['block_deal_ratio'] = block_activity['block_deal_count'] / total_rows
    
    # 기존 데이터에 매핑
    df = pd.merge(df, block_activity[['spatial_block_id', 'block_deal_ratio']], on='spatial_block_id', how='left')
    print("- Block activity features generated.")

    # 4. 모델 학습 피처 구성 및 전처리
    features = [
        'houseType', 'buildYear', 'plottageAr', 'totalFloorAr', 'dealYear', 
        'Age', 'floor_area_ratio', 'deal_month_sin', 'deal_month_cos', 
        'sggCd', 'umdNm', 'spatial_block_id',
        'umd_block_count', 'is_urban_core', 'block_deal_ratio'
    ]
    target = 'target_price_log'
    
    # 범주형 인코딩
    if df['houseType'].dtype == object:
        df['houseType'] = LabelEncoder().fit_transform(df['houseType'].astype(str))
    
    X = df[features]
    y = df[target]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Target Encoding (umdNm, sggCd)
    for col in ['umdNm', 'sggCd']:
        if df[col].dtype == object:
            mapping = y_train.groupby(X_train[col]).mean()
            global_mean = y_train.mean()
            X_train[col] = X_train[col].map(mapping).fillna(global_mean)
            X_val[col] = X_val[col].map(mapping).fillna(global_mean)

    # 5. XGBoost 고도화 학습
    print("Training Data-Driven XGBoost Regressor...")
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

    # 6. 결과 리포트 및 성능 비교
    y_pred_log = model.predict(X_val)
    y_val_real = np.expm1(y_val)
    y_pred_real = np.expm1(y_pred_log)
    
    r2 = r2_score(y_val_real, y_pred_real)
    mae = mean_absolute_error(y_val_real, y_pred_real)
    mape = np.mean(np.abs((y_val_real - y_pred_real) / y_val_real)) * 100
    
    # 이전 성적 가이드라인
    base_r2 = 0.8551
    base_mape = 39.60
    
    print("\n" + "="*50)
    print(f"--- [Data-Driven Infrastructure Model Results] ---")
    print(f"R2 Score: {r2:.4f} (Base: {base_r2})")
    print(f"MAE: {mae:,.2f} 원/m2")
    print(f"Final MAPE: {mape:.2f} % (Base: {base_mape}%)")
    print(f"MAPE Improved: {base_mape - mape:+.2f} %p")
    print("="*50)

    # 피처 중요도 확인
    importances = pd.Series(model.feature_importances_, index=features)
    print("\n--- [Top 10 Feature Importance] ---")
    print(importances.sort_values(ascending=False).head(10))

if __name__ == "__main__":
    # 최신 데이터셋 상속
    target_data = "national_single_house_optimized.csv"
    run_data_driven_pipeline(target_data)
