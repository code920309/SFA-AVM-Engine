import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import os

# ==========================================
# XGBoost 기반 전국 부동산 가치 산정 Baseline 모델
# ==========================================

def run_model_pipeline(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # 1. 데이터 로드
    df = pd.read_csv(file_path, encoding='cp949')
    print(f"Data Loaded: {len(df):,} rows")

    # 2. 파생 피처 생성 (Feature Engineering)
    # 건물 노후도 (Age)
    df['Age'] = df['dealYear'] - df['buildYear']
    df['Age'] = df['Age'].apply(lambda x: max(x, 0))
    
    # 용적률성 면적비 (floor_area_ratio)
    df['floor_area_ratio'] = df['totalFloorAr'] / (df['plottageAr'] + 1e-5)
    
    # 계약월 계절성 반영 (Sin/Cos 변환)
    df['deal_month_sin'] = np.sin(2 * np.pi * df['dealMonth'] / 12)
    df['deal_month_cos'] = np.cos(2 * np.pi * df['dealMonth'] / 12)
    
    # 3. 데이터 전처리 및 인코딩
    features = [
        'houseType', 'buildYear', 'plottageAr', 'totalFloorAr', 'dealYear', 
        'Age', 'floor_area_ratio', 'deal_month_sin', 'deal_month_cos', 
        'sggCd', 'umdNm'
    ]
    target = 'target_price_log'
    
    # 범주형 변수 Label Encoding (houseType)
    le = LabelEncoder()
    df['houseType'] = le.fit_transform(df['houseType'].astype(str))
    
    # 데이터 분할 (Train 80%, Validation 20%)
    X = df[features]
    y = df[target]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Split: Train({len(X_train):,}), Val({len(X_val):,})")

    # Target Encoding (umdNm, sggCd) - Leakage-Free
    # Train 셋의 평균값으로 매핑 생성
    for col in ['umdNm', 'sggCd']:
        # Train 데이터의 타겟 평균 계산
        mapping = y_train.groupby(X_train[col]).mean()
        X_train[col] = X_train[col].map(mapping)
        # Train의 전체 평균으로 결측치(Val에만 있는 범주) 채우기
        global_mean = y_train.mean()
        X_val[col] = X_val[col].map(mapping).fillna(global_mean)
        # Train 내의 결측치 처리 (동일하게 글로벌 평균)
        X_train[col] = X_train[col].fillna(global_mean)

    # 4. XGBoost 모델 학습
    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror',
        early_stopping_rounds=30 # 2.x+ 버전 사양에 맞춰 생성자로 이동
    )
    
    # 학습 실행
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    # 5. 최종 평가 및 역변환
    y_pred_log = model.predict(X_val)
    
    # 로그 복원 (log1p -> expm1)
    y_val_real = np.expm1(y_val)
    y_pred_real = np.expm1(y_pred_log)
    
    # 지표 계산
    r2 = r2_score(y_val_real, y_pred_real)
    mae = mean_absolute_error(y_val_real, y_pred_real)
    mape = np.mean(np.abs((y_val_real - y_pred_real) / y_val_real)) * 100
    
    print("\n" + "="*50)
    print(f"--- [Model Evaluation Results] ---")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:,.2f} 원/m2")
    print(f"MAPE: {mape:.2f} %")
    print("="*50)

    # 6. 피처 중요도 출력
    importances = pd.Series(model.feature_importances_, index=features)
    top_10 = importances.sort_values(ascending=False).head(10)
    
    print("\n--- [Top 10 Feature Importance] ---")
    print(top_10)

if __name__ == "__main__":
    target_data = "national_single_house_cleaned.csv"
    run_model_pipeline(target_data)
