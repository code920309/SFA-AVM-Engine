import pandas as pd
import numpy as np
import os
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# ==========================================
# 4대 하이브리드 모델(XGB, LGBM, CAT, TabNet) 비교 파이프라인
# ==========================================

def run_hybrid_comparison(file_path):
    # 1. 데이터 로드 및 정제 (Pure Data 고도화 로직 적용)
    df = pd.read_csv(file_path, encoding='cp949')
    
    # Trimming (1.5% ~ 98.5%)
    p_low, p_high = np.percentile(df['adj_price_per_m2'], [1.5, 98.5])
    df = df[(df['adj_price_per_m2'] >= p_low) & (df['adj_price_per_m2'] <= p_high)].copy()
    
    # 피처 생성
    df['Age'] = df['dealYear'] - df['buildYear']
    df['Age'] = df['Age'].apply(lambda x: max(x, 0))
    df['is_old_building'] = (df['Age'] >= 30).astype(int)
    df['floor_area_ratio'] = df['totalFloorAr'] / (df['plottageAr'] + 1e-5)
    df['deal_month_sin'] = np.sin(2 * np.pi * df['dealMonth'] / 12)
    df['deal_month_cos'] = np.cos(2 * np.pi * df['dealMonth'] / 12)
    
    # 공간 블록 생성
    df['jibun'] = df['jibun'].fillna('').astype(str)
    df['spatial_block_id_str'] = df['sggCd'].astype(str) + "_" + df['umdNm'] + "_" + df['jibun']
    
    # 2. 범주형 변수 처리
    cat_features = ['houseType', 'sggCd', 'umdNm', 'spatial_block_id_str']
    le_dict = {}
    for col in cat_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    features = [
        'houseType', 'buildYear', 'plottageAr', 'totalFloorAr', 'dealYear', 
        'Age', 'floor_area_ratio', 'deal_month_sin', 'deal_month_cos', 
        'sggCd', 'umdNm', 'spatial_block_id_str', 'is_old_building'
    ]
    target = 'target_price_log'
    
    X = df[features]
    y = df[target].values.reshape(-1, 1)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. 모델 정의 및 학습 (비교용)
    results = []
    
    def evaluate(name, y_true_log, y_pred_log):
        y_true = np.expm1(y_true_log).flatten()
        y_pred = np.expm1(y_pred_log).flatten()
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return {'Model': name, 'R2': round(r2, 4), 'MAE': round(mae, 0), 'MAPE(%)': round(mape, 2)}

    # (1) XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    results.append(evaluate("XGBoost", y_val, xgb_model.predict(X_val)))

    # (2) LightGBM
    print("Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train, y_train.flatten())
    results.append(evaluate("LightGBM", y_val, lgb_model.predict(X_val)))

    # (3) CatBoost
    print("Training CatBoost...")
    cat_model = CatBoostRegressor(n_estimators=500, learning_rate=0.05, depth=8, random_state=42, verbose=False)
    cat_model.fit(X_train, y_train)
    results.append(evaluate("CatBoost", y_val, cat_model.predict(X_val)))

    # (4) TabNet
    print("Training TabNet (PyTorch)...")
    cat_idxs = [X.columns.get_loc(c) for c in cat_features]
    cat_dims = [len(le_dict[c].classes_) for c in cat_features]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tabnet_model = TabNetRegressor(
        cat_idxs=cat_idxs, cat_dims=cat_dims,
        cat_emb_dim=4, optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2), verbose=0
    )
    
    tabnet_model.fit(
        X_train.values, y_train,
        eval_set=[(X_val.values, y_val)],
        max_epochs=50, patience=10, batch_size=1024, virtual_batch_size=128
    )
    results.append(evaluate("TabNet", y_val, tabnet_model.predict(X_val.values)))

    # 4. 비교 결과 출력
    print("\n" + "="*50)
    print("--- [Hybrid AVM Model Comparison] ---")
    print(pd.DataFrame(results))
    print("="*50 + "\n")

    # 5. TabNet XAI (Attention Sampling)
    print("--- [TabNet XAI: Local Feature Importance] ---")
    sample_idx = 0
    explain_matrix, masks = tabnet_model.explain(X_val.values[sample_idx:sample_idx+1])
    
    importance = explain_matrix[0]
    feat_imp = pd.Series(importance, index=features).sort_values(ascending=False)
    print(f"Sample Property Target Mapping focus:")
    print(feat_imp.head(5))

if __name__ == "__main__":
    target_data = "national_single_house_time_corrected.csv"
    run_hybrid_comparison(target_data)
