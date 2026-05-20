import pandas as pd
import numpy as np
import os
import joblib
import torch
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# ==========================================
# 단독 모델 튜닝 + 스택킹 앙상블 파이프라인
# ==========================================

OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def prepare_data(file_path):
    """시점 보정 데이터 로드 및 공통 피처 엔지니어링"""
    df = pd.read_csv(file_path, encoding='cp949')
    
    # 아웃라이어 트리밍 (1.5% ~ 98.5%)
    p_low, p_high = np.percentile(df['adj_price_per_m2'], [1.5, 98.5])
    df = df[(df['adj_price_per_m2'] >= p_low) & (df['adj_price_per_m2'] <= p_high)].copy()
    
    # 피처 생성
    df['Age'] = (df['dealYear'] - df['buildYear']).clip(lower=0)
    df['is_old_building'] = (df['Age'] >= 30).astype(int)
    df['floor_area_ratio'] = df['totalFloorAr'] / (df['plottageAr'] + 1e-5)
    df['deal_month_sin'] = np.sin(2 * np.pi * df['dealMonth'] / 12)
    df['deal_month_cos'] = np.cos(2 * np.pi * df['dealMonth'] / 12)
    
    # 법정동 노후화 지수 (umd_old_building_ratio)
    umd_old = df.groupby('umdNm')['is_old_building'].mean().reset_index()
    umd_old.columns = ['umdNm', 'umd_old_building_ratio']
    df = pd.merge(df, umd_old, on='umdNm', how='left')
    
    # 미시 공간 블록 생성
    df['jibun'] = df['jibun'].fillna('').astype(str)
    df['spatial_block_id_str'] = df['sggCd'].astype(str) + "_" + df['umdNm'] + "_" + df['jibun']
    
    # 블록별 평균 나이 및 거래 비율
    block_age = df.groupby('spatial_block_id_str')['Age'].mean()
    df['block_mean_age'] = df['spatial_block_id_str'].map(block_age)
    block_deal_ratio = (df.groupby('spatial_block_id_str').size() / len(df))
    df['block_deal_ratio'] = df['spatial_block_id_str'].map(block_deal_ratio)
    
    # 법정동 블록 수
    umd_block = df.groupby('umdNm')['spatial_block_id_str'].nunique()
    df['umd_block_count'] = df['umdNm'].map(umd_block)
    threshold = np.percentile(df['umd_block_count'], 80)
    df['is_urban_core'] = (df['umd_block_count'] >= threshold).astype(int)
    
    # 범주형 변수 Label Encoding
    cat_cols = ['houseType', 'sggCd', 'umdNm', 'spatial_block_id_str']
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    features = [
        'houseType', 'buildYear', 'plottageAr', 'totalFloorAr', 'dealYear',
        'Age', 'floor_area_ratio', 'deal_month_sin', 'deal_month_cos',
        'sggCd', 'umdNm', 'spatial_block_id_str',
        'is_old_building', 'umd_old_building_ratio', 'block_mean_age',
        'block_deal_ratio', 'umd_block_count', 'is_urban_core'
    ]
    
    X = df[features]
    y = df['target_price_log']
    
    return X, y, features, le_dict


def target_encode(X_tr, X_va, y_tr, cols):
    """Leakage-Free 타겟 인코딩"""
    X_tr, X_va = X_tr.copy(), X_va.copy()
    global_mean = y_tr.mean()
    for col in cols:
        mapping = y_tr.groupby(X_tr[col]).mean()
        X_tr[col] = X_tr[col].map(mapping).fillna(global_mean)
        X_va[col] = X_va[col].map(mapping).fillna(global_mean)
    return X_tr, X_va


def evaluate(name, y_true_log, y_pred_log):
    y_true = np.expm1(np.array(y_true_log).flatten())
    y_pred = np.expm1(np.array(y_pred_log).flatten())
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'Model': name, 'R2': round(r2, 4), 'MAE': int(mae), 'MAPE(%)': round(mape, 2)}


def run_full_pipeline(file_path):
    print("Loading and preparing data...")
    X, y, features, le_dict = prepare_data(file_path)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Target Encoding (트리 모델용)
    te_cols = ['umdNm', 'sggCd']
    X_tr_tree, X_va_tree = target_encode(X_train, X_val, y_train, te_cols)
    
    print(f"Train: {len(X_tr_tree):,} / Val: {len(X_va_tree):,}")
    results = []
    oof_preds = {}  # 앙상블용 OOF 예측값 저장

    # ==========================================
    # 1. XGBoost (Full Tuned)
    # ==========================================
    print("\n[1/4] Training Tuned XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=2500, learning_rate=0.03, max_depth=12,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, objective='reg:squarederror',
        early_stopping_rounds=50
    )
    xgb_model.fit(X_tr_tree, y_train, eval_set=[(X_va_tree, y_val)], verbose=100)
    xgb_pred = xgb_model.predict(X_va_tree)
    oof_preds['XGBoost'] = xgb_pred
    results.append(evaluate("XGBoost (Tuned)", y_val, xgb_pred))
    joblib.dump(xgb_model, f"{OUTPUT_DIR}/xgb_model.pkl")

    # ==========================================
    # 2. LightGBM (Full Tuned)
    # ==========================================
    print("\n[2/4] Training Tuned LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=2500, learning_rate=0.03, max_depth=12,
        num_leaves=127, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    )
    lgb_callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100)
    ]
    lgb_model.fit(
        X_tr_tree, y_train,
        eval_set=[(X_va_tree, y_val)],
        callbacks=lgb_callbacks
    )
    lgb_pred = lgb_model.predict(X_va_tree)
    oof_preds['LightGBM'] = lgb_pred
    results.append(evaluate("LightGBM (Tuned)", y_val, lgb_pred))
    joblib.dump(lgb_model, f"{OUTPUT_DIR}/lgb_model.pkl")

    # ==========================================
    # 3. CatBoost (Full Tuned)
    # ==========================================
    print("\n[3/4] Training Tuned CatBoost...")
    cat_model = CatBoostRegressor(
        iterations=2500, learning_rate=0.03, depth=10,
        subsample=0.8, colsample_bylevel=0.8,
        random_seed=42, verbose=200, early_stopping_rounds=50
    )
    cat_model.fit(X_tr_tree, y_train, eval_set=(X_va_tree, y_val))
    cat_pred = cat_model.predict(X_va_tree)
    oof_preds['CatBoost'] = cat_pred
    results.append(evaluate("CatBoost (Tuned)", y_val, cat_pred))
    cat_model.save_model(f"{OUTPUT_DIR}/cat_model.cbm")

    # ==========================================
    # 4. TabNet (Tuned)
    # ==========================================
    print("\n[4/4] Training Tuned TabNet...")
    cat_idxs = [X.columns.get_loc(c) for c in ['houseType', 'sggCd', 'umdNm', 'spatial_block_id_str']]
    cat_dims  = [len(le_dict[c].classes_) for c in ['houseType', 'sggCd', 'umdNm', 'spatial_block_id_str']]
    
    tab_model = TabNetRegressor(
        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=8,
        n_d=64, n_a=64, n_steps=5, gamma=1.3, momentum=0.02,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-2, weight_decay=1e-5),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=10, gamma=0.9),
        mask_type='entmax', verbose=50
    )
    tab_model.fit(
        X_train.values, y_train.values.reshape(-1, 1),
        eval_set=[(X_val.values, y_val.values.reshape(-1, 1))],
        max_epochs=200, patience=20, batch_size=2048, virtual_batch_size=256
    )
    tab_pred = tab_model.predict(X_val.values).flatten()
    oof_preds['TabNet'] = tab_pred
    results.append(evaluate("TabNet (Tuned)", y_val, tab_pred))
    tab_model.save_model(f"{OUTPUT_DIR}/tab_model")

    # ==========================================
    # 5. 스택킹 앙상블 (Simple Weighted Average)
    # ==========================================
    print("\n[Ensemble] Building stacking ensemble...")
    # 각 모델의 Val R2를 가중치로 사용해 가중 평균 앙상블 산출
    r2_scores = {k: r2_score(y_val, v) for k, v in oof_preds.items()}
    total_r2 = sum(r2_scores.values())
    weights = {k: v / total_r2 for k, v in r2_scores.items()}
    
    ensemble_pred = sum(oof_preds[m] * w for m, w in weights.items())
    results.append(evaluate("Ensemble (R2-Weighted)", y_val, ensemble_pred))
    
    # ==========================================
    # 최종 성적표 출력
    # ==========================================
    print("\n" + "="*60)
    print("--- [Final Model Scorecard] ---")
    score_df = pd.DataFrame(results).sort_values('R2', ascending=False)
    print(score_df.to_string(index=False))
    print("="*60)
    
    print("\n--- [Ensemble Weights] ---")
    for m, w in weights.items():
        print(f"  {m}: {w:.4f}")
    
    print(f"\nAll models saved to '{OUTPUT_DIR}/' directory.")


if __name__ == "__main__":
    target_data = "national_single_house_time_corrected.csv"
    run_full_pipeline(target_data)
