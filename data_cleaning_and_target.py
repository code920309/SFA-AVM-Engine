import pandas as pd
import numpy as np
import os

# ==========================================
# 부동산 데이터 정제 및 타겟 변수 엔지니어링 스크립트
# ==========================================

def clean_and_feature_engineering(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # 1. 데이터 로드
    print(f"Loading data: {input_file}")
    df = pd.read_csv(input_file, encoding='cp949')
    initial_count = len(df)
    
    # 2. 거래 취소 데이터(Noise) 제거
    # cdealDay와 cdealType이 모두 Null인 행만 유지
    # API 응답에 따라 ' ' (공백)이나 None으로 올 수 있으므로 정밀하게 처리
    df = df[df['cdealDay'].isna() | (df['cdealDay'].str.strip() == '')]
    df = df[df['cdealType'].isna() | (df['cdealType'].str.strip() == '')]
    after_filter_noise = len(df)
    print(f"- Noise 제거 후 행 수: {after_filter_noise:,} (제거됨: {initial_count - after_filter_noise:,})")

    # 3. 핵심 변수 타입 변환 및 정제 (dealAmount)
    # dealAmount: "12,000" -> 12000 -> 120,000,000 (만 원 -> 원)
    if 'dealAmount' in df.columns:
        df['dealAmount'] = df['dealAmount'].astype(str).str.replace(',', '').str.strip()
        df['dealAmount'] = pd.to_numeric(df['dealAmount'], errors='coerce')
        # 결측치 제거
        df = df.dropna(subset=['dealAmount'])
        # 단위 보정: 국토부 실거래가는 '만 원' 단위이므로 '원' 단위로 통합
        df['dealAmount'] = (df['dealAmount'] * 10000).astype(np.int64)

    # 4. 건축년도(buildYear) 정제
    if 'buildYear' in df.columns:
        df['buildYear'] = pd.to_numeric(df['buildYear'], errors='coerce')
        # 1900년 이전이거나 0이거나 결측치인 경우 제거
        df = df[df['buildYear'] >= 1900]
        print(f"- 비정상 건축년도 제거 후 행 수: {len(df):,}")

    # 5. 타겟 변수 생성 (대지면적당 단가)
    # 단독주택의 핵심인 '땅값(대지면적당 가격)' 산출
    if 'plottageAr' in df.columns:
        df['plottageAr'] = pd.to_numeric(df['plottageAr'], errors='coerce')
        # 면적이 0이거나 결측치인 경우 방지
        df = df[df['plottageAr'] > 0]
        
        # price_per_m2 생성 (단위: 원/m2)
        df['price_per_m2'] = df['dealAmount'] / df['plottageAr']
        
        # 타겟 로그 변환 (로그 정규화)
        df['target_price_log'] = np.log1p(df['price_per_m2'])
        print(f"- 타겟 변수(target_price_log) 생성 완료.")

    # 6. 불필요 컬럼 제거 (취소 관련 및 결측치 60% 이상)
    cols_to_drop = [
        'cdealDay', 'cdealType', 'buyerGbn', 'slerGbn', 'estateAgentSggNm'
    ]
    # 실제 존재하는 컬럼만 선택하여 제거
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drops)
    
    # 7. 최종 결과 확인 및 저장
    print(f"--- [최종 데이터셋 요약] ---")
    print(f"최종 행 수: {len(df):,}")
    print(f"최종 컬럼 수: {len(df.columns)}")
    print(df[['dealAmount', 'price_per_m2', 'target_price_log']].head())
    
    # 저장 (CP949 인코딩)
    df.to_csv(output_file, index=False, encoding='cp949')
    print(f"Cleaned data saved to: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    input_f = "national_single_house_5years_raw.csv"
    output_f = "national_single_house_cleaned.csv"
    clean_and_feature_engineering(input_f, output_f)
