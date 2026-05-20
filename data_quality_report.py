import pandas as pd
import numpy as np
import os

# ==========================================
# 부동산 데이터 품질 진단 및 결측치 분석 스크립트
# ==========================================

def analyze_data_quality(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # 데이터 로드 (G2B 도메인 규칙에 따라 cp949 인코딩 사용)
    try:
        df = pd.read_csv(file_path, encoding='cp949')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='utf-8')

    # 1. 기본 크기 확인
    rows, cols = df.shape
    print(f"--- [1. 데이터 기본 정보] ---")
    print(f"총 행(Row) 수: {rows:,}")
    print(f"총 열(Column) 수: {cols:,}")
    print("-" * 30)

    # 2. 결측치 통계 산출
    null_counts = df.isnull().sum()
    null_percentage = (df.isnull().sum() / rows) * 100
    
    quality_df = pd.DataFrame({
        'Missing_Count': null_counts,
        'Missing_Percentage(%)': null_percentage.round(2)
    })
    
    # 결측치 많은 순으로 정렬
    quality_df = quality_df.sort_values(by='Missing_Count', ascending=False)
    
    print(f"--- [2. 컬럼별 결측치 통계 (내림차순)] ---")
    print(quality_df)
    print("-" * 30)

    # 3. 주요 컬럼 분석
    print(f"--- [3. 주요 컬럼 진단 인사이트] ---")
    
    # 거래 취소 관련
    cancellation_cols = ['cdealDay', 'cdealType']
    print(f"\n[거래 취소 정보]")
    for col in cancellation_cols:
        if col in df.columns:
            m_count = df[col].isnull().sum()
            print(f"- {col}: 결측치 {m_count:,}건 (정상 거래 비율: {(m_count/rows*100):.2f}%)")
        else:
            print(f"- {col}: 데이터셋에 해당 컬럼이 존재하지 않습니다.")

    # 학습 필수 컬럼
    essential_cols = ['plottageAr', 'totalFloorAr', 'dealAmount']
    print(f"\n[모델 학습 필수 데이터]")
    for col in essential_cols:
        if col in df.columns:
            m_count = df[col].isnull().sum()
            status = "정삭(결측 없음)" if m_count == 0 else f"주의(결측 {m_count:,}건 존재)"
            print(f"- {col}: {status}")
        else:
            print(f"- {col}: 데이터셋에 해당 컬럼이 존재하지 않습니다.")

    # 데이터 타입 확인 (dealAmount 등 숫자형 변환 필요 여부)
    if 'dealAmount' in df.columns:
        sample_value = df['dealAmount'].iloc[0]
        if isinstance(sample_value, str):
            print("\n* 참고: 'dealAmount'가 문자열 타입으로 저장되어 있습니다. 분석 전 콤마 제거 및 정수 변환이 필요합니다.")

if __name__ == "__main__":
    target_file = "national_single_house_5years_raw.csv"
    analyze_data_quality(target_file)
