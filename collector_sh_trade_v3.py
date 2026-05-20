import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ==========================================
# G2B IT Bid Prediction Project - Advanced Data Engineering
# 목적: 국토교통부 단독/다가구 매매 실거래가 고속 병렬 수집 (이어받기 기능 포함)
# ==========================================

class MOLITResumeCollector:
    def __init__(self, service_key, max_workers=6):
        self.endpoint = "https://apis.data.go.kr/1613000/RTMSDataSvcSHTrade/getRTMSDataSvcSHTrade"
        self.service_key = service_key
        self.max_workers = max_workers
        self.output_file = "national_single_house_5years_raw.csv"
        self.lock = threading.Lock()
        self.collected_set = set() # (sigungu, ym) 집합
        self.all_results = []
        
    def load_existing_data(self):
        """기존에 수집된 데이터를 로드하여 중복 수집을 방지합니다."""
        if os.path.exists(self.output_file):
            try:
                df = pd.read_csv(self.output_file, encoding='cp949')
                if not df.empty:
                    # 기존 데이터에서 (지역코드, 연월) 쌍을 추출
                    # API 응답 필드명에 따라 다를 수 있으나, 보통 sggCd(또는 LAWD_CD)와 dealYear/Month를 조합
                    # 여기서는 안전하게 전체 행을 유지하고 작업 대상에서 제외할 수 있도록 로직 설계
                    print(f"Existing data found: {len(df)} rows. Resuming...")
                    self.all_results = df.to_dict('records')
                    
                    # 수집 완료된 지역/연월 식별 (법정동 및 계약년월 기준)
                    # 단독/다가구 API는 'sggCd'와 'dealYear', 'dealMonth' 제공
                    # 또는 수집 시점에 사용한 LAWD_CD와 DEAL_YMD를 기록해두는 것이 가장 정확함
                    # 여기서는 데이터 내의 sggCd와 년월을 조합해 set 생성
                    for _, row in df.iterrows():
                        key = (str(row.get('sggCd', '')), f"{row.get('dealYear', '')}{int(row.get('dealMonth', 0)):02d}")
                        self.collected_set.add(key)
                return True
            except Exception as e:
                print(f"Error loading existing data: {e}")
        return False

    def get_valid_sigungu_codes(self):
        sigungu_patterns = {
            '서울': [f'11{i:03d}' for i in range(110, 750, 10)],
            '부산': [f'26{i:03d}' for i in range(110, 720, 10)],
            '대구': [f'27{i:03d}' for i in range(110, 720, 10)] + ['27290'],
            '인천': [f'28{i:03d}' for i in range(110, 750, 10)],
            '광주': [f'29{i:03d}' for i in range(110, 210, 10)],
            '대전': [f'30{i:03d}' for i in range(110, 240, 10)],
            '울산': [f'31{i:03d}' for i in range(110, 710, 10)],
            '세종': ['36110'],
            '경기': [f'41{i:03d}' for i in range(110, 840, 10)],
            '강원': [f'51{i:03d}' for i in range(110, 830, 10)],
            '충북': [f'43{i:03d}' for i in range(110, 810, 10)],
            '충남': [f'44{i:03d}' for i in range(110, 810, 10)],
            '전북': [f'52{i:03d}' for i in range(110, 800, 10)],
            '전남': [f'46{i:03d}' for i in range(110, 910, 10)],
            '경북': [f'47{i:03d}' for i in range(110, 950, 10)],
            '경남': [f'48{i:03d}' for i in range(110, 900, 10)],
            '제주': ['50110', '50130']
        }
        all_codes = []
        for codes in sigungu_patterns.values():
            all_codes.extend(codes)
        return sorted(list(set(all_codes)))

    def get_deal_ymd_list(self, start_ym="202101", end_ym="202605"):
        start_year, start_month = int(start_ym[:4]), int(start_ym[4:])
        end_year, end_month = int(end_ym[:4]), int(end_ym[4:])
        ym_list = []
        curr_year, curr_month = start_year, start_month
        while (curr_year < end_year) or (curr_year == end_year and curr_month <= end_month):
            ym_list.append(f"{curr_year}{curr_month:02d}")
            curr_month += 1
            if curr_month > 12:
                curr_month = 1
                curr_year += 1
        return ym_list

    def fetch_api(self, code, ym):
        """단일 호출 및 결과 반환"""
        if (code, ym) in self.collected_set:
            return [] # 이미 수집됨
            
        params = {'serviceKey': self.service_key, 'LAWD_CD': code, 'DEAL_YMD': ym, 'numOfRows': '9999'}
        try:
            time.sleep(0.1)
            resp = requests.get(self.endpoint, params=params, timeout=30)
            if resp.status_code == 429:
                print("Quota exceeded again. Stop.")
                return None
            if resp.status_code != 200: return []
            
            root = ET.fromstring(resp.content)
            items = root.findall('.//item')
            res = []
            for item in items:
                row = {child.tag: (child.text.strip() if child.text else "") for child in item}
                # 보조 필드 추가
                row['sggCd'] = code 
                res.append(row)
            return res
        except:
            return []

    def run(self):
        self.load_existing_data()
        codes = self.get_valid_sigungu_codes()
        yms = self.get_deal_ymd_list()
        
        # 전체 작업 리스트 생성 (미수집 항목만)
        tasks = []
        for c in codes:
            for y in yms:
                if (c, y) not in self.collected_set:
                    tasks.append((c, y))
        
        if not tasks:
            print("All data already collected.")
            return

        print(f"Resuming Collection: {len(tasks)} tasks remaining | 6 Threads")
        
        start_time = time.time()
        success_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {executor.submit(self.fetch_api, c, y): (c, y) for c, y in tasks}
            
            for i, future in enumerate(as_completed(future_to_task)):
                c, y = future_to_task[future]
                data = future.result()
                
                if data is None: # 429 에러 등 중단 신호
                    print("Terminating due to API issues.")
                    break
                
                if data:
                    with self.lock:
                        self.all_results.extend(data)
                        success_count += 1
                
                # 100회 호출마다 저장 (실시간 저장성 강화)
                if (i + 1) % 100 == 0:
                    self.save_to_csv()
                    elapsed = time.time() - start_time
                    print(f"Progress: [{i+1}/{len(tasks)}] tasks finished. Elapsed: {elapsed/60:.1f}m")

        self.save_to_csv()
        print("Collection finished.")

    def save_to_csv(self):
        with self.lock:
            if self.all_results:
                df = pd.DataFrame(self.all_results).drop_duplicates()
                df.to_csv(self.output_file, index=False, encoding='cp949')

import os
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    SERVICE_KEY = os.getenv("MOLIT_DECODED_KEY")
    
    if not SERVICE_KEY:
        print("Error: MOLIT_DECODED_KEY not found in .env file.")
    else:
        collector = MOLITResumeCollector(SERVICE_KEY)
        collector.run()
