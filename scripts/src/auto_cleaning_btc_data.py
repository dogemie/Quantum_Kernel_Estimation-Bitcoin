import pandas as pd
import numpy as np
import os
import argparse

def main():
    # 1. 인자 처리 (자동화 스크립트로부터 seed를 받음)
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Random seed for sampling')
    args = parser.parse_args()

    # 2. 경로 설정
    # 현재 파일 위치: (Project)/scripts/src/auto_cleaning_btc_data.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..")
    
    input_path = os.path.join(project_root, "archive", "btcusd_1-min_data.csv")
    output_dir = os.path.join(project_root, "data", f"run_{args.seed}")
    output_file = os.path.join(output_dir, f"cleaned_btc_data_{args.seed}.csv")

    # 결과 저장 폴더 생성 (없을 경우)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_path):
        return # 에러 발생 시 조용히 종료하거나 로그만 남김

    # 3. 데이터 로드 (필요한 컬럼만 최적화하여 로드)
    df = pd.read_csv(input_path, usecols=['Timestamp', 'High', 'Low', 'Close', 'Volume'])

    # 4. 전처리 및 리샘플링 (5분 단위)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.dropna()
    df.set_index('Timestamp', inplace=True)

    df_5min = pd.DataFrame()
    df_5min['Close'] = df['Close'].resample('5min').last()
    df_5min['High'] = df['High'].resample('5min').max()
    df_5min['Low'] = df['Low'].resample('5min').min()
    df_5min['Volume'] = df['Volume'].resample('5min').sum()
    df_5min = df_5min.dropna()

    # 5. 지표 생성 (Feature Engineering)
    df_5min['Return'] = np.log(df_5min['Close'] / df_5min['Close'].shift(1))
    df_5min['Range'] = (df_5min['High'] - df_5min['Low']) / df_5min['Close']

    # RSI 계산
    delta = df_5min['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df_5min['RSI'] = 100 - (100 / (1 + rs))

    # 거래량 변화율
    df_5min['Vol_Ratio'] = df_5min['Volume'] / df_5min['Volume'].rolling(window=5).mean()
    df_5min = df_5min.dropna()

    # 6. 다중 클래스 라벨링 (0~3)
    df_5min['Label'] = 0
    std_low = df_5min['Return'].rolling(window=100).std()

    # Label 1: Dip / Label 2: Flash Crash / Label 3: High Vol
    df_5min.loc[(df_5min['Return'] <= std_low * -1.5) & (df_5min['Return'] > std_low * -2.5), 'Label'] = 1
    df_5min.loc[(df_5min['Return'] <= std_low * -2.5) & (df_5min['Vol_Ratio'] > 3.0), 'Label'] = 2
    df_5min.loc[(df_5min['Range'] > df_5min['Range'].rolling(100).mean() * 3) & (df_5min['Label'] == 0), 'Label'] = 3

    # 7. 클래스별 균등 샘플링 (전달받은 seed 사용)
    samples_per_class = 50
    final_df = pd.concat([
        df_5min[df_5min['Label'] == i].sample(
            n=min(samples_per_class, len(df_5min[df_5min['Label'] == i])), 
            random_state=args.seed
        )
        for i in [0, 1, 2, 3]
    ]).sort_index()

    # 8. 저장
    final_df.to_csv(output_file)

if __name__ == "__main__":
    main()