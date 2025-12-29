import pandas as pd
import numpy as np
import os
import argparse
import warnings
import sys

warnings.filterwarnings('ignore')

def get_weights_optimized(d, size):
    """분수 차분 가중치 생성"""
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] / k * (d - k + 1)
        w.append(w_k)
    return np.array(w)

def frac_diff_vectorized(series, d, thres=1e-5):
    """컨볼루션을 이용한 전역적 분수 차분 연산"""
    w = get_weights_optimized(d, 500)
    w = w[np.abs(w) > thres]
    diff_values = np.convolve(series.values, w[::-1], mode='valid')
    return pd.Series(diff_values, index=series.index[len(w)-1:])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(current_dir, "..", ".."))
    input_path = os.path.join(project_root, "archive", "btcusd_1-min_data.csv")
    output_dir = os.path.join(project_root, "data", f"run_{args.seed}")
    output_file = os.path.join(output_dir, f"cleaned_btc_data_{args.seed}.csv")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_path): return

    # 1. [전역 로드] 전체 시계열 로드 및 5분 리샘플링
    try:
        # 전체 역사를 로드 (메모리 내 약 100만 행 처리 가능)
        df_raw = pd.read_csv(input_path, usecols=['Timestamp', 'High', 'Low', 'Close', 'Volume'])
        df_raw['Timestamp'] = pd.to_datetime(df_raw['Timestamp'], unit='s')
        df_raw.set_index('Timestamp', inplace=True)
        
        df_5min = df_raw.resample('5min').agg({
            'Close': 'last', 'High': 'max', 'Low': 'min', 'Volume': 'sum'
        }).dropna()
        del df_raw # 원본 메모리 즉시 해제
    except Exception as e:
        print(f"Error loading full data: {e}")
        return

    # 2. [전역 특징량] 전체 역사에 대한 분수 차분 적용
    # d=0.4로 정상성을 확보하며 역사적 추세를 보존
    df_5min['FracDiff_Close'] = frac_diff_vectorized(df_5min['Close'], d=0.4)
    df_5min = df_5min.dropna()

    # 3. [전역 라벨링] 전체 기간에 대한 트리플 배리어 적용
    log_ret = np.log(df_5min['Close'] / df_5min['Close'].shift(1))
    vol = log_ret.rolling(window=20).std()
    
    close_vals = df_5min['Close'].values
    vol_vals = vol.values
    pt_sl = [5.0, 5.0] # 5% 이익/손실 목표
    v_barrier = 4 # 20분 후 강제 종료
    
    labels = []
    for i in range(len(close_vals) - v_barrier):
        start_price = close_vals[i]
        curr_vol = vol_vals[i]
        if np.isnan(curr_vol):
            labels.append(np.nan)
            continue
        
        up_b, low_b = start_price * (1 + curr_vol * pt_sl[0]), start_price * (1 - curr_vol * pt_sl[1])
        label = 0 # Neutral
        for j in range(1, v_barrier + 1):
            if close_vals[i+j] >= up_b: label = 2; break # Up
            elif close_vals[i+j] <= low_b: label = 1; break # Down
        labels.append(label)

    df_full = df_5min.iloc[:len(labels)].copy()
    df_full['Target_Label'] = labels
    df_full = df_full.dropna()

    # 4. 보조 지표 생성
    df_full['Range'] = (df_full['High'] - df_full['Low']) / df_full['Close']
    df_full['Vol_Ratio'] = df_full['Volume'] / df_full['Volume'].rolling(window=5).mean()
    delta = df_full['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df_full['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

    for f in ['FracDiff_Close', 'Range', 'RSI', 'Vol_Ratio']:
        for i in range(1, 4):
            df_full[f'{f}_lag{i}'] = df_full[f].shift(i)
    df_full = df_full.dropna()

    # 5. [전역 샘플링] CUSUM 필터를 통한 역사적 이벤트 풀 형성
    h = vol.mean() * 0.5 # 전역 변동성 평균의 50%를 임계값으로 설정
    diff = df_full['FracDiff_Close'].diff().fillna(0).values
    t_events, s_pos, s_neg = [], 0, 0
    full_indices = df_full.index
    
    for i in range(len(diff)):
        s_pos = max(0, s_pos + diff[i])
        s_neg = min(0, s_neg + diff[i])
        if s_pos > h or s_neg < -h:
            t_events.append(full_indices[i])
            s_pos, s_neg = 0, 0
    
    event_pool = df_full.loc[df_full.index.intersection(t_events)]

    # 6. [균형 샘플링] 클래스별 100개씩 추출 (전체 역사에서 무작위 선택)
    samples_per_class = 150
    sampled_list = []
    for i in [0, 1, 2]:
        class_set = event_pool[event_pool['Target_Label'] == i]
        if not class_set.empty:
            # [수정] 무작위 시드를 활용해 전체 역사에서 150개 추출
            sampled_list.append(class_set.sample(n=min(samples_per_class, len(class_set)), random_state=args.seed))
    
    if sampled_list:
        final_df = pd.concat(sampled_list).sort_index()
        final_df.to_csv(output_file)
        print(f"[Done] Seed {args.seed}: Global Historical Sampling (100 per class) Complete")

if __name__ == "__main__":
    main()