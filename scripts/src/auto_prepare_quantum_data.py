import pandas as pd
import numpy as np
import os
import argparse
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import matplotlib
matplotlib.use('Agg')

def main():
    # 1. 인자 처리
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Random seed to locate the run directory')
    args = parser.parse_args()

    # 2. 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(current_dir, "..", ".."))
    
    run_dir = os.path.join(project_root, "data", f"run_{args.seed}")
    input_file = os.path.join(run_dir, f"cleaned_btc_data_{args.seed}.csv")
    
    output_x = os.path.join(run_dir, "X_quantum.npy")
    output_y = os.path.join(run_dir, "y_label.npy")

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # 3. 데이터 로드
    df = pd.read_csv(input_file)
    
    # [수정] 분수 차분 특징을 포함한 16차원 벡터 구성
    base_features = ['FracDiff_Close', 'Range', 'RSI', 'Vol_Ratio']
    lagged_features = []
    
    # t 시점부터 t-3 시점(총 4개 캔들) 정보 수집
    for f in base_features:
        lagged_features.append(f)
        for i in range(1, 4):
            lagged_features.append(f + f'_lag{i}')
    
    X = df[lagged_features]
    y = df['Target_Label'].values

    # 4. [혁신] Robust Scaling 도입
    # 비트코인 특유의 꼬리가 두꺼운(Fat-tail) 분포와 이상치를 보존하기 위해 
    # StandardScaler 대신 중앙값과 사분위수를 사용하는 RobustScaler 적용
    scaler_robust = RobustScaler()
    X_robust = scaler_robust.fit_transform(X)

    # 5. [핵심] Kernel PCA를 통한 비선형 특징 추출
    # 단순 선형 PCA는 금융 데이터의 복잡한 상관관계를 뭉개버릴 수 있음
    # 'rbf' 커널을 사용한 KPCA를 통해 양자 힐베르트 공간에 적합한 비선형 특징을 먼저 추출
    kpca = KernelPCA(n_components=4, kernel='rbf', gamma=None, fit_inverse_transform=True, random_state=args.seed)
    X_kpca = kpca.fit_transform(X_robust)

    # 6. 양자 회로용 최적화 스케일링 (0 ~ pi)
    # ZZ Feature Map의 위상 변이(Phase Shift) 범위에 맞게 정렬
    scaler_minmax = MinMaxScaler(feature_range=(0, np.pi))
    X_qs = scaler_minmax.fit_transform(X_kpca)

    # 7. 최종 데이터 저장
    np.save(output_x, X_qs)
    np.save(output_y, y)

    # 정보 보존율 근사 보고 (Kernel PCA는 직접적인 분산 비율 제공이 어려워 변동성 보존으로 대체)
    print(f"[Prep 완료] Seed {args.seed}: 16D -> 4D (Non-linear Geometric Alignment applied)")

if __name__ == "__main__":
    main()