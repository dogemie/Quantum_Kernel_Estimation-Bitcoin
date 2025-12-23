import pandas as pd
import numpy as np
import os
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def main():
    # 1. 인자 처리 (자동화 스크립트로부터 seed를 받음)
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Random seed to locate the run directory')
    args = parser.parse_args()

    # 2. 경로 설정
    # 현재 파일 위치: (Project)/scripts/src/auto_prepare_quantum_data.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..")
    
    # 해당 시드의 작업 폴더 지정
    run_dir = os.path.join(project_root, "data", f"run_{args.seed}")
    input_file = os.path.join(run_dir, f"cleaned_btc_data_{args.seed}.csv")
    
    output_x = os.path.join(run_dir, "X_quantum.npy")
    output_y = os.path.join(run_dir, "y_label.npy")

    if not os.path.exists(input_file):
        return # 파일이 없으면 조용히 종료

    # 3. 데이터 로드
    df = pd.read_csv(input_file)
    features = ['Return', 'Range', 'RSI', 'Vol_Ratio']
    X = df[features]
    y = df['Label'].values

    # 4. 데이터 표준화 (Standardization)
    # PCA 전 필수: 각 지표의 단위를 평균 0, 표준편차 1로 정규화
    scaler_std = StandardScaler()
    X_std = scaler_std.fit_transform(X)

    # 5. PCA를 이용한 특징 재배열
    # 4개의 지표를 유지하되, 정보 밀도가 높은 축(주성분)으로 투영
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X_std)

    # 6. 양자 회로용 스케일링 (0 ~ pi)
    # 데이터를 양자 게이트의 회전 각도로 사용하기 위해 변환
    scaler_minmax = MinMaxScaler(feature_range=(0, np.pi))
    X_qs = scaler_minmax.fit_transform(X_pca)

    # 7. 최종 데이터 저장 (해당 시드 폴더 내)
    np.save(output_x, X_qs)
    np.save(output_y, y)

    # 자동화 파이프라인을 위해 요약 정보만 한 줄 출력 (선택 사항)
    # print(f"[PCA 완료] Seed {args.seed}: Explained Variance {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

if __name__ == "__main__":
    main()