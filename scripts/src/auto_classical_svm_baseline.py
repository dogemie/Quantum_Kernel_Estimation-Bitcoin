# 1. Matplotlib 백엔드 설정을 반드시 'pyplot' 임포트보다 위에 배치
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import joblib
import os
import argparse
import warnings # 경고 메시지 제어용
from datetime import datetime

# 2. Sklearn 및 Matplotlib의 경고 메시지가 stderr 버퍼를 채우지 않도록 무시 설정
warnings.filterwarnings('ignore')

def main():
    # 1. 인자 처리
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Random seed for data split')
    args = parser.parse_args()

    # 2. 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(current_dir, "..", ".."))
    run_dir = os.path.join(project_root, "data", f"run_{args.seed}")
    
    input_x = os.path.join(run_dir, "X_quantum.npy")
    input_y = os.path.join(run_dir, "y_label.npy")

    # 결과 저장 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(run_dir, f"classical_svm_prediction_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(input_x) or not os.path.exists(input_y):
        print(f"Error: Data not found for seed {args.seed}")
        return

    # 3. 데이터 로드 및 분할
    X = np.load(input_x)
    y = np.load(input_y).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    # 데이터 저장
    np.save(os.path.join(save_dir, 'X_test_final.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_test_final.npy'), y_test)
    np.save(os.path.join(save_dir, 'X_train_final.npy'), X_train)
    np.save(os.path.join(save_dir, 'y_train_final.npy'), y_train)

    # 4. 고전 SVM 모델 생성 및 학습
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    model.fit(X_train, y_train)

    joblib.dump(model, os.path.join(save_dir, 'classical_svm_model.pkl'))

    # 5. 3클래스 결과 예측 및 평가
    y_pred = model.predict(X_test)
    
    # 수정된 레이블 체계 적용: 0=Neutral, 1=Down, 2=Up
    target_labels = [0, 1, 2]
    target_names = ['Neutral', 'Down', 'Up']
    
    present_labels = sorted(list(set(y_test) | set(y_pred)))
    valid_labels = [l for l in target_labels if l in present_labels]
    valid_names = [target_names[l] for l in valid_labels]

    report_dict = classification_report(y_test, y_pred, labels=valid_labels, 
                                        target_names=valid_names, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(save_dir, 'performance_metrics.csv'), index=True)
    # 6. 혼동 행렬 저장 (Agg 백엔드로 GUI 없이 파일만 생성)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=valid_labels, display_labels=valid_names, cmap='Blues', ax=ax
    )
    plt.title(f'Classical SVM Prediction (t+1)\nSeed: {args.seed}')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    plt.close(fig) # 메모리 해제 필수

    # 부모 프로세스에 종료 신호(0)를 보내기 위해 명시적 성공 메시지 출력
    print(f"Seed {args.seed}: Classical SVM 완료")

if __name__ == "__main__":
    main()