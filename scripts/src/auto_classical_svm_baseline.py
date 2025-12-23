import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import joblib
import os
import argparse
from datetime import datetime

def main():
    # 1. 인자 처리
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Random seed for data split and directory location')
    args = parser.parse_args()

    # 2. 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..")
    run_dir = os.path.join(project_root, "data", f"run_{args.seed}")
    
    # 입력 데이터 경로
    input_x = os.path.join(run_dir, "X_quantum.npy")
    input_y = os.path.join(run_dir, "y_label.npy")

    # 결과 저장 서브 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(run_dir, f"classical_svm_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(input_x) or not os.path.exists(input_y):
        return

    # 3. 데이터 로드 및 분할
    X = np.load(input_x)
    y = np.load(input_y).astype(int) # 라벨 정수형 강제 변환

    # 추출 시 사용한 seed를 분할 시에도 적용하여 데이터 일관성 유지
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    # --- [저장 1: 학습 데이터] ---
    np.save(os.path.join(save_dir, 'X_train_final.npy'), X_train)
    np.save(os.path.join(save_dir, 'y_train_final.npy'), y_train)

    # 4. 고전 SVM 모델 생성 및 학습
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    model.fit(X_train, y_train)

    # --- [저장 2: 학습된 모델] ---
    joblib.dump(model, os.path.join(save_dir, 'classical_svm_model.pkl'))

    # 5. 결과 예측 및 평가
    y_pred = model.predict(X_test)
    
    # 실제 존재하는 라벨에 맞춰 타겟 네임 설정 (에러 방지)
    target_labels = [0, 1, 2, 3]
    target_names = ['Normal', 'Dip', 'Flash', 'Vol']
    present_labels = sorted(list(set(y_test) | set(y_pred)))
    valid_labels = [l for l in target_labels if l in present_labels]
    valid_names = [target_names[l] for l in valid_labels]

    # 1. TXT 저장용 (문자열)
    report_text = classification_report(y_test, y_pred, labels=valid_labels, target_names=valid_names, zero_division=0)
    
    # 2. CSV 저장용 (딕셔너리 옵션 추가)
    report_dict = classification_report(y_test, y_pred, labels=valid_labels, target_names=valid_names, zero_division=0, output_dict=True)

    # TXT 저장
    with open(os.path.join(save_dir, 'performance_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)

    # CSV 저장 (딕셔너리를 DataFrame으로 변환)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(save_dir, 'performance_metrics.csv'), index=True)
    # ------------------------------

    # 6. 혼동 행렬 시각화 및 저장
    # 자동화 실행 시 GUI 창이 뜨지 않도록 파일로 바로 저장
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, 
        labels=valid_labels,
        display_labels=valid_names, 
        cmap='Blues', ax=ax
    )
    plt.title(f'Classical SVM Confusion Matrix\nSeed: {args.seed}')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    plt.close(fig) # 메모리 해제

if __name__ == "__main__":
    main()