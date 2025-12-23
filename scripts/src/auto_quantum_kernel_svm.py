import pennylane as qml
from pennylane import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import joblib
import argparse
from datetime import datetime

def manual_zz_feature_map(x, wires):
    """라이브러리 버전 이슈를 방지하기 위해 수동으로 설계한 ZZ Feature Map"""
    for i in range(len(wires)):
        qml.Hadamard(wires=wires[i])
    for i in range(len(wires)):
        qml.RZ(2.0 * x[i], wires=wires[i])
    for i in range(len(wires)):
        for j in range(i + 1, len(wires)):
            qml.CNOT(wires=[wires[i], wires[j]])
            qml.RZ(2.0 * (np.pi - x[i]) * (np.pi - x[j]), wires=wires[j])
            qml.CNOT(wires=[wires[i], wires[j]])

def main():
    # 1. 인자 처리
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Random seed for experiment')
    args = parser.parse_args()

    # 2. 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..")
    run_dir = os.path.join(project_root, "data", f"run_{args.seed}")
    
    input_x = os.path.join(run_dir, "X_quantum.npy")
    input_y = os.path.join(run_dir, "y_label.npy")

    # 결과 저장 서브 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(run_dir, f"quantum_kernel_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(input_x) or not os.path.exists(input_y):
        return

    # 3. 데이터 로드 및 분할
    X = np.load(input_x)
    y = np.load(input_y).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    # --- [저장 1: 학습 데이터] ---
    np.save(os.path.join(save_dir, 'X_train_final.npy'), X_train)
    np.save(os.path.join(save_dir, 'y_train_final.npy'), y_train)

    # 4. 양자 장치 및 커널 회로 설정
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        manual_zz_feature_map(x1, wires=range(4))
        qml.adjoint(manual_zz_feature_map)(x2, wires=range(4))
        return qml.probs(wires=range(4))

    def quantum_kernel(A, B, desc=""):
        total = len(A) * len(B)
        matrix = np.zeros((len(A), len(B)))
        count = 0
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                matrix[i, j] = kernel_circuit(a, b)[0]
                count += 1
                if count % 500 == 0: # 자동화 시에는 출력을 조금 더 간결하게 함
                    print(f"Seed {args.seed} {desc}: {count}/{total} ({(count/total)*100:.1f}%)")
        return matrix

    # 5. 커널 행렬 계산 및 저장
    gram_train = quantum_kernel(X_train, X_train, "Train")
    np.save(os.path.join(save_dir, 'gram_train.npy'), gram_train)

    gram_test = quantum_kernel(X_test, X_train, "Test")
    np.save(os.path.join(save_dir, 'gram_test.npy'), gram_test)

    # 6. QSVM 학습 및 저장
    model = SVC(kernel="precomputed", probability=True)
    model.fit(gram_train, y_train)
    joblib.dump(model, os.path.join(save_dir, 'qsvm_model.pkl'))

    # 7. 결과 예측 및 평가
    y_pred = model.predict(gram_test)
    
    target_labels = [0, 1, 2, 3]
    target_names = ['Normal', 'Dip', 'Flash', 'Vol']
    present_labels = sorted(list(set(y_test) | set(y_pred)))
    valid_labels = [l for l in target_labels if l in present_labels]
    valid_names = [target_names[l] for l in valid_labels]

    # --- [저장 3: 성능 리포트 TXT] ---
    # 1. TXT 저장용
    report_text = classification_report(y_test, y_pred, labels=valid_labels, target_names=valid_names, zero_division=0)
    
    # 2. CSV 저장용 (딕셔너리)
    report_dict = classification_report(y_test, y_pred, labels=valid_labels, target_names=valid_names, zero_division=0, output_dict=True)

    # TXT 저장
    with open(os.path.join(save_dir, 'performance_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)

    # CSV 저장
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(save_dir, 'performance_metrics.csv'), index=True)

    # --- [저장 4: Test 결과 CSV] ---
    test_results_df = pd.DataFrame(X_test, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    test_results_df['True_Label'] = y_test
    test_results_df['Predicted_Label'] = y_pred
    test_results_df.to_csv(os.path.join(save_dir, 'qsvm_test_results.csv'), index=False)

    # 8. 혼동 행렬 시각화 및 저장
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, 
        labels=valid_labels,
        display_labels=valid_names, 
        cmap='Purples', ax=ax
    )
    plt.title(f'Quantum SVM Confusion Matrix\nSeed: {args.seed}')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()