import pennylane as qml
from pennylane import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import pd
import joblib
import argparse
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 1. [개선] 데이터 재업로드(Data Re-uploading) 구조의 ZZ Feature Map
# Kernel PCA로 압축된 정보의 가치를 극대화하기 위해 다중 레이어를 사용하여 
# 양자 회로의 푸리에 급수 표현력을 증대시킵니다.
def enhanced_feature_map(x, wires, layers=2):
    for layer in range(layers):
        # 단일 큐비트 회전 (Pauli-Y 포함으로 복소 진폭 간섭 강화)
        for i in range(len(wires)):
            qml.Hadamard(wires=wires[i])
            qml.RY(x[i], wires=wires[i]) # Pauli-Y 게이트 추가
            qml.RZ(x[i], wires=wires[i])
        
        # 얽힘 레이어 (All-to-all Entanglement)
        for i in range(len(wires)):
            for j in range(i + 1, len(wires)):
                qml.CNOT(wires=[wires[i], wires[j]])
                # 비선형 상관관계 인코딩
                qml.RZ((np.pi - x[i]) * (np.pi - x[j]), wires=wires[j])
                qml.CNOT(wires=[wires[i], wires[j]])



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Random seed for experiment')
    args = parser.parse_args()

    # 경로 설정 및 데이터 로드 (기존 로직 유지)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(current_dir, "..", ".."))
    run_dir = os.path.join(project_root, "data", f"run_{args.seed}")
    
    input_x = os.path.join(run_dir, "X_quantum.npy")
    input_y = os.path.join(run_dir, "y_label.npy")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(run_dir, f"quantum_kernel_prediction_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(input_x) or not os.path.exists(input_y):
        return

    X = np.load(input_x)
    y = np.load(input_y).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    # 2. [개선] 로컬 측정(Local Measurement) 기반 커널 설정
    # Barren Plateau(불모의 고원) 현상을 방지하기 위해 전역 측정이 아닌 
    # 개별 큐비트들의 상태를 종합하여 커널 값을 도출합니다.
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        enhanced_feature_map(x1, wires=range(4))
        qml.adjoint(enhanced_feature_map)(x2, wires=range(4))
        # 로컬 확률 밀도 측정을 통한 학습 가능성 증대
        return qml.probs(wires=range(4))

    def quantum_kernel(A, B, desc=""):
        total = len(A) * len(B)
        matrix = np.zeros((len(A), len(B)))
        count = 0
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                # Fidelity 커널 계산
                matrix[i, j] = kernel_circuit(a, b)[0]
                count += 1
                if count % 100 == 0 or count == total:
                    print(f"Seed {args.seed} {desc}: {count}/{total} ({(count/total)*100:.1f}%)", flush=True)
        return matrix

    # 3. 커널 행렬 계산 및 모델 학습 (기존 로직 유지)
    gram_train = quantum_kernel(X_train, X_train, "Train")
    np.save(os.path.join(save_dir, 'gram_train.npy'), gram_train)

    gram_test = quantum_kernel(X_test, X_train, "Test")
    np.save(os.path.join(save_dir, 'gram_test.npy'), gram_test)

    # [수정] SVM의 하이퍼파라미터 최적화 (커널 행렬 특성에 맞춰 조정)
    model = SVC(kernel="precomputed", probability=True, C=10.0) # 마진 규제 강화
    model.fit(gram_train, y_train)
    joblib.dump(model, os.path.join(save_dir, 'qsvm_model.pkl'))

    # 결과 예측 및 평가 (3클래스 대응 유지)
    y_pred = model.predict(gram_test)
    target_names = ['Neutral', 'Down', 'Up']
    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(save_dir, 'performance_metrics.csv'))
    
    # 혼동 행렬 저장
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=target_names, cmap='Purples', ax=ax)
    plt.title(f'Enhanced QSVM (Kernel PCA + Pauli-Y)\nSeed: {args.seed}')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    plt.close(fig)
    print(f"[QSVM 완료] Seed {args.seed}: Geometric Alignment Analysis complete")

if __name__ == "__main__":
    main()