import numpy as np
import pandas as pd
import os
import joblib
import argparse
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

'''
python src/auto_triple_hybrid_compare.py --seed 223
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    # 1. 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..")
    run_dir = os.path.join(project_root, "data", f"run_{args.seed}")
    
    # 최신 폴더 찾기 함수
    def get_latest(prefix):
        folders = [f for f in os.listdir(run_dir) if f.startswith(prefix)]
        return os.path.join(run_dir, sorted(folders)[-1]) if folders else None

    csvm_folder = get_latest("classical_svm_")
    qsvm_folder = get_latest("quantum_kernel_")
    
    save_dir = os.path.join(run_dir, f"hybrid_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)

    # 2. 모델 및 데이터 로드
    csvm = joblib.load(os.path.join(csvm_folder, "classical_svm_model.pkl"))
    qsvm = joblib.load(os.path.join(qsvm_folder, "qsvm_model.pkl"))
    
    X = np.load(os.path.join(run_dir, "X_quantum.npy"))
    y = np.load(os.path.join(run_dir, "y_label.npy")).astype(int)
    gram_train = np.load(os.path.join(qsvm_folder, "gram_train.npy"))
    gram_test = np.load(os.path.join(qsvm_folder, "gram_test.npy"))
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)

    # 3. 각 모델의 확률값 추출
    # CSVM은 특징 데이터를, QSVM은 Precomputed Gram 행렬을 입력으로 사용
    prob_c = csvm.predict_proba(X_test)
    prob_q = qsvm.predict_proba(gram_test)
    
    # 학습 세트 확률 (Stacking용)
    prob_c_train = csvm.predict_proba(train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)[0])
    prob_q_train = qsvm.predict_proba(gram_train)

    results = {}

    # --- [기법 1: 의사결정 융합 (Decision Fusion)] ---
    # Dip(1) 구간에서 양자 우위를 반영하여 가중치 설정
    weights = np.array([0.5, 0.7, 0.4, 0.4]) # 각 클래스별 양자 모델 가중치
    prob_fusion = (prob_q * weights) + (prob_c * (1 - weights))
    y_fusion = np.argmax(prob_fusion, axis=1)
    results['Hybrid_Fusion'] = y_fusion

    # --- [기법 2: 계층적 분류 (Cascading)] ---
    # CSVM의 확신도가 낮은(0.6 미만) 샘플만 QSVM이 판단
    threshold = 0.6
    y_cascading = []
    for i in range(len(prob_c)):
        if np.max(prob_c[i]) < threshold:
            y_cascading.append(np.argmax(prob_q[i]))
        else:
            y_cascading.append(np.argmax(prob_c[i]))
    results['Hybrid_Cascading'] = np.array(y_cascading)

    # --- [기법 3: 메타 학습 (Stacking)] ---
    # 두 모델의 확률값을 새로운 특징으로 사용하여 로지스틱 회귀 학습
    X_meta_train = np.hstack([prob_c_train, prob_q_train])
    X_meta_test = np.hstack([prob_c, prob_q])
    
    y_train_internal = train_test_split(y, test_size=0.2, random_state=args.seed, stratify=y)[0]
    
    meta_model = LogisticRegression().fit(X_meta_train, y_train_internal)
    results['Hybrid_Stacking'] = meta_model.predict(X_meta_test)

    # 4. 성능 비교 및 저장
    comparison_data = []
    target_names = ['Normal', 'Dip', 'Flash', 'Vol']
    
    # 베이스라인 추가
    results['Baseline_CSVM'] = np.argmax(prob_c, axis=1)
    results['Baseline_QSVM'] = np.argmax(prob_q, axis=1)

    for name, y_pred in results.items():
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(save_dir, f"metrics_{name}.csv"))
        
        comparison_data.append({
            "Method": name,
            "Accuracy": report['accuracy'],
            "Dip_F1": report['Dip']['f1-score'],
            "Flash_F1": report['Flash']['f1-score']
        })

    pd.DataFrame(comparison_data).to_csv(os.path.join(save_dir, "hybrid_total_comparison.csv"), index=False)
    print(f"하이브리드 비교 완료: {save_dir}")

if __name__ == "__main__":
    main()