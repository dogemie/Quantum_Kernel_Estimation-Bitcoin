import numpy as np
import pandas as pd
import os
import joblib
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# [설정] 프로젝트 루트 및 데이터 경로
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def get_latest_folder(run_dir, prefix):
    """특정 접두사로 시작하는 폴더 중 가장 최근 폴더 반환"""
    folders = [f for f in os.listdir(run_dir) if f.startswith(prefix) and os.path.isdir(os.path.join(run_dir, f))]
    return os.path.join(run_dir, sorted(folders)[-1]) if folders else None

def has_hybrid_folder(run_dir):
    """이미 하이브리드 결과 폴더가 존재하는지 확인"""
    folders = [f for f in os.listdir(run_dir) if f.startswith("hybrid_comparison_")]
    return len(folders) > 0

def run_hybrid_analysis(run_folder_name):
    run_dir = os.path.join(DATA_DIR, run_folder_name)
    seed = int(run_folder_name.replace("run_", ""))
    
    # 1. 중복 실행 방지 체크
    if has_hybrid_folder(run_dir):
        print(f"⏩ [Skip] {run_folder_name}: 하이브리드 결과가 이미 존재합니다.")
        return

    print(f"🚀 [Process] {run_folder_name}: 하이브리드 분석을 시작합니다.")

    # 2. 경로 설정 및 모델 로드
    csvm_folder = get_latest_folder(run_dir, "classical_svm_")
    qsvm_folder = get_latest_folder(run_dir, "quantum_kernel_")
    
    if not csvm_folder or not qsvm_folder:
        print(f"⚠️ [Error] {run_folder_name}: 모델 폴더를 찾을 수 없습니다. 건너뜁니다.")
        return

    save_dir = os.path.join(run_dir, f"hybrid_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)

    # 모델 및 데이터 로드
    csvm = joblib.load(os.path.join(csvm_folder, "classical_svm_model.pkl"))
    qsvm = joblib.load(os.path.join(qsvm_folder, "qsvm_model.pkl"))
    X = np.load(os.path.join(run_dir, "X_quantum.npy"))
    y = np.load(os.path.join(run_dir, "y_label.npy")).astype(int)
    gram_train = np.load(os.path.join(qsvm_folder, "gram_train.npy"))
    gram_test = np.load(os.path.join(qsvm_folder, "gram_test.npy"))
    
    # 데이터 분할 (기존 실험과 동일한 seed 사용)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    # 3. 확률값 추출
    prob_c = csvm.predict_proba(X_test)
    prob_q = qsvm.predict_proba(gram_test)
    prob_c_train = csvm.predict_proba(X_train)
    prob_q_train = qsvm.predict_proba(gram_train)

    results = {}
    target_names = ['Normal', 'Dip', 'Flash', 'Vol']

    # --- [기법 1: 의사결정 융합 (Decision Fusion)] ---
    # 각 모델의 예측 확률을 클래스별 가중치에 따라 합산
    weights = np.array([0.6005, 0.6755, 0.2815, 0.4825]) # Dip(1)에 양자 가중치 집중
    prob_fusion = (prob_q * weights) + (prob_c * (1 - weights))
    results['Hybrid_Fusion'] = np.argmax(prob_fusion, axis=1)

    # --- [기법 2: 계층적 분류 (Cascading)] ---
    # CSVM 결과가 모호할 때만 QSVM에게 최종 판단을 맡김
    threshold = 0.6
    y_cascading = []
    for i in range(len(prob_c)):
        if np.max(prob_c[i]) < threshold:
            y_cascading.append(np.argmax(prob_q[i]))
        else:
            y_cascading.append(np.argmax(prob_c[i]))
    results['Hybrid_Cascading'] = np.array(y_cascading)

    # --- [기법 3: 메타 학습 (Stacked Generalization)] ---
    # CSVM과 QSVM의 예측값을 새로운 특징으로 삼아 최종 결정
    X_meta_train = np.hstack([prob_c_train, prob_q_train])
    X_meta_test = np.hstack([prob_c, prob_q])
    meta_model = LogisticRegression().fit(X_meta_train, y_train) # 인덱스 오류 수정 완료
    results['Hybrid_Stacking'] = meta_model.predict(X_meta_test)

    # 베이스라인 기록
    results['Baseline_CSVM'] = np.argmax(prob_c, axis=1)
    results['Baseline_QSVM'] = np.argmax(prob_q, axis=1)

    # 4. 결과 저장
    comparison_data = []
    for name, y_pred in results.items():
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(os.path.join(save_dir, f"metrics_{name}.csv"))
        
        comparison_data.append({
            "Method": name,
            "Accuracy": report['accuracy'],
            "Macro_F1": report['macro avg']['f1-score'],
            "Dip_F1": report['Dip']['f1-score'],
            "Vol_F1": report['Vol']['f1-score']
        })

    pd.DataFrame(comparison_data).to_csv(os.path.join(save_dir, "hybrid_total_comparison.csv"), index=False)
    print(f"✅ [Done] {run_folder_name}: 분석 완료 및 저장 ({save_dir})")

def main():
    if not os.path.exists(DATA_DIR):
        print(f"❌ 데이터 폴더가 없습니다: {DATA_DIR}")
        return

    run_folders = [f for f in os.listdir(DATA_DIR) if f.startswith("run_") and os.path.isdir(os.path.join(DATA_DIR, f))]
    print(f"🔍 총 {len(run_folders)}개의 실험 폴더를 검사합니다.")

    for run_folder in sorted(run_folders):
        try:
            run_hybrid_analysis(run_folder)
        except Exception as e:
            print(f"❌ {run_folder} 처리 중 예외 발생: {e}")

if __name__ == "__main__":
    main()