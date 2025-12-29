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
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis_results")

def get_latest_summary_report():
    """3-클래스 형식의 최신 summary_report를 찾아 Neutral, Down, Up에 대한 QSVM 가중치 반환"""
    if not os.path.exists(ANALYSIS_DIR):
        print("⚠️ analysis_results 폴더가 없어 기본 가중치(0.5)를 사용합니다.")
        return np.array([0.5, 0.5, 0.5])

    # 3-클래스 리포트 파일 필터링
    files = [f for f in os.listdir(ANALYSIS_DIR) if f.startswith("summary_report_3class_") and f.endswith(".csv")]
    if not files:
        print("⚠️ 3-클래스 요약 리포트가 없어 기본 가중치를 사용합니다.")
        return np.array([0.5, 0.5, 0.5])

    latest_file = sorted(files)[-1]
    report_path = os.path.join(ANALYSIS_DIR, latest_file)
    print(f"📊 최신 3-클래스 리포트 로드: {latest_file}")

    try:
        # MultiIndex(Class, Metric) 구조 로드
        df = pd.read_csv(report_path, index_col=0, header=[0, 1])
        target_classes = ['Neutral', 'Down', 'Up']
        weights = []

        for cls in target_classes:
            if cls in df.index:
                csvm_f1 = df.loc[cls, ('CSVM_F1', 'mean')]
                qsvm_f1 = df.loc[cls, ('QSVM_F1', 'mean')]
                # 상대적 성능 비중 계산
                w_q = qsvm_f1 / (csvm_f1 + qsvm_f1 + 1e-9)
                weights.append(w_q)
            else:
                weights.append(0.5)
        
        return np.array(weights)
    except Exception as e:
        print(f"❌ 가중치 계산 중 오류: {e}")
        return np.array([0.5, 0.5, 0.5])

def get_latest_folder(run_dir, prefix):
    """최신 예측 폴더(_prediction_) 반환"""
    folders = [f for f in os.listdir(run_dir) if f.startswith(prefix) and os.path.isdir(os.path.join(run_dir, f))]
    return os.path.join(run_dir, sorted(folders)[-1]) if folders else None

def has_hybrid_folder(run_dir):
    folders = [f for f in os.listdir(run_dir) if f.startswith("hybrid_comparison_")]
    return len(folders) > 0

def run_hybrid_analysis(run_folder_name, dynamic_weights):
    run_dir = os.path.join(DATA_DIR, run_folder_name)
    seed = int(run_folder_name.replace("run_", ""))
    
    if has_hybrid_folder(run_dir):
        print(f"⏩ [Skip] {run_folder_name}: 하이브리드 결과가 이미 존재합니다.")
        return

    # 2. 모델 로드 (최신 네이밍 규칙 적용)
    csvm_folder = get_latest_folder(run_dir, "classical_svm_prediction_")
    qsvm_folder = get_latest_folder(run_dir, "quantum_kernel_prediction_")
    
    if not csvm_folder or not qsvm_folder:
        return

    save_dir = os.path.join(run_dir, f"hybrid_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)

    # 데이터 및 모델 로드
    csvm = joblib.load(os.path.join(csvm_folder, "classical_svm_model.pkl"))
    qsvm = joblib.load(os.path.join(qsvm_folder, "qsvm_model.pkl"))
    X = np.load(os.path.join(run_dir, "X_quantum.npy"))
    y = np.load(os.path.join(run_dir, "y_label.npy")).astype(int)
    gram_train = np.load(os.path.join(qsvm_folder, "gram_train.npy"))
    gram_test = np.load(os.path.join(qsvm_folder, "gram_test.npy"))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    # 확률값 추출
    prob_c = csvm.predict_proba(X_test)
    prob_q = qsvm.predict_proba(gram_test)
    prob_c_train = csvm.predict_proba(X_train)
    prob_q_train = qsvm.predict_proba(gram_train)

    results = {}
    target_names = ['Neutral', 'Down', 'Up']

    # --- [기법 1: 동적 의사결정 융합 (Dynamic Decision Fusion)] ---
    # 3-클래스별 동적 가중치 적용
    prob_fusion = (prob_q * dynamic_weights) + (prob_c * (1 - dynamic_weights))
    results['Hybrid_Fusion'] = np.argmax(prob_fusion, axis=1)

    # --- [기법 2: 계층적 분류 (Cascading)] ---
    threshold = 0.6
    y_cascading = []
    for i in range(len(prob_c)):
        if np.max(prob_c[i]) < threshold:
            y_cascading.append(np.argmax(prob_q[i]))
        else:
            y_cascading.append(np.argmax(prob_c[i]))
    results['Hybrid_Cascading'] = np.array(y_cascading)

    # --- [기법 3: 메타 학습 (Stacked Generalization)] ---
    X_meta_train = np.hstack([prob_c_train, prob_q_train])
    X_meta_test = np.hstack([prob_c, prob_q])
    meta_model = LogisticRegression(max_iter=1000).fit(X_meta_train, y_train)
    results['Hybrid_Stacking'] = meta_model.predict(X_meta_test)

    results['Baseline_CSVM'] = np.argmax(prob_c, axis=1)
    results['Baseline_QSVM'] = np.argmax(prob_q, axis=1)

    # 4. 결과 저장 (3-클래스 지표 반영)
    comparison_data = []
    for name, y_pred in results.items():
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(os.path.join(save_dir, f"metrics_{name}.csv"))
        
        comparison_data.append({
            "Method": name,
            "Accuracy": report['accuracy'],
            "Macro_F1": report['macro avg']['f1-score'],
            "Down_F1": report['Down']['f1-score'],
            "Up_F1": report['Up']['f1-score'],
            "Neutral_F1": report['Neutral']['f1-score']
        })

    pd.DataFrame(comparison_data).to_csv(os.path.join(save_dir, "hybrid_total_comparison.csv"), index=False)
    print(f"✅ [Done] {run_folder_name}: 가중치 {dynamic_weights} 적용 분석 완료")

def main():
    if not os.path.exists(DATA_DIR): return

    # 동적 가중치 로드 (Neutral, Down, Up 순서)
    dynamic_weights = get_latest_summary_report()
    print(f"🚀 결정된 융합 가중치 (QSVM 비중):")
    for i, name in enumerate(['Neutral', 'Down', 'Up']):
        print(f"   - {name:8}: {dynamic_weights[i]:.4f}")

    run_folders = [f for f in os.listdir(DATA_DIR) if f.startswith("run_") and os.path.isdir(os.path.join(DATA_DIR, f))]
    for run_folder in sorted(run_folders):
        try:
            run_hybrid_analysis(run_folder, dynamic_weights)
        except Exception as e:
            print(f"❌ {run_folder} 예외: {e}")

if __name__ == "__main__":
    main()