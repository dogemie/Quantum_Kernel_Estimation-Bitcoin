import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib

# GUI 충돌 방지
matplotlib.use('Agg')

# [설정]
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, "..", "data"))
RESULT_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, "..", "analysis_results"))
os.makedirs(RESULT_DIR, exist_ok=True)

def get_latest_metrics(base_path, prefix):
    """특정 접두사로 시작하는 폴더 중 가장 최근의 metrics 파일을 반환"""
    if not os.path.exists(base_path): return None
    folders = [f for f in os.listdir(base_path) if f.startswith(prefix) and os.path.isdir(os.path.join(base_path, f))]
    if not folders: return None
    latest_folder = sorted(folders)[-1]
    metrics_path = os.path.join(base_path, latest_folder, "performance_metrics.csv")
    return metrics_path if os.path.exists(metrics_path) else None

def main():
    all_results = []
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. 모든 run_{seed} 폴더 순회
    run_folders = [f for f in os.listdir(DATA_DIR) if f.startswith("run_")]
    print(f"📊 총 {len(run_folders)}개의 실험 데이터를 발견했습니다.")

    # [수정] 3클래스 레이블 리스트 정의
    target_labels = ['Neutral', 'Down', 'Up', 'accuracy']

    for run in run_folders:
        run_path = os.path.join(DATA_DIR, run)
        seed = run.split("_")[1]

        # 최신 예측 결과 경로 확보 (접두사 주의)
        csvm_path = get_latest_metrics(run_path, "classical_svm_prediction_")
        qsvm_path = get_latest_metrics(run_path, "quantum_kernel_prediction_")

        if csvm_path and qsvm_path:
            try:
                df_csvm = pd.read_csv(csvm_path, index_col=0)
                df_qsvm = pd.read_csv(qsvm_path, index_col=0)

                # [핵심 수정] 새로운 클래스 명칭으로 데이터 추출
                for label in target_labels:
                    if label in df_csvm.index and label in df_qsvm.index:
                        all_results.append({
                            "Seed": seed,
                            "Class": label,
                            "CSVM_F1": df_csvm.loc[label, "f1-score"] if label != 'accuracy' else df_csvm.loc[label, "precision"],
                            "QSVM_F1": df_qsvm.loc[label, "f1-score"] if label != 'accuracy' else df_qsvm.loc[label, "precision"]
                        })
            except Exception as e:
                print(f"⚠️ Seed {seed} 처리 중 오류: {e}")

    if not all_results:
        print("❌ 분석할 수 있는 결과 데이터가 없습니다. (CSV 인덱스를 확인하세요)")
        return

    # 2. 데이터프레임 변환 및 통계 계산
    master_df = pd.DataFrame(all_results)
    
    # 통계 요약 (Mean, Std)
    summary = master_df.groupby("Class").agg({
        "CSVM_F1": ["mean", "std"],
        "QSVM_F1": ["mean", "std"]
    })
    
    summary_filename = f"summary_report_3class_{timestamp_str}.csv"
    summary_path = os.path.join(RESULT_DIR, summary_filename)
    summary.to_csv(summary_path)
    print(f"✅ 통계 요약 리포트 저장 완료: {summary_path}")

    # 3. 시각화 (Box Plot)
    plt.figure(figsize=(14, 8))
    plot_data = master_df.melt(id_vars=["Seed", "Class"], value_vars=["CSVM_F1", "QSVM_F1"], 
                               var_name="Model", value_name="F1-Score")
    
    # 모델명 가독성 개선
    plot_data['Model'] = plot_data['Model'].replace({"CSVM_F1": "Classical SVM", "QSVM_F1": "Quantum SVM"})
    
    # [수정] 박스플롯 생성 시 x축 순서 고정
    order = ['Neutral', 'Down', 'Up', 'accuracy']
    sns.boxplot(data=plot_data, x="Class", y="F1-Score", hue="Model", palette="Set2", order=order)
    
    plt.title(f"Forecasting Performance Comparison: 3-Class Strategy ($t+1$)\n({timestamp_str})", fontsize=16)
    plt.ylabel("F1-Score / Accuracy", fontsize=12)
    plt.xlabel("Evaluation Metric", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    
    plot_filename = f"performance_comparison_3class_{timestamp_str}.png"
    plot_path = os.path.join(RESULT_DIR, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📈 비교 그래프 저장 완료: {plot_path}")

    # 4. 분석 결과 요약 출력
    print("\n" + "="*55)
    print(f" 🚀 [ 3클래스 모델 성능 비교 요약 ]")
    print("="*55)
    for label in ['Neutral', 'Down', 'Up']:
        if label in summary.index:
            c_m = summary.loc[label, ("CSVM_F1", "mean")]
            q_m = summary.loc[label, ("QSVM_F1", "mean")]
            diff = q_m - c_m
            status = "🟢 QSVM 우세" if diff > 0 else "🔴 CSVM 우세"
            print(f"{label:8} | {status} | 차이: {diff:+.4f}")
    
    if 'accuracy' in summary.index:
        acc_diff = summary.loc['accuracy', ("QSVM_F1", "mean")] - summary.loc['accuracy', ("CSVM_F1", "mean")]
        print("-" * 55)
        print(f"Total Accuracy Difference: {acc_diff:+.4f}")
    print("="*55)

if __name__ == "__main__":
    main()