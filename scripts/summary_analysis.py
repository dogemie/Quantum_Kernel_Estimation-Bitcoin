import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# [설정]
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "..", "data")
RESULT_DIR = os.path.join(PROJECT_ROOT, "..", "analysis_results")
os.makedirs(RESULT_DIR, exist_ok=True)

def get_latest_metrics(base_path, prefix):
    """특정 접두사(classical_svm_ 등)로 시작하는 폴더 중 가장 최근의 metrics 파일을 반환"""
    folders = [f for f in os.listdir(base_path) if f.startswith(prefix) and os.path.isdir(os.path.join(base_path, f))]
    if not folders:
        return None
    # 가장 최근 생성된 폴더 선택
    latest_folder = sorted(folders)[-1]
    metrics_path = os.path.join(base_path, latest_folder, "performance_metrics.csv")
    return metrics_path if os.path.exists(metrics_path) else None

def main():
    all_results = []

    # 1. 모든 run_{seed} 폴더 순회
    run_folders = [f for f in os.listdir(DATA_DIR) if f.startswith("run_")]
    print(f"총 {len(run_folders)}개의 실험 데이터를 발견했습니다.")

    for run in run_folders:
        run_path = os.path.join(DATA_DIR, run)
        seed = run.split("_")[1]

        # 고전 및 양자 모델의 최신 결과 경로 확보
        csvm_path = get_latest_metrics(run_path, "classical_svm_")
        qsvm_path = get_latest_metrics(run_path, "quantum_kernel_")

        if csvm_path and qsvm_path:
            df_csvm = pd.read_csv(csvm_path, index_col=0)
            df_qsvm = pd.read_csv(qsvm_path, index_col=0)

            # 필요한 지표(F1-score) 추출
            for label in ['Normal', 'Dip', 'Flash', 'Vol', 'accuracy']:
                if label in df_csvm.index and label in df_qsvm.index:
                    all_results.append({
                        "Seed": seed,
                        "Class": label,
                        "CSVM_F1": df_csvm.loc[label, "f1-score"] if label != 'accuracy' else df_csvm.loc[label, "precision"],
                        "QSVM_F1": df_qsvm.loc[label, "f1-score"] if label != 'accuracy' else df_qsvm.loc[label, "precision"]
                    })

    if not all_results:
        print("분석할 수 있는 결과 데이터가 없습니다.")
        return

    # 2. 데이터프레임 변환 및 통계 계산
    master_df = pd.DataFrame(all_results)
    
    # 통계 요약 (Mean, Std)
    summary = master_df.groupby("Class").agg({
        "CSVM_F1": ["mean", "std"],
        "QSVM_F1": ["mean", "std"]
    })
    
    # 결과 저장
    summary_path = os.path.join(RESULT_DIR, f"summary_report_{datetime.now().strftime('%Y%m%d')}.csv")
    summary.to_csv(summary_path)
    print(f"통계 요약 리포트 저장 완료: {summary_path}")

    # 3. 시각화 (Box Plot)
    plt.figure(figsize=(12, 6))
    plot_data = master_df.melt(id_vars=["Seed", "Class"], value_vars=["CSVM_F1", "QSVM_F1"], 
                               var_name="Model", value_name="F1-Score")
    
    sns.boxplot(data=plot_data, x="Class", y="F1-Score", hue="Model", palette="Set2")
    plt.title("Statistical Comparison: Classical vs Quantum SVM", fontsize=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(RESULT_DIR, "performance_comparison_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"비교 그래프 저장 완료: {plot_path}")
    plt.show()

    # 4. 양자 우위 구간 출력
    print("\n" + "="*40)
    print("   [ 분석 결과 요약 ]")
    print("="*40)
    for label in ['Normal', 'Dip', 'Flash', 'Vol']:
        c_mean = summary.loc[label, ("CSVM_F1", "mean")]
        q_mean = summary.loc[label, ("QSVM_F1", "mean")]
        diff = q_mean - c_mean
        status = "🟢 양자 우세" if diff > 0 else "🔴 고전 우세"
        print(f"{label:7} : {status} (차이: {diff:+.4f})")
    print("="*40)

if __name__ == "__main__":
    main()