import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# [설정] 경로 자동 인식
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..")) 
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULT_DIR = os.path.join(PROJECT_ROOT, "analysis_results")
os.makedirs(RESULT_DIR, exist_ok=True)

def get_latest_hybrid_dir(run_path):
    """특정 run 폴더 내에서 가장 최근의 하이브리드 결과 폴더 반환"""
    if not os.path.exists(run_path): return None
    hybrids = [d for d in os.listdir(run_path) if d.startswith("hybrid_comparison_")]
    return os.path.join(run_path, sorted(hybrids)[-1]) if hybrids else None

def main():
    all_metrics = []
    models = ["Baseline_CSVM", "Baseline_QSVM", "Hybrid_Fusion", "Hybrid_Cascading", "Hybrid_Stacking"]
    
    # 1. 모든 run_ 폴더 순회
    if not os.path.exists(DATA_DIR):
        print(f"❌ 데이터 폴더를 찾을 수 없습니다: {DATA_DIR}")
        return

    run_folders = [f for f in os.listdir(DATA_DIR) if f.startswith("run_") and os.path.isdir(os.path.join(DATA_DIR, f))]
    print(f"✅ 총 {len(run_folders)}개의 실험 데이터를 발견했습니다.")

    for run in run_folders:
        run_path = os.path.join(DATA_DIR, run)
        hybrid_path = get_latest_hybrid_dir(run_path)
        
        if not hybrid_path:
            print(f"⚠️ {run} : 하이브리드 결과가 없어 건너뜜 (먼저 auto_triple_hybrid_compare.py를 실행하세요)")
            continue

        for model in models:
            csv_path = os.path.join(hybrid_path, f"metrics_{model}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, index_col=0)
                try:
                    all_metrics.append({
                    "Seed": run.replace("run_", ""),
                    "Model": model,
                    "Accuracy": float(df.loc["accuracy", "precision"]),
                    "Dip_F1": float(df.loc["Dip", "f1-score"]),
                    "Flash_F1": float(df.loc["Flash", "f1-score"]),
                    "Vol_F1": float(df.loc["Vol", "f1-score"]),  
                    "Normal_F1": float(df.loc["Normal", "f1-score"]),
                    "Macro_F1": float(df.loc["macro avg", "f1-score"])
                })
                except Exception as e:
                    print(f"⚠️ {run} {model} 데이터 읽기 오류: {e}")

    if not all_metrics:
        print("❌ 분석할 데이터가 없습니다.")
        return

    # 2. 통합 데이터프레임 생성
    master_df = pd.DataFrame(all_metrics)
    
    # [핵심 수정] 수치 계산에 사용할 컬럼들만 명시적으로 선택
    metric_cols = ["Accuracy", "Dip_F1", "Flash_F1", "Vol_F1", "Normal_F1", "Macro_F1"]
    
    # 모델별 평균 및 표준편차 계산 (수치형 컬럼만 필터링)
    # stats = master_df.groupby("Model")[metric_cols].agg(['mean', 'std']).round(4)
    stats = master_df.groupby("Model")[metric_cols].agg(['mean', 'std'])
    
    stats.columns = [f"{col[0]}_{col[1]}" for col in stats.columns.values]
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_path = os.path.join(RESULT_DIR, f"total_statistical_summary_{timestamp}.csv")
    stats.to_csv(stats_path)
    print(f"📊 통계 요약 저장 완료: {stats_path}")

    # 3. 시각화 (Box Plot)
    plt.figure(figsize=(14, 8))
    plot_data = master_df.melt(id_vars=["Seed", "Model"], value_vars=metric_cols, var_name="Metric", value_name="Score")
    
    sns.boxplot(data=plot_data, x="Metric", y="Score", hue="Model", palette="husl")
    plt.title("Statistical Performance Comparison: Baselines vs Hybrid Methods", fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plot_path = os.path.join(RESULT_DIR, f"total_performance_boxplot_{timestamp}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"📈 비교 그래프 저장 완료: {plot_path}")
    
    # 4. 결론 도출
    best_dip_model = master_df.groupby("Model")["Dip_F1"].mean().idxmax()
    print("\n" + "="*50)
    print(f"🏆 [ 분석 결론 ]")
    print(f" - Dip(이상치) 탐지에 가장 효과적인 모델: {best_dip_model}")
    print(f" - 데이터 건수가 1개인 경우 표준편차(std)는 NaN으로 표시됩니다.")
    print("="*50)

if __name__ == "__main__":
    main()