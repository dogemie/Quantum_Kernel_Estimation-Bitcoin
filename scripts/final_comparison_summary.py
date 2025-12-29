import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib

# [설정] GUI 충돌 방지 및 경로 자동 인식
matplotlib.use('Agg')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..")) 
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
    # 분석 대상 모델 리스트
    models = ["Baseline_CSVM", "Baseline_QSVM", "Hybrid_Fusion", "Hybrid_Cascading", "Hybrid_Stacking"]
    
    # 1. 모든 run_ 폴더 순회 및 데이터 수집
    if not os.path.exists(DATA_DIR):
        print(f"❌ 데이터 폴더를 찾을 수 없습니다: {DATA_DIR}")
        return

    run_folders = [f for f in os.listdir(DATA_DIR) if f.startswith("run_") and os.path.isdir(os.path.join(DATA_DIR, f))]
    print(f"✅ 총 {len(run_folders)}개의 실험 데이터셋 분석을 시작합니다.")

    for run in run_folders:
        run_path = os.path.join(DATA_DIR, run)
        hybrid_path = get_latest_hybrid_dir(run_path)
        
        if not hybrid_path:
            continue

        for model in models:
            csv_path = os.path.join(hybrid_path, f"metrics_{model}.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path, index_col=0)
                    # [수정] 3-클래스 지표 추출 (Neutral, Down, Up)
                    all_metrics.append({
                        "Seed": run.replace("run_", ""),
                        "Model": model,
                        "Accuracy": float(df.loc["accuracy", "f1-score"] if "f1-score" in df.columns else df.loc["accuracy", "precision"]),
                        "Down_F1": float(df.loc["Down", "f1-score"]),
                        "Up_F1": float(df.loc["Up", "f1-score"]),
                        "Neutral_F1": float(df.loc["Neutral", "f1-score"]),
                        "Macro_F1": float(df.loc["macro avg", "f1-score"])
                    })
                except Exception as e:
                    print(f"⚠️ {run} {model} 파싱 오류: {e}")

    if not all_metrics:
        print("❌ 분석할 하이브리드 데이터가 없습니다. triple_hybrid_all.py를 실행했는지 확인하세요.")
        return

    # 2. 통합 데이터프레임 생성 및 통계 요약
    master_df = pd.DataFrame(all_metrics)
    metric_cols = ["Accuracy", "Down_F1", "Up_F1", "Neutral_F1", "Macro_F1"]
    
    # 모델별 평균 및 표준편차 계산
    stats = master_df.groupby("Model")[metric_cols].agg(['mean', 'std'])
    stats.columns = [f"{col[0]}_{col[1]}" for col in stats.columns.values]
    
    # 타임스탬프 기반 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_path = os.path.join(RESULT_DIR, f"total_hybrid_statistical_summary_{timestamp}.csv")
    stats.to_csv(stats_path)
    print(f"📊 통계 요약 저장 완료: {stats_path}")

    # 3. 시각화 (Box Plot)
    plt.figure(figsize=(16, 9))
    plot_data = master_df.melt(id_vars=["Seed", "Model"], value_vars=metric_cols, var_name="Metric", value_name="Score")
    
    sns.boxplot(data=plot_data, x="Metric", y="Score", hue="Model", palette="husl")
    plt.title(f"Final Performance Comparison: 3-Class Forecasting ($t+1$)\nMeasured on {timestamp}", fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    plot_path = os.path.join(RESULT_DIR, f"total_hybrid_performance_boxplot_{timestamp}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📈 최종 비교 그래프 저장 완료: {plot_path}")
    
    # 4. 결론 도출 (하락장 예측 특화 모델 선정)
    best_down_model = stats["Down_F1_mean"].idxmax()
    best_up_model = stats["Up_F1_mean"].idxmax()
    best_normal_model = stats["Neutral_F1_mean"].idxmax()
    best_overall_model = stats["Macro_F1_mean"].idxmax()
    
    print("\n" + "="*65)
    print(f" 🏆 [ 하이브리드 3-클래스 분석 결론 ]")
    print("-" * 65)
    print(f" 1. 전체 예측 안정성 (Macro F1) 최우수: {best_overall_model}")
    print(f" 2. 하락(Down) 예측 성능 최우수       : {best_down_model}")
    print(f" 3. 상승(Up) 예측 성능 최우수         : {best_up_model}")
    print(f" 4. 중립(Neutral) 예측 성능 최우수     : {best_normal_model}")
    print(f" * 상세 데이터는 analysis_results 폴더를 참조하십시오.")
    print("="*65)

if __name__ == "__main__":
    main()