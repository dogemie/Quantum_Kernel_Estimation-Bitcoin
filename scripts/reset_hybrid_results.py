import os
import shutil

# [설정] 프로젝트 루트 및 데이터 경로 인식
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def reset_hybrid_folders():
    if not os.path.exists(DATA_DIR):
        print(f"❌ 데이터 폴더를 찾을 수 없습니다: {DATA_DIR}")
        return

    # 1. 모든 run_ 폴더 찾기
    run_folders = [f for f in os.listdir(DATA_DIR) if f.startswith("run_") and os.path.isdir(os.path.join(DATA_DIR, f))]
    
    total_deleted = 0
    print(f"🔍 총 {len(run_folders)}개의 실험 폴더에서 하이브리드 결과 삭제를 시작합니다.")

    for run_folder in run_folders:
        run_path = os.path.join(DATA_DIR, run_folder)
        
        # 2. 각 run_ 폴더 내부의 hybrid_comparison_ 폴더 찾기
        targets = [d for d in os.listdir(run_path) if d.startswith("hybrid_comparison_") and os.path.isdir(os.path.join(run_path, d))]
        
        for target in targets:
            target_path = os.path.join(run_path, target)
            try:
                # 폴더와 내부 파일 모두 삭제
                shutil.rmtree(target_path)
                print(f"🗑️ 삭제 완료: {run_folder}/{target}")
                total_deleted += 1
            except Exception as e:
                print(f"❌ {target_path} 삭제 중 오류 발생: {e}")

    print("-" * 50)
    print(f"✨ 총 {total_deleted}개의 하이브리드 결과 폴더를 정리했습니다.")
    print("🚀 이제 새로운 가중치 설정으로 'auto_hybrid_all.py'를 실행할 수 있습니다.")

if __name__ == "__main__":
    # 실행 전 사용자 확인 (실수 방지)
    confirm = input("❗ 모든 run 폴더 내의 하이브리드 분석 결과가 삭제됩니다. 계속하시겠습니까? (y/n): ")
    if confirm.lower() == 'y':
        reset_hybrid_folders()
    else:
        print("❌ 작업을 취소했습니다.")