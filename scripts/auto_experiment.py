import os
import random
import subprocess
import argparse
from multiprocessing import Manager, Process
from datetime import datetime
import time
import re

# [설정]
NUM_ITERATIONS = 1
MAX_WORKERS = 4
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "..", "data")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

def run_pipeline(seed, status_dict):
    status_dict[seed] = {"Cleaning": "대기", "Prep": "대기", "CSVM": "대기", "QSVM": "대기", "ErrorLog": ""}
    steps = [
        ("Cleaning", "auto_cleaning_btc_data.py"),
        ("Prep", "auto_prepare_quantum_data.py"),
        ("CSVM", "auto_classical_svm_baseline.py"),
        ("QSVM", "auto_quantum_kernel_svm.py")
    ]
    
    try:
        for step_name, script_name in steps:
            temp_status = status_dict[seed]
            temp_status[step_name] = "🔄 실행 중"
            status_dict[seed] = temp_status
            
            script_path = os.path.join(SRC_DIR, script_name)
            
            # stderr를 stdout으로 통합하여 모든 에러 메시지를 캡처
            process = subprocess.Popen(
                ["python", "-u", script_path, "--seed", str(seed)],
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1,
                errors='ignore'
            )

            last_output = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None: break
                if line:
                    line_strip = line.strip()
                    last_output.append(line_strip)
                    if len(last_output) > 10: last_output.pop(0) # 마지막 10줄만 유지

                    match = re.search(r".*: (\d+/\d+) \((\d+\.\d+)%\)", line)
                    if match:
                        temp_status = status_dict[seed]
                        temp_status[step_name] = f"⏳ {match.group(2)}%"
                        status_dict[seed] = temp_status

            if process.returncode == 0:
                temp_status = status_dict[seed]
                temp_status[step_name] = "✅ 완료"
                status_dict[seed] = temp_status
            else:
                # 에러 발생 시 해당 단계와 마지막 출력을 저장
                temp_status = status_dict[seed]
                temp_status[step_name] = "❌ 에러"
                temp_status["ErrorLog"] = f"[{step_name} 실패] " + " | ".join(last_output[-3:])
                status_dict[seed] = temp_status
                return
    except Exception as e:
        temp_status = status_dict[seed]
        temp_status["ErrorLog"] = f"[예외 발생] {str(e)}"
        status_dict[seed] = temp_status

def monitor_display(status_dict, target_seeds, log_path, stop_event):
    start_time = time.time()
    while not stop_event.is_set():
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            
            dashboard = "="*70 + "\n"
            dashboard += f" 🚀 양자 머신러닝 실험 실시간 대시보드 (경과 시간: {elapsed})\n"
            dashboard += "="*70 + "\n"
            
            all_done = True
            for seed in target_seeds:
                info = status_dict.get(seed, {})
                dashboard += f"Seed {seed:5} :\n"
                dashboard += f"  - Data Prep     : {info.get('Cleaning', '대기')} / {info.get('Prep', '대기')}\n"
                dashboard += f"  - Classical SVM : {info.get('CSVM', '대기')}\n"
                dashboard += f"  - Quantum SVM   : {info.get('QSVM', '대기')}\n"
                dashboard += "-" * 35 + "\n"
                
                # 하나라도 진행 중이면 완료 아님
                steps_status = [info.get(s) for s in ["Cleaning", "Prep", "CSVM", "QSVM"]]
                if not all(s == "✅ 완료" or s == "❌ 에러" for s in steps_status):
                    all_done = False

            print(dashboard)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(dashboard)
            
            if all_done: break
            time.sleep(1)
        except (BrokenPipeError, EOFError): break

def main():
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR, exist_ok=True)
    existing_folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
    target_seeds = []
    while len(target_seeds) < NUM_ITERATIONS:
        seed = random.randint(0, 10000)
        if f"run_{seed}" not in existing_folders and seed not in target_seeds:
            target_seeds.append(seed)

    log_dir = os.path.join(PROJECT_ROOT, "logs")
    if not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)

    log_filename = f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join(log_dir, log_filename)

    with Manager() as manager:
        status_dict = manager.dict()
        stop_event = manager.Event()
        
        monitor_p = Process(target=monitor_display, args=(status_dict, target_seeds, log_path, stop_event))
        monitor_p.start()

        processes = []
        for seed in target_seeds:
            p = Process(target=run_pipeline, args=(seed, status_dict))
            processes.append(p)

        for i in range(0, len(processes), MAX_WORKERS):
            chunk = processes[i : i + MAX_WORKERS]
            for p in chunk: p.start()
            for p in chunk: p.join()
        
        stop_event.set()
        monitor_p.join(timeout=2)

        # --- [추가] 에러 구체적 리포트 ---
        print("\n" + "!"*70)
        print(" 🔍 에러 발생 시드 리포트")
        print("!"*70)
        error_found = False
        for seed in target_seeds:
            info = status_dict.get(seed, {})
            if info.get("ErrorLog"):
                error_found = True
                print(f"❌ Seed {seed:5} 실패 사유: {info['ErrorLog']}")
        
        if not error_found:
            print("✅ 모든 시드가 성공적으로 완료되었습니다.")
        print("!"*70)

    print(f"\n최종 리포트가 {log_filename}에 저장되었습니다.")

if __name__ == "__main__":
    main()