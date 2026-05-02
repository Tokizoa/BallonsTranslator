import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
# Toplevel이 추가되었습니다.
from tkinter import Tk, Toplevel, filedialog, messagebox, Label, Button

# --- 추가된 부분 ---
import queue
from tkinter.scrolledtext import ScrolledText

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    class TkinterDnD:
        @staticmethod
        def Tk():
            return Tk()
    DND_FILES = None
    print("[경고] tkinterdnd2 라이브러리가 설치되지 않아 드래그 앤 드롭 기능을 사용할 수 없습니다.")
    print("         'pip install tkinterdnd2' 명령으로 설치할 수 있습니다.")
# --- 여기까지 ---


# --- 공통 및 유틸리티 함수 ---

# 라벨링에 사용될 문자 정의
LABEL_START = '『'
LABEL_END = '』'

# print 출력을 위한 Lock 객체 (스레드 간 메시지 섞임 방지)
print_lock = threading.Lock()

# GUI 로그 업데이트를 위한 전역 큐
log_queue = None

def log_message(message, folder_context=""):
    """
    스레드 안전하게 메시지를 CLI, 로그 파일, GUI에 기록하는 함수
    """
    # 로그 파일 이름 설정 (스크립트와 같은 위치에 생성)
    log_file_path = os.path.join(os.path.dirname(sys.argv[0]), "app_log.txt")

    with print_lock:
        if folder_context:
            # 현재 처리 중인 폴더명을 메시지 앞에 붙여서 출력
            log_line = f"[{os.path.basename(folder_context)}] {message}"
        else:
            log_line = message
        
        # 1. 콘솔에 출력 (python.exe로 실행 시 보임)
        print(log_line)

        # 2. 파일에 기록 (pythonw.exe로 실행 시 확인용)
        try:
            with open(log_file_path, "a", encoding="utf-8") as f:
                # 타임스탬프 추가하여 로그의 가독성 높임
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                f.write(f"{timestamp} - {log_line}\n")
        except Exception as e:
            # 파일 쓰기 오류는 콘솔에만 출력
            print(f"[CRITICAL] 로그 파일 작성 중 오류 발생: {e}")

        # 3. GUI에 표시하기 위해 큐에 추가 (큐가 초기화된 경우)
        if log_queue:
            log_queue.put(log_line)

def get_cpu_count():
    """사용할 CPU 코어 수를 결정합니다. (최대 4개)"""
    cpu_count = os.cpu_count()
    return min(cpu_count if cpu_count else 1, 4)

def select_directory():
    """
    GUI를 통해 폴더를 선택하고 경로를 반환합니다.
    """
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="작업할 폴더를 선택하세요")
    root.destroy()
    if folder_path:
        return Path(folder_path)
    return None

# --- 기능 3: 폴더 후처리 ---

def process_single_folder(folder_path):
    folder_basename_for_log = os.path.basename(folder_path)
    log_message(f"처리 시작...", folder_basename_for_log)
    all_steps_successful = True
    error_messages =[]
    try:
        result_folder_path = os.path.join(folder_path, 'result')
        moved_items_count = 0
        if os.path.isdir(result_folder_path):
            items_in_result = os.listdir(result_folder_path)
            if not items_in_result:
                log_message(f"'result' 폴더는 비어있습니다.", folder_basename_for_log)
            else:
                for item_name in items_in_result:
                    source_item_path = os.path.join(result_folder_path, item_name)
                    destination_item_path = os.path.join(folder_path, item_name)
                    try:
                        if os.path.exists(destination_item_path):
                            if os.path.isfile(destination_item_path) and os.path.isfile(source_item_path):
                                os.remove(destination_item_path)
                            elif os.path.isdir(destination_item_path) and os.path.isdir(source_item_path):
                                shutil.rmtree(destination_item_path)
                            elif os.path.isfile(destination_item_path) and os.path.isdir(source_item_path):
                                log_message(f"경고: '{item_name}' 폴더 이동 불가. 대상 위치에 동일 이름의 '파일'이 존재합니다.", folder_basename_for_log)
                                continue
                            elif os.path.isdir(destination_item_path) and os.path.isfile(source_item_path):
                                log_message(f"경고: '{item_name}' 파일 이동 불가. 대상 위치에 동일 이름의 '폴더'가 존재합니다.", folder_basename_for_log)
                                continue
                        shutil.move(source_item_path, destination_item_path)
                        moved_items_count += 1
                    except Exception as e:
                        msg = f"오류: 'result/{item_name}' 이동 중 - {e}"
                        log_message(msg, folder_basename_for_log)
                        error_messages.append(msg)
                        all_steps_successful = False
                if moved_items_count > 0:
                    log_message(f"'result' 폴더에서 {moved_items_count}개 항목을 상위로 이동했습니다.", folder_basename_for_log)
        else:
            log_message(f"'result' 폴더가 없습니다. 파일 이동을 건너뜁니다.", folder_basename_for_log)
        folders_to_delete =['inpainted', 'mask', 'result']
        deleted_items_log =[]
        for folder_name in folders_to_delete:
            path_to_delete = os.path.join(folder_path, folder_name)
            if os.path.isdir(path_to_delete):
                try:
                    shutil.rmtree(path_to_delete)
                    deleted_items_log.append(f"'{folder_name}' 폴더")
                except Exception as e:
                    msg = f"오류: '{folder_name}' 폴더 삭제 중 - {e}"
                    log_message(msg, folder_basename_for_log)
                    error_messages.append(msg)
                    all_steps_successful = False
        items_in_main_folder = os.listdir(folder_path)
        for item_name in items_in_main_folder:
            if item_name.lower().endswith('.json'):
                file_path_to_delete = os.path.join(folder_path, item_name)
                if os.path.isfile(file_path_to_delete):
                    try:
                        os.remove(file_path_to_delete)
                        deleted_items_log.append(f"'{item_name}' JSON 파일")
                    except Exception as e:
                        msg = f"오류: '{item_name}' JSON 파일 삭제 중 - {e}"
                        log_message(msg, folder_basename_for_log)
                        error_messages.append(msg)
                        all_steps_successful = False
        if deleted_items_log:
            log_message(f"삭제 완료: {', '.join(deleted_items_log)}.", folder_basename_for_log)
        if all_steps_successful:
            log_message(f"처리 성공.", folder_basename_for_log)
            return True, f"'{folder_basename_for_log}' 처리 성공."
        else:
            log_message(f"처리 실패. (오류 수: {len(error_messages)})", folder_basename_for_log)
            return False, f"'{folder_basename_for_log}' 처리 중 오류 발생. 상세 로그 확인."
    except Exception as e:
        critical_error_msg = f"처리 중 예기치 않은 심각한 오류 발생: {e}"
        log_message(f"오류: {critical_error_msg}", folder_basename_for_log)
        return False, f"'{folder_basename_for_log}' 처리 실패: {critical_error_msg}"

def run_post_processing(root_path: Path):
    folders_to_process =[]
    # 휴리스틱: 선택된 폴더 내에 'result', 'inpainted', 'mask' 같은 폴더가 있으면,
    # 해당 폴더 자체가 단일 처리 대상일 가능성이 높다고 판단합니다.
    is_single_job_folder = any((root_path / d).is_dir() for d in['result', 'inpainted', 'mask'])

    if is_single_job_folder:
        # 단일 처리 대상 폴더로 판단되면, 해당 폴더만 목록에 추가합니다.
        folders_to_process.append(root_path)
        log_message(f"폴더 '{root_path.name}'를 직접 후처리 대상으로 설정합니다.")
    else:
        # 그렇지 않으면, 하위 폴더들을 처리 대상으로 삼습니다.
        subdirs =[d for d in root_path.iterdir() if d.is_dir()]
        if not subdirs:
            log_message("후처리할 하위 폴더가 없습니다.")
            return
        folders_to_process.extend(subdirs)
        log_message(f"--- 총 {len(folders_to_process)}개 하위 폴더에 대한 후처리 시작 ---")
    
    if not folders_to_process:
        log_message("처리할 폴더를 찾지 못했습니다.")
        return

    threads =[]
    processing_results =[]
    
    def target_for_thread(folder_path):
        success, message = process_single_folder(str(folder_path))
        processing_results.append({'folder_name': folder_path.name, 'success': success, 'message': message})

    for folder_p in folders_to_process:
        thread = threading.Thread(target=target_for_thread, args=(folder_p,))
        threads.append(thread)
        thread.start()

    for t in threads:
        t.join()

    log_message("\n--- 후처리 작업 완료 ---")
    successful_count = sum(1 for r in processing_results if r['success'])
    failed_count = len(processing_results) - successful_count
    
    if processing_results:
        log_message("\n[최종 처리 결과 요약]")
        for result in sorted(processing_results, key=lambda x: x['folder_name']):
            status_symbol = "✔ 성공" if result['success'] else "✘ 실패"
            log_message(f"  폴더 '{result['folder_name']}': {status_symbol}")
        log_message(f"\n총 {len(processing_results)}개 폴더 중: 성공 {successful_count}개, 실패 {failed_count}개")
    else:
        log_message("처리된 폴더가 없습니다.")

# --- 기능 2: 폴더 병합 ---

def process_flatten_task(subdir_path: Path, root_path: Path):
    logs =[]
    processed_files_count = 0
    dir_name = subdir_path.name
    try:
        files_to_move =[]
        for file_path in subdir_path.iterdir():
            if file_path.is_file():
                new_name = f"{LABEL_START}{dir_name}{LABEL_END}{file_path.name}"
                if len(new_name.encode('utf-8')) > 255:
                    logs.append(f"[경고] 파일명이 너무 길어 건너뜁니다: {file_path.name}")
                    continue
                destination_path = root_path / new_name
                files_to_move.append((file_path, destination_path))

        for source_path, dest_path in files_to_move:
            source_path.rename(dest_path)
            processed_files_count += 1
        
        if not files_to_move:
             logs.append(f"처리할 파일이 없습니다.")
        else:
             logs.append(f"{processed_files_count}개 파일을 상위 폴더로 이동했습니다.")

        if not any(subdir_path.iterdir()):
            subdir_path.rmdir()
            logs.append("빈 폴더를 삭제했습니다.")
            
    except Exception as e:
        logs.append(f"오류 발생: {e}")
        return (logs, 0)
    
    return (logs, processed_files_count)

def flatten_directories(root_path: Path):
    subdirs =[d for d in root_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not subdirs:
        log_message("처리할 하위 폴더가 없습니다.")
        return
    log_message(f"총 {len(subdirs)}개의 폴더를 대상으로 파일 합치기를 시작합니다...")
    start_time = time.time()
    total_files_processed = 0
    with ProcessPoolExecutor(max_workers=get_cpu_count()) as executor:
        futures = {executor.submit(process_flatten_task, subdir, root_path): subdir for subdir in subdirs}
        for future in as_completed(futures):
            subdir_path = futures[future]
            try:
                logs, processed_count = future.result()
                total_files_processed += processed_count
                for log_entry in logs:
                    log_message(log_entry, subdir_path.name)
            except Exception as e:
                log_message(f"처리 중 예측하지 못한 오류: {e}", subdir_path.name)

    end_time = time.time()
    log_message(f"\n총 {total_files_processed}개의 파일을 처리했습니다.")
    log_message(f"작업 완료! (소요 시간: {end_time - start_time:.2f}초)")

# --- 기능 4: 폴더 복원 ---

def process_restore_group_task(dir_name: str, files_to_move: list, root_path: Path):
    """
    하나의 폴더 그룹에 대한 복원 작업을 처리합니다.
    폴더를 한 번 생성하고, 해당 폴더에 속한 모든 파일을 이동시킵니다.
    """
    logs =[]
    processed_count = 0
    try:
        target_dir_path = root_path / dir_name
        target_dir_path.mkdir(exist_ok=True)

        for source_path, original_file_name in files_to_move:
            try:
                destination_path = target_dir_path / original_file_name
                source_path.rename(destination_path)
                processed_count += 1
            except Exception as e:
                logs.append(f"'{source_path.name}' 이동 중 오류: {e}")
        
        if processed_count > 0:
            logs.append(f"{processed_count}개 파일을 '{dir_name}' 폴더로 복원했습니다.")

    except Exception as e:
        logs.append(f"'{dir_name}' 그룹 처리 중 심각한 오류 발생: {e}")

    return (logs, processed_count)


def restore_directories(root_path: Path):
    """
    라벨링된 파일들을 원래의 폴더 구조로 복원합니다.
    파일을 폴더별로 그룹화하여 병렬 처리 효율을 극대화합니다.
    """
    pattern = re.compile(re.escape(LABEL_START) + r"(.+?)" + re.escape(LABEL_END) + r"(.+)")
    
    log_message("파일 목록을 스캔하고 그룹화하는 중...")
    start_time = time.time()

    folders_to_restore = {}
    all_files =[f for f in root_path.iterdir() if f.is_file()]
    
    for file_path in all_files:
        match = pattern.match(file_path.name)
        if not match:
            continue
        
        try:
            dir_name = match.group(1)
            original_file_name = match.group(2)
            
            # 잘못된 파일명 형식 예외 처리
            if not dir_name or not original_file_name:
                log_message(f"[경고] 파일명 형식이 잘못되어 건너뜁니다: {file_path.name}")
                continue

            if dir_name not in folders_to_restore:
                folders_to_restore[dir_name] = []
            
            folders_to_restore[dir_name].append((file_path, original_file_name))

        except Exception as e:
            log_message(f"[경고] 파일 '{file_path.name}' 처리 중 오류 발생: {e}")

    if not folders_to_restore:
        log_message("복원할 파일을 찾지 못했습니다.")
        return

    grouping_end_time = time.time()
    log_message(f"총 {len(folders_to_restore)}개의 폴더 그룹으로 분류 완료! (소요 시간: {grouping_end_time - start_time:.2f}초)")
    log_message(f"본격적인 파일 복원을 시작합니다...")

    total_files_processed = 0
    with ProcessPoolExecutor(max_workers=get_cpu_count()) as executor:
        # 작업 제출
        futures = {
            executor.submit(process_restore_group_task, dir_name, files, root_path): dir_name 
            for dir_name, files in folders_to_restore.items()
        }

        for future in as_completed(futures):
            dir_name = futures[future]
            try:
                logs, processed_count = future.result()
                total_files_processed += processed_count
                for log_entry in logs:
                    log_message(log_entry, dir_name)
            except Exception as e:
                log_message(f"'{dir_name}' 그룹 처리 중 예측하지 못한 오류: {e}", dir_name)

    end_time = time.time()
    log_message(f"\n총 {total_files_processed}개의 파일을 복원했습니다.")
    log_message(f"작업 완료! (총 소요 시간: {end_time - start_time:.2f}초)")

# --- 신규 기능: 후처리 + 복원 (원터치) ---
def post_process_and_restore(root_path: Path):
    log_message(f"▶[1단계] '{root_path.name}'의 후처리 작업을 시작합니다.")
    run_post_processing(root_path)
    log_message(f"\n▶ [2단계] '{root_path.name}'의 파일 복원 작업을 시작합니다.")
    restore_directories(root_path)


# --- 메인 GUI 실행 함수 ---

def main_gui():
    app_state = {'target_paths':[], 'is_working': False}

    # --- GUI Setup ---
    if DND_FILES:
        window = TkinterDnD.Tk()
    else:
        window = Tk()
        
    window.title("파일 및 폴더 관리 프로그램 v3.1 (다중폴더 & 토스트 알림)")
    window.geometry("600x650")
    window.resizable(True, True)

    # --- 전역 로그 큐 초기화 ---
    global log_queue
    log_queue = queue.Queue()

    # --- GUI 위젯 ---
    title_label = Label(window, text="파일 및 폴더 관리 프로그램", font=("Helvetica", 16, "bold"))
    title_label.pack(pady=(15, 10))

    path_label = Label(window, text="▶ 현재 대상 폴더: 0개 선택됨", font=("Helvetica", 10), fg="red")
    path_label.pack(pady=(0, 5))
    
    info_label = Label(window, text="폴더를 선택하거나 이 창으로 여러 폴더를 드래그 앤 드롭하세요.", font=("Helvetica", 9), fg="gray")
    info_label.pack(pady=(0, 15))

    button_frame = Label(window)
    button_frame.pack(fill='x', padx=20, pady=5)

    log_label = Label(window, text="진행 상황 로그")
    log_label.pack(pady=(10, 0))
    log_widget = ScrolledText(window, height=15, font=("Consolas", 9), wrap='word', borderwidth=1, relief="solid")
    log_widget.pack(pady=5, padx=20, fill='both', expand=True)
    log_widget.config(state='disabled')

    def show_toast(message, duration=3000):
        """우측 하단에 나타나는 토스트 알림 창"""
        toast = Toplevel(window)
        toast.overrideredirect(True) # 타이틀 바 제거
        toast.attributes('-topmost', True) # 항상 위
        
        lbl = Label(toast, text=message, bg="#4CAF50", fg="white", font=("Helvetica", 10, "bold"), padx=15, pady=10)
        lbl.pack()
        
        toast.update_idletasks()
        w = toast.winfo_width()
        h = toast.winfo_height()
        
        # 화면 우측 하단 (작업 표시줄 고려)
        sw = toast.winfo_screenwidth()
        sh = toast.winfo_screenheight()
        x = sw - w - 20
        y = sh - h - 60
        
        toast.geometry(f"{w}x{h}+{x}+{y}")
        
        # 페이드 아웃 애니메이션
        def fade_out():
            try:
                alpha = toast.attributes("-alpha")
                if alpha > 0.05:
                    alpha -= 0.05
                    toast.attributes("-alpha", alpha)
                    window.after(40, fade_out)
                else:
                    toast.destroy()
            except:
                # 에러(플랫폼 미지원 등)가 발생하면 그냥 삭제
                toast.destroy()
                
        window.after(duration, fade_out)

    def update_log_text():
        """ 큐에서 로그 메시지를 가져와 GUI에 표시 """
        try:
            while not log_queue.empty():
                log_line = log_queue.get_nowait()
                log_widget.config(state='normal')
                log_widget.insert('end', log_line + '\n')
                log_widget.see('end')
                log_widget.config(state='disabled')
        finally:
            window.after(100, update_log_text)

    def update_path_label():
        paths = app_state['target_paths']
        if not paths:
            path_label.config(text="▶ 현재 대상 폴더: 지정되지 않음", fg="red")
        elif len(paths) == 1:
            display_path = str(paths[0])
            if len(display_path) > 60:
                display_path = "..." + display_path[-57:]
            path_label.config(text=f"▶ 현재 대상 폴더: {display_path}", fg="blue")
        else:
            path_label.config(text=f"▶ 현재 대상 폴더: {len(paths)}개 폴더 선택됨", fg="blue")
            log_message(f"--- 현재 선택된 폴더 목록 ({len(paths)}개) ---")
            for p in paths:
                log_message(f" * {p.name}")

    def set_target_paths(paths_data, append=False):
        if not append:
            app_state['target_paths'].clear()

        # DND로 들어온 문자열 분리 처리
        if isinstance(paths_data, str):
            try:
                parsed_paths = window.tk.splitlist(paths_data)
            except Exception:
                parsed_paths = [paths_data]
        elif isinstance(paths_data, (list, tuple)):
            parsed_paths = paths_data
        else:
            parsed_paths = [paths_data]

        added_count = 0
        for p in parsed_paths:
            p_str = str(p).strip()
            # 윈도우 경로에 감싸진 {} 제거
            if p_str.startswith('{') and p_str.endswith('}'):
                p_str = p_str[1:-1]
            
            path_obj = Path(p_str)
            if path_obj.is_dir():
                if path_obj not in app_state['target_paths']:
                    app_state['target_paths'].append(path_obj)
                    added_count += 1
            else:
                log_message(f"[경고] 폴더가 아니거나 존재하지 않는 경로: {p_str}")

        if added_count > 0:
            log_message(f"{added_count}개의 대상 폴더가 추가되었습니다.")
        update_path_label()

    def handle_select_directory():
        selected_path = select_directory()
        if selected_path:
            set_target_paths([selected_path], append=True)

    def clear_directories():
        app_state['target_paths'].clear()
        log_message("선택된 폴더 목록이 초기화되었습니다.")
        update_path_label()

    def on_drop(event):
        # 드래그 앤 드롭 시 목록에 폴더 누적 추가
        set_target_paths(event.data, append=True)

    def toggle_buttons_state(is_enabled):
        for child in button_frame.winfo_children():
            if isinstance(child, Button):
                child.config(state='normal' if is_enabled else 'disabled')

    def run_task_in_thread(task_function, task_name):
        if not app_state['target_paths']:
            messagebox.showwarning("경고", "먼저 하나 이상의 작업 대상 폴더를 추가해주세요.")
            return
        if app_state['is_working']:
            messagebox.showwarning("알림", "현재 다른 작업이 실행 중입니다.")
            return

        def task_wrapper():
            app_state['is_working'] = True
            toggle_buttons_state(False)
            log_message(f"\n{'='*30}\n'{task_name}' 작업을 시작합니다. (총 {len(app_state['target_paths'])}개 폴더)\n{'='*30}")
            
            try:
                # 선택된 모든 폴더를 순회하며 독립적으로 작업을 수행합니다.
                for idx, target_path in enumerate(app_state['target_paths'], 1):
                    log_message(f"\n>>>[{idx}/{len(app_state['target_paths'])}] 대상: {target_path}")
                    task_function(target_path)
                
                log_message(f"\n{'='*30}\n모든 대상 폴더에 대한 '{task_name}' 작업이 완료되었습니다.\n{'='*30}")
                
                # 기존의 messagebox를 토스트 알림으로 교체
                window.after(0, lambda: show_toast(f"✅ '{task_name}' 작업이 완료되었습니다.", 3500))
                
            except Exception as e:
                error_msg = f"[치명적 오류] 작업 중 예외 발생: {e}"
                log_message(error_msg)
                # 에러는 사용자가 확실히 인지해야 하므로 팝업창 유지
                window.after(0, lambda: messagebox.showerror("오류 발생", f"작업 중 심각한 오류가 발생했습니다:\n{e}"))
            finally:
                app_state['is_working'] = False
                toggle_buttons_state(True)
        
        threading.Thread(target=task_wrapper, daemon=True).start()

    # --- 위젯 배치 (다중 폴더 및 신규 버튼 반영) ---
    btn_pad = {'padx': 5, 'pady': 5}
    
    # 0행: 폴더 지정 및 초기화
    Button(button_frame, text="➕ 0. 폴더 추가하기", command=handle_select_directory, bg="#E6F2FF").grid(row=0, column=0, sticky="ew", **btn_pad)
    Button(button_frame, text="🔄 목록 초기화", command=clear_directories, bg="#FFF2E6").grid(row=0, column=1, sticky="ew", **btn_pad)
    
    # 1행: 하위폴더 병합, 후처리
    Button(button_frame, text="1. 하위 폴더 병합", command=lambda: run_task_in_thread(flatten_directories, "폴더 병합")).grid(row=1, column=0, sticky="ew", **btn_pad)
    Button(button_frame, text="2. 후처리만 하기", command=lambda: run_task_in_thread(run_post_processing, "폴더 후처리")).grid(row=1, column=1, sticky="ew", **btn_pad)
    
    # 2행: 복원, 후처리+복원 원터치
    Button(button_frame, text="3. 폴더 복원만 하기", command=lambda: run_task_in_thread(restore_directories, "폴더 복원")).grid(row=2, column=0, sticky="ew", **btn_pad)
    Button(button_frame, text="⭐ 4. 후처리 후 복원 (원터치)", command=lambda: run_task_in_thread(post_process_and_restore, "후처리 및 복원 통합"), bg="#E6FFE6").grid(row=2, column=1, sticky="ew", **btn_pad)
    
    button_frame.columnconfigure(0, weight=1)
    button_frame.columnconfigure(1, weight=1)

    Button(window, text="프로그램 종료 (Q)", command=window.quit, bg="#FFDDDD").pack(side="bottom", pady=20, ipadx=10)
    
    # --- 초기화 ---
    if len(sys.argv) > 1:
        # sys.argv를 통해 폴더 여러 개가 단축키 등으로 넘겨질 경우를 지원
        set_target_paths(sys.argv[1:], append=True)
    
    if DND_FILES:
        window.drop_target_register(DND_FILES)
        window.dnd_bind('<<Drop>>', on_drop)

    update_log_text()
    window.mainloop()


if __name__ == '__main__':
    main_gui()