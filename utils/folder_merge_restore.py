"""
하위 폴더 병합 및 복원 유틸리티

폴더병합복원v6.py의 핵심 로직을 BallonsTranslator에 내장하기 위해
Qt 환경에 맞게 재구성한 모듈입니다.

- flatten_directories: 하위 폴더의 파일들을 상위로 병합
- post_process_and_restore: 후처리(result 이동, 작업폴더 삭제) 후 복원
"""

import os
import re
import shutil
import threading
from pathlib import Path

# 라벨링에 사용될 문자 정의
LABEL_START = '『'
LABEL_END = '』'


def flatten_directories(root_path: Path, callback=None):
    """
    하위 폴더 내 파일들을 『폴더명』파일명 형식으로 상위 폴더에 병합합니다.
    빈 하위 폴더는 자동으로 삭제됩니다.

    Args:
        root_path: 대상 루트 폴더 경로
        callback: 진행 상황 로그를 전달할 콜백 함수 (str -> None)
    """
    def log(msg):
        if callback:
            callback(msg)

    subdirs = [d for d in root_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not subdirs:
        log("처리할 하위 폴더가 없습니다.")
        return 0

    log(f"총 {len(subdirs)}개의 폴더를 대상으로 파일 합치기를 시작합니다...")
    total_files_processed = 0

    for subdir_path in subdirs:
        dir_name = subdir_path.name
        processed_count = 0

        try:
            files_to_move = []
            for file_path in subdir_path.iterdir():
                if file_path.is_file():
                    new_name = f"{LABEL_START}{dir_name}{LABEL_END}{file_path.name}"
                    if len(new_name.encode('utf-8')) > 255:
                        log(f"[{dir_name}] [경고] 파일명이 너무 길어 건너뜁니다: {file_path.name}")
                        continue
                    destination_path = root_path / new_name
                    files_to_move.append((file_path, destination_path))

            for source_path, dest_path in files_to_move:
                source_path.rename(dest_path)
                processed_count += 1

            if processed_count > 0:
                log(f"[{dir_name}] {processed_count}개 파일을 상위 폴더로 이동했습니다.")
            else:
                log(f"[{dir_name}] 처리할 파일이 없습니다.")

            # 빈 폴더 삭제
            if not any(subdir_path.iterdir()):
                subdir_path.rmdir()
                log(f"[{dir_name}] 빈 폴더를 삭제했습니다.")

            total_files_processed += processed_count

        except Exception as e:
            log(f"[{dir_name}] 오류 발생: {e}")

    log(f"\n총 {total_files_processed}개의 파일을 처리했습니다.")
    return total_files_processed


def _process_single_folder_post(folder_path: str, callback=None):
    """
    단일 폴더에 대한 후처리를 수행합니다.
    - result 폴더의 내용을 상위로 이동
    - inpainted, mask, result 폴더 삭제
    - JSON 파일 삭제

    Args:
        folder_path: 처리할 폴더 경로 (문자열)
        callback: 로그 콜백 함수
    """
    def log(msg):
        if callback:
            callback(msg)

    folder_basename = os.path.basename(folder_path)
    log(f"[{folder_basename}] 후처리 시작...")
    all_steps_successful = True

    try:
        # 1. result 폴더 내용을 상위로 이동
        result_folder_path = os.path.join(folder_path, 'result')
        moved_items_count = 0

        if os.path.isdir(result_folder_path):
            items_in_result = os.listdir(result_folder_path)
            if not items_in_result:
                log(f"[{folder_basename}] 'result' 폴더는 비어있습니다.")
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
                                log(f"[{folder_basename}] 경고: '{item_name}' 폴더 이동 불가.")
                                continue
                            elif os.path.isdir(destination_item_path) and os.path.isfile(source_item_path):
                                log(f"[{folder_basename}] 경고: '{item_name}' 파일 이동 불가.")
                                continue
                        shutil.move(source_item_path, destination_item_path)
                        moved_items_count += 1
                    except Exception as e:
                        log(f"[{folder_basename}] 오류: 'result/{item_name}' 이동 중 - {e}")
                        all_steps_successful = False

                if moved_items_count > 0:
                    log(f"[{folder_basename}] 'result' 폴더에서 {moved_items_count}개 항목을 상위로 이동했습니다.")
        else:
            log(f"[{folder_basename}] 'result' 폴더가 없습니다. 파일 이동을 건너뜁니다.")

        # 2. 작업 폴더 삭제
        folders_to_delete = ['inpainted', 'mask', 'result']
        deleted_items_log = []
        for folder_name in folders_to_delete:
            path_to_delete = os.path.join(folder_path, folder_name)
            if os.path.isdir(path_to_delete):
                try:
                    shutil.rmtree(path_to_delete)
                    deleted_items_log.append(f"'{folder_name}' 폴더")
                except Exception as e:
                    log(f"[{folder_basename}] 오류: '{folder_name}' 폴더 삭제 중 - {e}")
                    all_steps_successful = False

        # 3. JSON 파일 삭제
        items_in_main_folder = os.listdir(folder_path)
        for item_name in items_in_main_folder:
            if item_name.lower().endswith('.json'):
                file_path_to_delete = os.path.join(folder_path, item_name)
                if os.path.isfile(file_path_to_delete):
                    try:
                        os.remove(file_path_to_delete)
                        deleted_items_log.append(f"'{item_name}' JSON 파일")
                    except Exception as e:
                        log(f"[{folder_basename}] 오류: '{item_name}' JSON 파일 삭제 중 - {e}")
                        all_steps_successful = False

        if deleted_items_log:
            log(f"[{folder_basename}] 삭제 완료: {', '.join(deleted_items_log)}.")

        if all_steps_successful:
            log(f"[{folder_basename}] 후처리 성공.")
        else:
            log(f"[{folder_basename}] 후처리 중 일부 오류 발생.")

    except Exception as e:
        log(f"[{folder_basename}] 후처리 중 예기치 않은 오류: {e}")

    return all_steps_successful


def _run_post_processing(root_path: Path, callback=None):
    """
    후처리를 실행합니다.
    - 선택된 폴더 자체에 result/inpainted/mask가 있으면 단일 대상으로 처리
    - 없으면 하위 폴더들을 각각 처리

    Args:
        root_path: 대상 루트 폴더 경로
        callback: 로그 콜백 함수
    """
    def log(msg):
        if callback:
            callback(msg)

    folders_to_process = []

    # 휴리스틱: 선택된 폴더 내에 result/inpainted/mask가 있으면 단일 대상
    is_single_job_folder = any((root_path / d).is_dir() for d in ['result', 'inpainted', 'mask'])

    if is_single_job_folder:
        folders_to_process.append(root_path)
        log(f"폴더 '{root_path.name}'를 직접 후처리 대상으로 설정합니다.")
    else:
        subdirs = [d for d in root_path.iterdir() if d.is_dir()]
        if not subdirs:
            log("후처리할 하위 폴더가 없습니다.")
            return
        folders_to_process.extend(subdirs)
        log(f"--- 총 {len(folders_to_process)}개 하위 폴더에 대한 후처리 시작 ---")

    if not folders_to_process:
        log("처리할 폴더를 찾지 못했습니다.")
        return

    # 스레드로 병렬 처리
    processing_results = []

    def target_for_thread(folder_path):
        success = _process_single_folder_post(str(folder_path), callback)
        processing_results.append({'folder_name': folder_path.name, 'success': success})

    threads = []
    for folder_p in folders_to_process:
        thread = threading.Thread(target=target_for_thread, args=(folder_p,))
        threads.append(thread)
        thread.start()

    for t in threads:
        t.join()

    # 결과 요약
    log("\n--- 후처리 작업 완료 ---")
    successful_count = sum(1 for r in processing_results if r['success'])
    failed_count = len(processing_results) - successful_count

    if processing_results:
        log("\n[후처리 결과 요약]")
        for result in sorted(processing_results, key=lambda x: x['folder_name']):
            status_symbol = "✔ 성공" if result['success'] else "✘ 실패"
            log(f"  폴더 '{result['folder_name']}': {status_symbol}")
        log(f"\n총 {len(processing_results)}개 폴더 중: 성공 {successful_count}개, 실패 {failed_count}개")


def _restore_directories(root_path: Path, callback=None):
    """
    라벨링된 파일들을 원래의 폴더 구조로 복원합니다.

    Args:
        root_path: 대상 루트 폴더 경로
        callback: 로그 콜백 함수
    """
    def log(msg):
        if callback:
            callback(msg)

    pattern = re.compile(re.escape(LABEL_START) + r"(.+?)" + re.escape(LABEL_END) + r"(.+)")

    log("파일 목록을 스캔하고 그룹화하는 중...")

    folders_to_restore = {}
    all_files = [f for f in root_path.iterdir() if f.is_file()]

    for file_path in all_files:
        match = pattern.match(file_path.name)
        if not match:
            continue

        try:
            dir_name = match.group(1)
            original_file_name = match.group(2)

            if not dir_name or not original_file_name:
                log(f"[경고] 파일명 형식이 잘못되어 건너뜁니다: {file_path.name}")
                continue

            if dir_name not in folders_to_restore:
                folders_to_restore[dir_name] = []

            folders_to_restore[dir_name].append((file_path, original_file_name))

        except Exception as e:
            log(f"[경고] 파일 '{file_path.name}' 처리 중 오류 발생: {e}")

    if not folders_to_restore:
        log("복원할 파일을 찾지 못했습니다.")
        return 0

    log(f"총 {len(folders_to_restore)}개의 폴더 그룹으로 분류 완료!")
    log("본격적인 파일 복원을 시작합니다...")

    total_files_processed = 0

    for dir_name, files_to_move in folders_to_restore.items():
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
                    log(f"[{dir_name}] '{source_path.name}' 이동 중 오류: {e}")

            if processed_count > 0:
                log(f"[{dir_name}] {processed_count}개 파일을 '{dir_name}' 폴더로 복원했습니다.")

        except Exception as e:
            log(f"[{dir_name}] 그룹 처리 중 오류 발생: {e}")

        total_files_processed += processed_count

    log(f"\n총 {total_files_processed}개의 파일을 복원했습니다.")
    return total_files_processed


def post_process_and_restore(root_path: Path, callback=None):
    """
    후처리 + 복원을 연속으로 실행합니다.

    1단계: 후처리 (result 폴더 이동, 작업 폴더/JSON 삭제)
    2단계: 복원 (『폴더명』파일명 → 원래 폴더로 이동)

    Args:
        root_path: 대상 루트 폴더 경로
        callback: 로그 콜백 함수
    """
    def log(msg):
        if callback:
            callback(msg)

    log(f"▶ [1단계] '{root_path.name}'의 후처리 작업을 시작합니다.")
    _run_post_processing(root_path, callback)
    log(f"\n▶ [2단계] '{root_path.name}'의 파일 복원 작업을 시작합니다.")
    _restore_directories(root_path, callback)
    log(f"\n✅ 후처리 + 복원 작업이 완료되었습니다.")
