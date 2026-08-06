from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from types import MethodType, SimpleNamespace
from typing import Callable, Dict, Optional


RENDER_WORKER_COUNT = 2
RENDER_STALL_TIMEOUT_SECONDS = 120.0
RENDER_RETRY_LIMIT = 1
RENDER_START_TIMEOUT_SECONDS = 30.0
_BLOCK_IPC_EXCLUDED_FIELDS = {"region_mask", "region_inpaint_dict"}


def encode_text_blocks(blocks) -> str:
    from utils.proj_imgtrans import TextBlkEncoder
    from utils.textblock import TextBlock

    field_names = set(TextBlock.__dataclass_fields__) - _BLOCK_IPC_EXCLUDED_FIELDS
    payload = [
        {name: value for name, value in vars(block).items() if name in field_names}
        for block in blocks
    ]
    return json.dumps(payload, ensure_ascii=False, cls=TextBlkEncoder)


@dataclass
class RenderRequest:
    run_id: str
    request_id: str
    page_key: str
    input_path: str
    output_path: str
    quality: int
    blocks_json: str
    settings: Dict[str, object]

    def to_message(self) -> dict:
        return {"command": "render", "request": asdict(self)}


@dataclass
class RenderResult:
    success: bool
    page_key: str
    blocks_json: Optional[str] = None
    timings: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    cancelled: bool = False
    attempts: int = 1


class _WorkerProject:
    def __init__(self, page_key, input_path, blocks, image_array):
        self.pages = {page_key: blocks}
        self.directory = os.path.dirname(input_path)
        self.current_img = page_key
        self.img_array = image_array
        self._v4_save_lock = threading.RLock()
        self._input_path = input_path

    def get_inpainted_path(self, _page_key):
        return self._input_path


def _apply_worker_settings(settings):
    from utils import shared
    from utils.config import pcfg

    shared.LDPI = float(settings.get("ldpi", shared.LDPI))
    shared.DEFAULT_FONT_FAMILY = str(
        settings.get("default_font_family", shared.DEFAULT_FONT_FAMILY)
    )
    shared.APP_DEFAULT_FONT = str(
        settings.get("app_default_font", shared.APP_DEFAULT_FONT)
    )
    pcfg.let_autolayout_flag = bool(settings.get("let_autolayout_flag", True))
    pcfg.let_fntsize_flag = int(settings.get("let_fntsize_flag", 0))
    pcfg.module.translate_source = settings.get(
        "translate_source", pcfg.module.translate_source
    )
    pcfg.module.translate_target = settings.get(
        "translate_target", pcfg.module.translate_target
    )


def _initialize_qt_worker(settings):
    from qtpy.QtGui import QFont, QFontDatabase, QGuiApplication
    from qtpy.QtWidgets import QApplication
    from utils import shared

    app = QApplication.instance() or QApplication([])
    _apply_worker_settings(settings)

    font_root = os.path.join(shared.PROGRAM_PATH, "fonts")
    if os.path.isdir(font_root):
        for root, _dirs, files in os.walk(font_root):
            for filename in files:
                if filename.lower().endswith((".ttf", ".otf", ".ttc", ".pfb")):
                    QFontDatabase.addApplicationFont(os.path.join(root, filename))

    app_font = QFont(shared.APP_DEFAULT_FONT)
    if not app_font.exactMatch():
        app_font = app.font()
    app_font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    app_font.setStyleStrategy(
        QFont.StyleStrategy.PreferAntialias
        | QFont.StyleStrategy.NoSubpixelAntialias
    )
    QGuiApplication.setFont(app_font)
    return app


def _temporary_output_path(request: RenderRequest, worker_id: int) -> str:
    directory, filename = os.path.split(request.output_path)
    stem, extension = os.path.splitext(filename)
    return os.path.join(
        directory,
        f".{stem}.v4-{request.run_id}-{request.request_id}-w{worker_id}.tmp{extension}",
    )


def _render_request(connection, worker_id: int, request: RenderRequest):
    from utils.io_utils import imread
    from utils.textblock import TextBlock

    temporary_path = _temporary_output_path(request, worker_id)
    page_started = time.perf_counter()
    try:
        _apply_worker_settings(request.settings)
        connection.send(
            {
                "type": "heartbeat",
                "page_key": request.page_key,
                "stage": "load",
            }
        )
        block_dicts = json.loads(request.blocks_json)
        blocks = [TextBlock(**block_dict) for block_dict in block_dicts]
        image_array = imread(request.input_path)
        if image_array is None:
            raise RuntimeError(f"Failed to load render input: {request.input_path}")

        # Import and install the exact V4 layout/paint behavior in this isolated
        # process. ui.module_manager is deliberately not imported, so pipeline
        # monkey patches are not installed in the worker.
        from ui import scenetext_manager
        from modules.translators import trans_llm_api_v4 as v4

        v4._V4_STOP_REQUESTED = False
        v4._install_patches()

        project = _WorkerProject(
            request.page_key,
            request.input_path,
            blocks,
            image_array,
        )
        layout_context = SimpleNamespace(
            imgtrans_proj=project,
            pairwidget_list=[],
            auto_textlayout_flag=True,
        )
        layout_context.layout_textblk = MethodType(
            scenetext_manager.SceneTextManager.layout_textblk,
            layout_context,
        )
        layout_context.restore_charfmts = MethodType(
            scenetext_manager.SceneTextManager.restore_charfmts,
            layout_context,
        )

        def heartbeat(stage, page_key, block_index=None, block_count=None):
            connection.send(
                {
                    "type": "heartbeat",
                    "page_key": page_key,
                    "stage": stage,
                    "block_index": block_index,
                    "block_count": block_count,
                }
            )

        helper = v4.UIHelper(
            None,
            project,
            layout_context,
            total_pages=1,
            translate_thread=SimpleNamespace(stop_requested=False),
            heartbeat_callback=heartbeat,
        )
        connection.send(
            {
                "type": "heartbeat",
                "page_key": request.page_key,
                "stage": "area_prepare",
            }
        )
        prepared_page = v4._prepare_layout_page(
            project,
            request.page_key,
            bool(request.settings.get("let_autolayout_flag", True)),
        )
        with helper.prepared_pages_lock:
            helper.prepared_pages[request.page_key] = prepared_page
        helper.render_page_task(request.page_key)
        image = helper.rendered_images.pop(request.page_key, None)
        render_seconds = helper.render_timings.pop(
            request.page_key,
            time.perf_counter() - page_started,
        )
        if image is None or image.isNull():
            raise RuntimeError("Qt renderer returned a null image.")

        rendered_blocks_json = encode_text_blocks(project.pages[request.page_key])
        connection.send(
            {
                "type": "layout_complete",
                "page_key": request.page_key,
                "blocks_json": rendered_blocks_json,
                "render_seconds": render_seconds,
            }
        )
        connection.send(
            {
                "type": "heartbeat",
                "page_key": request.page_key,
                "stage": "save",
            }
        )

        os.makedirs(os.path.dirname(request.output_path), exist_ok=True)
        save_started = time.perf_counter()
        if not image.save(temporary_path, quality=request.quality):
            raise RuntimeError(f"QImage save failed: {temporary_path}")
        os.replace(temporary_path, request.output_path)
        save_seconds = time.perf_counter() - save_started
        connection.send(
            {
                "type": "save_complete",
                "page_key": request.page_key,
                "blocks_json": rendered_blocks_json,
                "timings": {
                    "area_prepare": prepared_page.get("prepare_seconds", 0.0),
                    "qt_render": render_seconds,
                    "image_save": save_seconds,
                    "worker_total": time.perf_counter() - page_started,
                },
            }
        )
    except BaseException as exc:
        try:
            connection.send(
                {
                    "type": "failed",
                    "page_key": request.page_key,
                    "error": f"{exc}\n{traceback.format_exc()}",
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass


def _worker_main(connection, worker_id: int, initial_settings: dict):
    try:
        import faulthandler
        faulthandler.enable()
        app = _initialize_qt_worker(initial_settings)
        connection.send({"type": "ready", "worker_id": worker_id})
        while True:
            try:
                message = connection.recv()
            except (EOFError, OSError):
                break
            command = message.get("command")
            if command == "shutdown":
                break
            if command != "render":
                continue
            _render_request(
                connection,
                worker_id,
                RenderRequest(**message["request"]),
            )
            app.processEvents()
    except BaseException as exc:
        try:
            connection.send(
                {
                    "type": "startup_failed",
                    "worker_id": worker_id,
                    "error": f"{exc}\n{traceback.format_exc()}",
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        try:
            connection.close()
        except OSError:
            pass


class _WorkerSlot:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.process = None
        self.connection = None
        self.busy = False
        self.current_request = None


class V4RenderSupervisor:
    def __init__(
        self,
        settings: dict,
        worker_count: int = RENDER_WORKER_COUNT,
        stall_timeout: float = RENDER_STALL_TIMEOUT_SECONDS,
        retry_limit: int = RENDER_RETRY_LIMIT,
        worker_target=None,
    ):
        self.settings = dict(settings)
        self.worker_count = max(1, int(worker_count))
        self.stall_timeout = max(0.1, float(stall_timeout))
        self.retry_limit = max(0, int(retry_limit))
        self._worker_target = worker_target or _worker_main
        self._context = multiprocessing.get_context("spawn")
        self._condition = threading.Condition()
        self._slots = [_WorkerSlot(index) for index in range(self.worker_count)]
        self._cancelled = False
        self._started = False

    def _terminate_slot(self, slot):
        request = slot.current_request
        connection = slot.connection
        process = slot.process
        slot.connection = None
        slot.process = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if process is not None:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=1.0)
            try:
                process.close()
            except (OSError, ValueError):
                pass
        if request is not None:
            temporary_path = _temporary_output_path(request, slot.worker_id)
            if os.path.exists(temporary_path):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass

    def _start_slot(self, slot):
        if self._cancelled:
            return False
        self._terminate_slot(slot)
        parent_connection, child_connection = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=self._worker_target,
            args=(child_connection, slot.worker_id, self.settings),
            name=f"V4QtRenderer-{slot.worker_id}",
            daemon=True,
        )
        process.start()
        child_connection.close()
        slot.process = process
        slot.connection = parent_connection

        deadline = time.monotonic() + RENDER_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._cancelled:
                break
            try:
                if parent_connection.poll(0.05):
                    event = parent_connection.recv()
                    if event.get("type") == "ready":
                        return True
                    if event.get("type") == "startup_failed":
                        break
            except (EOFError, OSError):
                break
            if not process.is_alive():
                break
        self._terminate_slot(slot)
        return False

    def start(self):
        with self._condition:
            if self._started:
                return any(slot.process is not None for slot in self._slots)
            self._started = True
        started = 0
        for slot in self._slots:
            if self._start_slot(slot) or self._start_slot(slot):
                started += 1
        return started > 0

    def _acquire_slot(self, should_stop):
        with self._condition:
            while True:
                if self._cancelled or should_stop():
                    return None
                for slot in self._slots:
                    if not slot.busy:
                        slot.busy = True
                        return slot
                self._condition.wait(timeout=0.05)

    def _release_slot(self, slot):
        with self._condition:
            slot.current_request = None
            slot.busy = False
            self._condition.notify_all()

    def render(
        self,
        request: RenderRequest,
        on_event: Optional[Callable[[dict], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> RenderResult:
        on_event = on_event or (lambda _event: None)
        should_stop = should_stop or (lambda: False)
        if not self._started and not self.start():
            if self._cancelled or should_stop():
                return RenderResult(
                    False,
                    request.page_key,
                    error="Render cancelled.",
                    cancelled=True,
                )
            return RenderResult(
                False,
                request.page_key,
                error="No V4 Qt renderer process could be started.",
            )

        last_error = None
        for attempt in range(self.retry_limit + 1):
            last_error = None
            slot = self._acquire_slot(should_stop)
            if slot is None:
                return RenderResult(
                    False,
                    request.page_key,
                    error="Render cancelled.",
                    cancelled=True,
                    attempts=attempt + 1,
                )
            last_heartbeat = time.monotonic()
            blocks_json = None
            try:
                if slot.process is None or not slot.process.is_alive():
                    if not self._start_slot(slot):
                        last_error = "Renderer process failed to restart."
                        continue
                slot.current_request = request
                slot.connection.send(request.to_message())
                while True:
                    if self._cancelled or should_stop():
                        self.cancel()
                        return RenderResult(
                            False,
                            request.page_key,
                            error="Render cancelled.",
                            cancelled=True,
                            attempts=attempt + 1,
                        )
                    if slot.connection.poll(0.05):
                        event = slot.connection.recv()
                        event_type = event.get("type")
                        last_heartbeat = time.monotonic()
                        if event_type in ("heartbeat", "layout_complete"):
                            if event_type == "layout_complete":
                                blocks_json = event.get("blocks_json")
                            on_event(event)
                            continue
                        if event_type == "save_complete":
                            on_event(event)
                            return RenderResult(
                                True,
                                request.page_key,
                                blocks_json=event.get("blocks_json") or blocks_json,
                                timings=event.get("timings") or {},
                                attempts=attempt + 1,
                            )
                        if event_type in ("failed", "startup_failed"):
                            last_error = event.get("error") or "Renderer failed."
                            break
                    if slot.process is None or not slot.process.is_alive():
                        last_error = "Renderer process exited unexpectedly."
                        break
                    if time.monotonic() - last_heartbeat > self.stall_timeout:
                        last_error = (
                            f"Renderer heartbeat stalled for {self.stall_timeout:.0f}s."
                        )
                        break
            except (BrokenPipeError, EOFError, OSError) as exc:
                last_error = f"Renderer IPC failed: {exc}"
            except Exception as exc:
                last_error = f"Renderer controller failed: {exc}"
            finally:
                if last_error and not self._cancelled:
                    if attempt < self.retry_limit:
                        try:
                            on_event(
                                {
                                    "type": "retry",
                                    "page_key": request.page_key,
                                    "attempt": attempt + 2,
                                    "error": last_error,
                                }
                            )
                        except Exception:
                            pass
                    self._start_slot(slot)
                self._release_slot(slot)

        return RenderResult(
            False,
            request.page_key,
            error=last_error or "Renderer failed.",
            attempts=self.retry_limit + 1,
        )

    def cancel(self):
        with self._condition:
            if self._cancelled:
                return
            self._cancelled = True
            slots = list(self._slots)
            self._condition.notify_all()
        for slot in slots:
            self._terminate_slot(slot)

    def shutdown(self):
        with self._condition:
            slots = list(self._slots)
            self._cancelled = True
            self._condition.notify_all()
        for slot in slots:
            if slot.connection is not None and slot.process is not None:
                try:
                    if slot.process.is_alive():
                        slot.connection.send({"command": "shutdown"})
                        slot.process.join(timeout=2.0)
                except (BrokenPipeError, EOFError, OSError):
                    pass
            self._terminate_slot(slot)


def make_render_request(
    page_key,
    input_path,
    output_path,
    quality,
    blocks_json,
    settings,
    run_id=None,
):
    return RenderRequest(
        run_id=run_id or uuid.uuid4().hex,
        request_id=uuid.uuid4().hex,
        page_key=page_key,
        input_path=input_path,
        output_path=output_path,
        quality=int(quality),
        blocks_json=blocks_json,
        settings=dict(settings),
    )
