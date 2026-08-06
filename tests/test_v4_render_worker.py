import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qtpy.QtGui import QColor, QImage
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication

from modules.translators.v4_render_worker import (
    RenderRequest,
    V4RenderSupervisor,
    _initialize_qt_worker,
    encode_text_blocks,
    make_render_request,
)
from modules.translators import trans_llm_api_v4 as layout_mod
from utils.config import pcfg
from utils.io_utils import imread
from utils.textblock import TextBlock
from ui import scenetext_manager


def _hanging_worker(connection, worker_id, _settings):
    connection.send({'type': 'ready', 'worker_id': worker_id})
    while True:
        try:
            message = connection.recv()
        except (EOFError, OSError):
            return
        if message.get('command') == 'shutdown':
            return
        time.sleep(60)


class V4RenderWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        _initialize_qt_worker(cls._settings())
        from ui import scenetext_manager  # noqa: F401
        layout_mod._install_patches()

    @staticmethod
    def _settings():
        return {
            'ldpi': 96.0,
            'default_font_family': 'Microsoft YaHei UI',
            'app_default_font': 'Microsoft YaHei UI',
            'translate_source': '日本語',
            'translate_target': '한국어',
            'let_autolayout_flag': True,
            'let_fntsize_flag': 0,
        }

    def test_process_renderer_saves_atomically_and_emits_separate_stages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'input.png')
            output_path = os.path.join(tmpdir, 'output.png')
            image = QImage(200, 100, QImage.Format_ARGB32)
            image.fill(QColor(17, 34, 51, 255))
            self.assertTrue(image.save(input_path))
            block = TextBlock(
                xyxy=[20, 20, 180, 80],
                lines=[[[20, 20], [180, 20], [180, 80], [20, 80]]],
                translation='Qt renderer parity',
            )
            block.fontformat.stroke_width = 0.08
            block.fontformat.shadow_radius = 0.08
            block.fontformat.shadow_strength = 0.6
            block.fontformat.shadow_offset = [0.05, 0.05]
            block.fontformat.gradient_enabled = True
            block.fontformat.gradient_start_color = [10, 20, 30]
            block.fontformat.gradient_end_color = [150, 100, 50]
            blocks_json = encode_text_blocks([block])

            legacy_project = SimpleNamespace(
                directory=tmpdir,
                current_img='input.png',
                img_array=imread(input_path),
                pages={'input.png': [block]},
                _v4_save_lock=threading.RLock(),
                get_inpainted_path=lambda _page_key: input_path,
            )
            layout_context = SimpleNamespace(
                imgtrans_proj=legacy_project,
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
            legacy_helper = layout_mod.UIHelper(
                None,
                legacy_project,
                layout_context,
                total_pages=1,
                translate_thread=SimpleNamespace(stop_requested=False),
            )
            with legacy_helper.prepared_pages_lock:
                legacy_helper.prepared_pages['input.png'] = (
                    layout_mod._prepare_layout_page(
                        legacy_project,
                        'input.png',
                        True,
                    )
                )
            legacy_helper.render_page_task('input.png')
            legacy_image = legacy_helper.rendered_images.pop('input.png')
            self.assertIsNotNone(legacy_image)

            request = make_render_request(
                'input.png',
                input_path,
                output_path,
                -1,
                blocks_json,
                self._settings(),
                run_id='test-run',
            )
            events = []
            supervisor = V4RenderSupervisor(
                self._settings(),
                worker_count=1,
                stall_timeout=20,
            )
            results = []
            ticks = []
            timer = QTimer()
            timer.setInterval(25)
            timer.timeout.connect(lambda: ticks.append(time.monotonic()))
            render_thread = threading.Thread(
                target=lambda: results.append(
                    supervisor.render(request, on_event=events.append)
                ),
                daemon=True,
            )
            try:
                timer.start()
                render_thread.start()
                deadline = time.monotonic() + 60
                while render_thread.is_alive() and time.monotonic() < deadline:
                    QApplication.processEvents()
                    time.sleep(0.005)
                render_thread.join(timeout=1.0)
            finally:
                timer.stop()
                supervisor.shutdown()
            self.assertFalse(render_thread.is_alive())
            self.assertTrue(results)
            result = results[0]

            self.assertTrue(result.success, f"{result.error}\nevents={events}")
            self.assertTrue(os.path.exists(output_path))
            event_types = [event.get('type') for event in events]
            self.assertIn('layout_complete', event_types)
            self.assertIn('save_complete', event_types)
            self.assertLess(
                event_types.index('layout_complete'),
                event_types.index('save_complete'),
            )
            rendered = QImage(output_path).convertToFormat(QImage.Format_ARGB32)
            legacy = legacy_image.convertToFormat(QImage.Format_ARGB32)
            self.assertEqual(rendered.size(), legacy.size())
            rendered_bits = rendered.bits()
            legacy_bits = legacy.bits()
            rendered_bits.setsize(rendered.sizeInBytes())
            legacy_bits.setsize(legacy.sizeInBytes())
            self.assertEqual(bytes(rendered_bits), bytes(legacy_bits))
            self.assertEqual(
                json.loads(result.blocks_json),
                json.loads(encode_text_blocks(legacy_project.pages['input.png'])),
            )
            leftovers = [name for name in os.listdir(tmpdir) if '.v4-' in name]
            self.assertEqual(leftovers, [])
            self.assertGreater(len(ticks), 2)
            max_tick_gap = max(
                later - earlier for earlier, later in zip(ticks, ticks[1:])
            )
            self.assertLess(max_tick_gap, 0.5)

    def test_stalled_worker_is_restarted_and_retried_once(self):
        request = RenderRequest(
            run_id='stall-run',
            request_id='stall-request',
            page_key='stall.png',
            input_path='unused',
            output_path='unused.png',
            quality=-1,
            blocks_json='[]',
            settings=self._settings(),
        )
        supervisor = V4RenderSupervisor(
            self._settings(),
            worker_count=1,
            stall_timeout=0.2,
            retry_limit=1,
            worker_target=_hanging_worker,
        )
        started = time.monotonic()
        try:
            result = supervisor.render(request)
        finally:
            supervisor.shutdown()
        elapsed = time.monotonic() - started

        self.assertFalse(result.success)
        self.assertEqual(result.attempts, 2)
        self.assertIn('heartbeat stalled', result.error)
        self.assertLess(elapsed, 20.0)

    def test_ipc_block_snapshot_excludes_large_runtime_masks(self):
        block = TextBlock(
            xyxy=[0, 0, 10, 10],
            lines=[[[0, 0], [10, 0], [10, 10], [0, 10]]],
            translation='text',
        )
        block.region_mask = np.ones((64, 64), dtype=np.uint8)
        block.region_inpaint_dict = {'large': np.ones((64, 64), dtype=np.uint8)}
        payload = json.loads(encode_text_blocks([block]))[0]
        self.assertNotIn('region_mask', payload)
        self.assertNotIn('region_inpaint_dict', payload)

    def test_v4_save_entry_uses_process_renderer_and_commits_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = os.path.join(tmpdir, 'result')
            page_keys = ['page-a.png', 'page-b.png']
            for index, page_key in enumerate(page_keys):
                image = QImage(32 + index, 24 + index, QImage.Format_ARGB32)
                image.fill(QColor(80 + index, 90, 100, 255))
                self.assertTrue(image.save(os.path.join(tmpdir, page_key)))
            journaled = []
            project = SimpleNamespace(
                directory=tmpdir,
                pages={page_key: [] for page_key in page_keys},
                _image_info={
                    page_key: {'finish_code': 0} for page_key in page_keys
                },
                _v4_save_lock=threading.RLock(),
                result_dir=lambda: result_dir,
                get_inpainted_path=lambda page_key: os.path.join(tmpdir, page_key),
                append_progress_journal=journaled.append,
            )
            translate_thread = SimpleNamespace(
                imgtrans_proj=project,
                stop_requested=False,
                parent=lambda: None,
            )
            old_inpaint = pcfg.module.enable_inpaint
            old_repair_mode = layout_mod._V4_REPAIR_MODE
            old_repair_pages = layout_mod._V4_REPAIR_PAGES
            try:
                pcfg.module.enable_inpaint = False
                layout_mod._V4_REPAIR_MODE = False
                layout_mod._V4_REPAIR_PAGES = None
                layout_mod._v4_headless_save_entry(
                    translate_thread,
                    proj=project,
                    wait_for_pipeline=False,
                )
            finally:
                pcfg.module.enable_inpaint = old_inpaint
                layout_mod._V4_REPAIR_MODE = old_repair_mode
                layout_mod._V4_REPAIR_PAGES = old_repair_pages

            self.assertTrue(translate_thread._v4_save_completed)
            self.assertCountEqual(journaled, page_keys)
            for page_key in page_keys:
                self.assertTrue(os.path.exists(os.path.join(result_dir, page_key)))
                self.assertEqual(
                    translate_thread._v4_layout_states[page_key],
                    'completed',
                )

    def test_cancel_terminates_hung_worker_without_waiting_for_timeout(self):
        request = RenderRequest(
            run_id='cancel-run',
            request_id='cancel-request',
            page_key='cancel.png',
            input_path='unused',
            output_path='unused.png',
            quality=-1,
            blocks_json='[]',
            settings=self._settings(),
        )
        supervisor = V4RenderSupervisor(
            self._settings(),
            worker_count=1,
            stall_timeout=30,
            retry_limit=1,
            worker_target=_hanging_worker,
        )
        self.assertTrue(supervisor.start())
        results = []
        thread = threading.Thread(
            target=lambda: results.append(supervisor.render(request)),
            daemon=True,
        )
        thread.start()
        time.sleep(0.3)
        cancel_started = time.monotonic()
        supervisor.cancel()
        thread.join(timeout=2.0)
        elapsed = time.monotonic() - cancel_started
        supervisor.shutdown()

        self.assertFalse(thread.is_alive())
        self.assertTrue(results and results[0].cancelled)
        self.assertLess(elapsed, 2.0)


if __name__ == '__main__':
    unittest.main()
