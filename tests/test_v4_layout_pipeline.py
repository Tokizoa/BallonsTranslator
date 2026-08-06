import ast
import inspect
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from qtpy.QtGui import QImage, QColor

from modules.translators import trans_llm_api_v4 as layout_mod
from modules.translators import v4_render_worker as worker_mod


class _CaptureLogger:
    def __init__(self):
        self.messages = []

    def debug(self, message):
        self.messages.append(('debug', message))

    def info(self, message):
        self.messages.append(('info', message))

    def warning(self, message):
        self.messages.append(('warning', message))

    def error(self, message):
        self.messages.append(('error', message))


class _Block:
    def __init__(self):
        self.translation = '번역'
        self.rich_text = ''
        self.src_is_vertical = True
        self.xyxy = [4, 2, 8, 12]

    def bounding_rect(self):
        return [4, 2, 4, 10]


class V4LayoutPipelineTests(unittest.TestCase):
    def test_detail_log_toggle_defaults_off_and_gates_only_detail_messages(self):
        param = layout_mod.LLM_API_Translator_V4.params['autolayout_detail_logging']
        self.assertFalse(param['value'])
        self.assertEqual(param['type'], 'checkbox')

        logger = _CaptureLogger()
        with mock.patch.object(layout_mod, 'LOGGER', logger):
            with mock.patch.object(layout_mod, '_V4_AUTOLAYOUT_DETAIL_LOGGING', False):
                layout_mod._autolayout_detail_log('info', 'hidden')
                logger.warning('always-visible warning')
            with mock.patch.object(layout_mod, '_V4_AUTOLAYOUT_DETAIL_LOGGING', True):
                layout_mod._autolayout_detail_log('debug', 'visible')

        self.assertEqual(
            logger.messages,
            [('warning', 'always-visible warning'), ('debug', 'visible')],
        )

        translator = layout_mod.LLM_API_Translator_V4(
            '日本語',
            '한국어',
            autolayout_detail_logging=True,
        )
        self.assertTrue(translator.autolayout_detail_logging)
        self.assertTrue(layout_mod._V4_AUTOLAYOUT_DETAIL_LOGGING)
        translator.updateParam('autolayout_detail_logging', False)
        self.assertFalse(translator.autolayout_detail_logging)
        self.assertFalse(layout_mod._V4_AUTOLAYOUT_DETAIL_LOGGING)

    def test_page_readiness_requires_current_translation_ocr_and_inpaint(self):
        finish_none = 0
        project = SimpleNamespace(
            _image_info={'page.png': {'finish_code': finish_none}},
        )
        thread = SimpleNamespace(
            stop_requested=False,
            imgtrans_proj=project,
            _v4_layout_state_lock=threading.Lock(),
            _v4_translated_pages=set(),
            _v4_translation_failed_pages=set(),
        )
        cfg = SimpleNamespace(module=SimpleNamespace(enable_ocr=True, enable_inpaint=True))

        with mock.patch.object(layout_mod, 'pcfg', cfg), mock.patch.object(
            layout_mod, '_V4_STOP_REQUESTED', False
        ):
            self.assertEqual(
                layout_mod._layout_page_readiness(thread, 'page.png', True),
                'waiting',
            )
            thread._v4_translated_pages.add('page.png')
            project._image_info['page.png']['finish_code'] = layout_mod.RunStatus.FIN_OCR
            self.assertEqual(
                layout_mod._layout_page_readiness(thread, 'page.png', True),
                'waiting',
            )
            project._image_info['page.png']['finish_code'] |= layout_mod.RunStatus.FIN_INPAINT
            self.assertEqual(
                layout_mod._layout_page_readiness(thread, 'page.png', True),
                'ready',
            )
            thread._v4_inpaint_failed_pages = {'page.png'}
            self.assertEqual(
                layout_mod._layout_page_readiness(thread, 'page.png', True),
                'failed',
            )
            thread._v4_inpaint_failed_pages.clear()
            thread.stop_requested = True
            self.assertEqual(
                layout_mod._layout_page_readiness(thread, 'page.png', True),
                'cancelled',
            )

    def test_page_is_claimed_once_and_records_full_state_sequence(self):
        thread = SimpleNamespace(_v4_layout_state_lock=threading.Lock(), _v4_layout_states={})
        self.assertTrue(layout_mod._claim_v4_layout_page(thread, 'page.png'))
        self.assertFalse(layout_mod._claim_v4_layout_page(thread, 'page.png'))
        for state in ('preparing', 'rendering', 'saving', 'completed'):
            layout_mod._set_v4_layout_state(thread, 'page.png', state)
        self.assertEqual(thread._v4_layout_states['page.png'], 'completed')

        layout_mod._add_v4_layout_timing(thread, 'area_prepare', 0.25)
        layout_mod._add_v4_layout_timing(thread, 'area_prepare', 0.50)
        self.assertEqual(thread._v4_layout_timings['area_prepare'], 0.75)

    def test_only_one_layout_run_can_enqueue_pages_at_a_time(self):
        thread = SimpleNamespace(
            _v4_layout_state_lock=threading.Lock(),
            _v4_layout_states={'old-page.png': 'completed'},
            _v4_layout_failed_pages={'old-page.png'},
            _v4_save_completed=True,
        )

        self.assertTrue(layout_mod._begin_v4_layout_run(thread, reset_page_states=True))
        self.assertEqual(thread._v4_layout_states, {})
        self.assertEqual(thread._v4_layout_failed_pages, set())
        self.assertFalse(thread._v4_save_completed)
        self.assertFalse(layout_mod._begin_v4_layout_run(thread, reset_page_states=True))

        layout_mod._end_v4_layout_run(thread)
        self.assertTrue(layout_mod._begin_v4_layout_run(thread, reset_page_states=False))
        layout_mod._end_v4_layout_run(thread)

    def test_renderer_waits_for_keyed_gui_text_preparation(self):
        thread = SimpleNamespace(
            stop_requested=False,
            _v4_page_prepare_lock=threading.Lock(),
            _v4_page_prepare_events={},
        )

        class _ImmediatePrepareSignal:
            def emit(self, page_key):
                with thread._v4_page_prepare_lock:
                    thread._v4_page_prepare_events[page_key].set()

        thread.v4_prepare_page = _ImmediatePrepareSignal()
        with mock.patch.object(layout_mod, '_V4_STOP_REQUESTED', False):
            self.assertTrue(
                layout_mod._v4_prepare_translated_page_for_render(
                    thread,
                    'page.png',
                    timeout_seconds=0.1,
                )
            )
        self.assertEqual(thread._v4_page_prepare_events, {})

    def test_worker_preparation_does_not_mutate_block_geometry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, 'page.png')
            image = QImage(24, 24, QImage.Format_RGB888)
            image.fill(QColor('white'))
            self.assertTrue(image.save(image_path))

            block = _Block()
            original_xyxy = list(block.xyxy)
            project = SimpleNamespace(
                directory=tmpdir,
                pages={'page.png': [block]},
                get_inpainted_path=lambda _key: image_path,
            )
            mask = np.ones((10, 4), dtype=np.uint8)
            extracted = (mask, None, None, [3, 1, 8, 14])

            with mock.patch(
                'utils.imgproc_utils.extract_ballon_region',
                return_value=extracted,
            ) as extract:
                prepared = layout_mod._prepare_layout_page(project, 'page.png', True)

            self.assertEqual(block.xyxy, original_xyxy)
            self.assertNotEqual(prepared['blocks'][0]['widened_xyxy'], original_xyxy)
            self.assertIs(prepared['blocks'][0]['mask'], mask)
            self.assertEqual(prepared['blocks'][0]['bounding_rect'], [4, 2, 4, 10])
            extract.assert_called_once()

    def test_qt_layout_and_final_drawing_remain_in_main_thread_method(self):
        source = inspect.getsource(layout_mod.UIHelper.render_page_task)
        self.assertIn('TextBlkItem', source)
        self.assertIn('self.stm.layout_textblk', source)
        self.assertIn('QPainter(image)', source)
        self.assertIn("block_prep.get('mask')", source)
        self.assertIn('LOGGER.warning', source)

        pipeline_source = inspect.getsource(layout_mod._v4_headless_save_entry)
        self.assertIn('_begin_v4_layout_run', pipeline_source)
        self.assertIn('_end_v4_layout_run', pipeline_source)
        self.assertIn('V4RenderSupervisor', pipeline_source)
        self.assertNotIn('BlockingQueuedConnection', pipeline_source)
        self.assertNotIn('_v4_gui_invoke_lock', pipeline_source)

        worker_source = inspect.getsource(worker_mod._render_request)
        layout_done = worker_source.index('"type": "layout_complete"')
        image_save = worker_source.index('image.save(')
        save_done = worker_source.index('"type": "save_complete"')
        self.assertLess(layout_done, image_save)
        self.assertLess(image_save, save_done)

        signaler = layout_mod.SaveSignaler()
        self.assertTrue(hasattr(signaler, 'layout_progress_signal'))
        self.assertTrue(hasattr(signaler, 'save_progress_signal'))
        self.assertFalse(hasattr(signaler, 'progress_signal'))
        self.assertNotIn(
            'updateTranslateProgress',
            inspect.getsource(layout_mod.UIHelper),
        )

        mainwindow_source = Path('ui/mainwindow.py').read_text(encoding='utf-8')
        mainwindow_tree = ast.parse(mainwindow_source)
        gui_prepare_node = next(
            node
            for node in ast.walk(mainwindow_tree)
            if isinstance(node, ast.FunctionDef) and node.name == 'on_v4_prepare_page'
        )
        gui_prepare_source = ast.get_source_segment(mainwindow_source, gui_prepare_node)
        self.assertIn('_prepare_page_translation_output', gui_prepare_source)
        self.assertNotIn('updateCanvas', gui_prepare_source)
        self.assertNotIn('.save(', gui_prepare_source)

    def test_stop_click_is_processed_while_render_waits_for_project_lock(self):
        project_lock = threading.Lock()
        project_lock.acquire()
        translate_thread = SimpleNamespace(stop_requested=False)
        project = SimpleNamespace(
            pages={'page.png': []},
            _v4_save_lock=project_lock,
        )
        helper = layout_mod.UIHelper(
            None,
            project,
            total_pages=1,
            translate_thread=translate_thread,
        )

        class _FakeApp:
            def __init__(self):
                self.calls = 0

            def processEvents(self):
                self.calls += 1
                if self.calls >= 2:
                    translate_thread.stop_requested = True

        fake_app = _FakeApp()
        try:
            with mock.patch.object(layout_mod.QApplication, 'instance', return_value=fake_app), mock.patch.object(
                layout_mod, '_V4_STOP_REQUESTED', False
            ):
                started = __import__('time').perf_counter()
                helper.render_page_task('page.png')
                elapsed = __import__('time').perf_counter() - started
                self.assertTrue(layout_mod._V4_STOP_REQUESTED)
        finally:
            project_lock.release()

        self.assertGreaterEqual(fake_app.calls, 2)
        self.assertLess(elapsed, 0.5)
        self.assertIsNone(helper.rendered_images['page.png'])


if __name__ == '__main__':
    unittest.main()
