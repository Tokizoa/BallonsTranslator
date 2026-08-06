import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qtpy.QtWidgets import QApplication

from ui.custom_widget.message import ImgtransProgressMessageBox
from ui.module_manager import ModuleManager


class ImgtransProgressBarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.box = ImgtransProgressMessageBox()

    def tearDown(self):
        self.box.close()
        self.box.deleteLater()

    def test_six_stage_bars_have_fixed_order_and_labels(self):
        expected = [
            (self.box.detect_bar, 'Detecting: '),
            (self.box.ocr_bar, 'OCR: '),
            (self.box.inpaint_bar, 'Inpainting: '),
            (self.box.translate_bar, 'Translating: '),
            (self.box.layout_bar, 'AutoLayout: '),
            (self.box.saving_bar, 'Saving: '),
        ]

        for index, (bar, label) in enumerate(expected):
            self.assertIs(self.box.layout().itemAt(index).widget(), bar)
            self.assertEqual(bar.description, label)

    def test_layout_and_save_updates_are_isolated_and_monotonic(self):
        bars = {
            'detect': self.box.detect_bar,
            'ocr': self.box.ocr_bar,
            'inpaint': self.box.inpaint_bar,
            'translate': self.box.translate_bar,
            'layout': self.box.layout_bar,
            'save': self.box.saving_bar,
        }

        self.box.updateLayoutProgress(35, ' layout')
        self.assertEqual(bars['layout'].progressbar.value(), 35)
        for name in ('detect', 'ocr', 'inpaint', 'translate', 'save'):
            self.assertEqual(bars[name].progressbar.value(), 0)

        self.box.updateSavingProgress(20, ' save')
        self.assertEqual(bars['save'].progressbar.value(), 20)
        self.assertEqual(bars['layout'].progressbar.value(), 35)

        self.box.updateLayoutProgress(10, ' stale')
        self.box.updateSavingProgress(5, ' stale')
        self.assertEqual(bars['layout'].progressbar.value(), 35)
        self.assertEqual(bars['save'].progressbar.value(), 20)

        self.box.updateLayoutProgress(35, ' (중단됨: 7/20)')
        self.box.updateSavingProgress(20, ' (실패: 16, 완료: 4/20)')
        self.assertIn('중단됨: 7/20', bars['layout'].textlabel.text())
        self.assertIn('실패: 16', bars['save'].textlabel.text())
        self.assertEqual(bars['layout'].progressbar.value(), 35)
        self.assertEqual(bars['save'].progressbar.value(), 20)

        self.box.zero_progress()
        self.assertTrue(all(bar.progressbar.value() == 0 for bar in bars.values()))

    def test_v4_visibility_does_not_affect_the_original_four_bars(self):
        self.assertFalse(self.box.layout_bar.isVisible())
        self.assertFalse(self.box.saving_bar.isVisible())

        self.box.setV4BarsVisible(True)
        self.box.show()
        QApplication.processEvents()
        self.assertTrue(self.box.layout_bar.isVisible())
        self.assertTrue(self.box.saving_bar.isVisible())
        for bar in (
            self.box.detect_bar,
            self.box.ocr_bar,
            self.box.inpaint_bar,
            self.box.translate_bar,
        ):
            self.assertTrue(bar.isVisible())

        self.box.setV4BarsVisible(False)
        self.assertFalse(self.box.layout_bar.isVisible())
        self.assertFalse(self.box.saving_bar.isVisible())

    def test_v4_progress_does_not_queue_legacy_page_refreshes(self):
        manager = ModuleManager(SimpleNamespace())
        manager.translate_thread = SimpleNamespace(
            translator=SimpleNamespace(use_image_batching=True),
        )
        manager.imgtrans_thread = SimpleNamespace(
            num_pages=1407,
            recent_finished_index=lambda progress: progress - 1,
        )
        manager.progress_msgbox = SimpleNamespace(
            updateTranslateProgress=lambda progress: None,
        )
        manager.finishImgtransPipeline = lambda: None
        manager.last_finished_index = -1
        refreshed_pages = []
        manager.page_trans_finished.connect(refreshed_pages.append)

        for progress in range(1, 1408):
            manager.on_update_translate_progress(progress)

        self.assertEqual(refreshed_pages, [])

        manager.translate_thread.translator.use_image_batching = False
        manager.on_update_translate_progress(1)
        self.assertEqual(refreshed_pages, [0])
        manager.deleteLater()


if __name__ == '__main__':
    unittest.main()
