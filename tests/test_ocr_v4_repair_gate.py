import unittest

from modules.ocr.ocr_llm_api_v4 import _should_run_page_ocr


class _Block:
    def __init__(self, needs_ocr=False):
        self._v4_needs_ocr = needs_ocr


class OCRV4RepairGateTests(unittest.TestCase):
    def test_regular_pipeline_runs_ocr_when_globally_enabled(self):
        self.assertTrue(
            _should_run_page_ocr(
                enable_ocr=True,
                repair_mode=False,
                blk_list=[_Block(False)],
            )
        )

    def test_repair_pipeline_runs_ocr_only_for_marked_pages(self):
        self.assertFalse(
            _should_run_page_ocr(
                enable_ocr=True,
                repair_mode=True,
                blk_list=[_Block(False), _Block(False)],
            )
        )
        self.assertTrue(
            _should_run_page_ocr(
                enable_ocr=True,
                repair_mode=True,
                blk_list=[_Block(False), _Block(True)],
            )
        )

    def test_disabled_ocr_never_runs(self):
        self.assertFalse(
            _should_run_page_ocr(
                enable_ocr=False,
                repair_mode=False,
                blk_list=[_Block(True)],
            )
        )


if __name__ == "__main__":
    unittest.main()
