import threading
import unittest

import numpy as np
import torch

from modules.ocr.ocr_manga import MangaOcr, MangaOCRBatchScheduler


class _FakeBatchModel:
    def __init__(self, calls):
        self.calls = calls
        self.lock = threading.Lock()

    def all_model_loaded(self):
        return True

    def ocr_batch(self, images):
        with self.lock:
            self.calls.append(len(images))
        return [str(int(image[0, 0, 0])) for image in images]


class _PixelValues:
    def __init__(self, tensor):
        self.pixel_values = tensor


class _FakeProcessor:
    def __call__(self, images, return_tensors):
        assert return_tensors == "pt"
        return _PixelValues(torch.zeros((len(images), 3, 2, 2)))


class _FakeGenerateModel:
    device = "cpu"

    def generate(self, pixel_values):
        return torch.arange(pixel_values.shape[0]).reshape(-1, 1)


class _FakeTokenizer:
    def batch_decode(self, output_ids, skip_special_tokens):
        assert skip_special_tokens is True
        return [f" text {int(row[0])} " for row in output_ids]


class MangaOCRBatchTests(unittest.TestCase):
    def test_scheduler_forms_cross_submission_batches_and_preserves_results(self):
        calls = []
        scheduler = MangaOCRBatchScheduler(
            model_factory=lambda: _FakeBatchModel(calls),
            worker_count=1,
            batch_size=4,
            batch_wait_ms=100,
        )
        try:
            futures = [
                scheduler.submit(np.full((2, 2, 3), value, dtype=np.uint8))
                for value in range(8)
            ]
            self.assertEqual([future.result(timeout=2) for future in futures], [str(i) for i in range(8)])
            self.assertEqual(calls, [4, 4])
        finally:
            scheduler.shutdown()

    def test_batch_size_one_keeps_legacy_dispatch_shape(self):
        calls = []
        scheduler = MangaOCRBatchScheduler(
            model_factory=lambda: _FakeBatchModel(calls),
            worker_count=3,
            batch_size=1,
        )
        try:
            futures = [
                scheduler.submit(np.full((1, 1, 3), value, dtype=np.uint8))
                for value in range(3)
            ]
            self.assertEqual([future.result(timeout=2) for future in futures], ["0", "1", "2"])
            self.assertEqual(calls, [1, 1, 1])
        finally:
            scheduler.shutdown()

    def test_model_batch_path_maps_one_decoded_result_per_image(self):
        ocr = MangaOcr.__new__(MangaOcr)
        ocr.image_processor = _FakeProcessor()
        ocr.model = _FakeGenerateModel()
        ocr.tokenizer = _FakeTokenizer()

        images = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
        self.assertEqual(ocr.ocr_batch(images), ["ｔｅｘｔ０", "ｔｅｘｔ１", "ｔｅｘｔ２"])
        self.assertEqual(ocr.ocr_batch([]), [])


if __name__ == "__main__":
    unittest.main()
