# modified from https://github.com/kha-white/manga-ocr/blob/master/manga_ocr/ocr.py
import re
import concurrent.futures
import queue
import threading
import time
import jaconv
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
import numpy as np
import torch
from typing import Callable, List

from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock

MANGA_OCR_PATH = r'data/models/manga-ocr-base-2025'
class MangaOcr:
    def __init__(self, pretrained_model_name_or_path=MANGA_OCR_PATH, device='cpu'):
        self.image_processor = AutoImageProcessor.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name_or_path)
        self.to(device)
        
    def to(self, device):
        self.model.to(device)

    @torch.inference_mode()
    def __call__(self, img: np.ndarray):
        return self.ocr_batch([img])[0]

    @torch.inference_mode()
    def ocr_batch(self, images: List[np.ndarray]) -> List[str]:
        """Run true batched OCR for a list of RGB images."""
        if not images:
            return []

        pixel_values = self.image_processor(
            images,
            return_tensors="pt",
        ).pixel_values.to(self.model.device)
        output_ids = self.model.generate(pixel_values).cpu()
        decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return [post_process(text) for text in decoded]


_BATCH_STOP = object()


class MangaOCRBatchScheduler:
    """Persistent cross-page batch queue backed by one or more MangaOCR models."""

    def __init__(
        self,
        model_factory: Callable[[], "MangaOCR"],
        worker_count: int,
        batch_size: int,
        batch_wait_ms: int = 10,
        logger=None,
        debug: bool = False,
    ) -> None:
        self.worker_count = max(1, int(worker_count))
        self.batch_size = max(1, int(batch_size))
        self.batch_wait_ms = max(0, int(batch_wait_ms))
        self.logger = logger
        self.debug = bool(debug)
        self._queue = queue.Queue()
        self._closed = False
        self._models = []
        self._threads = []

        # Load models before starting workers so callers never observe a partial pool.
        for _ in range(self.worker_count):
            model = model_factory()
            if not model.all_model_loaded():
                model.load_model()
            self._models.append(model)

        for worker_index, model in enumerate(self._models):
            thread = threading.Thread(
                target=self._worker_loop,
                args=(worker_index, model),
                name=f"MangaOCRBatch-{worker_index + 1}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def is_compatible(self, worker_count: int, batch_size: int) -> bool:
        return (
            not self._closed
            and self.worker_count == max(1, int(worker_count))
            and self.batch_size == max(1, int(batch_size))
        )

    def submit(self, image: np.ndarray) -> concurrent.futures.Future:
        if self._closed:
            raise RuntimeError("MangaOCR batch scheduler is closed")
        future = concurrent.futures.Future()
        self._queue.put((image, future))
        return future

    def _worker_loop(self, worker_index: int, model: "MangaOCR") -> None:
        while True:
            first = self._queue.get()
            if first is _BATCH_STOP:
                self._queue.task_done()
                return

            tasks = [first]
            stop_after_batch = False
            if self.batch_size > 1:
                deadline = time.perf_counter() + self.batch_wait_ms / 1000.0
                while len(tasks) < self.batch_size:
                    timeout = deadline - time.perf_counter()
                    if timeout <= 0:
                        break
                    try:
                        item = self._queue.get(timeout=timeout)
                    except queue.Empty:
                        break
                    if item is _BATCH_STOP:
                        self._queue.task_done()
                        stop_after_batch = True
                        break
                    tasks.append(item)

            try:
                images = [image for image, _future in tasks]
                started_at = time.perf_counter()
                results = model.ocr_batch(images)
                elapsed = time.perf_counter() - started_at
                if len(results) != len(tasks):
                    raise RuntimeError(
                        f"MangaOCR batch returned {len(results)} results for {len(tasks)} images"
                    )
                for (_image, future), text in zip(tasks, results):
                    if not future.cancelled():
                        future.set_result(text or "")
                if self.debug and self.logger:
                    self.logger.info(
                        f"[MangaOCR batch] worker={worker_index + 1} "
                        f"size={len(tasks)} infer={elapsed:.3f}s"
                    )
            except Exception as exc:
                if self.logger:
                    self.logger.error(
                        f"MangaOCR batch worker {worker_index + 1} failed "
                        f"for {len(tasks)} image(s): {exc}"
                    )
                for _image, future in tasks:
                    if not future.cancelled():
                        future.set_exception(exc)
            finally:
                for _task in tasks:
                    self._queue.task_done()

            if stop_after_batch:
                return

    def shutdown(self, wait: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        for _ in self._threads:
            self._queue.put(_BATCH_STOP)
        if wait:
            for thread in self._threads:
                thread.join()
        self._threads.clear()
        self._models.clear()


def post_process(text):
    text = ''.join(text.split())
    text = text.replace('…', '...')
    text = re.sub('[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)
    text = jaconv.h2z(text, ascii=True, digit=True)

    return text


@register_OCR('manga_ocr')
class MangaOCR(OCRBase):
    params = {
        'device': DEVICE_SELECTOR()
    }
    device = DEFAULT_DEVICE

    download_file_list = [{
        'url': 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/',
        'files': ['pytorch_model.bin', 'config.json', 'preprocessor_config.json', 'README.md', 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.txt'],
        'sha256_pre_calculated': ['c63e0bb5b3ff798c5991de18a8e0956c7ee6d1563aca6729029815eda6f5c2eb', None, None, None, None, None, None],
        'save_dir': 'data/models/manga-ocr-base',
        'concatenate_url_filename': 1,
    }]
    _load_model_keys = {'model'}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.device = self.params['device']['value']
        self.model: MangaOCR = None

    def _load_model(self):
        if self.model is None:
            self.model = MangaOcr(device=self.device)

    def ocr_img(self, img: np.ndarray) -> str:
        return self.model(img)

    def ocr_batch(self, images: List[np.ndarray]) -> List[str]:
        return self.model.ocr_batch(images)

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
        im_h, im_w = img.shape[:2]
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            if y2 < im_h and x2 < im_w and \
                x1 > 0 and y1 > 0 and x1 < x2 and y1 < y2: 
                # Extract region and convert RGBA to RGB if necessary for model input
                region = img[y1:y2, x1:x2]
                blk.text = self.model(region)
            else:
                self.logger.warning('invalid textbbox to target img')
                blk.text = ''

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        device = self.params['device']['value']
        if self.device != device and self.model is not None:
            self.model.to(device)




if __name__ == '__main__':
    import cv2

    img_path = r'data/testpacks/textline/ballontranslator.png'
    manga_ocr = MangaOcr(pretrained_model_name_or_path=MANGA_OCR_PATH, device='cuda')

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dummy = np.zeros((1024, 1024, 3), np.uint8)
    manga_ocr(dummy)
    # preprocessed = manga_ocr(img_path)

    # im_batch = 
    # img = (torch.from_numpy(img[np.newaxis, ...]).float() - 127.5) / 127.5
    # img = einops.rearrange(img, 'N H W C -> N C H W')
    import time
    
    for ii in range(10):
        t0 = time.time()
        out = manga_ocr(dummy)
        print(out, time.time() - t0)
