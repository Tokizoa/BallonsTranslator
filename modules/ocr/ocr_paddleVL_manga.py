from transformers import AutoModelForCausalLM, AutoProcessor
import numpy as np
import torch
from typing import List

from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock
from modules.ocr.ocr_manga import MangaOCR

MODEL_PATH = 'data/models/PaddleOCR-VL-For-Manga'


def _apply_chunked_processing_patch():
    try:
        from ui.module_manager import ImgtransThread
        from utils.io_utils import text_is_empty
        from utils.imgproc_utils import get_block_mask
        from utils.textblock import sort_regions
        from modules.base import soft_empty_cache
        from ui.module_manager import unload_modules 
        from utils.logger import logger as LOGGER
        from utils.config import pcfg, RunStatus
        from utils import shared
        import ui.module_manager as mm_module
        
        # Check if already patched
        if not getattr(ImgtransThread, '_is_chunked_patched', False):
            
            ImgtransThread._original_imgtrans_pipeline = ImgtransThread._imgtrans_pipeline
            cfg_module = pcfg.module
            create_error_dialog = mm_module.create_error_dialog

            def _imgtrans_pipeline_chunked(self):
                # Check if current OCR module requires chunking (V2 or PaddleVL)
                is_target_ocr = getattr(self.ocr, 'use_page_batching', False) or \
                                self.ocr.__class__.__name__ == 'PaddleOCRVLManga'
                
                if not is_target_ocr:
                    return self._original_imgtrans_pipeline()

                if LOGGER:
                    LOGGER.info("🚀 Async Processing: Wait for 20 pages before starting OCR.")

                self.detect_counter = 0
                self.ocr_counter = 0
                self.translate_counter = 0
                self.inpaint_counter = 0
                
                # Setup pages to iterate
                all_pages = list(self.imgtrans_proj.pages.keys())
                if self.pages_to_process is not None and len(self.pages_to_process) > 0:
                    pages_to_iterate = self.pages_to_process
                    self.num_pages = len(self.pages_to_process)
                    for process_idx, page_name in enumerate(pages_to_iterate):
                        if page_name in all_pages:
                            self.process_idx_to_page_idx[process_idx] = all_pages.index(page_name)
                    LOGGER.info(f'Processing specific pages: {len(pages_to_iterate)} pages')
                else:
                    pages_to_iterate = all_pages
                    self.num_pages = len(self.imgtrans_proj.pages)
                    for i in range(self.num_pages):
                        self.process_idx_to_page_idx[i] = i
                    LOGGER.info(f'Processing all {self.num_pages} pages')

                self.textdetect_thread.num_process_pages = self.num_pages
                self.ocr_thread.num_process_pages = self.num_pages
                self.inpaint_thread.num_process_pages = self.num_pages
                self.translate_thread.num_process_pages = self.num_pages

                low_vram_trans = False
                if self.translator is not None:
                    low_vram_trans = self.translator.low_vram_mode
                    self.parallel_trans = not self.translator.is_computational_intensive() and not low_vram_trans
                else:
                    self.parallel_trans = False
                
                if self.parallel_trans and cfg_module.enable_translate:
                    self.translate_thread.runTranslatePipeline(self.imgtrans_proj)

                # --- ASYNC LOGIC START ---
                initial_buffer = 20 if self.num_pages >= 20 else max(1, int(self.num_pages / 2))
                ocr_started = False
                
                # OCR Executor (Background Thread)
                ocr_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                ocr_futures = []

                def run_ocr_step(imgname_arg):
                    if self.stop_requested: return
                    try:
                        img_ocr = self.imgtrans_proj.read_img(imgname_arg)
                        blk_list_ocr = self.imgtrans_proj.pages.get(imgname_arg, [])
                        mask = None
                        need_save_mask = False
                        blk_removed = []

                        if cfg_module.enable_ocr:
                            try:
                                self.ocr.run_ocr(img_ocr, blk_list_ocr)
                            except Exception as e:
                                if shared and getattr(shared, 'create_errdialog_in_mainthread', None):
                                    shared.create_errdialog_in_mainthread(str(e), "OCR Failed", "OCRFailed")
                            self.ocr_counter += 1

                            if pcfg.restore_ocr_empty:
                                blk_list_updated = []
                                for blk in blk_list_ocr:
                                    text = blk.get_text()
                                    if text_is_empty(text):
                                        blk_removed.append(blk)
                                    else:
                                        blk_list_updated.append(blk)

                                if len(blk_removed) > 0:
                                    blk_list_ocr.clear()
                                    blk_list_ocr += blk_list_updated
                                    
                                    if mask is None:
                                        mask = self.imgtrans_proj.load_mask_by_imgname(imgname_arg)
                                    if mask is not None:
                                        inpainted = None
                                        if not cfg_module.enable_inpaint:
                                            inpainted = self.imgtrans_proj.load_inpainted_by_imgname(imgname_arg)
                                        for blk in blk_removed:
                                            from modules.utils import get_block_mask
                                            xywh = blk.bounding_rect()
                                            blk_mask, xyxy = get_block_mask(xywh, mask, blk.angle)
                                            x1, y1, x2, y2 = xyxy
                                            if blk_mask is not None:
                                                mask[y1: y2, x1: x2] = 0
                                                if inpainted is not None:
                                                    mskpnt = np.where(blk_mask)
                                                    inpainted[y1: y2, x1: x2][mskpnt] = img_ocr[y1: y2, x1: x2][mskpnt]
                                                need_save_mask = True
                                        if inpainted is not None and need_save_mask:
                                            self.imgtrans_proj.save_inpainted(imgname_arg, inpainted)
                                        if need_save_mask:
                                            self.imgtrans_proj.save_mask(imgname_arg, mask)
                                            need_save_mask = False

                            self.imgtrans_proj.update_page_progress(imgname_arg, RunStatus.FIN_OCR)
                            self.update_ocr_progress.emit(self.ocr_counter)

                        if need_save_mask and mask is not None:
                            self.imgtrans_proj.save_mask(imgname_arg, mask)

                        if cfg_module.enable_translate:
                            if self.parallel_trans:
                                self.translate_thread.push_pagekey_queue(imgname_arg)
                            elif not low_vram_trans:
                                self.translator.translate_textblk_lst(blk_list_ocr)
                                self.translate_counter += 1
                                self.update_translate_progress.emit(self.translate_counter)

                        if cfg_module.enable_inpaint:
                            if mask is None:
                                mask = self.imgtrans_proj.load_mask_by_imgname(imgname_arg)
                            if mask is not None:
                                try:
                                    inpainted = self.inpainter.inpaint(img_ocr, mask, blk_list_ocr)
                                    self.imgtrans_proj.save_inpainted(imgname_arg, inpainted)
                                except Exception as e:
                                    if shared and getattr(shared, 'create_errdialog_in_mainthread', None):
                                        shared.create_errdialog_in_mainthread(str(e), "Inpainting Failed", "InpaintFailed")
                            self.inpaint_counter += 1
                            self.imgtrans_proj.update_page_progress(imgname_arg, RunStatus.FIN_INPAINT)
                            self.update_inpaint_progress.emit(self.inpaint_counter)

                    except Exception as e:
                        LOGGER.error(f"Async OCR failed for {imgname_arg}: {e}")

                # Main Detection Loop
                processed_count = 0
                detect_buffer = []
                
                for imgname in pages_to_iterate:
                    if self.stop_requested:
                        LOGGER.info('Pipeline stopped')
                        break

                    # 1. Detection (Main Thread)
                    img = self.imgtrans_proj.read_img(imgname)
                    mask = blk_list = None
                    need_save_mask = False
                    
                    if cfg_module.enable_detect:
                        try:
                            mask, blk_list = self.textdetector.detect(img, self.imgtrans_proj)
                            need_save_mask = True
                        except Exception as e:
                            if shared and getattr(shared, 'create_errdialog_in_mainthread', None):
                                shared.create_errdialog_in_mainthread(str(e), "Text Detection Failed", "TextDetectFailed")
                            blk_list = []
                        self.detect_counter += 1
                        if pcfg.module.keep_exist_textlines:
                            blk_list = self.imgtrans_proj.pages[imgname] + blk_list
                            blk_list = sort_regions(blk_list)
                            existed_mask = self.imgtrans_proj.load_mask_by_imgname(imgname)
                            if existed_mask is not None:
                                mask = np.bitwise_or(mask, existed_mask)
                        self.imgtrans_proj.pages[imgname] = blk_list

                        if mask is not None and not cfg_module.enable_ocr:
                            self.imgtrans_proj.save_mask(imgname, mask)
                            need_save_mask = False
                            
                        self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_DET)
                        self.update_detect_progress.emit(self.detect_counter)
                    
                    if need_save_mask and mask is not None:
                        self.imgtrans_proj.save_mask(imgname, mask)

                    # 2. Schedule OCR (Async)
                    processed_count += 1
                    
                    # Wait for initial buffer before starting OCR (only once)
                    if not ocr_started:
                        detect_buffer.append(imgname)
                        if processed_count >= initial_buffer:
                            ocr_started = True
                            if LOGGER:
                                LOGGER.info(f"Initial buffer reached ({processed_count} pages). Starting OCR.")
                            # Flush buffer
                            for buffered_img in detect_buffer:
                                future = ocr_executor.submit(run_ocr_step, buffered_img)
                                ocr_futures.append(future)
                            detect_buffer.clear()
                    else:
                        # Once started, submit immediately
                        future = ocr_executor.submit(run_ocr_step, imgname)
                        ocr_futures.append(future)

                # Flush remaining buffer
                for buffered_img in detect_buffer:
                    future = ocr_executor.submit(run_ocr_step, buffered_img)
                    ocr_futures.append(future)

                # Wait for completion
                concurrent.futures.wait(ocr_futures)
                ocr_executor.shutdown()
                # --- ASYNC LOGIC END ---

                if cfg_module.enable_translate and low_vram_trans:
                    unload_modules(self, ['textdetector', 'inpainter', 'ocr'])
                    for imgname in pages_to_iterate:
                        if self.stop_requested:
                            LOGGER.info('Translation stopped by user')
                            break
                            
                        blk_list = self.imgtrans_proj.pages[imgname]
                        self.translator.translate_textblk_lst(blk_list)
                        self.translate_counter += 1
                        self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_TRANSLATE)
                        self.update_translate_progress.emit(self.translate_counter)

                if self.stop_requested and (not cfg_module.enable_translate or not self.parallel_trans):
                    self.pipeline_stopped.emit()

            ImgtransThread._imgtrans_pipeline = _imgtrans_pipeline_chunked
            ImgtransThread._is_chunked_patched = True
            if LOGGER:
                LOGGER.info("Monkey patch applied: ImgtransThread Async Pipeline (Paddle VL)")

    except ImportError:
        pass


@register_OCR('PaddleOCRVLManga')
class PaddleOCRVLManga(OCRBase):
    # ... existing params ...
    params = {
        'device': DEVICE_SELECTOR(),
        "batch_size": {
            "value": 4,
            "description": "Number of text blocks to process simultaneously. Higher values are faster but use more VRAM."
        },
        "max_new_tokens": {
            "value": 512,
            "description": "Max generation tokens"
        },
        "retry_attempts": {
            "value": 2,
            "description": "인식 실패(공란) 시 재시도할 횟수입니다."
        }
    }
    device = DEFAULT_DEVICE

    # ... download_file_list and _load_model_keys are inherited or already there, 
    # but we need to inject __init__ to call _apply_chunked_processing_patch()

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.device = self.params['device']['value']
        self.model = None
        self.processor = None
        # Apply patch lazily
        _apply_chunked_processing_patch()

    @property
    def batch_size(self) -> int:
        return int(self.get_param_value("batch_size"))

    def ocr_img(self, img: np.ndarray) -> str:
        # 단일 이미지 처리를 배치 처리의 특수 케이스로 활용
        results = self.ocr_batch([img])
        return results[0] if results else ""

    def ocr_batch(self, images: List[np.ndarray]) -> List[str]:
        if not images:
            return []

        # Prepare messages for each image in the batch
        texts_input = []
        for img in images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "OCR:"},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts_input.append(text)

        # Process batch inputs
        inputs = self.processor(text=texts_input, images=images, return_tensors="pt", padding=True)
        inputs = {
            k: (v.to(self.model.device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }

        # Generate text in batch
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.get_param_value('max_new_tokens'),
                do_sample=False,
                use_cache=True
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = generated[:, input_length:]
        answers = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # Post-process: clean up "OCR:" prefix if it exists in response
        return [ans.strip() for ans in answers]

    def _load_model(self):
        if self.model is None or self.processor is None:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device).eval()

            processor = AutoProcessor.from_pretrained(
                MODEL_PATH, trust_remote_code=True, use_fast=True
            )

            # Set pad_token_id to avoid warning during generation
            if model.generation_config.pad_token_id is None:
                model.generation_config.pad_token_id = processor.tokenizer.eos_token_id
            
            # Ensure processor has padding token for batching
            if processor.tokenizer.pad_token_id is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token

            self.model = model
            self.processor = processor

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
        if not self.all_model_loaded() or self.processor is None:
            self.load_model()
            
        im_h, im_w = img.shape[:2]
        valid_blocks = []
        crops = []

        # 1. 유효한 블록과 크롭 이미지 수집
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            
            # 좌표 보정 시도
            x1_clamped = max(0, min(x1, im_w - 1))
            y1_clamped = max(0, min(y1, im_h - 1))
            x2_clamped = max(0, min(x2, im_w))
            y2_clamped = max(0, min(y2, im_h))
            
            # 보정 후에도 유효성 확인
            if x1_clamped < x2_clamped and y1_clamped < y2_clamped:
                # 좌표가 보정되었다면 경고 출력
                if (x1 != x1_clamped or y1 != y1_clamped or x2 != x2_clamped or y2 != y2_clamped):
                    self.logger.warning(f'Text bbox out of bounds - Original: ({x1},{y1},{x2},{y2}), Image: ({im_w}x{im_h}), Clamped: ({x1_clamped},{y1_clamped},{x2_clamped},{y2_clamped})')
                
                region = img[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
                # RGBA -> RGB (모델 입력 요구사항 대응)
                if region.shape[-1] == 4:
                    region = cv2.cvtColor(region, cv2.COLOR_RGBA2RGB)
                crops.append(region)
                valid_blocks.append(blk)
            else:
                self.logger.warning(f'Invalid textbbox (zero/negative area) - bbox: ({x1},{y1},{x2},{y2}), image size: ({im_w}x{im_h})')
                blk.text = ['']

        # 2. 배치 단위로 추론 실행
        bs = self.batch_size
        retry_limit = int(self.params.get('retry_attempts', {'value': 2})['value'])
        fallback_ocr = None

        for i in range(0, len(crops), bs):
            batch_crops = crops[i : i + bs]
            batch_blks = valid_blocks[i : i + bs]
            
            for attempt in range(retry_limit + 1):
                try:
                    batch_results = self.ocr_batch(batch_crops)
                    
                    # 모든 결과가 채워졌는지 확인
                    all_filled = all(res and res.strip() for res in batch_results)
                    
                    if all_filled or attempt == retry_limit:
                        # 결과 적용 및 폴백 처리
                        for idx, (blk, res) in enumerate(zip(batch_blks, batch_results)):
                            is_empty = not res or (isinstance(res, str) and not res.strip())
                            
                            if is_empty:
                                # Fallback to MangaOCR
                                if fallback_ocr is None:
                                    try:
                                        self.logger.info("Initializing MangaOCR for fallback...")
                                        fallback_ocr = MangaOCR()
                                        fallback_ocr.load_model()
                                    except Exception as e:
                                        self.logger.error(f"Failed to initialize MangaOCR fallback: {e}")
                                        fallback_ocr = False # Mark as failed to avoid retry

                                if fallback_ocr:
                                    try:
                                        fallback_res = fallback_ocr.ocr_img(batch_crops[idx])
                                        if fallback_res and fallback_res.strip():
                                            blk.text = [fallback_res]
                                            self.logger.info(f"Fallback successful for block: {fallback_res}")
                                            continue
                                    except Exception as e:
                                        self.logger.error(f"MangaOCR fallback failed: {e}")

                                blk.text = ['[ERROR]']
                            else:
                                blk.text = [res] if isinstance(res, str) else res
                        break
                    else:
                        self.logger.warning(f"Batch OCR result empty at index {i}, retrying ({attempt + 1}/{retry_limit})...")
                except Exception as e:
                    if attempt == retry_limit:
                        self.logger.exception(f"Batch OCR failed at index {i} after {retry_limit} retries")
                        for blk in batch_blks:
                            blk.text = ['[ERROR]']
                    else:
                        continue

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        device = self.params['device']['value']
        if self.device != device and self.model is not None:
            self.model.to(device)





