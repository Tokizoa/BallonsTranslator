import re
import time
import json
import asyncio
import traceback
import threading
import concurrent.futures
import faulthandler
from typing import List, Dict, Optional, Type
import os
import sys
import csv
import io
import base64
import httpx
import openai

# Global control for layout batching
_V4_GUI_LAYOUT_ENABLED = True
_V4_REPAIR_MODE = False
_V4_LAYOUT_START_TIME = 0.0  # Layout start timestamp for elapsed time tracking
_V4_LAYOUT_COMPLETED_COUNT = 0  # Completed page count for layout progress
_V4_REPAIR_PAGES = None      # Set[str] of page keys to process during repair, None = all pages
_V4_REPAIR_OCR_PAGES = None  # Set[str] of page keys where OCR is also needed (both src+trans empty)
_V4_REPAIR_TOTAL_COUNT = 0   # Total target blocks for repair
_V4_REPAIR_COMPLETED_COUNT = 0 # Completed blocks count during repair
_V4_AUTOLAYOUT_DETAIL_LOGGING = False  # Per-block AutoLayout diagnostics (default off)
_V4_HEADLESS_LAYOUT_CALL = False  # True only around the queued main-thread layout call

# Watchdog: heartbeat updated by save workers; a monitor thread dumps all
# thread stacks when no progress for too long. Diagnostic only - no recovery.
_V4_SAVE_HEARTBEAT = 0.0
_V4_SAVE_HEARTBEAT_INFO = ""  # short description of the last heartbeat event
_V4_WATCHDOG_STOP = None  # threading.Event, created per-run
_V4_STOP_REQUESTED = False  # Global stop flag for immediate cancellation during render/save
from pydantic import BaseModel, Field, ValidationError, RootModel, AliasChoices
from qtpy.QtCore import QObject, Signal, Qt, QRectF
from qtpy.QtGui import QImage, QPainter, QFont, QColor, QPen
from qtpy.QtWidgets import QApplication


def _autolayout_detail_log(level: str, message: str) -> None:
    """Emit expensive per-block layout diagnostics only when explicitly enabled."""
    if not _V4_AUTOLAYOUT_DETAIL_LOGGING or LOGGER is None:
        return
    log_func = getattr(LOGGER, level, LOGGER.debug)
    log_func(message)

# -------------------------------------------------------------------------
# Utility Imports (Safe at module level)
# -------------------------------------------------------------------------
try:
    from utils.logger import logger as LOGGER
    from utils.config import RunStatus, pcfg
    from utils import shared
    # Define Exception locally to avoid circular import with modules.translators package
    class MissingTranslatorParams(Exception): pass
except ImportError:
    LOGGER = None
    RunStatus = None
    pcfg = None
    shared = None
    class MissingTranslatorParams(Exception): pass

# -------------------------------------------------------------------------
# Monkey Patching Setup (Delayed)
# -------------------------------------------------------------------------
# These are delayed to avoid circular imports with ui.module_manager
TranslateThread = None
ImgtransThread = None
ImgTranlsatePipeline = None

def _install_patches():
    global TranslateThread, ImgtransThread, ImgTranlsatePipeline, shared
    
    # Avoid re-patching
    if getattr(_install_patches, '_applied', False):
        return
    
    print("DEBUG: _install_patches running...")
    try:
        import sys
        # Use sys.modules to avoid 'import' statement deadlocks
        if 'ui.module_manager' in sys.modules:
            mm = sys.modules['ui.module_manager']
            TranslateThread = getattr(mm, 'TranslateThread', None)
            ImgtransThread = getattr(mm, 'ImgtransThread', None)
        
        if 'utils.shared' in sys.modules:
            shared = sys.modules['utils.shared']
        else:
            # Fallback if not loaded (unlikely)
            import utils.shared as shared_mod
            shared = shared_mod
            
        if 'utils.proj_imgtrans' in sys.modules:
            pm = sys.modules['utils.proj_imgtrans']
            ImgTranlsatePipeline = getattr(pm, 'ImgTranlsatePipeline', None)
        else:
            from utils.proj_imgtrans import ImgTranlsatePipeline as ITP
            ImgTranlsatePipeline = ITP

        # Backup originals
        if TranslateThread and not hasattr(TranslateThread, '_original_run_translate_pipeline'):
            TranslateThread._original_run_translate_pipeline = TranslateThread._run_translate_pipeline

        if ImgtransThread and not hasattr(ImgtransThread, '_original_imgtrans_pipeline_trans_v4'):
            ImgtransThread._original_imgtrans_pipeline_trans_v4 = ImgtransThread._imgtrans_pipeline

        # Apply patches
        if TranslateThread:
            TranslateThread._run_translate_pipeline = _run_translate_pipeline_patched
            if LOGGER: LOGGER.info("Monkey patch applied: V4 Parallel + Headless Save")

        if ImgtransThread:
            ImgtransThread._imgtrans_pipeline = _imgtrans_pipeline_v4_orchestrator
            if LOGGER: LOGGER.info("Monkey patch applied: V4 Master Orchestrator")
            
        # --- NEW: Patch TextBlock.to_dict to exclude large runtime-only data (MemoryError fix) ---
        try:
            from utils.textblock import TextBlock
            if not hasattr(TextBlock, '_original_to_dict_v4_patch'):
                TextBlock._original_to_dict_v4_patch = TextBlock.to_dict
                _EXCLUDE_KEYS_FROM_DICT = {'region_mask', 'region_inpaint_dict', '_v4_last_layout_text'}
                
                def _patched_to_dict(self, deep_copy=False):
                    blk_dict = {k: v for k, v in vars(self).items() if k not in _EXCLUDE_KEYS_FROM_DICT}
                    if deep_copy:
                        import copy
                        blk_dict = copy.deepcopy(blk_dict)
                    return blk_dict
                
                TextBlock.to_dict = _patched_to_dict
                if LOGGER: LOGGER.info("Monkey patch applied: TextBlock.to_dict (Exclude runtime-only data)")
        except Exception as e:
            if LOGGER: LOGGER.error(f"Failed to patch TextBlock.to_dict: {e}")

        # --- NEW: Patch ParamLineEditor for Drag & Drop JSON ---
        try:
            from ui.module_parse_widgets import ParamLineEditor
            if not hasattr(ParamLineEditor, '_vertex_patched'):
                original_init = ParamLineEditor.__init__
                
                def new_init(self, param_key: str, force_digital, size='short', *args, **kwargs):
                    original_init(self, param_key, force_digital, size, *args, **kwargs)
                    self.setAcceptDrops(True)
                    
                def dragEnterEvent(self, event):
                    if event.mimeData().hasUrls():
                        event.acceptProposedAction()
                    else:
                        event.ignore()
                        
                def dropEvent(self, event):
                    urls = event.mimeData().urls()
                    if urls:
                        file_path = urls[0].toLocalFile()
                        if file_path.lower().endswith('.json'):
                            self.setText(file_path)
                    event.acceptProposedAction()
                    
                ParamLineEditor.__init__ = new_init
                ParamLineEditor.dragEnterEvent = dragEnterEvent
                ParamLineEditor.dropEvent = dropEvent
                ParamLineEditor._vertex_patched = True
                if LOGGER: LOGGER.info("Monkey patch applied: ParamLineEditor Drag & Drop for JSON")
        except Exception as e:
            if LOGGER: LOGGER.error(f"Failed to patch ParamLineEditor: {e}")

        # --- NEW: Patch ProjImgTrans.save for thread safety + debouncing + suppression ---
        if 'utils.proj_imgtrans' in sys.modules:
            try:
                ProjImgTrans = sys.modules['utils.proj_imgtrans'].ProjImgTrans
                if not hasattr(ProjImgTrans, '_original_save_v4_patch'):
                    ProjImgTrans._original_save_v4_patch = ProjImgTrans.save
                    
                    def _patched_save_thread_safe(self, *args, **kwargs):
                        global _HEADLESS_SAVE_IN_PROGRESS, _PIPELINE_ACTIVE, _LAST_SAVE_TIME, _DEBUG_PROFILING
                        import time
                        import gc

                        force_v4 = bool(kwargs.pop('_v4_force', False))

                        # A standalone re-layout owns the project while it renders. During the
                        # translation pipeline, however, periodic and final JSON saves are allowed
                        # and serialized with the exact same project lock as layout mutations.
                        if _HEADLESS_SAVE_IN_PROGRESS and not _PIPELINE_ACTIVE and not force_v4:
                            return

                        # Debounce ordinary pipeline autosaves. The final save bypasses this once.
                        if _PIPELINE_ACTIVE and not force_v4:
                            current_time = time.time()
                            if current_time - _LAST_SAVE_TIME < 15.0:
                                # Skip this save, too soon
                                if _DEBUG_PROFILING and LOGGER:
                                    LOGGER.debug(f"🔬 [PROF-SAVE] skipped (debounce {current_time - _LAST_SAVE_TIME:.2f}s)")
                                return
                            _LAST_SAVE_TIME = current_time

                        # Lazily create a lock for this project instance
                        if not hasattr(self, '_v4_save_lock'):
                            self._v4_save_lock = threading.RLock()

                        # [PROF] Lock wait + IO duration tracking
                        _prof_lock_wait = 0.0
                        _prof_io_dur = 0.0
                        _prof_lock_t0 = time.perf_counter() if _DEBUG_PROFILING else 0.0

                        with self._v4_save_lock:
                            if _DEBUG_PROFILING:
                                _prof_lock_wait = time.perf_counter() - _prof_lock_t0
                                _prof_io_t0 = time.perf_counter()

                            # Import here to avoid issues at definition time
                            from utils.proj_imgtrans import TextBlkEncoder
                            from utils.exceptions import ProjectDirNotExistException

                            # Free memory before serialization
                            gc.collect()

                            if not os.path.exists(self.directory):
                                raise ProjectDirNotExistException

                            tmp_save_tgt = self.proj_path + '.tmp'
                            try:
                                # [메모리 강화] json.dump 딕셔너리 분할 직렬화 (메모리 단편화 원천 방지)
                                with open(tmp_save_tgt, "w", encoding="utf-8") as f:
                                    f.write('{')
                                    f.write('"directory": ' + json.dumps(self.directory, ensure_ascii=False) + ', ')
                                    f.write('"current_img": ' + json.dumps(self.current_img, ensure_ascii=False) + ', ')
                                    f.write('"image_info": ' + json.dumps(self._image_info, ensure_ascii=False, cls=TextBlkEncoder) + ', ')
                                    
                                    f.write('"pages": {')
                                    _all_pages = self.pages.copy()
                                    _all_pages.update(self.not_found_pages)
                                    
                                    _first_page = True
                                    for _p_key, _blk_list in _all_pages.items():
                                        if not _first_page:
                                            f.write(', ')
                                        _first_page = False
                                        f.write(json.dumps(_p_key, ensure_ascii=False) + ': ')
                                        # Dump only a single page to the disk buffer explicitly
                                        json.dump(_blk_list, f, ensure_ascii=False, cls=TextBlkEncoder)
                                    
                                    f.write('}}')
                                
                                gc.collect()
                            except MemoryError:
                                if LOGGER: LOGGER.error(f"MemoryError while saving project to {self.proj_path}. Project data may be too large. Existing save file preserved.")
                                if os.path.exists(tmp_save_tgt):
                                    try:
                                        os.remove(tmp_save_tgt)
                                    except OSError:
                                        pass
                                return  # Don't crash, preserve existing save file
                            except Exception as e:
                                if LOGGER: LOGGER.error(f"Failed to save project to {self.proj_path}: {e}")
                                if os.path.exists(tmp_save_tgt):
                                    try:
                                        os.remove(tmp_save_tgt)
                                    except OSError:
                                        pass
                                return

                            # Atomic replace: tmp -> final
                            keep_exist_as_backup = kwargs.get('keep_exist_as_backup', False)
                            if len(args) > 0:
                                keep_exist_as_backup = args[0]

                            if os.path.exists(self.proj_path) and keep_exist_as_backup:
                                os.replace(self.proj_path, self.proj_path + '.backup')
                            os.replace(tmp_save_tgt, self.proj_path)
                            if LOGGER and not _REPLACE_RERENDER_ACTIVE:
                                LOGGER.debug(f'project saved to {self.proj_path}')

                            if _DEBUG_PROFILING and LOGGER:
                                _prof_io_dur = time.perf_counter() - _prof_io_t0
                                # Noise reduction: only log if above threshold
                                if _prof_io_dur > 0.1 or _prof_lock_wait > 0.1:
                                    try:
                                        _prof_size = os.path.getsize(self.proj_path)
                                    except Exception:
                                        _prof_size = -1
                                    LOGGER.info(
                                        f"🔬 [PROF-SAVE] lock_wait={_prof_lock_wait:.3f}s "
                                        f"io={_prof_io_dur:.3f}s size={_prof_size}B"
                                    )
                    
                    ProjImgTrans.save = _patched_save_thread_safe
                    if LOGGER: LOGGER.info("Monkey patch applied: ProjImgTrans.save (Thread-Safe + Debounce)")
            except Exception as e:
                if LOGGER: LOGGER.error(f"Failed to patch ProjImgTrans.save: {e}")
            
        # --- NEW: Patch SceneTextManager.layout_textblk for Font Size Issue ---
        if 'ui.scenetext_manager' in sys.modules:
            try:
                stm = sys.modules['ui.scenetext_manager']
                SceneTextManager = stm.SceneTextManager
                
                if not hasattr(SceneTextManager, '_original_layout_textblk_patch'):
                    SceneTextManager._original_layout_textblk_patch = SceneTextManager.layout_textblk
                    
                    def _patched_layout_textblk(self, blkitem, text: str = None, mask=None, bounding_rect: List = None, region_rect: List = None):
                        # [Universal Guard] Completely suppress layout during active pipeline or stop requested
                        global _V4_GUI_LAYOUT_ENABLED, _V4_SAVE_HEARTBEAT, _V4_SAVE_HEARTBEAT_INFO, _V4_STOP_REQUESTED, _V4_HEADLESS_LAYOUT_CALL
                        if (not _V4_GUI_LAYOUT_ENABLED and not _V4_HEADLESS_LAYOUT_CALL) or _V4_STOP_REQUESTED:
                            return

                        # [Diag] Entry log - DEBUG level to avoid flooding logs on 1300-page batches
                        try:
                            _autolayout_detail_log(
                                "debug",
                                f"[Layout] blk={getattr(blkitem, 'idx', '?')} "
                                f"txt_len={len(text) if text else -1} vert={blkitem.blk.vertical}",
                            )
                        except Exception:
                            pass

                        # Imports needed for this function
                        import numpy as np
                        import cv2
                        from qtpy.QtGui import QFont, QFontMetricsF, QTextCursor
                        from utils.config import pcfg
                        from utils.imgproc_utils import extract_ballon_region
                        from utils.text_processing import seg_text, is_cjk
                        from utils.text_layout import layout_text
                        from ui.scenetext_manager import get_text_size, get_words_length_list

                        img = self.imgtrans_proj.img_array
                        if img is None:
                            return

                        src_is_cjk = is_cjk(pcfg.module.translate_source)
                        tgt_is_cjk = is_cjk(pcfg.module.translate_target)

                        # disable for vertical writing
                        if blkitem.blk.vertical:
                            return
                        
                        old_br = blkitem.absBoundingRect(qrect=True)
                        old_br = [old_br.x(), old_br.y(), old_br.width(), old_br.height()]
                        if old_br[2] < 1:
                            return

                        blk_font = blkitem.font()
                        fmt = blkitem.get_fontformat()
                        blk_font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, fmt.letter_spacing * 100)
                        text_size_func = lambda text: get_text_size(QFontMetricsF(blk_font), text)

                        restore_charfmts = False
                        if text is None:
                            text = blkitem.toPlainText()
                            restore_charfmts = True

                        if not text.strip():
                            return

                        if mask is None:
                            im_h, im_w = img.shape[:2]
                            bounding_rect = blkitem.absBoundingRect(max_h=im_h, max_w=im_w)
                            
                            if bounding_rect[2] <= 0 or bounding_rect[3] <= 0:
                                blkitem.setPlainText(text)
                                if len(self.pairwidget_list) > blkitem.idx:
                                    self.pairwidget_list[blkitem.idx].e_trans.setPlainText(text)
                                return
                            
                            # Standard enlarge_ratio formula for parity
                            if tgt_is_cjk:
                                max_enf = 2.5
                            else:
                                max_enf = 3.0
                            w, h = bounding_rect[2], bounding_rect[3]
                            enlarge_ratio = min(max(w/h, h/w) * 1.5, max_enf)
                            
                            mask, ballon_area, mask_xyxy, region_rect = extract_ballon_region(img, bounding_rect, enlarge_ratio=enlarge_ratio, cal_region_rect=True)

                        else:
                            # [Fix] Ensure bounding_rect is valid when mask is provided
                            if bounding_rect is None:
                                if region_rect is not None:
                                    bounding_rect = region_rect
                                else:
                                    im_h, im_w = img.shape[:2]
                                    bounding_rect = blkitem.absBoundingRect(max_h=im_h, max_w=im_w)

                            if bounding_rect is None or len(bounding_rect) < 4:
                                if LOGGER: LOGGER.warning("AutoLayout: Invalid bounding_rect in patched layout")
                                return

                            mask_xyxy = [bounding_rect[0], bounding_rect[1], bounding_rect[0]+bounding_rect[2], bounding_rect[1]+bounding_rect[3]]

                            
                            
                            # [Fix] Removed custom erosion/padding logic.
                            # The goal is strict parity with the original 'Manual Auto Layout'.
                            # Original logic uses the raw mask from 'extract_ballon_region' without extra margins.
                            
                            # [Fix] Must define ballon_area for adaptive logic below
                            ballon_area = (mask > 0).sum()

                        
                        words, delimiter = seg_text(text, pcfg.module.translate_target)
                        if len(words) < 1:
                            return

                        wl_list = get_words_length_list(QFontMetricsF(blk_font), words)
                        text_w, text_h = text_size_func(text)
                        text_area = text_w * text_h
                        if text_area < 1:  # Guard: degenerate text metrics
                            text_area = 1
                        if tgt_is_cjk:
                            line_height = int(round(fmt.line_spacing * text_size_func('X木')[1]))
                        else:
                            line_height = int(round(fmt.line_spacing * text_size_func('X')[1]))
                        delimiter_len = text_size_func(delimiter)[0]
                
                        ref_src_lines = False
                        if not blkitem.blk.src_is_vertical:
                            ref_src_lines = blkitem.blk.line_coord_valid(old_br)

                        adaptive_fntsize = False
                        resize_ratio = 1
                        if self.auto_textlayout_flag and pcfg.let_fntsize_flag == 0 and pcfg.let_autolayout_flag:
                            # [Fix] Logic aligned with SceneTextManager original implementation
                            if blkitem.blk.src_is_vertical and blkitem.blk.vertical != blkitem.blk.src_is_vertical:
                                adaptive_fntsize = True
                                area_ratio = ballon_area / text_area  # text_area guaranteed >= 1 by guard above
                                ballon_area_thresh = 1.7
                                downscale_constraint = 0.6
                                # Safe max calculation
                                max_wl = max(wl_list) if wl_list else 1
                                max_wl = max(max_wl, 1)  # Guard: prevent ZeroDivisionError
                                region_w = max(region_rect[2], 1)  # Guard: region width can be 0
                                resize_ratio = np.clip(min(area_ratio / ballon_area_thresh, region_w / max_wl), downscale_constraint, 1.0)
                                _autolayout_detail_log(
                                    "info",
                                    f"AutoLayout: Adaptive Resize (Vertical->Horizontal) Ratio={resize_ratio:.2f}",
                                )

                            else:
                                # [MODIFIED] Width Constraint Logic
                                # Safe max calculation
                                max_wl = max(wl_list) if wl_list else 1
                                max_wl = max(max_wl, 1)  # Guard: prevent ZeroDivisionError
                                region_w = max(region_rect[2], 1)  # Guard: region width can be 0
                                width_ratio = region_w / max_wl
                                if not src_is_cjk:
                                    resize_ratio_ballon = max(ballon_area / 1.2 / text_area, 0.7)  # text_area >= 1
                                    if ref_src_lines:
                                        _, src_width = blkitem.blk.normalizd_width_list(normalize=False)
                                        _wl_denom = sum(wl_list) + max((len(wl_list) - 1 - len(blkitem.blk.lines_array())), 0) * delimiter_len
                                        _wl_denom = max(_wl_denom, 1)  # Guard: prevent ZeroDivisionError
                                        resize_ratio_src = src_width / _wl_denom
                                        resize_ratio = min(resize_ratio_ballon, resize_ratio_src, width_ratio)
                                    else:
                                        resize_ratio = min(resize_ratio_ballon, width_ratio)
                                elif not blkitem.blk.src_is_vertical and ref_src_lines:
                                    _, src_width = blkitem.blk.normalizd_width_list(normalize=False)
                                    _wl_denom = sum(wl_list) + max((len(wl_list) - 1 - len(blkitem.blk.lines_array())), 0) * delimiter_len
                                    _wl_denom = max(_wl_denom, 1)  # Guard: prevent ZeroDivisionError
                                    resize_ratio_src = src_width / _wl_denom
                                    resize_ratio = max(resize_ratio_src * 1.5, 0.5)
                                    resize_ratio = min(resize_ratio, width_ratio)
                                resize_ratio = min(max(resize_ratio, 0.6), 1)

                        if resize_ratio != 1:
                            new_font_size = blk_font.pointSizeF() * resize_ratio   
                            blk_font.setPointSizeF(new_font_size)
                            wl_list = (np.array(wl_list, np.float64) * resize_ratio).astype(np.int32).tolist()
                            line_height = int(line_height * resize_ratio)
                            text_w = int(text_w * resize_ratio)
                            delimiter_len = int(delimiter_len * resize_ratio)

                        max_central_width = np.inf
                        if fmt.alignment == 1:
                            if len(blkitem.blk) > 0:
                                centroid = blkitem.blk.center().astype(np.int64).tolist()
                                centroid[0] -= mask_xyxy[0]
                                centroid[1] -= mask_xyxy[1]
                            else:
                                centroid = [bounding_rect[2] // 2, bounding_rect[3] // 2]
                        else:
                            max_central_width = np.inf
                            centroid = [0, 0]
                            abs_centroid = [bounding_rect[0], bounding_rect[1]]
                            if len(blkitem.blk) > 0:
                                blkitem.blk.lines[0]
                                abs_centroid = blkitem.blk.lines[0][0]
                                centroid[0] = int(abs_centroid[0] - mask_xyxy[0])
                                centroid[1] = int(abs_centroid[1] - mask_xyxy[1])

                        # [Diag] Log inputs right before the (historically-hanging) layout_text call.
                        # The last occurrence of this INFO line before a freeze pinpoints the hang site.
                        try:
                            _mask_shape = getattr(mask, 'shape', None)
                            _page = getattr(getattr(self, 'imgtrans_proj', None), 'current_img', '?')
                            _autolayout_detail_log(
                                "info",
                                f"[Layout->text] page={_page} blk={getattr(blkitem, 'idx', '?')} "
                                f"mask={_mask_shape} words={len(words)} lh={line_height} "
                                f"rr={resize_ratio:.3f} adaptive={adaptive_fntsize}",
                            )
                        except Exception:
                            pass
                        _V4_SAVE_HEARTBEAT = time.time()
                        _V4_SAVE_HEARTBEAT_INFO = f"layout_text:blk={getattr(blkitem, 'idx', '?')}"
                        try:
                            new_text, xywh, start_from_top, adjust_xy = layout_text(
                                blkitem.blk,
                                mask,
                                mask_xyxy,
                                centroid,
                                words,
                                wl_list,
                                delimiter,
                                delimiter_len,
                                line_height,
                                0,
                                max_central_width,
                                src_is_cjk=src_is_cjk,
                                tgt_is_cjk=tgt_is_cjk,
                                ref_src_lines=ref_src_lines
                            )
                        except Exception:
                            if LOGGER:
                                _page = getattr(getattr(self, 'imgtrans_proj', None), 'current_img', '?')
                                LOGGER.exception(
                                    f"[Layout->text] FAILED page={_page} blk={getattr(blkitem, 'idx', '?')}"
                                )
                            # Skip this block - do not kill the page or the batch.
                            return False
                        _V4_SAVE_HEARTBEAT = time.time()
                        try:
                            _autolayout_detail_log(
                                "info",
                                f"[Layout/ok] blk={getattr(blkitem, 'idx', '?')} "
                                f"xywh={xywh} lines={new_text.count(chr(10)) + 1}",
                            )
                        except Exception:
                            pass

                        # font size post adjustment
                        post_resize_ratio = 1
                        if adaptive_fntsize:
                            downscale_constraint = 0.5
                            w = xywh[2]
                            if w < 1:  # Guard: layout_text returned zero-width result
                                if LOGGER: LOGGER.warning(f"[Layout] Skipping post-resize: xywh width is 0 (blk={getattr(blkitem, 'idx', '?')})")
                                return False
                            post_resize_ratio = np.clip(max(region_rect[2] / w, downscale_constraint), 0, 1)
                            resize_ratio *= post_resize_ratio

                        if post_resize_ratio != 1:
                            cx, cy = xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2
                            w, h = xywh[2] * post_resize_ratio, xywh[3] * post_resize_ratio
                            xywh = [int(cx - w / 2), int(cy - h / 2), int(w), int(h)]

                        if resize_ratio != 1:
                            new_font_size = blkitem.font().pointSizeF() * resize_ratio
                            blkitem.textCursor().clearSelection()
                            blkitem.setFontSize(new_font_size)
                            blk_font.setPointSizeF(new_font_size)

                        if restore_charfmts:
                            char_fmts = blkitem.get_char_fmts()

                        ffmt = QFontMetricsF(blk_font)
                        maxw = max([ffmt.horizontalAdvance(t) for t in new_text.split('\n')])

                        # [Signal Guard] Block QTextDocument signals during the mutation burst so
                        # HorizontalTextDocumentLayout.documentChanged / size_enlarged don't
                        # re-enter layout logic thousands of times across a 1,300-page batch.
                        _doc = None
                        _prev_blocked = False
                        _had_block_change_flag = hasattr(blkitem, 'block_change_signal')
                        _prev_block_change_flag = getattr(blkitem, 'block_change_signal', False)
                        try:
                            try:
                                _doc = blkitem.document()
                            except Exception:
                                _doc = None
                            if _doc is not None:
                                try:
                                    _prev_blocked = _doc.blockSignals(True)
                                except Exception:
                                    _doc = None
                            if _had_block_change_flag:
                                try:
                                    blkitem.block_change_signal = True
                                except Exception:
                                    pass

                            blkitem.set_size(maxw * 1.5, xywh[3], set_layout_maxsize=True)
                            blkitem.setPlainText(new_text)
                            if len(self.pairwidget_list) > blkitem.idx:
                                self.pairwidget_list[blkitem.idx].e_trans.setPlainText(new_text)
                            if restore_charfmts:
                                self.restore_charfmts(blkitem, text, new_text, char_fmts)
                            blkitem.squeezeBoundingRect()
                        finally:
                            if _doc is not None:
                                try:
                                    _doc.blockSignals(_prev_blocked)
                                except Exception:
                                    pass
                            if _had_block_change_flag:
                                try:
                                    blkitem.block_change_signal = _prev_block_change_flag
                                except Exception:
                                    pass
                        return True

                    SceneTextManager.layout_textblk = _patched_layout_textblk
                    if LOGGER: LOGGER.info("Monkey patch applied: SceneTextManager.layout_textblk (Font Size Width Fix)")

                # [NEW] Patch updateTranslation to force auto-layout on finish
                def _patched_updateTranslation(self):
                    global _V4_GUI_LAYOUT_ENABLED
                    # Call original to set text
                    SceneTextManager._original_updateTranslation_patch(self)
                    
                    # [Performance Optimization] Skip automated layout during active V4 pipeline processing.
                    # This prevents O(N^2) slowdown as the project grows.
                    if not _V4_GUI_LAYOUT_ENABLED:
                        return

                    # [Force Layout] Apply layout regardless of checking flags initially
                    # --- AutoLayout Progress ---
                    global _V4_LAYOUT_START_TIME, _V4_LAYOUT_COMPLETED_COUNT
                    _current_page = getattr(self.imgtrans_proj, 'current_img', '?')
                    _total = len(self.imgtrans_proj.pages) if hasattr(self.imgtrans_proj, 'pages') else '?'
                    _page_keys = list(self.imgtrans_proj.pages.keys()) if hasattr(self.imgtrans_proj, 'pages') else []
                    _page_idx = _page_keys.index(_current_page) + 1 if _current_page in _page_keys else '?'
                    _V4_LAYOUT_COMPLETED_COUNT += 1
                    from datetime import datetime as _dt
                    _time_str = _dt.now().strftime('%H:%M:%S')
                    if LOGGER: LOGGER.info(f"\U0001f4d0 AutoLayout: \ud398\uc774\uc9c0 [{_page_idx}/{_total}] (\uc644\ub8cc: {_V4_LAYOUT_COMPLETED_COUNT}/{_total}) \u23f1 {_time_str} - {_current_page}")

                    old_stm_flag = self.auto_textlayout_flag
                    old_pcfg_auto = pcfg.let_autolayout_flag
                    old_pcfg_size = pcfg.let_fntsize_flag
                    
                    self.auto_textlayout_flag = True
                    pcfg.let_autolayout_flag = True
                    pcfg.let_fntsize_flag = 0
                    
                    try:
                        for blk_item in self.textblk_item_list:
                            if not blk_item.blk.vertical:
                                # [Performance Optimization] O(N) Skip check
                                current_text = blk_item.blk.translation or blk_item.toPlainText()
                                if current_text == getattr(blk_item.blk, '_v4_last_layout_text', ''):
                                    continue

                                # [Fix] Pre-Widening Strategy: Temporarily widen box for balloon search
                                if getattr(blk_item.blk, 'src_is_vertical', False):
                                    x1, y1, x2, y2 = blk_item.blk.xyxy
                                    nw, nh = x2 - x1, y2 - y1
                                    new_w = max(nw, nh * 1.5)
                                    cx = (x1 + x2) / 2
                                    blk_item.blk.xyxy[0] = int(cx - new_w / 2)
                                    blk_item.blk.xyxy[2] = int(cx + new_w / 2)
                                    blk_item.setRect(blk_item.blk.bounding_rect())
                                
                                try:
                                    self.layout_textblk(blk_item)
                                    # Mark as processed
                                    blk_item.blk._v4_last_layout_text = current_text
                                except Exception as e:
                                    if LOGGER: LOGGER.error(f"Layout failed for item {blk_item.idx}: {e}")
                    finally:
                        # Restore flags
                        self.auto_textlayout_flag = old_stm_flag
                        pcfg.let_autolayout_flag = old_pcfg_auto
                        pcfg.let_fntsize_flag = old_pcfg_size

                if not hasattr(SceneTextManager, '_original_updateTranslation_patch'):
                    SceneTextManager._original_updateTranslation_patch = SceneTextManager.updateTranslation
                
                SceneTextManager.updateTranslation = _patched_updateTranslation
                if LOGGER: LOGGER.info("Monkey patch applied: SceneTextManager.updateTranslation (Auto Layout Trigger)")

            except Exception as e:
                if LOGGER: LOGGER.error(f"Failed to patch SceneTextManager.layout_textblk: {e}")

        # --- NEW: Patch GlobalSearchWidget.set_document_edited for Result Preservation ---
        if 'ui.global_search_widget' in sys.modules:
            try:
                gsw = sys.modules['ui.global_search_widget']
                GlobalSearchWidget = gsw.GlobalSearchWidget
                
                if not hasattr(GlobalSearchWidget, '_original_set_document_edited_patch'):
                    GlobalSearchWidget._original_set_document_edited_patch = GlobalSearchWidget.set_document_edited
                    
                    def _patched_set_document_edited(self):
                        """검색 결과를 삭제하지 않고 메시지만 업데이트 + 자동 새로고침(Debounced)"""
                        
                        # [Guard] 모두 바꾸기 후 렌더링 중에는 완전 무시 (clearPages 호출 시 루프 중 검색 트리 파손 방지)
                        if getattr(self, '_is_replacing', False):
                            return

                        # [Recursion Guard] If we are currently auto-searching, ignore updates triggered by it.
                        if getattr(self, '_is_auto_searching', False):
                            return

                        # 결과가 있으면 삭제하지 않고 메시지만 변경 (기존 유지)
                        if self.counter_sum > 0:
                            self.result_label.setText(self.doc_edited_str)
                            
                        # [Auto-Refresh] Trigger search after delay (Debounce)
                        # This restores the "Live Update" feel without crashing due to recursion loops.
                        try:
                            from qtpy.QtCore import QTimer
                            if not hasattr(self, '_debounce_timer'):
                                self._debounce_timer = QTimer(self)
                                self._debounce_timer.setSingleShot(True)
                                
                                def do_search():
                                    if getattr(self, '_is_auto_searching', False): return
                                    self._is_auto_searching = True
                                    try:
                                        if LOGGER: LOGGER.info("Auto-refreshing search results (Debounced)...")
                                        self.commit_search()
                                    except Exception as e:
                                        if LOGGER: LOGGER.error(f"Auto-search failed: {e}")
                                    finally:
                                        # Small delay before releasing lock to prevent trailing signals
                                        QTimer.singleShot(100, lambda: setattr(self, '_is_auto_searching', False))
                                        
                                self._debounce_timer.timeout.connect(do_search)
                            
                            # Restart timer on every keypress (Debounce 800ms)
                            self._debounce_timer.start(800)
                        except Exception as e:
                            if LOGGER: LOGGER.error(f"Failed to setup auto-refresh timer: {e}")
                    
                    GlobalSearchWidget.set_document_edited = _patched_set_document_edited
                    if LOGGER: LOGGER.info("Monkey patch applied: GlobalSearchWidget.set_document_edited (Result Preservation)")
            except Exception as e:
                if LOGGER: LOGGER.error(f"Failed to patch GlobalSearchWidget.set_document_edited: {e}")

        # --- Patch on_pagtrans_finished for MemoryError safety ---
        if 'ui.mainwindow' in sys.modules:
            try:
                _MainWindow = sys.modules['ui.mainwindow'].MainWindow
                if not hasattr(_MainWindow, '_original_on_pagtrans_finished_v4'):
                    _MainWindow._original_on_pagtrans_finished_v4 = _MainWindow.on_pagtrans_finished

                    def _patched_on_pagtrans_finished(self, page_index: int):
                        import gc
                        import numpy as np
                        
                        # [메모리 핵심 수정] 파이프라인 실행 중에는 이미지 로드를 완전히 스킵
                        # on_pagtrans_finished는 매 페이지마다 호출되며, 원본 함수가 set_current_img을
                        # 통해 img_array + mask_array + inpainted_array 3개 배열을 풀 로드합니다.
                        # 1000+ 페이지 파이프라인에서 이는 불필요한 메모리 낭비입니다.
                        if _PIPELINE_ACTIVE:
                            if LOGGER: LOGGER.debug(f"Pipeline active, skipping page display for index {page_index}")
                            return
                        
                        # 파이프라인 비활성 시 (수동 작업 등): 기존 로직 유지
                        try:
                            proj = getattr(self, 'imgtrans_proj', None)
                            if proj is not None:
                                for attr in ('img_array', 'mask_array', 'inpainted_array'):
                                    old = getattr(proj, attr, None)
                                    if old is not None:
                                        setattr(proj, attr, None)
                                        del old
                        except Exception:
                            pass
                        gc.collect()
                        try:
                            _MainWindow._original_on_pagtrans_finished_v4(self, page_index)
                        except (MemoryError, np.core._exceptions._ArrayMemoryError):
                            gc.collect()
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception: pass
                            gc.collect()
                            if LOGGER: LOGGER.warning(f"MemoryError on page {page_index}, retrying after aggressive gc...")
                            try:
                                _MainWindow._original_on_pagtrans_finished_v4(self, page_index)
                            except Exception as e2:
                                if LOGGER: LOGGER.error(f"Page {page_index} display failed even after gc: {e2}")

                    _MainWindow.on_pagtrans_finished = _patched_on_pagtrans_finished
                    if LOGGER: LOGGER.info("Monkey patch applied: MainWindow.on_pagtrans_finished (MemoryError safety)")
            except Exception as e:
                if LOGGER: LOGGER.error(f"Failed to patch on_pagtrans_finished: {e}")

        # --- Patch on_blktrans_finished for auto-save after individual block translation ---
        if 'ui.mainwindow' in sys.modules:
            try:
                _MainWindow = sys.modules['ui.mainwindow'].MainWindow
                if not hasattr(_MainWindow, '_original_on_blktrans_finished_v4'):
                    _MainWindow._original_on_blktrans_finished_v4 = _MainWindow.on_blktrans_finished

                    def _patched_on_blktrans_finished(self, mode, blk_ids):
                        _MainWindow._original_on_blktrans_finished_v4(self, mode, blk_ids)
                        if len(blk_ids) >= 1:
                            # Auto-layout after translation (mode != 0 means translation was performed)
                            # Guard: _blktrans_pipeline emits finish_blktrans twice (after OCR and after translation).
                            # Only run layout when translation is actually present to avoid duplicate work.
                            if mode != 0 and mode < 3:
                                stm = self.st_manager
                                has_translation = any(
                                    idx < len(stm.textblk_item_list) and stm.textblk_item_list[idx].blk.translation
                                    for idx in blk_ids
                                )
                                if not has_translation:
                                    if LOGGER: LOGGER.debug(f"Auto-layout skipped: no translation yet (OCR stage)")
                                else:
                                    try:
                                        old_flag = stm.auto_textlayout_flag
                                        old_pcfg_auto = pcfg.let_autolayout_flag
                                        old_pcfg_size = pcfg.let_fntsize_flag
                                        stm.auto_textlayout_flag = True
                                        pcfg.let_autolayout_flag = True
                                        pcfg.let_fntsize_flag = 0
                                        try:
                                            for idx in blk_ids:
                                                if idx < len(stm.textblk_item_list):
                                                    blk_item = stm.textblk_item_list[idx]
                                                    if not blk_item.blk.vertical:
                                                        stm.layout_textblk(blk_item)
                                        finally:
                                            stm.auto_textlayout_flag = old_flag
                                            pcfg.let_autolayout_flag = old_pcfg_auto
                                            pcfg.let_fntsize_flag = old_pcfg_size
                                        if LOGGER: LOGGER.info(f"Auto-layout applied after blktrans for {len(blk_ids)} block(s)")
                                    except Exception as e:
                                        if LOGGER: LOGGER.error(f"Auto-layout after blktrans failed: {e}")
                            try:
                                self.saveCurrentPage(
                                    update_scene_text=True,
                                    save_proj=True,
                                    restore_interface=True
                                )
                            except Exception as e:
                                if LOGGER: LOGGER.error(f"Auto-save after blktrans failed: {e}")

                    _MainWindow.on_blktrans_finished = _patched_on_blktrans_finished
                    if LOGGER: LOGGER.info("Monkey patch applied: MainWindow.on_blktrans_finished (Auto-save + Auto-layout)")

                    # CRITICAL: PyQt signal.connect() captures a bound method reference.
                    # Since MainWindow.__init__ ran BEFORE this patch,
                    # module_manager.blktrans_pipeline_finished still points to the OLD method.
                    # We must reconnect on existing instances.
                    try:
                        from qtpy.QtWidgets import QApplication as _QApp2
                        import types
                        _app2 = _QApp2.instance()
                        if _app2:
                            _orig_blktrans_func = _MainWindow._original_on_blktrans_finished_v4
                            for _w2 in _app2.topLevelWidgets():
                                if isinstance(_w2, _MainWindow):
                                    try:
                                        _w2.module_manager.blktrans_pipeline_finished.disconnect(
                                            types.MethodType(_orig_blktrans_func, _w2)
                                        )
                                    except (TypeError, RuntimeError):
                                        try:
                                            _w2.module_manager.blktrans_pipeline_finished.disconnect()
                                        except (TypeError, RuntimeError):
                                            pass
                                    _w2.module_manager.blktrans_pipeline_finished.connect(_w2.on_blktrans_finished)
                                    if LOGGER: LOGGER.info("Reconnected blktrans_pipeline_finished signal to patched on_blktrans_finished")
                                    break
                    except Exception as _sig_e:
                        if LOGGER: LOGGER.warning(f"Failed to reconnect blktrans_pipeline_finished signal: {_sig_e}")
            except Exception as e:
                if LOGGER: LOGGER.error(f"Failed to patch on_blktrans_finished: {e}")

        # --- Patch GlobalRepalceAllCommand for reliable multi-textbox replacement ---
        try:
            from ui.textedit_commands import GlobalRepalceAllCommand
            if not hasattr(GlobalRepalceAllCommand, '_original_init_v4'):
                GlobalRepalceAllCommand._original_init_v4 = GlobalRepalceAllCommand.__init__

                def _patched_global_replace_init(self, sceneitem_list, background_list, target_text, proj):
                    # Block document signals during replacement to prevent
                    # on_textstack_changed -> set_document_edited() cascade
                    blocked_docs = []
                    for trans_dict in sceneitem_list.get('trans', []):
                        edit = trans_dict.get('edit')
                        if edit and hasattr(edit, 'document'):
                            doc = edit.document()
                            if not doc.signalsBlocked():
                                doc.blockSignals(True)
                                blocked_docs.append(doc)
                    for src_dict in sceneitem_list.get('src', []):
                        edit = src_dict.get('edit')
                        if edit and hasattr(edit, 'document'):
                            doc = edit.document()
                            if not doc.signalsBlocked():
                                doc.blockSignals(True)
                                blocked_docs.append(doc)
                    try:
                        GlobalRepalceAllCommand._original_init_v4(self, sceneitem_list, background_list, target_text, proj)
                    finally:
                        for doc in blocked_docs:
                            doc.blockSignals(False)

                GlobalRepalceAllCommand.__init__ = _patched_global_replace_init
                if LOGGER: LOGGER.info("Monkey patch applied: GlobalRepalceAllCommand.__init__ (Signal blocking)")
        except Exception as e:
            if LOGGER: LOGGER.error(f"Failed to patch GlobalRepalceAllCommand: {e}")

        # --- Patch on_replace_rerender to also update TransTextEdit ---
        if 'ui.global_search_widget' in sys.modules:
            try:
                gsw = sys.modules['ui.global_search_widget']
                GlobalSearchWidget = gsw.GlobalSearchWidget
                if not hasattr(GlobalSearchWidget, '_original_on_replace_rerender_v4'):
                    GlobalSearchWidget._original_on_replace_rerender_v4 = GlobalSearchWidget.on_replace_rerender

                    def _patched_on_replace_rerender(self):
                        """Optimized: current page via UI, others via in-memory TextBlock edit."""
                        if self.counter_sum < 1:
                            return
                        pattern = self.replace_thread.searched_pattern
                        if pattern is None:
                            return

                        from qtpy.QtWidgets import QMessageBox, QApplication
                        from qtpy.QtGui import QTextDocument as _QTD
                        from ui.misc import doc_replace

                        msg = QMessageBox()
                        msg.setText(self.tr('Replace all occurrences re-render all pages? It can\'t be undone.'))
                        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                        ret = msg.exec_()
                        if ret == QMessageBox.StandardButton.No:
                            return

                        global _REPLACE_RERENDER_ACTIVE
                        self._is_replacing = True
                        _REPLACE_RERENDER_ACTIVE = True
                        try:
                            total = self.search_tree.rowCount()
                            target = self.replace_editor.toPlainText()
                            replace_src = self.range_combobox.currentIndex() != 0
                            replace_trans = self.range_combobox.currentIndex() != 1
                            current_img = self.imgtrans_proj.current_img

                            # Temp doc for processing rich_text on non-current pages
                            temp_doc = _QTD()
                            temp_doc.setUndoRedoEnabled(False)

                            for ii in range(total):
                                page_rst_item = self.search_tree.sm.item(ii, 0)
                                pagename = page_rst_item.pagename

                                if LOGGER: LOGGER.info(f"Replace & Rerender [{ii+1}/{total}]: {pagename}")

                                if pagename == current_img:
                                    # === Current page: modify via live UI widgets ===
                                    if replace_src:
                                        for idx in page_rst_item.blkid2match['src']:
                                            src = self.replace_thread.pairwidget_list[idx].e_source
                                            src.setPlainText(pattern.sub(target, src.toPlainText()))

                                    if replace_trans:
                                        for idx, rstitem_list in page_rst_item.blkid2match['trans'].items():
                                            item = self.textblk_item_list[idx]
                                            span_list = [[rstitem.start, rstitem.end] for rstitem in rstitem_list]
                                            doc_replace(item.document(), span_list, target)
                                            if idx < len(self.pairwidget_list):
                                                self.pairwidget_list[idx].e_trans.setPlainText(item.toPlainText())
                                else:
                                    # === Non-current pages: modify TextBlock in memory (no page load!) ===
                                    page_blks = self.imgtrans_proj.pages[pagename]

                                    if replace_src:
                                        for idx in page_rst_item.blkid2match['src']:
                                            blk = page_blks[idx]
                                            blk.text = pattern.sub(target, blk.get_text())

                                    if replace_trans:
                                        for idx, rstitem_list in page_rst_item.blkid2match['trans'].items():
                                            blk = page_blks[idx]
                                            if blk.rich_text:
                                                temp_doc.setHtml(blk.rich_text)
                                                span_list = [[rstitem.start, rstitem.end] for rstitem in rstitem_list]
                                                doc_replace(temp_doc, span_list, target)
                                                blk.rich_text = temp_doc.toHtml()
                                                blk.translation = temp_doc.toPlainText()
                                            else:
                                                blk.translation = pattern.sub(target, blk.translation)

                                QApplication.processEvents()

                            # Save all changes (current page save includes the project JSON)
                            if total > 0:
                                if LOGGER: LOGGER.info("Replace & Rerender: Saving project...")
                                self.page_set = set()
                                self.num_pages = 0
                                self.fin_page_counter = 0
                                self.req_move_page.emit(current_img, True)

                        except Exception as e:
                            if LOGGER: LOGGER.error(f"Replace & Rerender failed: {e}")
                            import traceback
                            traceback.print_exc()
                        finally:
                            _REPLACE_RERENDER_ACTIVE = False
                            self._is_replacing = False
                            self.progress_bar.hide()
                        # Clean up search results (after _is_replacing is cleared)
                        GlobalSearchWidget._original_set_document_edited_patch(self)

                    GlobalSearchWidget.on_replace_rerender = _patched_on_replace_rerender
                    if LOGGER: LOGGER.info("Monkey patch applied: GlobalSearchWidget.on_replace_rerender (TransTextEdit sync)")

                    # [CRITICAL] PyQt6 signal.connect() captures a bound method reference.
                    # Since GlobalSearchWidget.__init__ ran BEFORE this patch,
                    # the button signal still points to the OLD function.
                    # We must reconnect existing instances' button signals.
                    try:
                        from qtpy.QtWidgets import QApplication as _QApp
                        _app = _QApp.instance()
                        if _app:
                            for _w in _app.allWidgets():
                                if isinstance(_w, GlobalSearchWidget):
                                    _w.replace_rerender_btn.clicked.disconnect()
                                    _w.replace_rerender_btn.clicked.connect(_w.on_replace_rerender)
                                    if LOGGER: LOGGER.info("Reconnected replace_rerender_btn signal to patched method")
                    except Exception as _e:
                        if LOGGER: LOGGER.error(f"Failed to reconnect replace_rerender_btn signal: {_e}")
            except Exception as e:
                if LOGGER: LOGGER.error(f"Failed to patch on_replace_rerender: {e}")

        # --- Patch ParamWidget.__init__ to fix grid row variable shadowing bug ---
        # Bug: Line 193 in module_parse_widgets.py uses `for ii, device in enumerate(...)` 
        # which overwrites the outer loop's `ii` (grid row index), causing device/backend
        # widgets to overlap with other parameter widgets in the config panel.
        try:
            from ui.module_parse_widgets import ParamWidget, ParamCheckBox, ParamLineEditor, ParamEditor, \
                ParamComboBox, ParamCheckGroup, ParamPushButton, ParamNameLabel, get_param_display_name
            from modules import DEFAULT_DEVICE, GPUINTENSIVE_SET
            from utils.shared import size2width

            if not hasattr(ParamWidget, '_original_init_v4'):
                ParamWidget._original_init_v4 = ParamWidget.__init__

                def _patched_param_widget_init(self, params, scrollWidget=None, *args, **kwargs):
                    from qtpy.QtWidgets import QHBoxLayout, QGridLayout, QWidget
                    from qtpy.QtCore import Qt
                    QWidget.__init__(self, *args, **kwargs)

                    layout = QHBoxLayout(self)
                    self.param_layout = param_layout = QGridLayout()
                    param_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
                    param_layout.setContentsMargins(0, 0, 0, 0)
                    param_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
                    layout.addLayout(param_layout)
                    layout.addStretch(-1)

                    if 'description' in params:
                        self.setToolTip(params['description'])

                    row_idx = 0  # [FIX] Use separate row counter instead of loop variable
                    for param_key in params:
                        if param_key == 'description' or param_key.startswith('__'):
                            continue
                        display_param_name = param_key

                        require_label = True
                        is_str = isinstance(params[param_key], str)
                        is_digital = isinstance(params[param_key], float) or isinstance(params[param_key], int)
                        param_widget = None

                        if isinstance(params[param_key], bool):
                            param_widget = ParamCheckBox(param_key)
                            val = params[param_key]
                            param_widget.setChecked(val)
                            param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)

                        elif is_str or is_digital:
                            param_widget = ParamLineEditor(param_key, force_digital=is_digital)
                            val = params[param_key]
                            if is_digital:
                                val = str(val)
                            param_widget.setText(val)
                            param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)

                        elif isinstance(params[param_key], dict):
                            param_dict = params[param_key]
                            display_param_name = get_param_display_name(param_key, param_dict)
                            value = params[param_key]['value']
                            param_widget = None
                            param_type = param_dict['type'] if 'type' in param_dict else 'line_editor'
                            flush_btn = param_dict.get('flush_btn', False)
                            path_selector = param_dict.get('path_selector', False)
                            param_size = param_dict.get('size', 'short')
                            if param_type == 'selector':
                                if 'url' in param_key:
                                    size = size2width('median')
                                else:
                                    size = size2width(param_size)

                                param_widget = ParamComboBox(
                                    param_key, param_dict['options'], size=size, scrollWidget=scrollWidget, flush_btn=flush_btn, path_selector=path_selector)

                                if param_key == 'device' and DEFAULT_DEVICE == 'cpu':
                                    param_dict['value'] = 'cpu'
                                    for dev_idx, device in enumerate(param_dict['options']):  # [FIX] dev_idx instead of ii
                                        if device in GPUINTENSIVE_SET:
                                            model = param_widget.model()
                                            item = model.item(dev_idx, 0)
                                            item.setEnabled(False)
                                param_widget.setCurrentText(str(value))
                                param_widget.setEditable(param_dict.get('editable', False))

                            elif param_type == 'editor':
                                param_widget = ParamEditor(param_key)
                                param_widget.setText(value)

                            elif param_type == 'checkbox':
                                param_widget = ParamCheckBox(param_key)
                                if isinstance(value, str):
                                    value = value.lower().strip() == 'true'
                                    params[param_key]['value'] = value
                                param_widget.setChecked(value)

                            elif param_type == 'pushbtn':
                                param_widget = ParamPushButton(param_key, param_dict)
                                require_label = False

                            elif param_type == 'line_editor':
                                param_widget = ParamLineEditor(param_key, force_digital=is_digital)
                                param_widget.setText(str(value))

                            elif param_type == 'check_group':
                                param_widget = ParamCheckGroup(param_key, check_group=value)

                            if param_widget is not None:
                                param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)
                                if 'description' in param_dict:
                                    param_widget.setToolTip(param_dict['description'])

                        widget_idx = 0
                        if require_label:
                            param_label = ParamNameLabel(display_param_name)
                            param_layout.addWidget(param_label, row_idx, 0)
                            widget_idx = 1
                        if param_widget is not None:
                            pw_lo = None
                            if hasattr(param_widget, 'flush_btn') or hasattr(param_widget, 'path_select_btn'):
                                pw_lo = QHBoxLayout()
                                pw_lo.addWidget(param_widget)
                            if hasattr(param_widget, 'flush_btn'):
                                pw_lo.addWidget(param_widget.flush_btn)
                                param_widget.flushbtn_clicked.connect(self.on_flushbtn_clicked)
                            if hasattr(param_widget, 'path_select_btn'):
                                pw_lo.addWidget(param_widget.path_select_btn)
                                param_widget.pathbtn_clicked.connect(self.on_pathbtn_clicked)
                            if pw_lo is None:
                                param_layout.addWidget(param_widget, row_idx, widget_idx)
                            else:
                                param_layout.addLayout(pw_lo, row_idx, widget_idx)
                        else:
                            v = params[param_key]
                            raise ValueError(f"Failed to initialize widget for key-value pair: {param_key}-{v}")
                        row_idx += 1  # [FIX] Increment row counter safely

                ParamWidget.__init__ = _patched_param_widget_init
                if LOGGER: LOGGER.info("Monkey patch applied: ParamWidget.__init__ (Grid row index fix)")
        except Exception as e:
            if LOGGER: LOGGER.error(f"Failed to patch ParamWidget: {e}")

        # --- Repair Button Monkey Patch ---
        # Adds "Repair" button to the Run confirmation dialog.
        # Repair mode only re-translates blocks whose translation contains "error".
        if 'ui.mainwindow' in sys.modules:
            try:
                _MainWindow = sys.modules['ui.mainwindow'].MainWindow
                from qtpy.QtWidgets import QMessageBox, QApplication

                if not hasattr(_MainWindow, '_original_run_imgtrans_v4_repair'):
                    _MainWindow._original_run_imgtrans_v4_repair = _MainWindow.run_imgtrans

                    def _patched_run_imgtrans(self):
                        if not self.imgtrans_proj.is_all_pages_no_text and not pcfg.module.keep_exist_textlines:
                            msgBox = QMessageBox(self)
                            msgBox.setIcon(QMessageBox.Question)
                            msgBox.setWindowTitle(self.tr('Confirmation'))
                            msgBox.setText(self.tr('"Run" will clear previous results, "Continue" will try to run from previous progress'))

                            restart_btn = msgBox.addButton(self.tr('Run'), QMessageBox.YesRole)
                            continue_btn = msgBox.addButton(self.tr('Continue'), QMessageBox.AcceptRole)
                            repair_btn = msgBox.addButton('Repair', QMessageBox.ActionRole)
                            relayout_btn = msgBox.addButton('Re-Layout', QMessageBox.ActionRole)
                            cancel_btn = msgBox.addButton(self.tr('Cancel'), QMessageBox.RejectRole)

                            msgBox.setDefaultButton(continue_btn)
                            msgBox.exec_()

                            clicked = msgBox.clickedButton()
                            if clicked == cancel_btn:
                                return
                            elif clicked == continue_btn:
                                self.on_run_imgtrans(continue_mode=True)
                                return
                            elif clicked == repair_btn:
                                self.on_run_imgtrans(repair_mode=True)
                                return
                            elif clicked == relayout_btn:
                                self.on_run_imgtrans(relayout_mode=True)
                                return
                        self.on_run_imgtrans()

                    _MainWindow.run_imgtrans = _patched_run_imgtrans

                    # CRITICAL: The signal was already connected to the old bound method
                    # at MainWindow.__init__ time. We must reconnect to the new method.
                    # We find the MainWindow instance and reconnect the signal.
                    _orig_run_imgtrans_func = _MainWindow._original_run_imgtrans_v4_repair
                    try:
                        app = QApplication.instance()
                        if app:
                            import types
                            for widget in app.topLevelWidgets():
                                if isinstance(widget, _MainWindow):
                                    # Create bound method from the original unbound function
                                    old_bound = types.MethodType(_orig_run_imgtrans_func, widget)
                                    try:
                                        widget.leftBar.run_imgtrans_clicked.disconnect(old_bound)
                                    except (TypeError, RuntimeError):
                                        # If exact disconnect fails, try disconnecting all and reconnecting
                                        try:
                                            widget.leftBar.run_imgtrans_clicked.disconnect()
                                        except (TypeError, RuntimeError):
                                            pass
                                    widget.leftBar.run_imgtrans_clicked.connect(widget.run_imgtrans)
                                    if LOGGER: LOGGER.info("Repair: Signal reconnected on MainWindow instance")
                                    break
                    except Exception as sig_e:
                        if LOGGER: LOGGER.warning(f"Repair: Failed to reconnect signal: {sig_e}")

                    if LOGGER: LOGGER.info("Monkey patch applied: MainWindow.run_imgtrans (Repair button)")

                if not hasattr(_MainWindow, '_original_on_run_imgtrans_v4_repair'):
                    _MainWindow._original_on_run_imgtrans_v4_repair = _MainWindow.on_run_imgtrans

                    def _patched_on_run_imgtrans(self, continue_mode=False, repair_mode=False, relayout_mode=False):
                        global _V4_REPAIR_MODE, _V4_REPAIR_PAGES, _V4_REPAIR_OCR_PAGES, _V4_REPAIR_TOTAL_COUNT, _V4_REPAIR_COMPLETED_COUNT

                        # --- Re-Layout Mode ---
                        # Skip OCR/Translation/Inpainting entirely.
                        # Only re-run Auto Layout + Headless Save for pages
                        # that already have valid translations.
                        if relayout_mode:
                            if LOGGER: LOGGER.info("Re-Layout mode: scanning for pages with valid translations...")

                            pages_to_relayout = []
                            skip_count = 0

                            for page_name, blk_list in self.imgtrans_proj.pages.items():
                                if not blk_list:
                                    skip_count += 1
                                    continue

                                has_valid_translation = False
                                for blk in blk_list:
                                    trans = getattr(blk, 'translation', '')
                                    if isinstance(trans, str) and trans.strip():
                                        # Blocks with "error" in translation are NOT valid
                                        if 'error' not in trans.lower():
                                            has_valid_translation = True
                                            break

                                if has_valid_translation:
                                    pages_to_relayout.append(page_name)
                                else:
                                    skip_count += 1

                            if not pages_to_relayout:
                                if LOGGER: LOGGER.info("Re-Layout mode: no pages with valid translations found.")
                                from utils.message import create_info_dialog
                                create_info_dialog("레이아웃을 재실행할 유효한 페이지가 없습니다.")
                                return

                            if LOGGER: LOGGER.info(f"Re-Layout mode: {len(pages_to_relayout)} pages to process, {skip_count} pages skipped")

                            # --- Check which pages need inpainting ---
                            import os as _os
                            pages_needing_inpaint = []
                            for page_name in pages_to_relayout:
                                inpaint_path = self.imgtrans_proj.get_inpainted_path(page_name, get_last_modified=True)
                                if not _os.path.exists(inpaint_path):
                                    pages_needing_inpaint.append(page_name)

                            if pages_needing_inpaint:
                                if LOGGER: LOGGER.info(f"Re-Layout mode: {len(pages_needing_inpaint)} pages need inpainting first")

                                # Save original pipeline settings
                                _orig_detect = pcfg.module.enable_detect
                                _orig_ocr = pcfg.module.enable_ocr
                                _orig_inpaint = pcfg.module.enable_inpaint
                                _orig_translate = pcfg.module.enable_translate

                                # Enable only detect + inpaint (inpainter needs mask from detector)
                                pcfg.module.enable_detect = True
                                pcfg.module.enable_ocr = False
                                pcfg.module.enable_translate = False
                                pcfg.module.enable_inpaint = True

                                # Common setup
                                self.backup_blkstyles.clear()
                                if self.bottomBar.textblockChecker.isChecked():
                                    self.bottomBar.textblockChecker.click()
                                self.postprocess_mt_toggle = False

                                import threading
                                _relayout_pages_set = set(pages_to_relayout)

                                # Define callback: when inpaint pipeline finishes, run headless save
                                def _on_inpaint_finished_then_relayout():
                                    # Restore original pipeline settings
                                    pcfg.module.enable_detect = _orig_detect
                                    pcfg.module.enable_ocr = _orig_ocr
                                    pcfg.module.enable_inpaint = _orig_inpaint
                                    pcfg.module.enable_translate = _orig_translate
                                    if LOGGER: LOGGER.info("Re-Layout mode: inpainting complete, starting headless save...")

                                    # Disconnect this one-shot handler
                                    try:
                                        self.module_manager.imgtrans_pipeline_finished.disconnect(_on_inpaint_finished_then_relayout)
                                    except Exception:
                                        pass

                                    # Now run headless save in background thread
                                    def _run_relayout():
                                        global _V4_REPAIR_MODE, _V4_REPAIR_PAGES
                                        translate_thread = self.module_manager.translate_thread
                                        translate_thread.imgtrans_proj = self.imgtrans_proj
                                        translate_thread._v4_pipeline_start_time = time.time()

                                        # Temporarily disable inpaint flag to skip FIN_INPAINT wait
                                        # in process_page_hybrid (inpainting is already done at this point)
                                        _saved_inpaint = pcfg.module.enable_inpaint
                                        pcfg.module.enable_inpaint = False

                                        _V4_REPAIR_MODE = True
                                        _V4_REPAIR_PAGES = _relayout_pages_set
                                        try:
                                            _v4_headless_save_entry(translate_thread, proj=self.imgtrans_proj)
                                        finally:
                                            _V4_REPAIR_MODE = False
                                            _V4_REPAIR_PAGES = None
                                            pcfg.module.enable_inpaint = _saved_inpaint
                                            if LOGGER: LOGGER.info("Re-Layout mode: finished.")

                                    t = threading.Thread(target=_run_relayout, name="V4ReLayoutThread", daemon=True)
                                    t.start()

                                # Connect one-shot signal
                                self.module_manager.imgtrans_pipeline_finished.connect(_on_inpaint_finished_then_relayout)

                                # Run inpaint-only pipeline for pages that need it
                                self.module_manager.runImgtransPipeline(pages_needing_inpaint)
                                return

                            # --- All pages already inpainted: run headless save directly ---
                            # Show progress box
                            self.imgtrans_progress_msgbox.zero_progress()
                            self.imgtrans_progress_msgbox.show()

                            # Run headless save in background thread
                            import threading
                            _relayout_pages_set = set(pages_to_relayout)

                            def _run_relayout():
                                global _V4_REPAIR_MODE, _V4_REPAIR_PAGES
                                translate_thread = self.module_manager.translate_thread
                                translate_thread.imgtrans_proj = self.imgtrans_proj
                                # Record start time for elapsed display
                                translate_thread._v4_pipeline_start_time = time.time()

                                # Temporarily disable inpaint flag to skip FIN_INPAINT wait
                                # in process_page_hybrid (inpainting is already done)
                                _saved_inpaint = pcfg.module.enable_inpaint
                                pcfg.module.enable_inpaint = False

                                # Use REPAIR_PAGES filter to limit which pages get rendered
                                _V4_REPAIR_MODE = True
                                _V4_REPAIR_PAGES = _relayout_pages_set
                                try:
                                    _v4_headless_save_entry(translate_thread, proj=self.imgtrans_proj)
                                finally:
                                    _V4_REPAIR_MODE = False
                                    _V4_REPAIR_PAGES = None
                                    pcfg.module.enable_inpaint = _saved_inpaint
                                    if LOGGER: LOGGER.info("Re-Layout mode: finished.")

                            t = threading.Thread(target=_run_relayout, name="V4ReLayoutThread", daemon=True)
                            t.start()
                            return

                        if not repair_mode:
                            # A repair run temporarily changes the shared pipeline
                            # configuration.  The normal completion callback should
                            # restore it, but a cancelled/stalled run can miss that
                            # signal.  Never let that temporary state leak into the
                            # next ordinary project run.
                            restore_repair_settings = getattr(self, '_v4_repair_restore_settings', None)
                            if callable(restore_repair_settings):
                                if LOGGER:
                                    LOGGER.warning(
                                        "Repair mode cleanup was still pending; restoring "
                                        "the normal pipeline before this run."
                                    )
                                restore_repair_settings()

                            # Delegate to the original (or previously patched) version
                            return self._original_on_run_imgtrans_v4_repair(continue_mode=continue_mode)

                        # --- Repair Mode ---
                        # Find all blocks whose translation contains "error",
                        # is empty with source present, has both source AND translation empty,
                        # or contains untranslated Japanese text (when source also had Japanese).
                        if LOGGER: LOGGER.info("Repair mode: scanning for error/empty/blank/untranslated Japanese translations...")

                        jp_char_regex = re.compile(r'[\u3040-\u309f\u30a0-\u30ff]')
                        # A translated box can legitimately retain a Japanese name,
                        # SFX, or quoted phrase.  Treat it as untranslated only when
                        # it contains no Korean at all (syllables or jamo).
                        ko_char_regex = re.compile(r'[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]')

                        pages_to_repair = []
                        pages_needing_ocr = set()  # Pages where at least one block needs OCR
                        error_count = 0
                        empty_count = 0
                        blank_count = 0  # Both source AND translation are empty
                        untranslated_jp_count = 0  # Translation is present but still in Japanese (and source also had Japanese)
                        for page_name, blk_list in self.imgtrans_proj.pages.items():
                            page_needs_repair = False
                            for blk in blk_list:
                                trans = getattr(blk, 'translation', '')
                                source_text = getattr(blk, 'text', [])
                                # Normalize source text: list of strings -> joined
                                if isinstance(source_text, list):
                                    joined_source = ' '.join(s for s in source_text if isinstance(s, str)).strip()
                                else:
                                    joined_source = str(source_text).strip() if source_text else ''
                                has_source = bool(joined_source) and 'error' not in joined_source.lower()
                                has_trans = bool(trans) and (isinstance(trans, str) and trans.strip())

                                needs_repair = False
                                needs_ocr = False
                                if isinstance(trans, str) and 'error' in trans.lower():
                                    needs_repair = True
                                    error_count += 1
                                elif has_source and not has_trans:
                                    # Has OCR text but no translation
                                    needs_repair = True
                                    empty_count += 1
                                elif not has_source and not has_trans:
                                    # Both source and translation are empty → need OCR + translation
                                    needs_repair = True
                                    needs_ocr = True
                                    blank_count += 1
                                elif has_source and has_trans:
                                    # A box that contains both Japanese and Korean is
                                    # already translated; only Japanese-only output is
                                    # a repair target.
                                    translation_text = str(trans)
                                    if (jp_char_regex.search(translation_text)
                                            and not ko_char_regex.search(translation_text)):
                                        if jp_char_regex.search(joined_source):
                                            needs_repair = True
                                            untranslated_jp_count += 1

                                if needs_repair:
                                    blk.translation = ''
                                    blk.rich_text = ''
                                    blk._v4_needs_repair = True
                                    if needs_ocr:
                                        blk._v4_needs_ocr = True
                                        pages_needing_ocr.add(page_name)
                                    page_needs_repair = True
                            if page_needs_repair:
                                pages_to_repair.append(page_name)

                        total_count = error_count + empty_count + blank_count + untranslated_jp_count
                        if total_count == 0:
                            if LOGGER: LOGGER.info("Repair mode: no errors, empty/blank, or untranslated Japanese translations found, nothing to do.")
                            from utils.message import create_info_dialog
                            create_info_dialog("No translation errors, empty/blank translations, or untranslated Japanese blocks found. Nothing to repair.")
                            return

                        _V4_REPAIR_TOTAL_COUNT = total_count
                        _V4_REPAIR_COMPLETED_COUNT = 0

                        log_msg = (
                            f"[Repair 대상 확정] 총 {total_count}개 블록이 대기열에 등록되었습니다. "
                            f"(페이지: {len(pages_to_repair)}개 | 에러: {error_count}개, 비어있음: {empty_count}개, "
                            f"OCR필요: {blank_count}개, 일본어 미번역: {untranslated_jp_count}개)"
                        )
                        if LOGGER: LOGGER.info(log_msg)

                        has_ocr_pages = bool(pages_needing_ocr)
                        if has_ocr_pages:
                            if LOGGER: LOGGER.info(f"Repair mode: {len(pages_needing_ocr)} pages need OCR (both src+trans empty)")

                        # If a previous repair did not reach its completion signal,
                        # restore it before recording this run's baseline settings.
                        restore_repair_settings = getattr(self, '_v4_repair_restore_settings', None)
                        if callable(restore_repair_settings):
                            restore_repair_settings()

                        # Activate repair mode filter
                        _V4_REPAIR_MODE = True
                        _V4_REPAIR_PAGES = set(pages_to_repair)
                        _V4_REPAIR_OCR_PAGES = pages_needing_ocr if has_ocr_pages else None

                        # Save original settings to restore later
                        _orig_detect = pcfg.module.enable_detect
                        _orig_ocr = pcfg.module.enable_ocr
                        _orig_inpaint = pcfg.module.enable_inpaint
                        _orig_translate = pcfg.module.enable_translate

                        # Enable translate; enable OCR only if there are blank blocks needing it
                        pcfg.module.enable_detect = False
                        pcfg.module.enable_ocr = has_ocr_pages  # Conditionally enable OCR
                        pcfg.module.enable_inpaint = False
                        pcfg.module.enable_translate = True

                        # Common setup from original on_run_imgtrans
                        self.backup_blkstyles.clear()
                        if self.bottomBar.textblockChecker.isChecked():
                            self.bottomBar.textblockChecker.click()
                        self.postprocess_mt_toggle = False

                        # Store font styles for repair pages
                        for page_name in pages_to_repair:
                            blklist = self.imgtrans_proj.pages[page_name]
                            ffmt_list = []
                            self.backup_blkstyles.append(ffmt_list)
                            for textblk in blklist:
                                ffmt_list.append(textblk.fontformat.deepcopy())

                        # Define an idempotent restore callback.  Keep it on the
                        # window as a fallback for the next normal run because the
                        # completion signal is not guaranteed after cancellation.
                        restore_state = {'done': False}
                        def _restore_settings():
                            if restore_state['done']:
                                return
                            restore_state['done'] = True
                            global _V4_REPAIR_MODE, _V4_REPAIR_PAGES, _V4_REPAIR_OCR_PAGES
                            _V4_REPAIR_MODE = False
                            _V4_REPAIR_PAGES = None
                            _V4_REPAIR_OCR_PAGES = None
                            pcfg.module.enable_detect = _orig_detect
                            pcfg.module.enable_ocr = _orig_ocr
                            pcfg.module.enable_inpaint = _orig_inpaint
                            pcfg.module.enable_translate = _orig_translate
                            if getattr(self, '_v4_repair_restore_settings', None) is _restore_settings:
                                self._v4_repair_restore_settings = None
                            if LOGGER: LOGGER.info("Repair mode: original pipeline settings restored. Repair filter deactivated.")

                        self._v4_repair_restore_settings = _restore_settings

                        # Connect one-shot restore when pipeline finishes
                        def _on_repair_finished():
                            _restore_settings()
                            try:
                                self.module_manager.imgtrans_pipeline_finished.disconnect(_on_repair_finished)
                            except Exception:
                                pass

                        self.module_manager.imgtrans_pipeline_finished.connect(_on_repair_finished)

                        # Run the pipeline with only the error pages
                        self.module_manager.runImgtransPipeline(pages_to_repair)

                    _MainWindow.on_run_imgtrans = _patched_on_run_imgtrans
                    if LOGGER: LOGGER.info("Monkey patch applied: MainWindow.on_run_imgtrans (Repair mode)")

            except Exception as e:
                if LOGGER: LOGGER.error(f"Failed to apply Repair button patches: {e}")

        # --- FIX: Patch TextBlkItem.paint_stroke for stroke position mismatch ---
        # setTextOutline() changes font metrics (ascent etc.), causing the stroke
        # document's HorizontalTextDocumentLayout to compute different line positions
        # (dy = -tbr.top() - line.ascent()). Fix: copy original line positions to
        # stroke document after its layout is calculated.
        try:
            from ui.textitem import TextBlkItem
            from qtpy.QtWidgets import QStyle
            if not hasattr(TextBlkItem, '_original_paint_stroke_v4'):
                TextBlkItem._original_paint_stroke_v4 = TextBlkItem.paint_stroke

                def _patched_paint_stroke(self, painter):
                    from qtpy.QtGui import QPainter, QPen, QFont, QTextDocument, QTextCursor
                    from qtpy.QtCore import Qt, QPointF
                    from ui.scene_textlayout import VerticalTextDocumentLayout, HorizontalTextDocumentLayout

                    # 1. Collect original line positions BEFORE creating stroke doc
                    orig_doc = self.document()
                    orig_line_positions = []
                    orig_block = orig_doc.firstBlock()
                    while orig_block.isValid():
                        block_lines = []
                        tl = orig_block.layout()
                        for ii in range(tl.lineCount()):
                            line = tl.lineAt(ii)
                            block_lines.append(QPointF(line.position()))
                        orig_line_positions.append(block_lines)
                        orig_block = orig_block.next()

                    # 2. Create stroke document (same as original paint_stroke)
                    doc = QTextDocument()
                    doc.setUndoRedoEnabled(False)
                    doc.setDocumentMargin(orig_doc.documentMargin())
                    doc.setDefaultFont(orig_doc.defaultFont())
                    doc.setHtml(orig_doc.toHtml())
                    doc.setDefaultTextOption(orig_doc.defaultTextOption())

                    cursor = QTextCursor(doc)
                    block = doc.firstBlock()
                    stroke_pen = QPen(self.stroke_qcolor, 0, Qt.PenStyle.SolidLine,
                                      Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                    letter_spacing = self.fontformat.letter_spacing * 100

                    while block.isValid():
                        it = block.begin()
                        while not it.atEnd():
                            fragment = it.fragment()
                            cfmt = fragment.charFormat()
                            from utils.fontformat import pt2px
                            sw = pt2px(cfmt.fontPointSize()) * self.fontformat.stroke_width
                            stroke_pen.setWidthF(sw)
                            pos1 = fragment.position()
                            pos2 = pos1 + fragment.length()
                            cursor.setPosition(pos1)
                            cursor.setPosition(pos2, QTextCursor.MoveMode.KeepAnchor)
                            cfmt.setTextOutline(stroke_pen)
                            if letter_spacing != 100 and not self.fontformat.vertical:
                                cfmt.setFontLetterSpacingType(QFont.SpacingType.PercentageSpacing)
                                cfmt.setFontLetterSpacing(letter_spacing)
                            cursor.mergeCharFormat(cfmt)
                            it += 1
                        block = block.next()

                    # 3. Create layout (triggers reLayoutEverything with wrong positions)
                    layout = VerticalTextDocumentLayout(doc, self.fontformat) if self.fontformat.vertical \
                        else HorizontalTextDocumentLayout(doc, self.fontformat)
                    layout._draw_offset = self.layout._draw_offset
                    layout._is_painting_stroke = True
                    layout.setMaxSize(self.layout.max_width, self.layout.max_height, False)
                    doc.setDocumentLayout(layout)

                    # 4. FIX: Overwrite stroke doc line positions with original positions
                    stroke_block = doc.firstBlock()
                    blk_idx = 0
                    while stroke_block.isValid() and blk_idx < len(orig_line_positions):
                        tl = stroke_block.layout()
                        orig_lines = orig_line_positions[blk_idx]
                        for ii in range(min(tl.lineCount(), len(orig_lines))):
                            line = tl.lineAt(ii)
                            line.setPosition(orig_lines[ii])
                        blk_idx += 1
                        stroke_block = stroke_block.next()

                    # 5. Draw
                    layout.relayout_on_changed = False
                    doc.drawContents(painter)

                TextBlkItem.paint_stroke = _patched_paint_stroke
                if LOGGER: LOGGER.info("Monkey patch applied: TextBlkItem.paint_stroke (Position fix)")
        except Exception as e:
            if LOGGER: LOGGER.error(f"Failed to patch TextBlkItem.paint_stroke: {e}")

        # --- FIX: Patch TextBlkItem.paint for stroke/outline rendering ---
        # Commit 3829d745 changed paint() to use CompositionMode_DestinationOver
        # for stroke rendering. This breaks when QGraphicsScene::render() is called
        # (e.g. saving result images) because the painter already has an opaque
        # background (inpainted image), causing DestinationOver to hide the stroke.
        # Fix: render to intermediate transparent QPixmap first, then draw to painter.
        try:
            if not hasattr(TextBlkItem, '_original_paint_v4_stroke_fix'):
                TextBlkItem._original_paint_v4_stroke_fix = TextBlkItem.paint

                def _patched_paint(self, painter, option, widget):
                    from qtpy.QtGui import QPainter, QPixmap
                    from qtpy.QtCore import Qt

                    if self.is_editting():
                        # Editing mode: draw accessories first, then text on top (original logic)
                        self._draw_accessories(painter)
                        option.state = QStyle.State_None
                        super(TextBlkItem, self).paint(painter, option, widget)
                    else:
                        # Non-editing mode: render to intermediate transparent pixmap
                        # This prevents subpixel antialiasing (issue #919) AND
                        # ensures stroke renders correctly during scene.render()
                        br = self.boundingRect()
                        size = br.size().toSize()
                        if size.width() > 0 and size.height() > 0:
                            tmp = QPixmap(size)
                            tmp.fill(Qt.GlobalColor.transparent)
                            tmp_painter = QPainter(tmp)
                            tmp_painter.setRenderHints(painter.renderHints())
                            # 1. Draw stroke/shadow underneath
                            self._draw_accessories(tmp_painter)
                            # 2. Draw text on top
                            option.state = QStyle.State_None
                            super(TextBlkItem, self).paint(tmp_painter, option, widget)
                            tmp_painter.end()
                            painter.drawPixmap(br.toRect(), tmp)

                TextBlkItem.paint = _patched_paint
                if LOGGER: LOGGER.info("Monkey patch applied: TextBlkItem.paint (Stroke rendering fix)")
        except Exception as e:
            if LOGGER: LOGGER.error(f"Failed to patch TextBlkItem.paint: {e}")

        _install_patches._applied = True
        
    except Exception as e:
        if LOGGER: LOGGER.error(f"V4 Patch failed: {e}")

# Global save lock to prevent conflicts
_GLOBAL_SAVE_LOCK = threading.Lock()

# Global flags for save suppression
_HEADLESS_SAVE_IN_PROGRESS = False
_PIPELINE_ACTIVE = False  # Set to True when OCR/Translation pipeline is running
_LAST_SAVE_TIME = 0.0  # For debouncing
_REPLACE_RERENDER_ACTIVE = False  # Suppress save debug logs during replace-rerender
_DEBUG_PROFILING = False  # OCR 파이프라인 프로파일링 토글 (ocr_llm_api_v4가 pipeline 시작 시 세팅)

# -------------------------------------------------------------------------
# UI Monkey Patching: Add Saving Bar dynamically (Delayed)
# -------------------------------------------------------------------------
def _install_ui_patches():
    from qtpy.QtWidgets import QApplication
    from qtpy.QtCore import QThread
    
    app = QApplication.instance()
    if app and QThread.currentThread() is not app.thread():
        print("DEBUG: _install_ui_patches called from background thread. SKIPPING to avoid freeze.")
        return

    print("DEBUG: _install_ui_patches running...")
    try:
        from ui.custom_widget import ImgtransProgressMessageBox, TaskProgressBar
        
        # 1. Inject updateSavingProgress method
        def updateSavingProgress(self, value: int, msg: str = ''):
            if hasattr(self, 'saving_bar'):
                self.saving_bar.updateProgress(value, msg)
                
        if not hasattr(ImgtransProgressMessageBox, 'updateSavingProgress'):
            setattr(ImgtransProgressMessageBox, 'updateSavingProgress', updateSavingProgress)

        # 2. Patch __init__ to add saving_bar widget
        _original_init = ImgtransProgressMessageBox.__init__

        def _patched_init(self, *args, **kwargs):
            _original_init(self, *args, **kwargs)
            # Add Saving Bar
            self.saving_bar = TaskProgressBar(self.tr('Saving: '), True, self)
            
            # Insert into layout (index 4 is after Translate bar)
            layout = self.layout()
            layout.insertWidget(4, self.saving_bar)
            
        if not getattr(ImgtransProgressMessageBox, '_saving_bar_patched', False):
            ImgtransProgressMessageBox.__init__ = _patched_init
            setattr(ImgtransProgressMessageBox, '_saving_bar_patched', True)
                
        # 3. Patch zero_progress to reset saving bar too
        _original_zero = ImgtransProgressMessageBox.zero_progress
        
        def _patched_zero(self):
            _original_zero(self)
            if hasattr(self, 'saving_bar'):
                self.saving_bar.updateProgress(0)
                
        if not getattr(ImgtransProgressMessageBox, '_zero_patched', False):
            ImgtransProgressMessageBox.zero_progress = _patched_zero
            setattr(ImgtransProgressMessageBox, '_zero_patched', True)

        # 4. Patch Existing Instances (Runtime Injection)
        # Since the box might be created before we patch __init__, we need to fix existing ones.
        from qtpy.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            for widget in app.allWidgets():
                # Check by class name to avoid import issues if class is different object
                if widget.__class__.__name__ == 'ImgtransProgressMessageBox':
                    print("DEBUG: Found ImgtransProgressMessageBox instance! Patching...")
                    if not hasattr(widget, 'saving_bar'):
                        widget.saving_bar = TaskProgressBar(widget.tr('Saving: '), True, widget)
                        # Insert before buttons (usually index 4)
                        widget.layout().insertWidget(4, widget.saving_bar)
                        # Hide by default or show? Usually others are shown.
                        widget.saving_bar.show()

    except ImportError:
        pass

# -------------------------------------------------------------------------
# Orchestration Patch
# -------------------------------------------------------------------------

def _imgtrans_pipeline_v4_orchestrator(self):
    """
    Patched main pipeline that ensures Verified Save runs if V4 Translator is used.
    """
    # Record start time
    import time
    start_time = time.time()

    # Run the actual pipeline (either original or OCR-patched version)
    if hasattr(self, '_original_imgtrans_pipeline_trans_v4'):
        self._original_imgtrans_pipeline_trans_v4()
    
    # After EVERYTHING is done (Detection, OCR, Translation, Inpainting)
    # Check if we are using the V4 Translator
    is_v4 = getattr(self.translator, 'use_image_batching', False)
    
    # Check if save was already handled by OCR pipeline (Robust Flag Check)
    if getattr(self.translate_thread, '_v4_save_completed', False):
        if LOGGER:
            LOGGER.info("Save already completed by OCR pipeline. Orchestrator skipping.")
        # Log total duration before returning
        elapsed = time.time() - start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        if LOGGER:
            LOGGER.info(f"📊 총 작업시간: {hours}시간 {minutes}분 {seconds}초 (OCR 파이프라인)")
        return
    
    # Check if OCR is V4 (Async Pipeline). If so, it handles orchestration and saving.
    # We should NOT duplicate the save here.
    is_v4_ocr = getattr(self.ocr, 'use_page_batching', False)
    
    if is_v4_ocr:
        if LOGGER:
            LOGGER.info("OCR is V4-capable. Orchestrator delegating final save to OCR pipeline.")
        # Log total duration before returning
        elapsed = time.time() - start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        if LOGGER:
            LOGGER.info(f"📊 총 작업시간: {hours}시간 {minutes}분 {seconds}초")
        return

    if is_v4:
        if LOGGER:
            LOGGER.info("Master Pipeline detected LLM V4 Translator. Initiating Final Verified Save...")
        
        # Ensure the translation thread is actually finished
        target = getattr(self.translate_thread, 'num_process_pages', 0) or getattr(self.translate_thread, 'num_pages', 0)
        while self.translate_thread.finished_counter < target:
            if self.stop_requested: break
            time.sleep(0.5)
            
        # Trigger Save
        try:
            _v4_headless_save_entry(self.translate_thread)
        except Exception as e:
            if LOGGER: LOGGER.error(f"Final save failed: {e}")
    
    # Log total duration at the very end
    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    if LOGGER:
        LOGGER.info(f"📊 총 작업시간: {hours}시간 {minutes}분 {seconds}초")

# -------------------------------------------------------------------------
# Base Translator & Models
# -------------------------------------------------------------------------
from .base import BaseTranslator, register_translator
from qtpy.QtCore import QObject, Signal, Qt, QTimer, Slot

class SaveSignaler(QObject):
    save_signal = Signal()
    progress_signal = Signal(int, str)
    finished_signal = Signal()


def _set_v4_layout_state(translate_thread, page_key: str, state: str) -> None:
    """Record one page's layout lifecycle."""
    lock = getattr(translate_thread, '_v4_layout_state_lock', None)
    if lock is None:
        lock = threading.Lock()
        translate_thread._v4_layout_state_lock = lock
    with lock:
        states = getattr(translate_thread, '_v4_layout_states', None)
        if states is None:
            states = {}
            translate_thread._v4_layout_states = states
        states[page_key] = state


def _claim_v4_layout_page(translate_thread, page_key: str) -> bool:
    """Atomically place a page in waiting state exactly once for this run."""
    lock = getattr(translate_thread, '_v4_layout_state_lock', None)
    if lock is None:
        lock = threading.Lock()
        translate_thread._v4_layout_state_lock = lock
    with lock:
        states = getattr(translate_thread, '_v4_layout_states', None)
        if states is None:
            states = {}
            translate_thread._v4_layout_states = states
        if page_key in states:
            return False
        states[page_key] = 'waiting'
        return True


def _add_v4_layout_timing(translate_thread, phase: str, elapsed: float) -> None:
    lock = getattr(translate_thread, '_v4_layout_timing_lock', None)
    if lock is None:
        lock = threading.Lock()
        translate_thread._v4_layout_timing_lock = lock
    with lock:
        timings = getattr(translate_thread, '_v4_layout_timings', None)
        if timings is None:
            timings = {}
            translate_thread._v4_layout_timings = timings
        timings[phase] = timings.get(phase, 0.0) + elapsed


def _layout_page_readiness(translate_thread, page_key: str, wait_for_pipeline: bool) -> str:
    """Return ready/waiting/failed/cancelled for one page without mutating it."""
    if getattr(translate_thread, 'stop_requested', False) or _V4_STOP_REQUESTED:
        return 'cancelled'

    proj = getattr(translate_thread, 'imgtrans_proj', None)
    info = proj._image_info.get(page_key, {}) if proj is not None else {}
    if info.get('corrupted', False):
        return 'failed'

    if wait_for_pipeline:
        state_lock = getattr(translate_thread, '_v4_layout_state_lock', None)
        if state_lock is None:
            translated_pages = getattr(translate_thread, '_v4_translated_pages', set())
            failed_pages = getattr(translate_thread, '_v4_translation_failed_pages', set())
            inpaint_failed_pages = getattr(translate_thread, '_v4_inpaint_failed_pages', set())
        else:
            with state_lock:
                translated_pages = set(getattr(translate_thread, '_v4_translated_pages', set()))
                failed_pages = set(getattr(translate_thread, '_v4_translation_failed_pages', set()))
                inpaint_failed_pages = set(getattr(translate_thread, '_v4_inpaint_failed_pages', set()))
        if page_key in failed_pages or page_key in inpaint_failed_pages:
            return 'failed'
        if page_key not in translated_pages:
            return 'waiting'

    if RunStatus is None:
        return 'ready'

    finish_code = info.get('finish_code', 0)
    module_cfg = getattr(pcfg, 'module', None) if pcfg is not None else None
    if wait_for_pipeline and getattr(module_cfg, 'enable_ocr', False):
        if not finish_code & RunStatus.FIN_OCR:
            return 'waiting'
    if getattr(module_cfg, 'enable_inpaint', False):
        if not finish_code & RunStatus.FIN_INPAINT:
            return 'waiting'
    return 'ready'


def _prepare_layout_page(proj, page_key: str, enable_autolayout: bool = True):
    """Load a page and perform only the OpenCV balloon analysis off the GUI thread."""
    started_at = time.perf_counter()
    img_path = proj.get_inpainted_path(page_key)
    if not os.path.exists(img_path):
        img_path = os.path.join(proj.directory, page_key)

    image = QImage(img_path)
    if image.isNull():
        raise RuntimeError(f"V4 Render: Failed to load image {img_path}")

    prepared_blocks = {}
    blk_list = proj.pages.get(page_key, [])
    if enable_autolayout and blk_list:
        import numpy as np
        from utils.imgproc_utils import extract_ballon_region

        image_rgb = image.convertToFormat(QImage.Format_RGB888)
        width, height = image_rgb.width(), image_rgb.height()
        ptr = image_rgb.bits()
        ptr.setsize(height * width * 3)
        img_array = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3))

        for index, blk in enumerate(blk_list):
            txt = getattr(blk, 'translation', '') or getattr(blk, 'rich_text', '')
            if not txt or not str(txt).strip():
                continue

            widened_xyxy = None
            if getattr(blk, 'src_is_vertical', False):
                x1, y1, x2, y2 = blk.xyxy
                width_now, height_now = x2 - x1, y2 - y1
                new_width = max(width_now, height_now * 1.5)
                center_x = (x1 + x2) / 2
                widened_xyxy = [
                    int(center_x - new_width / 2),
                    y1,
                    int(center_x + new_width / 2),
                    y2,
                ]

            if blk.translation == getattr(blk, '_v4_last_layout_text', ''):
                prepared_blocks[index] = {
                    'skip_layout': True,
                    'widened_xyxy': widened_xyxy,
                }
                continue

            blk_rect = blk.bounding_rect()
            if blk_rect is None or len(blk_rect) < 4:
                prepared_blocks[index] = {
                    'error': f"Invalid Rect: {blk_rect}",
                    'widened_xyxy': widened_xyxy,
                }
                continue

            width_now, height_now = blk_rect[2], blk_rect[3]
            if width_now <= 0 or height_now <= 0:
                prepared_blocks[index] = {
                    'error': f"Invalid Rect: {blk_rect}",
                    'widened_xyxy': widened_xyxy,
                }
                continue

            enlarge_ratio = min(max(width_now / height_now, height_now / width_now) * 1.5, 2.5)
            result = extract_ballon_region(
                img_array,
                blk_rect,
                enlarge_ratio=enlarge_ratio,
                cal_region_rect=True,
            )
            if result and len(result) >= 4:
                prepared_blocks[index] = {
                    'mask': result[0],
                    'region_rect': result[3],
                    'bounding_rect': blk_rect,
                    'widened_xyxy': widened_xyxy,
                }
            else:
                prepared_blocks[index] = {
                    'error': f"Extraction failed: {result}",
                    'widened_xyxy': widened_xyxy,
                }

    return {
        'image': image,
        'blocks': prepared_blocks,
        'prepare_seconds': time.perf_counter() - started_at,
    }


class UIHelper(QObject):
    def __init__(self, msgbox, proj=None, stm=None, total_pages=None, pipeline_start_time=None):
        super().__init__()
        self.msgbox = msgbox
        self.proj = proj
        self.stm = stm # [NEW] SceneTextManager for layout/refresh
        self.rendered_images = {} # Thread-safe storage for cross-thread return values
        self.render_timings = {}
        self.prepared_pages = {}
        self.prepared_pages_lock = threading.Lock()
        self._layout_start_time = time.time()  # Track layout start time
        self._layout_completed_count = 0  # Track completed page count
        self._layout_total_pages = total_pages if total_pages is not None else (len(proj.pages) if proj else 0)
        self._pipeline_start_time = pipeline_start_time  # Track overall pipeline start time

    def update_ui(self, percent, text):
        if self.msgbox:
            if not self.msgbox.isVisible():
                self.msgbox.show()
            if hasattr(self.msgbox, 'updateSavingProgress'):
                self.msgbox.updateSavingProgress(percent, text)
            else:
                self.msgbox.updateTranslateProgress(percent, text)

    def finish_ui(self):
        # Calculate elapsed time
        elapsed_str = ""
        if self._pipeline_start_time:
            elapsed = time.time() - self._pipeline_start_time
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                elapsed_str = f" ({hours}시간 {minutes}분 {seconds}초)"
            elif minutes > 0:
                elapsed_str = f" ({minutes}분 {seconds}초)"
            else:
                elapsed_str = f" ({seconds}초)"

        if self.msgbox:
            finish_msg = f" (저장 완료!{elapsed_str})"
            if hasattr(self.msgbox, 'updateSavingProgress'):
                self.msgbox.updateSavingProgress(100, finish_msg)
            else:
                self.msgbox.updateTranslateProgress(100, finish_msg)
            QTimer.singleShot(1500, self.msgbox.accept)

            # [NEW] Refresh GUI to show latest translations (layout already done in render_page_task)
            if self.stm:
                if hasattr(self.stm, '_original_updateTranslation_patch'):
                    self.stm._original_updateTranslation_patch()
                else:
                    self.stm.updateTranslation()

            # Show Native System Tray Notification (Safe on Main Thread)
            try:
                from qtpy.QtWidgets import QSystemTrayIcon, QApplication, QStyle
                # Use standard icon if app icon not available
                icon = QApplication.style().standardIcon(QStyle.SP_DialogApplyButton)
                # Create tray icon attached to the msgbox to prevent early GC
                self.tray = QSystemTrayIcon(icon, self.msgbox)
                self.tray.show()
                notify_msg = f"번역 및 저장이 완료되었습니다!{elapsed_str}"
                self.tray.showMessage(
                    "BallonsTranslator",
                    notify_msg,
                    QSystemTrayIcon.Information,
                    3000
                )
                # Auto-hide after 3.5 seconds to clear the icon
                QTimer.singleShot(3500, self.tray.hide)
            except Exception as e:
                print(f"Notification Error: {e}")

    # ----------------------------------------------------------------------
    # HYBRID RENDERING: Run heavily GUI-dependent logic on Main Thread
    # ----------------------------------------------------------------------
    @Slot(str)
    def render_page_task(self, page_key):
        """
        Renders the page to a QImage on the Main Thread.
        Stores result in self.rendered_images[page_key]
        """
        render_started = time.perf_counter()
        # --- AutoLayout Progress ---
        _total = self._layout_total_pages if self._layout_total_pages else (len(self.proj.pages) if self.proj else '?')
        _page_keys = list(self.proj.pages.keys()) if self.proj else []
        _page_idx = _page_keys.index(page_key) + 1 if page_key in _page_keys else '?'
        self._layout_completed_count += 1
        from datetime import datetime as _dt
        _time_str = _dt.now().strftime('%H:%M:%S')
        _elapsed = time.time() - self._layout_start_time
        _elapsed_m, _elapsed_s = divmod(int(_elapsed), 60)
        _elapsed_str = f"{_elapsed_m}분 {_elapsed_s}초" if _elapsed_m > 0 else f"{_elapsed_s}초"
        if LOGGER: LOGGER.info(f"\U0001f4d0 AutoLayout(Render): \ud398\uc774\uc9c0 [{_page_idx}/{_total}] (\uc644\ub8cc: {self._layout_completed_count}/{_total}) \u23f1 {_time_str} (+{_elapsed_str}) - {page_key}")

        with self.prepared_pages_lock:
            prepared_page = self.prepared_pages.pop(page_key, None)
        prepared_blocks = prepared_page.get('blocks', {}) if prepared_page else {}

        project_lock = getattr(self.proj, '_v4_save_lock', None) if self.proj else None
        if project_lock is not None:
            project_lock.acquire()

        try:
            # Import GUI classes locally to avoid circular dependencies at module level
            from ui.textitem import TextBlkItem
            from qtpy.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
            from qtpy.QtGui import QPixmap
            
            if not self.proj:
                print("V4 Render: Project is None")
                self.rendered_images[page_key] = None
                return
                
            # The worker-loaded QImage and OpenCV results are safe to hand over.
            # QPixmap/QGraphicsScene/TextBlkItem creation remains on this GUI thread.
            image = prepared_page.get('image') if prepared_page else None
            if image is None:
                img_path = self.proj.get_inpainted_path(page_key)
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.proj.directory, page_key)
                image = QImage(img_path)
            if image.isNull():
                print(f"V4 Render: Failed to load image for {page_key}")
                self.rendered_images[page_key] = None
                return

            # Setup Scene (Main Thread Safe)
            scene = QGraphicsScene()
            scene.setSceneRect(0, 0, image.width(), image.height())
            
            # Add Background (Z=0)
            # QGraphicsPixmapItem might need QPixmap, which is fine in Main Thread
            bg_pixmap = QPixmap.fromImage(image)
            bg_item = QGraphicsPixmapItem(bg_pixmap)
            bg_item.setZValue(0)
            scene.addItem(bg_item)
            
            # Add Text Blocks
            blk_list = self.proj.pages.get(page_key, [])
            
            # Legacy/direct re-layout fallback. Incremental pipeline pages already
            # performed this pure OpenCV work in a bounded background worker.
            img_array = None
            image_rgb = None
            if prepared_page is None and self.stm and pcfg.let_autolayout_flag and blk_list:
                try:
                    import numpy as np
                    image_rgb = image.convertToFormat(QImage.Format_RGB888)
                    width, height = image_rgb.width(), image_rgb.height()
                    ptr = image_rgb.bits()
                    ptr.setsize(height * width * 3)
                    img_array = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3))
                    # Note: image_rgb must stay alive while img_array references its buffer
                except Exception as img_err:
                    print(f"V4 Render: Image conversion failed: {img_err}")
                    image_rgb = None  # Ensure cleanup

            global _V4_SAVE_HEARTBEAT, _V4_SAVE_HEARTBEAT_INFO, _V4_HEADLESS_LAYOUT_CALL
            for i, blk in enumerate(blk_list):
                # [Diag] Per-block heartbeat + debug log. Last logged (page, blk) before a
                # freeze is exactly the block where the main thread hung.
                _V4_SAVE_HEARTBEAT = time.time()
                _V4_SAVE_HEARTBEAT_INFO = f"render:{page_key}:blk={i}/{len(blk_list)}"
                _autolayout_detail_log('debug', f"[Render/blk] page={page_key} blk={i}/{len(blk_list)}")

                # Try getting translation, fallback to rich_text if empty
                txt = getattr(blk, 'translation', '')
                if not txt:
                    txt = getattr(blk, 'rich_text', '')

                # If still empty, skip
                if not txt or not str(txt).strip():
                    continue

                try:
                    # Force horizontal for rendering translated text
                    if hasattr(blk, 'vertical'):
                        blk.vertical = False

                    # TextBlkItem requires Main Thread for FontMetrics & Layouts
                    text_item = TextBlkItem(blk, idx=i, set_format=True, show_rect=False)
                    text_item.setZValue(10) # Ensure it's above background
                    scene.addItem(text_item)

                    # Apply worker-prepared layout data without changing the Qt path.
                    if self.stm and prepared_page is not None and pcfg.let_autolayout_flag:
                        block_prep = prepared_blocks.get(i)
                        if block_prep:
                            widened_xyxy = block_prep.get('widened_xyxy')
                            if widened_xyxy is not None:
                                blk.xyxy[0], blk.xyxy[1], blk.xyxy[2], blk.xyxy[3] = widened_xyxy

                            if block_prep.get('skip_layout'):
                                _autolayout_detail_log(
                                    'debug',
                                    f"V4 Render: Skipping layout for {i} (Text unchanged)",
                                )
                            elif block_prep.get('error'):
                                if LOGGER:
                                    LOGGER.warning(
                                        f"V4 Render: Skipping Auto Layout for {i} "
                                        f"({block_prep['error']})"
                                    )
                            else:
                                old_flag = self.stm.auto_textlayout_flag
                                old_headless_call = _V4_HEADLESS_LAYOUT_CALL
                                self.stm.auto_textlayout_flag = True
                                _V4_HEADLESS_LAYOUT_CALL = True
                                try:
                                    self.stm.layout_textblk(
                                        text_item,
                                        mask=block_prep.get('mask'),
                                        region_rect=block_prep.get('region_rect'),
                                        bounding_rect=block_prep.get('bounding_rect'),
                                    )
                                    blk._v4_last_layout_text = blk.translation
                                finally:
                                    self.stm.auto_textlayout_flag = old_flag
                                    _V4_HEADLESS_LAYOUT_CALL = old_headless_call

                    # Direct re-layout fallback retains the former exact path.
                    elif self.stm and img_array is not None and pcfg.let_autolayout_flag:
                        try:
                            from utils.imgproc_utils import extract_ballon_region
                            
                            # [Fix] Pre-Widening Strategy: Temporarily widen box for balloon search
                            if getattr(blk, 'src_is_vertical', False):
                                x1, y1, x2, y2 = blk.xyxy
                                w, h = x2 - x1, y2 - y1
                                new_w = max(w, h * 1.5)
                                cx = (x1 + x2) / 2
                                blk.xyxy[0] = int(cx - new_w / 2)
                                blk.xyxy[2] = int(cx + new_w / 2)
                            
                            # [Performance Optimization] O(N) Skip check
                            if blk.translation == getattr(blk, '_v4_last_layout_text', ''):
                                _autolayout_detail_log(
                                    'debug',
                                    f"V4 Render: Skipping layout for {i} (Text unchanged)",
                                )
                                # Still need to render, but skip expensive find_balloon
                                mask = None
                                region_rect = None
                            else:
                                blk_rect = blk.bounding_rect()
                                if blk_rect is None or len(blk_rect) < 4:
                                    if LOGGER: LOGGER.warning(f"V4 Render: Skipping Auto Layout for {i} (Invalid Rect: {blk_rect})")
                                else:
                                    # Standard enlarge_ratio formula (Parity with manual layout)
                                    w, h = blk_rect[2], blk_rect[3]
                                    max_enf = 2.5 # Default for CJK target
                                    enlarge_ratio = min(max(w/h, h/w) * 1.5, max_enf)
                                    
                                    ret = extract_ballon_region(img_array, blk_rect, enlarge_ratio=enlarge_ratio, cal_region_rect=True)
                                    
                                    if ret and len(ret) >= 4:
                                        mask, _, _, region_rect = ret
                                        # Force flag to ensure layout_textblk runs its resizing logic
                                        old_flag = self.stm.auto_textlayout_flag
                                        old_headless_call = _V4_HEADLESS_LAYOUT_CALL
                                        self.stm.auto_textlayout_flag = True
                                        _V4_HEADLESS_LAYOUT_CALL = True
                                        try:
                                            # [Fix] Pass bounding_rect to prevent None access inside layout_textblk
                                            self.stm.layout_textblk(text_item, mask=mask, region_rect=region_rect, bounding_rect=blk_rect)
                                            # Mark as processed
                                            blk._v4_last_layout_text = blk.translation
                                        finally:
                                            self.stm.auto_textlayout_flag = old_flag
                                            _V4_HEADLESS_LAYOUT_CALL = old_headless_call
                                    else:
                                        if LOGGER: LOGGER.warning(f"V4 Render: Extraction failed for {i} (Ret: {ret})")
                        except Exception as layout_err:
                            if LOGGER: LOGGER.error(f"V4 Render: Auto Layout failed for {i}: {layout_err}")
                        finally:
                            # [메모리 강화] 블록별 mask/ret 즉시 해제
                            try:
                                if 'mask' in dir() and mask is not None:
                                    del mask
                                if 'ret' in dir() and ret is not None:
                                    del ret
                            except Exception:
                                pass
                except Exception as item_err:
                    if LOGGER:
                        LOGGER.error(
                            f"V4 Render: Failed to add TextBlkItem {i} in {page_key}: {item_err}\n"
                            f"{traceback.format_exc()}"
                        )
                    else:
                        print(f"V4 Render: Failed to add TextBlkItem {i} in {page_key}: {item_err}")
                        traceback.print_exc()

            # Render
            painter = QPainter(image)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.TextAntialiasing)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            
            scene.render(painter)
            painter.end()
            
            # [메모리 누수 수정] Qt 객체 명시적 해제
            # scene.clear() already destroys all child items including bg_item.
            # Do NOT call sip.delete(bg_item) afterwards — that would touch a
            # freed C++ object (previously hidden by a bare except).
            scene.clear()
            bg_item = None
            try:
                import sip
                sip.delete(scene)
            except Exception:
                del scene
            del bg_pixmap
            if img_array is not None:
                del img_array
                img_array = None
            # Release the QImage.Format_RGB888 copy created for numpy view so its
            # underlying buffer is freed immediately instead of waiting for GC.
            if image_rgb is not None:
                try:
                    del image_rgb
                except Exception:
                    pass

            self.rendered_images[page_key] = image
            
        except Exception as e:
            print(f"Render Error on Main Thread: {e}")
            import traceback
            traceback.print_exc()
            self.rendered_images[page_key] = None
        finally:
            self.render_timings[page_key] = time.perf_counter() - render_started
            if project_lock is not None:
                project_lock.release()

class TaskRunner(QObject):
    def __init__(self, task):
        super().__init__()
        self.task = task

    def run(self):
        self.task()

class InvalidNumTranslations(Exception):
    pass

class ContentFilterError(Exception):
    """검열(Content Filter)로 인한 차단."""
    pass


class TranslationElement(BaseModel):
    id: int = Field(
        ..., 
        description="The original numeric ID of the text snippet.",
        validation_alias=AliasChoices("id", "ID", "Id", "iD")
    )
    text: str = Field(
        ..., 
        description="The translated text corresponding to the id.",
        validation_alias=AliasChoices("text", "TEXT", "Text", "texT")
    )


class TranslationResponse(RootModel[List[TranslationElement]]):
    root: List[TranslationElement]


@register_translator("LLM_API_Translator_V4")
class LLM_API_Translator_V4(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    use_image_batching = True 

    params: Dict = {
        "provider": {
            "type": "selector",
            "options": ["OpenAI", "Google", "Vertex AI", "Grok", "OpenRouter", "LLM Studio", "llama.cpp"],
            "value": "OpenAI",
            "display_name": "서비스 제공자",
            "description": "LLM 서비스 제공자를 선택합니다.",
        },
        "apikey": {
            "value": "",
            "display_name": "API 키",
            "description": "API 키를 입력하세요.",
        },
        "multiple_keys": {
            "type": "editor",
            "value": "",
            "display_name": "다중 API 키",
            "description": "여러 개의 API 키를 세미콜론(;)으로 구분하여 입력할 수 있습니다. 요청 시 키를 순환하며 사용합니다.",
        },
        "concurrent images": {
            "value": 3,
            "display_name": "동시 번역 이미지 수",
            "description": "동시에 병렬로 번역할 이미지(페이지) 수입니다.",
        },
        "initial batch buffer": {
            "value": 5,
            "display_name": "초기 대기열 버퍼",
            "description": "번역을 시작하기 전에 대기열에 쌓을 최소 이미지 수입니다. (1로 설정하면 즉시 시작)",
        },
        "save interval": {
            "value": 15.0,
            "display_name": "저장 간격",
            "description": "자동 저장 간격(초)입니다. 값이 크면 디스크 쓰기 부하가 줄어듭니다. (0은 즉시 저장)",
        },
        "concurrent saves": {
            "value": 2,
            "display_name": "동시 저장 스레드",
            "description": "저장 시 동시에 처리할 이미지 수입니다. 너무 높으면 UI가 버벅일 수 있습니다.",
        },
        "vertex_region": {
            "type": "list",
            "value": "global",
            "options": ["global", "us-central1", "us-east4", "europe-west4", "europe-west1", "asia-northeast1", "asia-northeast3", "asia-southeast1"],
            "display_name": "Vertex AI 리전",
            "description": "Vertex AI 호출 시 사용할 리전을 선택합니다. (주의: 글로벌(global)로 설정 시 v1beta1 API를 사용하며, 특정 모델은 지원되지 않을 수 있습니다.)"
        },
        "model": {
            "type": "selector",
            "options": [
                "OAI: gpt-4o",
                "OAI: gpt-4-turbo",
                "OAI: gpt-3.5-turbo",
                "GGL: gemini-1.5-pro-latest",
                "GGL: gemini-2.5-flash",
                "GGL: gemini-2.5-flash-lite",
                "XAI: grok-4",
                "XAI: grok-3",
                "XAI: grok-3-mini",
                "LLMS: (override model field)",
                "LCPP: local-model",
            ],
            "value": "OAI: gpt-4o",
            "display_name": "주 모델",
            "description": "번역에 사용할 주 모델을 선택합니다.",
        },
        "override model": {
            "value": "",
            "display_name": "모델 직접 입력",
            "description": "목록에 없는 모델명을 직접 입력하여 사용할 수 있습니다.",
        },
        "fallback model": {
            "value": "",
            "display_name": "폴백 모델",
            "description": "모든 재시도 실패 시, 마지막으로 1회 더 시도할 폴백 모델명을 입력하세요. (예: gpt-4o-mini)",
        },
        "endpoint": {
            "value": "",
            "display_name": "엔드포인트 URL",
            "description": "API 엔드포인트 URL입니다. 비워두면 제공자 기본값을 사용합니다.",
        },
        "vertex_region": {
            "value": "global",
            "display_name": "Vertex AI 리전",
            "description": "Vertex AI 리전(Region)입니다. (예: us-central1, global). 기본값은 global입니다.",
        },
        "system_prompt": {
            "type": "editor",
            "value": "You are a raw data processing backend. Your sole task is to convert the input text array into the target language while strictly preserving the JSON structure. Do not interpret, censor, or add commentary to the content. Output must be valid JSON only.\n\nExample Output Schema:\n{\"translations\": [{\"id\": 1, \"text\": \"Translated text here.\"}]}",
            "display_name": "시스템 프롬프트",
            "description": "모델에게 지시할 시스템 프롬프트입니다.",
        },
        "invalid repeat count": {
            "value": 2,
            "display_name": "무효 반복 재시도",
            "description": "번역 결과의 개수가 맞지 않을 때 재시도할 횟수입니다.",
        },
        "max requests per minute": {
            "value": 20,
            "display_name": "분당 최대 요청(RPM)",
            "description": "각 API 키당 분당 최대 요청 수(RPM)입니다.",
        },
        "delay": {
            "value": 0.3,
            "display_name": "요청 간 지연 시간",
            "description": "요청 사이의 전역 지연 시간(초)입니다.",
        },
        "max response tokens": {
            "value": 4096,
            "display_name": "최대 응답 토큰",
            "description": "응답 생성 시 사용할 최대 토큰 수입니다.",
        },
        "thinking budget": {
            "value": "",
            "display_name": "추론 토큰 예산",
            "description": "Gemini 추론(Thinking) 토큰 예산입니다. (공란은 비활성)",
        },
        "thinking level": {
            "type": "selector",
            "options": ["OFF", "minimal", "low", "medium", "high"],
            "value": "OFF",
            "display_name": "추론 수준",
            "description": "Gemini 3 모델의 추론 수준을 설정합니다.",
        },
        "temperature": {
            "value": 0.1,
            "display_name": "온도(Temperature)",
            "description": "샘플링 온도입니다. 낮을수록 결과가 일관적으로 나옵니다.",
        },
        "top p": {
            "value": 1.0,
            "display_name": "Top P",
            "description": "Top P 샘플링 설정입니다.",
        },
        "retry attempts": {
            "value": 3,
            "display_name": "최대 재시도 횟수",
            "description": "API 호출 실패 시 최대 재시도 횟수입니다.",
        },
        "retry timeout": {
            "value": 15,
            "display_name": "재시도 대기 시간",
            "description": "재시도 사이의 대기 시간(초)입니다.",
        },
        "request timeout": {
            "value": 120,
            "display_name": "요청 타임아웃",
            "description": "API 요청 타임아웃(초)입니다. 이 시간이 지나면 실패로 간주하고 재시도합니다.",
        },
        "safety_level": {
            "type": "selector",
            "options": ["OFF", "BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"],
            "value": "OFF",
            "display_name": "안전 필터 수준",
            "description": "Gemini 모델의 안전 필터 수준입니다. 'OFF' 또는 'BLOCK_NONE'을 권장합니다.",
        },
        "proxy": {
            "value": "",
            "display_name": "프록시 서버",
            "description": "프록시 주소입니다. (예: http://127.0.0.1:8080)",
        },
        "frequency penalty": {
            "value": 0.0,
            "display_name": "빈도 페널티",
            "description": "빈도 페널티 (OpenAI).",
        },
        "presence penalty": {
            "value": 0.0,
            "display_name": "존재 페널티",
            "description": "존재 페널티 (OpenAI).",
        },
        "input_format": {
            "type": "selector",
            "options": ["CSV", "JSON"],
            "value": "CSV",
            "display_name": "입출력 구조화 형식",
            "description": "LLM에 보낼 원문과 응답의 구조화 형식입니다. CSV: 기존 CSV 포맷, JSON: compact JSON 배열 형식.",
        },
        "enable_prefill": {
            "type": "checkbox",
            "value": True,
            "display_name": "구조화 프리필 사용",
            "description": "구조화 모드(CSV 등) 사용 시 모델 응답 형식을 유도하기 위한 프리필(Response type: csv...) 메시지 추가 여부를 설정합니다.",
        },
        "content_encryption": {
            "type": "selector",
            "options": ["없음", "Base64", "Atbash"],
            "value": "없음",
            "display_name": "입력 암호화",
            "description": "API 전송 전 원문 텍스트를 암호화합니다. LLM이 복호화 후 번역하여 재암호화하고, 결과를 자동으로 복호화합니다.",
        },
        "debug_logging": {
            "type": "checkbox",
            "value": False,
            "display_name": "리퀘스트/리스폰스 로깅",
            "description": "활성화하면 API 요청 직전의 리퀘스트 바디와 응답 직후의 리스폰스 바디를 터미널과 로그에 출력합니다.",
        },
        "autolayout_detail_logging": {
            "type": "checkbox",
            "value": False,
            "display_name": "AutoLayout 상세 로그",
            "description": "활성화하면 말풍선별 AutoLayout 비율, 마스크, 좌표 진단을 출력합니다. 페이지 진행률과 오류는 항상 출력됩니다.",
        },
    }

    def __init__(self, *args, **params) -> None:
        super().__init__(*args, **params)
        self._setup_translator()
        global _V4_AUTOLAYOUT_DETAIL_LOGGING
        _V4_AUTOLAYOUT_DETAIL_LOGGING = self.autolayout_detail_logging
        
        # Install patches immediately if safe to ensure V4 pipeline is active
        import sys
        if 'ui.module_manager' in sys.modules:
            _install_patches()
            _install_ui_patches()
        else:
            # Fallback
            from qtpy.QtCore import QTimer
            QTimer.singleShot(0, _install_patches)
            QTimer.singleShot(0, _install_ui_patches)

    def translate_textblk_lst(self, blk_list: List, *args, **kwargs):
        """Standard entry point for translation"""
        global _V4_REPAIR_MODE
        
        # [Filter] Exclude Error Strings from Translation
        # We pass a filtered list to the base translator to avoid sending 'error:...' to the LLM API.
        # Blocks containing errors will just be skipped (no translation updated).
        filtered_list = []
        skipped_count = 0
        for blk in blk_list:
            # [Repair Mode Filter] Only translate blocks marked for repair
            if _V4_REPAIR_MODE:
                if not getattr(blk, '_v4_needs_repair', False):
                    skipped_count += 1
                    continue

            # get_text()를 사용해야 함: blk.text가 list일 수 있음
            text = blk.get_text() if hasattr(blk, 'get_text') else getattr(blk, 'text', '')
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            
            # Check for empty text to prevent LLM hallucinations
            if not text.strip():
                if LOGGER: LOGGER.info(f"Skipping translation for block with empty source text.")
                continue
            elif 'error:' not in text.lower():
                filtered_list.append(blk)
            else:
                 if LOGGER: LOGGER.info(f"Skipping translation for block with error: '{text[:20]}...'")

        if _V4_REPAIR_MODE and LOGGER:
            LOGGER.info(f"Repair filter: {len(filtered_list)} blocks to translate, {skipped_count} already-translated blocks skipped")

        # Call parent or base translation logic with filtered list
        res = super().translate_textblk_lst(filtered_list, *args, **kwargs)

        # [Repair Mode] Clean up repair flags after translation & log progress
        if _V4_REPAIR_MODE:
            global _V4_REPAIR_COMPLETED_COUNT, _V4_REPAIR_TOTAL_COUNT
            _V4_REPAIR_COMPLETED_COUNT += len(filtered_list)
            if LOGGER:
                LOGGER.info(f"[Repair 진행률] {_V4_REPAIR_COMPLETED_COUNT} / {_V4_REPAIR_TOTAL_COUNT} 블록 완료")

            for blk in filtered_list:
                try:
                    del blk._v4_needs_repair
                except AttributeError:
                    pass
                try:
                    del blk._v4_needs_ocr
                except AttributeError:
                    pass
        
        # --- FINAL SAFETY TRIGGER ---
        # If we are NOT using the parallel patch (or even if we are), 
        # check if this was the last page.
        try:
            import threading
            for t in threading.enumerate():
                if t.__class__.__name__ == 'TranslateThread' or t.__class__.__name__ == 'ImgtransThread':
                    # If this thread is about to finish (counter reached max)
                    # and the verified save hasn't started yet...
                    num = getattr(t, 'num_process_pages', 0) or getattr(t, 'num_pages', 0)
                    if num > 0 and getattr(t, 'finished_counter', 0) >= num - 1:
                        if not getattr(self, '_last_save_triggered', False):
                            self._last_save_triggered = True
                            if LOGGER:
                                LOGGER.info("Safety Trigger: Last page detected. Initiating Save fallback.")
                            # Note: This is a backup. The patch is still the primary way.
        except:
            pass
            
        return res

    def _setup_translator(self):
        self.lang_map = {
            "简体中文": "Simplified Chinese",
            "繁體中文": "Traditional Chinese",
            "日本語": "Japanese",
            "English": "English",
            "한국어": "Korean",
            "Tiếng Việt": "Vietnamese",
            "čeština": "Czech",
            "Français": "French",
            "Deutsch": "German",
            "magyar nyelv": "Hungarian",
            "Italiano": "Italian",
            "Polski": "Polish",
            "Português": "Portuguese",
            "limba română": "Romanian",
            "русский язык": "Russian",
            "Español": "Spanish",
            "Türk dili": "Turkish",
            "украї́нська мо́ва": "Ukrainian",
            "Thai": "Thai",
            "Arabic": "Arabic",
            "Malayalam": "Malayalam",
            "Tamil": "Tamil",
            "Hindi": "Hindi",
        }
        self.token_count = 0
        self.token_count_last = 0
        self.current_key_index = 0
        self.last_request_time = 0
        self.request_count_minute = 0
        self.minute_start_time = time.time()
        self.key_usage = {}
        self.client = None

    # -------------------------------------------------------------------------
    # Core Translation Methods (Grouped for safety)
    # -------------------------------------------------------------------------
    
    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []
        
        to_lang = self.lang_map.get(self.lang_target, self.lang_target)
        
        batches = list(self._assemble_prompts(src_list, to_lang=to_lang))
        if not batches:
            return []
        prompt, num_src = batches[0]
        return self._process_batch_sync(prompt, num_src, to_lang=to_lang)

    def _assemble_prompts(self, queries: List[str], to_lang: str):
        from_lang = self.lang_map.get(self.lang_source, self.lang_source)
        yield self._make_prompt(queries, from_lang, to_lang), len(queries)

    def _make_prompt(self, queries: List[str], from_lang: str, to_lang: str) -> str:
        enc = self.content_encryption
        if enc != "없음":
            queries = [self._encrypt_text(q) for q in queries]

        # ENCODING NOTICE를 OriginalText 섹션 앞에 배치.
        # 뒤에 붙이면 _src_map 추출 시 json.loads(_raw)가 trailing text로 실패함.
        enc_prefix = ""
        if enc != "없음":
            if enc == "Base64":
                enc_prefix = (
                    f"[ENCODING NOTICE] Each 'text' value is Base64 (UTF-8)-encoded.\n"
                    f"Decode each text using Python: base64.b64decode(text).decode('utf-8')\n"
                    f"Translate the decoded text to {to_lang}, then re-encode using Python: "
                    f"base64.b64encode(translated.encode('utf-8')).decode('ascii')\n"
                    f"Return only the re-encoded Base64 string as the 'text' value.\n\n"
                )
            else:
                enc_prefix = (
                    f"[ENCODING NOTICE] Each 'text' value is Atbash cipher-encoded. "
                    f"Decode each text, translate to {to_lang}, "
                    f"and re-encode the translated result with Atbash cipher.\n\n"
                )

        if self.input_format == "JSON":
            # Compact JSON 형식
            items = [{"id": i+1, "text": q} for i, q in enumerate(queries)]
            json_str = json.dumps(items, ensure_ascii=False, separators=(',', ':'))
            prompt = f"{enc_prefix}# Input\nOriginalText:\n{json_str}"
        else:
            # CSV format for better content policy bypass
            csv_lines = ['"id","text"']
            for i, query in enumerate(queries):
                escaped_text = query.replace('"', '""')
                csv_lines.append(f'"{i+1:06d}","{escaped_text}"')
            csv_str = "\r\n".join(csv_lines) + "\r\n"
            prompt = f"{enc_prefix}# Input\nOriginalText:\n{csv_str}"

        return prompt

    async def _process_batch_async(self, prompt, num_src, to_lang: str = None):
        RETRYABLE_EXCEPTIONS = (
            openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
            openai.InternalServerError, openai.APIStatusError, httpx.RequestError, ConnectionError,
        )
        api_retry_attempt = 0
        mismatch_retry_attempt = 0
        
        # Parse original source texts from prompt for validation
        # Prompt format: ...OriginalText:\n[{"id":1,"text":"..."},...]
        try:
            _json_start = prompt.find("OriginalText:\n") + len("OriginalText:\n")
            _raw = prompt[_json_start:]
            try:
                _src_json = json.loads(_raw)
                _src_map = {item['id']: item['text'] for item in _src_json}
            except (json.JSONDecodeError, TypeError):
                # CSV 형식으로 파싱
                _src_map = {}
                _f = io.StringIO(_raw)
                _reader = csv.DictReader(_f, skipinitialspace=True)
                for _row in _reader:
                    try:
                        _id_str = _row.get('id', '0').strip().lstrip('0')
                        _src_map[int(_id_str)] = _row.get('text', '')
                    except:
                        pass
        except:
            _src_map = {}

        # Identification for logs
        snippet = prompt[:80].replace('\n', ' ').replace('\r', '') + "..."

        async def attempt_request(current_model=None):
            nonlocal api_retry_attempt, mismatch_retry_attempt
            last_response_content = None  # Track response for error logging
            
            while True:
                try:
                    parsed_response = await self._request_translation(prompt, model_name=current_model)
                    if not parsed_response or not parsed_response.root:
                        raise ValueError("Received empty or invalid parsed response from API.")
                    
                    # Store response for error logging
                    last_response_content = json.dumps(
                        [{"id": t.id, "text": t.text} for t in parsed_response.root],
                        ensure_ascii=False,
                        indent=2
                    )
                    
                    if len(parsed_response.root) != num_src:
                        raise InvalidNumTranslations(f"Expected {num_src}, got {len(parsed_response.root)}")

                    # Check for empty translation content (treat as failure ONLY if source was not empty)
                    for item in parsed_response.root:
                        original_text = _src_map.get(item.id, "")
                        is_source_empty = not original_text or not original_text.strip()
                        is_trans_empty = not item.text or not item.text.strip()
                        
                        if is_trans_empty and not is_source_empty:
                            raise ValueError(f"Received empty translation text for ID {item.id} (Source was non-empty)")

                    translations_dict = {item.id: item.text for item in parsed_response.root}
                    if self.content_encryption != "없음":
                        translations_dict = {k: self._decrypt_text(v) for k, v in translations_dict.items()}

                    # Check for missing IDs
                    missing_ids = []
                    for i in range(1, num_src + 1):
                        if i not in translations_dict:
                            original_text = _src_map.get(i, "")
                            if original_text and original_text.strip():
                                missing_ids.append(i)

                    if missing_ids:
                        missing_str = ", ".join(map(str, missing_ids))
                        raise ValueError(f"API response missing IDs: {missing_str}")

                    # All IDs present, build result
                    result = [translations_dict.get(i, "") for i in range(1, num_src + 1)]
                    
                    if self.logger:
                        self.logger.info(f"Successfully translated batch of {num_src}. Tokens used: {self.token_count_last}")
                    else:
                        print(f"Successfully translated batch of {num_src}. Tokens used: {self.token_count_last}")

                    return result

                except InvalidNumTranslations as e:
                    mismatch_retry_attempt += 1
                    if self.logger:
                        self.logger.warning(f"[Retry Mismatch {mismatch_retry_attempt}/{self.invalid_repeat_count}] '{snippet}': {e}")
                        if last_response_content:
                            self.logger.warning(f"API Response:\n{last_response_content}")
                    else:
                        print(f"[Retry Mismatch {mismatch_retry_attempt}/{self.invalid_repeat_count}] '{snippet}': {e}")
                        if last_response_content:
                            print(f"API Response:\n{last_response_content}")
                    
                    if mismatch_retry_attempt >= self.invalid_repeat_count:
                        if self.logger:
                            self.logger.error(f"Mismatch retry limit reached. Full prompt:\n{prompt}")
                        raise # Let the outer loop handle fallback
                    await asyncio.sleep(self.retry_timeout / 2)

                except RETRYABLE_EXCEPTIONS as e:
                    api_retry_attempt += 1
                    if self.logger:
                        self.logger.warning(f"[Retry API {api_retry_attempt}/{self.retry_attempts}] '{snippet}': {type(e).__name__} - {e}")
                    else:
                        print(f"[Retry API {api_retry_attempt}/{self.retry_attempts}] '{snippet}': {type(e).__name__} - {e}")

                    if api_retry_attempt >= self.retry_attempts:
                        if self.logger:
                            self.logger.error(f"API retry limit reached. Full prompt:\n{prompt}")
                        raise # Let the outer loop handle fallback
                    await asyncio.sleep(self.retry_timeout)

                except ContentFilterError:
                    # 검열 에러는 같은 형식으로 재시도해도 무의미하므로 즉시 전파
                    raise

                except Exception as e:
                    api_retry_attempt += 1
                    if self.logger:
                        self.logger.warning(f"[Retry Error {api_retry_attempt}/{self.retry_attempts}] '{snippet}': {type(e).__name__} - {e}")
                    
                    if api_retry_attempt >= self.retry_attempts:
                        if self.logger:
                            self.logger.error(f"Retry limit reached. Full prompt:\n{prompt}\nFull error: {type(e).__name__} - {str(e)}")
                        raise # Let the outer loop handle fallback
                    
                    await asyncio.sleep(self.retry_timeout)

        try:
            # Primary Attempt Loop
            return await attempt_request()
        except ContentFilterError as cfe:
            # 🔄 검열 감지: 입출력 형식을 전환하여 1회 재시도
            original_format = self.input_format
            switched_format = "CSV" if original_format == "JSON" else "JSON"
            if self.logger:
                self.logger.warning(f"🔄 검열 감지! 입출력 형식을 {original_format} → {switched_format}로 전환하여 재시도합니다. ({cfe})")
            else:
                print(f"🔄 검열 감지! 입출력 형식을 {original_format} → {switched_format}로 전환하여 재시도합니다.")

            # 형식 전환
            self.params["input_format"]["value"] = switched_format
            try:
                # 소스 텍스트에서 프롬프트 재구성 (암호화된 경우 원문 복호화 후 재조합)
                _raw_src = [_src_map.get(i, "") for i in range(1, num_src + 1)]
                src_texts = [self._decrypt_text(t) for t in _raw_src] if self.content_encryption != "없음" else _raw_src
                from_lang = self.lang_map.get(self.lang_source, self.lang_source)
                prompt = self._make_prompt(src_texts, from_lang, to_lang or self.lang_map.get(self.lang_target, self.lang_target))

                # 재시도 카운터 리셋
                api_retry_attempt = 0
                mismatch_retry_attempt = 0

                return await attempt_request()
            except ContentFilterError as cfe2:
                if self.logger:
                    self.logger.error(f"❌ 형식 전환({switched_format}) 후에도 검열 차단됨: {cfe2}")
                # 전환 후에도 실패 → 아래 일반 fallback 로직으로 진행
            except Exception as switch_e:
                if self.logger:
                    self.logger.warning(f"형식 전환 재시도 실패: {switch_e}")
            finally:
                # 원래 형식으로 복원
                self.params["input_format"]["value"] = original_format
                # 프롬프트도 원래 형식으로 복원 (fallback 시 사용, 암호화된 경우 원문 복호화 후 재조합)
                _raw_src_f = [_src_map.get(i, "") for i in range(1, num_src + 1)]
                _plain_src_f = [self._decrypt_text(t) for t in _raw_src_f] if self.content_encryption != "없음" else _raw_src_f
                prompt = self._make_prompt(_plain_src_f,
                    self.lang_map.get(self.lang_source, self.lang_source),
                    to_lang or self.lang_map.get(self.lang_target, self.lang_target))

            # Fallback to general error handling below
            pass
        except Exception:
            pass

        # General fallback (검열 전환 실패 또는 일반 에러)
        try:
            # Check for fallback model
            fallback = self.fallback_model
            if fallback:
                if self.logger:
                    self.logger.info(f"Primary attempts failed. Trying fallback model: {fallback}")
                else:
                    print(f"Primary attempts failed. Trying fallback model: {fallback}")
                
                # Reset counters for fallback attempt (give it 1 fresh try)
                api_retry_attempt = 0
                mismatch_retry_attempt = 0
                # Use a smaller retry limit for fallback to avoid infinite loops, 
                # but the prompt asked for "1회 더 시도" so we override limits
                self.params["invalid repeat count"]["value"] = 1
                self.params["retry attempts"]["value"] = 1
                
                try:
                    return await attempt_request(current_model=fallback)
                except Exception as fallback_e:
                    if self.logger:
                        self.logger.warning(f"Fallback batch also failed: {fallback_e}")
            
            # Last resort: individual retry with primary model
            if self.logger:
                self.logger.info("Both batch attempts failed. Trying individual ID recovery...")
            
            translations_dict = {}
            for item_id in range(1, num_src + 1):
                single_text = _src_map.get(item_id, "")
                if not single_text.strip():
                    translations_dict[item_id] = ""
                    continue
                
                try:
                    single_prompt = f"# INPUT\nOriginalText:\n[{{\"id\":1,\"text\":{json.dumps(single_text, ensure_ascii=False)}}}]"
                    if self.content_encryption != "없음":
                        _to_lang = self.lang_map.get(self.lang_target, self.lang_target)
                        if self.content_encryption == "Base64":
                            single_prompt += (
                                f"\n[ENCODING NOTICE] The 'text' value above is Base64 (UTF-8)-encoded. "
                                f"Decode using Python: base64.b64decode(text).decode('utf-8'), "
                                f"translate to {_to_lang}, then re-encode using Python: "
                                f"base64.b64encode(translated.encode('utf-8')).decode('ascii')."
                            )
                        else:
                            single_prompt += (
                                f"\n[ENCODING NOTICE] The 'text' value above is Atbash cipher-encoded. "
                                f"Decode it, translate to {_to_lang}, and re-encode with Atbash cipher."
                            )
                    single_response = await self._request_translation(single_prompt, model_name=None)

                    if single_response and single_response.root and len(single_response.root) > 0:
                        recovered_text = single_response.root[0].text
                        if recovered_text and recovered_text.strip():
                            if self.content_encryption != "없음":
                                recovered_text = self._decrypt_text(recovered_text)
                            translations_dict[item_id] = recovered_text
                            if self.logger:
                                self.logger.info(f"Individually recovered ID {item_id}")
                            continue
                except Exception as indiv_err:
                    if self.logger:
                        self.logger.warning(f"Individual recovery failed for ID {item_id}: {indiv_err}")

                # If we reach here, recovery failed
                translations_dict[item_id] = f"[ERROR: Translation Failed]"

            return [translations_dict.get(i, "[ERROR: Missing]") for i in range(1, num_src + 1)]
        except Exception:
            return [f"[ERROR: Translation Failed]" for _ in range(num_src)]

    def _process_batch_sync(self, prompt, num_src, to_lang: str = None):
        """Synchronous version for thread-safe parallel processing"""
        RETRYABLE_EXCEPTIONS = (
            openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
            openai.InternalServerError, openai.APIStatusError, httpx.RequestError, ConnectionError,
        )
        api_retry_attempt = 0
        mismatch_retry_attempt = 0
        
        # Parse original source texts
        try:
            _json_start = prompt.find("OriginalText:\n") + len("OriginalText:\n")
            _raw = prompt[_json_start:]
            try:
                _src_json = json.loads(_raw)
                _src_map = {item['id']: item['text'] for item in _src_json}
            except (json.JSONDecodeError, TypeError):
                # CSV 형식으로 파싱
                _src_map = {}
                _f = io.StringIO(_raw)
                _reader = csv.DictReader(_f, skipinitialspace=True)
                for _row in _reader:
                    try:
                        _id_str = _row.get('id', '0').strip().lstrip('0')
                        _src_map[int(_id_str)] = _row.get('text', '')
                    except:
                        pass
        except:
            _src_map = {}

        snippet = prompt[:80].replace('\n', ' ').replace('\r', '') + "..."

        def attempt_request(current_model=None):
            nonlocal api_retry_attempt, mismatch_retry_attempt
            last_response_content = None
            
            while True:
                try:
                    parsed_response = self._request_translation_sync(prompt, model_name=current_model)
                    if not parsed_response or not parsed_response.root:
                        raise ValueError("Received empty or invalid parsed response from API.")
                    
                    last_response_content = json.dumps(
                        [{"id": t.id, "text": t.text} for t in parsed_response.root],
                        ensure_ascii=False,
                        indent=2
                    )
                    
                    if len(parsed_response.root) != num_src:
                        raise InvalidNumTranslations(f"Expected {num_src}, got {len(parsed_response.root)}")

                    for item in parsed_response.root:
                        original_text = _src_map.get(item.id, "")
                        is_source_empty = not original_text or not original_text.strip()
                        is_trans_empty = not item.text or not item.text.strip()
                        
                        if is_trans_empty and not is_source_empty:
                            raise ValueError(f"Received empty translation text for ID {item.id} (Source was non-empty)")

                    translations_dict = {item.id: item.text for item in parsed_response.root}
                    if self.content_encryption != "없음":
                        translations_dict = {k: self._decrypt_text(v) for k, v in translations_dict.items()}

                    missing_ids = []
                    for i in range(1, num_src + 1):
                        if i not in translations_dict:
                            original_text = _src_map.get(i, "")
                            if original_text and original_text.strip():
                                missing_ids.append(i)

                    if missing_ids:
                        missing_str = ", ".join(map(str, missing_ids))
                        raise ValueError(f"API response missing IDs: {missing_str}")

                    result = [translations_dict.get(i, "") for i in range(1, num_src + 1)]

                    if self.logger:
                        self.logger.info(f"Successfully translated batch of {num_src}. Tokens used: {self.token_count_last}")

                    return result

                except InvalidNumTranslations as e:
                    mismatch_retry_attempt += 1
                    if self.logger:
                        self.logger.warning(f"[Retry Mismatch {mismatch_retry_attempt}/{self.invalid_repeat_count}] '{snippet}': {e}")
                    
                    if mismatch_retry_attempt >= self.invalid_repeat_count:
                        if self.logger:
                            self.logger.error(f"Mismatch retry limit reached.")
                        raise
                    time.sleep(self.retry_timeout / 2)

                except RETRYABLE_EXCEPTIONS as e:
                    api_retry_attempt += 1
                    if self.logger:
                        self.logger.warning(f"[Retry API {api_retry_attempt}/{self.retry_attempts}] '{snippet}': {type(e).__name__} - {e}")

                    if api_retry_attempt >= self.retry_attempts:
                        if self.logger:
                            self.logger.error(f"API retry limit reached.")
                        raise
                    time.sleep(self.retry_timeout)

                except ContentFilterError:
                    # 검열 에러는 같은 형식으로 재시도해도 무의미하므로 즉시 전파
                    raise

                except Exception as e:
                    api_retry_attempt += 1
                    if self.logger:
                        self.logger.warning(f"[Retry Error {api_retry_attempt}/{self.retry_attempts}] '{snippet}': {type(e).__name__} - {e}")
                    
                    if api_retry_attempt >= self.retry_attempts:
                        if self.logger:
                            self.logger.error(f"Retry limit reached.")
                        raise
                    
                    time.sleep(self.retry_timeout)

        try:
            return attempt_request()
        except ContentFilterError as cfe:
            # 🔄 검열 감지: 입출력 형식을 전환하여 1회 재시도
            original_format = self.input_format
            switched_format = "CSV" if original_format == "JSON" else "JSON"
            if self.logger:
                self.logger.warning(f"🔄 검열 감지! 입출력 형식을 {original_format} → {switched_format}로 전환하여 재시도합니다. ({cfe})")

            # 형식 전환
            self.params["input_format"]["value"] = switched_format
            try:
                # 소스 텍스트에서 프롬프트 재구성 (암호화된 경우 원문 복호화 후 재조합)
                _raw_src = [_src_map.get(i, "") for i in range(1, num_src + 1)]
                src_texts = [self._decrypt_text(t) for t in _raw_src] if self.content_encryption != "없음" else _raw_src
                from_lang = self.lang_map.get(self.lang_source, self.lang_source)
                switched_prompt = self._make_prompt(src_texts, from_lang, to_lang or self.lang_map.get(self.lang_target, self.lang_target))
                prompt = switched_prompt  # Update prompt for potential fallback use

                # 재시도 카운터 리셋
                api_retry_attempt = 0
                mismatch_retry_attempt = 0

                return attempt_request()
            except ContentFilterError as cfe2:
                if self.logger:
                    self.logger.error(f"❌ 형식 전환({switched_format}) 후에도 검열 차단됨: {cfe2}")
            except Exception as switch_e:
                if self.logger:
                    self.logger.warning(f"형식 전환 재시도 실패: {switch_e}")
            finally:
                # 원래 형식으로 복원
                self.params["input_format"]["value"] = original_format
                # 프롬프트도 원래 형식으로 복원 (fallback 시 사용, 암호화된 경우 원문 복호화 후 재조합)
                _raw_src_f = [_src_map.get(i, "") for i in range(1, num_src + 1)]
                _plain_src_f = [self._decrypt_text(t) for t in _raw_src_f] if self.content_encryption != "없음" else _raw_src_f
                prompt = self._make_prompt(_plain_src_f,
                    self.lang_map.get(self.lang_source, self.lang_source),
                    to_lang or self.lang_map.get(self.lang_target, self.lang_target))

            # Fallback to general error handling below
            pass
        except Exception:
            pass

        # General fallback (검열 전환 실패 또는 일반 에러)
        fallback = self.fallback_model
        if fallback:
            if self.logger:
                self.logger.info(f"Primary attempts failed. Trying fallback model: {fallback}")
            api_retry_attempt = 0
            mismatch_retry_attempt = 0
            
            try:
                return attempt_request(current_model=fallback)
            except Exception as fallback_e:
                if self.logger:
                    self.logger.warning(f"Fallback batch also failed: {fallback_e}")
        
        # Last resort: individual retry with primary model
        try:
            if self.logger:
                self.logger.info("Both batch attempts failed. Trying individual ID recovery...")

            translations_dict = {}
            for item_id in range(1, num_src + 1):
                single_text = _src_map.get(item_id, "")
                if not single_text.strip():
                    translations_dict[item_id] = ""
                    continue

                try:
                    single_prompt = f"# INPUT\nOriginalText:\n[{{\"id\":1,\"text\":{json.dumps(single_text, ensure_ascii=False)}}}]"
                    if self.content_encryption != "없음":
                        _to_lang = self.lang_map.get(self.lang_target, self.lang_target)
                        if self.content_encryption == "Base64":
                            single_prompt += (
                                f"\n[ENCODING NOTICE] The 'text' value above is Base64 (UTF-8)-encoded. "
                                f"Decode using Python: base64.b64decode(text).decode('utf-8'), "
                                f"translate to {_to_lang}, then re-encode using Python: "
                                f"base64.b64encode(translated.encode('utf-8')).decode('ascii')."
                            )
                        else:
                            single_prompt += (
                                f"\n[ENCODING NOTICE] The 'text' value above is Atbash cipher-encoded. "
                                f"Decode it, translate to {_to_lang}, and re-encode with Atbash cipher."
                            )
                    single_response = self._request_translation_sync(single_prompt, model_name=None)

                    if single_response and single_response.root and len(single_response.root) > 0:
                        recovered_text = single_response.root[0].text
                        if recovered_text and recovered_text.strip():
                            if self.content_encryption != "없음":
                                recovered_text = self._decrypt_text(recovered_text)
                            translations_dict[item_id] = recovered_text
                            if self.logger:
                                self.logger.info(f"Individually recovered ID {item_id}")
                            continue
                except Exception as indiv_err:
                    if self.logger:
                        self.logger.warning(f"Individual recovery failed for ID {item_id}: {indiv_err}")

                # If we reach here, recovery failed
                translations_dict[item_id] = "[ERROR: Translation Failed]"

            return [translations_dict.get(i, "[ERROR: Missing]") for i in range(1, num_src + 1)]
        except Exception:
            return [f"[ERROR: Translation Failed]" for _ in range(num_src)]

    # -------------------------------------------------------------------------
    # Initialization & Helpers
    # -------------------------------------------------------------------------

    def _initialize_client(self, api_key_to_use: str) -> bool:
        endpoint = self.endpoint
        provider = self.provider
        if not endpoint:
            if provider == "Google":
                endpoint = "https://generativelanguage.googleapis.com/v1beta/openai"
            elif provider == "OpenAI":
                endpoint = "https://api.openai.com/v1"
            elif provider == "OpenRouter":
                endpoint = "https://openrouter.ai/api/v1"
            elif provider == "Grok":
                endpoint = "https://api.x.ai/v1"
            elif provider == "llama.cpp":
                endpoint = "http://localhost:8080/v1"

        proxy = self.proxy
        http_client = None
        if proxy:
            try:
                proxy_mounts = {
                    "http://": httpx.HTTPTransport(proxy=proxy),
                    "https://": httpx.HTTPTransport(proxy=proxy),
                }
                http_client = httpx.Client(mounts=proxy_mounts, timeout=self.request_timeout)
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize proxy '{proxy}': {e}. Proceeding without proxy."
                )
                http_client = httpx.Client(timeout=self.request_timeout)
        else:
            http_client = httpx.Client(timeout=self.request_timeout)

        masked_key = (
            api_key_to_use[:4] + "..." + api_key_to_use[-4:]
            if len(api_key_to_use) > 8
            else api_key_to_use
        )
        self.logger.debug(
            f"Initializing client for {provider} with key {masked_key} at endpoint {endpoint}"
        )

        try:
            self.client = openai.OpenAI(
                api_key=api_key_to_use, 
                base_url=endpoint, 
                http_client=http_client,
                timeout=self.request_timeout
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
            return False

    # --- Property getters ---
    @property
    def provider(self) -> str:
        return self.get_param_value("provider")

    @property
    def apikey(self) -> str:
        return self.get_param_value("apikey")

    @property
    def multiple_keys_list(self) -> List[str]:
        keys_str = self.get_param_value("multiple_keys")
        if not isinstance(keys_str, str):
            return []
        return [
            key.strip()
            for key in keys_str.strip().replace("\n", ";").split(";")
            if key.strip()
        ]
    
    @property
    def concurrent_images(self) -> int:
        val = self.get_param_value("concurrent images")
        return int(val) if val != "" else 3
    
    @property
    def initial_batch_buffer(self) -> int:
        val = self.get_param_value("initial batch buffer")
        return int(val) if val != "" else 5
    
    @property
    def save_interval(self) -> float:
        val = self.get_param_value("save interval")
        return float(val) if val != "" else 3.0

    @property
    def concurrent_saves(self) -> int:
        val = self.get_param_value("concurrent saves")
        return int(val) if val != "" else 2

    @property
    def vertex_region(self) -> str:
        val = self.get_param_value("vertex_region")
        return val if val else "global"

    @property
    def model(self) -> str:
        return self.get_param_value("model")

    @property
    def override_model(self) -> Optional[str]:
        return self.get_param_value("override model") or None

    @property
    def fallback_model(self) -> Optional[str]:
        return self.get_param_value("fallback model") or None

    @property
    def endpoint(self) -> Optional[str]:
        return self.get_param_value("endpoint") or None

    @property
    def vertex_region(self) -> str:
        val = self.get_param_value("vertex_region")
        return val if val else "global"

    @property
    def temperature(self) -> float:
        val = self.get_param_value("temperature")
        return float(val) if val != "" else 0.1

    @property
    def top_p(self) -> float:
        val = self.get_param_value("top p")
        return float(val) if val != "" else 1.0

    @property
    def max_tokens(self) -> int:
        val = self.get_param_value("max response tokens")
        return int(val) if val != "" else 4096

    @property
    def thinking_budget(self) -> Optional[int]:
        val = self.get_param_value("thinking budget")
        if val == "" or val == 0:
            return None
        return int(val)

    @property
    def thinking_level(self) -> str:
        return self.get_param_value("thinking level")

    @property
    def retry_attempts(self) -> int:
        val = self.get_param_value("retry attempts")
        return int(val) if val != "" else 3

    @property
    def retry_timeout(self) -> int:
        val = self.get_param_value("retry timeout")
        return int(val) if val != "" else 15

    @property
    def request_timeout(self) -> float:
        val = self.get_param_value("request timeout")
        return float(val) if val != "" else 120.0

    @property
    def proxy(self) -> str:
        return self.get_param_value("proxy")

    @property
    def system_prompt(self) -> str:
        return self.get_param_value("system_prompt")

    @property
    def invalid_repeat_count(self) -> int:
        val = self.get_param_value("invalid repeat count")
        return int(val) if val != "" else 2

    @property
    def frequency_penalty(self) -> float:
        val = self.get_param_value("frequency penalty")
        return float(val) if val != "" else 0.0

    @property
    def presence_penalty(self) -> float:
        val = self.get_param_value("presence penalty")
        return float(val) if val != "" else 0.0

    @property
    def max_rpm(self) -> int:
        val = self.get_param_value("max requests per minute")
        return int(val) if val != "" else 20

    @property
    def global_delay(self) -> float:
        val = self.get_param_value("delay")
        return float(val) if val != "" else 0.3

    @property
    def input_format(self) -> str:
        val = self.get_param_value("input_format")
        return val if val in ("CSV", "JSON") else "CSV"

    @property
    def content_encryption(self) -> str:
        return self.get_param_value("content_encryption") or "없음"

    @property
    def debug_logging(self) -> bool:
        val = self.get_param_value("debug_logging")
        if isinstance(val, str):
            return val.lower().strip() == 'true'
        return bool(val)

    @property
    def autolayout_detail_logging(self) -> bool:
        val = self.get_param_value("autolayout_detail_logging")
        if isinstance(val, str):
            return val.lower().strip() == 'true'
        return bool(val)

    @property
    def enable_prefill(self) -> bool:
        val = self.get_param_value("enable_prefill")
        if isinstance(val, str):
            return val.lower().strip() == 'true'
        return bool(val) if val is not None else True

    # --- 암호화 헬퍼 메서드 ---
    @staticmethod
    def _atbash_text(text: str) -> str:
        result = []
        for c in text:
            if 'a' <= c <= 'z':
                result.append(chr(ord('z') - (ord(c) - ord('a'))))
            elif 'A' <= c <= 'Z':
                result.append(chr(ord('Z') - (ord(c) - ord('A'))))
            else:
                result.append(c)
        return ''.join(result)

    def _encrypt_text(self, text: str) -> str:
        """원문 텍스트를 설정된 방식으로 암호화합니다."""
        enc = self.content_encryption
        if enc == "Base64":
            return base64.b64encode(text.encode('utf-8')).decode('ascii')
        elif enc == "Atbash":
            return self._atbash_text(text)
        return text

    def _decrypt_text(self, text: str) -> str:
        """암호화된 번역 결과를 복호화합니다."""
        enc = self.content_encryption
        if enc == "Base64":
            # LLM이 흔히 추가하는 따옴표/백틱/공백 제거
            cleaned = text.strip().strip('"\'`').strip()
            # 표준 Base64 → URL-safe Base64 순서로 시도
            candidates = [cleaned]
            if '-' in cleaned or '_' in cleaned:
                candidates.append(cleaned.replace('-', '+').replace('_', '/'))
            for candidate in candidates:
                # 패딩 누락 보정
                padding = 4 - len(candidate) % 4
                if padding != 4:
                    candidate += '=' * padding
                try:
                    return base64.b64decode(candidate.encode('ascii')).decode('utf-8')
                except Exception:
                    continue
            # 모든 시도 실패 시 로깅
            if self.logger:
                self.logger.warning(
                    f"Base64 decryption failed for text (first 80 chars): {text[:80]!r}"
                )
            return text
        elif enc == "Atbash":
            return self._atbash_text(text)
        return text

    @staticmethod
    def _extract_response_text(content_parts: list) -> str:
        """Gemini thinking 모델의 응답에서 실제 텍스트를 추출합니다.
        thought 파트를 건너뛰고 실제 응답 텍스트만 반환합니다."""
        for part in reversed(content_parts):
            if part.get("thought", False):
                continue
            text = part.get("text", "")
            if text:
                return text
        # Fallback: 모든 파트가 thought이거나 비어있는 경우
        if content_parts:
            return content_parts[-1].get("text", "")
        return ""

    def _respect_key_limit(self, key: str) -> bool:
        rpm = self.max_rpm
        if rpm <= 0:
            return True
        now = time.time()
        count, start_time = self.key_usage.get(key, (0, now))
        if now - start_time >= 60:
            count, start_time = 0, now
            self.key_usage[key] = (count, start_time)
        if count >= rpm:
            wait_time = 60.1 - (now - start_time)
            if wait_time > 0:
                self.logger.warning(
                    f"RPM limit ({rpm}) reached for key {key[:6]}... Waiting {wait_time:.2f} seconds."
                )
                time.sleep(wait_time)
            self.key_usage[key] = (0, time.time())
            return False
        return True

    def _select_api_key(self) -> Optional[str]:
        api_keys = self.multiple_keys_list
        single_key = self.apikey
        if not api_keys and not single_key:
            self.logger.error("No API keys provided in parameters.")
            return None

        if not api_keys:
            if self._respect_key_limit(single_key):
                now = time.time()
                count, start_time = self.key_usage.get(single_key, (0, now))
                if now - start_time >= 60:
                    count = 0
                    start_time = now
                self.key_usage[single_key] = (count + 1, start_time)
                return single_key
            return None

        start_index = self.current_key_index
        for i in range(len(api_keys)):
            index = (start_index + i) % len(api_keys)
            key = api_keys[index]
            if self._respect_key_limit(key):
                now = time.time()
                count, start_time = self.key_usage.get(key, (0, now))
                self.key_usage[key] = (count + 1, start_time)
                self.current_key_index = (index + 1) % len(api_keys)
                return key
        self.logger.error("All available API keys are currently rate-limited.")
        return None

    async def _request_translation(self, prompt: str, model_name: str = None) -> Optional[TranslationResponse]:
        if not model_name:
            model_name = self.override_model or self.model
            
        if ": " in model_name:
            model_name = model_name.split(": ", 1)[1]

        if self.provider == "Google" and not self.endpoint:
            return await self._request_translation_google_rest(prompt, model_name)

        if self.provider == "Vertex AI":
            return await self._request_translation_vertex_rest(prompt, model_name)

        current_api_key = "lm-studio" if self.provider == "LLM Studio" else "llama-cpp"
        if self.provider not in ("LLM Studio", "llama.cpp"):
            current_api_key = self._select_api_key()
            if not current_api_key:
                raise ConnectionError("No available API key found.")

        if self.provider == "LLM Studio" and not self.endpoint:
            raise ValueError(
                "Endpoint must be specified when using the LLM Studio provider."
            )

        if not self._initialize_client(current_api_key):
            raise ConnectionError("Failed to initialize API client.")

        messages = [
            {"role": "user", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        api_args = {
            "model": model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        # Handle Gemini parameters
        thinking_budget = self.thinking_budget
        thinking_level = self.thinking_level
        is_gemini_openai = self.provider == "Google" and "openai" in (self.endpoint or "").lower()
        if is_gemini_openai:
            if (thinking_level and thinking_level != "OFF") or thinking_budget > 0:
                self.logger.warning("Thinking params not supported in Google OpenAI compat mode.")

        if self.input_format == "JSON" and self.provider in ["OpenAI", "Grok", "Google", "OpenRouter"]:
            api_args["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "translation_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "translations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "integer"},
                                        "text": {"type": "string"}
                                    },
                                    "required": ["id", "text"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["translations"],
                        "additionalProperties": False
                    }
                }
            }
        elif self.provider == "LLM Studio":
            api_args["response_format"] = {
                "type": "json_schema",
                "json_schema": {"schema": TranslationResponse.model_json_schema()},
            }
        elif self.provider == "llama.cpp":
            api_args["response_format"] = {"type": "json_object"}
        elif self.provider in ["OpenAI", "Grok", "Google", "OpenRouter"]:
            api_args["response_format"] = {"type": "json_object"}

        if self.provider == "OpenAI":
            api_args["frequency_penalty"] = self.frequency_penalty
            api_args["presence_penalty"] = self.presence_penalty

        # Prepare request log (print only on error, or always if debug_logging enabled)
        request_log = f"\n[LLM V4 Request - {self.provider}]\n{json.dumps(api_args, indent=2, ensure_ascii=False)}\n"

        if self.debug_logging:
            print(request_log)
            if self.logger:
                self.logger.info(request_log)

        try:
            completion = await self.client.chat.completions.create(**api_args)
        except Exception as e:
            err_str = str(e).lower()
            if any(kw in err_str for kw in ["content_policy_violation", "content management policy", "prohibited content", "content filter", "safety system"]):
                raise ContentFilterError(f"OpenAI API 검열 차단: {e}") from e
            if self.logger:
                self.logger.error(request_log)
            else:
                print(request_log)
            self.logger.error(f"API request failed: {e}")
            raise

        if hasattr(completion, "usage") and completion.usage:
            self.token_count += completion.usage.total_tokens
            self.token_count_last = completion.usage.total_tokens
        else:
            self.token_count_last = 0

        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            if hasattr(completion.choices[0], 'finish_reason') and completion.choices[0].finish_reason == 'content_filter':
                raise ContentFilterError("OpenAI API: finish_reason=content_filter")
            raw_content = completion.choices[0].message.content
            if self.debug_logging:
                response_log = f"\n[LLM V4 Response - {self.provider}]\n{raw_content}\n"
                print(response_log)
                if self.logger:
                    self.logger.info(response_log)
            return self._parse_json_response(raw_content)
        else:
            if self.debug_logging:
                response_log = f"\n[LLM V4 Response - {self.provider}] (no content)\nRaw completion: {completion}\n"
                print(response_log)
                if self.logger:
                    self.logger.info(response_log)
            if completion.choices and hasattr(completion.choices[0], 'finish_reason') and completion.choices[0].finish_reason == 'content_filter':
                raise ContentFilterError("OpenAI API: finish_reason=content_filter (no content)")
            return None

    def _request_translation_sync(self, prompt: str, model_name: str = None) -> Optional[TranslationResponse]:
        """Synchronous version for thread-safe parallel processing"""
        if not model_name:
            model_name = self.override_model or self.model
            
        if ": " in model_name:
            model_name = model_name.split(": ", 1)[1]

        if self.provider == "Google" and not self.endpoint:
            return self._request_translation_google_rest_sync(prompt, model_name)

        if self.provider == "Vertex AI":
            return self._request_translation_vertex_rest_sync(prompt, model_name)

        current_api_key = "lm-studio" if self.provider == "LLM Studio" else "llama-cpp"
        if self.provider not in ("LLM Studio", "llama.cpp"):
            current_api_key = self._select_api_key()
            if not current_api_key:
                raise ConnectionError("No available API key found.")

        if self.provider == "LLM Studio" and not self.endpoint:
            raise ValueError("Endpoint must be specified when using the LLM Studio provider.")

        if not self._initialize_client(current_api_key):
            raise ConnectionError("Failed to initialize API client.")

        messages = [
            {"role": "user", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        api_args = {
            "model": model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        if self.input_format == "JSON" and self.provider in ["OpenAI", "Grok", "Google", "OpenRouter"]:
            api_args["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "translation_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "translations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "integer"},
                                        "text": {"type": "string"}
                                    },
                                    "required": ["id", "text"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["translations"],
                        "additionalProperties": False
                    }
                }
            }
        elif self.provider == "LLM Studio":
            api_args["response_format"] = {
                "type": "json_schema",
                "json_schema": {"schema": TranslationResponse.model_json_schema()},
            }
        elif self.provider == "llama.cpp":
            api_args["response_format"] = {"type": "json_object"}
        elif self.provider in ["OpenAI", "Grok", "Google", "OpenRouter"]:
            api_args["response_format"] = {"type": "json_object"}

        if self.provider == "OpenAI":
            api_args["frequency_penalty"] = self.frequency_penalty
            api_args["presence_penalty"] = self.presence_penalty

        request_log = f"\n[LLM V4 Sync Request - {self.provider}]\n{json.dumps(api_args, indent=2, ensure_ascii=False)}\n"

        if self.debug_logging:
            print(request_log)
            if self.logger:
                self.logger.info(request_log)

        try:
            # Use synchronous client
            completion = self.client.chat.completions.create(**api_args)
        except Exception as e:
            err_str = str(e).lower()
            if any(kw in err_str for kw in ["content_policy_violation", "content management policy", "prohibited content", "content filter", "safety system"]):
                raise ContentFilterError(f"OpenAI API 검열 차단: {e}") from e
            if self.logger:
                self.logger.error(request_log)
                self.logger.error(f"API request failed: {e}")
            raise

        if hasattr(completion, "usage") and completion.usage:
            self.token_count += completion.usage.total_tokens
            self.token_count_last = completion.usage.total_tokens
        else:
            self.token_count_last = 0

        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            if hasattr(completion.choices[0], 'finish_reason') and completion.choices[0].finish_reason == 'content_filter':
                raise ContentFilterError("OpenAI API: finish_reason=content_filter")
            raw_content = completion.choices[0].message.content
            if self.debug_logging:
                response_log = f"\n[LLM V4 Sync Response - {self.provider}]\n{raw_content}\n"
                print(response_log)
                if self.logger:
                    self.logger.info(response_log)
            return self._parse_json_response(raw_content)
        else:
            if self.debug_logging:
                response_log = f"\n[LLM V4 Sync Response - {self.provider}] (no content)\nRaw completion: {completion}\n"
                print(response_log)
                if self.logger:
                    self.logger.info(response_log)
            if completion.choices and hasattr(completion.choices[0], 'finish_reason') and completion.choices[0].finish_reason == 'content_filter':
                raise ContentFilterError("OpenAI API: finish_reason=content_filter (no content)")
            return None

    def _get_vertex_project_id(self, json_path: str) -> str:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("project_id", "")

    def _get_vertex_token(self, json_path: str) -> str:
        from google.oauth2 import service_account
        import google.auth.transport.requests
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Vertex AI Service Account JSON file not found: {json_path}")
        credentials = service_account.Credentials.from_service_account_file(
            json_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
        return credentials.token

    async def _request_translation_vertex_rest(self, prompt: str, model_name: str) -> Optional[TranslationResponse]:
        apijson_path = self._select_api_key()
        if not apijson_path:
            raise ConnectionError("No available API key (JSON path) found for Vertex AI API.")

        project_id = self._get_vertex_project_id(apijson_path)
        token = self._get_vertex_token(apijson_path)
        region = self.vertex_region

        if ": " in model_name:
            model_name = model_name.split(": ", 1)[1]
        
        pure_model_name = model_name.replace("models/", "").replace("publishers/google/models/", "").lower()
        url = f"https://aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{region}/publishers/google/models/{pure_model_name}:generateContent"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        # Vertex AI generally caps output tokens at 65536 for gemini models.
        # Clamp maxOutputTokens to 65536 if the user inputs a higher value like 66535.
        safe_max_tokens = min(self.max_tokens, 65536)

        generation_config = {
            "temperature": self.temperature,
            "topP": self.top_p if self.top_p < 1.0 else None,
            "maxOutputTokens": safe_max_tokens
        }

        thinking_budget = self.thinking_budget
        thinking_level = self.thinking_level
        thinking_config = {}
        
        if thinking_level and isinstance(thinking_level, str):
            thinking_level = thinking_level.strip()
            
        if pure_model_name.startswith("gemini-3"):
            if thinking_level and thinking_level.upper() != "OFF":
                 thinking_config["thinkingLevel"] = thinking_level
            elif thinking_budget is not None and thinking_budget > 0:
                 thinking_config["thinkingBudget"] = thinking_budget
        elif pure_model_name.startswith("gemini-2"):
            if thinking_budget is not None and thinking_budget > 0:
                 thinking_config["thinkingBudget"] = thinking_budget
        else:
            if thinking_budget is not None and thinking_budget > 0:
                 thinking_config["thinkingBudget"] = thinking_budget
        
        if thinking_config:
             generation_config["thinkingConfig"] = thinking_config

        safety_threshold = self.get_param_value("safety_level") if "safety_level" in self.params else "OFF"
        safety_settings = [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": safety_threshold},
        ]

        if self.input_format == "JSON":
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {"type": "INTEGER"},
                        "text": {"type": "STRING"}
                    },
                    "required": ["id", "text"]
                }
            }

        contents = [
            {"role": "user", "parts": [{"text": self.system_prompt}]},
            {"role": "user", "parts": [{"text": prompt}]},
        ]
        if self.input_format != "JSON" and self.enable_prefill:
            contents.append({"role": "model", "parts": [{"text": 'Response type: csv\n"id","text"'}]})
            contents.append({"role": "user", "parts": [{"text": "continue"}]})

        payload = {
            "contents": contents,
            "generationConfig": generation_config,
            "safetySettings": safety_settings
        }

        proxy = self.proxy
        mounts = {}
        if proxy:
             mounts = {
                 "http://": httpx.HTTPTransport(proxy=proxy),
                 "https://": httpx.HTTPTransport(proxy=proxy),
             }
        
        async with httpx.AsyncClient(mounts=mounts, timeout=self.request_timeout) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"\n[LLM V4 Request - Vertex AI REST]\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n")
                    self.logger.error(f"Vertex AI REST connection failed: {e}")
                raise

        try:
            candidates = data.get("candidates", [])
            if not candidates:
                 prompt_feedback = data.get("promptFeedback", {})
                 block_reason = prompt_feedback.get("blockReason", "UNKNOWN")
                 safety_ratings = prompt_feedback.get("safetyRatings", [])
                 raise ContentFilterError(f"Vertex AI REST 검열 차단됨 (BlockReason: {block_reason})")

            candidate = candidates[0]
            finish_reason = candidate.get("finishReason", "UNKNOWN")
            
            content_parts = candidate.get("content", {}).get("parts", [])
            if not content_parts:
                 if finish_reason in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"):
                      raise ContentFilterError(f"Vertex AI REST 검열 차단됨 (FinishReason: {finish_reason})")
                 raise ValueError(f"Vertex AI REST: Empty content. Finish Reason: {finish_reason}")

            raw_text = self._extract_response_text(content_parts)
            if not raw_text:
                 raise ValueError("Vertex AI REST: Content parts exist but no non-thinking text found.")

            return self._parse_json_response(raw_text)
        except Exception as e:
             self.logger.error(f"Failed to parse Vertex AI REST response: {e}")
             raise

    def _request_translation_vertex_rest_sync(self, prompt: str, model_name: str) -> Optional[TranslationResponse]:
        apijson_path = self._select_api_key()
        if not apijson_path:
            raise ConnectionError("No available API key (JSON path) found for Vertex AI API.")

        project_id = self._get_vertex_project_id(apijson_path)
        token = self._get_vertex_token(apijson_path)
        region = self.vertex_region

        if ": " in model_name:
            model_name = model_name.split(": ", 1)[1]
        
        pure_model_name = model_name.replace("models/", "").replace("publishers/google/models/", "").lower()
        url = f"https://aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{region}/publishers/google/models/{pure_model_name}:generateContent"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        # Vertex AI generally caps output tokens at 65536 for gemini models.
        # Clamp maxOutputTokens to 65536 if the user inputs a higher value like 66535.
        safe_max_tokens = min(self.max_tokens, 65536)

        generation_config = {
            "temperature": self.temperature,
            "topP": self.top_p if self.top_p < 1.0 else None,
            "maxOutputTokens": safe_max_tokens
        }

        thinking_budget = self.thinking_budget
        thinking_level = self.thinking_level
        thinking_config = {}
        
        if thinking_level and isinstance(thinking_level, str):
            thinking_level = thinking_level.strip()
            
        if pure_model_name.startswith("gemini-3"):
            if thinking_level and thinking_level.upper() != "OFF":
                 thinking_config["thinkingLevel"] = thinking_level
            elif thinking_budget is not None and thinking_budget > 0:
                 thinking_config["thinkingBudget"] = thinking_budget
        elif pure_model_name.startswith("gemini-2"):
            if thinking_budget is not None and thinking_budget > 0:
                 thinking_config["thinkingBudget"] = thinking_budget
        else:
            if thinking_budget is not None and thinking_budget > 0:
                 thinking_config["thinkingBudget"] = thinking_budget
        
        if thinking_config:
             generation_config["thinkingConfig"] = thinking_config

        safety_threshold = self.get_param_value("safety_level") if "safety_level" in self.params else "OFF"
        safety_settings = [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": safety_threshold},
        ]

        if self.input_format == "JSON":
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {"type": "INTEGER"},
                        "text": {"type": "STRING"}
                    },
                    "required": ["id", "text"]
                }
            }

        contents = [
            {"role": "user", "parts": [{"text": self.system_prompt}]},
            {"role": "user", "parts": [{"text": prompt}]},
        ]
        if self.input_format != "JSON" and self.enable_prefill:
            contents.append({"role": "model", "parts": [{"text": 'Response type: csv\n"id","text"'}]})
            contents.append({"role": "user", "parts": [{"text": "continue"}]})

        payload = {
            "contents": contents,
            "generationConfig": generation_config,
            "safetySettings": safety_settings
        }

        proxy = self.proxy
        mounts = {}
        if proxy:
             mounts = {
                 "http://": httpx.HTTPTransport(proxy=proxy),
                 "https://": httpx.HTTPTransport(proxy=proxy),
             }
        
        with httpx.Client(mounts=mounts, timeout=self.request_timeout) as client:
            try:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"\n[LLM V4 Request - Vertex AI REST Sync]\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n")
                    self.logger.error(f"Vertex AI REST connection failed: {e}")
                raise

        try:
            candidates = data.get("candidates", [])
            if not candidates:
                 prompt_feedback = data.get("promptFeedback", {})
                 block_reason = prompt_feedback.get("blockReason", "UNKNOWN")
                 safety_ratings = prompt_feedback.get("safetyRatings", [])
                 raise ContentFilterError(f"Vertex AI REST 검열 차단됨 (BlockReason: {block_reason})")

            candidate = candidates[0]
            finish_reason = candidate.get("finishReason", "UNKNOWN")
            
            content_parts = candidate.get("content", {}).get("parts", [])
            if not content_parts:
                 if finish_reason in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"):
                      raise ContentFilterError(f"Vertex AI REST 검열 차단됨 (FinishReason: {finish_reason})")
                 raise ValueError(f"Vertex AI REST: Empty content. Finish Reason: {finish_reason}")

            raw_text = self._extract_response_text(content_parts)
            if not raw_text:
                 raise ValueError("Vertex AI REST: Content parts exist but no non-thinking text found.")

            return self._parse_json_response(raw_text)
        except Exception as e:
             self.logger.error(f"Failed to parse Vertex AI REST response: {e}")
             raise

    async def _request_translation_google_rest(self, prompt: str, model_name: str) -> Optional[TranslationResponse]:
        api_key = self._select_api_key()
        if not api_key:
            raise ConnectionError("No available API key found for Google REST API.")

        if ": " in model_name:
            model_name = model_name.split(": ", 1)[1]
        if not model_name.startswith("models/"):
             model_name = f"models/{model_name}"

        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}

        generation_config = {
            "temperature": self.temperature,
            "topP": self.top_p if self.top_p < 1.0 else None,
            "maxOutputTokens": self.max_tokens
        }

        thinking_budget = self.thinking_budget
        thinking_level = self.thinking_level
        thinking_config = {}
        
        # Strip whitespace and normalize
        if thinking_level and isinstance(thinking_level, str):
            thinking_level = thinking_level.strip()
        
        # Extract pure model name without "models/" prefix for checking
        pure_model_name = model_name.replace("models/", "").lower()
        
        if pure_model_name.startswith("gemini-3"):
            # Gemini 3: Use thinkingLevel (recommended). Cannot use both level and budget.
            if thinking_level and thinking_level.upper() != "OFF":
                 thinking_config["thinkingLevel"] = thinking_level
            elif thinking_budget is not None and thinking_budget > 0:
                 thinking_config["thinkingBudget"] = thinking_budget
        elif pure_model_name.startswith("gemini-2"):
            # Gemini 2.5: Use thinkingBudget only
            if thinking_budget is not None:
                 thinking_config["thinkingBudget"] = thinking_budget
        else:
            # Other models: Use budget if available
            if thinking_budget is not None:
                 thinking_config["thinkingBudget"] = thinking_budget
        
        if thinking_config:
             generation_config["thinkingConfig"] = thinking_config

        safety_threshold = self.get_param_value("safety_level") if "safety_level" in self.params else "OFF"
        safety_settings = [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": safety_threshold},
        ]

        if self.input_format == "JSON":
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {"type": "INTEGER"},
                        "text": {"type": "STRING"}
                    },
                    "required": ["id", "text"]
                }
            }

        contents = [
            {"role": "user", "parts": [{"text": self.system_prompt}]},
            {"role": "user", "parts": [{"text": prompt}]},
        ]
        if self.input_format != "JSON" and self.enable_prefill:
            contents.append({"role": "model", "parts": [{"text": 'Response type: csv\n"id","text"'}]})
            contents.append({"role": "user", "parts": [{"text": "continue"}]})

        payload = {
            "contents": contents,
            "generationConfig": generation_config,
            "safetySettings": safety_settings
        }

        # Log request (only on error) moved to exception block
        
        proxy = self.proxy
        mounts = {}
        if proxy:
             mounts = {
                 "http://": httpx.HTTPTransport(proxy=proxy),
                 "https://": httpx.HTTPTransport(proxy=proxy),
             }
        
        async with httpx.AsyncClient(mounts=mounts, timeout=self.request_timeout) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"\n[LLM V4 Request - Google REST]\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n")
                    self.logger.error(f"Google REST API connection failed: {e}")
                raise

        try:
            candidates = data.get("candidates", [])
            if not candidates:
                 prompt_feedback = data.get("promptFeedback", {})
                 block_reason = prompt_feedback.get("blockReason", "UNKNOWN")
                 safety_ratings = prompt_feedback.get("safetyRatings", [])
                 if self.logger:
                     self.logger.error(f"\n[Request]\n{json.dumps(payload, indent=2, ensure_ascii=False)}")
                     self.logger.error(f"\n[Response]\n{json.dumps(data, indent=2, ensure_ascii=False)}")
                 raise ContentFilterError(f"Google REST 검열 차단됨 (BlockReason: {block_reason})")

            candidate = candidates[0]
            finish_reason = candidate.get("finishReason", "UNKNOWN")
            
            content_parts = candidate.get("content", {}).get("parts", [])
            if not content_parts:
                 if finish_reason in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"):
                      raise ContentFilterError(f"Google REST 검열 차단됨 (FinishReason: {finish_reason})")
                 raise ValueError(f"Google REST: Empty content. Finish Reason: {finish_reason}")

            raw_text = self._extract_response_text(content_parts)
            if not raw_text:
                 raise ValueError("Google REST: Content parts exist but no non-thinking text found.")

            return self._parse_json_response(raw_text)
        except Exception as e:
             self.logger.error(f"Failed to parse Google REST API response: {e}")
             raise

    def _request_translation_google_rest_sync(self, prompt: str, model_name: str) -> Optional[TranslationResponse]:
        """Synchronous version for thread-safe parallel processing"""
        api_key = self._select_api_key()
        if not api_key:
            raise ConnectionError("No available API key found for Google REST API.")

        if ": " in model_name:
            model_name = model_name.split(": ", 1)[1]
        if not model_name.startswith("models/"):
             model_name = f"models/{model_name}"

        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}

        generation_config = {
            "temperature": self.temperature,
            "topP": self.top_p if self.top_p < 1.0 else None,
            "maxOutputTokens": self.max_tokens
        }

        thinking_budget = self.thinking_budget
        thinking_level = self.thinking_level
        thinking_config = {}
        
        # Strip whitespace and normalize
        if thinking_level and isinstance(thinking_level, str):
            thinking_level = thinking_level.strip()
        
        # Extract pure model name without "models/" prefix for checking
        pure_model_name = model_name.replace("models/", "").lower()
        
        if pure_model_name.startswith("gemini-3"):
            # Gemini 3: Use thinkingLevel (recommended). Cannot use both level and budget.
            if thinking_level and thinking_level.upper() != "OFF":
                 thinking_config["thinkingLevel"] = thinking_level
            elif thinking_budget is not None and thinking_budget > 0:
                 thinking_config["thinkingBudget"] = thinking_budget
        elif pure_model_name.startswith("gemini-2"):
            # Gemini 2.5: Use thinkingBudget only
            if thinking_budget is not None:
                 thinking_config["thinkingBudget"] = thinking_budget
        else:
            # Other models: Use budget if available
            if thinking_budget is not None:
                 thinking_config["thinkingBudget"] = thinking_budget
        
        if thinking_config:
             generation_config["thinkingConfig"] = thinking_config

        safety_threshold = self.get_param_value("safety_level") if "safety_level" in self.params else "OFF"
        safety_settings = [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": safety_threshold},
        ]

        if self.input_format == "JSON":
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {"type": "INTEGER"},
                        "text": {"type": "STRING"}
                    },
                    "required": ["id", "text"]
                }
            }

        contents = [
            {"role": "user", "parts": [{"text": self.system_prompt}]},
            {"role": "user", "parts": [{"text": prompt}]},
        ]
        if self.input_format != "JSON" and self.enable_prefill:
            contents.append({"role": "model", "parts": [{"text": 'Response type: csv\n"id","text"'}]})
            contents.append({"role": "user", "parts": [{"text": "continue"}]})

        payload = {
            "contents": contents,
            "generationConfig": generation_config,
            "safetySettings": safety_settings
        }
        
        # Log request (only on error) moved to exception block
        
        proxy = self.proxy
        mounts = {}
        if proxy:
             mounts = {
                 "http://": httpx.HTTPTransport(proxy=proxy),
                 "https://": httpx.HTTPTransport(proxy=proxy),
             }
        
        with httpx.Client(mounts=mounts, timeout=self.request_timeout) as client:
            try:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"\n[LLM V4 Request - Google REST Sync]\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n")
                    self.logger.error(f"Google REST API connection failed: {e}")
                raise

        try:
            candidates = data.get("candidates", [])
            if not candidates:
                 prompt_feedback = data.get("promptFeedback", {})
                 block_reason = prompt_feedback.get("blockReason", "UNKNOWN")
                 safety_ratings = prompt_feedback.get("safetyRatings", [])
                 if self.logger:
                     self.logger.error(f"\n[Request]\n{json.dumps(payload, indent=2, ensure_ascii=False)}")
                     self.logger.error(f"\n[Response]\n{json.dumps(data, indent=2, ensure_ascii=False)}")
                 raise ContentFilterError(f"Google REST 검열 차단됨 (BlockReason: {block_reason})")

            candidate = candidates[0]
            finish_reason = candidate.get("finishReason", "UNKNOWN")
            
            content_parts = candidate.get("content", {}).get("parts", [])
            if not content_parts:
                 if finish_reason in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"):
                      raise ContentFilterError(f"Google REST 검열 차단됨 (FinishReason: {finish_reason})")
                 raise ValueError(f"Google REST: Empty content. Finish Reason: {finish_reason}")

            raw_text = self._extract_response_text(content_parts)
            if not raw_text:
                 raise ValueError("Google REST: Content parts exist but no non-thinking text found.")

            return self._parse_json_response(raw_text)
        except Exception as e:
             self.logger.error(f"Failed to parse Google REST API response: {e}")
             raise

    @staticmethod
    def _sanitize_translation_text(text: str) -> str:
        """번역 텍스트에서 '원문 -> 번역문' 패턴의 원문 부분을 제거합니다."""
        if not text:
            return text if isinstance(text, str) else ""
        # list인 경우 문자열로 합치기
        if isinstance(text, list):
            text = ' '.join(str(t) for t in text if t)
        if not isinstance(text, str):
            return str(text)
        # 패턴: "원문 -> 번역문" 또는 "원문 → 번역문"
        arrow_match = re.search(r'\s*(?:->|→)\s*', text)
        if arrow_match:
            after_arrow = text[arrow_match.end():].strip()
            if after_arrow:
                # 따옴표로 감싸진 경우 제거
                if (after_arrow.startswith('"') and after_arrow.endswith('"')) or \
                   (after_arrow.startswith("'") and after_arrow.endswith("'")):
                    after_arrow = after_arrow[1:-1]
                return after_arrow
        return text

    def _parse_json_response(self, raw_content: str) -> Optional[TranslationResponse]:
        json_to_parse = raw_content.strip()
        
        # 1. Quick Refusal Check
        refusal_keywords = ["I cannot translate", "unable to translate", "cannot provide", "against my policies"]
        if any(kw in json_to_parse for kw in refusal_keywords) and len(json_to_parse) < 200:
             if self.logger:
                 self.logger.error(f"Model refused to translate. Raw response:\n{raw_content}")
             raise ContentFilterError(f"Model refused to translate: {json_to_parse}")

        # 2. CSV parsing (for Google REST API) - V111 방식 사용
        # Check if response looks like CSV
        csv_header_quoted = '"id","text"'
        csv_header_unquoted = 'id,text'
        
        csv_start_pos = json_to_parse.find(csv_header_quoted)
        has_header = False
        if csv_start_pos != -1:
            has_header = True
        else:
            csv_start_pos = json_to_parse.find(csv_header_unquoted)
            if csv_start_pos != -1:
                has_header = True
        
        # 헤더가 있는 경우
        if csv_start_pos != -1 and has_header:
            if self.logger:
                self.logger.debug(f"CSV 구조 감지 (헤더 있음)")
            
            csv_content = json_to_parse[csv_start_pos:]
            
            try:
                f = io.StringIO(csv_content)
                # Use skipinitialspace=True to handle spaces after commas
                reader = csv.DictReader(f, quoting=csv.QUOTE_ALL, skipinitialspace=True)
                
                extracted_items = []
                for row in reader:
                    try:
                        # More lenient ID processing
                        id_str = row.get('id', '0').strip().lstrip('0')
                        item_id = int(id_str) if id_str else 0
                        text = self._sanitize_translation_text(row.get('text', ''))
                        if item_id > 0:
                            extracted_items.append({"id": item_id, "text": text})
                    except (ValueError, KeyError) as e:
                        if self.logger:
                            self.logger.warning(f"CSV row parsing error: {e}")
                        continue
                
                if extracted_items:
                    if self.logger:
                        self.logger.debug(f"CSV 파싱 성공: {len(extracted_items)}개 항목")
                    return TranslationResponse.model_validate(extracted_items)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"CSV (헤더 있음) 파싱 중 예외: {e}")
        
        # 헤더 없이 CSV 데이터만 있는 경우 감지 (ID가 숫자로 시작하거나 "로 감싸진 경우)
        if re.match(r'^\s*["\d]', json_to_parse):
            if self.logger:
                self.logger.debug("헤더 없는 CSV 데이터 감지 시도")
            
            try:
                f = io.StringIO(json_to_parse)
                # Use skipinitialspace=True to handle spaces after commas
                reader = csv.reader(f, skipinitialspace=True)
                
                extracted_items = []
                for row in reader:
                    # Handle rows that might contain multiple ID-Text pairs
                    current_idx = 0
                    while current_idx + 1 < len(row):
                        try:
                            id_str = str(row[current_idx]).strip().lstrip('0')
                            if id_str.isdigit():
                                idx = int(id_str)
                                if idx > 0:
                                    text = self._sanitize_translation_text(row[current_idx + 1])
                                    extracted_items.append({"id": idx, "text": text})
                                current_idx += 2
                            else:
                                current_idx += 1
                        except:
                            current_idx += 1
                
                if extracted_items:
                    if self.logger:
                        self.logger.debug(f"헤더 없는 CSV 파싱 성공: {len(extracted_items)}개 항목")
                    return TranslationResponse.model_validate(extracted_items)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"헤더 없는 CSV 파싱 실패: {e}")

        # 3. Extract JSON block (handles both array and object)
        match = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", json_to_parse, re.DOTALL)
        if match:
            json_to_parse = match.group(1)
        else:
            # Try array first, then object
            start_array = json_to_parse.find("[")
            end_array = json_to_parse.rfind("]")
            start_obj = json_to_parse.find("{")
            end_obj = json_to_parse.rfind("}")
            
            if start_array != -1 and end_array != -1 and (start_obj == -1 or start_array < start_obj):
                json_to_parse = json_to_parse[start_array : end_array + 1]
            elif start_obj != -1 and end_obj != -1:
                json_to_parse = json_to_parse[start_obj : end_obj + 1]

        # 4. Try Standard Parsing
        try:
            try:
                data_to_validate = json.loads(json_to_parse)
            except json.JSONDecodeError:
                data_to_validate, _ = json.JSONDecoder().raw_decode(json_to_parse)

            # If data is already array, use directly; if object with translations key, extract it
            if isinstance(data_to_validate, list):
                for item in data_to_validate:
                    if isinstance(item, dict) and 'text' in item:
                        item['text'] = self._sanitize_translation_text(item['text'])
                validated_response = TranslationResponse.model_validate(data_to_validate)
            elif isinstance(data_to_validate, dict) and "translations" in data_to_validate:
                for item in data_to_validate["translations"]:
                    if isinstance(item, dict) and 'text' in item:
                        item['text'] = self._sanitize_translation_text(item['text'])
                validated_response = TranslationResponse.model_validate(data_to_validate["translations"])
            else:
                raise ValueError("Invalid JSON structure")
            
            return validated_response
        except (ValidationError, json.JSONDecodeError) as e:
            if self.logger:
                self.logger.warning(f"Standard JSON parsing failed: {e}")
            pass # Proceed to Fallback

        # 5. Fallback: Aggressive Regex Extraction
        extracted_items = []
        
        # Method A: Standard JSON-like objects
        item_pattern = re.compile(r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"text"\s*:\s*"(.*?)"\s*\}', re.DOTALL)
        for match in item_pattern.finditer(json_to_parse):
            try:
                raw_trans = match.group(2)
                trans = raw_trans.replace(r'\"', '"').replace(r'\\', '\\').replace(r'\/', '/')
                trans = self._sanitize_translation_text(trans)
                extracted_items.append({"id": int(match.group(1)), "text": trans})
            except:
                continue

        # Method B: Robust Regex Engine (Always run to catch what other methods missed)
        # Use a non-backtracking approach: Find all potential ID markers, then slice.
        try:
            # Find all "ID," patterns (Start of string or after a separator)
            id_matches = list(re.finditer(r'(?:^|[,\uff0c\n\r])\s*"?(\d+)"?\s*[,\uff0c]', json_to_parse))
            for j, match in enumerate(id_matches):
                try:
                    idx_reg = int(match.group(1).lstrip('0') or '0')
                    start_pos = match.end()
                    # Next ID starts at the beginning of the next match
                    end_pos = id_matches[j+1].start() if j+1 < len(id_matches) else len(json_to_parse)
                    val_reg = json_to_parse[start_pos:end_pos].strip()
                    
                    # Clean up trailing separators
                    val_reg = re.sub(r'[,\uff0c\s]+$', '', val_reg).strip()
                    
                    # Unquote if wrapped
                    if (val_reg.startswith('"') and val_reg.endswith('"')) or \
                       (val_reg.startswith("'") and val_reg.endswith("'")):
                        val_reg = val_reg[1:-1].strip()
                    
                    # Only add if text is non-empty and ID not already found with text
                    already_found = False
                    for item in extracted_items:
                        if item['id'] == idx_reg:
                            if not item['text'] and val_reg:
                                item['text'] = val_reg
                            already_found = True
                            break
                    
                    if not already_found and idx_reg > 0:
                        extracted_items.append({"id": idx_reg, "text": self._sanitize_translation_text(val_reg)})
                except: continue
        except Exception as reg_err:
            if self.logger: self.logger.warning(f"Robust Regex parse failed: {reg_err}")

        if extracted_items:
            try:
                return TranslationResponse.model_validate(extracted_items)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Parsing failed even after reconstruction. Raw response:\n{raw_content}")
                raise ValueError(f"Failed to parse and reconstruction failed: {e}")
        
        if self.logger:
            self.logger.error(f"All parsing methods failed. Raw response:\n{raw_content}")
        raise ValueError("Failed to parse JSON response.")

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in ["proxy", "multiple_keys", "apikey", "provider", "endpoint"]:
            self.client = None
        if param_key == "autolayout_detail_logging":
            global _V4_AUTOLAYOUT_DETAIL_LOGGING
            _V4_AUTOLAYOUT_DETAIL_LOGGING = self.autolayout_detail_logging

# -------------------------------------------------------------------------
# Monkey Patch Implementation
# -------------------------------------------------------------------------

def _v4_headless_save_entry(translate_thread, proj=None, wait_for_pipeline=False):
    """
    V4: TRUE HEADLESS BACKGROUND SAVING (Parallel & Optimized).
    Executes entirely in background thread, updates UI via Signals.
    """
    global _HEADLESS_SAVE_IN_PROGRESS, _V4_LAYOUT_START_TIME, _V4_LAYOUT_COMPLETED_COUNT
    
    import time
    
    translate_thread._v4_layout_save_error = None
    if not hasattr(translate_thread, '_v4_layout_timing_lock'):
        translate_thread._v4_layout_timing_lock = threading.Lock()
    if not hasattr(translate_thread, '_v4_layout_timings'):
        translate_thread._v4_layout_timings = {}

    # Standalone re-layout suppresses other saves. During a pipeline, saves use
    # the same project lock as layout mutations.
    _HEADLESS_SAVE_IN_PROGRESS = True
    
    # Initialize layout timer for headless save
    _V4_LAYOUT_START_TIME = time.time()
    _V4_LAYOUT_COMPLETED_COUNT = 0
    
    print("DEBUG: _v4_headless_save_entry called!")
    if LOGGER: LOGGER.info("🏁 V4: Starting Headless Background Save Sequence...")
    
    try:
        from qtpy.QtWidgets import QApplication
        from qtpy.QtGui import QImage, QPainter, QFont, QColor, QPen
        import os
        
        # 1. Setup Signaler for UI Updates
        # Since this runs in a background thread, we need signals to update the UI safely.
        # We attach to translate_thread to prevent garbage collection before signals are processed.
        translate_thread._save_signaler = SaveSignaler()
        signaler = translate_thread._save_signaler
        
        app = QApplication.instance()
        mainwindow = None
        for widget in app.topLevelWidgets():
            if widget.__class__.__name__ == 'MainWindow':
                mainwindow = widget
                break
        
        # 2. Get Project Data
        if not proj:
            proj = getattr(translate_thread, 'imgtrans_proj', None)
            if not proj and hasattr(translate_thread, 'parent'):
                 proj = getattr(translate_thread.parent, 'imgtrans_proj', None)

        if not proj:
            if LOGGER: LOGGER.error("❌ V4 Save: Project not found.")
            return

        if not hasattr(proj, '_v4_save_lock'):
            proj._v4_save_lock = threading.RLock()

        # --- DEBOUNCE LOGIC (Prevent Double Saves) ---
        import time
        current_time = time.time()
        last_save = getattr(proj, '_last_v4_save_timestamp', 0)
        # Skip if saved less than 5 seconds ago
        if not wait_for_pipeline and current_time - last_save < 5.0:
            if LOGGER: LOGGER.info(f"Skipping redundant save request (Debounce: {current_time - last_save:.2f}s ago)")
            return
        proj._last_v4_save_timestamp = current_time
        # ---------------------------------------------

        output_dir = proj.result_dir()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img_keys_list = [k for k in proj.pages.keys() if not proj._image_info.get(k, {}).get('corrupted', False)]

        parent_obj = translate_thread.parent() if callable(getattr(translate_thread, 'parent', None)) else None
        selected_pages = getattr(parent_obj, 'pages_to_process', None)
        if selected_pages:
            selected_set = set(selected_pages)
            img_keys_list = [k for k in img_keys_list if k in selected_set]

        # [Repair Mode] Only process pages that were repaired
        if _V4_REPAIR_MODE and _V4_REPAIR_PAGES is not None:
            img_keys_list = [k for k in img_keys_list if k in _V4_REPAIR_PAGES]
            if LOGGER: LOGGER.info(f"Repair mode save: {len(img_keys_list)}/{len(proj.pages)} pages selected for rendering")

        total_pages = len(img_keys_list)

        if LOGGER: LOGGER.info(f"Target: {total_pages} pages. Output: {output_dir}")

        # Pass 'proj' to UIHelper so it can load data on Main Thread
        ui_helper = None
        if mainwindow:
            msgbox = None
            if hasattr(mainwindow, 'imgtrans_progress_msgbox'):
                msgbox = mainwindow.imgtrans_progress_msgbox
            elif hasattr(mainwindow, 'module_manager'):
                msgbox = getattr(mainwindow.module_manager, 'progress_msgbox', None)
            
            if msgbox:
                # Create helper and move to main thread
                # Attach to thread to prevent GC
                _pipeline_start = getattr(translate_thread, '_v4_pipeline_start_time', None)
                translate_thread._ui_helper = UIHelper(msgbox, proj, getattr(mainwindow, 'st_manager', None) if mainwindow else None, total_pages=total_pages, pipeline_start_time=_pipeline_start)
                ui_helper = translate_thread._ui_helper
                
                ui_helper.moveToThread(mainwindow.thread())
                
                signaler.progress_signal.connect(ui_helper.update_ui)
                signaler.finished_signal.connect(ui_helper.finish_ui)
        else:
            # Fallback if no UI (shouldn't happen)
            _pipeline_start = getattr(translate_thread, '_v4_pipeline_start_time', None)
            ui_helper = UIHelper(None, proj, total_pages=total_pages, pipeline_start_time=_pipeline_start)

        if ui_helper is None:
            _pipeline_start = getattr(translate_thread, '_v4_pipeline_start_time', None)
            ui_helper = UIHelper(
                None,
                proj,
                getattr(mainwindow, 'st_manager', None) if mainwindow else None,
                total_pages=total_pages,
                pipeline_start_time=_pipeline_start,
            )
            if mainwindow:
                ui_helper.moveToThread(mainwindow.thread())
            translate_thread._ui_helper = ui_helper
        
        # 3. Headless Render Function (Background Thread - Stable)
        def process_page_hybrid(page_key):
            prepared_stored = False
            try:
                if not _claim_v4_layout_page(translate_thread, page_key):
                    if LOGGER:
                        LOGGER.warning(f"Duplicate layout request ignored for {page_key}.")
                    return True

                # [손상된 이미지 스킵] corrupted 플래그 설정된 페이지는 렌더링/저장 즉시 스킵
                if proj and proj._image_info.get(page_key, {}).get('corrupted', False):
                    if LOGGER: LOGGER.warning(f"Pre-flight check skipped corrupted image '{page_key}'. Skipping save.")
                    _set_v4_layout_state(translate_thread, page_key, 'failed')
                    return False

                # [페이지별 인페인트 대기] 이 페이지의 인페인트가 완료될 때까지 기다림.
                # 전체 인페인트 완료를 기다리는 대신 페이지 단위로 대기하여
                # 번역이 끝나는 즉시 레이아웃을 시작할 수 있도록 함.
                if not wait_for_pipeline and pcfg and pcfg.module and pcfg.module.enable_inpaint and RunStatus is not None:
                    _inpaint_max_wait = 3600  # 1시간 타임아웃
                    _inpaint_wait_start = time.time()
                    _inpaint_last_log = _inpaint_wait_start
                    while True:
                        try:
                            _info = translate_thread.imgtrans_proj._image_info.get(page_key, {})
                            if _info.get('corrupted', False):
                                if LOGGER: LOGGER.warning(f"Skipping inpaint wait for corrupted image: {page_key}")
                                return False
                            _finish_code = _info.get('finish_code', 0)
                            if _finish_code & RunStatus.FIN_INPAINT:
                                break
                        except Exception:
                            break
                        _now = time.time()
                        if _now - _inpaint_wait_start > _inpaint_max_wait:
                            if LOGGER: LOGGER.warning(f"⚠️ 인페인트 대기 타임아웃: {page_key}, 그냥 진행")
                            break
                        if _now - _inpaint_last_log >= 10:
                            _waited = int(_now - _inpaint_wait_start)
                            if LOGGER: LOGGER.info(f"⏳ 인페인트 대기 중: {page_key} ({_waited}초 경과)")
                            _inpaint_last_log = _now
                        time.sleep(0.5)

                wait_started = time.time()
                last_wait_log = wait_started
                while True:
                    readiness = _layout_page_readiness(
                        translate_thread,
                        page_key,
                        wait_for_pipeline,
                    )
                    if readiness == 'ready':
                        break
                    if readiness in ('failed', 'cancelled'):
                        _set_v4_layout_state(translate_thread, page_key, readiness)
                        if LOGGER and readiness == 'failed':
                            LOGGER.error(f"Layout prerequisites failed for {page_key}; page was not saved.")
                        return False
                    now = time.time()
                    if now - wait_started > 3600:
                        _set_v4_layout_state(translate_thread, page_key, 'failed')
                        if LOGGER:
                            LOGGER.error(f"Layout prerequisite timeout for {page_key}; page was not saved.")
                        return False
                    if now - last_wait_log >= 10:
                        if LOGGER:
                            LOGGER.info(
                                f"Layout waiting for OCR/translation/inpaint: {page_key} "
                                f"({int(now - wait_started)}s)"
                            )
                        last_wait_log = now
                    time.sleep(0.2)

                _set_v4_layout_state(translate_thread, page_key, 'preparing')
                prepared_page = _prepare_layout_page(
                    proj,
                    page_key,
                    bool(ui_helper.stm and pcfg.let_autolayout_flag),
                )
                with ui_helper.prepared_pages_lock:
                    ui_helper.prepared_pages[page_key] = prepared_page
                prepared_stored = True
                _add_v4_layout_timing(
                    translate_thread,
                    'area_prepare',
                    prepared_page['prepare_seconds'],
                )
                _autolayout_detail_log(
                    'debug',
                    f"[Layout/prepare] {page_key} "
                    f"({prepared_page['prepare_seconds']:.3f}s)",
                )

                # Use invokeMethod to run render_page_task on Main Thread (Blocking)
                # This ensures we use the exact same rendering logic (TextBlkItem) as the GUI
                from qtpy.QtCore import QMetaObject, Q_ARG, Qt

                if not ui_helper:
                    if LOGGER: LOGGER.error("UIHelper is None")
                    return False

                # Call render_page_task(page_key) on the main thread
                # We do not use Q_RETURN_ARG as it causes issues in some PyQt6 versions.
                # Instead, we rely on the thread-safe dictionary in UIHelper.
                global _V4_SAVE_HEARTBEAT, _V4_SAVE_HEARTBEAT_INFO
                _V4_SAVE_HEARTBEAT = time.time()
                _V4_SAVE_HEARTBEAT_INFO = f"invoke->render:{page_key}"
                _invoke_t0 = time.time()
                _set_v4_layout_state(translate_thread, page_key, 'rendering')
                _autolayout_detail_log('debug', f"[Save/invoke->] {page_key}")
                QMetaObject.invokeMethod(
                    ui_helper,
                    "render_page_task",
                    Qt.BlockingQueuedConnection,
                    Q_ARG(str, page_key)
                )
                prepared_stored = False
                _V4_SAVE_HEARTBEAT = time.time()
                _invoke_elapsed = _V4_SAVE_HEARTBEAT - _invoke_t0
                _render_elapsed = ui_helper.render_timings.pop(page_key, _invoke_elapsed)
                _add_v4_layout_timing(translate_thread, 'qt_render', _render_elapsed)
                if LOGGER:
                    if _invoke_elapsed > 30:
                        LOGGER.warning(
                            f"[Save/invoke] Slow render: {page_key} took {_invoke_elapsed:.1f}s"
                        )
                _autolayout_detail_log(
                    'debug',
                    f"[Save/invoke/ok] {page_key} ({_invoke_elapsed:.2f}s)",
                )

                # Retrieve result
                result_image = ui_helper.rendered_images.pop(page_key, None)

                if result_image is None or result_image.isNull():
                    if LOGGER: LOGGER.warning(f"Main thread returned null image for {page_key}")
                    _set_v4_layout_state(translate_thread, page_key, 'failed')
                    return False

                # 2. Save File (Background Thread - Slow I/O)
                _set_v4_layout_state(translate_thread, page_key, 'saving')
                ext = getattr(pcfg, 'imgsave_ext', '.png')
                if not ext.startswith('.'): ext = '.' + ext
                
                save_path = os.path.join(output_dir, os.path.splitext(page_key)[0] + ext)
                
                quality = -1
                if hasattr(pcfg, 'imgsave_quality') and pcfg.imgsave_quality is not None:
                    quality = int(pcfg.imgsave_quality)
                
                save_started = time.perf_counter()
                if not result_image.save(save_path, quality=quality):
                    raise RuntimeError(f"QImage save failed: {save_path}")
                _add_v4_layout_timing(
                    translate_thread,
                    'image_save',
                    time.perf_counter() - save_started,
                )
        
                # [메모리 강화] sip.delete로 Qt C++ 메모리 즉시 해제
                try:
                    import sip
                    sip.delete(result_image)
                except Exception:
                    del result_image
                
                _set_v4_layout_state(translate_thread, page_key, 'completed')
                return True
                
            except Exception as e:
                _set_v4_layout_state(translate_thread, page_key, 'failed')
                if LOGGER: LOGGER.error(f"Hybrid Save failed for {page_key}: {e}")
                import traceback
                if LOGGER: LOGGER.error(traceback.format_exc())
                return False
            finally:
                if prepared_stored and ui_helper is not None:
                    with ui_helper.prepared_pages_lock:
                        ui_helper.prepared_pages.pop(page_key, None)

        # 4. Execute Parallel Rendering
        signaler.progress_signal.emit(0, " (저장 시작...)")
        
        # Max workers: Get from settings or default to 2
        max_workers = 2
        if hasattr(translate_thread, 'translator') and hasattr(translate_thread.translator, 'concurrent_saves'):
             max_workers = translate_thread.translator.concurrent_saves
             
        # Limit max_workers to prevent RAM explosion with large images
        if max_workers > 4: max_workers = 4
        
        if LOGGER: LOGGER.info(f"V4 Save: Using {max_workers} concurrent save threads.")

        # [Watchdog] Diagnostic-only: if no save progress for a long time, dump all
        # thread stacks to the log. Never kills the process, never skips pages.
        # The user has chosen "leave it stuck" — we only want evidence of where
        # the main thread was when the freeze started.
        global _V4_SAVE_HEARTBEAT, _V4_SAVE_HEARTBEAT_INFO, _V4_WATCHDOG_STOP
        _V4_SAVE_HEARTBEAT = time.time()
        _V4_SAVE_HEARTBEAT_INFO = "save-start"
        _V4_WATCHDOG_STOP = threading.Event()

        def _save_watchdog(stop_event):
            STALL_WARN_SEC = 60
            STALL_RENOTIFY_SEC = 300
            _last_dump_at = 0.0
            _second_dump_done = False
            while not stop_event.wait(30.0):
                try:
                    now = time.time()
                    stalled = now - _V4_SAVE_HEARTBEAT if _V4_SAVE_HEARTBEAT else 0
                    if stalled >= STALL_WARN_SEC and now - _last_dump_at >= STALL_WARN_SEC:
                        import io as _io
                        buf = _io.StringIO()
                        try:
                            faulthandler.dump_traceback(file=buf, all_threads=True)
                        except Exception as _fh_err:
                            buf.write(f"(faulthandler.dump_traceback failed: {_fh_err})\n")
                        dump = buf.getvalue()
                        if LOGGER:
                            LOGGER.error(
                                f"[Watchdog] No save progress for {stalled:.0f}s "
                                f"(last={_V4_SAVE_HEARTBEAT_INFO!r}). Dumping all thread stacks:\n{dump}"
                            )
                        else:
                            print(f"[Watchdog] Stalled {stalled:.0f}s\n{dump}")
                        _last_dump_at = now
                        # Second dump after the stall persists past STALL_RENOTIFY_SEC,
                        # then stop spamming.
                        if stalled >= STALL_RENOTIFY_SEC and _second_dump_done:
                            # Already dumped twice — keep watching but don't log again.
                            _last_dump_at = now + 10 ** 9
                        elif stalled >= STALL_RENOTIFY_SEC:
                            _second_dump_done = True
                except Exception as _wd_err:
                    if LOGGER:
                        LOGGER.error(f"[Watchdog] internal error: {_wd_err}")

        _watchdog_thread = threading.Thread(
            target=_save_watchdog,
            args=(_V4_WATCHDOG_STOP,),
            name="V4SaveWatchdog",
            daemon=True,
        )
        _watchdog_thread.start()

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        failed_pages = []
        try:
            futures = {executor.submit(process_page_hybrid, key): key for key in img_keys_list}
            
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                # Small delay to keep UI responsive
                time.sleep(0.02)
                
                try:
                    if not future.result():
                        failed_pages.append(futures[future])
                except Exception as e:
                    failed_pages.append(futures[future])
                    if LOGGER: LOGGER.error(f"Background save task failed: {e}")
                
                completed += 1
                percent = int((completed / total_pages) * 100)
                
                # [메모리 강화] 5페이지마다 GC + 적응형 CUDA 정리
                if completed % 20 == 0:
                    import gc
                    gc.collect()
                    try:
                        import psutil
                        mem = psutil.virtual_memory()
                        if mem.percent > 75:
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception: pass
                            gc.collect()
                    except Exception: pass
                
                if completed == total_pages:
                    signaler.progress_signal.emit(100, " (저장 완료!)")
                else:
                    signaler.progress_signal.emit(percent, f" (저장 중: {completed}/{total_pages})")
        finally:
            executor.shutdown(wait=True)
            # Stop the watchdog thread; daemon, so it will also die with the process.
            try:
                if _V4_WATCHDOG_STOP is not None:
                    _V4_WATCHDOG_STOP.set()
            except Exception:
                pass

        translate_thread._v4_layout_failed_pages = set(failed_pages)
        if failed_pages and LOGGER:
            LOGGER.error(
                f"V4 layout/save failed for {len(failed_pages)} page(s): "
                f"{', '.join(failed_pages[:10])}"
            )
                    
        # 5. Log total elapsed time and finish
        _start = getattr(translate_thread, '_v4_pipeline_start_time', None)
        if _start:
            elapsed = time.time() - _start
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                elapsed_str = f"{hours}시간 {minutes}분 {seconds}초"
            elif minutes > 0:
                elapsed_str = f"{minutes}분 {seconds}초"
            else:
                elapsed_str = f"{seconds}초"
            if LOGGER:
                LOGGER.info(f"📊 총 작업시간: {elapsed_str}")

        timings = getattr(translate_thread, '_v4_layout_timings', {})
        if LOGGER:
            LOGGER.info(
                "V4 stage totals: "
                f"area_prepare={timings.get('area_prepare', 0.0):.2f}s, "
                f"qt_render={timings.get('qt_render', 0.0):.2f}s, "
                f"image_save={timings.get('image_save', 0.0):.2f}s"
            )

        translate_thread._v4_save_completed = not failed_pages and not _V4_STOP_REQUESTED
        if _V4_STOP_REQUESTED:
            signaler.progress_signal.emit(0, " (저장 중단됨)")
            if LOGGER:
                LOGGER.warning("V4 incremental layout/save stopped; pending page data was cleared.")
        elif failed_pages:
            signaler.progress_signal.emit(100, f" (저장 실패: {len(failed_pages)}페이지)")
        else:
            signaler.finished_signal.emit()
            if LOGGER:
                LOGGER.info("✅ V4: Hybrid Background Save Complete.")
        
        # Reset flag to allow normal saves again
        _HEADLESS_SAVE_IN_PROGRESS = False

    except Exception as e:
        import traceback
        translate_thread._v4_layout_save_error = e
        err = f"🚨 V4 Save Critical Error: {e}\n{traceback.format_exc()}"
        if LOGGER: LOGGER.error(err)
        print(err)
    finally:
        # Always reset flag on exit (success or failure)
        _HEADLESS_SAVE_IN_PROGRESS = False
        # Safety net: ensure the watchdog thread is signaled to stop even if we
        # bail out before the inner finally runs.
        try:
            if _V4_WATCHDOG_STOP is not None:
                _V4_WATCHDOG_STOP.set()
        except Exception:
            pass
        helper = getattr(translate_thread, '_ui_helper', None)
        if helper is not None and hasattr(helper, 'prepared_pages_lock'):
            with helper.prepared_pages_lock:
                helper.prepared_pages.clear()

def _run_translate_pipeline_patched(self):
    """
    Monkey patched version with save-throttling and global lock protection.
    """
    global _V4_GUI_LAYOUT_ENABLED, _PIPELINE_ACTIVE, _V4_LAYOUT_START_TIME, _V4_LAYOUT_COMPLETED_COUNT
    global _V4_AUTOLAYOUT_DETAIL_LOGGING
    
    # Initialize layout progress tracking for this pipeline run
    _V4_LAYOUT_START_TIME = time.time()
    _V4_LAYOUT_COMPLETED_COUNT = 0
    # Record pipeline start time for elapsed time display
    self._v4_pipeline_start_time = time.time()
    is_v4_translator = getattr(self.translator, 'use_image_batching', False)
    if not is_v4_translator:
        if hasattr(self, '_original_run_translate_pipeline'):
             return self._original_run_translate_pipeline()
        return

    # TranslateThread often uses 'num_process_pages' instead of 'num_pages'
    target_num_pages = getattr(self, 'num_process_pages', 0) or getattr(self, 'num_pages', 0)
    if target_num_pages == 0:
        return

    # [메모리 수정] 이전 실행 잔여물 강제 정리 (반복 실행 시 메모리 누적 방지)
    try:
        import gc
        # 이전 UIHelper 정리
        prev_helper = getattr(self, '_ui_helper', None)
        if prev_helper is not None:
            if hasattr(prev_helper, 'rendered_images'):
                prev_helper.rendered_images.clear()
            if hasattr(prev_helper, 'render_timings'):
                prev_helper.render_timings.clear()
            if hasattr(prev_helper, 'tray'):
                try:
                    prev_helper.tray.hide()
                    import sip
                    sip.delete(prev_helper.tray)
                except Exception:
                    pass
            try:
                import sip
                sip.delete(prev_helper)
            except Exception:
                pass
            self._ui_helper = None
        # 이전 SaveSignaler 정리
        prev_signaler = getattr(self, '_save_signaler', None)
        if prev_signaler is not None:
            try:
                import sip
                sip.delete(prev_signaler)
            except Exception:
                pass
            self._save_signaler = None
        gc.collect()
        # CUDA 캐시도 정리
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if LOGGER:
            try:
                import psutil
                rss_gb = psutil.Process().memory_info().rss / 1024**3
                LOGGER.info(f"🧹 이전 실행 잔여물 정리 완료. RSS={rss_gb:.2f} GB")
            except Exception:
                LOGGER.info("🧹 이전 실행 잔여물 정리 완료.")
    except Exception as e:
        if LOGGER: LOGGER.warning(f"이전 실행 정리 중 오류 (무시됨): {e}")

    # Initialize flags for suppression and debouncing
    _V4_GUI_LAYOUT_ENABLED = False
    _PIPELINE_ACTIVE = True
    _V4_AUTOLAYOUT_DETAIL_LOGGING = bool(
        getattr(self.translator, 'autolayout_detail_logging', False)
    )
    self._v4_layout_state_lock = threading.Lock()
    self._v4_layout_states = {}
    self._v4_layout_timing_lock = threading.Lock()
    self._v4_layout_timings = {}
    self._v4_translated_pages = set()
    self._v4_translation_failed_pages = set()
    if not hasattr(self, '_v4_inpaint_failed_pages'):
        self._v4_inpaint_failed_pages = set()
    self._v4_layout_failed_pages = set()
    self._v4_save_completed = False
    if not hasattr(self.imgtrans_proj, '_v4_save_lock'):
        self.imgtrans_proj._v4_save_lock = threading.RLock()
    layout_thread = None
    
    try:
        global _V4_STOP_REQUESTED
        _V4_STOP_REQUESTED = False
        if LOGGER:
            LOGGER.info(f"🚀 V4 Parallel Processing: {target_num_pages} pages, {self.translator.concurrent_images} workers.")

        layout_thread = threading.Thread(
            target=_v4_headless_save_entry,
            args=(self,),
            kwargs={'wait_for_pipeline': True},
            name='V4IncrementalLayoutSave',
            daemon=True,
        )
        layout_thread.start()

        max_workers = getattr(self.translator, 'concurrent_images', 3)
        save_interval = getattr(self.translator, 'save_interval', 3.0)
        initial_buffer = getattr(self.translator, 'initial_batch_buffer', 5)
        
        has_started = False
        local_completed_count = 0
        last_save_time = 0.0
        completed_pages = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            while local_completed_count < target_num_pages:
                if self.stop_requested or _V4_STOP_REQUESTED:
                    _V4_STOP_REQUESTED = True
                    self.module_thread_stopped.emit()
                    self.stop_requested = False
                    for f in futures:
                        f.cancel()
                    try:
                        executor.shutdown(wait=False, cancel_futures=True)
                    except TypeError:
                        executor.shutdown(wait=False)
                    break

                if not has_started:
                    queue_len = len(self.pipeline_pagekey_queue)
                    if queue_len >= initial_buffer or queue_len >= target_num_pages:
                        has_started = True
                        if LOGGER:
                            LOGGER.info(f"Buffer reached ({queue_len}/{initial_buffer}). Starting parallel translation.")
                    else:
                        time.sleep(0.1)
                        continue

                while len(futures) < max_workers and len(self.pipeline_pagekey_queue) > 0:
                    page_key = self.pipeline_pagekey_queue.pop(0)
                    future = executor.submit(self._translate_page, self.imgtrans_proj.pages, page_key, False)
                    futures[future] = page_key
                
                if not futures:
                    time.sleep(0.05)
                    continue

                done, _ = concurrent.futures.wait(
                    futures.keys(), timeout=0.1, return_when=concurrent.futures.FIRST_COMPLETED
                )

                for future in done:
                    page_key = futures.pop(future)
                    trans_success = True
                    try:
                        future.result()
                    except Exception as e:
                        trans_success = False
                        if LOGGER:
                            LOGGER.error(f"Translation failed for {page_key}: {e}")

                    local_completed_count += 1
                    if trans_success:
                        with _GLOBAL_SAVE_LOCK:
                            if local_completed_count < target_num_pages:
                                self.finished_counter += 1
                                self.progress_changed.emit(self.finished_counter)
                            
                            self.imgtrans_proj.update_page_progress(page_key, RunStatus.FIN_TRANSLATE)
                            with self._v4_layout_state_lock:
                                self._v4_translated_pages.add(page_key)
                            completed_pages.append(page_key)
                            
                            current_time = time.time()
                            if self.finished_counter == 1 or current_time - last_save_time >= save_interval:
                                if completed_pages:
                                    # [버그 수정] OCR 완료 확인 후 저장
                                    # 병렬 OCR 워커가 아직 blk.text를 쓰는 중인 페이지가
                                    # 빈 상태로 저장되는 경쟁 조건을 방지한다.
                                    # RunStatus.FIN_OCR 플래그는 run_ocr_step()이
                                    # update_page_progress(imgname, RunStatus.FIN_OCR)를
                                    # 호출한 직후 세팅되므로, 이 플래그가 있는 페이지는
                                    # blk.text가 이미 확정된 상태임이 보장된다.
                                    if RunStatus is not None:
                                        ocr_ready = [
                                            pk for pk in completed_pages
                                            if self.imgtrans_proj._image_info.get(pk, {}).get('finish_code', 0)
                                               & RunStatus.FIN_OCR
                                        ]
                                        not_ready = len(completed_pages) - len(ocr_ready)
                                        if not_ready > 0 and LOGGER:
                                            LOGGER.debug(
                                                f"Save deferred: {not_ready} page(s) waiting for OCR completion"
                                            )
                                        if ocr_ready:
                                            self.imgtrans_proj.save()
                                            completed_pages.clear()
                                            last_save_time = current_time
                                    else:
                                        # RunStatus를 사용할 수 없는 환경 → 기존 동작 유지
                                        self.imgtrans_proj.save()
                                        completed_pages.clear()
                                        last_save_time = current_time
                    else:
                        with self._v4_layout_state_lock:
                            self._v4_translation_failed_pages.add(page_key)
                        if local_completed_count < target_num_pages:
                            self.finished_counter += 1
                            self.progress_changed.emit(self.finished_counter)

                    # [메모리 강화] 10페이지마다 주기적 GC + 적응형 CUDA 정리
                    if local_completed_count % 10 == 0:
                        import gc
                        gc.collect()
                        try:
                            import psutil
                            mem = psutil.virtual_memory()
                            if mem.percent > 75:
                                try:
                                    import torch
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                except Exception: pass
                                gc.collect()
                            if LOGGER:
                                rss_gb = psutil.Process().memory_info().rss / 1024**3
                                LOGGER.info(f"📊 GC @{local_completed_count}/{target_num_pages} pages, RSS={rss_gb:.2f} GB, SysMem={mem.percent}%")
                        except Exception:
                            if LOGGER: LOGGER.info(f"📊 GC @{local_completed_count}/{target_num_pages} pages")

        # --- ALL PAGES TRANSLATED ---
        if LOGGER:
            LOGGER.info(
                f"Translation stage finished for {target_num_pages} pages; "
                "waiting for incremental layout/save queue."
            )

        if layout_thread is not None:
            layout_thread.join()
        _V4_GUI_LAYOUT_ENABLED = True

        layout_error = getattr(self, '_v4_layout_save_error', None)
        if layout_error is not None:
            raise RuntimeError("Incremental layout/save queue failed") from layout_error

        # One authoritative project JSON save after every layout worker has stopped.
        with _GLOBAL_SAVE_LOCK:
            self.imgtrans_proj.save(_v4_force=True)
            if not _V4_STOP_REQUESTED:
                self.finished_counter = target_num_pages
                self.progress_changed.emit(self.finished_counter)
            if LOGGER:
                LOGGER.info("Incremental layout queue drained; final project save completed.")

    except Exception as e:
        _V4_STOP_REQUESTED = True
        if LOGGER:
            LOGGER.error(f"Critical error in translation pipeline: {e}")
            LOGGER.error(traceback.format_exc())
    finally:
        # [메모리 수정] 파이프라인 완료 후 강화된 정리 루틴
        import gc

        if layout_thread is not None and layout_thread.is_alive():
            _V4_STOP_REQUESTED = True
            layout_thread.join()
        
        # 1. UIHelper 내부 Qt 객체 명시적 해제
        try:
            _helper = getattr(self, '_ui_helper', None)
            if _helper is not None:
                # rendered_images dict 비우기
                if hasattr(_helper, 'rendered_images'):
                    _helper.rendered_images.clear()
                if hasattr(_helper, 'render_timings'):
                    _helper.render_timings.clear()
                if hasattr(_helper, 'prepared_pages_lock'):
                    with _helper.prepared_pages_lock:
                        _helper.prepared_pages.clear()
                # QSystemTrayIcon 해제
                if hasattr(_helper, 'tray'):
                    try:
                        _helper.tray.hide()
                        import sip
                        sip.delete(_helper.tray)
                    except Exception:
                        pass
                    _helper.tray = None
                # stm 참조 끊기
                _helper.stm = None
                _helper.proj = None
                _helper.msgbox = None
                # UIHelper 자체 sip.delete 시도
                try:
                    import sip
                    sip.delete(_helper)
                except Exception:
                    pass
            self._ui_helper = None
        except Exception as e:
            if LOGGER: LOGGER.warning(f"UIHelper cleanup failed: {e}")
        
        # 2. SaveSignaler 해제
        try:
            _signaler = getattr(self, '_save_signaler', None)
            if _signaler is not None:
                try:
                    import sip
                    sip.delete(_signaler)
                except Exception:
                    pass
            self._save_signaler = None
        except Exception:
            pass
        
        # 3. 1차 GC (Python 객체 해제)
        gc.collect()
        
        # 4. CUDA 캐시 정리
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                if LOGGER: LOGGER.info("CUDA cache cleared after pipeline completion.")
        except Exception as e:
            if LOGGER: LOGGER.error(f"Post-pipeline CUDA cleanup failed: {e}")
        
        # 5. 2차 GC (Qt C++ destructor 후 발생하는 추가 가비지 수거)
        gc.collect()
        
        # 6. 경량 메모리 진단 리포트
        try:
            _memory_diagnostic_report(self)
        except Exception as e:
            if LOGGER: LOGGER.error(f"메모리 진단 실패: {e}")
        
        # ALWAYS restore flags on exit
        _V4_GUI_LAYOUT_ENABLED = True
        _PIPELINE_ACTIVE = False
        
    # Notification is now handled in UIHelper.finish_ui (Main Thread)
    # _show_completion_notification()

def _memory_diagnostic_report(translate_thread=None):
    """
    경량 메모리 진단 리포트.
    gc.get_objects()를 최소 1회만 순회하여 메모리 소모를 줄입니다.
    """
    import gc
    import sys
    
    if not LOGGER:
        return
    
    LOGGER.info("=" * 60)
    LOGGER.info("🔍 메모리 진단 리포트")
    LOGGER.info("=" * 60)
    
    # 1. OS 레벨 메모리
    try:
        import psutil
        mem = psutil.Process().memory_info()
        LOGGER.info(f"  RSS: {mem.rss / 1024**3:.2f} GB, VMS: {mem.vms / 1024**3:.2f} GB")
    except Exception as e:
        LOGGER.info(f"  psutil 실패: {e}")
    
    # 2. CUDA 메모리
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / 1024**3
                resv = torch.cuda.memory_reserved(i) / 1024**3
                LOGGER.info(f"  GPU {i}: alloc={alloc:.3f} GB, reserved={resv:.3f} GB")
    except Exception:
        pass
    
    # 3. 단일 순회로 타입별 집계 + QImage/numpy 카운트 (메모리 절약)
    gc.collect()
    qimage_count = 0
    qimage_bytes = 0
    np_count = 0
    np_bytes = 0
    
    try:
        from qtpy.QtGui import QImage
        import numpy as np
        
        for obj in gc.get_objects():
            try:
                if isinstance(obj, QImage):
                    qimage_count += 1
                    try:
                        qimage_bytes += obj.sizeInBytes()
                    except Exception:
                        pass
                elif isinstance(obj, np.ndarray):
                    np_count += 1
                    np_bytes += obj.nbytes
            except Exception:
                pass
        
        LOGGER.info(f"  QImage: {qimage_count}개, {qimage_bytes / 1024**2:.1f} MB")
        LOGGER.info(f"  numpy: {np_count}개, {np_bytes / 1024**3:.3f} GB")
    except Exception as e:
        LOGGER.info(f"  객체 분석 실패: {e}")
    
    # 4. 프로젝트 데이터 요약
    try:
        proj = getattr(translate_thread, 'imgtrans_proj', None) if translate_thread else None
        if proj:
            LOGGER.info(f"  프로젝트: {len(proj.pages)}페이지, {sum(len(b) for b in proj.pages.values())}블록")
    except Exception:
        pass
    
    LOGGER.info("=" * 60)

def _show_completion_notification():
    """Log completion instead of showing unsafe Windows toast"""
    if LOGGER:
        LOGGER.info("Job Finished: Translation and Save completed successfully.")
