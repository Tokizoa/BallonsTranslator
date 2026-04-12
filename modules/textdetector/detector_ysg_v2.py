import os
import os.path as osp
from typing import Tuple, List

import torch
import numpy as np
import cv2

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEVICE_SELECTOR
from utils.textblock import mit_merge_textlines, sort_regions, examine_textblk, sort_pnts
from utils.imgproc_utils import xywh2xyxypoly
from utils.proj_imgtrans import ProjImgTrans

MODEL_DIR = 'data/models'
CKPT_LIST = []

def update_ckpt_list():
    if not osp.exists(MODEL_DIR):
        return
    global CKPT_LIST
    CKPT_LIST.clear()
    for p in os.listdir(MODEL_DIR):
        if p.startswith('ysgyolo') or p.startswith('ultralyticsyolo'):
            CKPT_LIST.append(osp.join(MODEL_DIR, p).replace('\\', '/'))


update_ckpt_list()

@register_textdetectors('ysgyolo_v2')
class YSGYoloDetectorV2(TextDetectorBase):
    params = {
        'model path': {
            'type': 'selector',
            'options': CKPT_LIST,
            'value': 'data/models/ysgyolo_1.2_OS1.0.pt',
            'editable': True,
            'flush_btn': True,
            'path_selector': True,
            'path_filter': '*.pt *.ckpt *.pth *.safetensors',
            'size': 'median',
            'display_name': '모델 경로'
        },
        'merge text lines': {
            'display_name': '텍스트 라인 병합', 'type': 'checkbox', 'value': True
        },
        'confidence threshold': {
            'display_name': '신뢰도 임계값', 'type': 'line_editor', 'value': 0.3
        },
        'IoU threshold': {
            'display_name': 'IoU 임계값', 'type': 'line_editor', 'value': 0.5
        },
        'font size multiplier': {
            'display_name': '글꼴 크기 배율', 'type': 'line_editor', 'value': 1.
        },
        'font size max': {
            'display_name': '최대 글꼴 크기', 'type': 'line_editor', 'value': -1
        },
        'font size min': {
            'display_name': '최소 글꼴 크기', 'type': 'line_editor', 'value': -1
        },
        'detect size': {
            'display_name': '감지 크기', 'type': 'line_editor', 'value': 1024
        },
        'det_batch_size': {
            'type': 'selector',
            'options': [1, 2, 4, 6, 8, 12, 16],
            'value': 4,
            'display_name': '배치 크기'
        },
        'device': {
            **DEVICE_SELECTOR(),
            'display_name': '디바이스'
        },
        'label': {
            'value': {
                'balloon': True,
                'qipao': True,
                'shuqing': True,
                'changfangtiao': True,
                'hengxie': True,
                'other': True
            },
            'type': 'check_group',
            'display_name': '라벨'
        },
        'source text is vertical': {
            'display_name': '세로쓰기 텍스트', 'type': 'checkbox', 'value': True
        },
        'mask dilate size': {
            'display_name': '마스크 확장 크기', 'type': 'line_editor', 'value': 2
        }
    }

    _load_model_keys = {'model'}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        update_ckpt_list()
    
    def _load_model(self):
        model_path = self.get_param_value('model path')
        if not osp.exists(model_path):
            global CKPT_LIST
            df_model_path = model_path
            for p in CKPT_LIST:
                if osp.exists(p):
                    df_model_path = p
                    break
            self.logger.warning(f'{model_path} does not exist, try fall back to default value {df_model_path}')
            model_path = df_model_path

        if 'rtdetr' in os.path.basename(model_path):
            from ultralytics import RTDETR as MODEL
        else:
            from ultralytics import YOLO as MODEL
        if not hasattr(self, 'model') or self.model is None:
            self.model = MODEL(model_path).to(device=self.get_param_value('device'))

    def get_valid_labels(self):
        return [k for k, v in self.params['label']['value'].items() if v]

    @property
    def is_ysg(self):
        return osp.basename(self.get_param_value('model path').startswith('ysg'))

    @property
    def det_batch_size(self) -> int:
        return int(self.get_param_value('det_batch_size'))

    def _process_yolo_results(self, result, img_shape: Tuple[int, int]) -> Tuple[np.ndarray, List[dict]]:
        """Process YOLO results and extract detected items"""
        valid_labels = set(self.get_valid_labels())
        valid_ids = [idx for idx, name in result.names.items() if name in valid_labels]

        im_h, im_w = img_shape
        mask = np.zeros((im_h, im_w), dtype=np.uint8)
        
        if not valid_ids:
            return mask, []

        detected_items = []

        # Process standard boxes
        dets = result.boxes
        if dets is not None and len(dets.cls) > 0:
            for i in range(len(dets.cls)):
                cls_idx = int(dets.cls[i])
                if cls_idx in valid_ids:
                    label_name = result.names[cls_idx]

                    xyxy = dets.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy.astype(int)
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    pts = xywh2xyxypoly(np.array([[x1, y1, x2 - x1, y2 - y1]])).reshape(4, 2).tolist()
                    detected_items.append({'pts': pts, 'label': label_name})

        # Process oriented boxes
        dets = result.obb
        if dets is not None and len(dets.cls) > 0:
            for i in range(len(dets.cls)):
                cls_idx = int(dets.cls[i])
                if cls_idx in valid_ids:
                    label_name = result.names[cls_idx]
                    pts = dets.xyxyxyxy[i].cpu().numpy().astype(int)
                    cv2.fillPoly(mask, [pts], 255)
                    detected_items.append({'pts': pts.tolist(), 'label': label_name})

        return mask, detected_items

    def _create_textblocks_from_items(self, detected_items: List[dict], img_shape: Tuple[int, int]) -> List[TextBlock]:
        """Convert detected items to TextBlock list"""
        im_h, im_w = img_shape
        blk_list = []
        
        if self.get_param_value('merge text lines'):
            pts_only_list = [item['pts'] for item in detected_items]
            blk_list = mit_merge_textlines(pts_only_list, width=im_w, height=im_h)
        else:
            for item in detected_items:
                pts_sorted, is_vertical = sort_pnts(item['pts'])
                blk = TextBlock(lines=[pts_sorted], src_is_vertical=is_vertical, label=item['label'])
                blk.vertical = is_vertical
                blk.adjust_bbox()
                examine_textblk(blk, im_w, im_h)
                blk_list.append(blk)
        
        blk_list = sort_regions(blk_list)

        # Apply font size settings
        fnt_rsz = self.get_param_value('font size multiplier')
        fnt_max = self.get_param_value('font size max')
        fnt_min = self.get_param_value('font size min')
        for blk in blk_list:
            sz = blk._detected_font_size * fnt_rsz
            if fnt_max > 0:
                sz = min(fnt_max, sz)
            if fnt_min > 0:
                sz = max(fnt_min, sz)
            blk.font_size = sz
            blk._detected_font_size = sz

        return blk_list

    def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:
        """Detect text regions with batch processing support"""
        batch_size = self.det_batch_size
        
        # For batch_size=1, use original single-image processing
        if batch_size <= 1:
            return self._detect_single(img)
        
        # For batch processing, split image into patches
        im_h, im_w = img.shape[:2]
        detect_size = int(self.get_param_value('detect size'))
        
        # Check if image needs splitting (similar to CTD's rearrangement logic)
        asp_ratio = max(im_h, im_w) / min(im_h, im_w)
        down_scale_ratio = max(im_h, im_w) / detect_size
        
        # Use batch processing for extreme aspect ratios or large images
        require_batch = down_scale_ratio > 2.5 and asp_ratio > 3
        
        if not require_batch:
            return self._detect_single(img)
        
        return self._detect_batched(img, batch_size, detect_size)

    def _detect_single(self, img: np.ndarray) -> Tuple[np.ndarray, List[TextBlock]]:
        """Single image detection (original behavior)"""
        result = self.model.predict(
            source=img, save=False, show=False, verbose=False,
            conf=self.get_param_value('confidence threshold'), 
            iou=self.get_param_value('IoU threshold'),
            agnostic_nms=True
        )[0]

        mask, detected_items = self._process_yolo_results(result, img.shape[:2])
        blk_list = self._create_textblocks_from_items(detected_items, img.shape[:2])

        # Apply mask dilation
        ksize = self.get_param_value('mask dilate size')
        if ksize > 0:
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1), (ksize, ksize))
            mask = cv2.dilate(mask, element)

        return mask, blk_list

    def _detect_batched(self, img: np.ndarray, batch_size: int, detect_size: int) -> Tuple[np.ndarray, List[TextBlock]]:
        """Batched detection for large/extreme aspect ratio images"""
        im_h, im_w = img.shape[:2]
        transpose = im_h < im_w
        
        if transpose:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            im_h, im_w = im_w, im_h

        # Calculate patch parameters
        patch_h = detect_size
        patch_w = im_w
        
        num_patches = int(np.ceil(im_h / patch_h))
        step_h = int((im_h - patch_h) / (num_patches - 1)) if num_patches > 1 else 0
        
        # Create patches
        patches = []
        patch_positions = []
        for i in range(num_patches):
            t = i * step_h
            b = min(t + patch_h, im_h)
            if b - t < patch_h // 2:  # Skip too-small patches
                continue
            patch = img[t:b, :]
            patches.append(patch)
            patch_positions.append((t, b))

        # Process patches in batches
        all_detected_items = []
        all_masks = []
        
        for batch_start in range(0, len(patches), batch_size):
            batch_end = min(batch_start + batch_size, len(patches))
            batch_imgs = patches[batch_start:batch_end]
            
            # Run batch prediction
            results = self.model.predict(
                source=batch_imgs, save=False, show=False, verbose=False,
                conf=self.get_param_value('confidence threshold'),
                iou=self.get_param_value('IoU threshold'),
                agnostic_nms=True
            )
            
            # Process each result in batch
            for idx, result in enumerate(results):
                patch_idx = batch_start + idx
                t, b = patch_positions[patch_idx]
                
                patch_mask, detected_items = self._process_yolo_results(result, (b - t, im_w))
                
                # Adjust coordinates to full image space
                for item in detected_items:
                    for pt in item['pts']:
                        pt[1] += t  # Adjust y-coordinate
                
                all_detected_items.extend(detected_items)
                
                # Store mask with position info
                all_masks.append((patch_mask, t, b))

        # Merge masks
        full_mask = np.zeros((im_h, im_w), dtype=np.uint8)
        for patch_mask, t, b in all_masks:
            full_mask[t:b, :] = np.maximum(full_mask[t:b, :], patch_mask)

        # Create text blocks
        img_shape = (im_h, im_w) if not transpose else (im_w, im_h)
        blk_list = self._create_textblocks_from_items(all_detected_items, img_shape)

        # Rotate back if needed
        if transpose:
            full_mask = cv2.rotate(full_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # Rotate text block coordinates
            for blk in blk_list:
                # Transform coordinates back
                for line in blk.lines:
                    for pt in line:
                        pt[0], pt[1] = pt[1], im_h - pt[0]
                blk.adjust_bbox()

        # Apply mask dilation
        ksize = self.get_param_value('mask dilate size')
        if ksize > 0:
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1), (ksize, ksize))
            full_mask = cv2.dilate(full_mask, element)

        return full_mask, blk_list

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        
        if param_key == 'model path':
            if hasattr(self, 'model'):
                del self.model

    def flush(self, param_key: str):
        if param_key == 'model path':
            update_ckpt_list()
            global CKPT_LIST
            return CKPT_LIST
