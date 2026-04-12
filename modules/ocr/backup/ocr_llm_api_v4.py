import re
import time
import base64
import json
import csv
import io
import cv2
import numpy as np
import threading
from typing import List, Optional
import concurrent.futures

import openai
import httpx
from qtpy.QtCore import QObject, Signal, Qt

from .base import register_OCR, OCRBase, TextBlock

class SaveSignaler(QObject):
    save_signal = Signal()

class TaskRunner(QObject):
    def __init__(self, task):
        super().__init__()
        self.task = task

    def run(self):
        self.task()

@register_OCR("llm_ocr_v4")
class LLM_OCR_V4(OCRBase):
    use_page_batching = True  # Signal for async pipeline
    _patch_applied = False  # Class variable to prevent multiple patch attempts
    
    lang_map = {
        "Auto Detect": None,
        "Japanese": "ja",
        "English": "en",
        "Korean": "ko",
        "Chinese (Simplified)": "zh-CN",
        "Chinese (Traditional)": "zh-TW",
        "French": "fr",
        "German": "de",
        "Spanish": "es",
        "Russian": "ru",
        "Arabic": "ar",
        "Thai": "th",
        "Vietnamese": "vi",
    }

    popular_models = [
        "OAI: gpt-4o-mini",
        "OAI: gpt-4o",
        "OAI: gpt-4-turbo",
        "GGL: gemini-1.5-pro-latest",
        "GGL: gemini-1.5-flash-latest",
        "GGL: gemini-2.0-flash-exp",
        "GGL: gemini-2.5-flash",
        "XAI: grok-vision-beta",
    ]

    params = {
        "provider": {
            "type": "selector",
            "options": ["OpenAI", "Google", "Grok", "OpenRouter", "LLM Studio"],
            "value": "OpenAI",
            "description": "LLM 서비스 제공자를 선택합니다.",
        },
        "api_key": {
            "value": "",
            "description": "API 키를 입력하세요.",
        },
        "multiple_keys": {
            "type": "editor",
            "value": "",
            "description": "여러 개의 API 키를 세미콜론(;)으로 구분하여 입력할 수 있습니다. 요청 시 키를 순환하며 사용합니다.",
        },
        "endpoint": {
            "value": "",
            "description": "API 엔드포인트 URL입니다. 비워두면 제공자 기본값을 사용합니다.",
        },
        "model": {
            "type": "selector",
            "options": popular_models,
            "value": "OAI: gpt-4o-mini",
            "description": "OCR에 사용할 주 모델을 선택합니다.",
        },
        "override_model": {
            "value": "",
            "description": "목록에 없는 모델명을 직접 입력하여 사용할 수 있습니다.",
        },
        "fallback_model": {
            "value": "",
            "description": "모든 재시도 실패 시 사용할 폴백 모델명입니다.",
        },
        "language": {
            "type": "selector",
            "options": list(lang_map.keys()),
            "value": "Japanese",
            "description": "OCR 대상 언어입니다.",
        },
        "detail_level": {
            "type": "selector",
            "options": ["auto", "low", "high"],
            "value": "auto",
            "description": "이미지 디테일 수준을 조절합니다.",
        },
        "prompt": {
            "type": "editor",
            "value": "You are a specialized OCR engine for manga and comics. The language is **{language}**.\nTask: Perform OCR on the {num_images} images attached.\nOutput Requirement: Return ONLY a CSV with columns: id, text.\nFormatting: Consolidate text into a single line. Remove all line breaks within the text field.\nID Mapping: IDs must be 1-based integers (1 to {num_images}) corresponding to the image order.\nExample Output:\n1, Text of image 1\n2, Text of image 2\nNote: If an image has no text, return an empty string for that ID.\n**CRITICAL:** If you see jumbled characters, it is likely vertical text read horizontally. Reconstruct the correct vertical text.\nOutput ONLY the CSV data with header. No explanations.",
            "description": "OCR 작업을 위한 프롬프트입니다. {language} 및 {num_images}(이미지 수) 플레이스홀더를 반드시 포함해야 합니다.",
        },
        "max_response_tokens": {
            "value": 4096,
            "description": "응답 생성 시 사용할 최대 토큰 수입니다.",
        },
        "thinking_budget": {
            "value": 0,
            "description": "Gemini 추론(Thinking) 토큰 예산입니다. (0은 비활성)",
        },
        "thinking_level": {
            "type": "selector",
            "options": ["OFF", "minimal", "low", "medium", "high"],
            "value": "OFF",
            "description": "Gemini 3 모델의 추론 수준을 설정합니다.",
        },
        "safety_level": {
            "type": "selector",
            "options": ["OFF", "BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"],
            "value": "BLOCK_NONE",
            "description": "Gemini 모델의 안전 필터 수준입니다.",
        },
        "temperature": {
            "value": 0.1,
            "description": "샘플링 온도입니다.",
        },
        "top_p": {
            "value": 1.0,
            "description": "Top P 샘플링 설정입니다.",
        },
        "retry_attempts": {
            "value": 3,
            "description": "API 호출 실패 시 최대 재시도 횟수입니다.",
        },
        "retry_timeout": {
            "value": 5,
            "description": "재시도 사이의 대기 시간(초)입니다.",
        },
        "request_timeout": {
            "value": 120,
            "description": "HTTP 요청 타임아웃(초)입니다. Gemini는 60초 이상 권장.",
        },
        "proxy": {
            "value": "",
            "description": "프록시 주소입니다. (예: http://127.0.0.1:8080)",
        },
        "delay": {
            "value": 0.5,
            "description": "요청 사이의 전역 지연 시간(초)입니다.",
        },
        "requests_per_minute": {
            "value": 1000,
            "description": "각 API 키당 분당 최대 요청 수(RPM)입니다.",
        },
        "ocr_parallel_workers": {
            "value": 200,
            "description": "동시에 병렬 처리할 OCR 작업 수입니다. API 호출이라 VRAM 무관, RPM만 체크하세요.",
        },
        "initial_buffer_pages": {
            "value": 20,
            "description": "OCR 시작 전 초기 버퍼링할 페이지 수입니다.",
        },
        "batch_size": {
            "value": 10,
            "description": "한 번의 API 요청에 묶어서 보낼 이미지 수입니다. (기본값: 10)",
        },
        "description": "비전 LLM을 사용한 OCR with CSV mode for censorship bypass.",
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.last_request_time = 0
        self.client = None
        self.http_client = None # Reusable HTTP Client
        self.request_count_minute = 0
        self.minute_start_time = time.time()
        self.key_usage = {}
        self.current_key_index = 0
        self.fallback_ocr = None  # MangaOCR fallback instance
        self.use_page_batching = True  # Enable async pipeline batching
        
        # Apply patch immediately if safe, otherwise verify module loading
        if not LLM_OCR_V4._patch_applied:
            import sys
            if 'ui.module_manager' in sys.modules:
                LLM_OCR_V4._patch_applied = True
                _apply_chunked_processing_patch()
            else:
                # Fallback only if module_manager isn't loaded yet (unlikely)
                from qtpy.QtCore import QTimer
                QTimer.singleShot(0, _apply_chunked_processing_patch)

        # Schedule UI Patch (Main Thread Safe)
        from qtpy.QtCore import QTimer
        QTimer.singleShot(0, self._safe_install_ui_patch)

    def _safe_install_ui_patch(self):
        """Safely install UI patches (Saving Bar) from Main Thread"""
        try:
            import sys
            if 'modules.translators.trans_llm_api_v4' in sys.modules:
                mod = sys.modules['modules.translators.trans_llm_api_v4']
                if hasattr(mod, '_install_ui_patches'):
                    mod._install_ui_patches()
            else:
                import modules.translators.trans_llm_api_v4 as mod
                if hasattr(mod, '_install_ui_patches'):
                    mod._install_ui_patches()
        except Exception as e:
            if self.logger: self.logger.warning(f"Failed to install UI patches: {e}")

    def _initialize_http_client(self):
        """Initialize reusable HTTP client with high connection limits for parallel OCR"""
        if self.http_client:
            try:
                self.http_client.close()
            except:
                pass
        
        # Determine pool size based on workers * 2 (Safety margin)
        # Default 200 workers -> 400 connections
        pool_size = self.ocr_parallel_workers * 2
        if pool_size < 100: pool_size = 100
        
        limits = httpx.Limits(max_keepalive_connections=pool_size, max_connections=pool_size)
        
        mounts = {}
        if self.proxy:
            try:
                mounts = {
                    "http://": httpx.HTTPTransport(proxy=self.proxy),
                    "https://": httpx.HTTPTransport(proxy=self.proxy),
                }
            except Exception as e:
                self.logger.error(f"Failed to initialize proxy for http_client: {e}")

        self.http_client = httpx.Client(
            mounts=mounts, 
            timeout=self.request_timeout,
            limits=limits,
            http2=False # Enable HTTP/2 for Google/OpenAI
        )
        self.logger.debug(f"Initialized HTTP Client with pool_size={pool_size}, http2=False")

    def __del__(self):
        if hasattr(self, 'http_client') and self.http_client:
            try:
                self.http_client.close()
            except:
                pass

    def _initialize_client(self, api_key_to_use: str):
        endpoint = self.endpoint
        provider = self.provider
        if not endpoint:
            if provider == "OpenAI":
                endpoint = "https://api.openai.com/v1"
            elif provider == "Google":
                endpoint = "https://generativelanguage.googleapis.com/v1beta/openai"
            elif provider == "OpenRouter":
                endpoint = "https://openrouter.ai/api/v1"

        http_client = None
        if self.proxy:
            try:
                proxy_mounts = {"all://": httpx.HTTPTransport(proxy=self.proxy)}
                http_client = httpx.Client(mounts=proxy_mounts)
            except Exception as e:
                self.logger.error(f"Failed to initialize proxy '{self.proxy}': {e}.")

        masked_key = (
            api_key_to_use[:4] + "..." + api_key_to_use[-4:]
            if len(api_key_to_use) > 8
            else api_key_to_use
        )
        self.logger.debug(
            f"Initializing client for {provider} with key {masked_key} at endpoint {endpoint}"
        )

        self.client = openai.OpenAI(
            api_key=api_key_to_use, base_url=endpoint, http_client=http_client
        )

    # --- Property Getters ---
    @property
    def provider(self) -> str:
        return self.get_param_value("provider")

    @property
    def api_key(self) -> str:
        return self.get_param_value("api_key")

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
    def endpoint(self) -> Optional[str]:
        return self.get_param_value("endpoint") or None

    @property
    def model(self) -> str:
        return self.get_param_value("model")

    @property
    def override_model(self) -> Optional[str]:
        return self.get_param_value("override_model") or None

    @property
    def language(self) -> str:
        return self.get_param_value("language")

    @property
    def detail_level(self) -> str:
        return self.get_param_value("detail_level")

    @property
    def prompt(self) -> str:
        return self.get_param_value("prompt")

    @property
    def proxy(self) -> str:
        return self.get_param_value("proxy")

    @property
    def requests_per_minute(self) -> int:
        return int(self.get_param_value("requests_per_minute"))

    @property
    def max_response_tokens(self) -> int:
        return int(self.get_param_value("max_response_tokens"))

    @property
    def request_delay(self) -> float:
        try:
            return float(self.get_param_value("delay"))
        except (ValueError, TypeError):
            return 0.5

    @property
    def thinking_budget(self) -> Optional[int]:
        val = self.get_param_value("thinking_budget")
        if val == "":
            return None
        return int(val)

    @property
    def thinking_level(self) -> str:
        return self.get_param_value("thinking_level")

    @property
    def safety_level(self) -> str:
        return self.get_param_value("safety_level")

    @property
    def temperature(self) -> float:
        val = self.get_param_value("temperature")
        return float(val) if val != "" else 0.1

    @property
    def top_p(self) -> float:
        val = self.get_param_value("top_p")
        return float(val) if val != "" else 1.0

    @property
    def retry_attempts(self) -> int:
        val = self.get_param_value("retry_attempts")
        return int(val) if val != "" else 3

    @property
    def retry_timeout(self) -> int:
        val = self.get_param_value("retry_timeout")
        return int(val) if val != "" else 5

    @property
    def request_timeout(self) -> int:
        val = self.get_param_value("request_timeout")
        return int(val) if val != "" else 120

    @property
    def fallback_model(self) -> Optional[str]:
        return self.get_param_value("fallback_model") or None

    @property
    def ocr_parallel_workers(self) -> int:
        val = self.get_param_value("ocr_parallel_workers")
        return int(val) if val != "" else 200

    @property
    def initial_buffer_pages(self) -> int:
        val = self.get_param_value("initial_buffer_pages")
        return int(val) if val != "" else 20

    @property
    def batch_size(self) -> int:
        val = self.get_param_value("batch_size")
        return int(val) if val != "" else 10

    def _respect_delay(self):
        current_time = time.time()
        rpm = self.requests_per_minute
        if rpm > 0:
            if current_time - self.minute_start_time >= 60:
                self.request_count_minute = 0
                self.minute_start_time = current_time
            if self.request_count_minute >= rpm:
                wait_time = 60.1 - (current_time - self.minute_start_time)
                if wait_time > 0:
                    self.logger.warning(
                        f"Global RPM limit ({rpm}) reached. Waiting {wait_time:.2f}s."
                    )
                    time.sleep(wait_time)
                self.request_count_minute = 0
                self.minute_start_time = time.time()

        # Removed global delay enforcement for parallel processing
        # Each request proceeds immediately without waiting for others
        self.request_count_minute += 1

    def _respect_key_limit(self, key: str) -> bool:
        rpm = self.requests_per_minute
        if rpm <= 0:
            return True
        now = time.time()
        count, start_time = self.key_usage.get(key, (0, now))
        if now - start_time >= 60:
            count, start_time = 0, now
        if count >= rpm:
            wait_time = 60.1 - (now - start_time)
            if wait_time > 0:
                self.logger.warning(
                    f"RPM limit ({rpm}) for key {key[:6]}... reached. Waiting {wait_time:.2f}s."
                )
                time.sleep(wait_time)
            self.key_usage[key] = (0, time.time())
            return False
        return True

    def _select_api_key(self) -> Optional[str]:
        api_keys = self.multiple_keys_list
        single_key = self.api_key
        if not api_keys and not single_key:
            self.logger.error("No API keys provided.")
            return None

        if not api_keys:
            if self._respect_key_limit(single_key):
                now = time.time()
                count, start_time = self.key_usage.get(single_key, (0, now))
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
        self.logger.error("All API keys are rate-limited.")
        return None

    def ocr(self, img_base64: str, prompt_override: str = None) -> str:
        # 1. Primary Model Attempts
        # _ocr_with_retry handles the retry loop internally
        try:
            result = self._ocr_with_retry(img_base64, prompt_override, retry_override=self.retry_attempts)
            if result and result.strip():
                return result
            self.logger.warning(f"Primary model returned empty text after {self.retry_attempts} attempts.")
        except Exception as e:
            self.logger.warning(f"Primary model failed after {self.retry_attempts} attempts: {e}")

        # 2. Fallback Model Attempts
        if self.fallback_model:
            self.logger.info(f"Switching to Fallback Model: {self.fallback_model}")
            try:
                # Use same retry count for fallback to ensure robustness
                result = self._ocr_with_retry(img_base64, prompt_override, 
                                            model_override=self.fallback_model, 
                                            retry_override=self.retry_attempts)
                if result and result.strip() and not result.startswith("[ERROR"):
                    self.logger.info(f"Fallback model succeeded: {self.fallback_model}")
                    return result
                self.logger.warning(f"Fallback model returned empty text.")
            except Exception as e:
                self.logger.error(f"Fallback model failed: {e}")

        # 3. MangaOCR Fallback
        self.logger.info("Trying MangaOCR as final fallback...")
        manga_result = self._try_manga_ocr_fallback(img_base64)
        if manga_result:
            return manga_result
        
        self.logger.error("OCR failed after all strategies (Primary -> Fallback -> MangaOCR).")
        return "[ERROR: OCR Failed]"

    def _try_manga_ocr_fallback(self, img_base64: str) -> Optional[str]:
        """Try MangaOCR as final fallback when LLM fails"""
        try:
            # Check if fallback was previously marked as failed
            if self.fallback_ocr is False:
                return None
                
            if self.fallback_ocr is None:
                from .ocr_manga import MangaOCR
                self.logger.info("Initializing MangaOCR for fallback...")
                self.fallback_ocr = MangaOCR()
                self.fallback_ocr.load_model()
            
            # Decode base64 to image
            import io
            from PIL import Image
            img_data = base64.b64decode(img_base64)
            img_pil = Image.open(io.BytesIO(img_data))
            img_np = np.array(img_pil)
            
            result = self.fallback_ocr.ocr_img(img_np)
            if result and result.strip():
                self.logger.info(f"MangaOCR fallback successful: {result}")
                return result
            return None
        except Exception as e:
            self.logger.error(f"MangaOCR fallback failed: {e}")
            self.fallback_ocr = False  # Mark as failed to avoid retry
            return None

    def _ocr_with_retry(self, img_base64: str, prompt_override: str = None, model_override: str = None, retry_override: int = None) -> str:
        """
        Internal OCR with retry logic.
        retry_override: Override retry_attempts for this call only (used for fallback)
        """
        # Use retry_override if provided, otherwise use self.retry_attempts
        max_attempts = retry_override if retry_override is not None else self.retry_attempts

        for attempt in range(max_attempts):
            try:
                # Select API Key INSIDE the loop to rotate keys on failure
                api_key_to_use = self._select_api_key()
                if not api_key_to_use:
                    return "[ERROR: No available API key]"

                if not self.client or self.client.api_key != api_key_to_use:
                    self._initialize_client(api_key_to_use)

                self._respect_delay()

                lang_name = self.language
                # Provide num_images=1 for single mode to support unified prompt
                try:
                    raw_prompt = prompt_override or self.prompt
                    prompt_text = raw_prompt.format(language=lang_name, num_images=1, num_imgs=1)
                except KeyError as e:
                    # If format fails (e.g. prompt has other placeholders or already formatted), use raw
                    self.logger.warning(f"Prompt formatting failed (KeyError: {e}). Using raw prompt.")
                    prompt_text = raw_prompt
                except Exception as e:
                    self.logger.warning(f"Prompt formatting failed: {e}. Using raw prompt.")
                    prompt_text = raw_prompt

                model_name = model_override or self.override_model or self.model
                if ": " in model_name:
                    model_name = model_name.split(": ", 1)[1]
                
                # [LOGGING ADDITION] Explicitly log attempt and model
                self.logger.info(f"OCR Attempt {attempt + 1}/{max_attempts} using Model: {model_name}")

                # Use Google REST API if provider is Google and no custom endpoint
                if self.provider == "Google" and not self.endpoint:
                    return self._ocr_google_rest(img_base64, prompt_text, model_name)

                # OpenAI-compatible API
                image_content_part = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                }

                if self.provider in ["OpenAI", "Google", "OpenRouter", "Grok"]:
                    detail_setting = self.detail_level
                    if detail_setting in ["low", "high"]:
                        image_content_part["image_url"]["detail"] = detail_setting

                # Format prompt with language
                prompt_text = self.prompt.format(language=lang_name, num_images=1, num_imgs=1)

                # Turn 1: User (prompt text only)
                # Turn 2: User (image only)
                messages = [
                    {
                        "role": "user",
                        "content": prompt_text,
                    },
                    {
                        "role": "user",
                        "content": [image_content_part],
                    }
                ]

                self.logger.debug(f"OCR request with model: {model_name}")

                req_start = time.time()
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=self.max_response_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                elapsed = time.time() - req_start
                end_time_str = time.strftime("%H:%M:%S", time.localtime())
                self.logger.debug(f"OCR Request took {elapsed:.2f}s ({end_time_str})")

                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    full_text = response.choices[0].message.content
                    # CSV 파싱 시도
                    parsed_text = self._parse_csv_response(full_text)
                    if parsed_text:
                        self.logger.debug(f"OCR CSV result: {parsed_text}")
                        return parsed_text
                    else:
                        # CSV 파싱 실패 시 원본 반환
                        full_text = full_text.replace("\n", " ").strip()
                        self.logger.debug(f"OCR result: {full_text}")
                        return full_text
                else:
                    self.logger.warning("No text found in OCR response.")
                    return ""
                    
            except Exception as e:
                if attempt < max_attempts - 1:
                    self.logger.warning(f"OCR attempt {attempt + 1} failed: {e}. Retrying with new key...")
                    time.sleep(self.retry_timeout)
                else:
                    raise e # Let the caller handle the final exception (fallback logic)
        
        return "[ERROR: Max retries exceeded]"

    def _parse_csv_response(self, response_text: str) -> Optional[str]:
        """CSV 응답을 파싱하여 텍스트만 추출"""
        try:
            # CSV 헤더 확인
            csv_header_quoted = '"id","text"'
            csv_header_unquoted = 'id,text'
            
            csv_start_pos = response_text.find(csv_header_quoted)
            has_header = False
            if csv_start_pos != -1:
                has_header = True
            else:
                csv_start_pos = response_text.find(csv_header_unquoted)
                if csv_start_pos != -1:
                    has_header = True
            
            # 헤더가 있는 경우
            if csv_start_pos != -1 and has_header:
                csv_content = response_text[csv_start_pos:]
                f = io.StringIO(csv_content)
                reader = csv.DictReader(f, quoting=csv.QUOTE_ALL)
                
                text_parts = []
                for row in reader:
                    text = row.get('text', '')
                    if text:
                        text_parts.append(text)
                
                if text_parts:
                    result = " ".join(text_parts)
                    self.logger.debug(f"CSV 파싱 성공: {len(text_parts)}개 항목")
                    return result
            
            # 헤더 없이 CSV 데이터만 있는 경우
            if re.match(r'^\s*"', response_text):
                f = io.StringIO(response_text)
                reader = csv.reader(f, quoting=csv.QUOTE_ALL)
                
                rows = [row for row in reader if row]
                if rows:
                    first_row = rows[0]
                    # 첫 열이 6자리 숫자인지 확인
                    if first_row and first_row[0] and re.match(r'^\d{6}$', first_row[0]):
                        text_parts = []
                        for row in rows:
                            if len(row) >= 2:
                                text = row[1]
                                if text:
                                    text_parts.append(text)
                        
                        if text_parts:
                            result = " ".join(text_parts)
                            self.logger.debug(f"헤더 없는 CSV 파싱 성공: {len(text_parts)}개 항목")
                            return result
            
            return None
        except Exception as e:
            self.logger.warning(f"CSV 파싱 실패: {e}")
            return None

    def _ocr_google_rest(self, img_base64: str, prompt_text: str, model_name: str) -> str:
        """Google REST API를 사용한 OCR with CSV mode"""
        api_key = self._select_api_key()
        if not api_key:
            raise ConnectionError("No available API key found for Google REST API.")

        # Extract pure model name for checking
        pure_model_name = model_name.replace("models/", "")
        
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}

        # Get language name for prompt formatting
        lang_name = self.language

        generation_config = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_response_tokens,
        }
        
        # topP: null when value is 1.0 (like translator)
        if self.top_p != 1.0:
            generation_config["topP"] = self.top_p
        else:
            generation_config["topP"] = None

        thinking_budget = self.thinking_budget
        thinking_level = self.thinking_level
        thinking_config = {}
        
        # Use snake_case keys like translator
        if pure_model_name.lower().startswith("gemini-3"):
            # gemini-3: level 우선, 없으면 budget
            if thinking_level and thinking_level != "OFF":
                thinking_config["thinking_level"] = thinking_level
            elif thinking_budget is not None and thinking_budget > 0:
                thinking_config["thinking_budget"] = thinking_budget
        else:
            # 그 외 모델: budget만 사용
            if thinking_budget is not None and thinking_budget > 0:
                thinking_config["thinking_budget"] = thinking_budget
        
        if thinking_config:
            generation_config["thinking_config"] = thinking_config

        safety_threshold = self.safety_level if self.safety_level != "OFF" else "BLOCK_NONE"
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": safety_threshold},
        ]

        # Build contents with model priming (like translator)
        # Turn 1: User (prompt text only)
        # Turn 2: User (image)
        # Turn 3: Model (CSV header priming - censorship bypass)
        prompt_formatted = self.prompt.format(language=lang_name)
        
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt_formatted}]},
                {"role": "user", "parts": [{"inline_data": {"mime_type": "image/jpeg", "data": img_base64}}]},
                {"role": "model", "parts": [{"text": 'Response type: csv\n"id","text"'}]}
            ],
            "generationConfig": generation_config,
            "safetySettings": safety_settings
        }

        # Use reusable client
        if self.http_client is None:
            self._initialize_http_client()

        try:
            req_start = time.time()
            response = self.http_client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            elapsed = time.time() - req_start
            end_time_str = time.strftime("%H:%M:%S", time.localtime())
            self.logger.debug(f"Google REST OCR Request took {elapsed:.2f}s ({end_time_str})")
        except (httpx.RequestError, httpx.TimeoutException, ConnectionError) as e:
            self.logger.warning(f"Connection failed ({e}). Refreshing client and retrying...")
            self._initialize_http_client()
            try:
                req_start = time.time()
                response = self.http_client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            except Exception as e2:
                self.logger.error(f"Google REST API connection failed after refresh: {e2}")
                raise
        except Exception as e:
            self.logger.error(f"Google REST API connection failed: {e}")
            raise

        try:
            candidates = data.get("candidates", [])
            if not candidates:
                finish_reason = data.get("promptFeedback", {}).get("blockReason", "UNKNOWN")
                # Log full response for debugging
                self.logger.debug(f"Google REST full response: {data}")
                raise ValueError(f"Google REST: No candidates returned. Blocked Reason: {finish_reason}")

            candidate = candidates[0]
            
            # Check finish reason first
            finish_reason = candidate.get("finishReason", "")
            if finish_reason == "SAFETY":
                safety_ratings = candidate.get("safetyRatings", [])
                self.logger.warning(f"Google REST: Content blocked by safety filter. Ratings: {safety_ratings}")
                raise ValueError(f"Google REST: Content blocked by safety filter")
            
            content_parts = candidate.get("content", {}).get("parts", [])
            if not content_parts:
                self.logger.debug(f"Google REST candidate: {candidate}")
                raise ValueError(f"Google REST: Empty content parts. Finish reason: {finish_reason}")

            raw_text = content_parts[0].get("text", "")
            if not raw_text:
                self.logger.debug(f"Google REST content_parts: {content_parts}")
                raise ValueError(f"Google REST: Content parts exist but text is empty. Finish reason: {finish_reason}")

            # CSV 파싱 시도
            parsed_text = self._parse_csv_response(raw_text)
            if parsed_text:
                return parsed_text
            else:
                # CSV 파싱 실패 시 원본 반환
                return raw_text.replace("\n", " ").strip()
        except Exception as e:
            self.logger.error(f"Failed to parse Google REST API response: {e}")
            raise

    def _ocr_google_rest_batch(self, content_payload: List[dict], model_name: str) -> List[str]:
        """Google REST API를 사용한 Batch OCR with CSV mode"""
        api_key = self._select_api_key()
        if not api_key:
            raise ConnectionError("No available API key found for Google REST API Batch.")

        pure_model_name = model_name.replace("models/", "")
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}

        # Convert OpenAI-format payload to Gemini REST format
        parts = []
        for item in content_payload:
            if item["type"] == "text":
                parts.append({"text": item["text"]})
            elif item["type"] == "image_url":
                # OpenAI: data:image/jpeg;base64,.....
                data_url = item["image_url"]["url"]
                base64_data = data_url.split(",")[1]
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": base64_data}})

        # Config & Safety
        generation_config = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_response_tokens,
        }
        if self.top_p != 1.0:
            generation_config["topP"] = self.top_p
        else:
            generation_config["topP"] = None

        thinking_budget = self.thinking_budget
        thinking_level = self.thinking_level
        thinking_config = {}
        if pure_model_name.lower().startswith("gemini-3"):
            if thinking_level and thinking_level != "OFF":
                thinking_config["thinking_level"] = thinking_level
            elif thinking_budget is not None and thinking_budget > 0:
                thinking_config["thinking_budget"] = thinking_budget
        else:
            if thinking_budget is not None and thinking_budget > 0:
                thinking_config["thinking_budget"] = thinking_budget
        if thinking_config:
            generation_config["thinking_config"] = thinking_config

        safety_threshold = self.safety_level if self.safety_level != "OFF" else "BLOCK_NONE"
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": safety_threshold},
        ]

        # Payload with Prefill
        payload = {
            "contents": [
                {"role": "user", "parts": parts},
                # Prefill to force CSV format
                {"role": "model", "parts": [{"text": 'Response type: csv\n"id","text"'}]}
            ],
            "generationConfig": generation_config,
            "safetySettings": safety_settings
        }

        if self.http_client is None:
            self._initialize_http_client()

        try:
            req_start = time.time()
            response = self.http_client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            elapsed = time.time() - req_start
            self.logger.debug(f"Google REST Batch OCR took {elapsed:.2f}s")
        except Exception as e:
            self.logger.error(f"Google REST Batch API failed: {e}")
            raise e

        # Response Parsing
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                finish_reason = data.get("promptFeedback", {}).get("blockReason", "UNKNOWN")
                raise ValueError(f"Google REST Batch: No candidates. Blocked: {finish_reason}")

            candidate = candidates[0]
            finish_reason = candidate.get("finishReason", "")
            
            if finish_reason == "SAFETY":
                raise ValueError(f"Google REST Batch: Blocked by safety filter")
            
            if finish_reason == "RECITATION":
                raise ValueError(f"Google REST Batch: Blocked by recitation/copyright filter")

            content_parts = candidate.get("content", {}).get("parts", [])
            if not content_parts:
                self.logger.debug(f"Google REST Batch empty candidate: {candidate}")
                raise ValueError(f"Google REST Batch: Empty content parts. Finish Reason: {finish_reason}")

            full_text = content_parts[0].get("text", "")
            
            # Parse CSV Result
            results = {}
            try:
                import csv
                clean_text = full_text.replace("```csv", "").replace("```", "").strip()
                f = io.StringIO(clean_text)
                reader = csv.reader(f, skipinitialspace=True)
                for row in reader:
                    if len(row) >= 2:
                        try:
                            idx_str = row[0].strip()
                            if idx_str.isdigit():
                                idx = int(idx_str)
                                text = row[1].strip()
                                # Clean up newlines
                                text = text.replace('\\n', '').replace('\n', '')
                                results[idx] = text
                        except:
                            continue
            except Exception as e:
                self.logger.warning(f"Batch CSV parse failed: {e}")
            
            num_images = len(content_payload) - 1 
            ordered_texts = []
            for i in range(1, num_images + 1):
                ordered_texts.append(results.get(i, ""))
            
            return ordered_texts

        except Exception as e:
            raise ValueError(f"Failed to parse Google REST Batch response: {e}")

    def _ocr_batch_call(self, content_payload: List[dict], model_name: str, retry_override: int = None) -> List[str]:
        """
        Process a batch of images in a single API call.
        content_payload: List of dicts (text prompt + images)
        """
        api_key_to_use = self._select_api_key()
        if not api_key_to_use:
            self.logger.error("No API key for batch OCR")
            return []

        if not self.client or self.client.api_key != api_key_to_use:
            self._initialize_client(api_key_to_use)

        self._respect_delay()
        
        # Use retry_override if provided, otherwise use self.retry_attempts
        max_attempts = retry_override if retry_override is not None else self.retry_attempts
        
        messages = [
            {"role": "user", "content": content_payload}
        ]

        self.logger.debug(f"Batch OCR request with {len(content_payload)-1} images")

        for attempt in range(max_attempts):
            try:
                # Use Google REST API for Batch if applicable (supports Prefill)
                if self.provider == "Google" and not self.endpoint:
                    try:
                        return self._ocr_google_rest_batch(content_payload, model_name)
                    except Exception as e:
                        if attempt < max_attempts - 1:
                            self.logger.warning(f"Google Batch attempt {attempt+1} failed: {e}. Retrying...")
                            time.sleep(self.retry_timeout)
                            continue
                        else:
                            raise e

                req_start = time.time()
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=self.max_response_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                elapsed = time.time() - req_start
                self.logger.debug(f"Batch OCR took {elapsed:.2f}s")

                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    full_text = response.choices[0].message.content
                    
                    # Parse CSV: id, text
                    results = {}
                    try:
                        import csv
                        clean_text = full_text.replace("```csv", "").replace("```", "").strip()
                        f = io.StringIO(clean_text)
                        reader = csv.reader(f, skipinitialspace=True)
                        for row in reader:
                            if len(row) >= 2:
                                try:
                                    idx_str = row[0].strip()
                                    if idx_str.isdigit():
                                        idx = int(idx_str)
                                        text = row[1].strip()
                                        # Clean up newlines
                                        text = text.replace('\\n', '').replace('\n', '')
                                        results[idx] = text
                                except:
                                    continue
                        if not results:
                            raise ValueError("Batch OCR: No valid results parsed from CSV.")

                    except Exception as e:
                        self.logger.warning(f"Batch CSV parse failed: {e}")
                        raise e # Trigger retry
                    
                    # Convert dict to ordered list
                    num_images = len(content_payload) - 1
                    ordered_texts = []
                    for i in range(1, num_images + 1):
                        ordered_texts.append(results.get(i, ""))
                    
                    return ordered_texts
                else:
                    raise ValueError("Batch OCR: Empty response from API")
                
            except Exception as e:
                if attempt < max_attempts - 1:
                    self.logger.warning(f"Batch OCR attempt {attempt+1} failed: {e}. Retrying...")
                    time.sleep(self.retry_timeout)
                else:
                    self.logger.error(f"Batch OCR failed after retries: {e}")
                    num_images = len(content_payload) - 1
                    return ["[ERROR: Batch Failed]"] * num_images
        
        return []

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
        im_h, im_w = img.shape[:2]
        
        valid_blks = []
        image_payloads = []
        
        # 1. Prepare images
        for i, blk in enumerate(blk_list):
            x1, y1, x2, y2 = blk.xyxy
            if 0 <= x1 < x2 <= im_w and 0 <= y1 < y2 <= im_h:
                cropped_img = img[y1:y2, x1:x2]
                _, buffer = cv2.imencode(".jpg", cropped_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                img_base64 = base64.b64encode(buffer).decode("utf-8")
                
                valid_blks.append(blk)
                
                # OpenAI Format
                image_payloads.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                        "detail": self.detail_level if self.detail_level in ["low", "high"] else "auto"
                    }
                })
            else:
                blk.text = ""

        if not valid_blks:
            return

        # 2. Chunking (User Configurable)
        CHUNK_SIZE = self.batch_size
        
        model_name = self.override_model or self.model
        if ": " in model_name:
            model_name = model_name.split(": ", 1)[1]

        for i in range(0, len(valid_blks), CHUNK_SIZE):
            chunk_blks = valid_blks[i : i + CHUNK_SIZE]
            chunk_imgs = image_payloads[i : i + CHUNK_SIZE]
            
            # Prompt Construction
            lang_name = self.language
            num_imgs = len(chunk_imgs)
            
            user_prompt_template = self.prompt
            
            # Check if user provided batch instructions via {num_images} placeholder
            # Support both {num_images} and {num_imgs}
            has_placeholder = "{num_images}" in user_prompt_template or "{num_imgs}" in user_prompt_template
            
            # Safe formatting args
            fmt_args = {"language": lang_name, "num_images": num_imgs, "num_imgs": num_imgs}

            if has_placeholder:
                full_prompt = user_prompt_template.format(**fmt_args)
            else:
                # 1. Base Prompt
                base_prompt = user_prompt_template.format(**fmt_args)
                
                # 2. Batch Specific Instructions (Legacy Support)
                batch_instruction = (
                    f"\n\n[BATCH PROCESSING INSTRUCTION]\n"
                    f"Task: Perform OCR on the {num_imgs} images attached.\n"
                    f"Output Requirement: Return ONLY a CSV with columns: id, text.\n"
                    f"Formatting: Consolidate text into a single line. Remove all line breaks within the text field.\n"
                    f"ID Mapping: IDs must be 1-based integers (1 to {num_imgs}) corresponding to the image order.\n"
                    f"Example Output:\n1, Text of image 1\n2, Text of image 2\n"
                    f"Note: If an image has no text, return an empty string for that ID."
                )
                
                # Combine
                full_prompt = base_prompt + batch_instruction
            
            content = [{"type": "text", "text": full_prompt}] + chunk_imgs
            
            # Call Batch API
            texts = self._ocr_batch_call(content, model_name)
            
            # Assign with Fallback Logic
            for j, text in enumerate(texts):
                if j < len(chunk_blks):
                    # Check for error or empty result from Batch
                    if text.startswith("[ERROR") or not text.strip():
                        self.logger.warning(f"Batch OCR item {j} failed: {text}. Fallback to individual OCR.")
                        try:
                            # Extract base64 from the payload we prepared earlier
                            # chunk_imgs[j] is {"type": "image_url", "image_url": {"url": "data:...", ...}}
                            data_url = chunk_imgs[j]["image_url"]["url"]
                            base64_data = data_url.split(",", 1)[1]
                            
                            # Use the robust single-image OCR method (Retries -> Fallback Model -> MangaOCR)
                            fallback_text = self.ocr(base64_data)
                            chunk_blks[j].text = fallback_text
                        except Exception as e:
                            self.logger.error(f"Fallback OCR failed for item {j}: {e}")
                            chunk_blks[j].text = text # Keep original error
                    else:
                        chunk_blks[j].text = text

    def ocr_img(self, img: np.ndarray, prompt: str = "") -> str:
        _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        return self.ocr(img_base64, prompt_override=prompt)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in ["api_key", "multiple_keys", "endpoint", "proxy", "provider", "request_timeout", "ocr_parallel_workers"]:
            self.client = None
            if self.http_client:
                try:
                    self.http_client.close()
                except:
                    pass
                self.http_client = None
        if param_key in ["requests_per_minute", "delay"]:
            self.request_count_minute = 0
            self.minute_start_time = time.time()
            self.last_request_time = 0


# -------------------------------------------------------------------------
# Monkey Patch for Async Pipeline
# -------------------------------------------------------------------------
def _apply_chunked_processing_patch():
    LOGGER = None  # Initialize before try block
    try:
        from ui.module_manager import ImgtransThread, text_is_empty
        from utils.config import RunStatus, pcfg
        from utils.logger import logger as LOGGER
        from utils import shared
        from utils.imgproc_utils import get_block_mask
        from utils.textblock import sort_regions
        
        cfg_module = pcfg.module  # cfg_module is created in module_manager

        if LOGGER:
            LOGGER.info("LLM OCR V3: Attempting to apply async pipeline patch...")

        if not hasattr(ImgtransThread, '_original_imgtrans_pipeline'):
             ImgtransThread._original_imgtrans_pipeline = ImgtransThread._imgtrans_pipeline

        def _imgtrans_pipeline_async(self):
            # --- RESOURCE MANAGEMENT ---
            import torch
            import cv2
            import os
            
            # --- PHYSICAL RESOURCE CAPPING (OS LEVEL) ---
            try:
                import psutil
                p = psutil.Process(os.getpid())
                # 1. Lower priority
                if os.name == 'nt':
                    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                
                # 2. CPU Affinity: Force process to use 70% of cores
                all_cores = list(range(psutil.cpu_count()))
                target_cores = all_cores[:max(1, int(len(all_cores) * 0.7))]
                p.cpu_affinity(target_cores)
                
                # 3. Environment Variables for math libraries
                os.environ["OMP_NUM_THREADS"] = "2"
                os.environ["MKL_NUM_THREADS"] = "2"
                
                if LOGGER:
                    LOGGER.info(f"LLM OCR V3: OS-Level Cap (70%). Using Cores: {target_cores}")
            except Exception as e:
                if LOGGER: LOGGER.warning(f"Could not set CPU affinity: {e}")
        
            # Limit internal math libraries to 2 for better balance
            torch.set_num_threads(2)
            cv2.setNumThreads(2)
            # Check if current OCR module requires chunking
            is_target_ocr = getattr(self.ocr, 'use_page_batching', False) or \
                            self.ocr.__class__.__name__ == 'PaddleOCRVLManga'
            
            if LOGGER:
                LOGGER.info(f"LLM OCR V3: is_target_ocr={is_target_ocr}, OCR class={self.ocr.__class__.__name__}, use_page_batching={getattr(self.ocr, 'use_page_batching', 'N/A')}")
            
            if not is_target_ocr:
                if LOGGER:
                    LOGGER.info("LLM OCR V3: Using original pipeline (not target OCR)")
                return self._original_imgtrans_pipeline()

            if LOGGER:
                LOGGER.info("🚀 Async Pipeline: Wait for 20 pages before starting OCR.")

            self.detect_counter = 0
            self.ocr_counter = 0
            self.translate_counter = 0
            self.inpaint_counter = 0
            
            # Setup pages
            all_pages = list(self.imgtrans_proj.pages.keys())
            if self.pages_to_process is not None and len(self.pages_to_process) > 0:
                pages_to_iterate = self.pages_to_process
                self.num_pages = len(self.pages_to_process)
                for process_idx, page_name in enumerate(pages_to_iterate):
                    if page_name in all_pages:
                        self.process_idx_to_page_idx[process_idx] = all_pages.index(page_name)
            else:
                pages_to_iterate = all_pages
                self.num_pages = len(self.imgtrans_proj.pages)
                for i in range(self.num_pages):
                    self.process_idx_to_page_idx[i] = i

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

            # --- ASYNC LOGIC ---
            # Respect user UI settings from the OCR module
            ocr_workers = getattr(self.ocr, 'ocr_parallel_workers', 50)
            buffer_pages = getattr(self.ocr, 'initial_buffer_pages', 20)
            
            # Initial buffer: Wait for N pages before starting OCR
            initial_buffer = buffer_pages if self.num_pages >= buffer_pages else max(1, int(self.num_pages / 2))
            ocr_task_buffer = []
            ocr_started = False
            
            # OCR Executor (Background Thread)
            if LOGGER:
                LOGGER.info(f"🚀 Async Pipeline: Using {ocr_workers} workers as per settings.")
            ocr_executor = concurrent.futures.ThreadPoolExecutor(max_workers=ocr_workers)
            ocr_futures = []
            
            # Batch save control (like translator)
            last_save_time = 0.0
            save_interval = 3.0  # Save every 3 seconds
            save_lock = threading.Lock()
            inpaint_lock = threading.Lock() # New lock for sequential inpainting

            def run_ocr_step(imgname_arg):
                if self.stop_requested: return
                try:
                    # Read fresh data
                    img_ocr = self.imgtrans_proj.read_img(imgname_arg)
                    blk_list_ocr = self.imgtrans_proj.pages[imgname_arg]
                    
                    if cfg_module.enable_ocr:
                        self.ocr.run_ocr(img_ocr, blk_list_ocr)
                        
                        # --- OCR Postprocess (Keyword Substitution) ---
                        # Explicitly call the hook since we are bypassing the original pipeline
                        
                        # 1. Try finding hook on self (ImgtransThread)
                        hook_func = getattr(self, 'ocr_postprocess', None)
                        
                        # 2. Fallback: Try finding hook via MainWindow
                        if not hook_func:
                            try:
                                from qtpy.QtWidgets import QApplication
                                for widget in QApplication.topLevelWidgets():
                                    if widget.__class__.__name__ == 'MainWindow':
                                        if hasattr(widget, 'ocr_postprocess'):
                                            hook_func = widget.ocr_postprocess
                                            if LOGGER: LOGGER.info("Found ocr_postprocess hook via MainWindow fallback.")
                                        break
                            except Exception as e:
                                if LOGGER: LOGGER.warning(f"Failed to find MainWindow fallback: {e}")

                        if hook_func:
                            try:
                                # Snapshot for logging changes (Use enumerate index as key)
                                original_texts = {i: blk.text for i, blk in enumerate(blk_list_ocr)}
                                
                                if LOGGER:
                                    LOGGER.info(f"Running OCR Postprocess on {len(blk_list_ocr)} text blocks...")

                                # Call the hook (modifies blk_list_ocr in-place)
                                hook_func(blk_list_ocr, img_ocr, self.ocr)
                                
                                # Log any changes (User requested logs)
                                change_count = 0
                                if LOGGER:
                                    for i, blk in enumerate(blk_list_ocr):
                                        orig = original_texts.get(i, "")
                                        if orig != blk.text:
                                            LOGGER.info(f"OCR Subst: '{orig}' -> '{blk.text}'")
                                            change_count += 1
                                    
                                    if change_count == 0:
                                        LOGGER.info("OCR Postprocess: No keywords matched or substituted.")

                            except Exception as e:
                                if LOGGER:
                                    LOGGER.error(f"Error in OCR postprocess hook: {e}")
                        
                        # Batch save with interval control
                        with save_lock:
                            nonlocal last_save_time
                            self.ocr_counter += 1
                            current_time = time.time()
                            # Update immediately for the very first item to show responsiveness
                            if self.ocr_counter == 1 or current_time - last_save_time >= save_interval:
                                self.update_ocr_progress.emit(self.ocr_counter)
                                self.imgtrans_proj.update_page_progress(imgname_arg, RunStatus.FIN_OCR)
                                last_save_time = current_time

                    # Always proceed to translation after OCR completes (success or final failure)
                    # This ensures the page is processed even if OCR failed
                    if cfg_module.enable_translate:
                        if self.parallel_trans:
                            self.translate_thread.push_pagekey_queue(imgname_arg)
                        elif not low_vram_trans:
                            self.translator.translate_textblk_lst(blk_list_ocr)
                            self.translate_counter += 1
                            self.update_translate_progress.emit(self.translate_counter)

                    if cfg_module.enable_inpaint:
                        # SEQUENTIAL GATE: Only one thread can use the inpainter at a time
                        with inpaint_lock:
                            mask_ocr = self.imgtrans_proj.load_mask_by_imgname(imgname_arg)
                            if mask_ocr is not None:
                                try:
                                    inpainted = self.inpainter.inpaint(img_ocr, mask_ocr, blk_list_ocr)
                                    self.imgtrans_proj.save_inpainted(imgname_arg, inpainted)
                                except Exception as e:
                                    if LOGGER:
                                        LOGGER.error(f"Inpainting failed for {imgname_arg}: {e}")
                            
                        self.inpaint_counter += 1
                        self.update_inpaint_progress.emit(self.inpaint_counter)
                        self.imgtrans_proj.update_page_progress(imgname_arg, RunStatus.FIN_INPAINT)
                        
                except Exception as e:
                    LOGGER.error(f"Async OCR failed for {imgname_arg}: {e}")

            # Main Detection Loop
            processed_count = 0
            for imgname in pages_to_iterate:
                if self.stop_requested:
                    LOGGER.info('Pipeline stopped')
                    break

                # --- ADVANCED CPU THROTTLE: Wave Throttling ---
                # Only throttle if OCR is actually active, otherwise we just sprint
                if ocr_started and cfg_module.enable_ocr:
                    # Calculate the "Lead" (how far Detection is ahead of OCR)
                    lead = self.detect_counter - self.ocr_counter
                    
                    if lead >= initial_buffer:
                        # Massive lead brake
                        time.sleep(0.05)
                    
                    # Politeness yield: 0.01s is enough for OS to context-switch to OCR/Inpaint
                    if cfg_module.enable_ocr or cfg_module.enable_translate:
                        time.sleep(0.01)
                
                # 1. Detection (Main Thread)
                img = self.imgtrans_proj.read_img(imgname)
                mask = blk_list = None
                need_save_mask = False
                
                if cfg_module.enable_detect:
                    try:
                        mask, blk_list = self.textdetector.detect(img, self.imgtrans_proj)
                        need_save_mask = True
                    except Exception:
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
                    ocr_task_buffer.append(imgname)
                    if processed_count >= initial_buffer:
                        ocr_started = True
                        if LOGGER:
                            LOGGER.info(f"Initial buffer reached ({processed_count} pages). Starting OCR.")
                        # Flush buffer
                        for buffered_img in ocr_task_buffer:
                            future = ocr_executor.submit(run_ocr_step, buffered_img)
                            ocr_futures.append(future)
                        ocr_task_buffer.clear()
                else:
                    # Once started, submit immediately
                    future = ocr_executor.submit(run_ocr_step, imgname)
                    ocr_futures.append(future)

            # Flush remaining buffer
            for buffered_img in ocr_task_buffer:
                future = ocr_executor.submit(run_ocr_step, buffered_img)
                ocr_futures.append(future)

            # Wait for completion
            concurrent.futures.wait(ocr_futures)
            ocr_executor.shutdown()
            
            # Final save after all OCR tasks complete
            if self.ocr_counter > 0:
                self.update_ocr_progress.emit(self.ocr_counter)
                for imgname in pages_to_iterate:
                    self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_OCR)

            if cfg_module.enable_translate and low_vram_trans:
                self.unload_modules(['textdetector', 'inpainter', 'ocr'])
                for imgname in pages_to_iterate:
                    if self.stop_requested: break
                    blk_list = self.imgtrans_proj.pages[imgname]
                    self.translator.translate_textblk_lst(blk_list)
                    self.translate_counter += 1
                    self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_TRANSLATE)
                    self.update_translate_progress.emit(self.translate_counter)
            
            # -----------------------------------------------------------------
            # FINAL COORDINATION STEP
            # -----------------------------------------------------------------
            # Image saving is now exclusively handled by the Translator module.
            # We just need to wait until the Translator is completely done, 
            # including its final image saving process.
            # -----------------------------------------------------------------
            # FINAL COORDINATION STEP
            # -----------------------------------------------------------------
            if cfg_module.enable_translate:
                # Target count identification
                target = getattr(self.translate_thread, 'num_process_pages', 0)
                if target == 0:
                    target = getattr(self.translate_thread, 'num_pages', 0)
                if target == 0:
                    target = self.num_pages
                
                if LOGGER:
                    LOGGER.info(f"OCR finished. Waiting for Translator and Final Save... (Target: {target})")
                
                # We wait for the counter to hit 100%. 
                while self.translate_thread.finished_counter < target:
                    if self.stop_requested:
                        break
                    time.sleep(0.5)
                
                # --- FINAL VERIFIED SAVE ---
                # Import the save function from the translator module and call it
                try:
                    # Use sys.modules to avoid import lock deadlocks
                    import sys
                    v4_save_func = None
                    
                    if 'modules.translators.trans_llm_api_v4' in sys.modules:
                        mod = sys.modules['modules.translators.trans_llm_api_v4']
                        v4_save_func = getattr(mod, '_v4_headless_save_entry', None)
                    else:
                        # Should be loaded by now, but just in case
                        import modules.translators.trans_llm_api_v4 as mod
                        v4_save_func = getattr(mod, '_v4_headless_save_entry', None)

                    if v4_save_func:
                        # Check if translator handles saving itself (V4 Translator)
                        translator = getattr(self, 'translator', None)
                        if translator and getattr(translator, 'use_image_batching', False):
                            if LOGGER:
                                LOGGER.info("Translator is V4-capable. Skipping redundant save in OCR pipeline.")
                        else:
                            if LOGGER:
                                LOGGER.info("Initiating Final Verified Save from OCR Pipeline context...")
                            # Pass project explicitly to avoid lookup failures
                            v4_save_func(self.translate_thread, self.imgtrans_proj)
                            
                            # Mark as completed to prevent double-save in orchestrator
                            self.translate_thread._v4_save_completed = True
                    else:
                        if LOGGER: LOGGER.error("V4 Save function not found in module.")
                except Exception as save_err:
                    if LOGGER:
                        LOGGER.error(f"Failed to trigger verified save: {save_err}")
                
                if LOGGER:
                    LOGGER.info(f"Pipeline complete ({self.translate_thread.finished_counter}/{target}).")
            else:
                if LOGGER:
                    LOGGER.info("OCR Pipeline tasks completed.")

            if self.stop_requested and (not cfg_module.enable_translate or not self.parallel_trans):
                self.pipeline_stopped.emit()

            if self.stop_requested and (not cfg_module.enable_translate or not self.parallel_trans):
                self.pipeline_stopped.emit()
            else:
                # If no translation, we just finish (OCR only does not need image save)
                if LOGGER:
                    LOGGER.info("OCR Pipeline tasks completed.")

            if self.stop_requested and (not cfg_module.enable_translate or not self.parallel_trans):
                self.pipeline_stopped.emit()

        ImgtransThread._imgtrans_pipeline = _imgtrans_pipeline_async
        if LOGGER:
            LOGGER.info("Async Pipeline applied to ImgtransThread")

    except ImportError as e:
        if LOGGER:
            LOGGER.error(f"LLM OCR V3: Failed to apply async pipeline patch (ImportError): {e}")
    except Exception as e:
        if LOGGER:
            LOGGER.error(f"LLM OCR V3: Failed to apply async pipeline patch: {e}")
