import re
import time
import json
import asyncio
import traceback
from typing import List, Dict, Optional, Type

import httpx
import openai
from pydantic import BaseModel, Field, ValidationError

from .base import BaseTranslator, register_translator


class InvalidNumTranslations(Exception):
    """Exception raised when the number of translations does not match the number of sources."""

    pass


class TranslationElement(BaseModel):
    id: int = Field(..., description="The original numeric ID of the text snippet.")
    translation: str = Field(
        ..., description="The translated text corresponding to the id."
    )


class TranslationResponse(BaseModel):
    translations: List[TranslationElement] = Field(
        ..., description="A list of all translated elements."
    )


@register_translator("LLM_API_Translator_V2")
class LLM_API_Translator_V2(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    params: Dict = {
        "provider": {
            "type": "selector",
            "options": ["OpenAI", "Google", "Grok", "OpenRouter", "LLM Studio"],
            "value": "OpenAI",
            "description": "Select the LLM provider.",
        },
        "apikey": {
            "value": "",
            "description": "Single API key to use if multiple keys are not provided.",
        },
        "multiple_keys": {
            "type": "editor",
            "value": "",
            "description": "API keys separated by semicolons (;). Requests will rotate through these keys.",
        },
        "concurrent requests": {
            "value": 5,
            "description": "Number of concurrent API requests allowed for a single key.",
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
            ],
            "value": "OAI: gpt-4o",
            "description": "Select a model that supports JSON Mode for structured output.",
        },
        "override model": {
            "value": "",
            "description": "Specify a custom model name to override the selected model.",
        },
        "endpoint": {
            "value": "",
            "description": "Base URL for the API. Leave empty for provider default.",
        },
        "system_prompt": {
            "type": "editor",
            "value": "You are an expert translator. Your task is to accurately translate the given text snippets. You MUST provide the output strictly in the specified JSON format, without any additional explanations or markdown formatting. The JSON object must have a single key \'translations\', which is a list of objects, each with an \'id\' (integer) and a \'translation\' (string).\n\nExample Output Schema:\n{\"translations\": [{\"id\": 1, \"translation\": \"Translated text here.\"}]}",
            "description": "System message to instruct the LLM on its role and required output format.",
        },
        "invalid repeat count": {
            "value": 2,
            "description": "Number of retries if the count of translations mismatches the source count.",
        },
        "max requests per minute": {
            "value": 20,
            "description": "Maximum requests per minute for EACH API key.",
        },
        "delay": {
            "value": 0.3,
            "description": "Global delay in seconds between requests.",
        },
        "max response tokens": {
            "value": 4096,
            "description": "Maximum tokens for the response generation (Inference).",
        },
        "max prompt tokens": {
            "value": 3000,
            "description": "Maximum tokens for the input prompt (Context/Batch limit). Requests will be split if they exceed this limit.",
        },
        "thinking budget": {
            "value": 0,
            "description": "Token budget for reasoning/thinking (0 to disable). Applies to Gemini models supporting 'thinkingBudget'.",
        },
        "thinking level": {
            "type": "selector",
            "options": ["OFF", "minimal", "low", "medium", "high"],
            "value": "OFF",
            "description": "Thinking level for Gemini 3 models. Overrides 'thinking budget' if set. (minimal/low/medium/high)",
        },
        "temperature": {
            "value": 0.1,
            "description": "Sampling temperature. Lower values are recommended for structured output.",
        },
        "top p": {
            "value": 1.0,
            "description": "Top P for sampling.",
        },
        "retry attempts": {
            "value": 3,
            "description": "Number of retry attempts on API connection or parsing failures.",
        },
        "retry timeout": {
            "value": 15,
            "description": "Timeout between retry attempts (seconds).",
        },
        "proxy": {
            "value": "",
            "description": "Proxy address (e.g., http(s)://user:password@host:port or socks4/5://user:password@host:port)",
        },
        "frequency penalty": {
            "value": 0.0,
            "description": "Frequency penalty (OpenAI).",
        },
        "presence penalty": {"value": 0.0, "description": "Presence penalty (OpenAI)."},
    }

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

        proxy = self.proxy
        http_client = None
        if proxy:
            try:
                proxy_mounts = {
                    "http://": httpx.HTTPTransport(proxy=proxy),
                    "https://": httpx.HTTPTransport(proxy=proxy),
                }
                http_client = httpx.AsyncClient(mounts=proxy_mounts)
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize proxy '{proxy}': {e}. Proceeding without proxy."
                )
                http_client = httpx.AsyncClient()
        else:
            http_client = httpx.AsyncClient()

        masked_key = (
            api_key_to_use[:4] + "..." + api_key_to_use[-4:]
            if len(api_key_to_use) > 8
            else api_key_to_use
        )
        self.logger.debug(
            f"Initializing client for {provider} with key {masked_key} at endpoint {endpoint}"
        )

        try:
            self.client = openai.AsyncOpenAI(
                api_key=api_key_to_use, base_url=endpoint, http_client=http_client
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
    def model(self) -> str:
        return self.get_param_value("model")

    @property
    def override_model(self) -> Optional[str]:
        return self.get_param_value("override model") or None

    @property
    def endpoint(self) -> Optional[str]:
        return self.get_param_value("endpoint") or None

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
    def max_prompt_tokens(self) -> int:
        val = self.get_param_value("max prompt tokens")
        # Default fallback if not set
        return int(val) if val != "" else 3000

    @property
    def thinking_budget(self) -> int:
        val = self.get_param_value("thinking budget")
        return int(val) if val != "" else 0

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
    def concurrent_requests(self) -> int:
        val = self.get_param_value("concurrent requests")
        return int(val) if val != "" else 5

    def _assemble_prompts(self, queries: List[str], to_lang: str):
        from_lang = self.lang_map.get(self.lang_source, self.lang_source)
        
        limit_tokens = self.max_prompt_tokens
        
        # Estimate template size
        # Template: "Please translate... from X to Y... INPUT:\n[]"
        # Approx 150-200 chars. 
        template_overhead = 200
        
        current_batch = []
        current_batch_tokens = 0
        
        for query in queries:
            # Estimate query tokens. 
            # Using 1 char = 1 token as a conservative estimate (covers CJK well).
            # + overhead for JSON item structure {"id": N, "source": "..."} (approx 40-50 chars)
            item_cost = len(query) + 50
            
            if current_batch and (current_batch_tokens + item_cost + template_overhead > limit_tokens):
                # Yield current batch
                yield self._make_prompt(current_batch, from_lang, to_lang), len(current_batch)
                current_batch = []
                current_batch_tokens = 0
                
            current_batch.append(query)
            current_batch_tokens += item_cost
            
        if current_batch:
            yield self._make_prompt(current_batch, from_lang, to_lang), len(current_batch)
            
    def _make_prompt(self, queries: List[str], from_lang: str, to_lang: str) -> str:
        input_elements = [
            {"id": i + 1, "source": query} for i, query in enumerate(queries)
        ]
        input_json_str = json.dumps(input_elements, ensure_ascii=False, indent=2)

        prompt = (
            f"Please translate the following text snippets from {from_lang} to {to_lang}. "
            f"The input is provided as a JSON array. Respond with a JSON object in the specified format.\n\n"
            f"INPUT:\n{input_json_str}"
        )
        return prompt

    def _respect_delay(self):
        # Delay logic is less relevant in async/concurrent mode for single requests,
        # but rate limiting per minute still applies.
        # We'll rely on rate limits mostly, but simple sleep can block event loop if not careful.
        # For true async rate limiting, we would need an async rate limiter.
        # For now, we will use a simple non-blocking sleep if needed, but 'time.sleep' blocks.
        # In async context, we should ideally use 'await asyncio.sleep'.
        # However, since this method is called inside async function, we can't easily change it 
        # without changing call signature everywhere or ignoring it for concurrency.
        # Given 'concurrent requests' is the main throttle, we will skip explicit blocking delays
        # per request to maximize throughput, relying on RPM checks.
        
        current_time = time.time()
        rpm = self.max_rpm
        
        if rpm > 0:
            if current_time - self.minute_start_time >= 60:
                self.request_count_minute = 0
                self.minute_start_time = current_time
            if self.request_count_minute >= rpm:
                wait_time = 60.1 - (current_time - self.minute_start_time)
                if wait_time > 0:
                    self.logger.warning(
                        f"Global RPM limit ({rpm}) reached. Waiting {wait_time:.2f} seconds."
                    )
                    # Blocking sleep here is bad for async, but acceptable if we hit global limit
                    time.sleep(wait_time)
                self.request_count_minute = 0
                self.minute_start_time = time.time()
        
        self.request_count_minute += 1

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

    async def _request_translation(self, prompt: str) -> Optional[TranslationResponse]:
        model_name = self.override_model or self.model
        if ": " in model_name:
            model_name = model_name.split(": ", 1)[1]

        # Route to Native REST API for Google if no custom endpoint is set
        # This bypasses OpenAI client initialization and the compatibility warning
        if self.provider == "Google" and not self.endpoint:
            return await self._request_translation_google_rest(prompt, model_name)

        current_api_key = "lm-studio"
        if self.provider != "LLM Studio":
            current_api_key = self._select_api_key()
            if not current_api_key:
                raise ConnectionError("No available API key found.")

        if self.provider == "LLM Studio" and not self.endpoint:
            raise ValueError(
                "Endpoint must be specified when using the LLM Studio provider (e.g., http://localhost:1234/v1)."
            )

        if not self._initialize_client(current_api_key):
            raise ConnectionError("Failed to initialize API client.")

        # self._respect_delay() # Skipped for async concurrency to prevent blocking

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        api_args = {
            "model": model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        # Handle 'thinking budget' and 'thinking level' for Gemini/Google
        thinking_budget = self.thinking_budget
        thinking_level = self.thinking_level

        is_gemini_openai = self.provider == "Google" and "openai" in (self.endpoint or "").lower()
        if is_gemini_openai:
            if (thinking_level and thinking_level != "OFF") or thinking_budget > 0:
                self.logger.warning(
                    "Thinking Level/Budget is currently not supported in Google's OpenAI compatibility mode. Ignoring these parameters to prevent errors."
                )

        if self.provider == "LLM Studio":
            self.logger.debug("Using 'json_schema' mode for LLM Studio.")
            api_args["response_format"] = {
                "type": "json_schema",
                "json_schema": {"schema": TranslationResponse.model_json_schema()},
            }
        elif self.provider in ["OpenAI", "Grok", "Google", "OpenRouter"]:
            self.logger.debug(f"Using 'json_object' mode for {self.provider}.")
            api_args["response_format"] = {"type": "json_object"}

        if self.provider == "OpenAI":
            api_args["frequency_penalty"] = self.frequency_penalty
            api_args["presence_penalty"] = self.presence_penalty

        try:
            completion = await self.client.chat.completions.create(**api_args)
        except Exception as e:
            self.logger.error(f"API request failed: {e}")
            raise

        if (
            completion.choices
            and completion.choices[0].message
            and completion.choices[0].message.content
        ):
            raw_content = completion.choices[0].message.content
            return self._parse_json_response(raw_content)
        else:
            self.logger.warning("No valid message content in API response.")
            return None

    async def _request_translation_google_rest(self, prompt: str, model_name: str) -> Optional[TranslationResponse]:
        """
        Handles requests to Google's native REST API (generateContent).
        This allows using 'generationConfig' with 'thinkingConfig' which is not supported in the OpenAI compatibility layer.
        """
        api_key = self._select_api_key()
        if not api_key:
            raise ConnectionError("No available API key found for Google REST API.")

        if not model_name.startswith("models/"):
             # Clean up model name if it has prefixes like "GGL: " or just "gemini-..."
            if ": " in model_name:
                model_name = model_name.split(": ", 1)[1]
            if not model_name.startswith("models/"):
                 model_name = f"models/{model_name}"

        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }

        # Construct Gemini-native JSON payload
        generation_config = {
            "temperature": self.temperature,
            "topP": self.top_p,
            "maxOutputTokens": self.max_tokens,
            "responseMimeType": "application/json" 
        }

        # Thinking Config
        thinking_budget = self.thinking_budget
        thinking_level = self.thinking_level
        thinking_config = {}
        
        if thinking_level and thinking_level != "OFF":
             thinking_config["thinkingLevel"] = thinking_level
        elif thinking_budget > 0:
             thinking_config["thinkingBudget"] = thinking_budget
        
        if thinking_config:
             generation_config["thinkingConfig"] = thinking_config
             self.logger.debug(f"Applied Gemini thinkingConfig: {thinking_config}")

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "systemInstruction": {
                "parts": [{"text": self.system_prompt}]
            },
            "generationConfig": generation_config
        }

        self.logger.debug(f"Sending Google REST API request to {url}")
        
        # Use a temporary client just for this request or reuse self.client's http_client if accessible.
        # Since self.client is openai.AsyncOpenAI, we can't easily reuse its transport.
        # We'll create a new httpx client or use a global one. 
        # For simplicity and isolation, we create a new one, but respecting proxy.
        
        proxy = self.proxy
        mounts = {}
        if proxy:
             mounts = {
                 "http://": httpx.HTTPTransport(proxy=proxy),
                 "https://": httpx.HTTPTransport(proxy=proxy),
             }
        
        async with httpx.AsyncClient(mounts=mounts, timeout=self.retry_timeout * 2) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as e:
                self.logger.error(f"Google REST API Error: {e.response.text}")
                raise
            except Exception as e:
                self.logger.error(f"Google REST API connection failed: {e}")
                raise

        # Parse Gemini response
        # Structure: {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                 # Check for promptFeedback if blocked
                 prompt_feedback = data.get("promptFeedback", {})
                 if prompt_feedback:
                      self.logger.warning(f"Google API blocked request: {prompt_feedback}")
                 return None
            
            # Usually the first candidate
            content_parts = candidates[0].get("content", {}).get("parts", [])
            if not content_parts:
                 return None
            
            raw_text = content_parts[0].get("text", "")
            if not raw_text:
                 return None

            return self._parse_json_response(raw_text)

        except Exception as e:
             self.logger.error(f"Failed to parse Google REST API response: {e}")
             self.logger.debug(f"Response data: {data}")
             raise

    def _parse_json_response(self, raw_content: str) -> Optional[TranslationResponse]:
        """Shared JSON parsing logic for both OpenAI and Google REST responses."""
        json_to_parse = raw_content.strip()

        match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", json_to_parse, re.DOTALL
        )
        if match:
            self.logger.debug(
                "Markdown code block detected. Extracting JSON content."
            )
            json_to_parse = match.group(1)
        else:
            start = json_to_parse.find("{")
            if start != -1:
                json_to_parse = json_to_parse[start:]
        try:
            try:
                data_to_validate = json.loads(json_to_parse)
            except json.JSONDecodeError:
                # Try to decode just the first valid JSON object if there's extra data
                data_to_validate, _ = json.JSONDecoder().raw_decode(json_to_parse)
                self.logger.debug("Successfully parsed JSON using raw_decode (ignoring trailing data).")

            validated_response = TranslationResponse.model_validate(
                data_to_validate
            )
            return validated_response
        except (ValidationError, json.JSONDecodeError) as e:
            self.logger.warning(
                f"Initial Pydantic validation failed: {e}. Attempting to fix simple dictionary or list format."
            )
            try:
                simple_data = json.loads(json_to_parse)
                fixed_translations = []

                if isinstance(simple_data, dict) and all(
                    k.isdigit() for k in simple_data.keys()
                ):
                    fixed_translations = [
                        {"id": int(k), "translation": v}
                        for k, v in simple_data.items()
                    ]
                elif isinstance(simple_data, list):
                    fixed_translations = simple_data

                if fixed_translations:
                    fixed_data = {"translations": fixed_translations}
                    self.logger.debug(
                        f"Transformed simple response to: {fixed_data}"
                    )
                    return TranslationResponse.model_validate(
                        fixed_data
                    )
                else:
                    raise e
            except (ValidationError, json.JSONDecodeError, Exception) as final_e:
                self.logger.error(
                    f"Pydantic validation or JSON parsing failed even after attempting fix: {final_e}"
                )
                self.logger.debug(f"Raw JSON content from API: {raw_content}")
                raise

    def _unused_method(self): # Placeholder to match indentation for the replacement block context if needed
        pass

    async def _process_batch_async(self, prompt, num_src, semaphore):
        async with semaphore:
            RETRYABLE_EXCEPTIONS = (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
                openai.APIStatusError,
                httpx.RequestError,
                ConnectionError,
            )
            
            api_retry_attempt = 0
            mismatch_retry_attempt = 0

            while True:
                try:
                    parsed_response = await self._request_translation(prompt)

                    if not parsed_response or not parsed_response.translations:
                        raise ValueError(
                            "Received empty or invalid parsed response from API."
                        )

                    if len(parsed_response.translations) != num_src:
                        raise InvalidNumTranslations(
                            f"Expected {num_src}, got {len(parsed_response.translations)}"
                        )

                    translations_dict = {
                        item.id: item.translation
                        for item in parsed_response.translations
                    }
                    # Result for this batch
                    return [translations_dict.get(i, "") for i in range(1, num_src + 1)]

                except InvalidNumTranslations as e:
                    mismatch_retry_attempt += 1
                    self.logger.warning(
                        f"Translation structure mismatch: {e}. Attempt {mismatch_retry_attempt}/{self.invalid_repeat_count}."
                    )
                    if mismatch_retry_attempt >= self.invalid_repeat_count:
                        self.logger.error(
                            "Fatal Error: Failed to get correct translation structure after retries."
                        )
                        return ["[ERROR: Structure Mismatch]"] * num_src
                    await asyncio.sleep(self.retry_timeout / 2)

                except RETRYABLE_EXCEPTIONS as e:
                    api_retry_attempt += 1
                    self.logger.warning(
                        f"API Error (retryable): {type(e).__name__} - {e}. Attempt {api_retry_attempt}/{self.retry_attempts}."
                    )
                    if api_retry_attempt >= self.retry_attempts:
                        self.logger.error(
                            f"Fatal Error: Failed to connect to API after {self.retry_attempts} attempts."
                        )
                        return [f"[ERROR: API Failed]"] * num_src
                    await asyncio.sleep(self.retry_timeout)

                except (
                    ValidationError,
                    json.JSONDecodeError,
                    openai.BadRequestError,
                    openai.AuthenticationError,
                    ValueError,
                ) as e:
                    self.logger.error(
                        f"Fatal Error: An unrecoverable error occurred: {type(e).__name__} - {e}"
                    )
                    self.logger.debug(traceback.format_exc())
                    return [f"[ERROR: {type(e).__name__}]"] * num_src

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []

        to_lang = self.lang_map.get(self.lang_target, self.lang_target)
        
        async def run_tasks():
            tasks = []
            semaphore = asyncio.Semaphore(self.concurrent_requests)
            
            # Prepare all batches
            batches = list(self._assemble_prompts(src_list, to_lang=to_lang))
            
            # Create async tasks for each batch
            for prompt, num_src in batches:
                tasks.append(self._process_batch_async(prompt, num_src, semaphore))
            
            # Run all tasks concurrently
            results = await asyncio.gather(*tasks)
            
            # Flatten results
            flat_results = []
            for batch_result in results:
                flat_results.extend(batch_result)
            return flat_results

        return asyncio.run(run_tasks())

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)

        if param_key in ["proxy", "multiple_keys", "apikey", "provider", "endpoint", "concurrent requests"]:
            self.client = None
