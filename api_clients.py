"""
Unified API client layer for OpenAI, Anthropic, Groq, and Gemini.
Each provider returns a standardized response dict.
Simple retry on transient errors (429, 500+).
"""
import json
import time
import hashlib
import logging
from typing import Any

import config

logger = logging.getLogger(__name__)

MAX_RETRIES = 6
BASE_DELAY = 1.0   # start at 1s, not 2s (paid tier recovers faster)


def _backoff(attempt: int) -> float:
    """Exponential backoff: 1, 2, 4, 8, 16, 32s — capped at 30s for paid tier."""
    return min(BASE_DELAY * (2 ** attempt), 30.0)


# ── Standardized response ──
def _make_response(
    raw_text: str,
    model: str,
    provider: str,
    usage: dict,
    response_id: str = "",
    latency: float = 0.0,
    prompt_hash: str = "",
) -> dict:
    return {
        "raw_text": raw_text,
        "model": model,
        "provider": provider,
        "response_id": response_id,
        "usage": usage,
        "latency": latency,
        "prompt_hash": prompt_hash,
    }


def _prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}||{user}".encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════
# OpenAI
# ═══════════════════════════════════════════════════════
def _call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str,
    json_schema: dict | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    store: bool = False,
    metadata: dict | None = None,
) -> dict:
    from openai import OpenAI, RateLimitError, APIError

    client = OpenAI(api_key=config.OPENAI_API_KEY)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "store": store,
    }
    if json_schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "game_response", "strict": True, "schema": json_schema},
        }
    else:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(**kwargs)
            latency = time.time() - t0

            usage = {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "total_tokens": resp.usage.total_tokens if resp.usage else 0,
                "cached_tokens": getattr(resp.usage, "prompt_tokens_details", {}).get("cached_tokens", 0) if resp.usage else 0,
            }

            return _make_response(
                raw_text=resp.choices[0].message.content or "",
                model=resp.model,
                provider="openai",
                usage=usage,
                response_id=resp.id,
                latency=latency,
                prompt_hash=_prompt_hash(system_prompt, user_prompt),
            )
        except RateLimitError as e:
            delay = _backoff(attempt)
            logger.warning(f"OpenAI 429 (attempt {attempt+1}): retrying in {delay:.1f}s")
            time.sleep(delay)
        except APIError as e:
            if attempt < MAX_RETRIES - 1 and getattr(e, 'status_code', 0) >= 500:
                delay = _backoff(attempt)
                logger.warning(f"OpenAI server error (attempt {attempt+1}): retrying in {delay:.1f}s")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"OpenAI: max retries ({MAX_RETRIES}) exceeded")


# ═══════════════════════════════════════════════════════
# Anthropic
# ═══════════════════════════════════════════════════════
def _call_anthropic(
    system_prompt: str,
    user_prompt: str,
    model: str,
    json_schema: dict | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    store: bool = False,
    metadata: dict | None = None,
) -> dict:
    from anthropic import Anthropic, RateLimitError, APIError

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

    messages = [{"role": "user", "content": user_prompt}]

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }

    # Force JSON output: instruct in system prompt + use prefill trick
    if json_schema:
        messages.append({"role": "assistant", "content": "{"})

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            resp = client.messages.create(**kwargs)
            latency = time.time() - t0

            raw_text = resp.content[0].text if resp.content else ""
            if json_schema:
                raw_text = "{" + raw_text

            usage = {
                "prompt_tokens": resp.usage.input_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.output_tokens if resp.usage else 0,
                "total_tokens": (resp.usage.input_tokens + resp.usage.output_tokens) if resp.usage else 0,
                "cached_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) if resp.usage else 0,
            }

            return _make_response(
                raw_text=raw_text,
                model=resp.model,
                provider="anthropic",
                usage=usage,
                response_id=resp.id,
                latency=latency,
                prompt_hash=_prompt_hash(system_prompt, user_prompt),
            )
        except RateLimitError as e:
            delay = _backoff(attempt)
            logger.warning(f"Anthropic 429 (attempt {attempt+1}): retrying in {delay:.1f}s")
            time.sleep(delay)
        except APIError as e:
            if attempt < MAX_RETRIES - 1 and getattr(e, 'status_code', 0) >= 500:
                delay = _backoff(attempt)
                logger.warning(f"Anthropic server error (attempt {attempt+1}): retrying in {delay:.1f}s")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"Anthropic: max retries ({MAX_RETRIES}) exceeded")


# ═══════════════════════════════════════════════════════
# Groq
# ═══════════════════════════════════════════════════════
def _call_groq(
    system_prompt: str,
    user_prompt: str,
    model: str,
    json_schema: dict | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    store: bool = False,
    metadata: dict | None = None,
) -> dict:
    from groq import Groq, RateLimitError, APIError

    client = Groq(api_key=config.GROQ_API_KEY)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(**kwargs)
            latency = time.time() - t0

            usage = {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "total_tokens": resp.usage.total_tokens if resp.usage else 0,
                "cached_tokens": 0,
            }

            return _make_response(
                raw_text=resp.choices[0].message.content or "",
                model=resp.model,
                provider="groq",
                usage=usage,
                response_id=resp.id,
                latency=latency,
                prompt_hash=_prompt_hash(system_prompt, user_prompt),
            )
        except RateLimitError as e:
            delay = _backoff(attempt)
            logger.warning(f"Groq 429 (attempt {attempt+1}): retrying in {delay:.1f}s")
            time.sleep(delay)
        except APIError as e:
            if attempt < MAX_RETRIES - 1 and getattr(e, 'status_code', 0) >= 500:
                delay = _backoff(attempt)
                logger.warning(f"Groq server error (attempt {attempt+1}): retrying in {delay:.1f}s")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"Groq: max retries ({MAX_RETRIES}) exceeded")


# ═══════════════════════════════════════════════════════
# Gemini (using google.genai SDK)
# ═══════════════════════════════════════════════════════
def _strip_unsupported_schema_fields(schema: dict) -> dict:
    """Remove fields not supported by the Gemini SDK."""
    import copy
    UNSUPPORTED = {"additionalProperties", "$schema", "title"}
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if k not in UNSUPPORTED}
        if isinstance(obj, list):
            return [_clean(item) for item in obj]
        return obj
    return _clean(copy.deepcopy(schema))


def _call_gemini(
    system_prompt: str,
    user_prompt: str,
    model: str,
    json_schema: dict | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    store: bool = False,
    metadata: dict | None = None,
) -> dict:
    from google import genai
    from google.genai import types
    from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

    client = genai.Client(api_key=config.GEMINI_API_KEY)

    # Build generation config
    gen_config_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_tokens,
        "response_mime_type": "application/json",
    }
    if json_schema:
        gen_config_kwargs["response_schema"] = _strip_unsupported_schema_fields(json_schema)

    # Disable thinking for 2.5 flash models only — 2.5 pro requires thinking mode (budget>=128)
    if "2.5" in model and "pro" not in model.lower():
        gen_config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    gen_config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        **gen_config_kwargs,
    )

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            resp = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=gen_config,
            )
            latency = time.time() - t0

            raw_text = resp.text if resp.text else ""
            usage_meta = resp.usage_metadata
            usage = {
                "prompt_tokens": getattr(usage_meta, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(usage_meta, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(usage_meta, "total_token_count", 0) or 0,
                "cached_tokens": getattr(usage_meta, "cached_content_token_count", 0) or 0,
            }

            return _make_response(
                raw_text=raw_text,
                model=model,
                provider="gemini",
                usage=usage,
                response_id="",
                latency=latency,
                prompt_hash=_prompt_hash(system_prompt, user_prompt),
            )
        except ResourceExhausted as e:
            delay = _backoff(attempt)
            logger.warning(f"Gemini 429 (attempt {attempt+1}): retrying in {delay:.1f}s")
            time.sleep(delay)
        except ServiceUnavailable as e:
            if attempt < MAX_RETRIES - 1:
                delay = _backoff(attempt)
                logger.warning(f"Gemini unavailable (attempt {attempt+1}): retrying in {delay:.1f}s")
                time.sleep(delay)
            else:
                raise
        except Exception as e:
            err_msg = str(e).lower()
            if "schema" in err_msg or "field" in err_msg:
                logger.warning(f"Gemini schema error, retrying without response_schema: {e}")
                gen_config_kwargs.pop("response_schema", None)
                gen_config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    **gen_config_kwargs,
                )
                continue
            raise
    raise RuntimeError(f"Gemini: max retries ({MAX_RETRIES}) exceeded")


# ═══════════════════════════════════════════════════════
# Vertex AI — shared auth helper
# ═══════════════════════════════════════════════════════
def _get_vertex_token() -> str:
    """Get a fresh GCP access token using Application Default Credentials.
    Run 'gcloud auth application-default login' once before using Vertex providers.
    """
    import google.auth
    import google.auth.transport.requests
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


# ═══════════════════════════════════════════════════════
# Vertex AI — Llama 4 (OpenAI-compatible endpoint)
# ═══════════════════════════════════════════════════════
def _call_vertex_llama(
    system_prompt: str,
    user_prompt: str,
    model: str,
    json_schema: dict | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    store: bool = False,
    metadata: dict | None = None,
) -> dict:
    from openai import OpenAI, RateLimitError, APIError

    project_id = config.GCP_PROJECT_ID
    location   = config.VERTEX_LLAMA_LOCATION
    if not project_id:
        raise ValueError("GCP_PROJECT_ID is not set. Add it to your .env file.")

    base_url = (
        f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}"
        f"/locations/{location}/endpoints/openapi"
    )
    token = _get_vertex_token()
    client = OpenAI(base_url=base_url, api_key=token)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    kwargs: dict[str, Any] = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "top_p":       top_p,
        "max_tokens":  max_tokens,
    }
    if json_schema:
        # Llama on Vertex supports json_object mode; strict schema not yet available
        kwargs["response_format"] = {"type": "json_object"}
    else:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(**kwargs)
            latency = time.time() - t0

            usage = {
                "prompt_tokens":     resp.usage.prompt_tokens     if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "total_tokens":      resp.usage.total_tokens      if resp.usage else 0,
                "cached_tokens":     0,
            }
            return _make_response(
                raw_text=resp.choices[0].message.content or "",
                model=model,
                provider="vertex_llama",
                usage=usage,
                response_id=resp.id,
                latency=latency,
                prompt_hash=_prompt_hash(system_prompt, user_prompt),
            )
        except RateLimitError:
            delay = _backoff(attempt)
            logger.warning(f"Vertex/Llama 429 (attempt {attempt+1}): retrying in {delay:.1f}s")
            time.sleep(delay)
        except APIError as e:
            if attempt < MAX_RETRIES - 1 and getattr(e, "status_code", 0) >= 500:
                delay = _backoff(attempt)
                logger.warning(f"Vertex/Llama server error (attempt {attempt+1}): retrying in {delay:.1f}s")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"Vertex/Llama: max retries ({MAX_RETRIES}) exceeded")


# ═══════════════════════════════════════════════════════
# Vertex AI — Anthropic Claude (AnthropicVertex SDK)
# ═══════════════════════════════════════════════════════
def _call_vertex_anthropic(
    system_prompt: str,
    user_prompt: str,
    model: str,
    json_schema: dict | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    store: bool = False,
    metadata: dict | None = None,
) -> dict:
    from anthropic import AnthropicVertex, RateLimitError, APIError

    project_id = config.GCP_PROJECT_ID
    region     = config.VERTEX_ANTHROPIC_LOCATION
    if not project_id:
        raise ValueError("GCP_PROJECT_ID is not set. Add it to your .env file.")

    client = AnthropicVertex(project_id=project_id, region=region)

    messages = [{"role": "user", "content": user_prompt}]
    kwargs: dict[str, Any] = {
        "model":      model,
        "max_tokens": max_tokens,
        "system":     system_prompt,
        "messages":   messages,
        "temperature": temperature,
        "top_p":      top_p,
    }
    # JSON prefill trick — same as direct Anthropic
    if json_schema:
        messages.append({"role": "assistant", "content": "{"})

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            resp = client.messages.create(**kwargs)
            latency = time.time() - t0

            raw_text = resp.content[0].text if resp.content else ""
            if json_schema:
                raw_text = "{" + raw_text

            usage = {
                "prompt_tokens":     resp.usage.input_tokens  if resp.usage else 0,
                "completion_tokens": resp.usage.output_tokens if resp.usage else 0,
                "total_tokens":      (resp.usage.input_tokens + resp.usage.output_tokens) if resp.usage else 0,
                "cached_tokens":     getattr(resp.usage, "cache_read_input_tokens", 0) if resp.usage else 0,
            }
            return _make_response(
                raw_text=raw_text,
                model=resp.model,
                provider="vertex_anthropic",
                usage=usage,
                response_id=resp.id,
                latency=latency,
                prompt_hash=_prompt_hash(system_prompt, user_prompt),
            )
        except RateLimitError:
            delay = _backoff(attempt)
            logger.warning(f"Vertex/Anthropic 429 (attempt {attempt+1}): retrying in {delay:.1f}s")
            time.sleep(delay)
        except APIError as e:
            if attempt < MAX_RETRIES - 1 and getattr(e, "status_code", 0) >= 500:
                delay = _backoff(attempt)
                logger.warning(f"Vertex/Anthropic server error (attempt {attempt+1}): retrying in {delay:.1f}s")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"Vertex/Anthropic: max retries ({MAX_RETRIES}) exceeded")


# ═══════════════════════════════════════════════════════
# Vertex AI — DeepSeek (OpenAI-compatible global endpoint)
# ═══════════════════════════════════════════════════════
def _call_vertex_deepseek(
    system_prompt: str,
    user_prompt: str,
    model: str,
    json_schema: dict | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    store: bool = False,
    metadata: dict | None = None,
) -> dict:
    from openai import OpenAI, RateLimitError, APIError

    project_id = config.GCP_PROJECT_ID
    location   = config.VERTEX_DEEPSEEK_LOCATION
    if not project_id:
        raise ValueError("GCP_PROJECT_ID is not set. Add it to your .env file.")

    # Global endpoint uses aiplatform.googleapis.com (no location prefix in host)
    if location == "global":
        base_url = (
            f"https://aiplatform.googleapis.com/v1/projects/{project_id}"
            f"/locations/global/endpoints/openapi"
        )
    else:
        base_url = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}"
            f"/locations/{location}/endpoints/openapi"
        )

    token = _get_vertex_token()
    client = OpenAI(base_url=base_url, api_key=token)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    kwargs: dict[str, Any] = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "top_p":       top_p,
        "max_tokens":  max_tokens,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(**kwargs)
            latency = time.time() - t0

            usage = {
                "prompt_tokens":     resp.usage.prompt_tokens     if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "total_tokens":      resp.usage.total_tokens      if resp.usage else 0,
                "cached_tokens":     0,
            }
            return _make_response(
                raw_text=resp.choices[0].message.content or "",
                model=model,
                provider="vertex_deepseek",
                usage=usage,
                response_id=resp.id,
                latency=latency,
                prompt_hash=_prompt_hash(system_prompt, user_prompt),
            )
        except RateLimitError:
            delay = _backoff(attempt)
            logger.warning(f"Vertex/DeepSeek 429 (attempt {attempt+1}): retrying in {delay:.1f}s")
            time.sleep(delay)
        except APIError as e:
            if attempt < MAX_RETRIES - 1 and getattr(e, "status_code", 0) >= 500:
                delay = _backoff(attempt)
                logger.warning(f"Vertex/DeepSeek server error (attempt {attempt+1}): retrying in {delay:.1f}s")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"Vertex/DeepSeek: max retries ({MAX_RETRIES}) exceeded")


# ═══════════════════════════════════════════════════════
# Unified dispatcher
# ═══════════════════════════════════════════════════════
_PROVIDER_MAP = {
    "openai":           _call_openai,
    "anthropic":        _call_anthropic,
    "groq":             _call_groq,
    "gemini":           _call_gemini,
    # Vertex AI / Model Garden
    "vertex_llama":     _call_vertex_llama,
    "vertex_anthropic": _call_vertex_anthropic,
    "vertex_deepseek":  _call_vertex_deepseek,
}


def call_llm(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    json_schema: dict | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    store: bool = False,
    metadata: dict | None = None,
) -> dict:
    """Unified LLM call. Returns standardized response dict."""
    if provider not in _PROVIDER_MAP:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(_PROVIDER_MAP.keys())}")

    model = model or config.DEFAULT_MODELS.get(provider, "")
    temperature = temperature if temperature is not None else config.TEMPERATURE
    top_p = top_p if top_p is not None else config.TOP_P
    max_tokens = max_tokens or config.MAX_TOKENS

    fn = _PROVIDER_MAP[provider]
    return fn(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        json_schema=json_schema,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        store=store,
        metadata=metadata,
    )


def parse_json_response(response: dict) -> dict:
    """Parse the raw_text from an LLM response as JSON. Returns {} on failure."""
    raw = response.get("raw_text", "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
    logger.warning(f"Failed to parse JSON from response: {raw[:200]}")
    return {}
