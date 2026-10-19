"""Unit tests for NVIDIA NIM hosted extractor (Nemotron 3.5 Lightning, Nemotron 3.5 Super VL, GPT-OSS) and dispatcher integration."""

import base64
import json
from unittest.mock import MagicMock, patch
import pytest
import openai

from src.config import DEFAULT_NVIDIA_MODEL, NVIDIA_BASE_URL, SUPPORTED_NVIDIA_MODELS
from src.schemas import ExtractionResult
from src.nvidia_extractor import (
    NvidiaExtractor,
    get_configured_nvidia_api_key,
    get_nvidia_model_config,
    sanitize_error_message,
    NVIDIA_MODEL_CONFIGS,
)
from src.extractor_dispatcher import extract_with_provider


RECEIPT_SAMPLE_TEXT = """
AEON CO. (M) BHD
LOT 12, JALAN MEDAN 1, 46000 PETALING JAYA
DATE: 27/04/2018
TOTAL RM 42.10
CASH RM 50.00
CHANGE RM 7.90
"""

INVOICE_SAMPLE_TEXT = """
Acme Solutions LLC
Invoice Number: INV-2024-889
Invoice Date: 2024-03-15
Due Date: 2024-04-15
Subtotal: $500.00
Tax: $50.00
Total Amount: $550.00
"""

DUMMY_IMAGE_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb"


def test_get_configured_nvidia_api_key(monkeypatch):
    """Test API key resolution order: explicit > env > None."""
    # 1. Explicit key takes precedence
    assert get_configured_nvidia_api_key("nvapi-explicit-123") == "nvapi-explicit-123"

    # 2. Environment variable
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-env-key-456")
    assert get_configured_nvidia_api_key() == "nvapi-env-key-456"

    # 3. None when unset
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    assert get_configured_nvidia_api_key() is None


def test_nvidia_sanitize_error_message():
    """Verify that credentials, nvapi- tokens, gsk_ tokens, and Bearer tokens are redacted."""
    raw_error = "Failed to authenticate with nvapi-abcdef1234567890 and Bearer secret_nv_token"
    sanitized = sanitize_error_message(raw_error, key_to_mask="nvapi-abcdef1234567890")

    assert "nvapi-abcdef1234567890" not in sanitized
    assert "secret_nv_token" not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized
    assert "Bearer [REDACTED]" in sanitized

    # Groq token test in NVIDIA sanitizer
    groq_err = "Auth error for gsk_9876543210zyxwvutsrqponmlkjihg"
    sanitized_groq = sanitize_error_message(groq_err)
    assert "gsk_9876543210" not in sanitized_groq
    assert "[REDACTED_API_KEY]" in sanitized_groq


def test_nvidia_supported_models_list():
    """Verify required models are registered in configuration."""
    assert "nvidia/nemotron-3.5-lightning-30b-a3b" in SUPPORTED_NVIDIA_MODELS
    assert "nvidia/nemotron-3.5-super-vl-120b-a12b" in SUPPORTED_NVIDIA_MODELS
    assert "openai/gpt-oss-20b" in SUPPORTED_NVIDIA_MODELS


def test_nvidia_model_configs_parameters():
    """Verify model-specific runtime parameters: timeout, token limits, and vision capabilities."""
    lightning_cfg = get_nvidia_model_config("nvidia/nemotron-3.5-lightning-30b-a3b")
    assert not lightning_cfg["is_vision"]
    assert lightning_cfg["timeout"] == 45.0
    assert lightning_cfg["max_tokens"] == 2048
    assert lightning_cfg["supports_thinking_toggle"] is True

    super_vl_cfg = get_nvidia_model_config("nvidia/nemotron-3.5-super-vl-120b-a12b")
    assert super_vl_cfg["is_vision"] is True
    assert super_vl_cfg["timeout"] == 90.0
    assert super_vl_cfg["max_tokens"] == 4096
    assert super_vl_cfg["temperature"] == 0.2
    assert super_vl_cfg["supports_thinking_toggle"] is True

    fallback_cfg = get_nvidia_model_config("openai/gpt-oss-20b")
    assert not fallback_cfg["is_vision"]
    assert fallback_cfg["timeout"] == 30.0
    assert fallback_cfg["max_tokens"] == 1024
    assert fallback_cfg["supports_thinking_toggle"] is False


def test_nvidia_offline_fallback_when_api_key_missing():
    """Verify that when no NVIDIA key exists, extractor falls back cleanly without crashing."""
    extractor = NvidiaExtractor(api_key="")
    assert not extractor.is_available()

    result = extractor.extract(RECEIPT_SAMPLE_TEXT, doc_type="receipt", doc_id="test_rcpt_nv_01")

    assert isinstance(result, ExtractionResult)
    assert result.source == "baseline_fallback"
    assert result.provider == "regex"
    assert result.model_used is None
    assert "NVIDIA API key not configured" in result.error_message
    assert result.fields["total"] == "42.10"
    assert "AEON" in result.fields["company"]


def test_nvidia_forced_fallback_mode():
    """Verify explicit offline testing via force_fallback=True."""
    extractor = NvidiaExtractor(api_key="nvapi-dummy-valid")
    result = extractor.extract(INVOICE_SAMPLE_TEXT, doc_type="invoice", doc_id="test_inv_nv_01", force_fallback=True)

    assert result.source == "baseline_fallback"
    assert result.provider == "regex"
    assert result.model_used is None
    assert result.fields["total_amount"] == 550.0


def test_nvidia_client_initialization_base_url_and_timeout():
    """Verify OpenAI client is instantiated with the NVIDIA base URL and model-appropriate timeout."""
    with patch("openai.OpenAI") as mock_openai_cls:
        # Default lightning model timeout
        extractor = NvidiaExtractor(api_key="nvapi-test-key", model="nvidia/nemotron-3.5-lightning-30b-a3b")
        _ = extractor.client
        mock_openai_cls.assert_called_with(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="nvapi-test-key",
            timeout=45.0,
        )

        # Super VL model timeout (generous 90s for 120B parameter model)
        extractor_vl = NvidiaExtractor(api_key="nvapi-test-key", model="nvidia/nemotron-3.5-super-vl-120b-a12b")
        _ = extractor_vl.client
        mock_openai_cls.assert_called_with(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="nvapi-test-key",
            timeout=90.0,
        )

        # Fallback GPT-OSS-20B timeout
        extractor_fallback = NvidiaExtractor(api_key="nvapi-test-key", model="openai/gpt-oss-20b")
        _ = extractor_fallback.client
        mock_openai_cls.assert_called_with(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="nvapi-test-key",
            timeout=30.0,
        )


def test_mocked_nvidia_lightning_ocr_text_receipt_extraction():
    """Test Requirement 4: Test Nemotron 3.5 Lightning using the existing OCR-text extraction pipeline."""
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "company": "AEON CO. (M) BHD",
        "date": "2018-04-27",
        "address": "LOT 12, JALAN MEDAN 1, 46000 PETALING JAYA",
        "total": 42.10,
    })
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_cls.return_value = mock_client

        extractor = NvidiaExtractor(api_key="nvapi-test-key-valid", model="nvidia/nemotron-3.5-lightning-30b-a3b")
        res = extractor.extract(RECEIPT_SAMPLE_TEXT, doc_type="receipt", doc_id="mock_rcpt_lightning_01")

        assert res.source == "nvidia"
        assert res.provider == "nvidia"
        assert res.model_used == "nvidia/nemotron-3.5-lightning-30b-a3b"
        assert res.error_message is None
        assert res.fields["company"] == "AEON CO. (M) BHD"
        assert res.fields["total"] == 42.10
        assert res.fields["date"] == "2018-04-27"
        assert res.metadata["input_mode"] == "ocr_text_only"

        # Verify request parameters
        create_call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert create_call_kwargs["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
        assert create_call_kwargs["response_format"] == {"type": "json_object"}
        assert create_call_kwargs["max_tokens"] == 2048
        assert create_call_kwargs["temperature"] == 0.0
        assert create_call_kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
        # Verify message content is OCR text string
        assert create_call_kwargs["messages"][1]["content"] == f"DOCUMENT TEXT:\n{RECEIPT_SAMPLE_TEXT}"


def test_mocked_nvidia_lightning_ocr_text_invoice_extraction():
    """Test Nemotron 3.5 Lightning extracting from structured invoice OCR text."""
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "vendor_name": "Acme Solutions LLC",
        "invoice_number": "INV-2024-889",
        "invoice_date": "2024-03-15",
        "due_date": "2024-04-15",
        "subtotal": 500.0,
        "tax_amount": 50.0,
        "discount_amount": 0.0,
        "total_amount": 550.0,
        "currency": "USD",
    })
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_cls.return_value = mock_client

        extractor = NvidiaExtractor(api_key="nvapi-test-key-valid", model="nvidia/nemotron-3.5-lightning-30b-a3b")
        res = extractor.extract(INVOICE_SAMPLE_TEXT, doc_type="invoice", doc_id="mock_inv_lightning_01")

        assert res.source == "nvidia"
        assert res.provider == "nvidia"
        assert res.fields["vendor_name"] == "Acme Solutions LLC"
        assert res.fields["subtotal"] == 500.0
        assert res.fields["total_amount"] == 550.0
        assert res.metadata["input_mode"] == "ocr_text_only"


def test_mocked_nvidia_super_vl_multimodal_request_construction_with_image():
    """Test Requirement 5: When image content is provided, Super VL constructs a multimodal message."""
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "company": "VISUAL STORE NIM",
        "date": "2024-07-01",
        "address": "456 Vision Way",
        "total": 77.25,
    })
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_cls.return_value = mock_client

        extractor = NvidiaExtractor(api_key="nvapi-test-key-valid", model="nvidia/nemotron-3.5-super-vl-120b-a12b")
        res = extractor.extract(
            text=RECEIPT_SAMPLE_TEXT,
            doc_type="receipt",
            doc_id="mock_super_vl_multimodal",
            image_bytes=DUMMY_IMAGE_BYTES,
        )

        assert res.source == "nvidia"
        assert res.model_used == "nvidia/nemotron-3.5-super-vl-120b-a12b"
        assert res.fields["company"] == "VISUAL STORE NIM"
        assert res.fields["total"] == 77.25
        assert res.metadata["input_mode"] == "multimodal_visual_text"
        assert res.metadata["is_vision_model"] is True

        # Verify request parameters
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "nvidia/nemotron-3.5-super-vl-120b-a12b"
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_tokens"] == 4096

        # Verify user message contains image_url
        user_msg = call_kwargs["messages"][1]["content"]
        assert isinstance(user_msg, list)
        assert any(item.get("type") == "image_url" for item in user_msg)
        assert any("data:image/jpeg;base64," in item.get("image_url", {}).get("url", "") for item in user_msg)


def test_mocked_nvidia_super_vl_text_only_when_no_image_provided():
    """Test Requirement 5: When image content is missing, Super VL does NOT claim visual extraction."""
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "company": "TEXT ONLY NIM",
        "date": "2024-07-02",
        "address": "789 Text Road",
        "total": 35.00,
    })
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_cls.return_value = mock_client

        extractor = NvidiaExtractor(api_key="nvapi-test-key-valid", model="nvidia/nemotron-3.5-super-vl-120b-a12b")
        res = extractor.extract(
            text=RECEIPT_SAMPLE_TEXT,
            doc_type="receipt",
            doc_id="mock_super_vl_text_only",
            image_bytes=None,  # No image passed
        )

        assert res.source == "nvidia"
        assert res.metadata["input_mode"] == "ocr_text_only"
        assert "visual_note" in res.metadata
        assert "Text-only extraction" in res.metadata["visual_note"]

        # Verify user message is standard string, NOT an image_url
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        user_msg = call_kwargs["messages"][1]["content"]
        assert isinstance(user_msg, str)
        assert "DOCUMENT TEXT:" in user_msg


def test_mocked_nvidia_lightning_ignores_image_bytes():
    """Verify text-only models (e.g. Lightning) never send image_url even if image_bytes are passed."""
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "company": "LIGHTNING STORE",
        "date": "2024-07-03",
        "address": "123 Lightning St",
        "total": 19.99,
    })
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_cls.return_value = mock_client

        extractor = NvidiaExtractor(api_key="nvapi-test-key-valid", model="nvidia/nemotron-3.5-lightning-30b-a3b")
        res = extractor.extract(
            text=RECEIPT_SAMPLE_TEXT,
            doc_type="receipt",
            doc_id="mock_lightning_with_image_param",
            image_bytes=DUMMY_IMAGE_BYTES,
        )

        assert res.metadata["input_mode"] == "ocr_text_only"
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        user_msg = call_kwargs["messages"][1]["content"]
        assert isinstance(user_msg, str)
        assert "image_url" not in str(user_msg)


def test_nvidia_extra_body_retry_on_bad_request():
    """Verify that if server rejects chat_template_kwargs with 400 Bad Request, it retries without extra_body."""
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "company": "RETRY SUCCESS STORE",
        "date": "2024-07-04",
        "address": "999 Retry Ave",
        "total": 45.00,
    })
    mock_success_completion = MagicMock()
    mock_success_completion.choices = [mock_choice]

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        # First call fails on extra_body, second call succeeds
        mock_client.chat.completions.create.side_effect = [
            openai.BadRequestError(
                message="Unrecognized extra_body parameter: chat_template_kwargs",
                response=mock_resp,
                body={"error": {"message": "Unrecognized extra_body parameter"}},
            ),
            mock_success_completion,
        ]
        mock_openai_cls.return_value = mock_client

        extractor = NvidiaExtractor(api_key="nvapi-valid", model="nvidia/nemotron-3.5-lightning-30b-a3b")
        res = extractor.extract(RECEIPT_SAMPLE_TEXT, doc_type="receipt", doc_id="test_retry_01")

        assert res.source == "nvidia"
        assert res.fields["company"] == "RETRY SUCCESS STORE"
        assert mock_client.chat.completions.create.call_count == 2
        # Verify second call dropped extra_body
        second_call_kwargs = mock_client.chat.completions.create.call_args_list[1][1]
        assert "extra_body" not in second_call_kwargs


def test_nvidia_authentication_error_fallback():
    """Verify that AuthenticationError triggers fallback and redacts keys."""
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
            message="Invalid API Key provided: nvapi-secret-1234567890",
            response=mock_response,
            body={"error": {"message": "Invalid API Key provided: nvapi-secret-1234567890"}},
        )
        mock_openai_cls.return_value = mock_client

        extractor = NvidiaExtractor(api_key="nvapi-secret-1234567890")
        res = extractor.extract(RECEIPT_SAMPLE_TEXT, doc_type="receipt", doc_id="auth_fail_nv_01")

        assert res.source == "baseline_fallback"
        assert res.provider == "regex"
        assert res.model_used is None
        assert "AuthenticationError" in res.error_message
        assert "nvapi-secret-1234567890" not in res.error_message
        assert "[REDACTED_API_KEY]" in res.error_message
        assert res.fields["total"] == "42.10"


def test_nvidia_rate_limit_error_fallback():
    """Verify that RateLimitError triggers fallback to deterministic baseline."""
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            message="NVIDIA NIM rate limit reached",
            response=mock_response,
            body={"error": {"message": "Rate limit reached"}},
        )
        mock_openai_cls.return_value = mock_client

        extractor = NvidiaExtractor(api_key="nvapi-valid")
        res = extractor.extract(RECEIPT_SAMPLE_TEXT, doc_type="receipt", doc_id="rate_fail_nv_01")

        assert res.source == "baseline_fallback"
        assert res.provider == "regex"
        assert res.model_used is None
        assert "RateLimitError" in res.error_message
        assert res.fields["total"] is not None


def test_nvidia_not_found_error_fallback():
    """Verify that NotFoundError (model not found / unavailable) triggers fallback with diagnostics."""
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_client.chat.completions.create.side_effect = openai.NotFoundError(
            message="Model nvidia/nemotron-3.5-lightning-30b-a3b does not exist or account is not entitled",
            response=mock_response,
            body={"error": {"message": "Model not found"}},
        )
        mock_openai_cls.return_value = mock_client

        extractor = NvidiaExtractor(api_key="nvapi-valid", model="nvidia/nemotron-3.5-lightning-30b-a3b")
        res = extractor.extract(RECEIPT_SAMPLE_TEXT, doc_type="receipt", doc_id="not_found_nv_01")

        assert res.source == "baseline_fallback"
        assert res.provider == "regex"
        assert res.model_used is None
        assert "NotFoundError" in res.error_message
        assert "Switched to fallback baseline extractor" in res.error_message
        assert res.fields["total"] == "42.10"


def test_nvidia_json_decode_error_fallback():
    """Verify that invalid non-JSON output triggers fallback rather than crashing."""
    mock_choice = MagicMock()
    mock_choice.message.content = "Here is the raw text without valid JSON format."
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_cls.return_value = mock_client

        extractor = NvidiaExtractor(api_key="nvapi-valid")
        res = extractor.extract(RECEIPT_SAMPLE_TEXT, doc_type="receipt", doc_id="bad_json_nv_01")

        assert res.source == "baseline_fallback"
        assert res.provider == "regex"
        assert res.model_used is None
        assert "JSONDecodeError" in res.error_message


def test_dispatcher_explicit_nvidia_mode():
    """Dispatcher with mode='nvidia' routes to NvidiaExtractor with specified model."""
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "company": "NVIDIA DISPATCH TEST",
        "date": "2024-06-01",
        "address": "2788 San Tomas Expressway, Santa Clara, CA",
        "total": 88.50,
    })
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_cls.return_value = mock_client

        res = extract_with_provider(
            text=RECEIPT_SAMPLE_TEXT,
            doc_type="receipt",
            doc_id="disp_nvidia_01",
            mode="nvidia",
            nvidia_api_key="nvapi-valid-test",
            nvidia_model="nvidia/nemotron-3.5-lightning-30b-a3b",
        )

        assert res.source == "nvidia"
        assert res.provider == "nvidia"
        assert res.model_used == "nvidia/nemotron-3.5-lightning-30b-a3b"
        assert res.fields["company"] == "NVIDIA DISPATCH TEST"
        assert res.fields["total"] == 88.50


def test_dispatcher_explicit_baseline_mode():
    """Dispatcher with mode='baseline' runs deterministic regex without calling hosted LLM."""
    res = extract_with_provider(
        text=RECEIPT_SAMPLE_TEXT,
        doc_type="receipt",
        doc_id="disp_base_01",
        mode="baseline",
        nvidia_api_key="nvapi-some-key",
    )
    assert res.source == "baseline"
    assert res.provider == "regex"
    assert res.model_used is None
    assert res.fields["total"] == "42.10"


def test_dispatcher_auto_mode_nvidia_success():
    """In Auto mode, when NVIDIA is configured and valid, it extracts successfully."""
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "company": "NVIDIA AUTO SUCCESS",
        "date": "2024-06-02",
        "address": "123 Silicon Way",
        "total": 150.00,
    })
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_cls.return_value = mock_client

        res = extract_with_provider(
            text=RECEIPT_SAMPLE_TEXT,
            doc_type="receipt",
            doc_id="auto_nvidia_01",
            mode="auto",
            nvidia_api_key="nvapi-present",
        )

        assert res.source == "nvidia"
        assert res.provider == "nvidia"
        assert res.fields["company"] == "NVIDIA AUTO SUCCESS"
        assert res.fields["total"] == 150.00


def test_dispatcher_auto_mode_key_missing_falls_back_to_baseline(monkeypatch):
    """In Auto mode when no NVIDIA key is provided or in env, routes cleanly to offline baseline."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    res = extract_with_provider(
        text=RECEIPT_SAMPLE_TEXT,
        doc_type="receipt",
        doc_id="auto_no_key_01",
        mode="auto",
        nvidia_api_key=None,
    )

    assert res.source == "baseline"
    assert res.provider == "regex"
    assert res.model_used is None
    assert "NVIDIA API key not configured" in res.error_message
    assert res.fields["total"] == "42.10"


def test_dispatcher_auto_mode_nvidia_failure_falls_back_to_baseline():
    """In Auto mode, when NVIDIA NIM fails (e.g. rate limit), safely falls back to baseline with diagnostics."""
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            message="NVIDIA rate limit exceeded",
            response=mock_resp,
            body={"error": {"message": "Rate limit"}},
        )
        mock_openai_cls.return_value = mock_client

        res = extract_with_provider(
            text=RECEIPT_SAMPLE_TEXT,
            doc_type="receipt",
            doc_id="auto_fallback_nvidia_to_baseline",
            mode="auto",
            nvidia_api_key="nvapi-failing",
        )

        assert res.source == "baseline_fallback"
        assert res.provider == "regex"
        assert res.model_used is None
        assert "Auto mode: NVIDIA NIM failed" in res.error_message
        assert res.fields["total"] == "42.10"


def test_dispatcher_no_other_provider_supported():
    """Verify no provider other than NVIDIA or baseline is selected; obsolete modes route to baseline."""
    res_unknown = extract_with_provider(
        text=RECEIPT_SAMPLE_TEXT,
        doc_type="receipt",
        doc_id="test_obsolete_mode",
        mode="groq",  # obsolete provider mode
        nvidia_api_key=None,
    )
    assert res_unknown.source == "baseline"
    assert res_unknown.provider == "regex"
    assert res_unknown.model_used is None
    assert res_unknown.fields["total"] == "42.10"

    # Even if an NVIDIA key is present, obsolete/unsupported modes must route directly to baseline
    res_obsolete_with_key = extract_with_provider(
        text=RECEIPT_SAMPLE_TEXT,
        doc_type="receipt",
        doc_id="test_obsolete_mode_with_key",
        mode="gemini",
        nvidia_api_key="nvapi-valid-dummy-key",
    )
    assert res_obsolete_with_key.source == "baseline"
    assert res_obsolete_with_key.provider == "regex"
    assert res_obsolete_with_key.model_used is None
    assert res_obsolete_with_key.fields["total"] == "42.10"


def test_nvidia_timeout_error_fallback():
    """Verify that APITimeoutError triggers safe fallback to baseline with sanitized error."""
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = openai.APITimeoutError(
            request=MagicMock()
        )
        mock_openai_cls.return_value = mock_client

        extractor = NvidiaExtractor(api_key="nvapi-valid", model="nvidia/nemotron-3.5-lightning-30b-a3b")
        res = extractor.extract(RECEIPT_SAMPLE_TEXT, doc_type="receipt", doc_id="timeout_nv_01")

        assert res.source == "baseline_fallback"
        assert res.provider == "regex"
        assert res.model_used is None
        assert "APITimeoutError" in res.error_message
        assert res.fields["total"] == "42.10"
