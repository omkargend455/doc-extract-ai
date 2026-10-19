"""Hosted LLM extractor using NVIDIA NIM OpenAI-compatible API (https://integrate.api.nvidia.com/v1)."""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Optional

import openai
from pydantic import ValidationError

from src.config import DEFAULT_NVIDIA_MODEL, NVIDIA_BASE_URL
from src.baseline_regex import RegexBaselineExtractor
from src.schemas import ExtractionResult, InvoiceGroundTruth, ReceiptGroundTruth


NVIDIA_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "nvidia/nemotron-3.5-lightning-30b-a3b": {
        "is_vision": False,
        "timeout": 45.0,
        "max_tokens": 2048,
        "temperature": 0.0,
        "supports_thinking_toggle": True,
        "description": "High-throughput hybrid Mamba-MoE text extractor (30B total, 3B active)",
    },
    "nvidia/nemotron-3.5-super-vl-120b-a12b": {
        "is_vision": True,
        "timeout": 90.0,
        "max_tokens": 4096,
        "temperature": 0.2,
        "supports_thinking_toggle": True,
        "description": "Multimodal Vision-Language Mamba-MoE extractor (120B total, 12B active)",
    },
    "openai/gpt-oss-20b": {
        "is_vision": False,
        "timeout": 30.0,
        "max_tokens": 1024,
        "temperature": 0.0,
        "supports_thinking_toggle": False,
        "description": "Hosted fallback text extractor",
    },
    "openai/gpt-oss-120b": {
        "is_vision": False,
        "timeout": 60.0,
        "max_tokens": 2048,
        "temperature": 0.0,
        "supports_thinking_toggle": False,
        "description": "Hosted fallback 120B text extractor",
    },
}


def get_nvidia_model_config(model_id: str) -> dict[str, Any]:
    """Retrieve runtime API parameters, timeout, and vision capabilities for a specific model."""
    if model_id in NVIDIA_MODEL_CONFIGS:
        return NVIDIA_MODEL_CONFIGS[model_id]

    is_vl = any(term in model_id.lower() for term in ["-vl", "vision", "multimodal"])
    is_nemotron = "nemotron" in model_id.lower()
    return {
        "is_vision": is_vl,
        "timeout": 90.0 if is_vl else 45.0,
        "max_tokens": 4096 if is_vl else 2048,
        "temperature": 0.2 if is_vl else 0.0,
        "supports_thinking_toggle": is_nemotron,
        "description": "Custom NVIDIA model",
    }


NVIDIA_RECEIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": ["string", "null"], "description": "Legal company or store name"},
        "date": {"type": ["string", "null"], "description": "Transaction date in YYYY-MM-DD or original format"},
        "address": {"type": ["string", "null"], "description": "Full street address of the store"},
        "total": {"type": ["number", "null"], "description": "Final total amount paid as a numeric float"},
    },
    "required": ["company", "date", "total"],
}

NVIDIA_INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor_name": {"type": ["string", "null"], "description": "Name of issuing company or vendor"},
        "invoice_number": {"type": ["string", "null"], "description": "Invoice or bill reference number"},
        "invoice_date": {"type": ["string", "null"], "description": "Issue date in YYYY-MM-DD format"},
        "due_date": {"type": ["string", "null"], "description": "Payment due date in YYYY-MM-DD format"},
        "subtotal": {
            "type": ["number", "null"],
            "description": "Subtotal representing pre-tax taxable value (base amount before taxes and discounts)",
        },
        "tax_amount": {
            "type": ["number", "null"],
            "description": "Total tax amount (mapped from sum of SGST+CGST or IGST, or general tax)",
        },
        "sgst_amount": {"type": ["number", "null"], "description": "SGST / UTGST tax component"},
        "cgst_amount": {"type": ["number", "null"], "description": "CGST tax component"},
        "igst_amount": {"type": ["number", "null"], "description": "IGST tax component"},
        "discount_amount": {"type": ["number", "null"], "description": "Discount amount applied"},
        "total_amount": {
            "type": ["number", "null"],
            "description": "Grand total payable (tax-inclusive printed grand total)",
        },
        "currency": {"type": ["string", "null"], "description": "Currency code (USD, EUR, MYR, INR, GBP)"},
    },
    "required": ["vendor_name", "total_amount"],
}


def sanitize_error_message(msg: str, key_to_mask: Optional[str] = None) -> str:
    """Mask credentials, bearer tokens, or sensitive patterns from error messages."""
    sanitized = str(msg)
    if key_to_mask and len(key_to_mask) > 4:
        sanitized = sanitized.replace(key_to_mask, "[REDACTED_API_KEY]")
    # Mask nvapi-... keys, gsk-... keys, Google keys, or general bearer tokens
    sanitized = re.sub(r"nvapi-[A-Za-z0-9_\-]+", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"gsk_[A-Za-z0-9_\-]+", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"AIza[0-9A-Za-z_\-]+", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer [REDACTED]", sanitized, flags=re.IGNORECASE)
    return sanitized


def get_configured_nvidia_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    """Retrieve NVIDIA API key from argument, environment variables, or Streamlit secrets.

    Never logs, persists, or commits credentials.
    """
    if explicit_key is not None:
        return explicit_key.strip() if explicit_key.strip() else None

    env_key = os.environ.get("NVIDIA_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    try:
        import streamlit as st
        if hasattr(st, "secrets") and "NVIDIA_API_KEY" in st.secrets:
            sec_key = st.secrets["NVIDIA_API_KEY"]
            if sec_key and str(sec_key).strip():
                return str(sec_key).strip()
    except Exception:
        pass

    return None


class NvidiaExtractor:
    """Hosted LLM extractor using NVIDIA NIM OpenAI-compatible endpoint with structured JSON output and baseline fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.api_key = get_configured_nvidia_api_key(api_key)
        self.model = model or os.environ.get("NVIDIA_MODEL") or DEFAULT_NVIDIA_MODEL
        self.base_url = base_url or os.environ.get("NVIDIA_BASE_URL") or NVIDIA_BASE_URL
        self.baseline_extractor = RegexBaselineExtractor()
        self.custom_timeout = timeout

    def get_client(self, timeout: Optional[float] = None) -> openai.OpenAI:
        """Return an initialized OpenAI client targeting the NVIDIA NIM base URL with model-appropriate timeout."""
        model_cfg = get_nvidia_model_config(self.model)
        eff_timeout = (
            timeout
            if timeout is not None
            else (self.custom_timeout if self.custom_timeout is not None else model_cfg["timeout"])
        )
        return openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=eff_timeout,
        )

    @property
    def client(self) -> openai.OpenAI:
        """Return an initialized OpenAI client targeting the NVIDIA NIM base URL."""
        return self.get_client()

    def is_available(self) -> bool:
        """Check whether a valid NVIDIA API key is configured."""
        return bool(self.api_key and self.api_key.strip())

    def _clean_json_response(self, text: str) -> str:
        """Strip markdown code fence wrappers from model JSON response."""
        s = text.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\s*```$", "", s)
        return s.strip()

    def _validate_against_schema(self, data: dict[str, Any], doc_type: str, doc_id: str) -> dict[str, Any]:
        """Validate parsed dictionary against project Pydantic ground-truth models without fabricating values."""
        if doc_type == "invoice":
            temp_dict = {
                "doc_id": doc_id,
                "doc_type": "invoice",
                "vendor_name": data.get("vendor_name"),
                "invoice_number": data.get("invoice_number"),
                "invoice_date": data.get("invoice_date"),
                "due_date": data.get("due_date"),
                "subtotal": data.get("subtotal"),
                "tax_amount": data.get("tax_amount"),
                "sgst_amount": data.get("sgst_amount"),
                "cgst_amount": data.get("cgst_amount"),
                "igst_amount": data.get("igst_amount"),
                "discount_amount": data.get("discount_amount", 0.0),
                "total_amount": data.get("total_amount"),
                "currency": data.get("currency") or "USD",
            }
            try:
                validated = InvoiceGroundTruth(**temp_dict)
                return {
                    "vendor_name": validated.vendor_name,
                    "invoice_number": validated.invoice_number,
                    "invoice_date": validated.invoice_date,
                    "due_date": validated.due_date,
                    "subtotal": validated.subtotal,
                    "tax_amount": validated.tax_amount,
                    "sgst_amount": validated.sgst_amount,
                    "cgst_amount": validated.cgst_amount,
                    "igst_amount": validated.igst_amount,
                    "discount_amount": validated.discount_amount,
                    "total_amount": validated.total_amount,
                    "currency": validated.currency,
                }
            except ValidationError:
                return data
        else:
            temp_dict = {
                "doc_id": doc_id,
                "doc_type": "receipt",
                "company": data.get("company"),
                "date": data.get("date"),
                "address": data.get("address"),
                "total": data.get("total"),
            }
            try:
                validated = ReceiptGroundTruth(**temp_dict)
                return {
                    "company": validated.company,
                    "date": validated.date,
                    "address": validated.address,
                    "total": validated.total,
                }
            except ValidationError:
                return data

    def extract(
        self,
        text: str,
        doc_type: str = "receipt",
        doc_id: str = "doc_unknown",
        force_fallback: bool = False,
        image_bytes: Optional[bytes] = None,
    ) -> ExtractionResult:
        """Extract structured fields using NVIDIA NIM hosted LLM or fallback to deterministic baseline."""
        # 1. Offline or missing API key
        if force_fallback or not self.is_available():
            baseline_fields = self.baseline_extractor.extract(text, doc_type=doc_type)
            reason = (
                "Offline mode forced."
                if force_fallback
                else "NVIDIA API key not configured. Used deterministic rule-based extractor."
            )
            return ExtractionResult(
                doc_id=doc_id,
                doc_type=doc_type,
                source="baseline_fallback",
                provider="regex",
                model_used=None,
                fields=baseline_fields,
                raw_text=text,
                error_message=reason,
                metadata={"input_mode": "offline_fallback"},
            )

        # 2. Retrieve model-specific configuration
        cfg = get_nvidia_model_config(self.model)
        is_vision = cfg["is_vision"]
        temperature = cfg["temperature"]
        max_tokens = cfg["max_tokens"]
        supports_thinking = cfg["supports_thinking_toggle"]

        # 3. Prepare JSON prompt and schema
        schema = NVIDIA_INVOICE_SCHEMA if doc_type == "invoice" else NVIDIA_RECEIPT_SCHEMA
        system_instruction = (
            f"You are a precise document extraction engine. Analyze the provided {doc_type} and "
            f"extract all relevant fields into a strictly valid JSON object adhering to this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Rules:\n"
            "- Return ONLY valid JSON; no markdown formatting, no commentary, no markdown code blocks.\n"
            "- Do NOT fabricate or hallucinate missing information. Use null for any field not present in the document.\n"
            "- Ensure monetary values are numeric floats without currency symbols.\n"
            "- Normalize dates to ISO-8601 (YYYY-MM-DD) format whenever unambiguous.\n"
            "- Subtotal Definition: For invoices, subtotal is consistently defined as the pre-tax taxable value "
            "(the net taxable base amount upon which taxes are assessed, before taxes and after line discounts). "
            "For tax-inclusive invoices (such as Indian GST invoices), subtotal is the Taxable Value, NOT the gross or final total.\n"
            "- For GST invoices, extract SGST/UTGST and CGST separately into sgst_amount and cgst_amount where present, "
            "and map their sum to tax_amount. Do NOT mistake an individual line item's tax for the invoice's total tax.\n"
            "- The printed grand total is tax-inclusive; do NOT add tax to total_amount again."
        )

        # 4. Construct user message (multimodal visual+text if supported and image provided, else text-only)
        has_visual_input = bool(image_bytes and is_vision)
        if has_visual_input:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            user_content: list[dict[str, Any]] | str = [
                {
                    "type": "text",
                    "text": (
                        f"DOCUMENT OCR TEXT (if available):\n{text}\n\n"
                        f"Please visually inspect the attached document image and extract all {doc_type} fields strictly according to the schema."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    },
                },
            ]
        else:
            user_content = f"DOCUMENT TEXT:\n{text}"

        request_params: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if supports_thinking:
            request_params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        try:
            client = self.client

            # Request JSON mode output with fallback retry if extra_body is rejected by server
            try:
                response = client.chat.completions.create(**request_params)
            except openai.BadRequestError as bad_req:
                if "extra_body" in request_params:
                    retry_params = dict(request_params)
                    retry_params.pop("extra_body", None)
                    response = client.chat.completions.create(**retry_params)
                else:
                    raise bad_req

            raw_response_text = response.choices[0].message.content or "{}"
            cleaned_json = self._clean_json_response(raw_response_text)
            parsed_raw = json.loads(cleaned_json)

            # Validate against Pydantic schema
            parsed_fields = self._validate_against_schema(parsed_raw, doc_type=doc_type, doc_id=doc_id)

            input_mode = "multimodal_visual_text" if has_visual_input else "ocr_text_only"
            result_meta = {
                "input_mode": input_mode,
                "is_vision_model": is_vision,
                "model_description": cfg.get("description", ""),
            }
            if is_vision and not has_visual_input:
                result_meta["visual_note"] = "Text-only extraction. No visual image input was sent to the model."

            return ExtractionResult(
                doc_id=doc_id,
                doc_type=doc_type,
                source="nvidia",
                provider="nvidia",
                model_used=self.model,
                fields=parsed_fields,
                raw_text=text,
                error_message=None,
                metadata=result_meta,
            )

        except openai.AuthenticationError as exc:
            sanitized = sanitize_error_message(str(exc), self.api_key)
            baseline_fields = self.baseline_extractor.extract(text, doc_type=doc_type)
            return ExtractionResult(
                doc_id=doc_id,
                doc_type=doc_type,
                source="baseline_fallback",
                provider="regex",
                model_used=None,
                fields=baseline_fields,
                raw_text=text,
                error_message=f"NVIDIA authentication failed (AuthenticationError: {sanitized}). Switched to fallback baseline extractor.",
            )

        except (openai.NotFoundError, openai.BadRequestError) as exc:
            sanitized = sanitize_error_message(str(exc), self.api_key)
            baseline_fields = self.baseline_extractor.extract(text, doc_type=doc_type)
            return ExtractionResult(
                doc_id=doc_id,
                doc_type=doc_type,
                source="baseline_fallback",
                provider="regex",
                model_used=None,
                fields=baseline_fields,
                raw_text=text,
                error_message=f"NVIDIA model '{self.model}' unavailable or invalid request ({type(exc).__name__}: {sanitized}). Switched to fallback baseline extractor.",
            )

        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.APIError) as exc:
            sanitized = sanitize_error_message(str(exc), self.api_key)
            baseline_fields = self.baseline_extractor.extract(text, doc_type=doc_type)
            return ExtractionResult(
                doc_id=doc_id,
                doc_type=doc_type,
                source="baseline_fallback",
                provider="regex",
                model_used=None,
                fields=baseline_fields,
                raw_text=text,
                error_message=f"NVIDIA API call failed ({type(exc).__name__}: {sanitized}). Switched to fallback baseline extractor.",
            )

        except json.JSONDecodeError as exc:
            sanitized = sanitize_error_message(str(exc), self.api_key)
            baseline_fields = self.baseline_extractor.extract(text, doc_type=doc_type)
            return ExtractionResult(
                doc_id=doc_id,
                doc_type=doc_type,
                source="baseline_fallback",
                provider="regex",
                model_used=None,
                fields=baseline_fields,
                raw_text=text,
                error_message=f"NVIDIA returned malformed JSON ({type(exc).__name__}: {sanitized}). Switched to fallback baseline extractor.",
            )

        except Exception as exc:
            sanitized = sanitize_error_message(str(exc), self.api_key)
            baseline_fields = self.baseline_extractor.extract(text, doc_type=doc_type)
            return ExtractionResult(
                doc_id=doc_id,
                doc_type=doc_type,
                source="baseline_fallback",
                provider="regex",
                model_used=None,
                fields=baseline_fields,
                raw_text=text,
                error_message=f"NVIDIA extraction failed ({type(exc).__name__}: {sanitized}). Switched to fallback baseline extractor.",
            )
