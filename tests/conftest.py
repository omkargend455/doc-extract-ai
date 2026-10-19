"""Pytest fixtures for DocExtract AI test isolation."""

from __future__ import annotations

import pytest
import streamlit as st


@pytest.fixture(autouse=True)
def isolate_secrets_and_env(monkeypatch):
    """Ensure automated tests never accidentally pick up local credentials or live environment variables."""
    # Reset NVIDIA environment variables for clean test isolation
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    # Mock st.secrets as an empty mapping by default so local .streamlit/secrets.toml is not used in tests
    # Individual tests can explicitly configure monkeypatch.setattr(st, "secrets", ...) or monkeypatch.setenv(...)
    try:
        monkeypatch.setattr(st, "secrets", {})
    except Exception:
        pass
