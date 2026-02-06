"""Tests for the Result type."""

import pytest
from hypothesis import given, strategies as st

from src.rag.result import Err, Ok


class TestOk:
    def test_is_ok(self) -> None:
        result = Ok(42)
        assert result.is_ok() is True
        assert result.is_err() is False

    def test_unwrap(self) -> None:
        result = Ok("hello")
        assert result.unwrap() == "hello"

    def test_unwrap_or(self) -> None:
        result = Ok(42)
        assert result.unwrap_or(0) == 42

    def test_map(self) -> None:
        result = Ok(5)
        mapped = result.map(lambda x: x * 2)
        assert mapped.unwrap() == 10

    def test_map_err_noop(self) -> None:
        result = Ok(5)
        mapped = result.map_err(lambda e: f"error: {e}")
        assert mapped.is_ok()
        assert mapped.unwrap() == 5

    @given(st.integers())
    def test_ok_preserves_value(self, value: int) -> None:
        result = Ok(value)
        assert result.unwrap() == value


class TestErr:
    def test_is_err(self) -> None:
        result = Err("something failed")
        assert result.is_err() is True
        assert result.is_ok() is False

    def test_unwrap_raises(self) -> None:
        result = Err("fail")
        with pytest.raises(ValueError, match="fail"):
            result.unwrap()

    def test_unwrap_or(self) -> None:
        result = Err("fail")
        assert result.unwrap_or(42) == 42

    def test_map_noop(self) -> None:
        result = Err("fail")
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_err()

    def test_map_err(self) -> None:
        result = Err("fail")
        mapped = result.map_err(lambda e: f"wrapped: {e}")
        assert mapped.is_err()

    @given(st.text(min_size=1))
    def test_err_preserves_error(self, error: str) -> None:
        result = Err(error)
        with pytest.raises(ValueError):
            result.unwrap()
