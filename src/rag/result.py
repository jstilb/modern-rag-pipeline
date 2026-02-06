"""Result type for explicit error handling without exceptions.

Provides a Rust-inspired Result[T, E] pattern for operations that can fail.
Forces callers to handle both success and error cases explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Union

T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """Successful result containing a value."""

    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default: T) -> T:  # type: ignore[override]
        return self.value

    def map(self, fn: Callable[[T], U]) -> Result[U, E]:  # type: ignore[type-var]
        return Ok(fn(self.value))

    def map_err(self, fn: Callable[[E], U]) -> Result[T, U]:  # type: ignore[type-var]
        return Ok(self.value)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    """Error result containing an error value."""

    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> T:  # type: ignore[type-var]
        raise ValueError(f"Called unwrap on Err: {self.error}")

    def unwrap_or(self, default: T) -> T:  # type: ignore[type-var]
        return default

    def map(self, fn: Callable[[T], U]) -> Result[U, E]:  # type: ignore[type-var]
        return Err(self.error)  # type: ignore[return-value]

    def map_err(self, fn: Callable[[E], U]) -> Result[T, U]:  # type: ignore[type-var]
        return Err(fn(self.error))  # type: ignore[return-value]


Result = Union[Ok[T], Err[E]]
