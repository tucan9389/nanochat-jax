"""Smoke tests for nanochat_jax.execution sandbox."""

from __future__ import annotations

import sys

import pytest

from nanochat_jax.execution import ExecutionResult, execute_code


def test_simple_print_succeeds():
    result = execute_code("print('hello world')", timeout=5.0)
    assert isinstance(result, ExecutionResult)
    assert result.success
    assert "hello world" in result.stdout
    assert not result.timeout


def test_syntax_error_marks_failure():
    result = execute_code("print(", timeout=5.0)
    assert not result.success
    assert result.error is not None


def test_runtime_error_marks_failure():
    result = execute_code("raise ValueError('boom')", timeout=5.0)
    assert not result.success
    assert "ValueError" in (result.error or "") or "ValueError" in result.stderr


def test_infinite_loop_times_out():
    result = execute_code("while True: pass", timeout=1.0)
    assert not result.success
    assert result.timeout


@pytest.mark.skipif(sys.platform == "win32", reason="signals/resource limits Unix only")
def test_arithmetic_returns_via_stdout():
    result = execute_code("print(2 + 2 * 5)", timeout=5.0)
    assert result.success
    assert "12" in result.stdout
