#!/usr/bin/env python3
"""Standalone test runner for SmartHire test suite.

Discovers and executes all test_* functions across tests/test_*.py files,
providing standard fixtures (tiny_resumes, tiny_jobs, tmp_path) and assertion
support (pytest.raises, pytest.approx).
"""
from __future__ import annotations

import inspect
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Built-in pytest shim if pytest is not installed ──────────────────────────

class _Approx:
    def __init__(self, expected, abs_tol=1e-5, rel_tol=1e-5):
        self.expected = expected
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

    def __eq__(self, actual):
        if self.expected == 0.0:
            return abs(actual) <= self.abs_tol
        return math.isclose(actual, self.expected, abs_tol=self.abs_tol, rel_tol=self.rel_tol)

    def __repr__(self):
        return f"approx({self.expected} ± {self.abs_tol})"


class _RaisesContext:
    def __init__(self, expected_exception, match=None):
        self.expected_exception = expected_exception
        self.match = match

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(f"Expected exception {self.expected_exception.__name__} was not raised")
        if not issubclass(exc_type, self.expected_exception):
            return False
        if self.match and not re.search(self.match, str(exc_val)):
            raise AssertionError(f"Exception message '{exc_val}' did not match pattern '{self.match}'")
        return True


class _PytestShim:
    @staticmethod
    def approx(expected, abs=1e-5, rel=1e-5):
        return _Approx(expected, abs_tol=abs, rel_tol=rel)

    @staticmethod
    def raises(expected_exception, match=None):
        return _RaisesContext(expected_exception, match=match)

    @staticmethod
    def fixture(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


if "pytest" not in sys.modules:
    try:
        import pytest  # noqa: F401
    except ImportError:
        sys.modules["pytest"] = _PytestShim()


# ── Shared fixtures ──────────────────────────────────────────────────────────

import tests.conftest as conftest_module  # noqa: E402


def get_fixtures(tmp_dir_path: Path) -> dict:
    fixtures = {
        "tmp_path": tmp_dir_path,
    }
    for name in dir(conftest_module):
        item = getattr(conftest_module, name)
        if not name.startswith("_"):
            func = getattr(item, "__wrapped__", getattr(item, "_fixture_function", getattr(item, "func", None)))
            if func is not None and callable(func):
                fixtures[name] = func()
            elif callable(item):
                try:
                    fixtures[name] = item()
                except Exception:
                    pass
    return fixtures


# ── Discovery & Runner ────────────────────────────────────────────────────────

def run_all_tests() -> int:
    try:
        import pytest
        return pytest.main(["tests/", "-v"])
    except ImportError:
        pass

    tests_dir = PROJECT_ROOT / "tests"
    test_files = sorted(tests_dir.glob("test_*.py"))

    total_count = 0
    passed_count = 0
    failed_count = 0
    errors = []

    print("\n" + "=" * 65)
    print("  SmartHire Test Suite")
    print("=" * 65)
    t_start = time.time()

    for test_file in test_files:
        module_name = f"tests.{test_file.stem}"
        try:
            if module_name in sys.modules:
                mod = sys.modules[module_name]
            else:
                mod = __import__(module_name, fromlist=["*"])
        except Exception as e:
            print(f"\n[FAIL] Error importing {test_file.name}: {e}")
            errors.append((test_file.name, "import", str(e)))
            continue

        test_funcs = [
            (name, func)
            for name, func in inspect.getmembers(mod, inspect.isfunction)
            if name.startswith("test_") and func.__module__ == module_name
        ]

        print(f"\n[-] {test_file.name} ({len(test_funcs)} tests)", flush=True)

        for name, func in test_funcs:
            total_count += 1
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                fixtures = get_fixtures(tmp_path)
                sig = inspect.signature(func)
                kwargs = {}
                for param in sig.parameters:
                    if param in fixtures:
                        kwargs[param] = fixtures[param]
                    else:
                        raise ValueError(f"Unknown fixture '{param}' requested by {name}")

                try:
                    func(**kwargs)
                    print(f"   [PASS] {name}", flush=True)
                    passed_count += 1
                except Exception as e:
                    print(f"   [FAIL] {name}: {e}", flush=True)
                    failed_count += 1
                    errors.append((test_file.name, name, str(e)))

    duration = time.time() - t_start
    print("\n" + "=" * 65, flush=True)
    if failed_count == 0:
        print(f"  All {passed_count}/{total_count} tests passed in {duration:.2f}s!", flush=True)
    else:
        print(f"  {passed_count} passed, {failed_count} failed out of {total_count} in {duration:.2f}s", flush=True)
        print("\nFailures:", flush=True)
        for file, func, msg in errors:
            print(f"  - {file}::{func}: {msg}", flush=True)
    print("=" * 65 + "\n", flush=True)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
