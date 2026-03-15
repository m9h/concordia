"""Tests for the calculator module.

Tests are organized by issue. Each issue has tests that FAIL on the
buggy version and PASS on the fixed version. This allows objective
scoring of LLM-generated patches.
"""

import pytest
import math
from calculator import Calculator


@pytest.fixture
def calc():
    return Calculator()


# === Issue 1: Division by zero (bug_fix, easy) ===

class TestIssue1_DivideByZero:
    def test_divide_normal(self, calc):
        assert calc.divide(10, 2) == 5.0

    def test_divide_by_zero_raises(self, calc):
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            calc.divide(10, 0)

    def test_modulo_by_zero_raises(self, calc):
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            calc.modulo(10, 0)


# === Issue 2: Negative sqrt (bug_fix, easy) ===

class TestIssue2_NegativeSqrt:
    def test_sqrt_positive(self, calc):
        assert calc.sqrt(4) == 2.0

    def test_sqrt_zero(self, calc):
        assert calc.sqrt(0) == 0.0

    def test_sqrt_negative_raises(self, calc):
        with pytest.raises(ValueError, match="Cannot take square root of negative"):
            calc.sqrt(-4)


# === Issue 3: Factorial validation (bug_fix, medium) ===

class TestIssue3_FactorialValidation:
    def test_factorial_zero(self, calc):
        assert calc.factorial(0) == 1

    def test_factorial_positive(self, calc):
        assert calc.factorial(5) == 120

    def test_factorial_negative_raises(self, calc):
        with pytest.raises(ValueError, match="Factorial.*negative"):
            calc.factorial(-1)

    def test_factorial_float_raises(self, calc):
        with pytest.raises(ValueError, match="Factorial.*integer"):
            calc.factorial(3.5)


# === Issue 4: Add percentage method (feature, easy) ===

class TestIssue4_Percentage:
    def test_percentage_basic(self, calc):
        assert calc.percentage(50, 200) == 25.0

    def test_percentage_full(self, calc):
        assert calc.percentage(200, 200) == 100.0

    def test_percentage_zero(self, calc):
        assert calc.percentage(0, 200) == 0.0

    def test_percentage_zero_total_raises(self, calc):
        with pytest.raises(ValueError):
            calc.percentage(50, 0)


# === Issue 5: Add absolute value method (feature, easy) ===

class TestIssue5_AbsoluteValue:
    def test_abs_positive(self, calc):
        assert calc.absolute(5) == 5

    def test_abs_negative(self, calc):
        assert calc.absolute(-5) == 5

    def test_abs_zero(self, calc):
        assert calc.absolute(0) == 0


# === Issue 6: Add logarithm method (feature, medium) ===

class TestIssue6_Logarithm:
    def test_log_base10(self, calc):
        assert calc.logarithm(100, 10) == pytest.approx(2.0)

    def test_log_natural(self, calc):
        assert calc.logarithm(math.e) == pytest.approx(1.0)

    def test_log_base2(self, calc):
        assert calc.logarithm(8, 2) == pytest.approx(3.0)

    def test_log_zero_raises(self, calc):
        with pytest.raises(ValueError):
            calc.logarithm(0)

    def test_log_negative_raises(self, calc):
        with pytest.raises(ValueError):
            calc.logarithm(-1)


# === Issue 7: Add mean method (feature, medium) ===

class TestIssue7_Mean:
    def test_mean_basic(self, calc):
        assert calc.mean([1, 2, 3, 4, 5]) == 3.0

    def test_mean_single(self, calc):
        assert calc.mean([42]) == 42.0

    def test_mean_empty_raises(self, calc):
        with pytest.raises(ValueError):
            calc.mean([])

    def test_mean_floats(self, calc):
        assert calc.mean([1.5, 2.5]) == 2.0


# === Issue 8: Add is_prime method (feature, hard) ===

class TestIssue8_IsPrime:
    def test_prime_small(self, calc):
        assert calc.is_prime(2) is True
        assert calc.is_prime(3) is True
        assert calc.is_prime(5) is True

    def test_not_prime(self, calc):
        assert calc.is_prime(1) is False
        assert calc.is_prime(4) is False
        assert calc.is_prime(9) is False

    def test_prime_large(self, calc):
        assert calc.is_prime(97) is True
        assert calc.is_prime(100) is False

    def test_prime_negative(self, calc):
        assert calc.is_prime(-5) is False

    def test_prime_zero(self, calc):
        assert calc.is_prime(0) is False


# === Issue 9: Write docstrings (documentation, easy) ===

class TestIssue9_Docstrings:
    def test_add_has_docstring(self, calc):
        assert calc.add.__doc__ is not None
        assert len(calc.add.__doc__.strip()) > 10

    def test_divide_has_docstring(self, calc):
        assert calc.divide.__doc__ is not None
        assert len(calc.divide.__doc__.strip()) > 10

    def test_factorial_has_docstring(self, calc):
        assert calc.factorial.__doc__ is not None
        assert len(calc.factorial.__doc__.strip()) > 10


# === Issue 10: History records errors (bug_fix, hard) ===

class TestIssue10_HistoryErrors:
    def test_history_records_success(self, calc):
        calc.add(1, 2)
        assert len(calc.get_history()) == 1

    def test_history_records_error(self, calc):
        with pytest.raises(ValueError):
            calc.divide(1, 0)
        history = calc.get_history()
        assert len(history) == 1
        assert history[0][0] == 'divide'
        assert 'error' in str(history[0]).lower() or history[0][-1] is None
