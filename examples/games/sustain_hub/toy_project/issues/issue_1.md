# Issue 1: Fix division by zero in divide() and modulo()

**Type**: bug_fix
**Difficulty**: easy
**File**: calculator/calculator.py

## Description

The `divide(a, b)` and `modulo(a, b)` methods do not check whether the divisor `b` is zero before performing the operation. When `b` is zero, Python raises a `ZeroDivisionError`, which is an unhandled internal exception. Both methods should instead raise a `ValueError` with a clear message indicating that division by zero is not allowed.

## Expected Behavior

- `calc.divide(10, 0)` should raise `ValueError` with the message "Cannot divide by zero".
- `calc.modulo(10, 0)` should raise `ValueError` with the message "Cannot divide by zero".
- Normal operations like `calc.divide(10, 2)` should still return `5.0` as before.

## Current Behavior

- `calc.divide(10, 0)` raises an unhandled `ZeroDivisionError: division by zero`.
- `calc.modulo(10, 0)` raises an unhandled `ZeroDivisionError: integer division or modulo by zero`.

## Test Cases

Run `pytest tests/test_calculator.py::TestIssue1_DivideByZero` to verify the fix.
