# Issue 3: Fix factorial validation for negatives and floats

**Type**: bug_fix
**Difficulty**: medium
**File**: calculator/calculator.py

## Description

The `factorial(n)` method has two validation bugs:

1. **Negative numbers**: When a negative integer is passed (e.g., `factorial(-1)`), the method enters infinite recursion because `n` will never reach `0`. This eventually causes a `RecursionError`. The method should check for negative input and raise a `ValueError`.

2. **Non-integer values**: When a float is passed (e.g., `factorial(3.5)`), the method recurses with non-integer values (`3.5, 2.5, 1.5, 0.5, -0.5, ...`), again causing infinite recursion. The method should check that `n` is an integer (or an integer-valued float like `5.0` should also be rejected -- only `int` types should be accepted) and raise a `ValueError` if not.

## Expected Behavior

- `calc.factorial(-1)` should raise `ValueError` with a message matching "Factorial.*negative" (e.g., "Factorial is not defined for negative numbers").
- `calc.factorial(3.5)` should raise `ValueError` with a message matching "Factorial.*integer" (e.g., "Factorial requires an integer argument").
- `calc.factorial(0)` should still return `1`.
- `calc.factorial(5)` should still return `120`.

## Current Behavior

- `calc.factorial(-1)` causes a `RecursionError: maximum recursion depth exceeded`.
- `calc.factorial(3.5)` causes a `RecursionError: maximum recursion depth exceeded`.

## Test Cases

Run `pytest tests/test_calculator.py::TestIssue3_FactorialValidation` to verify the fix.
