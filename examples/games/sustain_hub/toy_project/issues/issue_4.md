# Issue 4: Add percentage(value, total) method

**Type**: feature
**Difficulty**: easy
**File**: calculator/calculator.py

## Description

Add a `percentage(value, total)` method to the `Calculator` class that calculates what percentage `value` is of `total`. The formula is `(value / total) * 100`. The method should:

1. Check that `total` is not zero; if it is, raise a `ValueError` (e.g., "Total cannot be zero").
2. Compute and return the result as a float.
3. Record the operation in the history list as a tuple: `('percentage', value, total, result)`.

## Expected Behavior

- `calc.percentage(50, 200)` returns `25.0`.
- `calc.percentage(200, 200)` returns `100.0`.
- `calc.percentage(0, 200)` returns `0.0`.
- `calc.percentage(50, 0)` raises `ValueError`.

## Current Behavior

Method does not exist. Calling `calc.percentage(50, 200)` raises `AttributeError`.

## Test Cases

Run `pytest tests/test_calculator.py::TestIssue4_Percentage` to verify the fix.
