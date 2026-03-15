# Issue 5: Add absolute(n) method

**Type**: feature
**Difficulty**: easy
**File**: calculator/calculator.py

## Description

Add an `absolute(n)` method to the `Calculator` class that returns the absolute value of a number. The method should:

1. Compute the absolute value using Python's built-in `abs()` function (or equivalent logic).
2. Return the result.
3. Record the operation in the history list as a tuple: `('absolute', n, None, result)`.

## Expected Behavior

- `calc.absolute(5)` returns `5`.
- `calc.absolute(-5)` returns `5`.
- `calc.absolute(0)` returns `0`.

## Current Behavior

Method does not exist. Calling `calc.absolute(-5)` raises `AttributeError`.

## Test Cases

Run `pytest tests/test_calculator.py::TestIssue5_AbsoluteValue` to verify the fix.
