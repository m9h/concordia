# Issue 6: Add logarithm(x, base=e) method

**Type**: feature
**Difficulty**: medium
**File**: calculator/calculator.py

## Description

Add a `logarithm(x, base=math.e)` method to the `Calculator` class that computes the logarithm of `x` with a given base. The default base should be `math.e` (natural logarithm). The method should:

1. Validate that `x` is positive (greater than zero). If `x <= 0`, raise a `ValueError` (e.g., "Logarithm is not defined for zero or negative numbers").
2. Validate that `base` is positive and not equal to 1. If invalid, raise a `ValueError`.
3. Use `math.log(x, base)` for arbitrary bases, or `math.log(x)` when base is `math.e`.
4. Return the result as a float.
5. Record the operation in the history list as a tuple: `('logarithm', x, base, result)`.

## Expected Behavior

- `calc.logarithm(100, 10)` returns approximately `2.0`.
- `calc.logarithm(math.e)` returns approximately `1.0` (natural log).
- `calc.logarithm(8, 2)` returns approximately `3.0`.
- `calc.logarithm(0)` raises `ValueError`.
- `calc.logarithm(-1)` raises `ValueError`.

## Current Behavior

Method does not exist. Calling `calc.logarithm(100, 10)` raises `AttributeError`.

## Test Cases

Run `pytest tests/test_calculator.py::TestIssue6_Logarithm` to verify the fix.
