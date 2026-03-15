# Issue 7: Add mean(numbers) method

**Type**: feature
**Difficulty**: medium
**File**: calculator/calculator.py

## Description

Add a `mean(numbers)` method to the `Calculator` class that computes the arithmetic mean (average) of a list of numbers. The method should:

1. Validate that `numbers` is not an empty list. If it is, raise a `ValueError` (e.g., "Cannot compute mean of empty list").
2. Compute the sum of all elements divided by the length of the list.
3. Return the result as a float.
4. Record the operation in the history list as a tuple: `('mean', numbers, None, result)`.

## Expected Behavior

- `calc.mean([1, 2, 3, 4, 5])` returns `3.0`.
- `calc.mean([42])` returns `42.0`.
- `calc.mean([1.5, 2.5])` returns `2.0`.
- `calc.mean([])` raises `ValueError`.

## Current Behavior

Method does not exist. Calling `calc.mean([1, 2, 3])` raises `AttributeError`.

## Test Cases

Run `pytest tests/test_calculator.py::TestIssue7_Mean` to verify the fix.
