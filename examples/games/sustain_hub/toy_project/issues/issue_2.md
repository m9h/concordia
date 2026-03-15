# Issue 2: Fix sqrt of negative numbers

**Type**: bug_fix
**Difficulty**: easy
**File**: calculator/calculator.py

## Description

The `sqrt(a)` method calls `math.sqrt(a)` without first checking whether `a` is negative. When a negative number is passed, Python's `math.sqrt` raises a `ValueError: math domain error`, which is an unhelpful error message. The method should explicitly check for negative input and raise a `ValueError` with a descriptive message before calling `math.sqrt`.

## Expected Behavior

- `calc.sqrt(-4)` should raise `ValueError` with a message matching "Cannot take square root of negative" (e.g., "Cannot take square root of negative number").
- `calc.sqrt(4)` should still return `2.0`.
- `calc.sqrt(0)` should still return `0.0`.

## Current Behavior

- `calc.sqrt(-4)` raises `ValueError: math domain error`, which is the raw error from `math.sqrt` and does not clearly explain the problem to the user.

## Test Cases

Run `pytest tests/test_calculator.py::TestIssue2_NegativeSqrt` to verify the fix.
