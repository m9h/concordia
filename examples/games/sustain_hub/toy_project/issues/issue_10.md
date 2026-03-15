# Issue 10: Record failed operations in history

**Type**: bug_fix
**Difficulty**: hard
**File**: calculator/calculator.py

## Description

When an operation raises an exception (e.g., `divide(1, 0)` raises `ValueError` after Issue 1 is fixed), the failed operation is NOT recorded in the calculator's history. This makes it impossible to audit what operations were attempted, including failed ones.

The fix requires modifying methods that can raise exceptions so that they record the attempted operation in the history list BEFORE raising the exception. The history entry for a failed operation should use `None` as the result value, or include an error indicator in the tuple.

Specifically, for `divide(a, b)` when `b == 0`:
1. Append `('divide', a, b, None)` to `self.history`.
2. Then raise `ValueError("Cannot divide by zero")`.

This pattern should be applied to any method that performs validation and raises an exception. The history should record that the operation was attempted, with `None` (or an error string) as the result.

Note: This issue depends on Issue 1 being resolved first (so that `divide(1, 0)` raises `ValueError` instead of `ZeroDivisionError`). The test checks that after a failed `divide(1, 0)` call, the history contains exactly one entry whose first element is `'divide'` and whose last element is `None` or contains the word "error".

## Expected Behavior

- After calling `calc.divide(1, 0)` (which raises `ValueError`), `calc.get_history()` has one entry.
- That entry's first element is `'divide'`.
- That entry's last element is `None` or contains the word "error" (case-insensitive).
- Successful operations still record normally: `calc.add(1, 2)` produces history entry `('add', 1, 2, 3)`.

## Current Behavior

When `divide(1, 0)` raises an exception, no history entry is recorded. The history remains empty after the failed call.

## Test Cases

Run `pytest tests/test_calculator.py::TestIssue10_HistoryErrors` to verify the fix.
