# Issue 8: Add is_prime(n) method

**Type**: feature
**Difficulty**: hard
**File**: calculator/calculator.py

## Description

Add an `is_prime(n)` method to the `Calculator` class that determines whether a given integer `n` is a prime number. A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. The method should:

1. Return `False` for any value less than 2 (including negative numbers, 0, and 1).
2. Return `True` for 2 (the smallest prime).
3. Return `False` for even numbers greater than 2.
4. For odd numbers greater than 2, check divisibility by all odd numbers from 3 up to and including the square root of `n`. If any divides evenly, return `False`; otherwise return `True`.
5. Record the operation in the history list as a tuple: `('is_prime', n, None, result)`.

The method should handle edge cases correctly without raising exceptions -- negative numbers, zero, and 1 should simply return `False`.

## Expected Behavior

- `calc.is_prime(2)` returns `True`.
- `calc.is_prime(3)` returns `True`.
- `calc.is_prime(5)` returns `True`.
- `calc.is_prime(97)` returns `True`.
- `calc.is_prime(1)` returns `False`.
- `calc.is_prime(4)` returns `False`.
- `calc.is_prime(9)` returns `False`.
- `calc.is_prime(100)` returns `False`.
- `calc.is_prime(-5)` returns `False`.
- `calc.is_prime(0)` returns `False`.

## Current Behavior

Method does not exist. Calling `calc.is_prime(7)` raises `AttributeError`.

## Test Cases

Run `pytest tests/test_calculator.py::TestIssue8_IsPrime` to verify the fix.
