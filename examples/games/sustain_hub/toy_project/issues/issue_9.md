# Issue 9: Add docstrings to all public methods

**Type**: documentation
**Difficulty**: easy
**File**: calculator/calculator.py

## Description

Most public methods on the `Calculator` class lack docstrings entirely. Add meaningful docstrings to every public method. Each docstring should be at least 10 characters long (excluding leading/trailing whitespace) and should describe what the method does, its parameters, and its return value.

At minimum, the following methods must have docstrings:
- `add(self, a, b)`
- `subtract(self, a, b)`
- `multiply(self, a, b)`
- `divide(self, a, b)`
- `power(self, base, exp)`
- `sqrt(self, a)`
- `modulo(self, a, b)`
- `factorial(self, n)`
- `get_history(self)`
- `clear_history(self)`

Plus any new methods added by other issues (percentage, absolute, logarithm, mean, is_prime).

## Expected Behavior

- `calc.add.__doc__` is not `None` and has more than 10 non-whitespace characters.
- `calc.divide.__doc__` is not `None` and has more than 10 non-whitespace characters.
- `calc.factorial.__doc__` is not `None` and has more than 10 non-whitespace characters.
- The same applies to all other public methods.

## Current Behavior

Most methods have no docstring (`__doc__` is `None`), or only have inline comments (which are not docstrings).

## Test Cases

Run `pytest tests/test_calculator.py::TestIssue9_Docstrings` to verify the fix.
