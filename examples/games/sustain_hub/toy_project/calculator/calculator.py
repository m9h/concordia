"""Calculator module for SustainHub toy project.

This module has intentional bugs and missing features that serve as
task targets for LLM agents in the simulation.
"""

import math


class Calculator:
    """A basic calculator with history tracking."""

    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(('add', a, b, result))
        return result

    def subtract(self, a, b):
        result = a - b
        self.history.append(('subtract', a, b, result))
        return result

    def multiply(self, a, b):
        result = a * b
        self.history.append(('multiply', a, b, result))
        return result

    def divide(self, a, b):
        # BUG: No zero division check
        result = a / b
        self.history.append(('divide', a, b, result))
        return result

    def power(self, base, exp):
        # BUG: Doesn't handle negative exponents correctly
        result = base ** exp
        self.history.append(('power', base, exp, result))
        return result

    def sqrt(self, a):
        # BUG: No check for negative numbers
        result = math.sqrt(a)
        self.history.append(('sqrt', a, None, result))
        return result

    def modulo(self, a, b):
        # BUG: No zero division check
        result = a % b
        self.history.append(('modulo', a, b, result))
        return result

    def factorial(self, n):
        # BUG: No check for negative numbers or non-integers
        if n == 0:
            return 1
        result = n * self.factorial(n - 1)
        self.history.append(('factorial', n, None, result))
        return result

    def get_history(self):
        return list(self.history)

    def clear_history(self):
        self.history = []

    # MISSING: percentage method
    # MISSING: absolute value method
    # MISSING: logarithm method
    # MISSING: mean of a list method
    # MISSING: is_prime method
