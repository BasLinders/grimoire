"""Unit tests for grimoire_ai.tools.math_tool.

Coverage:
    _safe_eval           — arithmetic operators, functions, constants, rejections
    MathTool.detect      — keyword+expr, standalone expr, percent-of, exclusions
    MathTool.evaluate    — correct results, division-by-zero, bad expressions
    MathTool.run         — full pipeline returning inject-ready strings or None
    MathTool.process_response — <TOOL:python> tag substitution
"""

import math
import pytest

from grimoire_ai.tools.math_tool import MathTool, _safe_eval


# ---------------------------------------------------------------------------
# _safe_eval — low-level evaluator
# ---------------------------------------------------------------------------

class TestSafeEval:
    def test_addition(self):
        assert _safe_eval("2 + 3") == 5.0

    def test_subtraction(self):
        assert _safe_eval("10 - 4") == 6.0

    def test_multiplication(self):
        assert _safe_eval("6 * 7") == 42.0

    def test_division(self):
        assert _safe_eval("10 / 4") == pytest.approx(2.5)

    def test_floor_division(self):
        assert _safe_eval("10 // 3") == 3.0

    def test_modulo(self):
        assert _safe_eval("10 % 3") == 1.0

    def test_power(self):
        assert _safe_eval("2 ** 10") == 1024.0

    def test_unary_minus(self):
        assert _safe_eval("-5 + 10") == 5.0

    def test_parentheses(self):
        assert _safe_eval("(3 + 4) * 2") == 14.0

    def test_float_literal(self):
        assert _safe_eval("1.5 * 200") == pytest.approx(300.0)

    def test_named_constant_pi(self):
        assert _safe_eval("pi") == pytest.approx(math.pi)

    def test_named_constant_e(self):
        assert _safe_eval("e") == pytest.approx(math.e)

    def test_function_sqrt(self):
        assert _safe_eval("sqrt(144)") == pytest.approx(12.0)

    def test_function_abs(self):
        assert _safe_eval("abs(-7)") == 7.0

    def test_function_round(self):
        assert _safe_eval("round(3.7)") == 4.0

    def test_function_floor(self):
        assert _safe_eval("floor(3.9)") == 3.0

    def test_function_ceil(self):
        assert _safe_eval("ceil(3.1)") == 4.0

    def test_nested_expression(self):
        assert _safe_eval("sqrt(3 ** 2 + 4 ** 2)") == pytest.approx(5.0)

    def test_rejects_string_literal(self):
        with pytest.raises(ValueError):
            _safe_eval('"hello"')

    def test_rejects_import(self):
        with pytest.raises((ValueError, SyntaxError)):
            _safe_eval("__import__('os')")

    def test_rejects_unknown_name(self):
        with pytest.raises(ValueError):
            _safe_eval("secret_var")

    def test_rejects_unknown_function(self):
        with pytest.raises(ValueError):
            _safe_eval("exec('pass')")

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            _safe_eval("1 / 0")

    def test_rejects_comparison_operator(self):
        with pytest.raises(ValueError):
            _safe_eval("1 < 2")

    def test_rejects_boolean_operator(self):
        with pytest.raises(ValueError):
            _safe_eval("True and False")


# ---------------------------------------------------------------------------
# MathTool.detect
# ---------------------------------------------------------------------------

class TestDetect:
    def setup_method(self):
        self.tool = MathTool()

    def test_keyword_plus_expression(self):
        assert self.tool.detect("What is 2 + 3?") is not None

    def test_calculate_keyword(self):
        assert self.tool.detect("calculate 1.5 * 200") is not None

    def test_compute_keyword(self):
        assert self.tool.detect("compute 100 / 4") is not None

    def test_standalone_infix(self):
        assert self.tool.detect("3 * 7") is not None

    def test_percent_of_pattern(self):
        expr = self.tool.detect("25% of 1200")
        assert expr is not None
        assert "25" in expr and "1200" in expr

    def test_power_notation(self):
        assert self.tool.detect("what is 2^10") is not None

    def test_unicode_multiply(self):
        assert self.tool.detect("what is 6 × 7") is not None

    def test_no_math_query(self):
        assert self.tool.detect("What happens when a creature is grappled?") is None

    def test_no_math_word_only(self):
        assert self.tool.detect("How much damage does a greatsword deal?") is None

    def test_dice_notation_excluded(self):
        assert self.tool.detect("what is 2d6 damage?") is None

    def test_dice_with_modifier_excluded(self):
        assert self.tool.detect("2d6 + 3 average?") is None

    def test_single_number_no_operator(self):
        assert self.tool.detect("answer is 42") is None


# ---------------------------------------------------------------------------
# MathTool.evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def setup_method(self):
        self.tool = MathTool()

    def test_integer_result(self):
        result, error = self.tool.evaluate("2 + 2")
        assert error is None
        assert result == "4"

    def test_float_result(self):
        result, error = self.tool.evaluate("10 / 4")
        assert error is None
        assert result == "2.5"

    def test_large_integer(self):
        result, error = self.tool.evaluate("1234 * 56")
        assert error is None
        assert result == "69104"

    def test_division_by_zero_error(self):
        result, error = self.tool.evaluate("1 / 0")
        assert result == ""
        assert "zero" in error

    def test_bad_expression_error(self):
        result, error = self.tool.evaluate("import os")
        assert result == ""
        assert error is not None

    def test_sqrt(self):
        result, error = self.tool.evaluate("sqrt(9)")
        assert error is None
        assert result == "3"

    def test_power(self):
        result, error = self.tool.evaluate("2 ** 10")
        assert error is None
        assert result == "1024"

    def test_unicode_normalisation(self):
        result, error = self.tool.evaluate("6 × 7")
        assert error is None
        assert result == "42"

    def test_caret_power(self):
        result, error = self.tool.evaluate("2^8")
        assert error is None
        assert result == "256"


# ---------------------------------------------------------------------------
# MathTool.run
# ---------------------------------------------------------------------------

class TestRun:
    def setup_method(self):
        self.tool = MathTool()

    def test_returns_string_for_math(self):
        result = self.tool.run("What is 2 + 2?")
        assert result is not None
        assert "4" in result

    def test_returns_none_for_non_math(self):
        assert self.tool.run("What is grapple?") is None

    def test_returns_none_for_dice(self):
        assert self.tool.run("2d6+3 average?") is None

    def test_percent_of(self):
        result = self.tool.run("What is 50% of 200?")
        assert result is not None
        assert "100" in result

    def test_power_result(self):
        result = self.tool.run("what is 2^8?")
        assert result is not None
        assert "256" in result

    def test_returns_none_when_eval_fails(self):
        # A detected-but-unparseable fragment shouldn't crash
        assert self.tool.run("calculate nothing + nothing") is None


# ---------------------------------------------------------------------------
# MathTool.process_response
# ---------------------------------------------------------------------------

class TestProcessResponse:
    def setup_method(self):
        self.tool = MathTool()

    def test_replaces_single_tag(self):
        response = "The answer is <TOOL:python>2 + 2</TOOL>."
        processed = self.tool.process_response(response)
        assert "<TOOL:python>" not in processed
        assert "4" in processed

    def test_replaces_multiple_tags(self):
        response = "<TOOL:python>3 * 3</TOOL> and <TOOL:python>4 * 4</TOOL>"
        processed = self.tool.process_response(response)
        assert "9" in processed
        assert "16" in processed

    def test_no_tags_unchanged(self):
        response = "The grappled creature has speed zero."
        assert self.tool.process_response(response) == response

    def test_error_in_tag_is_annotated(self):
        response = "<TOOL:python>1 / 0</TOOL>"
        processed = self.tool.process_response(response)
        assert "error" in processed.lower()
        assert "<TOOL:python>" not in processed

    def test_multiline_tag(self):
        response = "Result: <TOOL:python>\n1234 * 56\n</TOOL>."
        processed = self.tool.process_response(response)
        assert "69104" in processed

    def test_sqrt_in_tag(self):
        response = "<TOOL:python>sqrt(144)</TOOL>"
        assert "12" in self.tool.process_response(response)
