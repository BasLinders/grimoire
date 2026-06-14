"""Unit tests for grimoire_ai.tools.math_tool.

Coverage:
    _safe_eval           — arithmetic operators, functions, constants, rejections
    MathTool.detect      — keyword+expr, standalone expr, percent-of, exclusions
    MathTool.evaluate    — correct results, division-by-zero, bad expressions
    MathTool.run         — full pipeline returning inject-ready strings or None
    MathTool.process_response — <TOOL:python> tag substitution
    Stdlib extensions    — factorial, exp, comb, perm, gcd, hypot, trig inverses,
                           degrees/radians, tau, inf
    Scipy stats          — norm_cdf, binom_pmf, etc. (skipped when scipy absent)
"""

import math
import pytest

from grimoire_ai.tools.math_tool import MathTool, _safe_eval

try:
    import scipy  # noqa: F401
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

scipy_only = pytest.mark.skipif(not _HAS_SCIPY, reason="scipy not installed")


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


# ---------------------------------------------------------------------------
# Stdlib math extensions
# ---------------------------------------------------------------------------

class TestStdlibExtensions:
    def test_factorial(self):
        assert _safe_eval("factorial(5)") == 120.0

    def test_exp(self):
        assert _safe_eval("exp(1)") == pytest.approx(math.e)

    def test_comb(self):
        assert _safe_eval("comb(10, 3)") == 120.0

    def test_perm(self):
        assert _safe_eval("perm(5, 2)") == 20.0

    def test_gcd(self):
        assert _safe_eval("gcd(12, 8)") == 4.0

    def test_hypot(self):
        assert _safe_eval("hypot(3, 4)") == pytest.approx(5.0)

    def test_asin(self):
        assert _safe_eval("asin(1)") == pytest.approx(math.pi / 2)

    def test_acos(self):
        assert _safe_eval("acos(1)") == pytest.approx(0.0)

    def test_atan(self):
        assert _safe_eval("atan(1)") == pytest.approx(math.pi / 4)

    def test_atan2(self):
        assert _safe_eval("atan2(1, 1)") == pytest.approx(math.pi / 4)

    def test_degrees(self):
        assert _safe_eval("degrees(pi)") == pytest.approx(180.0)

    def test_radians(self):
        assert _safe_eval("radians(180)") == pytest.approx(math.pi)

    def test_constant_tau(self):
        assert _safe_eval("tau") == pytest.approx(math.tau)

    def test_constant_inf(self):
        assert _safe_eval("inf") == math.inf

    def test_detect_comb_call(self):
        tool = MathTool()
        assert tool.detect("calculate comb(10, 3)") is not None

    def test_detect_atan2_call(self):
        tool = MathTool()
        assert tool.detect("what is atan2(1, 1)") is not None

    def test_evaluate_factorial(self):
        tool = MathTool()
        result, error = tool.evaluate("factorial(6)")
        assert error is None
        assert result == "720"

    def test_evaluate_hypot(self):
        tool = MathTool()
        result, error = tool.evaluate("hypot(3, 4)")
        assert error is None
        assert result == "5"

    def test_inf_constant_does_not_crash(self):
        result, error = MathTool().evaluate("inf")
        assert error is None
        assert result == "inf"

    def test_exp_overflow_returns_error(self):
        result, error = MathTool().evaluate("exp(1000)")
        assert result == ""
        assert error is not None and "large" in error

    def test_factorial_overflow_returns_error(self):
        result, error = MathTool().evaluate("factorial(200)")
        assert result == ""
        assert error is not None

    def test_factorial_negative_returns_error(self):
        result, error = MathTool().evaluate("factorial(-1)")
        assert result == ""
        assert error is not None

    def test_log_domain_error(self):
        result, error = MathTool().evaluate("log(0)")
        assert result == ""
        assert error is not None


# ---------------------------------------------------------------------------
# Scipy statistics functions (skipped when scipy is not installed)
# ---------------------------------------------------------------------------

class TestScipyStats:
    @scipy_only
    def test_norm_cdf_at_zero(self):
        result, error = MathTool().evaluate("norm_cdf(0)")
        assert error is None
        assert float(result) == pytest.approx(0.5, abs=1e-4)

    @scipy_only
    def test_norm_cdf_at_196(self):
        result, error = MathTool().evaluate("norm_cdf(1.96)")
        assert error is None
        assert float(result) == pytest.approx(0.975, abs=1e-3)

    @scipy_only
    def test_norm_pdf_at_zero(self):
        result, error = MathTool().evaluate("norm_pdf(0)")
        assert error is None
        assert float(result) == pytest.approx(1 / math.sqrt(2 * math.pi), rel=1e-4)

    @scipy_only
    def test_norm_ppf(self):
        result, error = MathTool().evaluate("norm_ppf(0.975)")
        assert error is None
        assert float(result) == pytest.approx(1.96, abs=1e-2)

    @scipy_only
    def test_binom_pmf(self):
        # P(X=3 | n=10, p=0.5) ≈ 0.1172
        result, error = MathTool().evaluate("binom_pmf(3, 10, 0.5)")
        assert error is None
        assert float(result) == pytest.approx(0.1172, abs=1e-3)

    @scipy_only
    def test_binom_cdf(self):
        # P(X≤5 | n=10, p=0.5) = 0.623
        result, error = MathTool().evaluate("binom_cdf(5, 10, 0.5)")
        assert error is None
        assert float(result) == pytest.approx(0.6230, abs=1e-3)

    @scipy_only
    def test_poisson_pmf(self):
        # P(X=2 | mu=3) ≈ 0.2240
        result, error = MathTool().evaluate("poisson_pmf(2, 3)")
        assert error is None
        assert float(result) == pytest.approx(0.2240, abs=1e-3)

    @scipy_only
    def test_t_ppf(self):
        # 97.5th percentile of t with 10 df ≈ 2.228
        result, error = MathTool().evaluate("t_ppf(0.975, 10)")
        assert error is None
        assert float(result) == pytest.approx(2.228, abs=1e-2)

    @scipy_only
    def test_chi2_ppf(self):
        # 95th percentile of chi2 with 5 df ≈ 11.07
        result, error = MathTool().evaluate("chi2_ppf(0.95, 5)")
        assert error is None
        assert float(result) == pytest.approx(11.07, abs=0.05)

    @scipy_only
    def test_detect_norm_cdf(self):
        tool = MathTool()
        assert tool.detect("what is norm_cdf(1.96)?") is not None

    @scipy_only
    def test_detect_binom_pmf(self):
        tool = MathTool()
        assert tool.detect("calculate binom_pmf(3, 10, 0.5)") is not None

    @scipy_only
    def test_run_norm_cdf(self):
        tool = MathTool()
        result = tool.run("what is norm_cdf(1.96)?")
        assert result is not None
        assert "norm_cdf" in result

    def test_norm_cdf_without_scipy_gives_error(self):
        """norm_cdf returns an error string when scipy is absent, not a crash."""
        if _HAS_SCIPY:
            pytest.skip("scipy is installed; graceful-degradation test not applicable")
        tool = MathTool()
        result, error = tool.evaluate("norm_cdf(1.96)")
        assert result == ""
        assert error is not None
