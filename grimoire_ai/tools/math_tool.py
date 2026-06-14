"""Safe arithmetic evaluator and math-intent detector.

Two public entry points:

    MathTool.run(query)             → inject-ready string or None
    MathTool.process_response(text) → response with <TOOL:python> tags resolved

Detection
---------
Patterns that trigger math detection (all require digits adjacent to operators):

    "what is 2 + 2?"               → "2 + 2"
    "calculate 1.5 * 200"          → "1.5 * 200"
    "25% of 1000"                  → "0.25 * 1000"
    "evaluate (3 + 4) * 2"         → "(3 + 4) * 2"
    "what's 2^10?"                 → "2**10"    (^ → ** normalised)
    "compute sqrt(144)"            → "sqrt(144)"

Patterns that do NOT trigger:

    "what is grapple?"             — no arithmetic operators between digits
    "2d6 damage"                   — dice notation excluded
    "how much damage?"             — no digits + operators

Evaluator safety
----------------
No ``eval()``, no ``exec()``, no imports.  The expression is parsed into an
``ast.Expression`` node and walked by ``_SafeEval`` which only accepts:

- Integer and float literals
- Unary ``+`` / ``-``
- Binary ``+``, ``-``, ``*``, ``/``, ``//``, ``%``, ``**``
- Parenthesised sub-expressions
- Whitelisted function calls: ``abs``, ``round``, ``min``, ``max``,
  ``sqrt``, ``floor``, ``ceil``, ``sin``, ``cos``, ``tan``,
  ``log``, ``log2``, ``log10``
- Named constants: ``pi``, ``e``

Any other node type raises ``ValueError``, which is caught and surfaced as an
error string rather than crashing the chat loop.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Safe AST evaluator
# ---------------------------------------------------------------------------

_SAFE_NAMES: dict[str, object] = {
    "abs":   abs,
    "round": round,
    "min":   min,
    "max":   max,
    "sqrt":  math.sqrt,
    "floor": math.floor,
    "ceil":  math.ceil,
    "sin":   math.sin,
    "cos":   math.cos,
    "tan":   math.tan,
    "log":   math.log,
    "log2":  math.log2,
    "log10": math.log10,
    "pi":    math.pi,
    "e":     math.e,
}

_BINARY_OPS: dict[type, object] = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod:      operator.mod,
    ast.Pow:      operator.pow,
}


class _SafeEval(ast.NodeVisitor):
    """AST visitor that evaluates arithmetic only — raises ValueError otherwise."""

    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported literal type: {type(node.value).__name__}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left  = self.visit(node.left)
        right = self.visit(node.right)
        fn = _BINARY_OPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        return fn(left, right)

    def visit_Call(self, node: ast.Call) -> float:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple built-in function calls are allowed.")
        fn = _SAFE_NAMES.get(node.func.id)
        if fn is None:
            raise ValueError(f"Unknown function: {node.func.id!r}")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported.")
        args = [self.visit(a) for a in node.args]
        return fn(*args)

    def visit_Name(self, node: ast.Name) -> float:
        val = _SAFE_NAMES.get(node.id)
        if val is None or not isinstance(val, (int, float)):
            raise ValueError(f"Unknown name: {node.id!r}")
        return float(val)

    def generic_visit(self, node: ast.AST) -> float:
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def _safe_eval(expression: str) -> float:
    """Parse and evaluate *expression* without calling ``eval()``.

    Raises ``ValueError`` for any expression that isn't pure arithmetic.
    Raises ``ZeroDivisionError`` for division by zero.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Syntax error: {exc}") from exc
    return _SafeEval().visit(tree)


def _format_result(value: float) -> str:
    """Format a float result for display.

    Integers are shown without a decimal point; floats are rounded to 6
    significant figures to avoid floating-point noise.
    """
    if value == math.floor(value) and abs(value) < 1e15:
        return str(int(value))
    # Round to 6 significant figures.
    mag = math.floor(math.log10(abs(value))) if value != 0 else 0
    rounded = round(value, max(0, 5 - mag))
    return f"{rounded:g}"


# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------

# Normalise Unicode operators before pattern matching.
_UNICODE_MAP = str.maketrans({
    "×": "*",
    "÷": "/",
    "^": "**",
    "−": "-",   # Unicode minus
})

# Detect "N% of M" patterns (e.g. "25% of 1000").
_PERCENT_OF = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Match an arithmetic expression: numbers separated by operators, allowing
# parentheses, spaces, and named math functions.
# Requires at least one arithmetic operator (+, -, *, /, %, **) between digits,
# OR a math function call like sqrt(144).
_ARITH_EXPR = re.compile(
    r"""
    (?:                             # option A: function call
        (?:sqrt|floor|ceil|abs|round|log(?:2|10)?|sin|cos|tan)
        \s*\([\d\s\.\+\-\*\/\%\(\)\*\*]+\)
    )
    |
    (?:                             # option B: infix expression with operator
        [\(\s]*                     # optional leading parens/spaces
        -?                          # optional leading minus
        \d+(?:\.\d+)?               # first number
        (?:                         # one or more (operator number) pairs
            \s*(?:\*\*|[+\-*/%/])\s*
            \(?\s*-?\d+(?:\.\d+)?\s*\)?
        )+
    )
    """,
    re.VERBOSE,
)

# Keywords that introduce math intent — the expression must still be present.
_MATH_KEYWORDS = re.compile(
    r"\b(?:calculat\w*|comput\w*|evaluat\w*|what\s+is|what'?s|how\s+much\s+is|result\s+of)\b",
    re.IGNORECASE,
)

# Dice notation — explicitly excluded to avoid misidentifying "2d6+3" as math.
_DICE_NOTATION = re.compile(r"\b\d+d\d+\b", re.IGNORECASE)

# Tool call tag emitted by a fine-tuned model.
_TOOL_TAG = re.compile(r"<TOOL:python>(.*?)</TOOL>", re.DOTALL)


# ---------------------------------------------------------------------------
# MathTool
# ---------------------------------------------------------------------------

class MathTool:
    """Detect arithmetic in user queries and evaluate it safely.

    Attach to an ``InferenceEngine`` via ``engine.math_tool = MathTool()`` to
    enable the math pre-processing path.

    Typical use:

        tool = MathTool()
        result = tool.run("What is 25% of 1200?")
        # → "0.25 * 1200 = 300"
    """

    def detect(self, query: str) -> Optional[str]:
        """Return the arithmetic expression found in *query*, or ``None``.

        Dice notation (e.g. ``2d6``) is never matched even when surrounded by
        arithmetic operators, because it is probabilistic rather than
        deterministic arithmetic.
        """
        if _DICE_NOTATION.search(query):
            return None

        # Normalise Unicode operators.
        normalised = query.translate(_UNICODE_MAP)

        # Handle "N% of M" idiom.
        m = _PERCENT_OF.search(normalised)
        if m:
            pct, total = m.group(1), m.group(2)
            return f"{pct} / 100 * {total}"

        # Require a math keyword OR a standalone infix expression.
        has_keyword = bool(_MATH_KEYWORDS.search(normalised))
        m = _ARITH_EXPR.search(normalised)
        if m and has_keyword:
            return m.group(0).strip()
        if m and not has_keyword:
            # Accept a standalone infix expression only when it looks
            # unambiguously mathematical (contains a digit-operator-digit triple).
            snippet = m.group(0).strip()
            if re.search(r"\d\s*[-+*/]\s*\d", snippet):
                return snippet

        return None

    def evaluate(self, expression: str) -> tuple[str, Optional[str]]:
        """Evaluate *expression* safely.

        Returns a ``(result_str, error_str)`` pair.  On success ``error_str`` is
        ``None``; on failure ``result_str`` is ``""`` and ``error_str`` describes
        the problem.
        """
        expr = expression.translate(_UNICODE_MAP).strip()
        try:
            value = _safe_eval(expr)
        except ZeroDivisionError:
            return "", "division by zero"
        except (ValueError, TypeError) as exc:
            return "", str(exc)
        except RecursionError:
            return "", "expression is too deeply nested"
        return _format_result(value), None

    def run(self, query: str) -> Optional[str]:
        """Detect arithmetic in *query* and return an inject-ready string.

        Returns a string of the form ``"expr = result"`` when math is found
        and evaluation succeeds, or ``None`` otherwise.  The caller can prepend
        this to the corpus context so the model sees the verified number.

        Example::

            tool.run("What is 25% of 1200?")   # → "0.25 / 100 * 1200 = 300"
            tool.run("Tell me about grapple")   # → None
        """
        expr = self.detect(query)
        if expr is None:
            return None
        result, error = self.evaluate(expr)
        if error:
            return None
        canonical = expr.replace("**", "^").replace("/ 100 *", "% of")
        return f"{canonical} = {result}"

    def process_response(self, response: str) -> str:
        """Replace ``<TOOL:python>expr</TOOL>`` tags with evaluated results.

        Called post-generation when the model (after fine-tuning) emits a tool
        tag.  Each tag is replaced with ``expr = result`` inline so the final
        visible response is self-contained.

        If evaluation fails the tag text is left in place with an error note.
        """
        def _replace(m: re.Match) -> str:
            expr = m.group(1).strip()
            result, error = self.evaluate(expr)
            if error:
                return f"{expr} [error: {error}]"
            return f"{expr} = {result}"

        return _TOOL_TAG.sub(_replace, response)
