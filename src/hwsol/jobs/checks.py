"""Exact numeric recomputation, without executing input strings as code."""

import ast
import operator

from hwsol.jobs.schema import Verification


def arithmetic(expression: str):
    from sympy import Rational

    if len(expression) > 512:
        raise ValueError("Numeric check exceeds 512 characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid arithmetic syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > 150:
        raise ValueError("Numeric check too complex")
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }

    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) in {float, int}:
            # Read decimal digits from the expression, before Python float rounding.
            token = ast.get_source_segment(expression, node)
            if "e" in token.lower() or len(token) > 100:
                raise ValueError("Use ordinary decimal or integer literals in checks")
            return Rational(token)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            result = value(node.operand)
            return -result if isinstance(node.op, ast.USub) else result
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            left, right = value(node.left), value(node.right)
            if isinstance(node.op, ast.Pow) and (right.q != 1 or abs(right) > 32):
                raise ValueError("Check exponent must be an integer between -32 and 32")
            result = operations[type(node.op)](left, right)
            if (
                not result.is_Rational
                or max(int(result.p).bit_length(), int(result.q).bit_length()) > 2048
            ):
                raise ValueError("Numeric check is undefined or too large")
            return result
        raise ValueError("Checks accept numbers, parentheses, + - * / and **")

    return value(tree.body)


def check_answer(v: Verification) -> dict:
    if v.status == "failed":
        raise ValueError(f"Agent verification failed: {v.evidence}")
    results = []
    for check in v.checks:
        lhs, rhs = arithmetic(check.lhs), arithmetic(check.rhs)
        if lhs != rhs:
            raise ValueError(f"Numeric check failed: {check.lhs} = {lhs}, {check.rhs} = {rhs}")
        results.append({**check.model_dump(), "value": str(lhs)})
    return {
        "agent_status": v.status,
        "agent_evidence": v.evidence,
        "arithmetic_status": "passed" if results else "not_checked",
        "equalities": results,
    }
