from __future__ import annotations

import ast
import random
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal, DivisionByZero, InvalidOperation

from app.training.tool_use.protocol import CalculateCall, GeneratedToolCall, GetTimeCall, SearchCall
from app.training.tool_use.schema import JsonObjectField, JsonObjectValue, JsonStringValue

BinaryOperation = Callable[[Decimal, Decimal], Decimal]
UnaryOperation = Callable[[Decimal], Decimal]

MAXIMUM_ABSOLUTE_RESULT = Decimal("1e100")
MAXIMUM_EXPONENT = Decimal("12")
TIME_SAMPLE_WINDOW = timedelta(days=365)


def call_arguments(call: GeneratedToolCall) -> JsonObjectValue:
    match call:
        case SearchCall(query=query):
            return JsonObjectValue(
                fields=(JsonObjectField(name="query", value=JsonStringValue(value=query)),)
            )
        case CalculateCall(expression=expression):
            return JsonObjectValue(
                fields=(
                    JsonObjectField(
                        name="expression",
                        value=JsonStringValue(value=expression),
                    ),
                )
            )
        case GetTimeCall():
            return JsonObjectValue(fields=())


def calculate_expression(expression: str) -> str:
    if not expression.strip():
        raise ValueError("Calculation expression cannot be empty.")
    if len(expression) > 160:
        raise ValueError("Calculation expression exceeds 160 characters.")
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("Calculation expression is invalid.") from error
    try:
        result = _evaluate_node(parsed.body)
    except (DivisionByZero, InvalidOperation, OverflowError, ZeroDivisionError) as error:
        raise ValueError("Calculation expression cannot be evaluated safely.") from error
    if not result.is_finite() or abs(result) > MAXIMUM_ABSOLUTE_RESULT:
        raise ValueError("Calculation result is outside the supported range.")
    return _format_decimal(result)


def seeded_time(random_seed: int, reference_time: datetime) -> str:
    if reference_time.tzinfo is None or reference_time.utcoffset() is None:
        raise ValueError("Time reference must include a UTC offset.")

    generator = random.Random(random_seed)
    reference_utc = reference_time.astimezone(UTC).replace(microsecond=0)
    seconds_ago = generator.randrange(0, int(TIME_SAMPLE_WINDOW.total_seconds()))
    utc_offset_hours = generator.randrange(-11, 15)
    target_timezone = timezone(timedelta(hours=utc_offset_hours))
    timestamp = reference_utc - timedelta(seconds=seconds_ago)
    return timestamp.astimezone(target_timezone).replace(microsecond=0).isoformat()


def _evaluate_node(node: ast.expr) -> Decimal:
    match node:
        case ast.Constant(value=bool()):
            raise ValueError("Booleans are not valid calculation values.")
        case ast.Constant(value=int() as integer_value):
            return Decimal(integer_value)
        case ast.Constant(value=float() as float_value):
            return Decimal(str(float_value))
        case ast.BinOp(left=left, op=operation, right=right):
            left_value = _evaluate_node(left)
            right_value = _evaluate_node(right)
            match operation:
                case ast.Pow() if abs(right_value) > MAXIMUM_EXPONENT:
                    raise ValueError("Calculation exponent is outside the supported range.")
                case _:
                    pass
            function = _binary_operation(operation)
            return function(left_value, right_value)
        case ast.UnaryOp(op=operation, operand=operand):
            function = _unary_operation(operation)
            return function(_evaluate_node(operand))
        case _:
            raise ValueError("Calculation contains a disallowed expression.")


def _binary_operation(operation: ast.operator) -> BinaryOperation:
    match operation:
        case ast.Add():
            return lambda left, right: left + right
        case ast.Sub():
            return lambda left, right: left - right
        case ast.Mult():
            return lambda left, right: left * right
        case ast.Div():
            return lambda left, right: left / right
        case ast.Mod():
            return lambda left, right: left % right
        case ast.Pow():
            return lambda left, right: left**right
        case _:
            raise ValueError("Calculation contains a disallowed binary operator.")


def _unary_operation(operation: ast.unaryop) -> UnaryOperation:
    match operation:
        case ast.UAdd():
            return lambda value: value
        case ast.USub():
            return lambda value: -value
        case _:
            raise ValueError("Calculation contains a disallowed unary operator.")


def _format_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return format(normalized, "f")
    return format(normalized, "f").rstrip("0").rstrip(".")
