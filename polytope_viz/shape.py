import ast
import operator
from collections.abc import Sequence

_ALLOWED_BINOPS = {ast.Mult: operator.mul, ast.Add: operator.add}


def _eval_shape_node(node):
    if isinstance(node, ast.Expression):
        return _eval_shape_node(node.body)
    if isinstance(node, ast.List):
        values = [_eval_shape_node(elt) for elt in node.elts]
        if not all(isinstance(v, int) and v > 0 for v in values):
            raise ValueError('shape lists must contain positive integers')
        return values
    if isinstance(node, ast.Tuple):
        values = [_eval_shape_node(elt) for elt in node.elts]
        if not all(isinstance(v, int) and v > 0 for v in values):
            raise ValueError('shape tuples must contain positive integers')
        return values
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        if node.value <= 0:
            raise ValueError('shape integers must be positive')
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_shape_node(node.left)
        right = _eval_shape_node(node.right)
        if isinstance(node.op, ast.Mult):
            if isinstance(left, list) and isinstance(right, int):
                return left * right
            if isinstance(left, int) and isinstance(right, list):
                return left * right
        result = _ALLOWED_BINOPS[type(node.op)](left, right)
        if isinstance(result, list):
            return result
        if isinstance(result, int) and result > 0:
            return result
    raise ValueError('shape must be a list of positive integers, e.g. "[10]*8" or "[8, 8]"')


def parse_layer_sizes(shape: str | Sequence[int]) -> list[int]:
    if isinstance(shape, str):
        parsed = _eval_shape_node(ast.parse(shape, mode='eval'))
    else:
        parsed = list(shape)
    if isinstance(parsed, int):
        parsed = [parsed]
    if not isinstance(parsed, list) or not all(isinstance(v, int) and v > 0 for v in parsed):
        raise ValueError('shape must resolve to a list of positive integers')
    return parsed
