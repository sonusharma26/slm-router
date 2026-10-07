from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import operator
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CodingTask:
    task_id: str
    group_id: str
    split: str
    family: str
    prompt: str
    cases: tuple[tuple[Any, Any], ...]
    reference_expression: str


@dataclass(frozen=True)
class Score:
    quality: float
    passed: int
    total: int
    rejection: str | None


_GRAMMAR = (
    'Write one bounded Python expression that transforms the JSON input named x. '
    'Reply with exactly one JSON object {"expression":"your expression"}. '
    'Do not include markdown, explanations, function definitions, assignments, or imports. '
    'The expression is interpreted, never executed as Python code. '
    'Allowed syntax is literals, x, local comprehension names, lists, tuples, dictionaries, '
    'arithmetic + - * / // %, comparisons == != < <= > >= in not in, and/or/not, '
    'conditional expressions, indexing, slices, and list comprehensions with optional filters. '
    'Allowed builtin calls are len, sorted, sum, min, max, abs, list, set, int, str. '
    'sorted accepts reverse=True or reverse=False but no key function. '
    'Safe string methods are lower, upper, strip, split, count, replace, startswith, endswith, join. '
    'Dictionary get, keys, values, items and list count/index are also supported. '
    'No other attributes, lambdas, generators, powers, reflection, or calls are allowed. '
    'Limits per input are 128 AST nodes, depth 16, 10000 evaluation steps, '
    '128 entries per collection, 2048 characters per string, and numeric magnitude at most 1000000000. '
    'The expression must be at most 4096 characters. Inputs contain at most 24 items. '
    'Return a JSON-compatible value. Preserve duplicates unless the specification says otherwise. '
    'Hidden tests include empty inputs, boundary values, repeated items and negative numbers where valid. '
    'Implement the complete rule for all valid inputs.\n\n'
)

_COMPACT_GRAMMAR = (
    'Return only JSON {"expression":"..."} containing a Python expression over input x. '
    'Use literals, arithmetic, comparisons, indexing, list comprehensions, len, sorted, sum, abs, '
    'and string strip/lower/upper methods. Keep duplicates; do not use set. '
    "JSON objects become Python dictionaries; use item['key'], not item.key. "
    'Use list comprehensions inside sorted and sum, never generators. '
    'Do not use imports, functions, lambdas, assignments, markdown or explanations. '
    'Implement the complete specification, including empty input and boundary cases.\n\n'
)


class _Rejected(ValueError):
    pass


_BUILTINS = frozenset({'len', 'sorted', 'sum', 'min', 'max', 'abs', 'list', 'set', 'int', 'str'})
_METHODS = frozenset({
    'lower', 'upper', 'strip', 'split', 'count', 'replace', 'startswith', 'endswith', 'join',
    'get', 'keys', 'values', 'items', 'index',
})
_COMPARISONS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
}
_NODES = (
    ast.Expression, ast.Constant, ast.Name, ast.Load, ast.Store, ast.List, ast.Tuple, ast.Dict,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Subscript, ast.Slice,
    ast.ListComp, ast.comprehension, ast.Call, ast.Attribute, ast.keyword,
    ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or, *_OPERATORS, *_COMPARISONS,
)


def _parse(expression: str) -> ast.Expression:
    if not expression or len(expression) > 4096:
        raise _Rejected('resource_limit')
    try:
        tree = ast.parse(expression, mode='eval')
    except (SyntaxError, ValueError, RecursionError):
        raise _Rejected('syntax') from None
    stack = [(tree, 1)]
    count = 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > 128 or depth > 16:
            raise _Rejected('resource_limit')
        if type(node) not in _NODES:
            raise _Rejected('unsupported')
        if isinstance(node, ast.Name) and (not node.id.isidentifier() or node.id.startswith('_')):
            raise _Rejected('unsupported')
        if isinstance(node, ast.Attribute) and node.attr not in _METHODS:
            raise _Rejected('unsupported')
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _BUILTINS:
                    raise _Rejected('unsupported')
            elif not isinstance(node.func, ast.Attribute):
                raise _Rejected('unsupported')
            if node.keywords and (
                not isinstance(node.func, ast.Name) or node.func.id != 'sorted'
                or len(node.keywords) != 1 or node.keywords[0].arg != 'reverse'
            ):
                raise _Rejected('unsupported')
        if isinstance(node, ast.comprehension) and (
            node.is_async or not isinstance(node.target, ast.Name) or node.target.id == 'x'
            or node.target.id in _BUILTINS
        ):
            raise _Rejected('unsupported')
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return tree


def _bounded(value: Any, depth: int = 0, budget: list[int] | None = None) -> Any:
    budget = [10000] if budget is None else budget
    budget[0] -= 1
    if depth > 16 or budget[0] < 0:
        raise _Rejected('resource_limit')
    kind = type(value)
    if kind in (bool, type(None)):
        return value
    if kind in (int, float):
        if abs(value) > 1_000_000_000 or (kind is float and not math.isfinite(value)):
            raise _Rejected('resource_limit')
        return value
    if kind is str:
        if len(value) > 2048:
            raise _Rejected('resource_limit')
        return value
    if kind in (list, tuple, set, dict):
        if len(value) > 128:
            raise _Rejected('resource_limit')
        for item in value:
            _bounded(item, depth + 1, budget)
            if kind is dict:
                _bounded(value[item], depth + 1, budget)
        return value
    raise _Rejected('unsupported')


def _work(value: Any) -> int:
    budget = [10000]
    _bounded(value, budget=budget)
    return 10000 - budget[0]


class _Interpreter:
    def __init__(self) -> None:
        self.steps = 0

    def charge(self, count: int = 1) -> None:
        self.steps += count
        if self.steps > 10000:
            raise _Rejected('resource_limit')

    def iterable(self, value: Any) -> Any:
        if type(value) not in (list, tuple, set, dict, str):
            raise _Rejected('runtime')
        self.charge(len(value))
        return value

    def evaluate(self, node: ast.AST, names: dict[str, Any]) -> Any:
        self.charge()
        result = self._evaluate(node, names)
        self.charge(_work(result))
        return result

    def _evaluate(self, node: ast.AST, names: dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise _Rejected('unsupported')
            return names[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            if len(node.elts) > 128:
                raise _Rejected('resource_limit')
            values = [self.evaluate(item, names) for item in node.elts]
            return values if isinstance(node, ast.List) else tuple(values)
        if isinstance(node, ast.Dict):
            if len(node.keys) > 128 or any(key is None for key in node.keys):
                raise _Rejected('unsupported')
            return {self.evaluate(key, names): self.evaluate(value, names)
                    for key, value in zip(node.keys, node.values, strict=True)}
        if isinstance(node, ast.BinOp):
            left = self.evaluate(node.left, names)
            right = self.evaluate(node.right, names)
            return self.binary(type(node.op), left, right)
        if isinstance(node, ast.UnaryOp):
            value = self.evaluate(node.operand, names)
            if isinstance(node.op, ast.Not):
                return not value
            if type(value) not in (int, float):
                raise _Rejected('runtime')
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BoolOp):
            for child in node.values:
                value = self.evaluate(child, names)
                if isinstance(node.op, ast.And) and not value:
                    return value
                if isinstance(node.op, ast.Or) and value:
                    return value
            return value
        if isinstance(node, ast.Compare):
            left = self.evaluate(node.left, names)
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                right = self.evaluate(comparator, names)
                if isinstance(op, (ast.In, ast.NotIn)):
                    self.iterable(right)
                    self.charge(_work(left) * len(right) + _work(right))
                else:
                    self.charge(_work(left) + _work(right))
                if not _COMPARISONS[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            branch = node.body if self.evaluate(node.test, names) else node.orelse
            return self.evaluate(branch, names)
        if isinstance(node, ast.Subscript):
            value = self.evaluate(node.value, names)
            if type(value) not in (list, tuple, str, dict):
                raise _Rejected('runtime')
            if isinstance(node.slice, ast.Slice):
                if type(value) is dict:
                    raise _Rejected('runtime')
                bounds = [self.evaluate(part, names) if part is not None else None
                          for part in (node.slice.lower, node.slice.upper, node.slice.step)]
                if any(part is not None and type(part) is not int for part in bounds):
                    raise _Rejected('runtime')
                self.charge(len(value))
                return value[slice(*bounds)]
            index = self.evaluate(node.slice, names)
            if type(value) is not dict and type(index) is not int:
                raise _Rejected('runtime')
            return value[index]
        if isinstance(node, ast.ListComp):
            result: list[Any] = []
            self.comprehend(node, 0, names, result)
            return result
        if isinstance(node, ast.Call):
            args = [self.evaluate(arg, names) for arg in node.args]
            if isinstance(node.func, ast.Name):
                kwargs = {kw.arg: self.evaluate(kw.value, names) for kw in node.keywords}
                return self.builtin(node.func.id, args, kwargs)
            if isinstance(node.func, ast.Attribute):
                value = self.evaluate(node.func.value, names)
                return self.method(value, node.func.attr, args)
        raise _Rejected('unsupported')

    def comprehend(self, node: ast.ListComp, index: int, names: dict[str, Any], result: list[Any]) -> None:
        if index == len(node.generators):
            if len(result) >= 128:
                raise _Rejected('resource_limit')
            result.append(self.evaluate(node.elt, names))
            return
        generator = node.generators[index]
        for value in self.iterable(self.evaluate(generator.iter, names)):
            self.charge()
            local = {**names, generator.target.id: value}
            if all(self.evaluate(condition, local) for condition in generator.ifs):
                self.comprehend(node, index + 1, local, result)

    def binary(self, op: type[ast.operator], left: Any, right: Any) -> Any:
        if op is ast.Add and type(left) in (list, tuple, str) and type(left) is type(right):
            limit = 2048 if type(left) is str else 128
            if len(left) + len(right) > limit:
                raise _Rejected('resource_limit')
            self.charge(len(left) + len(right))
            return left + right
        if op is ast.Mult:
            if type(left) is int and type(right) in (list, tuple, str):
                left, right = right, left
            if type(left) in (list, tuple, str) and type(right) is int:
                limit = 2048 if type(left) is str else 128
                length = len(left) * max(0, right)
                if length > limit:
                    raise _Rejected('resource_limit')
                self.charge(length)
                return left * right
        if type(left) not in (int, float) or type(right) not in (int, float):
            raise _Rejected('runtime')
        if op is ast.Mult and right and abs(left) > 1_000_000_000 / abs(right):
            raise _Rejected('resource_limit')
        return _OPERATORS[op](left, right)

    def builtin(self, name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        if name == 'len' and len(args) == 1:
            return len(self.iterable(args[0]))
        if name in {'sorted', 'list', 'set'} and len(args) == 1:
            values = self.iterable(args[0])
            if len(values) > 128:
                raise _Rejected('resource_limit')
            if name == 'sorted':
                reverse = kwargs.get('reverse', False)
                if type(reverse) is not bool:
                    raise _Rejected('runtime')
                weight = max((_work(item) for item in values), default=1)
                self.charge(weight * len(values) * max(1, len(values).bit_length()))
                return sorted(values, reverse=reverse)
            return list(values) if name == 'list' else set(values)
        if name == 'sum' and 1 <= len(args) <= 2:
            value = args[1] if len(args) == 2 else 0
            for item in self.iterable(args[0]):
                value = _bounded(self.binary(ast.Add, value, item))
            return value
        if name in {'min', 'max'} and args:
            values = self.iterable(args[0]) if len(args) == 1 else args
            self.charge(max((_work(item) for item in values), default=1) * len(values))
            return min(values) if name == 'min' else max(values)
        if name == 'abs' and len(args) == 1 and type(args[0]) in (int, float):
            return abs(args[0])
        if name == 'int' and len(args) == 1 and type(args[0]) in (str, int, float, bool):
            value = args[0]
            if type(value) is str and len(value.strip().lstrip('+-')) > 10:
                raise _Rejected('resource_limit')
            return int(value)
        if name == 'str' and len(args) == 1 and type(args[0]) in (str, int, float, bool, type(None)):
            return str(args[0])
        raise _Rejected('runtime')

    def method(self, value: Any, name: str, args: list[Any]) -> Any:
        if type(value) is dict:
            self.charge(len(value))
            if name == 'get' and 1 <= len(args) <= 2:
                return value.get(args[0], args[1] if len(args) == 2 else None)
            if name in {'keys', 'values', 'items'} and not args:
                if name == 'keys':
                    return list(value)
                if name == 'values':
                    return list(value.values())
                return list(value.items())
        if type(value) is list and name in {'count', 'index'} and len(args) == 1:
            self.charge(_work(args[0]) * len(value) + _work(value))
            return value.count(args[0]) if name == 'count' else value.index(args[0])
        if type(value) is not str:
            raise _Rejected('runtime')
        self.charge(len(value))
        if name == 'lower' and not args:
            if sum(len(char.lower()) for char in value) > 2048:
                raise _Rejected('resource_limit')
            return value.lower()
        if name == 'upper' and not args:
            if sum(len(char.upper()) for char in value) > 2048:
                raise _Rejected('resource_limit')
            return value.upper()
        if name == 'strip' and len(args) <= 1 and all(type(arg) is str for arg in args):
            return value.strip(*args)
        if name in {'count', 'startswith', 'endswith'} and len(args) == 1 and type(args[0]) is str:
            if name == 'count':
                return value.count(args[0])
            return value.startswith(args[0]) if name == 'startswith' else value.endswith(args[0])
        if name == 'replace' and 2 <= len(args) <= 3:
            old, new = args[:2]
            if type(old) is not str or type(new) is not str:
                raise _Rejected('runtime')
            count = args[2] if len(args) == 3 else -1
            if type(count) is not int:
                raise _Rejected('runtime')
            occurrences = value.count(old)
            if count >= 0:
                occurrences = min(occurrences, count)
            if len(value) + occurrences * (len(new) - len(old)) > 2048:
                raise _Rejected('resource_limit')
            return value.replace(old, new, count)
        if name == 'split' and len(args) <= 2:
            separator = args[0] if args else None
            count = args[1] if len(args) == 2 else -1
            if (separator is not None and type(separator) is not str) or type(count) is not int:
                raise _Rejected('runtime')
            if separator == '':
                raise _Rejected('runtime')
            if separator is None:
                pieces = sum(not char.isspace() and (i == 0 or value[i - 1].isspace())
                             for i, char in enumerate(value))
            else:
                pieces = value.count(separator) + 1
            if count >= 0:
                pieces = min(pieces, count + 1)
            if pieces > 128:
                raise _Rejected('resource_limit')
            return value.split(separator, count)
        if name == 'join' and len(args) == 1:
            items = self.iterable(args[0])
            if any(type(item) is not str for item in items):
                raise _Rejected('runtime')
            if sum(len(item) for item in items) + len(value) * max(0, len(items) - 1) > 2048:
                raise _Rejected('resource_limit')
            return value.join(items)
        raise _Rejected('runtime')


def task_messages(task: CodingTask) -> list[dict[str, str]]:
    return [{'role': 'user', 'content': task.prompt}]


def _json_value(value: Any) -> Any:
    if type(value) in (list, tuple):
        return [_json_value(item) for item in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise _Rejected('runtime')
        return {key: _json_value(item) for key, item in value.items()}
    if type(value) not in (str, int, float, bool, type(None)):
        raise _Rejected('runtime')
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise _Rejected('invalid_response')
    return result


def score_response(task: CodingTask, text: str) -> Score:
    total = len(task.cases)
    passed = 0
    try:
        if len(text) > 16384:
            raise _Rejected('resource_limit')
        response = json.loads(text, object_pairs_hook=_unique_object)
        if type(response) is not dict or set(response) != {'expression'}:
            raise _Rejected('invalid_response')
        expression = response['expression']
        if type(expression) is not str:
            raise _Rejected('invalid_response')
        tree = _parse(expression)
        for value, expected in task.cases:
            result = _json_value(_Interpreter().evaluate(tree.body, {'x': _bounded(value)}))
            actual_json = json.dumps(result, sort_keys=True, allow_nan=False)
            expected_json = json.dumps(expected, sort_keys=True, allow_nan=False)
            passed += actual_json == expected_json
        return Score(float(total > 0 and passed == total), passed, total, None)
    except _Rejected as error:
        return Score(0.0, passed, total, str(error))
    except (TypeError, ValueError, KeyError, IndexError, ZeroDivisionError, OverflowError, RecursionError):
        return Score(0.0, passed, total, 'runtime')


def _numeric_inputs(rng: random.Random, boundaries: list[int]) -> list[list[int]]:
    return [[], [0], [-1], [1], boundaries, list(reversed(boundaries)), [0, 0, 1, -1, -1],
            [-24, -12, 12, 24], [2, 2, 2], [-3, -2, -1], [1, 2, 3],
            *[[rng.randint(-24, 24) for _ in range(length)] for length in (4, 7, 12, 20, 24)]]


def _task_spec(family: str, params: tuple[Any, ...], rng: random.Random) -> tuple[str, str, tuple]:
    if family == 'numeric_filter_map':
        threshold, factor, offset, above, reverse = params
        comparison = '>=' if above else '<='
        specification = (
            f'x is a list of integers in [-24,24]. Keep each v satisfying v {comparison} {threshold}. '
            f'Transform each retained item to v * {factor} + {offset}. '
            f'Return the transformed list sorted {"descending" if reverse else "ascending"}. '
            'Filter before transforming, and keep repeated values.'
        )
        expression = (f'sorted([v * {factor} + {offset} for v in x if v {comparison} {threshold}], '
                      f'reverse={reverse})')
        inputs = _numeric_inputs(rng, [threshold - 1, threshold, threshold + 1])
        cases = []
        for values in inputs:
            retained = [value for value in values if (value >= threshold if above else value <= threshold)]
            expected = sorted([value * factor + offset for value in retained], reverse=reverse)
            cases.append((values, expected))
    elif family == 'string_normalize_filter':
        minimum, blocked, upper, reverse = params
        method = 'upper' if upper else 'lower'
        token = blocked.upper() if upper else blocked.lower()
        specification = (
            'x is a list of ASCII strings of length at most 40. Strip surrounding whitespace from each '
            f'string and convert it to {"uppercase" if upper else "lowercase"}. '
            f'Keep normalized strings of length at least {minimum} that do not contain {token!r}. '
            f'Return them sorted {"descending" if reverse else "ascending"}, preserving duplicates. '
            'Apply both filters to the normalized string.'
        )
        expression = (f'sorted([v.strip().{method}() for v in x if len(v.strip()) >= {minimum} '
                      f'and {token!r} not in v.strip().{method}()], reverse={reverse})')
        words = ['', ' ', ' A ', 'a', 'B', ' Bb ', 'cat', 'DOG', 'foo', 'zz', 'q', '_', ':',
                 'Hello', '\tMiXeD\n', 'b' * max(0, minimum - 1), 'b' * minimum, 'b' * (minimum + 1),
                 blocked, f'  {blocked.upper()}  ']
        inputs = [[], [''], [' '], [blocked], [blocked.upper()], ['B', 'B', 'b'], words[:12],
                  list(reversed(words[:12])), ['b' * minimum], ['b' * max(0, minimum - 1)],
                  ['b' * (minimum + 1)], *[rng.choices(words, k=k) for k in (4, 7, 12, 20, 24)]]
        cases = []
        for values in inputs:
            normalized = [value.strip().upper() if upper else value.strip().lower() for value in values]
            expected = sorted([value for value in normalized if len(value) >= minimum and token not in value],
                              reverse=reverse)
            cases.append((values, expected))
    elif family == 'record_select_map':
        threshold, factor, offset, field, reverse = params
        other = 'weight' if field == 'score' else 'score'
        specification = (
            'x is a list of objects with integer fields score and weight, each in [-24,24]. '
            f'Keep objects whose {field} is strictly greater than {threshold}. '
            f'For each retained object compute its {other} * {factor} + {offset}. '
            f'Return these numbers sorted {"descending" if reverse else "ascending"}. '
            'Do not filter on the output field. Keep duplicates and return [] for an empty input.'
        )
        expression = (f'sorted([v[{other!r}] * {factor} + {offset} for v in x '
                      f'if v[{field!r}] > {threshold}], reverse={reverse})')
        numeric = _numeric_inputs(rng, [threshold - 1, threshold, threshold + 1])
        inputs = [[{'score': value, 'weight': max(-24, min(24, -value + (index % 3)))}
                   for index, value in enumerate(values)] for values in numeric]
        inputs[4] = [{field: threshold + delta, other: output}
                     for delta, output in ((-1, 20), (0, -20), (1, 7), (1, 7), (2, -9))]
        cases = []
        for values in inputs:
            selected = [record for record in values if record[field] > threshold]
            expected = sorted([record[other] * factor + offset for record in selected], reverse=reverse)
            cases.append((values, expected))
    else:
        modulus, remainder, factor, offset = params
        specification = (
            f'x is a list of integers in [-24,24]. Keep v when its Python remainder v % {modulus} '
            f'equals {remainder}. For each retained integer compute abs(v * {factor} + {offset}). '
            'Return the sum of these values as an integer, counting duplicates separately. '
            'Python remainder is nonnegative for a positive divisor, including when v is negative. '
            'Return 0 when no item qualifies.'
        )
        expression = (f'sum([abs(v * {factor} + {offset}) for v in x '
                      f'if v % {modulus} == {remainder}])')
        inputs = _numeric_inputs(rng, [remainder - modulus, remainder, remainder + modulus,
                                     remainder - 1, remainder + 1])
        cases = []
        for values in inputs:
            expected = 0
            for value in values:
                if value % modulus == remainder:
                    expected += abs(value * factor + offset)
            cases.append((values, expected))
    return specification, expression, tuple(cases)


def prepare_tasks(seed: int = 42, counts: dict[str, int] | None = None, *, prompt_style: str = 'verbose') -> list[CodingTask]:
    counts = counts if counts is not None else {'train': 16, 'calibration': 96, 'validation': 16, 'test': 32}
    splits = ('train', 'calibration', 'validation', 'test')
    if prompt_style not in {'verbose', 'compact'}:
        raise ValueError('unknown prompt style')
    if set(counts) - set(splits) or any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError('counts must contain nonnegative integer counts for known splits')
    rng = random.Random(seed)
    factors = (-3, -2, -1, 1, 2, 3)
    families = {
        'numeric_filter_map': list(itertools.product(range(-12, 13), factors, range(-6, 7),
                                                     (False, True), (False, True))),
        'string_normalize_filter': list(itertools.product(range(8),
                                                          ('a', 'e', 'x', 'oo', 'zz', 'cat', 'dog', '_', ':', 'q'),
                                                          (False, True), (False, True))),
        'record_select_map': list(itertools.product(range(-12, 13), factors, range(-6, 7),
                                                    ('score', 'weight'), (False, True))),
        'modular_abs_reduction': [(m, r, a, b) for m in range(2, 8) for r in range(m)
                                  for a in (1, 2, 3) for b in range(-6, 7)],
    }
    for choices in families.values():
        rng.shuffle(choices)
    names = list(families)
    total = sum(counts.values())
    if total > 4 * min(map(len, families.values())):
        raise ValueError('requested population exceeds unique specification capacity')
    tasks = []
    index = 0
    for split in splits:
        for _ in range(counts.get(split, 0)):
            family = names[index % len(names)]
            params = families[family].pop()
            specification, expression, cases = _task_spec(family, params, rng)
            prompt = (_GRAMMAR if prompt_style == 'verbose' else _COMPACT_GRAMMAR) + 'Specification. ' + specification
            task_id = f'coding-{seed}-{index:04d}'
            group_id = hashlib.sha256(json.dumps((family, specification)).encode()).hexdigest()
            task = CodingTask(task_id, group_id, split, family, prompt, cases, expression)
            size = len(json.dumps(task_messages(task), ensure_ascii=False).encode('utf-8'))
            minimum, maximum = (1024, 2048) if prompt_style == 'verbose' else (512, 1024)
            if not minimum <= size < maximum:
                raise ValueError(f'prompt leaves the frozen input bucket: {size}')
            reference_score = score_response(task, json.dumps({'expression': expression}))
            if reference_score.quality != 1.0:
                raise ValueError(f'unrepresentable trusted reference for {task_id}: {reference_score}')
            tasks.append(task)
            index += 1
    return tasks
