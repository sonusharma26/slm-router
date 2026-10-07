import json

from inference_control.benchmarks.coding import CodingTask, prepare_tasks, score_response, task_messages


def _response(expression):
    return json.dumps({'expression': expression})


def test_frozen_population_references_and_prompt_bucket():
    tasks = prepare_tasks()
    assert len(tasks) == 160
    assert {split: sum(task.split == split for task in tasks)
            for split in ('train', 'calibration', 'validation', 'test')} == {
                'train': 16, 'calibration': 96, 'validation': 16, 'test': 32,
            }
    assert len({task.prompt for task in tasks}) == 160
    assert len({task.group_id for task in tasks}) == 160
    assert len({task.family for task in tasks}) == 4
    for task in tasks:
        score = score_response(task, _response(task.reference_expression))
        assert (score.quality, score.passed, score.total, score.rejection) == (1.0, 16, 16, None)
        assert 1024 <= len(json.dumps(task_messages(task), ensure_ascii=False).encode()) < 2048
        assert task_messages(task) == [{'role': 'user', 'content': task.prompt}]
        assert score_response(task, _response('x')).quality == 0.0
    small = prepare_tasks(counts={'train': 2, 'calibration': 0, 'validation': 0, 'test': 0})
    assert small == tasks[:2]
    assert [task.prompt for task in prepare_tasks(seed=7)] != [task.prompt for task in tasks]


def test_hidden_cases_require_whole_task_correctness():
    task = CodingTask('example', 'example', 'test', 'numeric', 'Transform all integers.',
                      (([], []), ([-2, 0, 2, 2], [5, 5]), ([1, 3], [3, 7])), '')
    score = score_response(task, _response('sorted([v * 2 + 1 for v in x if v > 0])'))
    assert (score.quality, score.passed, score.total, score.rejection) == (1.0, 3, 3, None)
    partial = score_response(task, _response('[]'))
    assert (partial.quality, partial.passed, partial.total) == (0.0, 1, 3)
    bool_task = CodingTask('boolean', 'boolean', 'test', 'boolean', '', (([], 1),), '')
    assert score_response(bool_task, _response('True')).quality == 0.0


def test_compact_prompts_preserve_tasks_and_hidden_scores():
    verbose = prepare_tasks(seed=43)
    compact = prepare_tasks(seed=43, prompt_style='compact')
    for original, shortened in zip(verbose, compact, strict=True):
        assert original.group_id == shortened.group_id
        assert original.cases == shortened.cases
        assert original.reference_expression == shortened.reference_expression
        assert 512 <= len(json.dumps(task_messages(shortened), separators=(',', ':')).encode()) < 1024
        assert score_response(shortened, _response(shortened.reference_expression)).quality == 1


def test_primitive_expressions_support_safe_methods_and_slices():
    task = CodingTask('strings', 'strings', 'test', 'strings', '',
                      (([' A ', 'bb', 'a'], ['BB', 'A']), ([], [])), '')
    expression = 'sorted(list(set([v.strip().upper() for v in x])), reverse=True)[:2]'
    score = score_response(task, _response(expression))
    assert (score.quality, score.passed, score.total) == (1.0, 2, 2)
    methods = CodingTask('methods', 'methods', 'test', 'methods', '', (({}, ['a', 'b']),), '')
    score = score_response(methods, _response("'a,b'.replace(',', ' ').split()"))
    assert (score.quality, score.passed) == (1.0, 1)


def test_untrusted_syntax_envelopes_and_allocations_are_rejected():
    task = CodingTask('safe', 'safe', 'test', 'safe', '', (([1, 2], []),), '')
    dangerous = (
        "__import__('os').system('echo unsafe')", 'x.__class__', '(lambda: 0)()',
        '(v for v in x)', '2 ** 999999999', '[v for x in x]',
        'sorted(x, key=str)', "int('9' * 100)", "'a' * 999999999", '[0] * 129',
        "('a ' * 128).split() + ['overflow']", "'x'.replace('', 'a' * 2048)",
        '1000000000 * 1000000000', '[[v for v in x] for v in x] * 129',
        "' '.join(['a' * 2048, 'b'])", "'ß' * 2048 + 'x'",
        "('İ' * 2048).lower()", "('ß' * 2048).upper()", '[[[[0] * 128] * 128] * 128] * 128',
    )
    for expression in dangerous:
        score = score_response(task, _response(expression))
        assert score.quality == 0.0
        assert score.rejection in {'unsupported', 'resource_limit', 'runtime'}
    for text in ('not JSON', '{"expression":"[]","expression":"x"}',
                 '{"expression":"[]","extra":1}', '{"expression":1}'):
        score = score_response(task, text)
        assert score.quality == 0.0
        assert score.rejection in {'invalid_response', 'runtime'}
    oversized = score_response(task, _response('1' * 4097))
    assert oversized.rejection == 'resource_limit'
