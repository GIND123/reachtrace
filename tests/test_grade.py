import pytest

from reachtrace.sample.grade import extract_answer, grade, last_boxed


@pytest.mark.parametrize(
    "text,expected",
    [
        ("work...\nAnswer: 34", "34"),
        ("Answer: 1\nWait, recheck.\nAnswer: $\\frac{3}{4}$", "$\\frac{3}{4}$"),
        ("so the result is \\boxed{\\frac{a}{b}} done", "\\frac{a}{b}"),
        ("Answer:\n\n  12", "12"),
    ],
)
def test_extract(text, expected):
    assert extract_answer(text)[0] == expected


@pytest.mark.parametrize(
    "text,gold,ok",
    [
        ("Answer: 34", "34", True),
        ("Answer: $34$.", "34", True),
        ("Answer: \\frac{6}{8}", "3/4", True),
        ("Answer: 0.75", "\\dfrac{3}{4}", True),
        ("Answer: 1,000", "1000", True),
        ("Answer: 35", "34", False),
        ("no final answer here", "34", False),
        ("Answer: 90^\\circ", "90", True),
    ],
)
def test_grade(text, gold, ok):
    assert grade(text, gold).correct is ok


def test_new_problem_is_cut_before_extraction():
    text = "x = 34\nAnswer: 34\n\nProblem 2: Compute 5+5.\nAnswer: 10"
    assert grade(text, "34").correct
    assert not grade(text, "34", truncate=False).correct


@pytest.mark.parametrize(
    "text,cut",
    [
        (".\nProblem\nFind x.", "."),
        ("work\n## Problem 3. Next", "work"),
        ("ok\nSolve the following problem step by step.", "ok"),
        ("The problem asks for x.\nProblems like this are common.", None),
    ],
)
def test_truncate_new_problem(text, cut):
    from reachtrace.sample.grade import truncate_new_problem

    assert truncate_new_problem(text) == (text if cut is None else cut)


def test_lenient_accepts_answer_without_answer_line():
    from reachtrace.sample.grade import grade_lenient

    assert grade_lenient("so the total is 1251 students.", "1251")
    assert grade_lenient("Answer: 7\nhmm, actually\nAnswer: 1251", "7")
    assert not grade_lenient("the total is 1250", "1251")


def test_last_boxed_nested():
    assert last_boxed("\\boxed{\\sqrt{\\frac{1}{2}}}") == "\\sqrt{\\frac{1}{2}}"
