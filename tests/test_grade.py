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


def test_last_boxed_nested():
    assert last_boxed("\\boxed{\\sqrt{\\frac{1}{2}}}") == "\\sqrt{\\frac{1}{2}}"
