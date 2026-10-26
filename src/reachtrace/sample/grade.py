"""Answer extraction and equivalence checking for RL-Zero-format responses.

Extraction follows the training reward: text after the last "Answer:". A trailing
\\boxed{} is the fallback. Equivalence tries normalized string match, then numeric match,
then math-verify when installed (`pip install reachtrace[grade]`).

Base models often finish a solution and then write a new problem of their own ("\\nProblem
2: ..."), sometimes with its own "Answer:" line. Responses are therefore cut at the first
new-problem marker before extraction, for both models. `grade_lenient` accepts any of
several candidate answers; it is a robustness check, used so that a problem only counts
as unsolved by the base if it is unsolved under both graders.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

_ANSWER_RE = re.compile(r"Answer\s*:")
_NEW_PROBLEM_RE = re.compile(
    r"\n[ \t]*(?:#+[ \t]*)?(?:Problem(?:[ \t]*\d+)?[ \t]*[:.\n]|Solve the following|Question[ \t]*:"
    r"|Remember to put your answer)"
)
_NUMBER_RE = re.compile(r"-?\\frac\{-?\d+\}\{-?\d+\}|-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")
_FRAC_RE = re.compile(r"^-?\\frac\{(-?\d+)\}\{(-?\d+)\}$")
_STRIP = ["\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\ ", "$", " "]


@dataclass(frozen=True)
class Grade:
    extracted: str | None
    correct: bool
    method: str  # how the answer was extracted and matched, for auditing


def last_boxed(text: str) -> str | None:
    """Contents of the last \\boxed{...} or \\fbox{...}, with brace matching."""
    start = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if start < 0:
        return None
    i = text.index("{", start) + 1
    depth = 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None


def extract_answer(text: str) -> tuple[str | None, str]:
    matches = list(_ANSWER_RE.finditer(text))
    if matches:
        rest = text[matches[-1].end() :]
        lines = [ln.strip() for ln in rest.split("\n")]
        line = next((ln for ln in lines if ln), "")
        if line:
            return line, "answer_line"
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed.strip(), "boxed"
    return None, "none"


def normalize(ans: str) -> str:
    s = ans.strip()
    if (b := last_boxed(s)) is not None:
        s = b
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "")
    s = s.replace("\\%", "").replace("%", "")
    for tok in _STRIP:
        s = s.replace(tok, "")
    s = s.rstrip(".")
    return s


def _as_number(s: str) -> Fraction | None:
    s = s.replace(",", "")
    if m := _FRAC_RE.match(s):
        num, den = int(m.group(1)), int(m.group(2))
        if den == 0:
            return None
        return Fraction(num, den) * (-1 if s.startswith("-\\frac") else 1)
    try:
        if "/" in s:
            num, den = s.split("/", 1)
            return Fraction(int(num), int(den))
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def _math_verify(pred: str, gold: str) -> bool | None:
    try:
        from math_verify import parse, verify
    except ImportError:
        return None
    try:
        return bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))
    except Exception:  # math-verify raises parser-specific errors on malformed LaTeX
        return None


def is_equiv(pred: str, gold: str) -> tuple[bool, str]:
    p, g = normalize(pred), normalize(gold)
    if p == g:
        return True, "exact"
    pn, gn = _as_number(p), _as_number(g)
    if pn is not None and gn is not None:
        return pn == gn, "numeric"
    mv = _math_verify(pred, gold)
    if mv is not None:
        return mv, "math_verify"
    return False, "no_match"


def truncate_new_problem(text: str) -> str:
    """Cut a response at the first marker of a new, self-generated problem."""
    m = _NEW_PROBLEM_RE.search(text)
    return text[: m.start()] if m else text


def grade(text: str, gold: str, truncate: bool = True) -> Grade:
    extracted, how = extract_answer(truncate_new_problem(text) if truncate else text)
    if extracted is None:
        return Grade(None, False, how)
    ok, match = is_equiv(extracted, gold)
    return Grade(extracted, ok, f"{how}+{match}")


def answer_candidates(text: str) -> list[str]:
    """First and last answer lines, the last \\boxed{}, and the last number in the response."""
    body = truncate_new_problem(text)
    out = []
    matches = list(_ANSWER_RE.finditer(body))
    for m in matches[:1] + matches[-1:]:
        line = next((ln.strip() for ln in body[m.end() :].split("\n") if ln.strip()), "")
        if line:
            out.append(line)
    if (b := last_boxed(body)) is not None:
        out.append(b.strip())
    nums = _NUMBER_RE.findall(body)
    if nums:
        out.append(nums[-1])
    return list(dict.fromkeys(out))


def grade_lenient(text: str, gold: str) -> bool:
    """True if any answer candidate matches; an upper bound on correctness, not the label."""
    return any(is_equiv(c, gold)[0] for c in answer_candidates(text))
