"""No editor module decides by a family's class or name, outside the places listed here.

The editor draws a term by its fitted link and reads parameter names from the family's
``ParameterSpec``. A family is named only where the browser has maths for that family
(the distribution panel and Explain tab) and where the library's distribution
re-profiling supports exactly the negative binomial's theta and the Tweedie power. A
family class or family string anywhere else under ``src/superglm/editor/`` fails here.

Python is read by tokens: a family class is a name token, a family string a string
token whose whole value is one. JavaScript is read line by line with comment lines
dropped: a family string is a quoted literal whose whole value is one. Prose in a
docstring, a comment or user text may name a family.
"""

from __future__ import annotations

import inspect
import io
import re
import tokenize
from pathlib import Path

import superglm.distributional.families as lss_families
import superglm.distributions as glm_distributions
from superglm.distributions import DISTRIBUTION_SHORTCUTS

EDITOR = Path(__file__).resolve().parents[1] / "src" / "superglm" / "editor"


def _family_classes() -> set[str]:
    lss = {
        name
        for name in lss_families.__all__
        if inspect.isclass(cls := getattr(lss_families, name)) and hasattr(cls, "parameters")
    }
    glm = {
        name
        for name, cls in vars(glm_distributions).items()
        if inspect.isclass(cls)
        and cls.__module__ == glm_distributions.__name__
        and cls is not glm_distributions.Distribution
        and hasattr(cls, "default_link")
    }
    return lss | glm


FAMILY_CLASSES = _family_classes()
# Compared lower-cased: the library's shortcuts, the parameterised families' names, the
# browser's fast-path names, the summary's display name, and every class name.
FAMILY_STRINGS = (
    set(DISTRIBUTION_SHORTCUTS)
    | {"tweedie", "negative_binomial", "nb2", "lognormal", "lognormal_mean", "neg. binomial"}
    | {name.lower() for name in FAMILY_CLASSES}
)

# (file under src/superglm/editor/, pattern the line matches, why it may name a family).
# A file ending in "/" allows every line under it. Each entry must still match a line.
ALLOWED = (
    ("app/distribution/", None, "the browser's laws for the fast-path families"),
    (
        "lss/panel.py",
        r"^    (GammaLS|LogNormalLS|NegativeBinomialLS|TweedieLSS),$",
        "imports the classes _FAST_PATHS keys on",
    ),
    (
        "lss/panel.py",
        r"^    (LogNormalLS|GammaLS|NegativeBinomialLS|TweedieLSS): _FastPath\(",
        "_FAST_PATHS: the families the browser has maths for",
    ),
    ("lss/panel.py", r'^        "lognormal_mean",$', "_FAST_PATHS: the log-normal's browser name"),
    (
        "lss/panel.py",
        r"^from superglm\.distributional\.families\.tweedie import BoundedPowerLink$",
        "a link the package does not export; the panel maps it to the browser's bounded logit",
    ),
    (
        "app/views/risk_panel.js",
        r'reply\.family === "tweedie"',
        "Explain draws the Tweedie's chance of zero from its fast-path law",
    ),
    (
        "app/views/distribution_panel.js",
        r'"(gamma|tweedie|lognormal_mean|negative_binomial)"',
        "the panel's fast-path laws: the gamma CV cap and the family cast",
    ),
    (
        "cv.py",
        r"(^from superglm\.distributions import NegativeBinomial$"
        r'|isinstance\(family, NegativeBinomial\) and family\.theta == "auto")',
        "refits the negative binomial's estimated theta, as profile_distribution supports",
    ),
    (
        "app/summary.js",
        r'family === "(Tweedie|Neg\. Binomial)"',
        "offers re-profiling for the two families profile_distribution supports",
    ),
    (
        "server.py",
        r'^\s+if parameter not in \{"tweedie", "tweedie_p", "p"\}:$',
        "the re-profile route's Tweedie power keys",
    ),
    *(
        (
            where,
            r'(\{"tweedie", "tweedie_p", "p"\}|\{"nb2", "nb2_theta", "negative_binomial", "theta"\})',
            "the keys naming the parameter profile_distribution re-profiles",
        )
        for where in ("session.py", "widget.py")
    ),
)


def _python_findings(text: str) -> list[tuple[int, str]]:
    found = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.NAME and token.string in FAMILY_CLASSES:
            found.append((token.start[0], token.string))
        elif token.type == tokenize.STRING:
            body = re.sub(r"^[rRbBuUfF]*('''|\"\"\"|'|\")|('''|\"\"\"|'|\")$", "", token.string)
            if body.lower() in FAMILY_STRINGS:
                found.append((token.start[0], token.string))
    # A family's own module (superglm.distributional.families.<family>) is a family by path.
    for number, line in enumerate(text.splitlines(), start=1):
        if re.search(r"distributional\.families\.(?!_links\b)\w+", line):
            found.append((number, line.strip()))
    return found


# One branch takes an escaped character, the other any other character but a backslash, so no
# string can be read two ways and the match stays linear in its length.
_JS_LITERAL = re.compile(r"""(["'`])((?:\\.|(?!\1)[^\\])*)\1""")
_JS_COMMENT = re.compile(r"^\s*(//|/\*|\*)")


def _js_findings(text: str) -> list[tuple[int, str]]:
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        if _JS_COMMENT.match(line):
            continue
        for match in _JS_LITERAL.finditer(line):
            if match.group(2).lower() in FAMILY_STRINGS:
                found.append((number, match.group(0)))
        code = _JS_LITERAL.sub('""', line.split("//")[0])
        found += [(number, name) for name in re.findall(r"\b\w+\b", code) if name in FAMILY_CLASSES]
    return found


def _findings():
    for path in sorted(EDITOR.rglob("*")):
        if path.suffix not in (".py", ".js") or "vendor" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        reader = _python_findings if path.suffix == ".py" else _js_findings
        relative = path.relative_to(EDITOR).as_posix()
        for number, token in reader(text):
            yield relative, number, lines[number - 1], token


def _allowed_by(relative: str, line: str) -> int | None:
    for index, (where, pattern, _why) in enumerate(ALLOWED):
        if where.endswith("/"):
            if relative.startswith(where):
                return index
        elif relative == where and re.search(pattern, line):
            return index
    return None


def test_editor_names_no_family():
    used = set()
    stray = []
    for relative, number, line, token in _findings():
        index = _allowed_by(relative, line)
        if index is None:
            stray.append(f"src/superglm/editor/{relative}:{number}: {token} in {line.strip()!r}")
        else:
            used.add(index)
    assert not stray, (
        "An editor module names a family; decide by the fitted link or the family's "
        "ParameterSpec, or add the place to ALLOWED with its reason:\n" + "\n".join(stray)
    )
    unused = [ALLOWED[index][:2] for index in range(len(ALLOWED)) if index not in used]
    assert not unused, f"ALLOWED entries that match no line any more: {unused}"


def test_the_rule_sees_every_exported_family():
    assert {"GammaLS", "TweedieLSS", "DirichletLSS", "Poisson", "Tweedie"} <= FAMILY_CLASSES
    assert _python_findings('x = "tweedie"\nif isinstance(f, GammaLS): pass\n') == [
        (1, '"tweedie"'),
        (2, "GammaLS"),
    ]
    assert _js_findings('if (reply.family === "gamma") {}\n// a "gamma" comment\n') == [
        (1, '"gamma"')
    ]


def test_the_js_literal_reader_reads_escaped_quotes_inside_a_literal():
    text = r"""const a = "say \"gamma\"" + 'it\'s' + `x\\`;"""
    assert [m.group(2) for m in _JS_LITERAL.finditer(text)] == [
        r"say \"gamma\"",
        r"it\'s",
        r"x\\",
    ]
