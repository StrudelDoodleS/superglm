"""Typed Numba helpers remain visible without pretending to be dispatchers."""

import pytest

from tests._numba_dispatchers import entry_point_sites


@pytest.mark.parametrize(
    "decoration",
    [
        "register_jitable",
        "register_jitable(inline='always')",
        "numba.extending.register_jitable",
        "typed_helper",
        "overload(target)",
    ],
)
def test_module_level_typed_registrations_are_classified(tmp_path, decoration):
    (tmp_path / "helpers.py").write_text(
        "import numba\n"
        "from numba import njit\n"
        "from numba.extending import overload, register_jitable\n"
        "from numba.extending import register_jitable as typed_helper\n"
        "def target(x): return x\n"
        f"@{decoration}\n"
        "def helper(x): return x\n"
        "@njit\n"
        "def kernel(x): return x\n"
    )
    sites, problems = entry_point_sites(tmp_path, package="fixture")
    assert not problems
    assert sites == {("fixture.helpers", "kernel"): "njit"}


def test_nested_registration_still_fails_the_census(tmp_path):
    (tmp_path / "helpers.py").write_text(
        "from numba.extending import register_jitable\n"
        "def outer():\n"
        "    @register_jitable\n"
        "    def hidden(x): return x\n"
        "    return hidden\n"
    )
    _, problems = entry_point_sites(tmp_path, package="fixture")
    assert len(problems) == 1
    assert "register_jitable outside a module-level kernel definition" in problems[0]


def test_unclassified_compiler_entry_point_still_fails_the_census(tmp_path):
    (tmp_path / "helpers.py").write_text(
        "from numba import vectorize\n@vectorize\ndef kernel(x): return x\n"
    )
    _, problems = entry_point_sites(tmp_path, package="fixture")
    assert any("vectorize makes no dispatcher the census classifies" in p for p in problems)
