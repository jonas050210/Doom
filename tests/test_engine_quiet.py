"""Regression test for the engine-console spam reported on Windows.

ViZDoom's controller only passes ``-stdout`` on Windows, and that flag is what
makes the engine child write its start-up banner into the parent terminal.
Passing ``-stdout`` ourselves plus ``set_console_enabled(True)`` reproduces the
same output on Linux, so the fix can be tested without a Windows machine.

What we assert:

* without the quiet config the banner really is printed (the harness is valid);
* with ``silent_engine_ini_path()`` the WAD list and the
  ``... is write protected.`` lines are gone;
* the few lines the engine prints *before* it reads any configuration
  (``D_DoomInit``/``M_LoadDefaults``) are still there, which is the documented
  floor - so this test also fails loudly if upstream ever improves/shrinks it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

PROBE = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {root!r})

    import os
    import vizdoom as vzd
    from src.env import silent_engine_ini_path

    quiet = sys.argv[1] == "quiet"
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
    game.set_window_visible(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.add_game_args("-stdout")          # what ViZDoom always adds on Windows
    game.set_console_enabled(True)         # ... and what makes it reach the terminal
    if quiet:
        game.set_doom_config_path(silent_engine_ini_path())
    game.init()
    game.close()
    """
)


def _engine_output(mode: str, root: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(root=root), mode],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    return result.stdout + result.stderr


@pytest.fixture(scope="module")
def probe_root() -> str:
    import os

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.slow
def test_quiet_config_silences_the_engine_banner(probe_root):
    loud = _engine_output("loud", probe_root)
    quiet = _engine_output("quiet", probe_root)

    # The harness reproduces the Windows symptom.
    assert "W_Init: Init WADfiles." in loud
    assert "viz_controlled is write protected." in loud

    # The fix removes everything from the first config read onwards.
    assert "W_Init: Init WADfiles." not in quiet
    assert "adding " not in quiet
    assert "is write protected." not in quiet
    assert "I_Init: Setting up machine state." not in quiet

    # These precede the engine's first config read and cannot be suppressed;
    # they are the documented residue.
    assert "D_DoomInit:" in quiet
    assert "M_LoadDefaults: Load system defaults." in quiet

    # And the quiet run must not be quieter in a *lossy* way: it is a real boot.
    assert len(quiet.splitlines()) < len(loud.splitlines())
