"""Tests for discrete action definitions and ViZDoom button mapping.

Regression focus: the old static action table produced all-zero (no-op) rows and
identical duplicate rows for scenarios such as ``basic`` and
``defend_the_center``, which wasted most of the action space.
"""

from collections.abc import Sequence

import pytest
import vizdoom as vzd

from src import actions
from src.actions import (
    ActionDef,
    build_action_matrix,
    get_action_definitions,
    get_action_names,
)

BASIC_BUTTONS = [vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK]
DEFEND_BUTTONS = [vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK]
FULL_BUTTONS = [
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.ATTACK,
    vzd.Button.USE,
]


def assert_meaningful(available: Sequence[vzd.Button]) -> None:
    matrix, names = build_action_matrix(available)
    assert len(matrix) == len(names)
    rows = [tuple(row) for row in matrix]
    assert len(set(rows)) == len(rows), f"duplicate action rows for {names}"
    for row, name in zip(rows, names, strict=True):
        assert sum(row) > 0, f"action {name!r} presses no button"
        assert len(row) == len(available)
    for row in rows:
        for pressed in row:
            assert pressed in (0, 1)


def test_basic_actions_are_meaningful():
    """`basic` only exposes strafe left/right + attack; no dead 'forward' rows."""
    names = get_action_names(BASIC_BUTTONS)
    assert names == ["strafe_left", "strafe_right", "attack"]


def test_defend_the_center_actions_are_meaningful():
    names = get_action_names(DEFEND_BUTTONS)
    assert names == ["turn_left", "turn_right", "attack"]


def test_full_button_set_uses_whole_catalogue():
    names = get_action_names(FULL_BUTTONS)
    assert names == [
        "forward",
        "backward",
        "turn_left",
        "turn_right",
        "strafe_left",
        "strafe_right",
        "forward_left",
        "forward_right",
        "attack",
        "use",
    ]


@pytest.mark.parametrize(
    "available",
    [
        BASIC_BUTTONS,
        DEFEND_BUTTONS,
        FULL_BUTTONS,
        [vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.MOVE_FORWARD],
    ],
)
def test_matrices_have_no_dead_or_duplicate_rows(available):
    assert_meaningful(available)


def test_action_vector_matches_buttons():
    matrix, names = build_action_matrix(BASIC_BUTTONS)
    index = dict(zip(names, matrix, strict=True))
    # buttons order: MOVE_LEFT, MOVE_RIGHT, ATTACK
    assert index["strafe_left"] == [1, 0, 0]
    assert index["strafe_right"] == [0, 1, 0]
    assert index["attack"] == [0, 0, 1]


def test_two_button_action_requires_both_buttons():
    matrix, names = build_action_matrix([vzd.Button.MOVE_FORWARD])
    assert names == ["forward"]
    assert matrix == [[1]]


def test_unsupported_buttons_raise_a_helpful_error():
    with pytest.raises(ValueError, match="No catalogue action is available"):
        build_action_matrix([vzd.Button.RELOAD])


def test_duplicate_catalogue_entries_are_rejected(monkeypatch):
    """The validator must fail loudly instead of training on identical actions."""
    duplicate_catalogue = (
        ActionDef("a", (vzd.Button.MOVE_FORWARD,)),
        ActionDef("b", (vzd.Button.MOVE_FORWARD,)),
    )
    monkeypatch.setattr(actions, "ACTION_CATALOGUE", duplicate_catalogue)
    with pytest.raises(ValueError, match="same buttons"):
        build_action_matrix([vzd.Button.MOVE_FORWARD])


def test_unfiltered_catalogue_is_returned_without_buttons():
    definitions: list[ActionDef] = get_action_definitions(None)
    assert definitions == list(actions.ACTION_CATALOGUE)
    assert len({definition.name for definition in definitions}) == len(definitions)
