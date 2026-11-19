"""Tests for discrete action definitions and ViZDoom button mapping."""

import vizdoom as vzd

from src.actions import build_action_matrix, get_action_names


def test_default_action_names():
    names = get_action_names(include_use_attack=False)
    assert names == ["forward", "left", "right", "forward_left", "forward_right"]


def test_optional_action_names():
    names = get_action_names(include_use_attack=True)
    assert len(names) == 9
    assert "use" in names
    assert "attack" in names
    assert "backward" in names
    assert "backward_attack" in names


def test_build_action_matrix_mapping():
    available = [vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.MOVE_FORWARD]
    matrix = build_action_matrix(available, include_use_attack=False)
    assert len(matrix) == 5
    # forward -> [0, 0, 1]
    assert matrix[0] == [0, 0, 1]
    # left -> [1, 0, 0]
    assert matrix[1] == [1, 0, 0]
    # right -> [0, 1, 0]
    assert matrix[2] == [0, 1, 0]
    # forward_left -> [1, 0, 1]
    assert matrix[3] == [1, 0, 1]
    # forward_right -> [0, 1, 1]
    assert matrix[4] == [0, 1, 1]
