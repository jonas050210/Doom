"""Discrete action space definitions and mapping for ViZDoom."""

from typing import Dict, List, Sequence
import vizdoom as vzd


BASE_ACTIONS: List[Dict[str, object]] = [
    {"name": "forward", "buttons": [vzd.Button.MOVE_FORWARD]},
    {"name": "left", "buttons": [vzd.Button.TURN_LEFT]},
    {"name": "right", "buttons": [vzd.Button.TURN_RIGHT]},
    {
        "name": "forward_left",
        "buttons": [vzd.Button.MOVE_FORWARD, vzd.Button.TURN_LEFT],
    },
    {
        "name": "forward_right",
        "buttons": [vzd.Button.MOVE_FORWARD, vzd.Button.TURN_RIGHT],
    },
]

OPTIONAL_ACTIONS: List[Dict[str, object]] = [
    {"name": "use", "buttons": [vzd.Button.USE]},
    {"name": "attack", "buttons": [vzd.Button.ATTACK]},
]


def get_action_definitions(include_use_attack: bool = False) -> List[Dict[str, object]]:
    """Return the list of discrete action definitions."""
    actions = list(BASE_ACTIONS)
    if include_use_attack:
        actions.extend(OPTIONAL_ACTIONS)
    return actions


def get_action_names(include_use_attack: bool = False) -> List[str]:
    """Return human-readable names for each discrete action."""
    return [str(a["name"]) for a in get_action_definitions(include_use_attack)]


def build_action_matrix(
    available_buttons: Sequence[vzd.Button],
    include_use_attack: bool = False,
) -> List[List[int]]:
    """Map discrete action indices to ViZDoom button vectors.

    Each row in the returned matrix has length ``len(available_buttons)`` and
    contains 1s for the buttons pressed in that discrete action, 0s otherwise.
    """
    button_index = {btn: idx for idx, btn in enumerate(available_buttons)}
    matrix: List[List[int]] = []

    for action_def in get_action_definitions(include_use_attack):
        vec = [0] * len(available_buttons)
        for btn in action_def["buttons"]:  # type: ignore[attr-defined]
            if btn in button_index:
                vec[button_index[btn]] = 1
        matrix.append(vec)

    return matrix
