"""Discrete action definitions derived from the buttons ViZDoom exposes.

Each ViZDoom scenario declares its own ``available_buttons`` in its ``.cfg``
file and pressing a button that a scenario does not expose is a no-op.  A
static action table therefore produces action vectors that are silently empty
or identical (e.g. ``basic`` only exposes strafe left/right and attack, so a
"forward" action does nothing at all).

This module keeps a catalogue of *named* actions and projects it onto the
buttons a concrete scenario offers.  Actions that need a missing button are
dropped, and the resulting matrix is validated so that every action presses at
least one button and no two actions share the same button set.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import vizdoom as vzd


@dataclass(frozen=True)
class ActionDef:
    """A named discrete action and the ViZDoom buttons it presses."""

    name: str
    buttons: tuple[vzd.Button, ...]


# Ordered catalogue of candidate actions.  Order matters only for readability:
# single-button actions come before two-button combinations.
ACTION_CATALOGUE: tuple[ActionDef, ...] = (
    ActionDef("forward", (vzd.Button.MOVE_FORWARD,)),
    ActionDef("backward", (vzd.Button.MOVE_BACKWARD,)),
    ActionDef("turn_left", (vzd.Button.TURN_LEFT,)),
    ActionDef("turn_right", (vzd.Button.TURN_RIGHT,)),
    ActionDef("strafe_left", (vzd.Button.MOVE_LEFT,)),
    ActionDef("strafe_right", (vzd.Button.MOVE_RIGHT,)),
    ActionDef("forward_left", (vzd.Button.MOVE_FORWARD, vzd.Button.TURN_LEFT)),
    ActionDef("forward_right", (vzd.Button.MOVE_FORWARD, vzd.Button.TURN_RIGHT)),
    ActionDef("attack", (vzd.Button.ATTACK,)),
    ActionDef("use", (vzd.Button.USE,)),
)


def get_action_definitions(
    available_buttons: Sequence[vzd.Button] | None = None,
) -> list[ActionDef]:
    """Return catalogue actions that are fully supported and distinct.

    Args:
        available_buttons: Buttons the scenario exposes.  ``None`` returns the
            whole catalogue unfiltered (useful for documentation/tests).

    Raises:
        ValueError: If no catalogue action can be expressed with the given
            buttons, or if two actions collapse to the same button set.
    """
    if available_buttons is None:
        return list(ACTION_CATALOGUE)

    available = set(available_buttons)
    filtered = [
        action
        for action in ACTION_CATALOGUE
        if action.buttons and all(btn in available for btn in action.buttons)
    ]

    distinct: list[ActionDef] = []
    seen: dict[frozenset, str] = {}
    for action in filtered:
        key = frozenset(action.buttons)
        if key in seen:
            # A duplicate in the catalogue is a bug, not something to hide.
            raise ValueError(
                f"Catalogue actions '{seen[key]}' and '{action.name}' press the "
                f"same buttons {[b.name for b in action.buttons]}."
            )
        seen[key] = action.name
        distinct.append(action)

    if not distinct:
        offered = ", ".join(sorted(btn.name for btn in available)) or "<none>"
        raise ValueError(
            "No catalogue action is available for this scenario. "
            f"Scenario exposes: {offered}. "
            "Extend ACTION_CATALOGUE in src/actions.py."
        )
    return distinct


def get_action_names(
    available_buttons: Sequence[vzd.Button] | None = None,
) -> list[str]:
    """Return human-readable names for each discrete action."""
    return [action.name for action in get_action_definitions(available_buttons)]


def build_action_matrix(
    available_buttons: Sequence[vzd.Button],
) -> tuple[list[list[int]], list[str]]:
    """Map discrete action indices to ViZDoom button vectors.

    Returns:
        ``(matrix, names)`` where each row of ``matrix`` has length
        ``len(available_buttons)`` with 1s for pressed buttons.

    Raises:
        ValueError: If a generated action presses no button or if two
            actions produce identical vectors.
    """
    definitions = get_action_definitions(available_buttons)
    button_index = {btn: idx for idx, btn in enumerate(available_buttons)}

    matrix: list[list[int]] = []
    for action in definitions:
        vec = [0] * len(available_buttons)
        for btn in action.buttons:
            vec[button_index[btn]] = 1
        matrix.append(vec)

    _validate_action_matrix(matrix, definitions)
    return matrix, [action.name for action in definitions]


def _validate_action_matrix(
    matrix: Sequence[Sequence[int]], definitions: Sequence[ActionDef]
) -> None:
    """Fail loudly instead of training with dead or duplicated actions."""
    seen: dict[tuple[int, ...], str] = {}
    for vec, action in zip(matrix, definitions, strict=True):
        key = tuple(vec)
        if not any(key):
            raise ValueError(
                f"Action '{action.name}' presses no available button; "
                "it would be indistinguishable from doing nothing."
            )
        if key in seen:
            raise ValueError(
                f"Actions '{seen[key]}' and '{action.name}' are identical "
                f"(button vector {list(key)})."
            )
        seen[key] = action.name
