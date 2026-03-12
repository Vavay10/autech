# -*- coding: utf-8 -*-
"""
Turing Machine logic extracted from turing3.py — no Flet, no matplotlib.
"""
from typing import Dict, List, Set, Optional, Any


def parse_transitions(text: str) -> dict:
    """
    Parses transition text.
    Format: state,read -> newState,write,direction (L/R/S)
    Returns: {state: {read: [newState, write, direction]}}
    """
    transitions = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if '->' not in line:
            continue
        left, right = line.split('->', 1)
        left_parts = [p.strip() for p in left.split(',')]
        right_parts = [p.strip() for p in right.split(',')]

        if len(left_parts) < 2 or len(right_parts) < 3:
            continue

        state = left_parts[0]
        read = left_parts[1]
        new_state = right_parts[0]
        write = right_parts[1]
        direction = right_parts[2].upper()

        if state not in transitions:
            transitions[state] = {}
        transitions[state][read] = [new_state, write, direction]

    return transitions


def simulate_turing(
    states: List[str],
    transitions: dict,
    initial_state: str,
    accept_states: List[str],
    tape_input: str,
    head_pos: int = 0,
    max_steps: int = 1000,
) -> dict:
    """
    Simulates a Turing Machine step-by-step.

    Returns a dict with:
      - steps: list of step dicts
      - result: 'ACCEPTED' | 'REJECTED' | 'TIMEOUT'
    """
    tape = list(tape_input) if tape_input else ['_']
    # Ensure head position is valid
    while len(tape) <= head_pos:
        tape.append('_')

    current_state = initial_state
    accept_set = set(accept_states)
    steps = []

    # Initial step
    steps.append({
        "step": 0,
        "state": current_state,
        "tape": list(tape),
        "headPos": head_pos,
        "message": f"Inicio: estado={current_state}",
        "isAccepted": False,
        "isRejected": False,
        "prevState": None,
        "symbolRead": None,
        "transitionTaken": None,
    })

    for i in range(1, max_steps + 1):
        # Check accept
        if current_state in accept_set:
            steps.append({
                "step": i,
                "state": current_state,
                "tape": list(tape),
                "headPos": head_pos,
                "message": f"✅ Cadena ACEPTADA en estado {current_state}",
                "isAccepted": True,
                "isRejected": False,
                "prevState": current_state,
                "symbolRead": None,
                "transitionTaken": None,
            })
            return {"steps": steps, "result": "ACCEPTED"}

        # Extend tape if needed
        while head_pos < 0:
            tape.insert(0, '_')
            head_pos = 0
        while head_pos >= len(tape):
            tape.append('_')

        symbol_read = tape[head_pos]
        prev_state = current_state

        # Find transition
        trans = transitions.get(current_state, {}).get(symbol_read)
        if trans is None:
            # Check accept again before rejecting
            if current_state in accept_set:
                steps.append({
                    "step": i,
                    "state": current_state,
                    "tape": list(tape),
                    "headPos": head_pos,
                    "message": f"✅ Cadena ACEPTADA en estado {current_state}",
                    "isAccepted": True,
                    "isRejected": False,
                    "prevState": prev_state,
                    "symbolRead": symbol_read,
                    "transitionTaken": None,
                })
                return {"steps": steps, "result": "ACCEPTED"}

            steps.append({
                "step": i,
                "state": current_state,
                "tape": list(tape),
                "headPos": head_pos,
                "message": f"❌ Sin transición para ({current_state}, {symbol_read}) — RECHAZADA",
                "isAccepted": False,
                "isRejected": True,
                "prevState": prev_state,
                "symbolRead": symbol_read,
                "transitionTaken": None,
            })
            return {"steps": steps, "result": "REJECTED"}

        new_state, write_sym, direction = trans
        tape[head_pos] = write_sym
        current_state = new_state
        prev_head = head_pos

        if direction == 'R':
            head_pos += 1
        elif direction == 'L':
            head_pos -= 1
        # 'S' = stay

        if head_pos < 0:
            tape.insert(0, '_')
            head_pos = 0
        while head_pos >= len(tape):
            tape.append('_')

        transition_label = f"δ({prev_state},{symbol_read})=({new_state},{write_sym},{direction})"
        steps.append({
            "step": i,
            "state": current_state,
            "tape": list(tape),
            "headPos": head_pos,
            "message": f"{transition_label}  cabeza: {prev_head}→{head_pos}",
            "isAccepted": False,
            "isRejected": False,
            "prevState": prev_state,
            "symbolRead": symbol_read,
            "transitionTaken": [new_state, write_sym, direction],
        })

        if current_state in accept_set:
            steps.append({
                "step": i + 1,
                "state": current_state,
                "tape": list(tape),
                "headPos": head_pos,
                "message": f"✅ Cadena ACEPTADA en estado {current_state}",
                "isAccepted": True,
                "isRejected": False,
                "prevState": prev_state,
                "symbolRead": None,
                "transitionTaken": None,
            })
            return {"steps": steps, "result": "ACCEPTED"}

    # Timeout
    steps.append({
        "step": max_steps + 1,
        "state": current_state,
        "tape": list(tape),
        "headPos": head_pos,
        "message": f"⚠️ Límite de {max_steps} pasos alcanzado — posible bucle infinito",
        "isAccepted": False,
        "isRejected": True,
        "prevState": current_state,
        "symbolRead": None,
        "transitionTaken": None,
    })
    return {"steps": steps, "result": "TIMEOUT"}


def build_graph_json(states: List[str], transitions: dict, initial: str,
                     accept_states: List[str]) -> dict:
    """Convert TM definition to Flutter AutomatonCanvas JSON."""
    states_json = [
        {"id": s, "isInitial": s == initial, "isAccepting": s in accept_states}
        for s in states
    ]
    # Group edges: (from, to) → [labels]
    grouped: Dict[tuple, List[str]] = {}
    for state, trans in transitions.items():
        for read, (new_state, write, direction) in trans.items():
            key = (state, new_state)
            label = f"{read}→{write},{direction}"
            grouped.setdefault(key, []).append(label)

    edges_json = [
        {"from": frm, "to": to, "label": "\n".join(labels)}
        for (frm, to), labels in grouped.items()
    ]
    return {"states": states_json, "edges": edges_json}