# -*- coding: utf-8 -*-
"""
PDA logic extracted from AP.py — usable as a pure backend module (no Flet).
"""
import re
import itertools
from typing import Dict, Set, Tuple, List, Optional, Any
from collections import deque

EPSILON = 'ε'


class PDA:
    def __init__(self):
        self.states: Set[str] = set()
        self.input_alphabet: Set[str] = set()
        self.stack_alphabet: Set[str] = set()
        self.transitions: Dict[Tuple[str, str, str], Set[Tuple[str, Tuple[str, ...]]]] = {}
        self.start_state: Optional[str] = None
        self.start_symbol: Optional[str] = None
        self.accept_states: Set[str] = set()

    def clear(self):
        self.__init__()

    def parse(self, states_str: str, input_alpha_str: str, stack_alpha_str: str,
              start_state_str: str, start_symbol_str: str, accept_states_str: str,
              transitions_str: str):
        self.clear()
        errors = []

        def clean_split(text: str) -> Set[str]:
            return set(s.strip() for s in text.split(',') if s.strip())

        self.states = clean_split(states_str)
        self.input_alphabet = clean_split(input_alpha_str)
        self.input_alphabet.discard(EPSILON)
        self.stack_alphabet = clean_split(stack_alpha_str)
        self.start_state = start_state_str.strip() if start_state_str else None
        self.start_symbol = start_symbol_str.strip() if start_symbol_str else None
        self.accept_states = clean_split(accept_states_str)

        if not self.states:
            errors.append("El conjunto de estados (Q) no puede estar vacío.")
        if not self.stack_alphabet:
            errors.append("El alfabeto de pila (Γ) no puede estar vacío.")
        if not self.start_state:
            errors.append("Debe definir un estado inicial (q₀).")
        if not self.start_symbol:
            errors.append("Debe definir un símbolo inicial de pila (Z₀).")
        if self.start_state and self.start_state not in self.states:
            errors.append(f"El estado inicial '{self.start_state}' no está en Q.")
        if self.start_symbol and self.start_symbol not in self.stack_alphabet:
            self.stack_alphabet.add(self.start_symbol)

        invalid_accept = self.accept_states - self.states
        if invalid_accept:
            errors.append(f"Estados de aceptación inválidos: {invalid_accept}")

        transition_pattern = re.compile(
            r"^\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*->\s*([^,]+)\s*,\s*(.*)\s*$"
        )
        processed_transitions = {}

        for i, line in enumerate(transitions_str.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            match = transition_pattern.match(line)
            if not match:
                errors.append(f"Línea {i}: Formato inválido. Use: 'origen,entrada,pop -> destino,push'")
                continue

            q_read, input_sym, stack_pop, q_write, stack_push_str = [m.strip() for m in match.groups()]

            if q_read not in self.states:
                errors.append(f"Línea {i}: Estado '{q_read}' no está en Q.")
            if q_write not in self.states:
                errors.append(f"Línea {i}: Estado '{q_write}' no está en Q.")
            if input_sym != EPSILON and input_sym not in self.input_alphabet:
                errors.append(f"Línea {i}: Símbolo '{input_sym}' no está en Σ.")
            if stack_pop != EPSILON and stack_pop not in self.stack_alphabet:
                errors.append(f"Línea {i}: Símbolo '{stack_pop}' no está en Γ.")

            stack_push_tuple = self._parse_stack_push(stack_push_str, i, errors)
            if stack_push_tuple is None:
                continue

            key = (q_read, input_sym, stack_pop)
            value = (q_write, stack_push_tuple)
            if key not in processed_transitions:
                processed_transitions[key] = set()
            processed_transitions[key].add(value)

        if errors:
            raise ValueError("\n".join(errors))

        self.transitions = processed_transitions

    def _parse_stack_push(self, stack_push_str: str, line_num: int, errors: List[str]) -> Optional[Tuple[str, ...]]:
        if stack_push_str == EPSILON:
            return tuple()
        symbols = sorted(self.stack_alphabet, key=len, reverse=True)
        result = []
        temp_str = stack_push_str
        while temp_str:
            found = False
            for symbol in symbols:
                if temp_str.startswith(symbol):
                    result.append(symbol)
                    temp_str = temp_str[len(symbol):]
                    found = True
                    break
            if not found:
                errors.append(f"Línea {line_num}: Símbolo inválido en '{stack_push_str}'")
                return None
        return tuple(result)

    def to_graph_json(self) -> dict:
        """Returns graph data for Flutter AutomatonCanvas."""
        states_list = []
        for s in self.states:
            states_list.append({
                "id": s,
                "isInitial": s == self.start_state,
                "isAccepting": s in self.accept_states,
            })

        edges_list = []
        # Group by (from, to) for multi-label edges
        grouped: Dict[Tuple[str, str], List[str]] = {}
        for (q, a, z), results in self.transitions.items():
            for (p, gamma) in results:
                push_str = "".join(gamma) if gamma else EPSILON
                label = f"{a},{z}/{push_str}"
                key = (q, p)
                grouped.setdefault(key, []).append(label)

        for (frm, to), labels in grouped.items():
            edges_list.append({
                "from": frm,
                "to": to,
                "label": "\n".join(labels),
            })

        return {"states": states_list, "edges": edges_list}


class PDASimulator:
    def __init__(self, pda: PDA):
        self.pda = pda
        self.max_steps = 2000
        self.max_stack_size = 200

    def simulate(self, input_string: str) -> Dict[str, Any]:
        if not self.pda.states or not self.pda.start_state or not self.pda.start_symbol:
            return {'accepted': False, 'error': 'PDA incompleto', 'trace': [], 'steps': 0}

        initial_config = (self.pda.start_state, input_string, [self.pda.start_symbol])
        queue = deque([initial_config])
        visited = {(self.pda.start_state, len(input_string), (self.pda.start_symbol,))}
        trace = [self._format_config(initial_config, 0, "Configuración inicial")]
        step = 0

        while queue and step < self.max_steps:
            step += 1
            current_state, current_input, current_stack = queue.popleft()

            if not current_input and current_state in self.pda.accept_states:
                trace.append(f"✅ Aceptada en el paso {step}. Estado: {current_state}")
                return {'accepted': True, 'acceptance_type': 'final_state', 'trace': trace, 'steps': step}

            self._explore(current_state, EPSILON, current_input, current_stack, queue, visited, trace, step)
            if current_input:
                self._explore(current_state, current_input[0], current_input[1:], current_stack, queue, visited, trace, step)

        if step >= self.max_steps:
            trace.append(f"⚠️ Límite de {self.max_steps} pasos alcanzado.")

        return {'accepted': False, 'trace': trace, 'steps': step}

    def _explore(self, state, char, input_after, stack, queue, visited, trace, step):
        stack_top = stack[-1] if stack else None
        if stack_top is not None:
            self._apply(state, char, stack_top, input_after, stack[:-1], queue, visited, trace, step)
        self._apply(state, char, EPSILON, input_after, stack, queue, visited, trace, step)

    def _apply(self, state, char, symbol_pop, input_after, stack_after_pop, queue, visited, trace, step):
        key = (state, char, symbol_pop)
        if key in self.pda.transitions:
            for next_state, push_symbols in self.pda.transitions[key]:
                new_stack = list(stack_after_pop)
                new_stack.extend(reversed(push_symbols))
                if len(new_stack) <= self.max_stack_size:
                    config_key = (next_state, len(input_after), tuple(new_stack))
                    if config_key not in visited:
                        visited.add(config_key)
                        new_config = (next_state, input_after, new_stack)
                        queue.append(new_config)
                        push_str = "".join(push_symbols) or EPSILON
                        move = f"Leer '{char}'" if char != EPSILON else "ε-movimiento"
                        trace.append(self._format_config(new_config, step, f"{move}, Pop '{symbol_pop}', Push '{push_str}'"))

    def _format_config(self, config, step, move=None):
        state, inp, stack = config
        stack_str = "".join(reversed(stack)) if stack else EPSILON
        inp_display = f"'{inp}'" if inp else EPSILON
        move_info = f"  ({move})" if move else ""
        return f"Paso {step}: Estado={state}, Entrada={inp_display}, Pila={stack_str}{move_info}"


def convert_pda_to_cfg(pda: PDA) -> str:
    if not pda.states or not pda.start_state or not pda.start_symbol:
        raise ValueError("PDA incompleto para conversión")

    non_terminals = {f"[{q},{A},{p}]" for q in pda.states for A in pda.stack_alphabet for p in pda.states}
    start_symbol = "S"
    productions = set()

    accept_states_to_use = pda.accept_states if pda.accept_states else pda.states
    for qf in sorted(list(accept_states_to_use)):
        productions.add(f"{start_symbol} → [{pda.start_state},{pda.start_symbol},{qf}]")

    for (q, a, Z), results in pda.transitions.items():
        for (r, gamma) in results:
            a_sym = a if a != EPSILON else 'ε'
            if not gamma:
                productions.add(f"[{q},{Z},{r}] → {a_sym}")
            else:
                k = len(gamma)
                for intermediate_states in itertools.product(pda.states, repeat=k - 1):
                    all_states = [r] + list(intermediate_states)
                    for p_k in pda.states:
                        rhs_parts = [a_sym]
                        current_states = all_states + [p_k]
                        for i in range(k):
                            rhs_parts.append(f"[{current_states[i]},{gamma[i]},{current_states[i + 1]}]")
                        productions.add(f"[{q},{Z},{p_k}] → {''.join(rhs_parts)}")

    sorted_productions = sorted(list(productions))

    result = f"""Gramática Libre de Contexto equivalente al PDA:
{'=' * 50}

Símbolos no terminales (V):
{{S, {', '.join(sorted(list(non_terminals)))}}}

Símbolos terminales (T):
{{{', '.join(sorted(pda.input_alphabet) if pda.input_alphabet else ['ε'])}}}

Símbolo inicial: S

Producciones (P):
{chr(10).join(f"  {prod}" for prod in sorted_productions)}

Nota: [q,A,p] representa cadenas que llevan al PDA del estado q al p consumiendo A de la pila.
"""
    return result