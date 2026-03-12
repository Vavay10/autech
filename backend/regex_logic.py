# -*- coding: utf-8 -*-
"""
Regex → NFA → DFA → Minimized DFA logic from expre.py.
All functions return JSON-serializable data (no matplotlib/images).
"""
import math
import itertools
from collections import defaultdict, deque
from typing import Dict, Set, List, Tuple, Optional, Any


# ─── Regex Parser ─────────────────────────────────────────────────────────────

def parse_regex(regex: str) -> list:
    if not isinstance(regex, str):
        raise ValueError("La expresión regular debe ser una cadena de texto.")
    regex = regex.strip()
    if not regex:
        raise ValueError("La expresión regular no puede estar vacía.")
    regex = ''.join(regex.split())

    if regex[0] in '.|*+?' or regex[-1] in '.|(':
        raise ValueError(f"Expresión inválida: empieza con '{regex[0]}' o termina con '{regex[-1]}'")

    # Convert a+ → aa*
    processed = ''
    i = 0
    while i < len(regex):
        char = regex[i]
        if char == '+' and i > 0:
            if regex[i - 1] == ')':
                balance, j = 0, i - 1
                while j >= 0:
                    if regex[j] == ')': balance += 1
                    elif regex[j] == '(': balance -= 1
                    if balance == 0:
                        grp = regex[j:i]
                        processed = processed[:-len(grp)]
                        processed += grp + grp + '*'
                        break
                    j -= 1
            elif regex[i - 1].isalnum() or regex[i - 1] == 'ε':
                prev = regex[i - 1]
                processed = processed[:-1]
                processed += prev + prev + '*'
            else:
                raise ValueError(f"Operador '+' inválido después de '{regex[i - 1]}'")
        else:
            processed += char
        i += 1
    regex = processed

    # Convert a? → (a|ε)
    processed = ''
    i = 0
    while i < len(regex):
        char = regex[i]
        if char == '?' and i > 0:
            if regex[i - 1] == ')':
                balance, j = 0, i - 1
                while j >= 0:
                    if regex[j] == ')': balance += 1
                    elif regex[j] == '(': balance -= 1
                    if balance == 0 and regex[j] == '(':
                        grp = regex[j:i]
                        processed = processed[:-(i - j)]
                        processed += f"({grp}|ε)"
                        break
                    j -= 1
            elif regex[i - 1].isalnum() or regex[i - 1] == 'ε':
                prev = regex[i - 1]
                processed = processed[:-1]
                processed += f"({prev}|ε)"
            else:
                raise ValueError(f"Operador '?' inválido después de '{regex[i - 1]}'")
        else:
            processed += char
        i += 1
    regex = processed

    # Add implicit concatenation
    new_regex = ''
    alnum_eps = 'abcdefghijklmnopqrstuvwxyz0123456789ε'
    for i in range(len(regex)):
        new_regex += regex[i]
        if i + 1 < len(regex):
            c1, c2 = regex[i], regex[i + 1]
            if (c1 in alnum_eps or c1 in ')*+?') and (c2 in alnum_eps or c2 == '('):
                new_regex += '.'
    regex = new_regex

    # Validate
    paren_count = 0
    last_char = None
    for i, char in enumerate(regex):
        if char == '(': paren_count += 1
        elif char == ')':
            paren_count -= 1
            if paren_count < 0:
                raise ValueError("Paréntesis desbalanceados.")
        if last_char is not None:
            if last_char in '.|' and char in '.|*+?)':
                raise ValueError(f"Operador inválido '{char}' después de '{last_char}'")
            if last_char in '*+?' and char in '*+?(':
                raise ValueError(f"Operador inválido '{char}' después de '{last_char}'")
            if last_char == '(' and char in '.|*+?)':
                raise ValueError(f"Operador inválido '{char}' después de '('")
        last_char = char

    if paren_count != 0:
        raise ValueError("Paréntesis desbalanceados.")

    # Shunting-yard → postfix
    def precedence(op):
        if op == '|': return 1
        if op == '.': return 2
        if op in '*+?': return 3
        return 0

    tokens, stack = [], []
    for char in regex:
        if char.isalnum():
            tokens.append(('SYMBOL', char))
        elif char == 'ε':
            tokens.append(('EPSILON', char))
        elif char == '(':
            stack.append(char)
        elif char == ')':
            while stack and stack[-1] != '(':
                tokens.append(('OPERATOR', stack.pop()))
            if stack: stack.pop()
        elif char in '.|*+?':
            while stack and stack[-1] != '(' and precedence(stack[-1]) >= precedence(char):
                tokens.append(('OPERATOR', stack.pop()))
            stack.append(char)
    while stack:
        op = stack.pop()
        tokens.append(('OPERATOR', op))

    return tokens


# ─── NFA Builder (Thompson) ───────────────────────────────────────────────────

def build_nfa(tokens: list) -> tuple:
    state_counter = 0
    states = set()
    transitions = defaultdict(lambda: defaultdict(set))

    def new_state():
        nonlocal state_counter
        s = f"q{state_counter}"
        states.add(s)
        state_counter += 1
        return s

    stack = []
    for token_type, value in tokens:
        if token_type in ('SYMBOL', 'EPSILON'):
            start, end = new_state(), new_state()
            sym = value if value != 'ε' else 'ε'
            transitions[start][sym].add(end)
            stack.append({'start': start, 'end': end})
        elif token_type == 'OPERATOR':
            if value == '.':
                r, l = stack.pop(), stack.pop()
                transitions[l['end']]['ε'].add(r['start'])
                stack.append({'start': l['start'], 'end': r['end']})
            elif value == '|':
                r, l = stack.pop(), stack.pop()
                ns, ne = new_state(), new_state()
                transitions[ns]['ε'].update([l['start'], r['start']])
                transitions[l['end']]['ε'].add(ne)
                transitions[r['end']]['ε'].add(ne)
                stack.append({'start': ns, 'end': ne})
            elif value == '*':
                nfa = stack.pop()
                ns, ne = new_state(), new_state()
                transitions[ns]['ε'].update([nfa['start'], ne])
                transitions[nfa['end']]['ε'].update([nfa['start'], ne])
                stack.append({'start': ns, 'end': ne})

    if len(stack) != 1:
        raise ValueError("Expresión regular inválida.")
    final = stack[0]
    return states, transitions, final['start'], final['end']


# ─── NFA → DFA ───────────────────────────────────────────────────────────────

def nfa_to_dfa(nfa_states, nfa_transitions, nfa_initial, nfa_final) -> tuple:
    eps_cache = {}

    def epsilon_closure(state_set):
        key = frozenset(state_set)
        if key in eps_cache: return eps_cache[key]
        closure = set(state_set)
        q = deque(state_set)
        while q:
            s = q.popleft()
            for ns in nfa_transitions.get(s, {}).get('ε', set()):
                if ns not in closure:
                    closure.add(ns)
                    q.append(ns)
        result = frozenset(closure)
        eps_cache[key] = result
        return result

    alphabet = set()
    for s in nfa_states:
        for sym in nfa_transitions.get(s, {}):
            if sym != 'ε':
                alphabet.add(sym)

    dfa_map = {}
    dfa_transitions = {}
    dfa_accepting = set()
    counter = 0
    worklist = deque()

    init_closure = epsilon_closure({nfa_initial})
    dfa_map[init_closure] = "D0"
    dfa_initial = "D0"
    worklist.append(init_closure)
    if nfa_final in init_closure:
        dfa_accepting.add("D0")
    counter = 1

    while worklist:
        cur_set = worklist.popleft()
        cur_name = dfa_map[cur_set]
        dfa_transitions[cur_name] = {}
        for sym in sorted(alphabet):
            next_direct = set()
            for s in cur_set:
                next_direct.update(nfa_transitions.get(s, {}).get(sym, set()))
            if not next_direct: continue
            next_closure = epsilon_closure(next_direct)
            if not next_closure: continue
            if next_closure not in dfa_map:
                new_name = f"D{counter}"
                counter += 1
                dfa_map[next_closure] = new_name
                worklist.append(next_closure)
                if nfa_final in next_closure:
                    dfa_accepting.add(new_name)
            dfa_transitions[cur_name][sym] = dfa_map[next_closure]

    dfa_states = list(dfa_map.values())
    # Add trap state for completeness
    trap = f"D{counter}"
    has_trap = False
    for s in list(dfa_states):
        if s not in dfa_transitions: dfa_transitions[s] = {}
        for sym in alphabet:
            if sym not in dfa_transitions[s]:
                if not has_trap:
                    dfa_states.append(trap)
                    dfa_transitions[trap] = {a: trap for a in alphabet}
                    has_trap = True
                dfa_transitions[s][sym] = trap

    return dfa_states, dfa_transitions, dfa_initial, dfa_accepting, alphabet


# ─── DFA Minimizer (Hopcroft) ─────────────────────────────────────────────────

def minimize_dfa(dfa_states, dfa_transitions, dfa_initial, dfa_accepting, alphabet) -> tuple:
    if not dfa_states:
        return [], {}, None, set()

    states = set(dfa_states)
    accepting = set(dfa_accepting)
    non_accepting = states - accepting
    partitions = []
    if accepting: partitions.append(accepting)
    if non_accepting: partitions.append(non_accepting)
    worklist = deque(partitions[:])

    while worklist:
        part = worklist.popleft()
        if not part: continue
        for sym in alphabet:
            preds = {s for s in states if dfa_transitions.get(s, {}).get(sym) in part}
            new_parts = []
            changed = False
            for P in partitions:
                inter = P & preds
                diff = P - preds
                if inter and diff:
                    new_parts += [inter, diff]
                    changed = True
                    if P in worklist:
                        worklist.remove(P)
                        worklist += [inter, diff]
                    else:
                        worklist.append(inter if len(inter) <= len(diff) else diff)
                else:
                    new_parts.append(P)
            if changed:
                partitions = new_parts

    # Build minimized DFA
    min_states = []
    state_reps = {}
    min_state_map = {}
    min_accepting = set()
    min_initial = None

    for i, partition in enumerate(partitions):
        if not partition: continue
        name = f"Min{i}"
        min_states.append(name)
        rep = next(iter(partition))
        state_reps[name] = rep
        for s in partition:
            min_state_map[s] = name
            if s == dfa_initial: min_initial = name
            if s in accepting: min_accepting.add(name)

    min_transitions = {}
    for name in min_states:
        rep = state_reps[name]
        min_transitions[name] = {}
        for sym in alphabet:
            dest = dfa_transitions.get(rep, {}).get(sym)
            if dest and dest in min_state_map:
                min_transitions[name][sym] = min_state_map[dest]

    # Keep only reachable
    if min_initial is None:
        return [], {}, None, set()
    reachable = {min_initial}
    q = deque([min_initial])
    while q:
        s = q.popleft()
        for sym in min_transitions.get(s, {}):
            ns = min_transitions[s][sym]
            if ns not in reachable:
                reachable.add(ns)
                q.append(ns)

    final_states = [s for s in min_states if s in reachable]
    final_transitions = {s: {sym: d for sym, d in min_transitions[s].items() if d in reachable}
                         for s in final_states}
    final_accepting = {s for s in min_accepting if s in reachable}
    final_initial = min_initial

    # Rename to M0, M1, ...
    rename = {old: f"M{i}" for i, old in enumerate(sorted(final_states))}
    return (
        list(rename.values()),
        {rename[s]: {sym: rename[d] for sym, d in final_transitions[s].items() if d in rename}
         for s in final_states},
        rename.get(final_initial),
        {rename[s] for s in final_accepting if s in rename},
    )


# ─── Convert automaton to JSON for Flutter ────────────────────────────────────

def automaton_to_json(states: list, transitions: dict, initial: str,
                      accepting: set, alphabet: set = None) -> dict:
    """Convert automaton data to Flutter-compatible JSON."""
    states_json = [
        {"id": s, "isInitial": s == initial, "isAccepting": s in accepting}
        for s in states
    ]
    edges_json = []
    for frm, trans in transitions.items():
        grouped: Dict[str, List[str]] = {}
        for sym, to in trans.items():
            grouped.setdefault(to, []).append(sym)
        for to, syms in grouped.items():
            edges_json.append({"from": frm, "to": to, "label": ",".join(sorted(syms))})

    return {
        "states": states_json,
        "edges": edges_json,
        "alphabet": sorted(list(alphabet)) if alphabet else [],
    }


# ─── Regex → Minimized DFA (all-in-one) ──────────────────────────────────────

def regex_to_min_dfa_json(regex: str) -> dict:
    tokens = parse_regex(regex)
    nfa_states, nfa_trans, nfa_init, nfa_final = build_nfa(tokens)
    dfa_states, dfa_trans, dfa_init, dfa_acc, alphabet = nfa_to_dfa(nfa_states, nfa_trans, nfa_init, nfa_final)
    min_states, min_trans, min_init, min_acc = minimize_dfa(dfa_states, dfa_trans, dfa_init, dfa_acc, alphabet)
    return automaton_to_json(min_states, min_trans, min_init, min_acc, alphabet)


# ─── Automaton → Regular Expression (State Elimination) ──────────────────────

def _wrap(r: str) -> str:
    """Wrap a regex string in parentheses only if needed (non-trivial)."""
    if not r or r == 'ε' or len(r) == 1:
        return r
    if r.startswith('(') and r.endswith(')'):
        # Check if the outer parens are matching
        depth = 0
        for i, c in enumerate(r):
            if c == '(': depth += 1
            elif c == ')': depth -= 1
            if depth == 0 and i < len(r) - 1:
                return f'({r})'
        return r
    return f'({r})'


def _concat(r1: Optional[str], r2: Optional[str]) -> Optional[str]:
    if r1 is None or r2 is None:
        return None
    if r1 == 'ε': return r2
    if r2 == 'ε': return r1
    return _wrap(r1) + _wrap(r2)


def _union(r1: Optional[str], r2: Optional[str]) -> Optional[str]:
    if r1 is None: return r2
    if r2 is None: return r1
    if r1 == r2: return r1
    return f'({r1}|{r2})'


def _star(r: Optional[str]) -> str:
    if r is None or r == 'ε':
        return 'ε'
    if len(r) == 1:
        return f'{r}*'
    return f'({r})*'


def automaton_to_regex(states: list, transitions: dict, initial: str,
                       accepting: set, alphabet: set = None) -> str:
    """
    Convert a DFA/NFA to a regular expression using the state elimination
    (Generalized NFA) algorithm.

    Returns a regex string.
    """
    if not states:
        raise ValueError("El autómata no tiene estados.")
    if not initial:
        raise ValueError("El autómata no tiene estado inicial.")
    if not accepting:
        raise ValueError("El autómata no tiene estados de aceptación.")

    # Filter unreachable states
    reachable = {initial}
    queue = deque([initial])
    while queue:
        s = queue.popleft()
        for sym, dest in transitions.get(s, {}).items():
            if isinstance(dest, str) and dest not in reachable:
                reachable.add(dest)
                queue.append(dest)
            elif isinstance(dest, set):
                for d in dest:
                    if d not in reachable:
                        reachable.add(d)
                        queue.append(d)

    states = [s for s in states if s in reachable]
    accepting = {s for s in accepting if s in reachable}

    # Special case: single accepting = initial state (accepts ε at least)
    QS = '__qs__'   # new unified start
    QA = '__qa__'   # new unified accept

    # Build GNFA as dict[from_state][to_state] = regex | None
    all_states = [QS] + list(states) + [QA]
    gnfa: Dict[str, Dict[str, Optional[str]]] = {s: {t: None for t in all_states} for s in all_states}

    # QS → initial with ε
    gnfa[QS][initial] = 'ε'

    # accepting states → QA with ε
    for acc in accepting:
        existing = gnfa[acc][QA]
        gnfa[acc][QA] = _union(existing, 'ε')

    # Original transitions
    # Edge labels may be comma-separated (e.g. 'a,b' means transitions on both a and b)
    for frm, trans in transitions.items():
        if frm not in reachable:
            continue
        if isinstance(trans, dict):
            for sym, dest in trans.items():
                # Split comma-separated labels into individual symbols
                raw_symbols = [s.strip() for s in sym.split(',') if s.strip()]
                dests = [dest] if isinstance(dest, str) else list(dest)
                for raw_sym in raw_symbols:
                    if raw_sym == 'ε' or raw_sym == '':
                        sym_label = 'ε'
                    else:
                        sym_label = raw_sym
                    for to in dests:
                        if to not in reachable:
                            continue
                        gnfa[frm][to] = _union(gnfa[frm][to], sym_label)

    # Eliminate states one by one (all except QS and QA)
    to_eliminate = list(states)

    for elim in to_eliminate:
        loop = gnfa[elim][elim]       # self-loop
        star_loop = _star(loop)       # (loop)*

        # Remaining states (not elim)
        remaining = [s for s in all_states if s != elim]

        for qi in remaining:
            for qj in remaining:
                r_in = gnfa[qi][elim]   # qi → elim
                r_out = gnfa[elim][qj]  # elim → qj
                if r_in is None or r_out is None:
                    continue

                # New path: r_in · star_loop · r_out
                mid = _concat(star_loop if star_loop != 'ε' else None, r_out)
                new_path = _concat(r_in, mid if mid else r_out)

                gnfa[qi][qj] = _union(gnfa[qi][qj], new_path)

        # Remove elim from GNFA
        all_states.remove(elim)
        del gnfa[elim]
        for s in all_states:
            if elim in gnfa[s]:
                del gnfa[s][elim]

    result = gnfa[QS][QA]
    if result is None:
        return '∅'  # empty language

    # Clean up ε in result
    result = result.replace('(ε|ε)', 'ε')
    return result


# ─── Language Operations ──────────────────────────────────────────────────────

def _dfa_accepts(transitions: dict, accepting: set, initial: str, word: str) -> bool:
    state = initial
    for ch in word:
        state = transitions.get(state, {}).get(ch)
        if state is None: return False
    return state in accepting


def _generate_words(alphabet: set, max_len: int) -> List[str]:
    words = ['']
    for _ in range(max_len):
        words += [''.join(p) for n in range(1, max_len + 1) for p in itertools.product(alphabet, repeat=n)]
    return list(dict.fromkeys(words))  # deduplicate


def operation_kleene(states, transitions, initial, accepting, alphabet):
    """Returns a sample JSON for Kleene closure: regex* of the accepted language."""
    new_states = list(states) + ["KLEENE_INIT"]
    new_transitions = dict(transitions)
    new_transitions["KLEENE_INIT"] = transitions.get(initial, {})
    new_accepting = set(accepting) | {"KLEENE_INIT"}
    for s in accepting:
        new_transitions[s] = dict(transitions.get(s, {}))
    return automaton_to_json(new_states, new_transitions, "KLEENE_INIT", new_accepting, alphabet)


def operation_union(s1, t1, i1, a1, s2, t2, i2, a2, alphabet):
    """Union of two automata using product construction."""
    product_states = [(p, q) for p in s1 for q in s2]
    product_initial = (i1, i2)
    product_accepting = {(p, q) for p, q in product_states if p in a1 or q in a2}
    product_transitions = {}
    for (p, q) in product_states:
        product_transitions[(p, q)] = {}
        for sym in alphabet:
            np_ = t1.get(p, {}).get(sym)
            nq = t2.get(q, {}).get(sym)
            if np_ and nq:
                product_transitions[(p, q)][sym] = (np_, nq)

    rename = {s: f"U{i}" for i, s in enumerate(product_states)}
    states_json = [{"id": rename[s], "isInitial": s == product_initial, "isAccepting": s in product_accepting}
                   for s in product_states]
    edges_json = []
    for s, trans in product_transitions.items():
        grouped: Dict[str, List[str]] = {}
        for sym, to in trans.items():
            grouped.setdefault(to, []).append(sym)
        for to, syms in grouped.items():
            edges_json.append({"from": rename[s], "to": rename[to], "label": ",".join(sorted(syms))})

    return {"states": states_json, "edges": edges_json, "alphabet": sorted(list(alphabet))}


def operation_intersection(s1, t1, i1, a1, s2, t2, i2, a2, alphabet):
    """Intersection of two automata."""
    product_states = [(p, q) for p in s1 for q in s2]
    product_accepting = {(p, q) for p, q in product_states if p in a1 and q in a2}
    product_transitions = {}
    for (p, q) in product_states:
        product_transitions[(p, q)] = {}
        for sym in alphabet:
            np_ = t1.get(p, {}).get(sym)
            nq = t2.get(q, {}).get(sym)
            if np_ and nq:
                product_transitions[(p, q)][sym] = (np_, nq)

    rename = {s: f"I{i}" for i, s in enumerate(product_states)}
    initial = (i1, i2)
    states_json = [{"id": rename[s], "isInitial": s == initial, "isAccepting": s in product_accepting}
                   for s in product_states]
    edges_json = []
    for s, trans in product_transitions.items():
        grouped: Dict[str, List[str]] = {}
        for sym, to in trans.items():
            grouped.setdefault(to, []).append(sym)
        for to, syms in grouped.items():
            edges_json.append({"from": rename[s], "to": rename[to], "label": ",".join(sorted(syms))})

    return {"states": states_json, "edges": edges_json, "alphabet": sorted(list(alphabet))}


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Test automaton_to_regex with a simple DFA for a*b
    states = ['M0', 'M1', 'M2']
    transitions = {
        'M0': {'a': 'M0', 'b': 'M1'},
        'M1': {'a': 'M2', 'b': 'M2'},
        'M2': {'a': 'M2', 'b': 'M2'},
    }
    initial = 'M0'
    accepting = {'M1'}
    alphabet = {'a', 'b'}
    regex = automaton_to_regex(states, transitions, initial, accepting, alphabet)
    print(f"a*b → regex: {regex}")

    # Test regex_to_min_dfa_json
    result = regex_to_min_dfa_json('a*b')
    print(f"a*b → DFA states: {[s['id'] for s in result['states']]}")