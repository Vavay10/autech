import flet as ft
import os
os.environ['MPLCONFIGDIR'] = '/data/user/0/com.flet.pythonproject/cache/mplconfig'
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
import numpy as np
import io
import math
import base64
from collections import defaultdict, deque
import copy  # Necesario para deepcopy
import traceback  # Para mostrar errores detallados


# Funciones auxiliares de ah.py
def parse_regex(regex):
    """
    Parsea una expresión regular y la convierte a notación postfija (RPN).
    """
    # Validación inicial
    if not isinstance(regex, str):
        raise ValueError("La expresión regular debe ser una cadena de texto.")

    regex = regex.strip()

    if not regex:
        raise ValueError("La expresión regular no puede estar vacía.")

    # Eliminar espacios en blanco no significativos
    regex = ''.join(regex.split())

    # Validaciones iniciales de operadores
    if regex[0] in '.|*+?' or regex[-1] in '.|(':
        raise ValueError(f"Expresión inválida: no puede empezar con '{regex[0]}' ni terminar con '{regex[-1]}'")

    # Convertir a+ en aa*
    processed_regex = ''
    i = 0
    while i < len(regex):
        char = regex[i]
        if char == '+' and i > 0:
            prev_char_or_group = ''
            if regex[i - 1] == ')':
                balance = 0
                j = i - 1
                while j >= 0:
                    if regex[j] == ')':
                        balance += 1
                    elif regex[j] == '(':
                        balance -= 1
                    if balance == 0:
                        prev_char_or_group = regex[j:i]
                        break
                    j -= 1
                if not prev_char_or_group:
                    raise ValueError(f"Error procesando '+' después de paréntesis desbalanceado cerca del índice {i}")
                processed_regex = processed_regex[:-len(prev_char_or_group)]
                processed_regex += prev_char_or_group + prev_char_or_group + '*'
            elif regex[i - 1].isalnum() or regex[i - 1] == 'ε':
                prev_char_or_group = regex[i - 1]
                processed_regex = processed_regex[:-1]
                processed_regex += prev_char_or_group + prev_char_or_group + '*'
            else:
                raise ValueError(f"Operador '+' inválido después de '{regex[i - 1]}' en el índice {i}")
        else:
            processed_regex += char
        i += 1
    regex = processed_regex

    # Convertir a? en (a|ε)
    processed_regex = ''
    i = 0
    while i < len(regex):
        char = regex[i]
        if char == '?' and i > 0:
            replacement = ''
            if regex[i - 1] == ')':
                balance = 0
                j = i - 1
                group_start_index = -1
                while j >= 0:
                    if regex[j] == ')':
                        balance += 1
                    elif regex[j] == '(':
                        balance -= 1
                    if balance == 0 and regex[j] == '(':
                        group_start_index = j
                        break
                    j -= 1
                if group_start_index == -1:
                    raise ValueError(f"Error procesando '?' después de paréntesis desbalanceado cerca del índice {i}")
                group = regex[group_start_index:i]
                replacement = f"({group}|ε)"
                processed_regex = processed_regex[:-(i - group_start_index)]
            elif regex[i - 1].isalnum() or regex[i - 1] == 'ε':
                prev_char = regex[i - 1]
                replacement = f"({prev_char}|ε)"
                processed_regex = processed_regex[:-1]
            else:
                raise ValueError(f"Operador '?' inválido después de '{regex[i - 1]}' en el índice {i}")
            processed_regex += replacement
        else:
            processed_regex += char
        i += 1
    regex = processed_regex

    # Añadir concatenaciones implícitas
    new_regex = ''
    i = 0
    alphanum_epsilon = 'abcdefghijklmnopqrstuvwxyz0123456789ε'
    while i < len(regex):
        new_regex += regex[i]
        if i + 1 < len(regex):
            c1, c2 = regex[i], regex[i + 1]
            if (c1 in alphanum_epsilon or c1 in ')*+?') and \
                    (c2 in alphanum_epsilon or c2 == '('):
                new_regex += '.'
        i += 1
    regex = new_regex

    # Validar la expresión
    paren_count = 0
    last_char = None
    for i, char in enumerate(regex):
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
            if paren_count < 0:
                raise ValueError("Paréntesis desbalanceados: más cierres que aperturas.")
        if last_char is not None:
            if last_char in '.|' and char in '.|*+?)':
                raise ValueError(f"Operador inválido '{char}' después de '{last_char}' en el índice {i}")
            if last_char in '*+?' and char in '*+?(':
                raise ValueError(f"Operador inválido '{char}' después de '{last_char}' en el índice {i}")
            if last_char == '(' and char in '.|*+?)':
                raise ValueError(f"Operador inválido '{char}' después de '(' en el índice {i}")
            if last_char in '.|' and char == ')':
                raise ValueError(f"Operador inválido '{last_char}' antes de ')' en el índice {i}")
        elif char in '.|*+?':
            raise ValueError(f"La expresión no puede comenzar con el operador '{char}'")
        last_char = char

    if paren_count != 0:
        raise ValueError("Paréntesis desbalanceados: no se cerraron todos los paréntesis.")
    if last_char in '.|(':
        raise ValueError(f"La expresión no puede terminar con el operador '{last_char}'")

    # Convertir a notación postfija
    def precedence(op):
        if op == '|': return 1
        if op == '.': return 2
        if op in '*+?': return 3
        return 0

    tokens = []
    stack = []
    i = 0

    while i < len(regex):
        char = regex[i]
        if char.isalnum():
            tokens.append(('SYMBOL', char))
        elif char == 'ε':
            tokens.append(('EPSILON', char))
        elif char == '(':
            stack.append(char)
        elif char == ')':
            while stack and stack[-1] != '(':
                tokens.append(('OPERATOR', stack.pop()))
            if stack and stack[-1] == '(':
                stack.pop()
            else:
                raise ValueError("Error interno: Paréntesis desbalanceado encontrado durante RPN.")
        elif char in '.|*+?':
            while stack and stack[-1] != '(' and precedence(stack[-1]) >= precedence(char):
                tokens.append(('OPERATOR', stack.pop()))
            stack.append(char)
        else:
            raise ValueError(f"Carácter inesperado encontrado durante RPN: {char}")
        i += 1

    while stack:
        op = stack.pop()
        if op == '(':
            raise ValueError("Error interno: Paréntesis de apertura sin cierre encontrado durante RPN.")
        tokens.append(('OPERATOR', op))

    return tokens


def build_nfa(tokens):
    """
    Construye un NFA a partir de la expresión regular en notación postfija.
    """
    state_counter = 0
    states = set()
    transitions = defaultdict(lambda: defaultdict(set))

    def new_state():
        nonlocal state_counter
        state = f"q{state_counter}"
        states.add(state)
        state_counter += 1
        return state

    stack = []

    for token_type, value in tokens:
        if token_type in ('SYMBOL', 'EPSILON'):
            start = new_state()
            end = new_state()
            transitions[start][value if value != 'ε' else 'ε'].add(end)
            stack.append({'start': start, 'end': end})

        elif token_type == 'OPERATOR':
            if value == '.':  # Concatenación
                if len(stack) < 2: raise ValueError("Faltan operandos para concatenación")
                right_nfa = stack.pop()
                left_nfa = stack.pop()
                transitions[left_nfa['end']]['ε'].add(right_nfa['start'])
                stack.append({'start': left_nfa['start'], 'end': right_nfa['end']})

            elif value == '|':  # Unión
                if len(stack) < 2: raise ValueError("Faltan operandos para unión")
                right_nfa = stack.pop()
                left_nfa = stack.pop()
                new_start = new_state()
                new_end = new_state()
                transitions[new_start]['ε'].add(left_nfa['start'])
                transitions[new_start]['ε'].add(right_nfa['start'])
                transitions[left_nfa['end']]['ε'].add(new_end)
                transitions[right_nfa['end']]['ε'].add(new_end)
                stack.append({'start': new_start, 'end': new_end})

            elif value == '*':  # Clausura de Kleene
                if not stack: raise ValueError("Faltan operandos para clausura de Kleene")
                nfa = stack.pop()
                new_start = new_state()
                new_end = new_state()
                transitions[new_start]['ε'].add(nfa['start'])
                transitions[new_start]['ε'].add(new_end)
                transitions[nfa['end']]['ε'].add(nfa['start'])
                transitions[nfa['end']]['ε'].add(new_end)
                stack.append({'start': new_start, 'end': new_end})

    if len(stack) != 1:
        raise ValueError("Expresión regular inválida: estructura RPN incorrecta")

    final_nfa = stack[0]
    return states, transitions, final_nfa['start'], final_nfa['end']


def nfa_to_dfa(nfa_states, nfa_transitions, nfa_initial_state, nfa_final_state):
    """
    Convierte un NFA a DFA usando el algoritmo de construcción por subconjuntos.
    """
    epsilon_closures = {}

    def get_epsilon_closure(state_set):
        frozen_state_set = frozenset(state_set)
        if frozen_state_set in epsilon_closures:
            return epsilon_closures[frozen_state_set]
        closure = set(state_set)
        queue = deque(list(state_set))
        while queue:
            state = queue.popleft()
            for next_state in nfa_transitions.get(state, {}).get('ε', set()):
                if next_state not in closure:
                    closure.add(next_state)
                    queue.append(next_state)
        result = frozenset(closure)
        epsilon_closures[frozen_state_set] = result
        return result

    alphabet = set()
    for state in nfa_states:
        for symbol in nfa_transitions.get(state, {}):
            if symbol != 'ε':
                alphabet.add(symbol)

    dfa_states_map = {}
    dfa_transitions = {}
    dfa_accepting = set()
    dfa_initial_state = None
    unmarked_dfa_states = deque()

    initial_closure = get_epsilon_closure({nfa_initial_state})
    if not initial_closure:
        raise ValueError("Error: Cierre-épsilon inicial vacío.")
    dfa_initial_state_name = "D0"
    dfa_states_map[initial_closure] = dfa_initial_state_name
    dfa_initial_state = dfa_initial_state_name
    unmarked_dfa_states.append(initial_closure)
    if nfa_final_state in initial_closure:
        dfa_accepting.add(dfa_initial_state_name)
    dfa_state_counter = 1
    processed_nfa_sets = {initial_closure}

    while unmarked_dfa_states:
        current_nfa_set = unmarked_dfa_states.popleft()
        current_dfa_state_name = dfa_states_map[current_nfa_set]
        dfa_transitions[current_dfa_state_name] = {}
        for symbol in sorted(list(alphabet)):
            next_nfa_states_direct = set()
            for nfa_state in current_nfa_set:
                next_nfa_states_direct.update(nfa_transitions.get(nfa_state, {}).get(symbol, set()))
            if not next_nfa_states_direct:
                continue
            next_closure = get_epsilon_closure(next_nfa_states_direct)
            if not next_closure:
                continue
            if next_closure not in dfa_states_map:
                next_dfa_state_name = f"D{dfa_state_counter}"
                dfa_state_counter += 1
                dfa_states_map[next_closure] = next_dfa_state_name
                unmarked_dfa_states.append(next_closure)
                processed_nfa_sets.add(next_closure)
                if nfa_final_state in next_closure:
                    dfa_accepting.add(next_dfa_state_name)
            else:
                next_dfa_state_name = dfa_states_map[next_closure]
            dfa_transitions[current_dfa_state_name][symbol] = next_dfa_state_name

    dfa_states_list = list(dfa_states_map.values())
    has_trap = False
    trap_state_name = f"D{dfa_state_counter}"
    for state in dfa_states_list:
        if state not in dfa_transitions:
            dfa_transitions[state] = {}
        for symbol in alphabet:
            if symbol not in dfa_transitions.get(state, {}):
                if not has_trap:
                    dfa_states_list.append(trap_state_name)
                    dfa_transitions[trap_state_name] = {s: trap_state_name for s in alphabet}
                    has_trap = True
                dfa_transitions[state][symbol] = trap_state_name

    return dfa_states_list, dfa_transitions, dfa_initial_state, dfa_accepting, alphabet


def minimize_dfa(dfa_states, dfa_transitions, dfa_initial_state, dfa_accepting_states, alphabet):
    """
    Minimiza un DFA usando el algoritmo de partición de estados (basado en Hopcroft).
    """
    if not dfa_states:
        return [], {}, None, set()
    states = set(dfa_states)
    accepting = set(dfa_accepting_states)
    non_accepting = states - accepting
    partitions = []
    if accepting: partitions.append(accepting)
    if non_accepting: partitions.append(non_accepting)
    worklist = deque(partitions[:])
    while worklist:
        current_partition = worklist.popleft()
        if not current_partition: continue
        for symbol in alphabet:
            predecessors = set()
            for state in states:
                dest_state = dfa_transitions.get(state, {}).get(symbol)
                if dest_state in current_partition:
                    predecessors.add(state)
            new_partitions = []
            changed = False
            for P in partitions:
                intersection = P & predecessors
                difference = P - predecessors
                if intersection and difference:
                    new_partitions.append(intersection)
                    new_partitions.append(difference)
                    changed = True
                    if P in worklist:
                        worklist.remove(P)
                        worklist.append(intersection)
                        worklist.append(difference)
                    else:
                        if len(intersection) <= len(difference):
                            worklist.append(intersection)
                        else:
                            worklist.append(difference)
                else:
                    new_partitions.append(P)
            if changed:
                partitions = new_partitions
    min_state_map = {}
    min_states = []
    min_transitions = {}
    min_initial_state = None
    min_accepting = set()
    state_reps = {}
    for i, partition in enumerate(partitions):
        if not partition: continue
        min_state_name = f"M{i}"
        min_states.append(min_state_name)
        rep = min(list(partition))
        state_reps[min_state_name] = rep
        for state in partition:
            min_state_map[state] = min_state_name
            if state == dfa_initial_state:
                min_initial_state = min_state_name
            if state in dfa_accepting_states:
                min_accepting.add(min_state_name)
    for min_state_name in min_states:
        rep = state_reps[min_state_name]
        min_transitions[min_state_name] = {}
        for symbol in alphabet:
            original_dest = dfa_transitions.get(rep, {}).get(symbol)
            if original_dest is not None and original_dest in min_state_map:
                min_transitions[min_state_name][symbol] = min_state_map[original_dest]
    if min_initial_state is None and min_states:
        if min_states:
            min_initial_state = min_states[0]
        else:
            return [], {}, None, set()
    elif min_initial_state is None and not min_states:
        return [], {}, None, set()
    reachable = {min_initial_state}
    queue = deque([min_initial_state])
    while queue:
        state = queue.popleft()
        for symbol in min_transitions.get(state, {}):
            next_state = min_transitions[state].get(symbol)
            if next_state is not None and next_state not in reachable:
                reachable.add(next_state)
                queue.append(next_state)
    final_states = [s for s in min_states if s in reachable]
    final_transitions = {s: trans for s, trans in min_transitions.items() if s in reachable}
    final_accepting = {s for s in min_accepting if s in reachable}
    final_initial = min_initial_state if min_initial_state in reachable else None
    for state in list(final_transitions.keys()):
        for symbol in list(final_transitions[state].keys()):
            if final_transitions[state][symbol] not in reachable:
                del final_transitions[state][symbol]
    rename_map = {old_name: f"M{i}" for i, old_name in enumerate(sorted(final_states))}
    renamed_states = list(rename_map.values())
    renamed_initial = rename_map.get(final_initial)
    renamed_accepting = {rename_map[s] for s in final_accepting if s in rename_map}
    renamed_transitions = {}
    for old_name, transitions_dict in final_transitions.items():
        new_name = rename_map.get(old_name)
        if new_name:
            renamed_transitions[new_name] = {}
            for symbol, old_dest in transitions_dict.items():
                new_dest = rename_map.get(old_dest)
                if new_dest:
                    renamed_transitions[new_name][symbol] = new_dest
    return renamed_states, renamed_transitions, renamed_initial, renamed_accepting


def draw_automata(states, transitions, initial_state, accepting_states, alphabet):
    """
    Dibuja el autómata con diseño mejorado y devuelve la imagen como bytes.
    """
    if not states:
        fig, ax = plt.subplots(figsize=(8, 2), dpi=200, facecolor='white')
        ax.set_facecolor('white')
        ax.text(0.5, 0.5, "Autómata vacío o no generable",
                ha='center', va='center', fontsize=14, color='#666666')
        ax.axis('off')
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight',
                    facecolor='white', dpi=300)
        img_buffer.seek(0)
        plt.close(fig)
        return img_buffer.getvalue()

    # Crear el grafo dirigido
    G = nx.DiGraph()
    G.add_nodes_from(states)

    # Agregar aristas para el layout (sin etiquetas aún)
    for state_from in states:
        if state_from in transitions:
            for symbol in alphabet:
                if symbol in transitions[state_from]:
                    state_to = transitions[state_from][symbol]
                    if state_to in states:
                        G.add_edge(state_from, state_to)

    # Configuración de tamaño de figura
    num_states = len(states)
    if num_states <= 3:
        figsize = (10, 8)
        node_size = 2000
        font_size_nodes = 14
        font_size_labels = 12
    elif num_states <= 6:
        figsize = (12, 10)
        node_size = 1800
        font_size_nodes = 13
        font_size_labels = 11
    else:
        figsize = (14, 12)
        node_size = 1500
        font_size_nodes = 12
        font_size_labels = 10

    # Crear figura con fondo limpio
    fig, ax = plt.subplots(figsize=figsize, dpi=200, facecolor='white')
    ax.set_facecolor('white')

    # Calcular posiciones optimizadas
    pos = calculate_optimal_positions(G, states, num_states)

    # Dibujar nodos con estilos diferenciados
    draw_nodes(ax, pos, states, initial_state, accepting_states, node_size, font_size_nodes)

    # Procesar y dibujar transiciones agrupadas
    draw_transitions(ax, pos, states, transitions, alphabet, font_size_labels, node_size)

    # Dibujar flecha de inicio
    draw_initial_arrow(ax, pos, initial_state)

    # Configurar título y leyenda
    setup_title_and_legend(ax, font_size_nodes)

    # Configurar ejes y márgenes
    setup_axes(ax, pos)

    plt.tight_layout()

    # Guardar imagen
    img_buffer = io.BytesIO()
    try:
        plt.savefig(img_buffer, format='png', bbox_inches='tight',
                    dpi=300, facecolor='white', edgecolor='none')
        img_buffer.seek(0)
        plt.close(fig)
        return img_buffer.getvalue()
    except Exception as e:
        plt.close(fig)
        raise e


def calculate_optimal_positions(G, states, num_states):
    """Calcula posiciones optimizadas para los nodos."""
    states_list = list(states)

    if num_states == 1:
        return {states_list[0]: (0, 0)}
    elif num_states == 2:
        return {states_list[0]: (-1, 0), states_list[1]: (1, 0)}
    elif num_states <= 6:
        # Layout circular para pocos estados
        pos = {}
        angle_step = 2 * np.pi / num_states
        radius = 2.5
        for i, state in enumerate(states_list):
            angle = i * angle_step - np.pi / 2  # Empezar desde arriba
            pos[state] = (radius * np.cos(angle), radius * np.sin(angle))
        return pos
    else:
        # Layout spring mejorado para muchos estados
        try:
            pos = nx.spring_layout(G, k=3 / math.sqrt(num_states), iterations=200, seed=42)
            # Escalar posiciones
            scale = 4.0
            for node in pos:
                pos[node] = (pos[node][0] * scale, pos[node][1] * scale)
            return pos
        except:
            # Fallback a layout circular
            pos = {}
            angle_step = 2 * np.pi / num_states
            radius = 3.0
            for i, state in enumerate(states_list):
                angle = i * angle_step
                pos[state] = (radius * np.cos(angle), radius * np.sin(angle))
            return pos


def draw_nodes(ax, pos, states, initial_state, accepting_states, node_size, font_size):
    """Dibuja los nodos del autómata con estilos diferenciados."""
    for state in states:
        if state not in pos:
            continue

        x, y = pos[state]

        # Determinar colores y estilos
        is_initial = state == initial_state
        is_accepting = state in accepting_states

        if is_initial and is_accepting:
            facecolor = '#E8F5E8'  # Verde claro
            edgecolor = '#2E7D32'  # Verde oscuro
            linewidth = 3
        elif is_initial:
            facecolor = '#E3F2FD'  # Azul claro
            edgecolor = '#1976D2'  # Azul oscuro
            linewidth = 3
        elif is_accepting:
            facecolor = '#FFF3E0'  # Naranja claro
            edgecolor = '#F57C00'  # Naranja oscuro
            linewidth = 3
        else:
            facecolor = '#FAFAFA'  # Gris muy claro
            edgecolor = '#616161'  # Gris oscuro
            linewidth = 2

        # Dibujar círculo principal
        radius = math.sqrt(node_size / math.pi) / 100
        circle = plt.Circle((x, y), radius, facecolor=facecolor,
                            edgecolor=edgecolor, linewidth=linewidth, zorder=3)
        ax.add_patch(circle)

        # Dibujar doble círculo para estados de aceptación
        if is_accepting:
            inner_circle = plt.Circle((x, y), radius * 0.8, fill=False,
                                      edgecolor=edgecolor, linewidth=linewidth - 1, zorder=4)
            ax.add_patch(inner_circle)

        # Dibujar etiqueta del estado
        ax.text(x, y, str(state), ha='center', va='center',
                fontsize=font_size, fontweight='bold', color='#212121', zorder=5)


def draw_transitions(ax, pos, states, transitions, alphabet, font_size, node_size):
    """Dibuja las transiciones agrupadas por pares de estados."""
    # Agrupar transiciones por pares origen-destino
    transition_groups = defaultdict(list)

    for state_from in states:
        if state_from in transitions:
            for symbol in alphabet:
                if symbol in transitions[state_from]:
                    state_to = transitions[state_from][symbol]
                    if state_to in states:
                        key = (state_from, state_to)
                        transition_groups[key].append(symbol)

    # Dibujar cada grupo de transiciones
    for (origin, dest), symbols in transition_groups.items():
        if origin not in pos or dest not in pos:
            continue

        draw_transition_group(ax, pos, origin, dest, symbols, font_size, node_size)


def draw_transition_group(ax, pos, origin, dest, symbols, font_size, node_size):
    """Dibuja un grupo de transiciones entre dos estados."""
    x1, y1 = pos[origin]
    x2, y2 = pos[dest]

    # Calcular radio del nodo para ajustar inicio/fin de flecha
    node_radius = math.sqrt(node_size / math.pi) / 100

    is_self_loop = origin == dest

    if is_self_loop:
        # Dibujar bucle
        draw_self_loop(ax, x1, y1, symbols, font_size, node_radius)
    else:
        # Dibujar transición normal
        draw_normal_transition(ax, x1, y1, x2, y2, symbols, font_size, node_radius)


def draw_self_loop(ax, x, y, symbols, font_size, node_radius):
    """Dibuja un bucle en un estado."""
    # Parámetros del bucle
    loop_radius = node_radius * 1.8
    loop_center_x = x
    loop_center_y = y + node_radius + loop_radius

    # Dibujar círculo del bucle
    loop_circle = plt.Circle((loop_center_x, loop_center_y), loop_radius,
                             fill=False, edgecolor='#424242', linewidth=2, zorder=1)
    ax.add_patch(loop_circle)

    # Dibujar flecha
    arrow_angle = np.pi * 0.1  # Pequeño ángulo para la flecha
    arrow_x = loop_center_x + loop_radius * np.cos(arrow_angle)
    arrow_y = loop_center_y + loop_radius * np.sin(arrow_angle)

    # Vector tangente para la dirección de la flecha
    tangent_x = -np.sin(arrow_angle) * 0.3
    tangent_y = np.cos(arrow_angle) * 0.3

    ax.annotate('', xy=(arrow_x, arrow_y),
                xytext=(arrow_x - tangent_x, arrow_y - tangent_y),
                arrowprops=dict(arrowstyle='->', color='#424242', lw=2))

    # Posicionar etiqueta
    label_text = ','.join(sorted(symbols))
    label_x = loop_center_x
    label_y = loop_center_y + loop_radius * 1.3

    ax.text(label_x, label_y, label_text, ha='center', va='center',
            fontsize=font_size, fontweight='bold', color='#1565C0',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#1565C0', linewidth=1))


def draw_normal_transition(ax, x1, y1, x2, y2, symbols, font_size, node_radius):
    """Dibuja una transición normal entre dos estados."""
    # Calcular vector dirección
    dx = x2 - x1
    dy = y2 - y1
    distance = math.sqrt(dx ** 2 + dy ** 2)

    if distance == 0:
        return

    # Normalizar vector dirección
    dx_norm = dx / distance
    dy_norm = dy / distance

    # Ajustar puntos de inicio y fin para no solapar con los nodos
    margin = node_radius * 1.1
    start_x = x1 + dx_norm * margin
    start_y = y1 + dy_norm * margin
    end_x = x2 - dx_norm * margin
    end_y = y2 - dy_norm * margin

    # Dibujar flecha
    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle='->', color='#424242',
                                lw=2, connectionstyle="arc3,rad=0.1"))

    # Posicionar etiqueta en el punto medio, ligeramente offset
    mid_x = (start_x + end_x) / 2
    mid_y = (start_y + end_y) / 2

    # Vector perpendicular para offset de etiqueta
    perp_x = -dy_norm * 0.3
    perp_y = dx_norm * 0.3

    label_x = mid_x + perp_x
    label_y = mid_y + perp_y

    label_text = ','.join(sorted(symbols))
    ax.text(label_x, label_y, label_text, ha='center', va='center',
            fontsize=font_size, fontweight='bold', color='#1565C0',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#1565C0', linewidth=1))


def draw_initial_arrow(ax, pos, initial_state):
    """Dibuja la flecha que indica el estado inicial."""
    if initial_state is None or initial_state not in pos:
        return

    x, y = pos[initial_state]

    # Flecha desde la izquierda
    arrow_start_x = x - 1.5
    arrow_end_x = x - 0.3

    ax.annotate('', xy=(arrow_end_x, y), xytext=(arrow_start_x, y),
                arrowprops=dict(arrowstyle='->', color='#1976D2', lw=3))

    ax.text(arrow_start_x - 0.2, y + 0.2, 'Inicio', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#1976D2')


def setup_title_and_legend(ax, font_size):
    """Configura el título y la leyenda."""
    ax.set_title('Autómata Finito Determinista', fontsize=font_size + 4,
                 fontweight='bold', color='#212121', pad=20)

    # Leyenda mejorada
    legend_elements = [
        mpatches.Patch(facecolor='#E3F2FD', edgecolor='#1976D2', label='Estado Inicial'),
        mpatches.Patch(facecolor='#FFF3E0', edgecolor='#F57C00', label='Estado de Aceptación'),
        mpatches.Patch(facecolor='#E8F5E8', edgecolor='#2E7D32', label='Inicial y Aceptación'),
        mpatches.Patch(facecolor='#FAFAFA', edgecolor='#616161', label='Estado Normal')
    ]

    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=True,
              fontsize=10, borderpad=1)


def setup_axes(ax, pos):
    """Configura los ejes y márgenes."""
    ax.set_aspect('equal')
    ax.axis('off')

    if pos:
        x_values = [x for x, y in pos.values()]
        y_values = [y for x, y in pos.values()]

        margin = 2.0
        ax.set_xlim(min(x_values) - margin, max(x_values) + margin)
        ax.set_ylim(min(y_values) - margin, max(y_values) + margin)


def obtener_regex_afd_jflap2(estados, alfabeto, estado_inicial, estados_aceptacion, transiciones):
    import sys
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(3000)
    try:
        matriz = {i: {j: set() for j in estados} for i in estados}
        for estado_origen, trans in transiciones.items():
            for simbolo, estado_destino in trans.items():
                if estado_destino:
                    matriz[estado_origen][estado_destino].add(simbolo)
        for i in estados:
            for j in estados:
                if matriz[i][j]:
                    matriz[i][j] = '+'.join(sorted(matriz[i][j]))
                    if len(matriz[i][j].split('+')) > 1:
                        matriz[i][j] = f"({matriz[i][j]})"
                else:
                    matriz[i][j] = ""
        estado_nuevo_inicial = "qi"
        estado_nuevo_final = "qf"
        estados_expandidos = [estado_nuevo_inicial] + estados + [estado_nuevo_final]
        matriz_expandida = {i: {j: "" for j in estados_expandidos} for i in estados_expandidos}
        for i in estados:
            for j in estados:
                matriz_expandida[i][j] = matriz[i][j]
        matriz_expandida[estado_nuevo_inicial][estado_inicial] = "ε"
        for estado in estados_aceptacion:
            matriz_expandida[estado][estado_nuevo_final] = "ε"

        def combinar_expresiones(expr1, expr2):
            if not expr1:
                return expr2
            if not expr2:
                return expr1
            if expr1 == expr2:
                return expr1
            if expr1 in expr2:
                return expr2
            if expr2 in expr1:
                return expr1
            return f"({expr1}+{expr2})"

        def concatenar_expresiones(*exprs):
            exprs = [e for e in exprs if e and e != "ε"]
            if not exprs:
                return "ε"
            if len(exprs) == 1:
                return exprs[0]
            resultado = []
            for e in exprs:
                if '+' in e and not (e.startswith('(') and e.endswith(')')):
                    resultado.append(f"({e})")
                else:
                    resultado.append(e)
            return ''.join(resultado)

        estados_a_eliminar = estados[:]
        for estado in estados_a_eliminar:
            loop = matriz_expandida[estado][estado]
            loop_clausura = f"({loop})*" if loop else ""
            for i in estados_expandidos:
                if i == estado:
                    continue
                for j in estados_expandidos:
                    if j == estado:
                        continue
                    directa = matriz_expandida[i][j]
                    i_to_k = matriz_expandida[i][estado]
                    k_to_j = matriz_expandida[estado][j]
                    if i_to_k and k_to_j:
                        via_estado = concatenar_expresiones(i_to_k, loop_clausura, k_to_j)
                        matriz_expandida[i][j] = combinar_expresiones(directa, via_estado)
        regex = matriz_expandida[estado_nuevo_inicial][estado_nuevo_final]

        def simplificar_regex(expr, max_iter=20):
            if not expr:
                return "∅"

            class SimplificadorRegex:
                @staticmethod
                def es_simple(expr):
                    return not any(c in expr for c in '+*()')

                @staticmethod
                def descomponer_alternativa(expr, profundidad=0):
                    if profundidad > 30:
                        return [expr]
                    if not expr.startswith('(') or not expr.endswith(')'):
                        return [expr]
                    contenido = expr[1:-1]
                    if '+' not in contenido:
                        return [expr]
                    resultado = []
                    nivel = 0
                    inicio = 0
                    for i, c in enumerate(contenido):
                        if c == '(':
                            nivel += 1
                        elif c == ')':
                            nivel -= 1
                        elif c == '+' and nivel == 0:
                            resultado.append(contenido[inicio:i])
                            inicio = i + 1
                    resultado.append(contenido[inicio:])
                    return resultado

                @staticmethod
                def extraer_factor_comun(alternativas):
                    if len(alternativas) <= 1:
                        return "", alternativas, ""
                    prefijo = ""
                    i = 0
                    max_longitud = min(len(alt) for alt in alternativas)
                    while i < max_longitud:
                        char_i = alternativas[0][i]
                        if all(alt[i] == char_i for alt in alternativas):
                            prefijo += char_i
                            i += 1
                        else:
                            break
                    if prefijo:
                        alternativas = [alt[len(prefijo):] for alt in alternativas]
                    sufijo = ""
                    i = 1
                    max_longitud = min(len(alt) for alt in alternativas)
                    while i <= max_longitud:
                        if all(len(alt) >= i for alt in alternativas):
                            char_i = alternativas[0][-i]
                            if all(alt[-i] == char_i for alt in alternativas):
                                sufijo = char_i + sufijo
                                i += 1
                            else:
                                break
                        else:
                            break
                    if sufijo:
                        alternativas = [alt[:-len(sufijo)] for alt in alternativas]
                    return prefijo, alternativas, sufijo

                @staticmethod
                def factorizar_alternativas(expr, profundidad=0):
                    if profundidad > 10:
                        return expr
                    alternativas = SimplificadorRegex.descomponer_alternativa(expr, profundidad)
                    if len(alternativas) <= 1:
                        return expr
                    prefijo, alternativas_sin_factores, sufijo = SimplificadorRegex.extraer_factor_comun(alternativas)
                    if all(not alt for alt in alternativas_sin_factores):
                        return prefijo + sufijo
                    alternativas_sin_factores = [alt for alt in alternativas_sin_factores if alt]
                    centro = ""
                    if alternativas_sin_factores:
                        if len(alternativas_sin_factores) == 1:
                            centro = alternativas_sin_factores[0]
                        else:
                            centro = f"({'+'.join(alternativas_sin_factores)})"
                    return prefijo + centro + sufijo

                @staticmethod
                def simplificar_repeticiones(expr):
                    if SimplificadorRegex.es_simple(expr):
                        return expr
                    if expr.endswith('*') and expr.startswith('(') and expr.endswith(')*'):
                        contenido = expr[1:-2]
                        if contenido.endswith('*'):
                            return contenido
                    if expr.endswith('**'):
                        return expr[:-1]
                    if expr.startswith('(') and expr.endswith(')*'):
                        alternativas = SimplificadorRegex.descomponer_alternativa(expr[:-1])
                        if 'ε' in alternativas:
                            alternativas.remove('ε')
                            if len(alternativas) == 1:
                                return alternativas[0] + '*'
                    return expr

                @staticmethod
                def simplificar_clausura_vacia(expr):
                    if expr == 'ε*':
                        return 'ε'
                    if expr == '∅*':
                        return 'ε'
                    return expr

                @staticmethod
                def eliminar_parentesis_redundantes(expr, profundidad=0):
                    if profundidad > 10:
                        return expr
                    if not expr or len(expr) <= 2:
                        return expr
                    if expr.startswith('(') and expr.endswith(')'):
                        contenido = expr[1:-1]
                        nivel_parentesis = 0
                        necesita_parentesis = False
                        for i, c in enumerate(contenido):
                            if c == '(':
                                nivel_parentesis += 1
                            elif c == ')':
                                nivel_parentesis -= 1
                            elif c == '+' and nivel_parentesis == 0:
                                necesita_parentesis = True
                                break
                            elif c == '*' and i < len(contenido) - 1 and contenido[i + 1] not in ')*+':
                                necesita_parentesis = True
                                break
                        if not necesita_parentesis:
                            return SimplificadorRegex.eliminar_parentesis_redundantes(contenido, profundidad + 1)
                    resultado = ""
                    nivel_parentesis = 0
                    inicio_subexpr = 0
                    for i, c in enumerate(expr):
                        if c == '(':
                            if nivel_parentesis == 0:
                                resultado += expr[inicio_subexpr:i]
                                inicio_subexpr = i
                            nivel_parentesis += 1
                        elif c == ')':
                            nivel_parentesis -= 1
                            if nivel_parentesis == 0:
                                subexpr = expr[inicio_subexpr:i + 1]
                                resultado += SimplificadorRegex.eliminar_parentesis_redundantes(subexpr,
                                                                                                profundidad + 1)
                                inicio_subexpr = i + 1
                    resultado += expr[inicio_subexpr:]
                    return resultado

                @staticmethod
                def fusionar_alternativas(expr):
                    alternativas = SimplificadorRegex.descomponer_alternativa(expr)
                    if len(alternativas) <= 1:
                        return expr
                    alternativas = sorted(set(alternativas))
                    i = 0
                    while i < len(alternativas) and i < 100:
                        j = 0
                        while j < len(alternativas) and j < 100:
                            if i != j and i < len(alternativas) and j < len(alternativas):
                                if alternativas[i] in alternativas[j]:
                                    alternativas.pop(i)
                                    i -= 1
                                    break
                            j += 1
                        i += 1
                    if len(alternativas) == 1:
                        return alternativas[0]
                    return f"({'+'.join(alternativas)})"

                @staticmethod
                def simplificar_concatenacion_vacia(expr):
                    if 'ε' not in expr and '∅' not in expr:
                        return expr
                    if 'ε' in expr and not (expr == 'ε'):
                        expr = expr.replace('εε', 'ε')
                        if expr.startswith('ε') and len(expr) > 1:
                            expr = expr[1:]
                        if expr.endswith('ε') and len(expr) > 1:
                            expr = expr[:-1]
                    if '∅' in expr:
                        if expr == '∅' or expr == '(∅)':
                            return '∅'
                    return expr

            anterior = ""
            expr = expr.replace('·', '')
            iter_count = 0
            while anterior != expr and iter_count < max_iter:
                anterior = expr
                iter_count += 1
                expr = SimplificadorRegex.simplificar_clausura_vacia(expr)
                expr = SimplificadorRegex.simplificar_concatenacion_vacia(expr)
                if len(expr) < 1000:
                    expr = SimplificadorRegex.factorizar_alternativas(expr)
                    expr = SimplificadorRegex.fusionar_alternativas(expr)
                    expr = SimplificadorRegex.simplificar_repeticiones(expr)
                    expr = SimplificadorRegex.eliminar_parentesis_redundantes(expr)
                simplificaciones = [
                    ('(ε)', 'ε'),
                    ('εε', 'ε'),
                    ('ε+ε', 'ε'),
                    ('∅+', ''),
                    ('+∅', ''),
                    ('(∅)', '∅'),
                    ('ε*', 'ε'),
                    ('∅*', 'ε'),
                    ('**', '*'),
                    ('((', '('),
                    ('))', ')'),
                    ('ε(', '('),
                    (')ε', ')'),
                    ('(a+b)+(a+b)', 'a+b'),
                    ('(a+b)+a', 'a+b'),
                    ('a+(a+b)', 'a+b'),
                    ('(a)(b)', 'ab'),
                    ('(ab)c', 'abc'),
                    ('a(bc)', 'abc'),
                    ('(ε+', '('),
                    ('+ε)', ')'),
                    ('a+a', 'a'),
                ]
                for patron, reemplazo in simplificaciones[:10]:
                    if patron in expr:
                        expr = expr.replace(patron, reemplazo)
                        break
            if not expr:
                return "∅"
            if expr == "()" or expr == "(ε)" or expr == "()":
                return "ε"
            return expr

        resultado = simplificar_regex(regex)
        sys.setrecursionlimit(old_limit)
        return resultado
    except RecursionError:
        sys.setrecursionlimit(old_limit)
        return "Error: Recursión excesiva al procesar la expresión regular"
    except Exception as e:
        sys.setrecursionlimit(old_limit)
        return f"Error: {str(e)}"

# --- Constantes ---
LAMBDA = "ε"
EMPTY_SET = "Ø"

# --- NUEVAS FUNCiones auxiliares con simplificación avanzada ---

def get_or_terms(s: str) -> list[str]:
    """Divide una expresión por el operador '+' de forma segura, respetando los paréntesis."""
    if s == EMPTY_SET: return []
    terms = []
    balance = 0
    start_index = 0
    for i, char in enumerate(s):
        if char == '(': balance += 1
        elif char == ')': balance -= 1
        elif char == '+' and balance == 0:
            terms.append(s[start_index:i])
            start_index = i + 1
    terms.append(s[start_index:])
    return terms

def smart_or(r1: str, r2: str) -> str:
    """Une dos expresiones con OR (+), desarmando, ordenando y eliminando duplicados."""
    if r1 == EMPTY_SET: return r2
    if r2 == EMPTY_SET: return r1

    terms1 = get_or_terms(r1)
    terms2 = get_or_terms(r2)
    unique_terms = set(terms1) | set(terms2)
    
    # Simplificación: R + R = R (manejado por el set)
    # Simplificación: R + Ø = R (manejado al inicio)

    # Ordenar para consistencia
    sorted_terms = sorted(list(unique_terms))
    return "+".join(sorted_terms)

def smart_concatenate(r1: str, r2: str) -> str:
    """Concatena dos expresiones, añadiendo paréntesis solo cuando es estrictamente necesario."""
    if r1 == EMPTY_SET or r2 == EMPTY_SET: return EMPTY_SET
    if r1 == LAMBDA: return r2
    if r2 == LAMBDA: return r1

    # Añadir paréntesis a una sub-expresión si contiene un '+' a nivel superior
    part1 = f"({r1})" if '+' in r1 else r1
    part2 = f"({r2})" if '+' in r2 else r2
    
    return f"{part1}{part2}"

def smart_star(r: str) -> str:
    """Aplica la cerradura de Kleene (*), con reglas de simplificación avanzadas."""
    # Limpiar paréntesis externos primero para analizar la expresión interna
    r_cleaned = final_cleanup(r)

    if r_cleaned == EMPTY_SET or r_cleaned == LAMBDA:
        return LAMBDA
    if r_cleaned.endswith("*"):
        return r_cleaned

    # --- REGLA DE SIMPLIFICACIÓN CLAVE: (ε+R)* = R* ---
    or_terms = get_or_terms(r_cleaned)
    if LAMBDA in or_terms:
        other_terms = [term for term in or_terms if term != LAMBDA]
        if not other_terms:
            return LAMBDA # (ε)* = ε
        
        # Reconstruir la expresión sin el ε y aplicar la estrella de nuevo
        new_r = "+".join(other_terms)
        return smart_star(new_r)
    
    # Si no aplica la regla del épsilon, añadir paréntesis si es necesario
    if len(r_cleaned) > 1 or '+' in r_cleaned:
        return f"({r_cleaned})*"
    
    return f"{r_cleaned}*"

def final_cleanup(expression: str) -> str:
    """Elimina repetidamente los paréntesis externos si envuelven toda la expresión."""
    if not isinstance(expression, str): return expression
    
    while len(expression) > 1 and expression.startswith('(') and expression.endswith(')'):
        balance = 0
        is_wrapper = True
        for i, char in enumerate(expression[1:-1]):
            if char == '(': balance += 1
            elif char == ')': balance -= 1
            if balance < 0:
                is_wrapper = False
                break
        
        if balance == 0 and is_wrapper:
            expression = expression[1:-1]
        else:
            break
            
    return expression

# --- Función Principal (Corregida) ---
def obtener_regex_jflap_exacto(estados_orig, alfabeto, estado_inicial_orig, estados_aceptacion_orig_set,
                                 transiciones_orig):
    """Calcula la ER usando eliminación de estados con simplificación avanzada."""
    try:
        # (El código de preprocesamiento y creación de matriz R se mantiene igual que en la versión anterior)
        # ...
        # --- 1. Preprocesamiento: Asegurar un único estado final ---
        # ... (Copiado de la versión anterior, es correcto)
        estados = list(copy.deepcopy(estados_orig))
        transiciones = copy.deepcopy(transiciones_orig)
        estado_inicial = copy.deepcopy(estado_inicial_orig)
        estados_aceptacion = copy.deepcopy(estados_aceptacion_orig_set)

        if not estados_aceptacion: return EMPTY_SET

        if len(estados_aceptacion) > 1 or estado_inicial in estados_aceptacion:
            qf_final = "qf_new"
            idx = 0
            while qf_final in estados:
                qf_final = f"qf_new_{idx}"
                idx += 1
            
            estados.append(qf_final)
            for viejo_final in list(estados_aceptacion):
                if viejo_final not in transiciones: transiciones[viejo_final] = {}
                if LAMBDA not in transiciones[viejo_final]: transiciones[viejo_final][LAMBDA] = []
                transiciones[viejo_final][LAMBDA].append(qf_final)
            
            estados_aceptacion = {qf_final}
        else:
            qf_final = list(estados_aceptacion)[0]

        # --- 2. Creación de Matriz de Expresiones R[p][q] ---
        R = defaultdict(lambda: defaultdict(lambda: EMPTY_SET))
        
        for p in estados:
            for q in estados:
                simbolos_directos = []
                if p in transiciones:
                    for simbolo, destinos in transiciones.get(p, {}).items():
                        lista_destinos = destinos if isinstance(destinos, list) else [destinos]
                        if q in lista_destinos:
                            simbolos_directos.append(simbolo if simbolo else LAMBDA)
                
                expr_union = EMPTY_SET
                if simbolos_directos:
                    expr_union = simbolos_directos[0]
                    for i in range(1, len(simbolos_directos)):
                        expr_union = smart_or(expr_union, simbolos_directos[i])
                
                if p == q:
                    R[p][q] = smart_or(LAMBDA, expr_union)
                else:
                    R[p][q] = expr_union
        
        # --- 3. Eliminación de Estados ---
        estados_a_eliminar = [s for s in estados_orig if s != estado_inicial and s not in estados_aceptacion]
        
        for k in estados_a_eliminar:
            for p in estados:
                if p == k: continue
                for q in estados:
                    if q == k: continue
                    
                    via_k = smart_concatenate(smart_concatenate(R[p][k], smart_star(R[k][k])), R[k][q])
                    R[p][q] = smart_or(R[p][q], via_k)
        
        # --- 4. Cálculo de la Expresión Final ---
        i, j = estado_inicial, qf_final
        
        if R[i][j] == EMPTY_SET: return EMPTY_SET
        
        r_ii_star = smart_star(R[i][i])
        r_jj_star = smart_star(R[j][j])
        r_ij = R[i][j]
        r_ji = R[j][i]

        ciclo_grande = smart_concatenate(smart_concatenate(r_ii_star, r_ij), smart_concatenate(r_jj_star, r_ji))
        camino_principal = smart_concatenate(smart_concatenate(r_ii_star, r_ij), r_jj_star)
        
        expression = smart_concatenate(smart_star(ciclo_grande), camino_principal)
        
        return final_cleanup(expression)

    except Exception as e:
        print(f"Error en obtener_regex_jflap_exacto: {e}")
        print(traceback.format_exc())
        return f"Error: {str(e)}"


# --- Funciones originales del usuario (verificar_lenguaje, kleene_cerradura, etc.) ---
def verificar_lenguaje(alfabeto, lenguaje):
    for cadena in lenguaje:
        for caracter in cadena:
            if caracter not in alfabeto:
                return False
    return True


def kleene_cerradura(alfabeto, n):
    kleene = ['']
    for i in range(1, n + 1):
        combinaciones = [''.join(p) for p in itertools.product(alfabeto, repeat=i)]
        kleene.extend(combinaciones)
    return kleene


def clausura_positiva(alfabeto, n):
    return kleene_cerradura(alfabeto, n)[1:]


def concatenar_lenguajes(lenguaje1, lenguaje2):
    # Asegurarse que son listas de strings
    l1 = [str(s) for s in lenguaje1]
    l2 = [str(s) for s in lenguaje2]
    return [x + y for x in l1 for y in l2]


def potenciar_lenguaje(lenguaje, potencia):
    if potencia == 0:
        return ['']  # L^0 es {epsilon}
    if potencia < 0:
        raise ValueError("La potencia no puede ser negativa")
    if potencia == 1:
        return lenguaje[:]  # Devolver copia

    resultado = lenguaje[:]
    base = lenguaje[:]
    for _ in range(potencia - 1):
        resultado = concatenar_lenguajes(resultado, base)
    return resultado


def reflexion_lenguaje(lenguaje):
    return [str(cadena)[::-1] for cadena in lenguaje]


def union_lenguajes(lenguaje1, lenguaje2):
    return list(set(map(str, lenguaje1)).union(set(map(str, lenguaje2))))


def interseccion_lenguajes(lenguaje1, lenguaje2):
    return list(set(map(str, lenguaje1)).intersection(set(map(str, lenguaje2))))


def diferencia_lenguajes(lenguaje1, lenguaje2):
    return list(set(map(str, lenguaje1)).difference(set(map(str, lenguaje2))))


def validar_cadena(alfabeto, estados, transiciones, estado_inicial, aceptados, cadena):
    if not estado_inicial or not aceptados or not transiciones:
        return False, "Debe definir el autómata completo antes de validar."
    if estado_inicial not in estados:
        return False, f"Estado inicial '{estado_inicial}' no definido en la lista de estados."
    for estado_acc in aceptados:
        if estado_acc not in estados:
            return False, f"Estado de aceptación '{estado_acc}' no definido en la lista de estados."

    est_actual = estado_inicial
    for caracter in cadena:
        if caracter not in alfabeto:
            return False, f"El carácter '{caracter}' no pertenece al alfabeto."
        # Asegurarse que la estructura de transiciones es correcta
        if est_actual not in transiciones:
            return False, f"Error: No hay transiciones definidas para el estado '{est_actual}'."
        if caracter not in transiciones[est_actual]:
            return False, f"Error: Transición no definida para estado {est_actual} y símbolo '{caracter}'."

        destino = transiciones[est_actual][caracter]
        # Verificar si el destino es válido
        if destino not in estados:
            return False, f"Error: La transición desde '{est_actual}' con '{caracter}' lleva a un estado inválido '{destino}'."
        est_actual = destino

    if est_actual in aceptados:
        return True, f"La cadena '{cadena}' es aceptada. Estado final: {est_actual}."
    else:
        return False, f"La cadena '{cadena}' no es aceptada. Estado final: {est_actual}."


# --- Interfaz Gráfica (Flet) ---

def create_automata_view(page: ft.Page):
    page.title = "Operaciones sobre Lenguajes y Autómatas"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 0
    page.bgcolor = ft.Colors.BLUE_GREY_50
    primary_color = ft.Colors.BLUE_600
    text_color = ft.Colors.GREY_900
    input_border_color = ft.Colors.BLUE_400

    def create_styled_textfield(label, multiline=False, tooltip=None, width=None):
        return ft.TextField(
            label=label,
            border_color=input_border_color,
            focused_border_color=primary_color,
            text_style=ft.TextStyle(color=text_color),
            label_style=ft.TextStyle(color=ft.Colors.BLUE_GREY_600),
            cursor_color=primary_color,
            bgcolor=ft.Colors.WHITE,
            dense=True,
            multiline=multiline,
            min_lines=1 if not multiline else 3,
            max_lines=1 if not multiline else 5,
            tooltip=tooltip,
            expand=width is None,  # Expandir solo si no se especifica ancho
            width=width,  # Permitir ancho fijo
            value="",
            border_radius=8,
        )

    alfabeto_input = create_styled_textfield("Alfabeto (ej: a,b,c)")
    n_input = create_styled_textfield("Número máximo de combinaciones", width=250)  # Ancho fijo
    lenguaje1_input = create_styled_textfield("Primer lenguaje (ej: a,ab,bba)")
    lenguaje2_input = create_styled_textfield("Segundo lenguaje (ej: b,ba)")
    potencia_input = create_styled_textfield("Potencia del lenguaje", width=200)  # Ancho fijo
    estados_input = create_styled_textfield("Estados (ej: q0,q1,q2)")
    estado_inicial_input = create_styled_textfield("Estado inicial", width=150)  # Ancho fijo
    aceptados_input = create_styled_textfield("Estados de aceptación (ej: q2,q3)")
    cadena_input = create_styled_textfield("Cadena a validar")
    regex_input = create_styled_textfield("Expresión regular", multiline=True)  # Permitir Regex largas

    # Contenedor para mostrar resultados y mensajes de error
    resultado_text = ft.Text("", size=16, color=text_color, selectable=True)  # Tamaño ajustado, seleccionable
    resultado_container = ft.Container(
        content=ft.Column([resultado_text], scroll=ft.ScrollMode.AUTO),  # Hacer scroll vertical si es largo
        padding=15,
        bgcolor=ft.Colors.WHITE,
        border_radius=10,
        border=ft.border.all(1, ft.Colors.BLUE_GREY_200),
        margin=ft.margin.only(top=10),
        expand=True,  # Permitir que crezca verticalmente
        height=150,  # Altura inicial, puede crecer si el contenido es mayor
    )

    # Estado para almacenar la UI de transiciones y la imagen
    transiciones_ui_columna = ft.Column([])
    imagen_automata_container = ft.Container(data='imagen_automataMt')  # Contenedor vacío inicial

    # --- Lógica de los botones ---

    # Limpiar controles dinámicos (transiciones, imagen)
    def limpiar_controles_dinamicos(layout_principal):
        indices_a_eliminar = []
        for i, control in enumerate(layout_principal.controls):
            # Eliminar columna de UI de transiciones o imagen anterior
            if isinstance(control, ft.Column) and control == transiciones_ui_columna:
                indices_a_eliminar.append(i)
            elif isinstance(control, ft.Container) and control.data == 'imagen_automata':
                indices_a_eliminar.append(i)
            # Eliminar botones de acción del autómata si existen
            elif isinstance(control, ft.Row) and control.data == 'automata_actions':
                indices_a_eliminar.append(i)

        # Eliminar en orden inverso para no afectar índices
        for i in sorted(indices_a_eliminar, reverse=True):
            del layout_principal.controls[i]
        # Limpiar el contenido de la columna de transiciones por si acaso
        transiciones_ui_columna.controls.clear()
        # Limpiar imagen
        imagen_automata_container.content = None
        # Limpiar resultado
        resultado_text.value = ""

    # Actualizar el layout principal con los inputs necesarios
    def show_inputs(layout_principal, *inputs_principales, botones_accion=None):
        limpiar_controles_dinamicos(layout_principal)
        # Reconstruir controles base
        controles = [layout_principal.controls[0], layout_principal.controls[1]]  # Títulos
        # Agrupar inputs en filas si es necesario para mejor layout
        filas = []
        controles_fila_actual = []
        ancho_total_fila = 0
        max_ancho_fila = 800  # Ajustar según necesidad

        for control in inputs_principales:
            ancho_control = control.width if control.width else 250  # Ancho estimado si expand=True
            if not controles_fila_actual or (ancho_total_fila + ancho_control) > max_ancho_fila:
                if controles_fila_actual:
                    filas.append(ft.Row(controles_fila_actual, spacing=10))
                controles_fila_actual = [control]
                ancho_total_fila = ancho_control
            else:
                controles_fila_actual.append(control)
                ancho_total_fila += ancho_control + 10  # Añadir espacio

        if controles_fila_actual:  # Añadir la última fila
            filas.append(ft.Row(controles_fila_actual, spacing=10))

        controles.extend(filas)

        # Añadir botones de acción si existen
        if botones_accion:
            controles.append(ft.Row(botones_accion, spacing=10, alignment=ft.MainAxisAlignment.START))

        controles.append(resultado_container)  # Resultado siempre al final
        layout_principal.controls = controles
        page.update()

    # --- Implementaciones de Operaciones ---

    def realizar_cerradura_kleene(e, layout_principal):
        def on_calculate(ev):
            alfabeto = [a.strip() for a in alfabeto_input.value.split(',') if a.strip()]
            if not alfabeto:
                resultado_text.value = "El alfabeto no puede estar vacío."
                resultado_text.color = ft.Colors.RED_600
                page.update()
                return
            try:
                n = int(n_input.value)
                if n < 0: raise ValueError("N debe ser no negativo.")
                resultado = kleene_cerradura(alfabeto, n)
                resultado_text.value = f"Cerradura de Kleene (hasta n={n}): {', '.join(resultado) if resultado else '{ε}'}"
                resultado_text.color = text_color
            except ValueError:
                resultado_text.value = "Por favor ingrese un número válido (entero no negativo) para n."
                resultado_text.color = ft.Colors.RED_600
            page.update()

        botones = [ft.ElevatedButton("Calcular Kleene", on_click=on_calculate)]
        show_inputs(layout_principal, alfabeto_input, n_input, botones_accion=botones)

    def realizar_clausura_positiva(e, layout_principal):
        def on_calculate(ev):
            alfabeto = [a.strip() for a in alfabeto_input.value.split(',') if a.strip()]
            if not alfabeto:
                resultado_text.value = "El alfabeto no puede estar vacío."
                resultado_text.color = ft.Colors.RED_600
                page.update()
                return
            try:
                n = int(n_input.value)
                if n < 0: raise ValueError("N debe ser no negativo.")
                resultado = clausura_positiva(alfabeto, n)
                resultado_text.value = f"Clausura Positiva (hasta n={n}): {', '.join(resultado) if resultado else '{}'}"
                resultado_text.color = text_color
            except ValueError:
                resultado_text.value = "Por favor ingrese un número válido (entero no negativo) para n."
                resultado_text.color = ft.Colors.RED_600
            page.update()

        botones = [ft.ElevatedButton("Calcular Positiva", on_click=on_calculate)]
        show_inputs(layout_principal, alfabeto_input, n_input, botones_accion=botones)

    def realizar_concatenacion(e, layout_principal):
        def on_calculate(ev):
            alfabeto = [a.strip() for a in alfabeto_input.value.split(',') if a.strip()]
            lenguaje1 = [l.strip() for l in lenguaje1_input.value.split(',') if
                         l.strip() or l == '']  # Permitir epsilon
            lenguaje2 = [l.strip() for l in lenguaje2_input.value.split(',') if
                         l.strip() or l == '']  # Permitir epsilon

            # Incluir epsilon si se omite pero los campos están vacíos
            if lenguaje1_input.value.strip() == "" and not lenguaje1: lenguaje1 = ['']
            if lenguaje2_input.value.strip() == "" and not lenguaje2: lenguaje2 = ['']

            if not verificar_lenguaje(alfabeto, [c for c in lenguaje1 if c]) or \
                    not verificar_lenguaje(alfabeto, [c for c in lenguaje2 if c]):
                resultado_text.value = "Error: Uno o más caracteres en los lenguajes no pertenecen al alfabeto."
                resultado_text.color = ft.Colors.RED_600
            else:
                resultado = concatenar_lenguajes(lenguaje1, lenguaje2)
                resultado_text.value = f"L1 . L2 = {{{', '.join(resultado) if resultado else ''}}}"
                resultado_text.color = text_color
            page.update()

        botones = [ft.ElevatedButton("Calcular Concatenación", on_click=on_calculate)]
        show_inputs(layout_principal, alfabeto_input, lenguaje1_input, lenguaje2_input, botones_accion=botones)

    def realizar_potenciacion(e, layout_principal):
        def on_calculate(ev):
            lenguaje = [l.strip() for l in lenguaje1_input.value.split(',') if l.strip() or l == '']  # Permitir epsilon
            if lenguaje1_input.value.strip() == "" and not lenguaje: lenguaje = ['']

            try:
                potencia = int(potencia_input.value)
                if potencia < 0: raise ValueError("Potencia debe ser >= 0")
                # Necesitamos alfabeto para verificar si es necesario
                # alfabeto = [a.strip() for a in alfabeto_input.value.split(',') if a.strip()]
                # if not verificar_lenguaje(alfabeto, [c for c in lenguaje if c]):
                #      resultado_text.value = "Error: Caracteres no pertenecen al alfabeto."
                #      resultado_text.color = ft.Colors.RED_600
                # else:
                resultado = potenciar_lenguaje(lenguaje, potencia)
                resultado_text.value = f"L^{potencia} = {{{', '.join(resultado) if resultado else ('ε' if potencia == 0 else '')}}}"
                resultado_text.color = text_color
            except ValueError as err:
                resultado_text.value = f"Error: Ingrese una potencia válida (entero >= 0). {err}"
                resultado_text.color = ft.Colors.RED_600
            page.update()

        botones = [ft.ElevatedButton("Calcular Potencia", on_click=on_calculate)]
        # Podríamos añadir alfabeto_input si queremos validar
        show_inputs(layout_principal, lenguaje1_input, potencia_input, botones_accion=botones)

    def realizar_reflexion(e, layout_principal):
        def on_calculate(ev):
            lenguaje = [l.strip() for l in lenguaje1_input.value.split(',') if l.strip() or l == '']  # Permitir epsilon
            if lenguaje1_input.value.strip() == "" and not lenguaje: lenguaje = ['']
            resultado = reflexion_lenguaje(lenguaje)
            resultado_text.value = f"L^R = {{{', '.join(resultado) if resultado else ''}}}"
            resultado_text.color = text_color
            page.update()

        botones = [ft.ElevatedButton("Calcular Reflexión", on_click=on_calculate)]
        show_inputs(layout_principal, lenguaje1_input, botones_accion=botones)

    def realizar_union(e, layout_principal):
        def on_calculate(ev):
            lenguaje1 = [l.strip() for l in lenguaje1_input.value.split(',') if l.strip() or l == '']
            lenguaje2 = [l.strip() for l in lenguaje2_input.value.split(',') if l.strip() or l == '']
            if lenguaje1_input.value.strip() == "" and not lenguaje1: lenguaje1 = ['']
            if lenguaje2_input.value.strip() == "" and not lenguaje2: lenguaje2 = ['']
            resultado = union_lenguajes(lenguaje1, lenguaje2)
            resultado_text.value = f"L1 ∪ L2 = {{{', '.join(sorted(resultado)) if resultado else ''}}}"
            resultado_text.color = text_color
            page.update()

        botones = [ft.ElevatedButton("Calcular Unión", on_click=on_calculate)]
        show_inputs(layout_principal, lenguaje1_input, lenguaje2_input, botones_accion=botones)

    def realizar_interseccion(e, layout_principal):
        def on_calculate(ev):
            lenguaje1 = [l.strip() for l in lenguaje1_input.value.split(',') if l.strip() or l == '']
            lenguaje2 = [l.strip() for l in lenguaje2_input.value.split(',') if l.strip() or l == '']
            if lenguaje1_input.value.strip() == "" and not lenguaje1: lenguaje1 = ['']
            if lenguaje2_input.value.strip() == "" and not lenguaje2: lenguaje2 = ['']
            resultado = interseccion_lenguajes(lenguaje1, lenguaje2)
            resultado_text.value = f"L1 ∩ L2 = {{{', '.join(sorted(resultado)) if resultado else ''}}}"
            resultado_text.color = text_color
            page.update()

        botones = [ft.ElevatedButton("Calcular Intersección", on_click=on_calculate)]
        show_inputs(layout_principal, lenguaje1_input, lenguaje2_input, botones_accion=botones)

    def realizar_diferencia(e, layout_principal):
        def on_calculate(ev):
            lenguaje1 = [l.strip() for l in lenguaje1_input.value.split(',') if l.strip() or l == '']
            lenguaje2 = [l.strip() for l in lenguaje2_input.value.split(',') if l.strip() or l == '']
            if lenguaje1_input.value.strip() == "" and not lenguaje1: lenguaje1 = ['']
            if lenguaje2_input.value.strip() == "" and not lenguaje2: lenguaje2 = ['']
            resultado = diferencia_lenguajes(lenguaje1, lenguaje2)
            resultado_text.value = f"L1 - L2 = {{{', '.join(sorted(resultado)) if resultado else ''}}}"
            resultado_text.color = text_color
            page.update()

        botones = [ft.ElevatedButton("Calcular Diferencia", on_click=on_calculate)]
        show_inputs(layout_principal, lenguaje1_input, lenguaje2_input, botones_accion=botones)

    # --- Lógica del Autómata ---
    def realizar_automata(e, layout_principal):
        # Diccionario para almacenar los TextFields de transiciones
        transiciones_fields = {}

        # Botón para definir transiciones (se activa después de ingresar estados/alfabeto)
        definir_transiciones_btn = ft.ElevatedButton(
            "Definir Transiciones",
            on_click=lambda ev: on_definir_transiciones(ev, layout_principal, transiciones_fields),
            disabled=False  # Se habilita/deshabilita dinámicamente
        )

        # Botones de acción del autómata (inicialmente ocultos)
        validar_btn = ft.ElevatedButton("Validar Cadena", on_click=lambda ev: on_validar_cadena(ev, layout_principal,
                                                                                                transiciones_fields)
                                        )
        regex_btn = ft.ElevatedButton("Obtener Expresión Regular (Estilo JFLAP)",
                                      on_click=lambda ev: on_obtener_regex_jflap(ev, layout_principal,
                                                                                 transiciones_fields))
        regex_app_btn = ft.ElevatedButton("Regex_AutomatApp",
                                          on_click=lambda ev: on_obtener_regex_jflap2(ev, layout_principal,
                                                                                      transiciones_fields)
                                         )
        visualizar_btn = ft.ElevatedButton("Visualizar Autómata",
                                           on_click=lambda ev: on_visualizar_automata(ev, layout_principal,
                                                                                      transiciones_fields)
                                           )
        botones_automata = ft.Column([validar_btn, regex_btn, regex_app_btn, visualizar_btn], spacing=10, visible=False,
                                  data='automata_actions')

        # Mostrar inputs iniciales y botón para definir transiciones
        show_inputs(layout_principal,
                    alfabeto_input, estados_input, estado_inicial_input, aceptados_input, cadena_input,
                    botones_accion=[definir_transiciones_btn])
        # Insertar placeholders para la UI de transiciones y botones de acción del autómata
        layout_principal.controls.insert(-1, transiciones_ui_columna)  # Penúltimo lugar
        layout_principal.controls.insert(-1, botones_automata)  # Antepenúltimo lugar
        layout_principal.controls.insert(-1, imagen_automata_container)  # Antes del resultado
        page.update()

    def on_definir_transiciones(e, layout_principal, transiciones_fields):
        alfabeto = [a.strip() for a in alfabeto_input.value.split(',') if a.strip()]
        estados = [s.strip() for s in estados_input.value.split(',') if s.strip()]

        if not alfabeto or not estados:
            resultado_text.value = "Error: Defina el alfabeto y los estados antes de crear transiciones."
            resultado_text.color = ft.Colors.RED_600
            page.update()
            return

        # Limpiar controles anteriores y campos
        transiciones_ui_columna.controls.clear()
        transiciones_fields.clear()
        imagen_automata_container.content = None

        # Crear UI para definir transiciones
        for estado in estados:
            fila_estado = [ft.Text(f"{estado}:", width=50, weight=ft.FontWeight.BOLD)]
            transiciones_fields[estado] = {}
            for simbolo in alfabeto:
                tooltip_txt = f"Estado destino para {estado} con símbolo '{simbolo}'"
                textfield = create_styled_textfield(f"{simbolo} →", width=100, tooltip=tooltip_txt)
                transiciones_fields[estado][simbolo] = textfield
                fila_estado.append(textfield)
            transiciones_ui_columna.controls.append(
                ft.Row(fila_estado, spacing=5, wrap=True))

        # --- INICIO DE LA CORRECCIÓN ---

        # 1. Buscar el control como ft.Column y renombrar la variable para mayor claridad
        botones_columna = next(
            (c for c in layout_principal.controls if isinstance(c, ft.Column) and c.data == 'automata_actions'), None)
        
        # 2. Usar la nueva variable para hacer visible la columna
        if botones_columna:
            botones_columna.visible = True
            
            # (Opcional pero recomendado): Si ya quitaste 'visible=False' de los botones
            # individuales como te sugerí antes, este bucle 'for' ya no es necesario.
            # Solo con hacer visible la columna es suficiente.
            for btn in botones_columna.controls:
                if isinstance(btn, ft.ElevatedButton):
                    btn.visible = True
        
        # --- FIN DE LA CORRECCIÓN ---

        resultado_text.value = "Ingrese los estados destino para cada transición."
        resultado_text.color = text_color
        page.update()
    def _get_automata_data(layout_principal, transiciones_fields):
        """ Función auxiliar para recolectar y validar datos del autómata """
        try:
            alfabeto = [a.strip() for a in alfabeto_input.value.split(',') if a.strip()]
            estados = [s.strip() for s in estados_input.value.split(',') if s.strip()]
            estado_inicial = estado_inicial_input.value.strip()
            aceptados_str = [s.strip() for s in aceptados_input.value.split(',') if s.strip()]
            aceptados = set(aceptados_str)  # Usar set

            # Validaciones básicas
            if not alfabeto: raise ValueError("El alfabeto no puede estar vacío.")
            if not estados: raise ValueError("La lista de estados no puede estar vacía.")
            if not estado_inicial: raise ValueError("Debe definir un estado inicial.")
            if estado_inicial not in estados: raise ValueError(
                f"Estado inicial '{estado_inicial}' no está en la lista de estados.")
            if not aceptados: raise ValueError("Debe definir al menos un estado de aceptación.")
            for acc in aceptados:
                if acc not in estados: raise ValueError(f"Estado de aceptación '{acc}' no está en la lista de estados.")

            # Obtener transiciones desde la UI
            transiciones_definidas = defaultdict(dict)
            for estado_origen, campos_simbolo in transiciones_fields.items():
                if estado_origen not in estados: continue  # Ignorar si el estado ya no existe
                for simbolo, textfield in campos_simbolo.items():
                    if simbolo not in alfabeto: continue  # Ignorar si el símbolo ya no existe
                    estado_destino = textfield.value.strip()
                    if estado_destino:  # Solo añadir si hay un destino definido
                        if estado_destino not in estados:
                            raise ValueError(
                                f"Transición desde '{estado_origen}' con '{simbolo}' lleva a estado inválido '{estado_destino}'.")
                        transiciones_definidas[estado_origen][simbolo] = estado_destino

            # Verificar que todos los estados (excepto quizás los finales sin salida) tengan transiciones definidas para todo el alfabeto (para AFD)
            # Opcional: podrías añadir esta validación si quieres forzar un AFD completo
            # for estado in estados:
            #      if estado not in transiciones_definidas and estado not in aceptados: # Asumiendo que aceptados pueden no tener salidas
            #           # Podría ser un estado inalcanzable o necesitar validación
            #           pass
            #      elif estado in transiciones_definidas:
            #            for simbolo in alfabeto:
            #                 if simbolo not in transiciones_definidas[estado]:
            #                      # Falta transición - ¿implica estado sumidero?
            #                      # raise ValueError(f"Falta definir transición para estado '{estado}' con símbolo '{simbolo}'.")
            #                      pass # Permitir AFD parciales por ahora

            return alfabeto, estados, estado_inicial, aceptados, transiciones_definidas

        except ValueError as ve:
            resultado_text.value = f"Error de Validación: {str(ve)}"
            resultado_text.color = ft.Colors.RED_600
            page.update()
            return None  # Indicar error

    def on_validar_cadena(e, layout_principal, transiciones_fields):
        automata_data = _get_automata_data(layout_principal, transiciones_fields)
        if automata_data is None: return  # Error ya mostrado

        alfabeto, estados, estado_inicial, aceptados, transiciones_definidas = automata_data
        cadena = cadena_input.value  # No necesita strip? Depende si quieres permitir espacios

        aceptada, mensaje = validar_cadena(alfabeto, estados, transiciones_definidas, estado_inicial, aceptados, cadena)
        resultado_text.value = mensaje
        resultado_text.color = ft.Colors.GREEN_600 if aceptada else ft.Colors.RED_600
        page.update()

    def on_obtener_regex_jflap(e, layout_principal, transiciones_fields):
        automata_data = _get_automata_data(layout_principal, transiciones_fields)
        if automata_data is None: return  # Error ya mostrado

        alfabeto, estados, estado_inicial, aceptados, transiciones_definidas = automata_data

        # --- Llamada a la nueva función ---
        regex = obtener_regex_jflap_exacto(estados, alfabeto, estado_inicial, aceptados,  # Aceptados ya es set
                                           transiciones_definidas)
        # -----------------------------------

        resultado_text.value = f"Expresión Regular (Estilo JFLAP): {regex}"
        resultado_text.color = ft.Colors.GREEN_600 if not regex.startswith("Error:") else ft.Colors.RED_600
        page.update()

    def on_obtener_regex_jflap2(e, layout_principal, transiciones_fields):
        automata_data = _get_automata_data(layout_principal, transiciones_fields)
        if automata_data is None: return  # Error ya mostrado

        alfabeto, estados, estado_inicial, aceptados, transiciones_definidas = automata_data

        try:
            regex = obtener_regex_afd_jflap2(
                estados=estados,
                alfabeto=alfabeto,
                estado_inicial=estado_inicial,
                estados_aceptacion=aceptados,  # Ya es un set
                transiciones=transiciones_definidas
            )
            resultado_text.value = f"Expresión Regular (Método Alternativo): {regex}"
            resultado_text.color = ft.Colors.GREEN_600 if not regex.startswith("Error:") else ft.Colors.RED_600
        except Exception as ex:
            resultado_text.value = f"Error al generar la expresión regular (Método Alternativo): {str(ex)}"
            resultado_text.color = ft.Colors.RED_600
        page.update()

    def on_visualizar_automata(e, layout_principal, transiciones_fields):
        try:
            # Buscar el contenedor principal (layout_principal es la Column dentro del Container)
            main_content = layout_principal

            # Obtener inputs desde los TextFields definidos en create_automata_view
            inputs = {}
            for control in main_content.controls:
                if isinstance(control, ft.Row):
                    for subcontrol in control.controls:
                        if isinstance(subcontrol, ft.TextField):
                            if subcontrol.label == "Alfabeto (ej: a,b,c)":
                                inputs['alfabeto'] = subcontrol
                            elif subcontrol.label == "Estados (ej: q0,q1,q2)":
                                inputs['estados'] = subcontrol
                            elif subcontrol.label == "Estado inicial":
                                inputs['estado_inicial'] = subcontrol
                            elif subcontrol.label == "Estados de aceptación (ej: q2,q3)":
                                inputs['aceptados'] = subcontrol
                            elif subcontrol.label == "Cadena a validar":
                                inputs['cadena'] = subcontrol

            if not all(key in inputs for key in ['alfabeto', 'estados', 'estado_inicial', 'aceptados']):
                raise ValueError("No se encontraron todos los campos de entrada necesarios.")

            alfabeto = [s.strip() for s in inputs['alfabeto'].value.split(',') if s.strip()]
            estados = [s.strip() for s in inputs['estados'].value.split(',') if s.strip()]
            estado_inicial = inputs['estado_inicial'].value.strip()
            aceptados = [s.strip() for s in inputs['aceptados'].value.split(',') if s.strip()]

            # Validaciones
            if not alfabeto:
                raise ValueError("El alfabeto no puede estar vacío.")
            if not estados:
                raise ValueError("La lista de estados no puede estar vacía.")
            if not estado_inicial:
                raise ValueError("Debe especificar un estado inicial.")
            if estado_inicial not in estados:
                raise ValueError(f"El estado inicial '{estado_inicial}' no está en la lista de estados.")
            for estado in aceptados:
                if estado not in estados:
                    raise ValueError(f"El estado de aceptación '{estado}' no está en la lista de estados.")

            # Obtener transiciones desde transiciones_fields
            transiciones = {}
            for estado_origen, campos_simbolo in transiciones_fields.items():
                transiciones[estado_origen] = {}
                for simbolo, textfield in campos_simbolo.items():
                    estado_destino = textfield.value.strip()
                    if estado_destino:
                        transiciones[estado_origen][simbolo] = estado_destino

            # Generar el grafo
            G = nx.DiGraph()
            G.add_nodes_from(estados)

            # Limpiar y estructurar transiciones
            transition_data = []
            for estado_origen, trans in transiciones.items():
                for simbolo, estado_destino in trans.items():
                    if estado_destino and estado_destino in estados:
                        transition_data.append({
                            'origen': estado_origen,
                            'destino': estado_destino,
                            'simbolo': simbolo
                        })

            # Crear la figura con fondo profesional
            fig, ax = plt.subplots(figsize=(14, 10), dpi=200, facecolor='#f8f9fa')
            ax.set_facecolor('#ffffff')

            # Posicionar nodos con mejor distribución
            pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

            # Ajustar posiciones para mejor distribución circular
            num_states = len(estados)
            if num_states <= 6:
                # Para pocos estados, usar distribución circular
                angle_step = 2 * np.pi / num_states
                radius = 1.2
                for i, state in enumerate(estados):
                    angle = i * angle_step
                    pos[state] = (radius * np.cos(angle), radius * np.sin(angle))

            # Configuración de nodos más profesional
            node_size = 1800
            node_colors = []
            node_edge_colors = []
            node_linewidths = []

            for node in G.nodes():
                if node == estado_inicial and node in aceptados:
                    color = '#4CAF50'  # Verde para inicial y aceptación
                    edge_color = '#2E7D32'
                    linewidth = 3
                elif node == estado_inicial:
                    color = '#2196F3'  # Azul para inicial
                    edge_color = '#1565C0'
                    linewidth = 3
                elif node in aceptados:
                    color = '#FF9800'  # Naranja para aceptación
                    edge_color = '#E65100'
                    linewidth = 3
                else:
                    color = '#FFFFFF'  # Blanco para normales
                    edge_color = '#424242'
                    linewidth = 2

                node_colors.append(color)
                node_edge_colors.append(edge_color)
                node_linewidths.append(linewidth)

            # Dibujar nodos con estilo profesional
            nx.draw_networkx_nodes(G, pos,
                                   node_color=node_colors,
                                   edgecolors=node_edge_colors,
                                   linewidths=node_linewidths,
                                   node_size=node_size,
                                   alpha=0.9)

            # Ya no necesitamos agrupar las transiciones
            # Cada transición será dibujada individualmente

            # Función mejorada para posicionar etiquetas arriba de las flechas
            def get_label_position(x1, y1, x2, y2, rad, is_selfloop=False, loop_angle=0):
                if is_selfloop:
                    # Para bucles, posicionar la etiqueta más alejada del centro
                    radius = 0.55  # Aumentado para estar más arriba del bucle
                    label_x = x1 + radius * np.cos(loop_angle)
                    label_y = y1 + radius * np.sin(loop_angle)
                    return label_x, label_y
                else:
                    # Para transiciones normales, calcular el punto en la curva donde va la etiqueta
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    dx, dy = x2 - x1, y2 - y1
                    length = math.sqrt(dx ** 2 + dy ** 2)
                    if length == 0:
                        return mid_x, mid_y

                    # Vector perpendicular normalizado (para la curvatura)
                    perpendicular_dx, perpendicular_dy = -dy / length, dx / length

                    # Calcular punto en la curva (punto medio de la curva)
                    curve_offset = rad * length * 0.5  # Punto medio de la curva
                    curve_mid_x = mid_x + perpendicular_dx * curve_offset
                    curve_mid_y = mid_y + perpendicular_dy * curve_offset

                    # Ahora calcular la posición "arriba" de este punto en la curva
                    # Vector tangente en el punto medio de la curva
                    if abs(rad) > 0.01:  # Si hay curvatura significativa
                        # Para curvas, el "arriba" es perpendicular a la tangente
                        tangent_dx = dx + perpendicular_dx * rad * length
                        tangent_dy = dy + perpendicular_dy * rad * length
                        tangent_length = math.sqrt(tangent_dx ** 2 + tangent_dy ** 2)
                        if tangent_length > 0:
                            # Vector perpendicular a la tangente (hacia "arriba")
                            up_dx = -tangent_dy / tangent_length
                            up_dy = tangent_dx / tangent_length
                        else:
                            up_dx, up_dy = perpendicular_dx, perpendicular_dy
                    else:
                        # Para líneas rectas, usar el vector perpendicular
                        up_dx, up_dy = perpendicular_dx, perpendicular_dy

                    # Offset hacia arriba de la flecha
                    label_offset = 0.15  # Distancia fija arriba de la flecha
                    final_x = curve_mid_x + up_dx * label_offset
                    final_y = curve_mid_y + up_dy * label_offset

                    return final_x, final_y

            # Dibujar cada transición como una flecha individual
            transition_counter = defaultdict(int)  # Contador para separar transiciones paralelas

            for t in transition_data:
                origen = t['origen']
                destino = t['destino']
                simbolo = t['simbolo']

                is_selfloop = origen == destino

                # Contar cuántas transiciones hay entre el mismo par de estados
                pair_key = (origen, destino)
                transition_counter[pair_key] += 1
                current_index = transition_counter[pair_key] - 1

                # Calcular el radio para esta transición específica
                if is_selfloop:
                    # Para bucles, cada uno en una posición angular diferente
                    base_rad = 0.8
                    rad = base_rad + (current_index * 0.3)
                    loop_angle = (np.pi / 2) + (current_index * np.pi / 3)
                else:
                    # Para transiciones normales, usar diferentes radios
                    base_rad = 0.15
                    # Alternar entre curvas positivas y negativas
                    if current_index % 2 == 0:
                        rad = base_rad + (current_index * 0.1)
                    else:
                        rad = -(base_rad + ((current_index - 1) * 0.1))

                # Dibujar la flecha individual
                edge = [(origen, destino)]
                nx.draw_networkx_edges(G, pos,
                                       edgelist=edge,
                                       arrowstyle='-|>',
                                       arrowsize=30,
                                       connectionstyle=f'arc3,rad={rad}',
                                       width=3,
                                       edge_color='#37474F',
                                       node_size=node_size,
                                       alpha=0.8)

                # Posicionar etiqueta para esta transición específica
                x1, y1 = pos[origen]
                x2, y2 = pos[destino]

                if is_selfloop:
                    label_x, label_y = get_label_position(x1, y1, x2, y2, rad, True, loop_angle)
                else:
                    label_x, label_y = get_label_position(x1, y1, x2, y2, rad)

                # Dibujar etiqueta individual para cada transición
                ax.text(label_x, label_y, simbolo,
                        fontsize=11,
                        color='#1A237E',
                        weight='bold',
                        ha='center',
                        va='center',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='#E8EAF6',
                                  edgecolor='#3F51B5',
                                  linewidth=1.5,
                                  alpha=0.95))

            # Dibujar etiquetas de nodos con mejor estilo
            labels = {}
            for node in G.nodes():
                labels[node] = node

            nx.draw_networkx_labels(G, pos, labels, font_size=13, font_weight='bold', font_color='#212121')

            # Dibujar círculos dobles para estados de aceptación (estilo JFLAP)
            for state in aceptados:
                if state in pos:
                    x, y = pos[state]
                    # Radio del círculo exterior (ligeramente más grande que el nodo)
                    outer_radius = 0.28
                    inner_radius = 0.22

                    # Círculo exterior
                    circle_outer = plt.Circle((x, y), outer_radius,
                                              fill=False,
                                              edgecolor='#E65100' if state not in [estado_inicial] else '#2E7D32',
                                              linestyle='solid',
                                              linewidth=3)

                    # Círculo interior
                    circle_inner = plt.Circle((x, y), inner_radius,
                                              fill=False,
                                              edgecolor='#E65100' if state not in [estado_inicial] else '#2E7D32',
                                              linestyle='solid',
                                              linewidth=2)

                    ax.add_patch(circle_outer)
                    ax.add_patch(circle_inner)

            # Dibujar flecha inicial más elegante
            if estado_inicial in pos:
                x, y = pos[estado_inicial]
                # Flecha más grande y con mejor estilo
                arrow_start_x = x - 0.9
                arrow_length = 0.6

                ax.annotate('', xy=(x - 0.3, y), xytext=(arrow_start_x, y),
                            arrowprops=dict(arrowstyle='-|>',
                                            color='#1565C0',
                                            lw=4,
                                            mutation_scale=25))

                # Etiqueta "Inicio"
                ax.text(arrow_start_x - 0.1, y + 0.15, 'Inicio',
                        fontsize=10,
                        color='#1565C0',
                        weight='bold',
                        ha='center',
                        va='bottom')

            # Leyenda mejorada
            legend_elements = [
                mpatches.Patch(color='#2196F3', label='Estado Inicial'),
                mpatches.Patch(color='#FF9800', label='Estado de Aceptación'),
                mpatches.Patch(color='#4CAF50', label='Inicial y Aceptación'),
                mpatches.Patch(color='#FFFFFF', label='Estado Normal', edgecolor='#424242')
            ]

            legend = ax.legend(handles=legend_elements,
                               loc='upper center',
                               bbox_to_anchor=(0.5, 1.08),
                               ncol=4,
                               frameon=True,
                               fontsize=11,
                               fancybox=True,
                               shadow=True,
                               borderpad=1)

            legend.get_frame().set_facecolor('#f8f9fa')
            legend.get_frame().set_edgecolor('#dee2e6')

            # Configurar ejes
            ax.set_aspect('equal')
            ax.axis('off')

            # Calcular límites con margen apropiado
            if pos:
                x_values = [x for x, y in pos.values()]
                y_values = [y for x, y in pos.values()]
                margin = 1.2
                ax.set_xlim(min(x_values) - margin, max(x_values) + margin)
                ax.set_ylim(min(y_values) - margin, max(y_values) + margin)

            # Añadir título
            ax.set_title('Diagrama del Autómata Finito',
                         fontsize=16,
                         fontweight='bold',
                         color='#212121',
                         pad=20)

            # Ajustar layout
            plt.tight_layout()

            # Guardar la imagen con mayor calidad
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer,
                        format='png',
                        bbox_inches='tight',
                        dpi=300,
                        facecolor='#f8f9fa',
                        edgecolor='none',
                        transparent=False)
            img_buffer.seek(0)
            plt.close()

            img_data = img_buffer.getvalue()
            img_base64 = base64.b64encode(img_data).decode('utf-8')

            # Mostrar la imagen en el contenedor existente
            imagen = ft.Image(
                src_base64=img_base64,
                width=700,
                height=550,
                fit=ft.ImageFit.CONTAIN
            )
            imagen_automata_container.content = imagen
            imagen_automata_container.visible = True

            # Actualizar el mensaje de resultado
            resultado_text.value = "Visualización generada correctamente con estilo profesional."
            resultado_text.color = ft.Colors.GREEN_600

            page.update()

        except Exception as ex:
            resultado_text.value = f"Error al visualizar el autómata: {str(ex)}"
            resultado_text.color = ft.Colors.RED_600
            imagen_automata_container.content = None
            imagen_automata_container.visible = False
            page.update()

    def create_regex_to_automata_view(page: ft.Page, main_content_column: ft.Column):
        """
        Crea la vista para convertir una expresión regular en un autómata finito determinista mínimo.
        """
        # Limpiar el contenido anterior
        main_content_column.controls.clear()

        # Definir componentes de la interfaz
        regex_input = ft.TextField(
            label="Expresión Regular",
            hint_text="Ej: (a|b)*abb",
            width=500,
            border_radius=10,
            border_color=ft.Colors.with_opacity(0.5, ft.Colors.BLACK),
            focused_border_color=ft.Colors.BLUE_600,
            text_size=16
        )

        resultado_text = ft.Text(
            value="Ingrese una expresión regular y presione Enter.",
            color=ft.Colors.with_opacity(0.7, ft.Colors.BLACK12),
            size=16,
            weight=ft.FontWeight.W_500,
            text_align=ft.TextAlign.CENTER
        )

        image_container = ft.Container(
            content=ft.Text("El autómata aparecerá aquí", italic=True, color=ft.Colors.BLACK54),
            alignment=ft.alignment.center,
            padding=10,
            border_radius=10,
            border=ft.border.all(1, ft.Colors.BLACK26),
            width=650,
            height=480,
            bgcolor=ft.Colors.WHITE,
            visible=False
        )

        progress_ring = ft.ProgressRing(visible=False, width=32, height=32, stroke_width=3)

        def on_generate(e):
            image_container.visible = False
            image_container.content = ft.Text("")
            resultado_text.value = "Procesando..."
            resultado_text.color = ft.Colors.BLUE_GREY_500
            progress_ring.visible = True
            page.update()

            regex = regex_input.value

            try:
                tokens = parse_regex(regex)
                nfa_states, nfa_transitions, nfa_initial, nfa_final = build_nfa(tokens)
                dfa_states, dfa_transitions, dfa_initial, dfa_accepting, alphabet = nfa_to_dfa(
                    nfa_states, nfa_transitions, nfa_initial, nfa_final
                )
                min_states, min_transitions, min_initial, min_accepting = minimize_dfa(
                    dfa_states, dfa_transitions, dfa_initial, dfa_accepting, alphabet
                )
                img_data = draw_automata(min_states, min_transitions, min_initial, min_accepting, alphabet)
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                imagen = ft.Image(
                    src_base64=img_base64,
                    fit=ft.ImageFit.CONTAIN,
                    width=image_container.width - 20,
                    height=image_container.height - 20
                )
                image_container.content = imagen
                image_container.visible = True
                resultado_text.value = "¡Autómata Mínimo Generado!"
                resultado_text.color = ft.Colors.GREEN_700
            except ValueError as ve:
                resultado_text.value = f"Error: {str(ve)}"
                resultado_text.color = ft.Colors.RED_700
                image_container.content = ft.Text("No se pudo generar el autómata.", color=ft.Colors.RED_700)
                image_container.visible = True
            except Exception as ex:
                resultado_text.value = f"Error inesperado: {str(ex)}"
                resultado_text.color = ft.Colors.RED_700
                image_container.content = ft.Text(f"Detalle: {type(ex).__name__}", color=ft.Colors.RED_700)
                image_container.visible = True
                print(f"Error Detallado: {ex}")
                import traceback
                traceback.print_exc()
            finally:
                progress_ring.visible = False
                page.update()

        regex_input.on_submit = on_generate

        main_content_column.controls = [
            ft.Text("Operaciones sobre Lenguajes y Autómatas", style=ft.TextThemeStyle.HEADLINE_MEDIUM,
                    color=primary_color),
            ft.Text("Seleccione una operación del menú lateral.", style=ft.TextThemeStyle.BODY_MEDIUM),
            ft.Text(
                "Convertidor de Expresión Regular a Autómata Finito Determinista Mínimo",
                size=20, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER
            ),
            ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
            ft.Row([regex_input], alignment=ft.MainAxisAlignment.CENTER),
            ft.Row(
                [resultado_text, progress_ring],
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10
            ),
            ft.Divider(height=10, color=ft.Colors.BLACK12),
            image_container
        ]

        page.update()

    # --- Configuración de la Interfaz Principal ---
    layout_principal = ft.Column(
        [
            ft.Text("Operaciones sobre Lenguajes y Autómatas", style=ft.TextThemeStyle.HEADLINE_MEDIUM,
                    color=primary_color),
            ft.Text("Seleccione una operación del menú lateral.", style=ft.TextThemeStyle.BODY_MEDIUM),
            # Los inputs y botones se añadirán dinámicamente aquí por show_inputs
            resultado_container,  # El contenedor de resultados va al final
        ],
        spacing=15,  # Espaciado ajustado
        expand=True,
        scroll=ft.ScrollMode.ADAPTIVE,  # Permitir scroll si el contenido excede
        # alignment=ft.MainAxisAlignment.START, # Alinear al inicio
    )

    rail = ft.NavigationRail(
        selected_index=None,  # Empezar sin selección
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=150,
        min_extended_width=250,
        group_alignment=-0.9,  # Alinear arriba
        bgcolor=ft.Colors.BLUE_GREY_100,  # Color de fondo más suave
        indicator_color=ft.Colors.BLUE_100,
        destinations=[
            ft.NavigationRailDestination(icon=ft.Icons.STAR_BORDER, selected_icon=ft.Icons.STAR, label="Kleene (*)"),
            ft.NavigationRailDestination(icon=ft.Icons.ADD_CIRCLE_OUTLINE, selected_icon=ft.Icons.ADD_CIRCLE,
                                         label="Positiva (+)"),
            ft.NavigationRailDestination(icon=ft.Icons.LINK, selected_icon=ft.Icons.LINK, label="Concatenar (.)"),
            ft.NavigationRailDestination(icon=ft.Icons.EXPOSURE_PLUS_1, selected_icon=ft.Icons.EXPOSURE_PLUS_1,
                                         label="Potencia (L^n)"),
            ft.NavigationRailDestination(icon=ft.Icons.FLIP_CAMERA_ANDROID, selected_icon=ft.Icons.FLIP_CAMERA_ANDROID,
                                         label="Reflexión (L^R)"),
            ft.NavigationRailDestination(icon=ft.Icons.MERGE_OUTLINED, selected_icon=ft.Icons.MERGE, label="Unión (∪)"),
            ft.NavigationRailDestination(icon=ft.Icons.LOCK_OUTLINE, selected_icon=ft.Icons.LOCK,
                                         label="Intersección (∩)"),
            ft.NavigationRailDestination(icon=ft.Icons.COMPARE_ARROWS_OUTLINED, selected_icon=ft.Icons.COMPARE_ARROWS,
                                         label="Diferencia (-)"),
            ft.NavigationRailDestination(icon=ft.Icons.SCHEMA, selected_icon=ft.Icons.SCHEMA, label="Autómata"),
            ft.NavigationRailDestination(icon=ft.Icons.FUNCTIONS, selected_icon=ft.Icons.FUNCTIONS,
                                         label="Regex a Autómata"),
        ],
        on_change=lambda e: {
            0: lambda: realizar_cerradura_kleene(e, layout_principal),
            1: lambda: realizar_clausura_positiva(e, layout_principal),
            2: lambda: realizar_concatenacion(e, layout_principal),
            3: lambda: realizar_potenciacion(e, layout_principal),
            4: lambda: realizar_reflexion(e, layout_principal),
            5: lambda: realizar_union(e, layout_principal),
            6: lambda: realizar_interseccion(e, layout_principal),
            7: lambda: realizar_diferencia(e, layout_principal),
            8: lambda: realizar_automata(e, layout_principal),
            9: lambda: create_regex_to_automata_view(page, layout_principal),
        }.get(e.control.selected_index, lambda: None)()
    )

    # CORRECCIÓN: Envolver el NavigationRail en un Container con altura fija
    # Layout final de la aplicación
    app_layout = ft.Row(
        [
            # CAMBIO PRINCIPAL: Envolver el rail en un Container con altura definida
            ft.Container(
                content=rail,
                height=650,  # Altura fija para evitar el error "unbounded height"
                width=220,  # Ancho fijo que acomode las etiquetas largas
            ),
            ft.VerticalDivider(width=1),
            ft.Container(  # Contenedor para el layout principal con padding
                content=layout_principal,
                padding=ft.padding.all(15),
                expand=True,
                alignment=ft.alignment.top_left
            )
        ],
        expand=True,
        vertical_alignment=ft.CrossAxisAlignment.START
    )
    return app_layout


# --- Punto de Entrada Principal (si ejecutas este archivo directamente) ---
if __name__ == "__main__":
    def main(page: ft.Page):
        page.add(create_automata_view(page))
        page.update()


    ft.app(target=main)