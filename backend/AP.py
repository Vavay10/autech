# -*- coding: utf-8 -*-
import flet as ft
import flet.canvas as cv
import re
import itertools
import math
from typing import Dict, Set, Tuple, List, Optional, Any
from collections import deque

# --- Constantes ---
EPSILON = 'ε'


# --- Constantes de Visualización ---
class VisualConstants:
    # Tamaños
    STATE_RADIUS = 28
    ACCEPT_RADIUS_OUTER = 32
    ARROW_SIZE = 10

    # Espaciado
    PADDING = 50
    NODE_SPACING_H = 140
    NODE_SPACING_V = 120
    SELF_LOOP_HEIGHT = 60
    CURVE_OFFSET = 45

    # Fuentes
    NODE_FONT_SIZE = 13
    EDGE_FONT_SIZE = 11
    LABEL_PADDING = 8

    # Colores (más profesionales)
    NORMAL_COLOR = ft.Colors.with_opacity(0.9, ft.Colors.BLUE_50)
    INITIAL_COLOR = ft.Colors.with_opacity(0.9, ft.Colors.GREEN_100)
    ACCEPT_COLOR = ft.Colors.with_opacity(0.9, ft.Colors.AMBER_50)

    STATE_BORDER_COLOR = ft.Colors.BLUE_GREY_700
    ACCEPT_BORDER_COLOR = ft.Colors.BLUE_GREY_800
    INITIAL_ARROW_COLOR = ft.Colors.GREEN_700

    TEXT_COLOR = ft.Colors.GREY_900
    EDGE_COLOR = ft.Colors.BLUE_GREY_600
    EDGE_LABEL_BG_COLOR = ft.Colors.with_opacity(0.95, ft.Colors.WHITE)
    SHADOW_COLOR = ft.Colors.with_opacity(0.15, ft.Colors.BLACK)


# --- Clase PDA Mejorada ---
class PDA:
    """Representa la definición de un Autómata de Pila con validaciones mejoradas."""

    def __init__(self):
        self.states: Set[str] = set()
        self.input_alphabet: Set[str] = set()
        self.stack_alphabet: Set[str] = set()
        self.transitions: Dict[Tuple[str, str, str], Set[Tuple[str, Tuple[str, ...]]]] = {}
        self.start_state: Optional[str] = None
        self.start_symbol: Optional[str] = None
        self.accept_states: Set[str] = set()

    def clear(self):
        """Reinicializa el objeto PDA a un estado vacío."""
        self.__init__()

    def parse_from_ui(self, states_str: str, input_alpha_str: str, stack_alpha_str: str,
                      start_state_str: str, start_symbol_str: str, accept_states_str: str,
                      transitions_str: str):
        """
        Popula el objeto PDA desde los campos de texto de la UI con validaciones mejoradas.
        """
        self.clear()
        errors = []

        # --- Parseo y limpieza de datos ---
        def clean_split(text: str) -> Set[str]:
            return set(s.strip() for s in text.split(',') if s.strip())

        self.states = clean_split(states_str)
        self.input_alphabet = clean_split(input_alpha_str)
        self.input_alphabet.discard(EPSILON)
        self.stack_alphabet = clean_split(stack_alpha_str)
        self.start_state = start_state_str.strip() if start_state_str else None
        self.start_symbol = start_symbol_str.strip() if start_symbol_str else None
        self.accept_states = clean_split(accept_states_str)

        # --- Validaciones básicas mejoradas ---
        if not self.states:
            errors.append("❌ El conjunto de estados (Q) no puede estar vacío.")
        if not self.stack_alphabet:
            errors.append("❌ El alfabeto de pila (Γ) no puede estar vacío.")
        if not self.start_state:
            errors.append("❌ Debe definir un estado inicial (q₀).")
        if not self.start_symbol:
            errors.append("❌ Debe definir un símbolo inicial de pila (Z₀).")

        if self.start_state and self.start_state not in self.states:
            errors.append(f"❌ El estado inicial '{self.start_state}' no está en Q.")

        # Auto-agregar símbolo inicial si no está en el alfabeto
        if self.start_symbol and self.start_symbol not in self.stack_alphabet:
            self.stack_alphabet.add(self.start_symbol)

        invalid_accept = self.accept_states - self.states
        if invalid_accept:
            errors.append(f"❌ Estados de aceptación inválidos: {invalid_accept}")

        # --- Parseo de transiciones mejorado ---
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
                errors.append(f"❌ Línea {i}: Formato inválido. Use: 'origen,entrada,pop -> destino,push'")
                continue

            q_read, input_sym, stack_pop, q_write, stack_push_str = [
                m.strip() for m in match.groups()
            ]

            # Validaciones de estados y símbolos
            if q_read not in self.states:
                errors.append(f"❌ Línea {i}: Estado '{q_read}' no está en Q.")
            if q_write not in self.states:
                errors.append(f"❌ Línea {i}: Estado '{q_write}' no está en Q.")
            if input_sym != EPSILON and input_sym not in self.input_alphabet:
                errors.append(f"❌ Línea {i}: Símbolo '{input_sym}' no está en Σ.")
            if stack_pop != EPSILON and stack_pop not in self.stack_alphabet:
                errors.append(f"❌ Línea {i}: Símbolo '{stack_pop}' no está en Γ.")

            # Parseo de símbolos de pila para push
            stack_push_tuple = self._parse_stack_push(stack_push_str, i, errors)
            if stack_push_tuple is None:
                continue

            # Almacenar transición
            key = (q_read, input_sym, stack_pop)
            value = (q_write, stack_push_tuple)
            if key not in processed_transitions:
                processed_transitions[key] = set()
            processed_transitions[key].add(value)

        if errors:
            raise ValueError("\n".join(errors))

        self.transitions = processed_transitions

    def _parse_stack_push(self, stack_push_str: str, line_num: int, errors: List[str]) -> Optional[Tuple[str, ...]]:
        """Parsea la cadena de símbolos a apilar."""
        if stack_push_str == EPSILON:
            return tuple()

        # Ordenar símbolos por longitud (más largo primero) para evitar ambigüedades
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
                errors.append(f"❌ Línea {line_num}: Símbolo inválido en '{stack_push_str}'")
                return None

        return tuple(result)

    def __str__(self) -> str:
        """Representación mejorada del PDA."""
        trans_lines = []
        for (q, a, z), results in sorted(self.transitions.items()):
            for (p, gamma) in sorted(results):
                gamma_str = "".join(gamma) if gamma else EPSILON
                trans_lines.append(f"  δ({q}, {a}, {z}) → ({p}, {gamma_str})")

        return f"""PDA Definition:
━━━━━━━━━━━━━━
Q = {{{', '.join(sorted(self.states))}}}
Σ = {{{', '.join(sorted(self.input_alphabet))}}}
Γ = {{{', '.join(sorted(self.stack_alphabet))}}}
q₀ = {self.start_state}
Z₀ = {self.start_symbol}
F = {{{', '.join(sorted(self.accept_states))}}}

Transitions:
{chr(10).join(trans_lines) if trans_lines else '  (None)'}"""


# --- Funciones de Visualización Mejoradas ---

class PDAVisualizer:
    """Clase para manejar la visualización profesional del PDA."""

    @staticmethod
    def get_text_dimensions(text: str, font_size: int) -> Tuple[float, float]:
        """Calcula las dimensiones aproximadas del texto."""
        lines = text.split('\n')
        max_width = max(len(line) for line in lines) if lines else 0
        width = max_width * font_size * 0.65
        height = len(lines) * font_size * 1.3
        return width, height

    @staticmethod
    def create_shadow_circle(x: float, y: float, radius: float, offset: float = 2) -> cv.Circle:
        """Crea un círculo de sombra."""
        return cv.Circle(
            x + offset, y + offset, radius,
            ft.Paint(color=VisualConstants.SHADOW_COLOR, style=ft.PaintingStyle.FILL)
        )

    @staticmethod
    def draw_curved_arrow(canvas: cv.Canvas, start_x: float, start_y: float,
                          end_x: float, end_y: float, control_x: float, control_y: float,
                          paint: ft.Paint):
        """Dibuja una flecha curvada con punta."""
        # Dibujar curva
        path = cv.Path([
            cv.Path.MoveTo(start_x, start_y),
            cv.Path.QuadraticTo(control_x, control_y, end_x, end_y),
        ], paint=paint)
        canvas.shapes.append(path)

        # Calcular ángulo para la punta de flecha
        angle = math.atan2(end_y - control_y, end_x - control_x)
        arrow_angle = math.radians(25)

        # Puntos de la punta de flecha
        p1_x = end_x - VisualConstants.ARROW_SIZE * math.cos(angle - arrow_angle)
        p1_y = end_y - VisualConstants.ARROW_SIZE * math.sin(angle - arrow_angle)
        p2_x = end_x - VisualConstants.ARROW_SIZE * math.cos(angle + arrow_angle)
        p2_y = end_y - VisualConstants.ARROW_SIZE * math.sin(angle + arrow_angle)

        canvas.shapes.extend([
            cv.Line(end_x, end_y, p1_x, p1_y, paint=paint),
            cv.Line(end_x, end_y, p2_x, p2_y, paint=paint)
        ])

    @classmethod
    def calculate_optimal_layout(cls, states: Set[str], start_state: str) -> Dict[str, Tuple[float, float]]:
        """Calcula un layout optimizado para los estados."""
        positions = {}
        state_list = [start_state] if start_state else []
        state_list.extend([s for s in sorted(states) if s != start_state])

        num_states = len(state_list)

        if not state_list:
            return {}
        if num_states == 1:
            positions[state_list[0]] = (200, 200)
        elif num_states <= 6:
            # Layout horizontal mejorado
            base_x = VisualConstants.PADDING + VisualConstants.ACCEPT_RADIUS_OUTER + 60
            base_y = VisualConstants.PADDING + VisualConstants.ACCEPT_RADIUS_OUTER + VisualConstants.SELF_LOOP_HEIGHT + 40

            for i, state in enumerate(state_list):
                if i < 4:  # Primera fila
                    x = base_x + i * VisualConstants.NODE_SPACING_H
                    y = base_y
                else:  # Segunda fila
                    x = base_x + (i - 4) * VisualConstants.NODE_SPACING_H + VisualConstants.NODE_SPACING_H // 2
                    y = base_y + VisualConstants.NODE_SPACING_V
                positions[state] = (x, y)
        else:
            # Layout circular para muchos estados
            center_x, center_y = 400, 300
            radius = max(120, num_states * 25)

            for i, state in enumerate(state_list):
                angle = 2 * math.pi * i / num_states
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                positions[state] = (x, y)

        return positions

    @classmethod
    def draw_pda(cls, pda: PDA, canvas: cv.Canvas):
        """Dibuja el PDA con estilo profesional mejorado."""
        canvas.shapes.clear()

        if not pda.states:
            cls._draw_empty_state(canvas)
            canvas.update()
            return

        # Calcular posiciones optimizadas
        positions = cls.calculate_optimal_layout(pda.states, pda.start_state)

        # Calcular dimensiones del canvas
        canvas_width, canvas_height = cls._calculate_canvas_size(positions)
        canvas.width = canvas_width
        canvas.height = canvas_height

        # Agrupar transiciones
        grouped_edges = cls._group_transitions(pda.transitions, positions)

        # Dibujar en capas para mejor apariencia
        cls._draw_shadows(canvas, positions, pda.accept_states)
        cls._draw_edges(canvas, grouped_edges, positions, pda.accept_states)
        cls._draw_states(canvas, positions, pda.start_state, pda.accept_states)
        cls._draw_initial_arrow(canvas, positions, pda.start_state)

        canvas.update()

    @staticmethod
    def _draw_empty_state(canvas: cv.Canvas):
        """Dibuja el estado vacío."""
        canvas.width = 400
        canvas.height = 100
        canvas.shapes.append(
            cv.Text(
                x=200, y=50,
                text="📝 Defina un autómata para visualizar",
                text_align=ft.TextAlign.CENTER,
                alignment=ft.alignment.center,
                style=ft.TextStyle(size=16, color=ft.Colors.GREY_600)
            )
        )

    @staticmethod
    def _calculate_canvas_size(positions: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
        """Calcula el tamaño óptimo del canvas."""
        if not positions:
            return 400, 200

        max_x = max(pos[0] for pos in positions.values())
        max_y = max(pos[1] for pos in positions.values())

        return (
            max(800, max_x + VisualConstants.PADDING + 100),
            max(500, max_y + VisualConstants.PADDING + 100)
        )

    @classmethod
    def _group_transitions(cls, transitions: Dict, positions: Dict) -> Dict:
        """Agrupa las transiciones por pares de estados."""
        grouped = {}

        for (q_read, input_sym, stack_pop), results in transitions.items():
            if q_read not in positions:
                continue

            for (q_write, stack_push_tuple) in results:
                if q_write not in positions:
                    continue

                edge_key = tuple(sorted((q_read, q_write))) + ((q_read, q_write),)
                push_str = "".join(stack_push_tuple) if stack_push_tuple else EPSILON
                label = f"{input_sym}, {stack_pop} / {push_str}"

                if edge_key not in grouped:
                    grouped[edge_key] = []
                grouped[edge_key].append(label)

        return grouped

    @classmethod
    def _draw_shadows(cls, canvas: cv.Canvas, positions: Dict, accept_states: Set[str]):
        """Dibuja sombras para los estados."""
        for state, (x, y) in positions.items():
            radius = VisualConstants.ACCEPT_RADIUS_OUTER if state in accept_states else VisualConstants.STATE_RADIUS
            canvas.shapes.append(cls.create_shadow_circle(x, y, radius))

    @classmethod
    def _draw_edges(cls, canvas: cv.Canvas, grouped_edges: Dict, positions: Dict, accept_states: Set[str]):
        """Dibuja todas las aristas con etiquetas."""
        edge_paint = ft.Paint(
            stroke_width=2,
            color=VisualConstants.EDGE_COLOR,
            style=ft.PaintingStyle.STROKE,
            stroke_cap=ft.StrokeCap.ROUND
        )

        processed_pairs = set()

        for edge_key, labels in grouped_edges.items():
            q_read, q_write = edge_key[2]

            x1, y1 = positions[q_read]
            x2, y2 = positions[q_write]
            full_label = "\n".join(labels)

            if q_read == q_write:
                cls._draw_self_loop(canvas, x1, y1, full_label, edge_paint)
            else:
                pair_key = tuple(sorted((q_read, q_write)))
                needs_curve = (q_write, q_read) in [k[2] for k in grouped_edges.keys()]

                cls._draw_transition_edge(
                    canvas, x1, y1, x2, y2, full_label, edge_paint,
                    q_read in accept_states, q_write in accept_states,
                    needs_curve, 1
                )

                processed_pairs.add(pair_key)

    @classmethod
    def _draw_self_loop(cls, canvas: cv.Canvas, x: float, y: float, label: str, paint: ft.Paint):
        """Dibuja un bucle propio."""
        radius = VisualConstants.STATE_RADIUS
        loop_height = VisualConstants.SELF_LOOP_HEIGHT

        angle_offset = math.pi / 6
        start_angle = -math.pi / 2 - angle_offset
        end_angle = -math.pi / 2 + angle_offset

        start_x = x + radius * math.cos(start_angle)
        start_y = y + radius * math.sin(start_angle)
        end_x = x + radius * math.cos(end_angle)
        end_y = y + radius * math.sin(end_angle)

        control_x = x
        control_y = y - radius - loop_height

        cls.draw_curved_arrow(canvas, start_x, start_y, end_x, end_y, control_x, control_y, paint)
        cls._draw_edge_label(canvas, control_x, control_y - 15, label)

    @classmethod
    def _draw_transition_edge(cls, canvas: cv.Canvas, x1: float, y1: float, x2: float, y2: float,
                              label: str, paint: ft.Paint, start_is_accept: bool, end_is_accept: bool,
                              needs_curve: bool, curve_direction: int):
        """Dibuja una arista de transición."""
        start_radius = VisualConstants.ACCEPT_RADIUS_OUTER if start_is_accept else VisualConstants.STATE_RADIUS
        end_radius = VisualConstants.ACCEPT_RADIUS_OUTER if end_is_accept else VisualConstants.STATE_RADIUS

        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)

        if dist < 1e-6:
            return

        if needs_curve:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            nx, ny = -dy / dist, dx / dist

            control_x = mid_x + nx * VisualConstants.CURVE_OFFSET * curve_direction
            control_y = mid_y + ny * VisualConstants.CURVE_OFFSET * curve_direction

            angle1 = math.atan2(control_y - y1, control_x - x1)
            angle2 = math.atan2(y2 - control_y, x2 - control_x)  # Corrected angle for end point

            start_x = x1 + start_radius * math.cos(angle1)
            start_y = y1 + start_radius * math.sin(angle1)
            end_x = x2 - end_radius * math.cos(angle2)
            end_y = y2 - end_radius * math.sin(angle2)

            cls.draw_curved_arrow(canvas, start_x, start_y, end_x, end_y, control_x, control_y, paint)
            cls._draw_edge_label(canvas, control_x, control_y, label)
        else:
            start_x = x1 + start_radius * dx / dist
            start_y = y1 + start_radius * dy / dist
            end_x = x2 - end_radius * dx / dist
            end_y = y2 - end_radius * dy / dist

            canvas.shapes.append(cv.Line(start_x, start_y, end_x, end_y, paint=paint))

            angle = math.atan2(end_y - start_y, end_x - start_x)
            arrow_angle = math.radians(25)

            p1_x = end_x - VisualConstants.ARROW_SIZE * math.cos(angle - arrow_angle)
            p1_y = end_y - VisualConstants.ARROW_SIZE * math.sin(angle - arrow_angle)
            p2_x = end_x - VisualConstants.ARROW_SIZE * math.cos(angle + arrow_angle)
            p2_y = end_y - VisualConstants.ARROW_SIZE * math.sin(angle + arrow_angle)

            canvas.shapes.extend([
                cv.Line(end_x, end_y, p1_x, p1_y, paint=paint),
                cv.Line(end_x, end_y, p2_x, p2_y, paint=paint)
            ])

            mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
            cls._draw_edge_label(canvas, mid_x, mid_y, label)

    @staticmethod
    def _draw_edge_label(canvas: cv.Canvas, x: float, y: float, label: str):
        """Dibuja una etiqueta de arista con fondo."""
        label_width, label_height = PDAVisualizer.get_text_dimensions(label, VisualConstants.EDGE_FONT_SIZE)

        bg_rect = cv.Rect(
            x=x - label_width / 2 - VisualConstants.LABEL_PADDING,
            y=y - label_height / 2 - VisualConstants.LABEL_PADDING,
            width=label_width + 2 * VisualConstants.LABEL_PADDING,
            height=label_height + 2 * VisualConstants.LABEL_PADDING,
            paint=ft.Paint(color=VisualConstants.EDGE_LABEL_BG_COLOR, style=ft.PaintingStyle.FILL),
            border_radius=4
        )

        border_rect = cv.Rect(
            x=x - label_width / 2 - VisualConstants.LABEL_PADDING,
            y=y - label_height / 2 - VisualConstants.LABEL_PADDING,
            width=label_width + 2 * VisualConstants.LABEL_PADDING,
            height=label_height + 2 * VisualConstants.LABEL_PADDING,
            paint=ft.Paint(color=ft.Colors.GREY_300, style=ft.PaintingStyle.STROKE, stroke_width=1),
            border_radius=4
        )

        canvas.shapes.extend([bg_rect, border_rect])

        canvas.shapes.append(
            cv.Text(
                x=x, y=y, text=label,
                text_align=ft.TextAlign.CENTER,
                alignment=ft.alignment.center,
                style=ft.TextStyle(
                    size=VisualConstants.EDGE_FONT_SIZE,
                    color=VisualConstants.TEXT_COLOR,
                    weight=ft.FontWeight.W_500
                )
            )
        )

    @classmethod
    def _draw_states(cls, canvas: cv.Canvas, positions: Dict, start_state: str, accept_states: Set[str]):
        """Dibuja todos los estados."""
        for state, (x, y) in positions.items():
            is_initial = state == start_state
            is_accept = state in accept_states

            if is_initial and is_accept:
                fill_color = ft.Colors.with_opacity(0.9, ft.Colors.LIGHT_GREEN_200)
            elif is_initial:
                fill_color = VisualConstants.INITIAL_COLOR
            elif is_accept:
                fill_color = VisualConstants.ACCEPT_COLOR
            else:
                fill_color = VisualConstants.NORMAL_COLOR

            main_circle = cv.Circle(
                x=x, y=y, radius=VisualConstants.STATE_RADIUS,
                paint=ft.Paint(color=fill_color, style=ft.PaintingStyle.FILL)
            )

            main_border = cv.Circle(
                x=x, y=y, radius=VisualConstants.STATE_RADIUS,
                paint=ft.Paint(
                    color=VisualConstants.STATE_BORDER_COLOR,
                    style=ft.PaintingStyle.STROKE,
                    stroke_width=2
                )
            )

            canvas.shapes.extend([main_circle, main_border])

            if is_accept:
                accept_border = cv.Circle(
                    x=x, y=y, radius=VisualConstants.ACCEPT_RADIUS_OUTER,
                    paint=ft.Paint(
                        color=VisualConstants.ACCEPT_BORDER_COLOR,
                        style=ft.PaintingStyle.STROKE,
                        stroke_width=2.5
                    )
                )
                canvas.shapes.append(accept_border)

            canvas.shapes.append(
                cv.Text(
                    x=x, y=y, text=state,
                    text_align=ft.TextAlign.CENTER,
                    alignment=ft.alignment.center,
                    style=ft.TextStyle(
                        size=VisualConstants.NODE_FONT_SIZE,
                        color=VisualConstants.TEXT_COLOR,
                        weight=ft.FontWeight.BOLD
                    )
                )
            )

    @staticmethod
    def _draw_initial_arrow(canvas: cv.Canvas, positions: Dict, start_state: str):
        """Dibuja la flecha que indica el estado inicial."""
        if not start_state or start_state not in positions:
            return

        x, y = positions[start_state]
        arrow_length = 55
        arrow_start_x = x - VisualConstants.STATE_RADIUS - arrow_length
        arrow_end_x = x - VisualConstants.STATE_RADIUS - 8

        arrow_paint = ft.Paint(
            stroke_width=3,
            color=VisualConstants.INITIAL_ARROW_COLOR,
            stroke_cap=ft.StrokeCap.ROUND
        )

        canvas.shapes.append(cv.Line(arrow_start_x, y, arrow_end_x, y, paint=arrow_paint))

        arrow_tip_size = 12
        angle = math.radians(25)

        p1_x = arrow_end_x - arrow_tip_size * math.cos(angle)
        p1_y = y - arrow_tip_size * math.sin(angle)
        p2_x = arrow_end_x - arrow_tip_size * math.cos(-angle)
        p2_y = y - arrow_tip_size * math.sin(-angle)

        canvas.shapes.extend([
            cv.Line(arrow_end_x, y, p1_x, p1_y, paint=arrow_paint),
            cv.Line(arrow_end_x, y, p2_x, p2_y, paint=arrow_paint)
        ])


# --- Conversión a CFG (mejorada) ---
def convert_pda_to_cfg(pda: PDA) -> str:
    """Convierte un PDA a CFG con mejor formato y validaciones."""
    if not pda.states or not pda.start_state or not pda.start_symbol:
        raise ValueError("PDA incompleto para conversión")

    non_terminals = {f"[{q},{A},{p}]" for q in pda.states for A in pda.stack_alphabet for p in pda.states}

    # Símbolo inicial de la gramática
    start_symbol = "S"

    # Generar producciones
    productions = set()

    # Regla 1: Producción inicial S -> [q0, Z0, qf] para cada estado final qf
    accept_states_to_use = pda.accept_states if pda.accept_states else pda.states
    for qf in sorted(list(accept_states_to_use)):
        productions.add(f"{start_symbol} → [{pda.start_state},{pda.start_symbol},{qf}]")

    # Regla 2: δ(q, a, Z) = (r, Y1Y2...Yk)
    for (q, a, Z), results in pda.transitions.items():
        for (r, gamma) in results:
            a_sym = a if a != EPSILON else 'ε'

            if not gamma:  # k = 0, se desapila Z y no se apila nada
                productions.add(f"[{q},{Z},{r}] → {a_sym}")
            else:  # k > 0
                k = len(gamma)
                # Generar todas las combinaciones de k-1 estados intermedios
                p_states = [r] + ['p'] * (k - 1)

                # Iterar sobre todas las posibles asignaciones de estados reales a los p_i
                for intermediate_states in itertools.product(pda.states, repeat=k - 1):
                    all_states = [r] + list(intermediate_states)

                    # Para cada estado final p_k
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

Nota: La gramática genera el lenguaje aceptado por estado final.
Una variable [q,A,p] representa las cadenas que llevan al PDA del estado q al p, 
consumiendo el símbolo A de la pila.
"""

    return result


# --- Simulación mejorada del PDA ---
class PDASimulator:
    """Simulador de PDA con capacidades de debugging y análisis (implementación BFS mejorada)."""

    def __init__(self, pda: PDA):
        self.pda = pda
        self.max_steps = 2000
        self.max_stack_size = 200

    def simulate(self, input_string: str) -> Dict[str, Any]:
        """Simula la ejecución del PDA en una cadena de entrada usando Búsqueda en Anchura (BFS)."""
        if not self.pda.states or not self.pda.start_state or not self.pda.start_symbol:
            return {'accepted': False, 'error': 'PDA incompleto', 'trace': [], 'final_configs': []}

        # Configuración: (estado, cadena_restante, pila_como_lista)
        initial_config = (self.pda.start_state, input_string, [self.pda.start_symbol])

        # Cola para BFS
        queue = deque([initial_config])

        # Conjunto para evitar ciclos y trabajo redundante.
        # Clave: (estado, longitud_cadena_restante, pila_como_tupla)
        visited = {(self.pda.start_state, len(input_string), (self.pda.start_symbol,))}

        trace = [self._format_config(initial_config, 0, "Initial Configuration")]
        step = 0

        while queue and step < self.max_steps:
            step += 1
            current_state, current_input, current_stack = queue.popleft()

            # --- Verificación de aceptación ---
            # Si la entrada está consumida y estamos en un estado de aceptación, terminamos.
            if not current_input and current_state in self.pda.accept_states:
                trace.append(
                    f"✅ Aceptada en el paso {step}. Estado final: {current_state}, Pila: {''.join(reversed(current_stack)) or 'ε'}")
                return {
                    'accepted': True,
                    'acceptance_type': 'final_state',
                    'trace': trace,
                    'final_configs': [(current_state, current_input, current_stack)],
                    'steps': step
                }

            # --- Lógica de Transiciones ---
            # Explorar todas las transiciones posibles desde la configuración actual.

            # Posibilidad 1: Movimiento-ε (sin consumir entrada)
            self._explore_transitions(current_state, EPSILON, current_input, current_stack, queue, visited, trace, step)

            # Posibilidad 2: Consumir un símbolo de la entrada
            if current_input:
                char_to_read = current_input[0]
                input_after_read = current_input[1:]
                self._explore_transitions(current_state, char_to_read, input_after_read, current_stack, queue, visited,
                                          trace, step)

        if step >= self.max_steps:
            trace.append(f"⚠️ Simulación detenida: se alcanzó el límite de {self.max_steps} pasos.")

        return {
            'accepted': False,
            'trace': trace,
            'final_configs': list(queue),
            'steps': step
        }

    def _explore_transitions(self, state, char_to_read, input_after_read, stack, queue, visited, trace, step):
        """Helper para explorar transiciones que desapilan un símbolo específico o épsilon."""
        # Caso A: La transición requiere desapilar un símbolo específico del tope.
        stack_top = stack[-1] if stack else None
        if stack_top is not None:
            self._apply_transition_rule(state, char_to_read, stack_top, input_after_read, stack[:-1], queue, visited,
                                        trace, step)

        # Caso B: La transición desapila épsilon (no desapila nada).
        self._apply_transition_rule(state, char_to_read, EPSILON, input_after_read, stack, queue, visited, trace, step)

    def _apply_transition_rule(self, state, char_to_read, symbol_to_pop, input_after_read, stack_after_pop, queue,
                               visited, trace, step):
        """Aplica una regla de transición si existe y genera nuevas configuraciones."""
        transition_key = (state, char_to_read, symbol_to_pop)

        if transition_key in self.pda.transitions:
            for next_state, push_symbols in self.pda.transitions[transition_key]:
                new_stack_list = list(stack_after_pop)
                # CORRECCIÓN: Apilar en orden inverso para que el primer símbolo quede en el tope.
                new_stack_list.extend(reversed(push_symbols))

                if len(new_stack_list) <= self.max_stack_size:
                    new_stack_tuple = tuple(new_stack_list)

                    # Usar longitud de la entrada para la clave `visited`
                    config_key = (next_state, len(input_after_read), new_stack_tuple)

                    if config_key not in visited:
                        visited.add(config_key)
                        new_config = (next_state, input_after_read, new_stack_list)
                        queue.append(new_config)

                        # Añadir al trace
                        push_str = "".join(push_symbols) or 'ε'
                        move_desc = f"Read '{char_to_read}'" if char_to_read != EPSILON else "ε-move"
                        trace_msg = f"{move_desc}, Pop '{symbol_to_pop}', Push '{push_str}'"
                        trace.append(self._format_config(new_config, step, trace_msg))

    def _format_config(self, config: Tuple[str, str, List[str]], step: int, move: Optional[str] = None) -> str:
        """Formatea una configuración para mostrar en el trace de forma más clara."""
        state, input_str, stack = config
        # En la pila, el tope está al final de la lista, por lo que lo mostramos primero.
        stack_str = "".join(reversed(stack)) if stack else "ε"
        input_display = f"'{input_str}'" if input_str else "ε"
        move_info = f"  (Action: {move})" if move else ""
        return f"Step {step}: State={state}, Input={input_display}, Stack={stack_str}{move_info}"


# --- Función para crear la vista del AP (para ser importada) ---
def create_ap_view(page: ft.Page):
    """Crea y retorna el layout principal para la herramienta de Autómata de Pila."""
    page.scroll = ft.ScrollMode.AUTO

    # Instancia del PDA
    pda = PDA()

    # --- Componentes de UI ---

    # Campos de entrada con mejor diseño
    states_field = ft.TextField(
        label="Estados (Q)",
        hint_text="q0, q1, q2, ...",
        border_radius=8,
        filled=True
    )

    input_alphabet_field = ft.TextField(
        label="Alfabeto de entrada (Σ)",
        hint_text="a, b, c, ...",
        border_radius=8,
        filled=True
    )

    stack_alphabet_field = ft.TextField(
        label="Alfabeto de pila (Γ)",
        hint_text="A, B, Z, ...",
        border_radius=8,
        filled=True
    )

    start_state_field = ft.TextField(
        label="Estado inicial (q₀)",
        hint_text="q0",
        border_radius=8,
        filled=True
    )

    start_symbol_field = ft.TextField(
        label="Símbolo inicial de pila (Z₀)",
        hint_text="Z",
        border_radius=8,
        filled=True
    )

    accept_states_field = ft.TextField(
        label="Estados de aceptación (F)",
        hint_text="q1, q2, ...",
        border_radius=8,
        filled=True
    )

    transitions_field = ft.TextField(
        label="Transiciones (δ)",
        hint_text="q0,a,Z -> q1,AZ\nq1,b,A -> q1,ε",
        multiline=True,
        min_lines=3,
        max_lines=10,
        border_radius=8,
        filled=True
    )

    # Campo para simulación
    simulation_input = ft.TextField(
        label="Cadena a simular",
        hint_text="Ingrese la cadena...",
        border_radius=8,
        filled=True
    )

    # Área de resultados
    result_text = ft.Text(
        value="📝 Complete la definición del PDA para ver los resultados.",
        size=14,
        color=ft.Colors.GREY_700
    )

    cfg_result = ft.Text(
        value="",
        size=12,
        color=ft.Colors.GREY_800,
        selectable=True,
        font_family="monospace"
    )

    simulation_result = ft.Text(
        value="",
        size=12,
        color=ft.Colors.GREY_800,
        selectable=True,
        font_family="monospace"
    )

    # Canvas para visualización
    canvas_container = ft.Container(
        content=cv.Canvas(
            width=800,
            height=500,
            shapes=[],
        ),
        border=ft.border.all(1, ft.Colors.GREY_300),
        border_radius=8,
        bgcolor=ft.Colors.WHITE,
        padding=10
    )
    canvas = canvas_container.content

    # --- Funciones de manejo de eventos ---

    def validate_and_update(e=None):
        """Valida y actualiza la visualización del PDA."""
        try:
            pda.parse_from_ui(
                states_field.value or "",
                input_alphabet_field.value or "",
                stack_alphabet_field.value or "",
                start_state_field.value or "",
                start_symbol_field.value or "",
                accept_states_field.value or "",
                transitions_field.value or ""
            )

            result_text.value = "✅ PDA válido y cargado correctamente."
            result_text.color = ft.Colors.GREEN_700

            # Actualizar visualización
            PDAVisualizer.draw_pda(pda, canvas)

        except ValueError as error:
            result_text.value = f"❌ Error en la definición:\n{str(error)}"
            result_text.color = ft.Colors.RED_700
            canvas.shapes.clear()
            canvas.update()
        except Exception as error:
            result_text.value = f"💥 Error inesperado: {str(error)}"
            result_text.color = ft.Colors.RED_700

        page.update()

    def convert_to_cfg(e):
        """Convierte el PDA a gramática libre de contexto."""
        try:
            if not pda.states:
                cfg_result.value = "⚠️ Primero debe definir un PDA válido."
                cfg_result.color = ft.Colors.ORANGE_700
            else:
                cfg_text = convert_pda_to_cfg(pda)
                cfg_result.value = cfg_text
                cfg_result.color = ft.Colors.BLUE_800
        except Exception as error:
            cfg_result.value = f"❌ Error en la conversión: {str(error)}"
            cfg_result.color = ft.Colors.RED_700

        page.update()

    def simulate_input(e):
        """Simula la ejecución del PDA con la cadena de entrada."""
        try:
            if not pda.states:
                simulation_result.value = "⚠️ Primero debe definir un PDA válido."
                simulation_result.color = ft.Colors.ORANGE_700
                page.update()
                return

            input_str = simulation_input.value or ""
            simulator = PDASimulator(pda)
            result = simulator.simulate(input_str)

            # Formatear resultado
            if result['accepted']:
                status = f"✅ ACEPTADA (por {result.get('acceptance_type', 'estado final')})"
                color = ft.Colors.GREEN_700
            else:
                status = "❌ RECHAZADA"
                color = ft.Colors.RED_700

            trace_text = "\n".join(result['trace'][:30])  # Limitar a 30 líneas
            if len(result['trace']) > 30:
                trace_text += f"\n... (mostrando primeras 30 de {len(result['trace'])} líneas)"

            simulation_result.value = f"""Simulación de: "{input_str or 'ε'}"
{status}
Pasos ejecutados: {result.get('steps', 0)}

Trace de ejecución:
{trace_text}"""

            simulation_result.color = color

        except Exception as error:
            simulation_result.value = f"❌ Error en la simulación: {str(error)}"
            simulation_result.color = ft.Colors.RED_700

        page.update()

    def load_example(e):
        """Carga un ejemplo predefinido."""
        # Ejemplo: PDA que acepta {a^n b^n | n ≥ 1}
        states_field.value = "q0, q1, q2"
        input_alphabet_field.value = "a, b"
        stack_alphabet_field.value = "A, Z"
        start_state_field.value = "q0"
        start_symbol_field.value = "Z"
        accept_states_field.value = "q2"
        transitions_field.value = """# Apilar 'A' por cada 'a'
q0,a,Z -> q0,AZ
q0,a,A -> q0,AA
# Cambiar de estado al leer 'b' y desapilar 'A'
q0,b,A -> q1,ε
# Desapilar 'A' por cada 'b'
q1,b,A -> q1,ε
# Si la pila solo contiene Z, pasar al estado de aceptación
q1,ε,Z -> q2,Z"""

        validate_and_update()

    def clear_all(e):
        """Limpia todos los campos."""
        for field in [states_field, input_alphabet_field, stack_alphabet_field,
                      start_state_field, start_symbol_field, accept_states_field,
                      transitions_field, simulation_input]:
            field.value = ""

        result_text.value = "📝 Complete la definición del PDA para ver los resultados."
        result_text.color = ft.Colors.GREY_700
        cfg_result.value = ""
        simulation_result.value = ""
        PDAVisualizer.draw_pda(PDA(), canvas)  # Dibuja el estado vacío
        page.update()

    # --- Botones ---
    validate_btn = ft.ElevatedButton(
        "🔍 Validar y Visualizar",
        on_click=validate_and_update,
        tooltip="Valida la definición del PDA y lo dibuja",
        style=ft.ButtonStyle(color=ft.Colors.WHITE, bgcolor=ft.Colors.BLUE_600)
    )

    cfg_btn = ft.ElevatedButton(
        "📝 Convertir a CFG",
        on_click=convert_to_cfg,
        tooltip="Genera una Gramática Libre de Contexto equivalente",
        style=ft.ButtonStyle(color=ft.Colors.WHITE, bgcolor=ft.Colors.GREEN_600)
    )

    simulate_btn = ft.ElevatedButton(
        "▶️ Simular",
        on_click=simulate_input,
        tooltip="Ejecuta la simulación con la cadena de entrada",
        style=ft.ButtonStyle(color=ft.Colors.WHITE, bgcolor=ft.Colors.PURPLE_600)
    )

    example_btn = ft.OutlinedButton(
        "📋 Cargar Ejemplo (aⁿbⁿ)",
        on_click=load_example,
        tooltip="Carga un PDA de ejemplo que reconoce a^n b^n"
    )

    clear_btn = ft.OutlinedButton(
        "🗑️ Limpiar Todo",
        on_click=clear_all,
        tooltip="Borra todos los campos y resultados"
    )

    # --- Layout de la aplicación ---

    title = ft.Text(
        "Simulador de Autómata de Pila (PDA)",
        size=28,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.BLUE_900
    )

    subtitle = ft.Text(
        "Herramienta profesional para definir, visualizar y simular autómatas de pila",
        size=16,
        color=ft.Colors.GREY_600
    )

    definition_section = ft.Card(
        elevation=2,
        content=ft.Container(
            padding=20,
            content=ft.Column([
                ft.Text("📋 Definición del PDA", size=20, weight=ft.FontWeight.BOLD),
                states_field,
                input_alphabet_field,
                stack_alphabet_field,
                start_state_field,
                start_symbol_field,
                accept_states_field,
                transitions_field,
                ft.Row([
                    validate_btn,
                    example_btn,
                    clear_btn
                ], spacing=10, alignment=ft.MainAxisAlignment.START),
                ft.Container(result_text, padding=ft.padding.only(top=10)),
            ])
        )
    )
    
    simulation_section = ft.Card(
        elevation=2,
        content=ft.Container(
            padding=20,
            content=ft.Column([
                ft.Row([
                    ft.Text("▶️ Simulación", size=20, weight=ft.FontWeight.BOLD),
                    simulate_btn
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                simulation_input,
                ft.Container(
                    content=ft.Column([simulation_result]),
                    bgcolor=ft.Colors.GREY_50,
                    border_radius=8,
                    padding=15,
                    margin=ft.margin.only(top=10),
                    border=ft.border.all(1, ft.Colors.GREY_200)
                )
            ])
        )
    )
    
    cfg_section = ft.Card(
        elevation=2,
        content=ft.Container(
            padding=20,
            content=ft.Column([
                ft.Row([
                    ft.Text("📝 Conversión a Gramática", size=20, weight=ft.FontWeight.BOLD),
                    cfg_btn
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Container(
                    content=ft.Column([cfg_result]),
                    bgcolor=ft.Colors.GREY_50,
                    border_radius=8,
                    padding=15,
                    margin=ft.margin.only(top=10),
                    border=ft.border.all(1, ft.Colors.GREY_200)
                )
            ])
        )
    )

    visualization_section = ft.Card(
        elevation=2,
        content=ft.Container(
            padding=20,
            content=ft.Column([
                ft.Text("🎨 Visualización Gráfica", size=20, weight=ft.FontWeight.BOLD),
                ft.Text("Usa 'Validar y Visualizar' para generar el diagrama.", size=12, color=ft.Colors.GREY_600),
                ft.Column([canvas_container], scroll=ft.ScrollMode.ADAPTIVE, expand=True)
            ])
        )
    )

    # --- MODIFICACIÓN DEL LAYOUT A UNA SOLA COLUMNA VERTICAL ---
    return ft.Column(
        spacing=20,
        controls=[
            ft.Container(content=ft.Column([title, subtitle]), padding=ft.padding.only(bottom=10)),
            definition_section,
            simulation_section,
            cfg_section,
            visualization_section,
        ]
    )


# --- Punto de entrada (para ejecución independiente y pruebas) ---
if __name__ == "__main__":
    def main(page: ft.Page):
        page.title = "Prueba de Herramienta AP"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.padding = 20
        page.vertical_alignment = ft.MainAxisAlignment.START
        page.horizontal_alignment = ft.CrossAxisAlignment.CENTER


        # Creamos la vista llamando a nuestra nueva función
        ap_view = create_ap_view(page)
        page.add(ap_view)
        page.update()


    ft.app(target=main)