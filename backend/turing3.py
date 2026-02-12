import flet as ft
import os
os.environ['MPLCONFIGDIR'] = '/data/user/0/com.flet.pythonproject/cache/mplconfig'
import networkx as nx
# Matplotlib sigue siendo necesario como backend de dibujo para networkx
import matplotlib
matplotlib.use('Agg')  # Esencial para generar archivos sin GUI
import matplotlib.pyplot as plt
import tempfile
import time
import shutil  # Para limpieza de directorio temporal


def crear_grafo_networkx(estados, transiciones, estado_inicial, estados_aceptacion,
                         estado_actual=None, estado_anterior=None, simbolo_leido=None, transicion_tomada=None):
    """
    Crea un objeto grafo de NetworkX representando la Máquina de Turing.
    Asigna atributos 'color' a nodos.
    Asigna atributos 'color', 'width', 'label' a aristas.
    Resalta la transición específica tomada (estado_anterior -> estado_actual).
    """
    G = nx.DiGraph()

    # --- Definición de colores ---
    colores_nodos = {
        'inicial': 'lightblue',
        'aceptacion': 'lightgreen',
        'normal': 'lightgray',
        'actual': 'yellow',
        'inicial_actual': 'lightskyblue',
        'aceptacion_actual': 'greenyellow'
    }
    color_transicion_activa = 'blue'  # Color para la arista de la transición tomada
    color_transicion_normal = 'black'
    color_transicion_actual_origen = 'red'  # Color aristas saliendo del estado *previo* a la transición actual
    ancho_transicion_activa = 2.5
    ancho_transicion_normal = 1.0

    # --- Añadir Nodos ---
    for estado in estados:
        # Determinar color del nodo
        color_nodo = colores_nodos['normal']  # Default
        if estado == estado_actual:
            if estado in estados_aceptacion:
                color_nodo = colores_nodos['aceptacion_actual']
            elif estado == estado_inicial:
                color_nodo = colores_nodos['inicial_actual']
            else:
                color_nodo = colores_nodos['actual']
        elif estado == estado_inicial and estado in estados_aceptacion:
            color_nodo = colores_nodos['aceptacion']
        elif estado == estado_inicial:
            color_nodo = colores_nodos['inicial']
        elif estado in estados_aceptacion:
            color_nodo = colores_nodos['aceptacion']

        G.add_node(estado, color=color_nodo)

    # --- Añadir Aristas ---
    if not isinstance(transiciones, dict):  # Asegurar que transiciones sea un dict
        transiciones = {}

    for origen, trans_dict in transiciones.items():
        if not isinstance(trans_dict, dict): continue

        for entrada, trans_info in trans_dict.items():
            if not isinstance(trans_info, (list, tuple)) or len(trans_info) != 3: continue
            destino, escribir, mover = trans_info
            label = f"{entrada} → {escribir},{mover}"

            # Determinar estilo de la arista
            edge_color = color_transicion_normal
            edge_width = ancho_transicion_normal

            # ¿Es esta la transición que se acaba de tomar?
            es_transicion_activa = (
                    estado_anterior == origen and
                    simbolo_leido == entrada and
                    transicion_tomada == trans_info
            )

            if es_transicion_activa:
                edge_color = color_transicion_activa
                edge_width = ancho_transicion_activa
            elif estado_anterior == origen:
                edge_color = color_transicion_actual_origen

            G.add_edge(origen, destino, label=label, color=edge_color, width=edge_width)

    return G


def simular_maquina_turing_paso_a_paso(estados, transiciones, estado_inicial, estados_aceptacion, cinta,
                                       posicion_cabezal, max_steps=1000):
    """
    Simula la Máquina de Turing. Cada paso incluye estado_anterior, simbolo_leido,
    y transicion_tomada para facilitar la visualización.
    """
    pasos = []
    estado_actual = estado_inicial
    estado_anterior = None
    steps = 0

    while len(cinta) <= posicion_cabezal: cinta.append("_")

    # Paso inicial
    pasos.append({
        "paso_num": steps,
        "estado": estado_actual,
        "estado_anterior": None,
        "simbolo_leido": None,
        "transicion_tomada": None,
        "cinta": cinta.copy(),
        "posicion": posicion_cabezal,
        "mensaje": "Estado inicial"
    })

    while steps < max_steps:
        estado_anterior = estado_actual

        if posicion_cabezal < 0:
            cinta.insert(0, "_")
            posicion_cabezal = 0
        elif posicion_cabezal >= len(cinta):
            cinta.append("_")

        simbolo_actual = cinta[posicion_cabezal]

        transicion_encontrada = False
        if estado_actual in transiciones and simbolo_actual in transiciones[estado_actual]:
            transicion_info = transiciones[estado_actual][simbolo_actual]

            if isinstance(transicion_info, (list, tuple)) and len(transicion_info) == 3:
                destino, escribir, mover = transicion_info
                transicion_encontrada = True

                mensaje_transicion = f"δ({estado_actual}, {simbolo_actual}) = ({destino}, {escribir}, {mover})"

                cinta[posicion_cabezal] = escribir
                if mover == "R":
                    posicion_cabezal += 1
                elif mover == "L":
                    posicion_cabezal -= 1
                elif mover != "S":
                    return pasos, f"Error: Movimiento inválido '{mover}'"

                estado_actual = destino
                steps += 1

                paso_info = {
                    "paso_num": steps,
                    "estado": estado_actual,
                    "estado_anterior": estado_anterior,
                    "simbolo_leido": simbolo_actual,
                    "transicion_tomada": transicion_info,
                    "cinta": cinta.copy(),
                    "posicion": posicion_cabezal,
                    "mensaje": mensaje_transicion
                }

                if estado_actual in estados_aceptacion:
                    paso_info["mensaje"] += " - ¡Estado de ACEPTACIÓN!"
                    pasos.append(paso_info)
                    return pasos, "Cadena ACEPTADA"

                pasos.append(paso_info)
            else:
                mensaje_error = f"Error interno: Formato de transición inválido para δ({estado_actual}, {simbolo_actual})"
                pasos.append({
                    "paso_num": steps + 1, "estado": estado_actual, "estado_anterior": estado_anterior,
                    "simbolo_leido": simbolo_actual, "transicion_tomada": None,
                    "cinta": cinta.copy(), "posicion": posicion_cabezal, "mensaje": mensaje_error
                })
                return pasos, "Error en definición interna de transiciones"

        if not transicion_encontrada:
            steps += 1
            pasos.append({
                "paso_num": steps,
                "estado": estado_actual,
                "estado_anterior": estado_anterior,
                "simbolo_leido": simbolo_actual,
                "transicion_tomada": None,
                "cinta": cinta.copy(),
                "posicion": posicion_cabezal,
                "mensaje": f"No hay transición para δ({estado_actual}, {simbolo_actual}) - ¡RECHAZADA!"
            })
            return pasos, "Cadena RECHAZADA"

    steps += 1
    pasos.append({
        "paso_num": steps,
        "estado": estado_actual, "estado_anterior": estado_anterior,
        "simbolo_leido": None, "transicion_tomada": None,
        "cinta": cinta.copy(), "posicion": posicion_cabezal,
        "mensaje": f"Límite de {max_steps} pasos alcanzado."
    })
    return pasos, "Límite de pasos alcanzado (posible bucle)"


# --- Interfaz Gráfica con Flet ---

def create_turing_view(page: ft.Page):
    """Crea la vista principal de la aplicación Flet."""
    page.title = "Simulador de Máquinas de Turing"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 10
    page.bgcolor = ft.Colors.BLUE_GREY_50

    primary_color = ft.Colors.BLUE_700
    text_color = ft.Colors.BLACK87
    input_border_color = ft.Colors.BLUE_GREY_300

    temp_dir = tempfile.mkdtemp(prefix="turing_flet_")
    print(f"Directorio temporal para grafos: {temp_dir}")

    def create_styled_textfield(label, multiline=False, min_lines=1, width=None, expand=False):
        return ft.TextField(label=label, border_color=input_border_color, focused_border_color=primary_color,
                            text_style=ft.TextStyle(color=text_color, size=14),
                            label_style=ft.TextStyle(color=ft.Colors.BLUE_GREY_700),
                            expand=expand, width=width, multiline=multiline, min_lines=min_lines,
                            max_lines=5 if multiline else 1,
                            shift_enter=True if multiline else False, border_radius=5)

    estados_input = create_styled_textfield("Estados (ej: q0,q1,q2)", expand=True)
    estado_inicial_input = create_styled_textfield("Estado Inicial (ej: q0)", width=150)
    aceptados_input = create_styled_textfield("Estados Aceptación (ej: qf)", expand=True)
    transiciones_input = create_styled_textfield("Transiciones (estado,leer -> nuevo_estado,escribir,mover)",
                                                 multiline=True, min_lines=4, expand=True)
    cinta_input = create_styled_textfield("Cinta Inicial (ej: 1101)", expand=True)
    posicion_cabezal_input = create_styled_textfield("Pos. Cabezal (ej: 0)", width=150)
    posicion_cabezal_input.value = "0"

    grafo_container = ft.Container(content=ft.Image(fit=ft.ImageFit.CONTAIN), width=700, height=400,
                                   bgcolor=ft.Colors.WHITE, border_radius=10, padding=10, visible=False)
    cinta_visualizacion = ft.Row(spacing=1, wrap=False, scroll=ft.ScrollMode.ADAPTIVE, visible=False)
    estado_texto = ft.Text(size=16, weight="bold", color=primary_color, visible=False)
    mensaje_texto = ft.Text(size=14, color=text_color, visible=False, selectable=True)
    resultado_texto = ft.Text(size=16, weight="bold", color=text_color, visible=False, selectable=True)
    contador_pasos = ft.Text("Paso 0 de 0", size=14, color=primary_color, visible=False)

    pasos_simulacion = []
    paso_actual = 0

    def actualizar_visualizacion():
        """Actualiza la UI con el estado del paso actual de la simulación."""
        nonlocal paso_actual
        if not pasos_simulacion or paso_actual < 0 or paso_actual >= len(pasos_simulacion):
            controles_navegacion.visible = False
            cinta_visualizacion.visible = False
            estado_texto.visible = False
            mensaje_texto.visible = False
            grafo_container.visible = False
            page.update()
            return

        paso_info = pasos_simulacion[paso_actual]
        estado = paso_info["estado"]
        cinta = paso_info["cinta"]
        posicion = paso_info["posicion"]
        mensaje = paso_info["mensaje"]
        estado_anterior = paso_info.get("estado_anterior")
        simbolo_leido = paso_info.get("simbolo_leido")
        transicion_tomada = paso_info.get("transicion_tomada")

        estado_texto.value = f"Estado Actual: {estado}"
        mensaje_texto.value = f"Info Paso {paso_info['paso_num']}: {mensaje}"
        estado_texto.visible = True
        mensaje_texto.visible = True

        cinta_visualizacion.controls.clear()
        for i, simbolo in enumerate(cinta):
            es_cabezal = (i == posicion)
            celda = ft.Container(
                content=ft.Text(simbolo if simbolo else "_", size=18,
                                weight=ft.FontWeight.BOLD if es_cabezal else ft.FontWeight.NORMAL,
                                text_align=ft.TextAlign.CENTER, color=ft.Colors.BLACK),
                width=35, height=35, bgcolor=ft.Colors.YELLOW_200 if es_cabezal else ft.Colors.WHITE,
                border=ft.border.all(2 if es_cabezal else 1, ft.Colors.BLUE_GREY_300), alignment=ft.alignment.center,
                border_radius=3
            )
            cinta_visualizacion.controls.append(celda)
        cinta_visualizacion.visible = True
        try:
            scroll_offset = max(0,
                                (posicion * 36) - (cinta_visualizacion.width / 2 if cinta_visualizacion.width else 300))
            cinta_visualizacion.scroll_to(offset=scroll_offset, duration=100)
        except Exception as scroll_err:
            print(f"Advertencia: no se pudo hacer scroll en cinta: {scroll_err}")

        actualizar_grafo(estado, estado_anterior, simbolo_leido, transicion_tomada)

        boton_paso_anterior.disabled = paso_actual <= 0
        boton_paso_inicial.disabled = paso_actual <= 0
        boton_paso_siguiente.disabled = paso_actual >= len(pasos_simulacion) - 1
        boton_paso_final.disabled = paso_actual >= len(pasos_simulacion) - 1
        controles_navegacion.visible = True
        contador_pasos.value = f"Paso {paso_info['paso_num']} de {pasos_simulacion[-1]['paso_num']}"
        contador_pasos.visible = True

        page.update()

    def actualizar_grafo(estado_actual, estado_anterior=None, simbolo_leido=None, transicion_tomada=None):
        """Actualiza y dibuja el grafo."""
        estados, estado_inicial, estados_aceptacion, transiciones = [], "", [], {}
        try:
            estados_str = estados_input.value.strip() if estados_input.value else ""
            inicial_str = estado_inicial_input.value.strip() if estado_inicial_input.value else ""
            aceptados_str = aceptados_input.value.strip() if aceptados_input.value else ""
            trans_str = transiciones_input.value.strip() if transiciones_input.value else ""

            if not estados_str or not inicial_str:
                raise ValueError("Estados o estado inicial no definidos.")

            estados = [s.strip() for s in estados_str.split(',') if s.strip()]
            estado_inicial = inicial_str
            estados_aceptacion = [s.strip() for s in aceptados_str.split(',') if s.strip()]

            for i, trans in enumerate(trans_str.split('\n')):
                linea = trans.strip()
                if not linea: continue
                try:
                    partes = linea.split('->')
                    if len(partes) != 2: raise ValueError("Formato '->' inválido")
                    o_part = partes[0].split(',', 1)
                    if len(o_part) != 2: raise ValueError("Formato 'estado,leer' inválido")
                    origen, entrada = o_part[0].strip(), o_part[1].strip()
                    d_part = partes[1].split(',', 2)
                    if len(d_part) != 3: raise ValueError("Formato 'estado,escribir,mover' inválido")
                    destino, esc, mov = d_part[0].strip(), d_part[1].strip(), d_part[2].strip().upper()
                    if mov not in ["L", "R", "S"]: raise ValueError("Movimiento debe ser L,R,S")
                    if origen not in transiciones: transiciones[origen] = {}
                    transiciones[origen][entrada] = (destino, esc, mov)
                except Exception as parse_err:
                    print(f"Advertencia al parsear grafo, línea {i + 1}: {parse_err} - '{linea}'")

        except Exception as e:
            mensaje_texto.value = f"Error leyendo definición para grafo: {e}"
            mensaje_texto.color = ft.Colors.ORANGE_700
            mensaje_texto.visible = True
            grafo_container.visible = False
            page.update()
            return

        try:
            G = crear_grafo_networkx(estados, transiciones, estado_inicial, estados_aceptacion,
                                     estado_actual, estado_anterior, simbolo_leido, transicion_tomada)

            if not G.nodes:
                grafo_container.visible = False
                page.update()
                return

            pos = nx.spring_layout(G, seed=42, k=0.6, iterations=50)

            plt.clf()
            fig, ax = plt.subplots(figsize=(10, 6))

            node_colors = [G.nodes[n].get('color', 'grey') for n in G.nodes()]
            edge_colors = [G.edges[e].get('color', 'black') for e in G.edges()]
            edge_widths = [G.edges[e].get('width', 1.0) for e in G.edges()]
            edge_labels = nx.get_edge_attributes(G, 'label')

            nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors,
                    edge_color=edge_colors, width=edge_widths,
                    node_size=2000, font_size=10, font_weight='bold', node_shape='o',
                    edgecolors='black', linewidths=1.0, arrowsize=15)

            nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=8,
                                         font_color='black', verticalalignment='bottom',
                                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none',
                                                   boxstyle='round,pad=0.1'))

            img_path = os.path.join(temp_dir, f'graph_{round(time.time() * 1000)}.png')
            plt.savefig(img_path, format='png', bbox_inches='tight', facecolor='white', edgecolor='none', dpi=90)
            plt.close(fig)

            grafo_container.content.src = img_path
            grafo_container.visible = True

            if isinstance(mensaje_texto.value, str) and mensaje_texto.value.startswith("Error leyendo definición"):
                mensaje_texto.visible = False

            page.update()

        except Exception as error:
            error_msg = f"Error al generar/dibujar grafo: {str(error)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            mensaje_texto.value = error_msg
            mensaje_texto.color = ft.Colors.RED
            mensaje_texto.visible = True
            grafo_container.visible = False
            page.update()

    def validar_entradas():
        errores = []
        estados_list = [s.strip() for s in estados_input.value.split(',') if s.strip()]
        inicial = estado_inicial_input.value.strip()
        aceptados_list = [s.strip() for s in aceptados_input.value.split(',') if s.strip()]
        if not estados_list: errores.append("Lista de estados vacía.")
        if not inicial:
            errores.append("Estado inicial no definido.")
        elif inicial not in estados_list:
            errores.append(f"Estado inicial '{inicial}' no está en la lista de estados.")
        for aceptado in aceptados_list:
            if aceptado not in estados_list: errores.append(
                f"Estado de aceptación '{aceptado}' no está en la lista de estados.")
        if not transiciones_input.value.strip():
            errores.append("No hay transiciones definidas.")
        else:
            for i, trans_line in enumerate(transiciones_input.value.strip().split('\n')):
                linea = trans_line.strip()
                if not linea: continue
                try:
                    partes = linea.split('->')
                    if len(partes) != 2: raise ValueError("Formato '->' inválido")
                    o_part = partes[0].split(',', 1)
                    if len(o_part) != 2: raise ValueError("Formato 'estado,leer' inválido")
                    origen, leer = o_part[0].strip(), o_part[1].strip()
                    d_part = partes[1].split(',', 2)
                    if len(d_part) != 3: raise ValueError("Formato 'estado,escribir,mover' inválido")
                    destino, escribir, mover = d_part[0].strip(), d_part[1].strip(), d_part[2].strip().upper()
                    if origen not in estados_list: errores.append(f"L{i + 1}: Estado origen '{origen}' no definido.")
                    if destino not in estados_list: errores.append(f"L{i + 1}: Estado destino '{destino}' no definido.")
                    if mover not in ["L", "R", "S"]: errores.append(
                        f"L{i + 1}: Movimiento '{mover}' inválido (L, R o S).")
                    if not leer: errores.append(f"L{i + 1}: Símbolo a leer no puede ser vacío.")
                except ValueError as e:
                    errores.append(f"L{i + 1}: Formato - {str(e)}")
                except Exception as e:
                    errores.append(f"L{i + 1}: Error - {str(e)}")
        try:
            pos = int(posicion_cabezal_input.value)
            if pos < 0: errores.append("Posición inicial cabezal < 0.")
        except ValueError:
            errores.append("Posición inicial cabezal debe ser número.")
        return errores

    def iniciar_simulacion(e):
        nonlocal pasos_simulacion, paso_actual
        pasos_simulacion, paso_actual = [], 0
        resultado_texto.visible = False
        cinta_visualizacion.visible = False
        estado_texto.visible = False
        mensaje_texto.visible = False
        grafo_container.visible = False
        controles_navegacion.visible = False
        page.update()

        errores = validar_entradas()
        if errores:
            resultado_texto.value = "Errores en definición:\n" + "\n".join([f"• {err}" for err in errores])
            resultado_texto.color = ft.Colors.RED
            resultado_texto.visible = True
            page.update()
            return

        try:
            estados = [s.strip() for s in estados_input.value.split(',') if s.strip()]
            estado_inicial = estado_inicial_input.value.strip()
            estados_aceptacion = [s.strip() for s in aceptados_input.value.split(',') if s.strip()]
            cinta = list(cinta_input.value) if cinta_input.value else ['_']
            posicion_cabezal = int(posicion_cabezal_input.value)
            transiciones = {}
            for trans_line in transiciones_input.value.strip().split('\n'):
                linea = trans_line.strip()
                if not linea: continue
                partes = linea.split('->')
                o_part = partes[0].split(',', 1)
                origen, entrada = o_part[0].strip(), o_part[1].strip()
                d_part = partes[1].split(',', 2)
                destino, escribir, mover = d_part[0].strip(), d_part[1].strip(), d_part[2].strip().upper()
                if origen not in transiciones: transiciones[origen] = {}
                transiciones[origen][entrada] = (destino, escribir, mover)

            pasos_simulacion, resultado = simular_maquina_turing_paso_a_paso(
                estados, transiciones, estado_inicial, estados_aceptacion, cinta.copy(), posicion_cabezal)

            paso_actual = 0
            resultado_texto.value = f"Resultado: {resultado}"
            resultado_texto.color = ft.Colors.GREEN_700 if "ACEPTADA" in resultado else (
                ft.Colors.RED_700 if "RECHAZADA" in resultado else ft.Colors.ORANGE_700)
            resultado_texto.visible = True

            if pasos_simulacion:
                actualizar_visualizacion()
                controles_navegacion.visible = True
            else:
                mensaje_texto.value = "Simulación no generó pasos."
                mensaje_texto.visible = True

            page.update()

        except Exception as error:
            print(f"Error durante simulación: {error}")
            import traceback
            traceback.print_exc()
            resultado_texto.value = f"Error crítico simulación: {str(error)}"
            resultado_texto.color = ft.Colors.RED
            resultado_texto.visible = True
            page.update()

    def mostrar_paso_anterior(e):
        nonlocal paso_actual
        if paso_actual > 0:
            paso_actual -= 1
            actualizar_visualizacion()

    def mostrar_paso_siguiente(e):
        nonlocal paso_actual
        if pasos_simulacion and paso_actual < len(pasos_simulacion) - 1:
            paso_actual += 1
            actualizar_visualizacion()

    def mostrar_paso_inicial(e):
        nonlocal paso_actual
        if paso_actual > 0:
            paso_actual = 0
            actualizar_visualizacion()

    def mostrar_paso_final(e):
        nonlocal paso_actual
        if pasos_simulacion and paso_actual < len(pasos_simulacion) - 1:
            paso_actual = len(pasos_simulacion) - 1
            actualizar_visualizacion()

    def cargar_ejemplo(estados, inicial, aceptados, trans, cinta, pos):
        estados_input.value = estados
        estado_inicial_input.value = inicial
        aceptados_input.value = aceptados
        transiciones_input.value = trans
        cinta_input.value = cinta
        posicion_cabezal_input.value = pos
        resultado_texto.visible = False
        mensaje_texto.visible = False
        pasos_simulacion.clear()
        paso_actual = 0
        controles_navegacion.visible = False
        if inicial:
            actualizar_grafo(inicial)
        else:
            grafo_container.visible = False
        page.update()

    def cargar_ejemplo_anbn(e):
        cargar_ejemplo(estados="q0,q1,q2,q3,qf", inicial="q0", aceptados="qf",
                       trans="q0,a->q1,X,R\nq0,Y->q3,Y,R\nq0,_->qf,_,S\nq1,a->q1,a,R\nq1,Y->q1,Y,R\nq1,b->q2,Y,L\nq2,a->q2,a,L\nq2,Y->q2,Y,L\nq2,X->q0,X,R\nq3,Y->q3,Y,R\nq3,_->qf,_,S",
                       cinta="aabb", pos="0")

    def cargar_ejemplo_copia_w_w(e):
        cargar_ejemplo(
            estados="q0,q1,q2,q3,q4,q5,qf", inicial="q0", aceptados="qf",
            trans="""q0,a->q1,A,R
q0,b->q2,B,R
q0,_->q5,_,R
q1,a->q1,a,R
q1,b->q1,b,R
q1,_->q1,_,R
q1,A->q1,A,R
q1,B->q1,B,R
q1,_->q3,a,L
q2,a->q2,a,R
q2,b->q2,b,R
q2,_->q2,_,R
q2,A->q2,A,R
q2,B->q2,B,R
q2,_->q4,b,L
q3,a->q3,a,L
q3,b->q3,b,L
q3,_->q3,_,L
q3,A->q0,A,R
q3,B->q0,B,R
q4,a->q4,a,L
q4,b->q4,b,L
q4,_->q4,_,L
q4,A->q0,A,R
q4,B->q0,B,R
q5,A->q5,a,R
q5,B->q5,b,R
q5,_->qf,_,S""",
            cinta="ab", pos="0"
        )

    def cargar_ejemplo_busy_beaver_3(e):
        cargar_ejemplo(
            estados="A,B,C,H", inicial="A", aceptados="H",
            trans="""A,_->B,1,R
A,1->C,1,L
B,_->C,1,R
B,1->B,1,R
C,_->H,1,S
C,1->A,1,L""",
            cinta="_", pos="0"
        )

    def cargar_ejemplo_suma_unaria(e):
        cargar_ejemplo(
            estados="q0,q1,qf", inicial="q0", aceptados="qf",
            trans="""q0,1->q0,1,R
q0,_->q1,1,R
q1,1->q1,1,R
q1,_->qf,_,L""",
            cinta="11_111", pos="0"
        )

    boton_iniciar = ft.ElevatedButton("Iniciar Simulación", icon=ft.Icons.PLAY_CIRCLE_OUTLINE,
                                      on_click=iniciar_simulacion,
                                      style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN_200, color=ft.Colors.BLACK87))
    boton_paso_inicial = ft.IconButton(icon=ft.Icons.SKIP_PREVIOUS, tooltip="Ir al inicio",
                                       on_click=mostrar_paso_inicial, disabled=True)
    boton_paso_anterior = ft.IconButton(icon=ft.Icons.NAVIGATE_BEFORE, tooltip="Paso anterior",
                                        on_click=mostrar_paso_anterior, disabled=True)
    boton_paso_siguiente = ft.IconButton(icon=ft.Icons.NAVIGATE_NEXT, tooltip="Paso siguiente",
                                         on_click=mostrar_paso_siguiente, disabled=True)
    boton_paso_final = ft.IconButton(icon=ft.Icons.SKIP_NEXT, tooltip="Ir al final", on_click=mostrar_paso_final,
                                     disabled=True)

    controles_navegacion = ft.Row(
        controls=[boton_paso_inicial, boton_paso_anterior, contador_pasos, boton_paso_siguiente, boton_paso_final],
        alignment=ft.MainAxisAlignment.CENTER, vertical_alignment=ft.CrossAxisAlignment.CENTER, visible=False)

    menu_ejemplos = ft.PopupMenuButton(icon=ft.Icons.LIST_ALT, tooltip="Cargar ejemplos", items=[
        ft.PopupMenuItem(text="a^n b^n (aabb)", on_click=cargar_ejemplo_anbn),
        ft.PopupMenuItem(text="Copia w -> w_w (ab)", on_click=cargar_ejemplo_copia_w_w),
        ft.PopupMenuItem(text="Suma Unaria (11_111)", on_click=cargar_ejemplo_suma_unaria),
        ft.PopupMenuItem(text="Busy Beaver (3-state)", on_click=cargar_ejemplo_busy_beaver_3),
    ])

    leyenda = ft.Container(
        content=ft.Column([
            ft.Text("Leyenda del Grafo:", weight="bold", size=14),
            ft.Row([ft.Container(width=15, height=15, bgcolor='yellow', border_radius=3), ft.Text("Actual", size=12)]),
            ft.Row(
                [ft.Container(width=15, height=15, bgcolor='lightblue', border_radius=3), ft.Text("Inicial", size=12)]),
            ft.Row([ft.Container(width=15, height=15, bgcolor='lightgreen', border_radius=3),
                    ft.Text("Aceptación", size=12)]),
            ft.Row(
                [ft.Container(width=15, height=15, bgcolor='lightgray', border_radius=3), ft.Text("Normal", size=12)]),
            ft.Row([ft.Container(width=20, height=3, bgcolor='blue'),
                    ft.Text("Transición Tomada", size=12, color='blue')]),
            ft.Row([ft.Container(width=20, height=2, bgcolor='red'),
                    ft.Text("Otras opciones desde estado anterior", size=12, color='red')]),
        ], spacing=3), padding=10, bgcolor=ft.Colors.WHITE, border=ft.border.all(1, ft.Colors.GREY_300),
        border_radius=5, margin=ft.margin.only(top=10))

    instrucciones = ft.Container(
        content=ft.Column([
            ft.Text("Instrucciones:", weight="bold", size=14),
            ft.Text("• Estados/Aceptados: separados por comas.", size=12),
            ft.Text("• Transiciones: estado,leer -> nuevo_estado,escribir,mover (L/R/S)", size=12),
            ft.Text("• Use '_' para el símbolo blanco.", size=12)
        ], spacing=2),
        padding=10, bgcolor=ft.Colors.BLUE_50, border=ft.border.all(1, ft.Colors.GREY_300), border_radius=5,
        margin=ft.margin.only(bottom=10))

    definicion_section = ft.Container(
        ft.Column([
            ft.Row([ft.Icon(ft.Icons.INFO_OUTLINE, color=primary_color),
                    ft.Text("Definición de la Máquina de Turing", style=ft.TextThemeStyle.TITLE_MEDIUM,
                            color=primary_color), menu_ejemplos], alignment=ft.MainAxisAlignment.START, spacing=5),
            instrucciones,
            ft.Row([estados_input, estado_inicial_input, aceptados_input], spacing=10),
            transiciones_input,
            ft.Row([cinta_input, posicion_cabezal_input], spacing=10),
            ft.Row([boton_iniciar], spacing=10, alignment=ft.MainAxisAlignment.END),
            # Botón de importar eliminado de esta fila
        ]),
        padding=15, bgcolor=ft.Colors.WHITE, border_radius=10, margin=ft.margin.only(bottom=15))

    visualizacion_section = ft.Container(
        ft.Column([
            ft.Row([ft.Icon(ft.Icons.VISIBILITY, color=primary_color),
                    ft.Text("Visualización de la Ejecución", style=ft.TextThemeStyle.TITLE_MEDIUM,
                            color=primary_color)], alignment=ft.MainAxisAlignment.START, spacing=5),
            ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
            estado_texto,
            mensaje_texto,
            ft.Text("Cinta:", weight="bold", visible=cinta_visualizacion.visible, size=14),
            cinta_visualizacion,
            ft.Divider(height=15, color=ft.Colors.TRANSPARENT),
            controles_navegacion,
            ft.Divider(height=15, color=ft.Colors.TRANSPARENT),
            ft.Row([
                ft.Column([grafo_container], alignment=ft.MainAxisAlignment.CENTER,
                          horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=5),
                ft.Column([leyenda], alignment=ft.MainAxisAlignment.START, expand=1)
            ], vertical_alignment=ft.CrossAxisAlignment.START, spacing=20),
            resultado_texto,
        ], spacing=10),
        padding=15, bgcolor=ft.Colors.WHITE, border_radius=10)

    main_layout = ft.Column([definicion_section, visualizacion_section], expand=True, scroll=ft.ScrollMode.ADAPTIVE)
    return main_layout


def main(page: ft.Page):
    """Configura y añade el componente principal a la página."""
    page.temp_dir_path = None
    turing_component = create_turing_view(page)
    page.add(turing_component)
    page.update()


def cleanup_temp_dir(path):
    if path and os.path.exists(path):
        try:
            shutil.rmtree(path)
            print(f"Directorio temporal {path} eliminado.")
        except Exception as e:
            print(f"Error limpiando directorio temporal {path}: {e}")


if __name__ == "__main__":
    original_mkdtemp = tempfile.mkdtemp
    app_temp_dir = None


    def custom_mkdtemp(*args, **kwargs):
        global app_temp_dir
        path = original_mkdtemp(*args, **kwargs)
        app_temp_dir = path
        tempfile.mkdtemp = original_mkdtemp
        return path


    tempfile.mkdtemp = custom_mkdtemp

    try:
        ft.app(target=main)
    finally:
        if app_temp_dir:
            cleanup_temp_dir(app_temp_dir)
        else:
            print("No se pudo determinar el directorio temporal para limpiar.")