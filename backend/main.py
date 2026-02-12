# AutomatApp.py
import flet as ft
import traceback  # Para imprimir errores detallados

# --- 1. Importar las funciones que crean la UI de cada herramienta ---
try:
    from expre import create_automata_view
    from AP import create_ap_view
    from turing3 import create_turing_view

    print("INFO: Funciones de UI importadas correctamente.")
except ImportError as e:
    print(f"ERROR CRÍTICO al importar vistas: {e}")
    print("--- Asegúrate que los archivos .py estén en la misma carpeta que AutomatApp.py ---")


    def create_error_view(tool_name, error_msg):
        return ft.Column([
            ft.Text(f"Error al importar {tool_name}", color=ft.Colors.RED, size=18),
            ft.Text(f"{error_msg}", selectable=True)
        ], expand=True, horizontal_alignment=ft.CrossAxisAlignment.CENTER)


    def create_automata_view(page):
        return create_error_view("Autómatas", e)


    def create_ap_view(page):
        return create_error_view("AP", e)


    def create_turing_view(page):
        return create_error_view("Turing", e)


# --- Función Principal de la Aplicación ---
def main(page: ft.Page):
    page.title = "Suite de Teoría de la Computación"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 0  # El padding se manejará en los contenedores internos

    # --- Área Principal de Contenido ---
    main_content_column = ft.Column(
        [ft.ProgressRing()],
        expand=True,
        alignment=ft.MainAxisAlignment.START,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH
    )

    # --- Función para Cambiar la Vista Mostrada (sin cambios en su lógica interna) ---
    def change_view(event_or_index):
        selected_index = -1

        if isinstance(event_or_index, ft.ControlEvent):
            if event_or_index.control:
                selected_index = event_or_index.control.selected_index
            else:
                return
        elif isinstance(event_or_index, int):
            selected_index = event_or_index
        else:
            return

        print(f"INFO: Cargando vista índice: {selected_index}")
        current_view = None

        try:
            if selected_index == 0:
                current_view = create_automata_view(page)
            elif selected_index == 1:
                current_view = create_ap_view(page)
            elif selected_index == 2:
                current_view = create_turing_view(page)
            else:
                current_view = ft.Text(f"Índice {selected_index} no válido")
        except Exception as e:
            print(f"ERROR al crear vista {selected_index}: {e}")
            traceback.print_exc()
            current_view = ft.Column(
                [
                    ft.Text(f"Error al cargar herramienta {selected_index}", color=ft.Colors.RED, size=16),
                    ft.Text(f"{e}", selectable=True)
                ],
                expand=True,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER
            )

        main_content_column.controls.clear()
        if current_view:
            main_content_column.controls.append(current_view)
        else:
            main_content_column.controls.append(ft.Text("Error: Vista no generada."))

        main_content_column.update()
        page.update()
        print(f"INFO: Columna de contenido y Página actualizadas para índice {selected_index}.")

    # --- Controles de Navegación Responsivos ---

    # 1. Menú para pantallas pequeñas (CORREGIDO: ft.Icons)
    app_bar_menu_button = ft.PopupMenuButton(
        icon=ft.Icons.MENU,
        items=[
            ft.PopupMenuItem(text="AF", on_click=lambda _: change_view(0)),
            ft.PopupMenuItem(text="AP", on_click=lambda _: change_view(1)),
            ft.PopupMenuItem(text="MT", on_click=lambda _: change_view(2)),
        ]
    )

    # 2. Barra lateral (Rail) para pantallas grandes (CORREGIDO: ft.Icons)
    navigation_rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=100,
        min_extended_width=200,
        group_alignment=-0.9,
        destinations=[
            ft.NavigationRailDestination(icon=ft.Icons.HUB_OUTLINED, label="AF", selected_icon=ft.Icons.HUB),
            ft.NavigationRailDestination(icon=ft.Icons.LAYERS_OUTLINED, label="AP", selected_icon=ft.Icons.LAYERS),
            ft.NavigationRailDestination(icon=ft.Icons.MEMORY_OUTLINED, label="MT", selected_icon=ft.Icons.MEMORY),
        ],
        on_change=change_view,
        # ELIMINADO: expand=True - esto causaba el problema
    )

    # --- Función para manejar el cambio de tamaño de la ventana ---
    def handle_resize(e):
        BREAKPOINT = 700  # Píxeles para cambiar de vista móvil a escritorio

        # CORRECCIÓN PRINCIPAL: Usar page.width en lugar de page.window_width
        if page.width <= BREAKPOINT:
            # VISTA PEQUEÑA (MÓVIL)
            navigation_rail.visible = False
            page.appbar.leading = app_bar_menu_button
            page.appbar.title = ft.Text("Teoría de la Computación")
            page.appbar.visible = True
        else:
            # VISTA GRANDE (ESCRITORIO)
            navigation_rail.visible = True
            page.appbar.visible = False  # Ocultamos el AppBar

        page.update()

    # Asignar la función al evento de redimensionar
    page.on_resize = handle_resize

    # --- Definir el Layout Principal de la Página ---
    page.appbar = ft.AppBar(
        leading=app_bar_menu_button,  # El botón de menú
        title=ft.Text("Teoría de la Computación"),
        visible=False  # Inicialmente oculto, handle_resize decidirá
    )

    # CORRECCIÓN PRINCIPAL: Envolver el Row principal en un Container con altura fija
    main_row = ft.Row(
        [
            # CORRECCIÓN: Envolver el NavigationRail en un Container con altura fija
            ft.Container(
                content=navigation_rail,
                height=page.height - 60 if page.height else 600,  # Altura fija menos espacio para appbar
                width=120,  # Ancho fijo para el rail
            ),
            ft.VerticalDivider(width=1),
            # El contenido principal va dentro de un contenedor para darle padding
            ft.Container(
                content=main_content_column,
                padding=ft.padding.all(10),
                expand=True,
            )
        ],
        expand=True,
        spacing=0,
    )

    page.add(main_row)

    # --- Cargar la Vista Inicial ---
    print("INFO: Realizando carga inicial...")
    # Llamar a handle_resize para establecer el layout correcto al inicio
    handle_resize(None)
    # Cargar la herramienta del índice 0
    change_view(0)
    print("INFO: Carga inicial completada.")


# --- Punto de Entrada ---
if __name__ == "__main__":
    ft.app(target=main)