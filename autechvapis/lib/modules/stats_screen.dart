// lib/modules/stats_screen.dart
import 'package:flutter/material.dart';

class StatsScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Métricas y Gestión")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _buildRoleButton(context, "Panel de Maestro", Icons.admin_panel_settings, Colors.indigo, _showTeacherMenu),
            const SizedBox(height: 20),
            _buildRoleButton(context, "Panel de Alumno", Icons.person, Colors.cyan, _showStudentMenu),
          ],
        ),
      ),
    );
  }

  Widget _buildRoleButton(BuildContext context, String label, IconData icon, Color color, Function(BuildContext) action) {
    return ElevatedButton.icon(
      style: ElevatedButton.styleFrom(
        backgroundColor: color,
        minimumSize: const Size(280, 60),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      ),
      onPressed: () => action(context),
      icon: Icon(icon, color: Colors.white),
      label: Text(label, style: const TextStyle(color: Colors.white, fontSize: 16)),
    );
  }

  // --- MENU MAESTRO ---
  void _showTeacherMenu(BuildContext context) {
    _showModal(context, "Funciones de Docente", [
      "Visualización de desempeño de alumno",
      "Gestión de clases",
      "Asignar ejercicios/tareas",
      "Habilitar aprendizaje guiado",
      "Reportes de grupo",
    ]);
  }

  // --- MENU ALUMNO ---
  void _showStudentMenu(BuildContext context) {
    _showModal(context, "Métricas de Usuario", [
      "Eliminar clase",
      "Ver calificaciones",
      "Gráficas de uso y aprendizaje",
      "Reporte de errores y aciertos",
      "Progreso autónomo (Simulador)",
    ]);
  }

  void _showModal(BuildContext context, String title, List<String> options) {
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
      builder: (context) => Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(title, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const Divider(),
            ...options.map((opt) => ListTile(
              leading: const Icon(Icons.check_circle_outline, color: Colors.cyan),
              title: Text(opt),
              onTap: () {},
            )).toList(),
          ],
        ),
      ),
    );
  }
}