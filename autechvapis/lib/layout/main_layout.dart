import 'package:flutter/material.dart';
import '../modules/dashboard_screen.dart'; // IMPORTANTE: Importar el Dashboard
import '../modules/simulator_screen.dart';
import '../modules/learning_path_screen.dart';
import '../modules/stats_screen.dart';
import '../modules/settings_screen.dart';

class MainLayout extends StatefulWidget {
  @override
  _MainLayoutState createState() => _MainLayoutState();
}

class _MainLayoutState extends State<MainLayout> {
  // 1. Asegúrate de que empiece en el índice 0
  int _selectedIndex = 0; 

  // 2. La lista debe tener el DashboardScreen() primero
// En lib/layout/main_layout.dart
final List<Widget> _pages = [
  DashboardScreen(),    // 0: Classroom
  SimulatorScreen(),    // 1: Simuladores (AF, AP, MT)
  LearningPathScreen(), // 2: Unidades de Aprendizaje
  StatsScreen(),        // 3: Métricas Maestro/Alumno
  SettingsScreen(),     // 4: AJUSTES (Nuevo)
];
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // Usamos IndexedStack para que no se recargue el dashboard cada vez
      body: IndexedStack(
        index: _selectedIndex,
        children: _pages,
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: (index) {
          setState(() {
            _selectedIndex = index;
          });
        },
        type: BottomNavigationBarType.fixed,
        selectedItemColor: Colors.cyan,
        unselectedItemColor: Colors.grey,
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: "Inicio"),
          BottomNavigationBarItem(icon: Icon(Icons.memory), label: "Simulador"),
          BottomNavigationBarItem(icon: Icon(Icons.psychology), label: "Aprendizaje"),
          BottomNavigationBarItem(icon: Icon(Icons.bar_chart), label: "Métricas"),
          BottomNavigationBarItem(icon: Icon(Icons.settings), label: "Ajustes"),
        ],
      ),
    );
  }
}