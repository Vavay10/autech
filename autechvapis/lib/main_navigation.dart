import 'package:flutter/material.dart';
import 'modules/learning_path_screen.dart'; 
import 'modules/dashboard_screen.dart';
import 'modules/simulator_screen.dart'; // <--- Importa tu pantalla del simulador

class MainNavigation extends StatefulWidget {
  @override
  _MainNavigationState createState() => _MainNavigationState();
}

class _MainNavigationState extends State<MainNavigation> {
  int _selectedIndex = 0; // Inicia en Dashboard (Casa)

  final List<Widget> _pages = [
    DashboardScreen(),      // Índice 0: Inicio
    SimulatorScreen(),      // Índice 1: Simulador (Cargará tu nuevo archivo)
    LearningPathScreen(),   // Índice 2: Aprendizaje
    const Center(child: Text("Estadísticas Detalladas")), // Índice 3
    const Center(child: Text("Ajustes")),                // Índice 4
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _selectedIndex,
        children: _pages,
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: (index) => setState(() => _selectedIndex = index),
        type: BottomNavigationBarType.fixed,
        backgroundColor: const Color(0xFF1E1E1E),
        selectedItemColor: Colors.cyan,
        unselectedItemColor: Colors.grey,
        showSelectedLabels: false,
        showUnselectedLabels: false,
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home_outlined), label: "Inicio"),
          BottomNavigationBarItem(icon: Icon(Icons.memory), label: "Simulador"),
          BottomNavigationBarItem(icon: Icon(Icons.psychology), label: "Aprendizaje"),
          BottomNavigationBarItem(icon: Icon(Icons.bar_chart), label: "Métricas"),
          BottomNavigationBarItem(icon: Icon(Icons.more_horiz), label: "Ajustes"),
        ],
      ),
    );
  }
}