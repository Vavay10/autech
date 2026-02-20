import 'package:flutter/material.dart';
import '../modules/learning_path_screen.dart';
import '../theme.dart';

class MainLayout extends StatefulWidget {
  @override
  _MainLayoutState createState() => _MainLayoutState();
}

class _MainLayoutState extends State<MainLayout> {
  // El índice 2 corresponde al Módulo de Aprendizaje (icono central)
  int _selectedIndex = 2; 

  final List<Widget> _pages = [
    const Center(child: Text("Módulo 1: Inicio")),
    const Center(child: Text("Módulo 2: Simulador Clásico")),
    LearningPathScreen(), // Tu nueva vista de camino de aprendizaje
    const Center(child: Text("Módulo 4: Estadísticas")),
    const Center(child: Text("Módulo 5: Configuración")),
  ];

  @override
  Widget build(BuildContext context) {
    bool isDark = Theme.of(context).brightness == Brightness.dark;

    return Scaffold(
      body: _pages[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: (index) => setState(() => _selectedIndex = index),
        type: BottomNavigationBarType.fixed,
        backgroundColor: isDark ? const Color(0xFF1A1A1A) : Colors.white,
        selectedItemColor: AutechColors.primaryCyan,
        unselectedItemColor: Colors.grey,
        showSelectedLabels: false,
        showUnselectedLabels: false,
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home_filled), label: ""),
          BottomNavigationBarItem(icon: Icon(Icons.psychology), label: ""),
          BottomNavigationBarItem(icon: Icon(Icons.auto_awesome_motion), label: ""),
          BottomNavigationBarItem(icon: Icon(Icons.bar_chart_rounded), label: ""),
          BottomNavigationBarItem(icon: Icon(Icons.more_horiz), label: ""),
        ],
      ),
    );
  }
}