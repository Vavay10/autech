import 'package:flutter/material.dart';
import 'modules/learning_path_screen.dart'; // La vista que pediste

class MainNavigation extends StatefulWidget {
  @override
  _MainNavigationState createState() => _MainNavigationState();
}

class _MainNavigationState extends State<MainNavigation> {
  int _selectedIndex = 2; // Empezamos en el centro (Aprendizaje)

  // Lista de las 5 pantallas de los módulos
  final List<Widget> _pages = [
    Center(child: Text("Inicio")),
    Center(child: Text("Autómatas")),
    LearningPathScreen(), // Módulo de Aprendizaje
    Center(child: Text("Estadísticas")),
    Center(child: Text("Ajustes")),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _pages[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: (index) => setState(() => _selectedIndex = index),
        type: BottomNavigationBarType.fixed,
        backgroundColor: const Color(0xFF1E1E1E), // Fondo oscuro del menú
        selectedItemColor: Colors.cyan,
        unselectedItemColor: Colors.grey,
        showSelectedLabels: false,
        showUnselectedLabels: false,
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home_outlined), label: ""),
          BottomNavigationBarItem(icon: Icon(Icons.memory), label: ""),
          BottomNavigationBarItem(icon: Icon(Icons.psychology), label: ""),
          BottomNavigationBarItem(icon: Icon(Icons.bar_chart), label: ""),
          BottomNavigationBarItem(icon: Icon(Icons.more_horiz), label: ""),
        ],
      ),
    );
  }
}