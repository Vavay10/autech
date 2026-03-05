import 'package:flutter/material.dart';

class UnitNodesScreen extends StatelessWidget {
  final String title;
  UnitNodesScreen({required this.title});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF222831), // Fondo oscuro estilo espacio/camino
      appBar: AppBar(title: Text(title), backgroundColor: Colors.transparent, elevation: 0),
      body: SingleChildScrollView(
        child: Column(
          children: [
            const SizedBox(height: 30),
            _buildNode(Icons.menu_book, "Conceptos Base", Alignment.centerLeft, true),
            _buildLine(),
            _buildNode(Icons.extension, "Ejemplos", Alignment.centerRight, false),
            _buildLine(),
            _buildNode(Icons.code, "Simulación", Alignment.center, false),
            const SizedBox(height: 50),
          ],
        ),
      ),
    );
  }

  Widget _buildNode(IconData icon, String label, Alignment align, bool active) {
    return Align(
      alignment: align,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 60, vertical: 10),
        child: Column(
          children: [
            Container(
              width: 70, height: 70,
              decoration: BoxDecoration(
                color: active ? Colors.orange : Colors.grey.shade700,
                shape: BoxShape.circle,
                border: Border.all(color: Colors.white, width: 4),
              ),
              child: Icon(icon, color: Colors.white, size: 35),
            ),
            const SizedBox(height: 8),
            Text(label, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
          ],
        ),
      ),
    );
  }

  Widget _buildLine() {
    return Container(height: 50, width: 3, color: Colors.white12);
  }
}