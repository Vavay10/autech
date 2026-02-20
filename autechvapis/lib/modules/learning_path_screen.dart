import 'package:flutter/material.dart';
import '../theme.dart';

class LearningPathScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Fondo decorativo (Cielo/Espacio)
          Container(color: const Color(0xFF222831)),
          
          SingleChildScrollView(
            child: Column(
              children: [
                const SizedBox(height: 50),
                _buildHeader(),
                const SizedBox(height: 30),
                _buildNode(Icons.menu_book, "Nivel 1", Alignment.centerLeft, true),
                _buildPathLine(true),
                _buildNode(Icons.extension, "Nivel 2", Alignment.centerRight, false),
                _buildPathLine(false),
                _buildNode(Icons.code, "Nivel 3", Alignment.center, false),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 20),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.orangeAccent,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: const [
              Text("NIVEL 1", style: TextStyle(fontWeight: FontWeight.bold, color: Colors.white)),
              Text("Introducción teórica", style: TextStyle(color: Colors.white)),
            ],
          ),
          const Icon(Icons.star, color: Colors.white, size: 30),
        ],
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

  Widget _buildPathLine(bool toRight) {
    return Container(
      height: 40,
      width: 2,
      color: Colors.white24,
    );
  }
}