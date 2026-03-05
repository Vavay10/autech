// lib/modules/dashboard_screen.dart
import 'package:flutter/material.dart';

class DashboardScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF5F5F5),
      appBar: AppBar(
        title: const Text("Mis Clases", style: TextStyle(color: Colors.black)),
        backgroundColor: Colors.white,
        elevation: 0,
      ),
      body: ListView(
        padding: const EdgeInsets.all(15),
        children: [
          _buildClassCard(
            context,
            "Teoría de la Computación",
            "Prof. Alan Turing",
            "3 Ejercicios activos",
            Colors.blue.shade700,
          ),
          _buildClassCard(
            context,
            "Compiladores",
            "Prof. Grace Hopper",
            "Tarea pendiente: AFN a AFD",
            Colors.teal.shade600,
          ),
        ],
      ),
    );
  }

  Widget _buildClassCard(BuildContext context, String materia, String prof, String info, Color color) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      margin: const EdgeInsets.only(bottom: 15),
      child: InkWell(
        onTap: () {
          // Aquí navegarías al detalle: Ejercicios, calificaciones, etc.
        },
        child: Column(
          children: [
            Container(
              height: 100,
              width: double.infinity,
              decoration: BoxDecoration(
                color: color,
                borderRadius: const BorderRadius.vertical(top: Radius.circular(15)),
              ),
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(materia, style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
                  Text(prof, style: const TextStyle(color: Colors.white70)),
                ],
              ),
            ),
            ListTile(
              title: Text(info, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w500)),
              trailing: const Icon(Icons.arrow_forward_ios, size: 16),
            ),
          ],
        ),
      ),
    );
  }
}