import 'package:flutter/material.dart';
import '../theme.dart';
import 'unit_nodes_screen.dart'; // Importa la pantalla del camino de nodos

class LearningPathScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Stack(
        children: [
          SingleChildScrollView(
            child: Column(
              children: [
                _buildHeader(),
                const SizedBox(height: 10),
                _buildUnitCard(
                  context,
                  "UNIDAD I",
                  "Unidad dedicada al aprendizaje de autómatas finitos y lenguajes regulares.",
                  'https://cdn-icons-png.flaticon.com/512/2103/2103633.png', // Icono red/nodos
                  Colors.greenAccent,
                ),
                _buildUnitCard(
                  context,
                  "UNIDAD II",
                  "Unidad dedicada al aprendizaje de expresiones regulares a partir de un autómata.",
                  'https://cdn-icons-png.flaticon.com/512/1491/1491214.png', // Icono cerebro/engrane
                  Colors.pinkAccent.shade100,
                ),
                const SizedBox(height: 100), // Espacio inferior para los botones
              ],
            ),
          ),

          // --- Botón de Basura (Izquierda) ---
          Positioned(
            bottom: 25,
            left: 20,
            child: _buildActionButton(
              context: context,
              icon: Icons.delete_outline,
              color: Colors.redAccent,
              title: "Confirmar eliminación",
              message: "¿Está de acuerdo con que se ejecute la operación asociada al botón (entrar a interfaz de eliminar usuario)?",
            ),
          ),

          // --- Botón de Más (Derecha) ---
          Positioned(
            bottom: 25,
            right: 20,
            child: _buildActionButton(
              context: context,
              icon: Icons.add,
              color: Colors.pinkAccent,
              title: "Nueva UA",
              message: "¿Está de acuerdo con que se ejecute la operación asociada al botón (entrar a interfaz para agregar una nueva UA)?",
            ),
          ),
        ],
      ),
    );
  }

  // --- Widgets Auxiliares ---

  Widget _buildHeader() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 40, horizontal: 20),
      color: const Color(0xFFF8BBD0), // Rosa pastel de la imagen
      child: Column(
        children: [
          const SizedBox(height: 20),
          const Text(
            "UNIDADES DE\nAPRENDIZAJE",
            textAlign: TextAlign.left,
            style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: Color(0xFF37474F)),
          ),
          const SizedBox(height: 20),
          Image.network('https://cdn-icons-png.flaticon.com/512/1043/1043236.png', height: 120), // Icono de libros y birrete
        ],
      ),
    );
  }

  Widget _buildUnitCard(BuildContext context, String title, String desc, String iconUrl, Color accent) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.grey.shade200),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 10, offset: const Offset(0, 5))],
      ),
      child: Column(
        children: [
          Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: TextStyle(color: Colors.pink.shade400, fontWeight: FontWeight.bold, fontSize: 18)),
                    const SizedBox(height: 10),
                    Text(desc, style: const TextStyle(color: Colors.grey, fontSize: 13)),
                  ],
                ),
              ),
              Image.network(iconUrl, height: 70), // Imagen circular decorativa
            ],
          ),
          const SizedBox(height: 15),
          OutlinedButton(
            style: OutlinedButton.styleFrom(
              minimumSize: const Size(double.infinity, 45),
              side: BorderSide(color: Colors.pink.shade200),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
            ),
            onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (context) => UnitNodesScreen(title: title))),
            child: const Text("VER UA", style: TextStyle(color: Colors.pinkAccent, fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
  }

  Widget _buildActionButton({required BuildContext context, required IconData icon, required Color color, required String title, required String message}) {
    return GestureDetector(
      onTap: () => _showConfirmDialog(context, title, message),
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(15), boxShadow: const [BoxShadow(color: Colors.black26, blurRadius: 5)]),
        child: Icon(icon, color: Colors.white, size: 30),
      ),
    );
  }

  void _showConfirmDialog(BuildContext context, String title, String msg) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: Text(msg),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text("CANCELAR")),
          TextButton(onPressed: () => Navigator.pop(context), child: const Text("ACEPTAR")),
        ],
      ),
    );
  }
}