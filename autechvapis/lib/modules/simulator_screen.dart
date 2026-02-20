import 'package:flutter/material.dart';
import '../theme.dart';
import 'pda_screen.dart';
import 'regex_screen.dart';
import 'turing_screen.dart';

class SimulatorScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    bool isDark = Theme.of(context).brightness == Brightness.dark;

    return Scaffold(
      backgroundColor: isDark ? AutechColors.darkBg : AutechColors.lightBg,
      appBar: AppBar(
        title: Text("HERRAMIENTAS", 
          style: TextStyle(color: isDark ? Colors.white : Colors.black, fontWeight: FontWeight.bold)),
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            const SizedBox(height: 20),
            Center(
              child: Icon(Icons.memory, size: 80, color: isDark ? Colors.greenAccent : Colors.blue),
            ),
            const SizedBox(height: 30),
            
            // Módulo de Expresiones Regulares
            _buildToolCard(
              context,
              title: "Expresiones Regulares",
              desc: "Convierte Regex a AFN/AFD y visualiza sus transiciones.",
              icon: Icons.code,
              screen: RegexScreen(),
              isDark: isDark,
            ),
            
            // Módulo de Autómatas de Pila
            _buildToolCard(
              context,
              title: "Autómatas de Pila (PDA)",
              desc: "Simula el procesamiento de cadenas con memoria de pila.",
              icon: Icons.layers, // Sustituye al error de Settings_input_component
              screen: PdaScreen(),
              isDark: isDark,
            ),
            
            // Módulo de Máquina de Turing
            _buildToolCard(
              context,
              title: "Máquina de Turing",
              desc: "Simulador completo con visualización de cinta y estados.",
              icon: Icons.settings_ethernet,
              screen: TuringScreen(),
              isDark: isDark,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildToolCard(BuildContext context, {
    required String title, 
    required String desc, 
    required IconData icon, 
    required Widget screen,
    required bool isDark
  }) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: isDark ? Colors.white.withOpacity(0.05) : Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: isDark ? Colors.white12 : Colors.grey.shade300),
      ),
      child: Column(
        children: [
          Row(
            children: [
              Icon(icon, size: 40, color: AutechColors.primaryCyan),
              const SizedBox(width: 15),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: TextStyle(
                      fontWeight: FontWeight.bold, color: isDark ? Colors.white : Colors.black)),
                    Text(desc, style: const TextStyle(color: Colors.grey, fontSize: 12)),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 15),
          ElevatedButton(
            onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (context) => screen)),
            style: ElevatedButton.styleFrom(
              backgroundColor: AutechColors.primaryCyan,
              minimumSize: const Size(double.infinity, 40),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
            child: const Text("COMENZAR", style: TextStyle(color: Colors.white)),
          )
        ],
      ),
    );
  }
}