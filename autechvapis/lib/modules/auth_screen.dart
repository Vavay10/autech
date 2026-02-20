import 'package:flutter/material.dart';
import 'login_screen.dart';
import 'register_screen.dart';

import '../theme.dart';

class AuthScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    bool isDark = Theme.of(context).brightness == Brightness.dark;

    return Scaffold(
      backgroundColor: isDark ? AutechColors.darkBg : AutechColors.lightBg,
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 40.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Logo AUTECH
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                shape: BoxShape.circle, 
                border: Border.all(color: AutechColors.primaryCyan, width: 3)
              ),
              child: const Icon(Icons.file_upload_outlined, size: 80, color: AutechColors.primaryCyan),
            ),
            const SizedBox(height: 30),
            Text(
              "AUTECH",
              style: TextStyle(
                fontSize: 35, 
                fontWeight: FontWeight.bold, 
                color: isDark ? Colors.white : Colors.black,
                letterSpacing: 2,
              ),
            ),
            const Text(
              "Tu app auxiliar para\nTeoría de la computación.",
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.grey, fontSize: 16),
            ),
            const SizedBox(height: 60),
            
            // Botón Iniciar Sesión
            ElevatedButton(
              onPressed: () {
                Navigator.push(context, MaterialPageRoute(builder: (context) => LoginScreen()));
              }, 
              style: ElevatedButton.styleFrom(
                backgroundColor: AutechColors.primaryCyan,
                minimumSize: const Size(double.infinity, 50),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
              ),
              child: const Text("INICIAR SESIÓN", style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
            ),
            const SizedBox(height: 15),
            
            // Botón Registrarse
// Busca el botón de Registrarse en auth_screen.dart y añade la navegación:
OutlinedButton(
  onPressed: () {
    Navigator.push(
      context, 
      MaterialPageRoute(builder: (context) => RegisterScreen())
    );
  },
              style: OutlinedButton.styleFrom(
                side: const BorderSide(color: AutechColors.primaryCyan, width: 2),
                minimumSize: const Size(double.infinity, 50),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
              ),
              child: const Text("REGISTRARSE", style: TextStyle(color: AutechColors.primaryCyan, fontWeight: FontWeight.bold)),
            ),
          ],
        ),
      ),
    );
  }
}