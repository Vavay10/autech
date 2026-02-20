import 'package:flutter/material.dart';
import 'login_screen.dart';
import '../theme.dart';

class AuthScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    bool isDark = Theme.of(context).brightness == Brightness.dark;

    return Scaffold(
      backgroundColor: isDark ? AutechColors.darkBg : AutechColors.lightBg,
      body: SingleChildScrollView( // Para evitar errores de espacio en pantallas pequeñas
        padding: const EdgeInsets.symmetric(horizontal: 40.0, vertical: 60),
        child: Column(
          children: [
            Align(alignment: Alignment.centerLeft, child: IconButton(icon: Icon(Icons.arrow_back, color: isDark ? Colors.white : Colors.black), onPressed: () {})),
            const SizedBox(height: 20),
            Text("Ingresa tus datos", style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: isDark ? Colors.white : Colors.black)),
            const SizedBox(height: 40),
            // Logo AUTECH
            Container(
              padding: EdgeInsets.all(15),
              decoration: BoxDecoration(shape: BoxShape.circle, border: Border.all(color: AutechColors.primaryCyan, width: 3)),
              child: Icon(Icons.file_upload_outlined, size: 70, color: AutechColors.primaryCyan),
            ),
            const SizedBox(height: 20),
            Text("AUTECH", style: TextStyle(fontSize: 30, fontWeight: FontWeight.bold, color: AutechColors.primaryCyan)),
            Text("Tu app auxiliar para Teoría de\nla computación.", textAlign: TextAlign.center, style: TextStyle(color: Colors.grey)),
            const SizedBox(height: 40),
            
            // Botón Iniciar Sesión (Navega al formulario)
            _buildButton("INICIAR SESIÓN", AutechColors.primaryCyan, Colors.white, () {
              Navigator.push(context, MaterialPageRoute(builder: (context) => LoginScreen()));
            }),
            const SizedBox(height: 12),
            _buildButton("REGISTRARSE", Colors.white, AutechColors.primaryCyan, () {}, isOutlined: true),
            const SizedBox(height: 12),
            _buildSocialButton("INICIAR SESIÓN CON GOOGLE", "assets/google_logo.png", isDark),
            const SizedBox(height: 12),
            _buildSocialButton("INICIAR SESIÓN CON FACEBOOK", "assets/fb_logo.png", isDark),
          ],
        ),
      ),
    );
  }

  Widget _buildButton(String text, Color bg, Color textCol, VoidCallback tap, {bool isOutlined = false}) {
    return ElevatedButton(
      onPressed: tap,
      style: ElevatedButton.styleFrom(
        backgroundColor: isOutlined ? Colors.transparent : bg,
        side: isOutlined ? BorderSide(color: AutechColors.primaryCyan, width: 2) : null,
        minimumSize: Size(double.infinity, 50),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      ),
      child: Text(text, style: TextStyle(color: isOutlined ? AutechColors.primaryCyan : textCol, fontWeight: FontWeight.bold)),
    );
  }

  Widget _buildSocialButton(String text, String asset, bool isDark) {
    return OutlinedButton.icon(
      onPressed: () {},
      icon: Icon(Icons.account_circle, size: 24), // Reemplazar con Image.asset cuando tengas los logos
      label: Text(text, style: TextStyle(color: isDark ? Colors.white : Colors.black, fontSize: 12)),
      style: OutlinedButton.styleFrom(
        minimumSize: Size(double.infinity, 50),
        side: BorderSide(color: Colors.grey.shade400),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      ),
    );
  }
}