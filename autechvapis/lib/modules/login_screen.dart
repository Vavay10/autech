import 'package:flutter/material.dart';
import '../theme.dart';
import '../main.dart';
import 'forgot_password_screen.dart';

class LoginScreen extends StatefulWidget {
  @override
  _LoginScreenState createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final TextEditingController _userCtrl = TextEditingController();
  final TextEditingController _passCtrl = TextEditingController();
  bool _obscure = true;

  @override
  Widget build(BuildContext context) {
    bool isDark = Theme.of(context).brightness == Brightness.dark;

    return Scaffold(
      backgroundColor: isDark ? AutechColors.darkBg : AutechColors.lightBg,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: isDark ? Colors.white : Colors.black),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 30),
        child: Column(
          children: [
            Text("Ingresa tus datos", 
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: isDark ? Colors.white : Colors.black)),
            const SizedBox(height: 30),
            _buildField("Correo, teléfono o usuario", _userCtrl, isDark),
            const SizedBox(height: 15),
            _buildField("Contraseña", _passCtrl, isDark, isPass: true),
            const SizedBox(height: 25),
            
            ElevatedButton(
              onPressed: () => Navigator.pushReplacement(context, MaterialPageRoute(builder: (context) => MainContainer())),
              style: ElevatedButton.styleFrom(
                backgroundColor: AutechColors.primaryCyan,
                minimumSize: const Size(double.infinity, 55),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
              ),
              child: const Text("INICIAR", style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
            ),
            const SizedBox(height: 15),
            TextButton(
              onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (context) => ForgotPasswordScreen())),
              child: const Text("OLVIDÉ MI CONTRASEÑA", style: TextStyle(color: AutechColors.primaryCyan, fontWeight: FontWeight.bold)),
            ),
            const SizedBox(height: 30),
            
            // Botones sociales movidos aquí según tu petición
            _socialBtn("INICIAR SESIÓN CON GOOGLE", Icons.g_mobiledata, isDark),
            const SizedBox(height: 12),
            _socialBtn("INICIAR SESIÓN CON FACEBOOK", Icons.facebook, isDark),
            
            const SizedBox(height: 30),
            const Text(
              "Al iniciar sesión en AUTECH, aceptas nuestros Términos y Condiciones y la Política de Privacidad.",
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.grey, fontSize: 11),
            )
          ],
        ),
      ),
    );
  }

  Widget _buildField(String hint, TextEditingController ctrl, bool isDark, {bool isPass = false}) {
    return TextField(
      controller: ctrl,
      obscureText: isPass ? _obscure : false,
      style: TextStyle(color: isDark ? Colors.white : Colors.black),
      decoration: InputDecoration(
        hintText: hint,
        hintStyle: const TextStyle(color: Colors.grey),
        filled: true,
        fillColor: isDark ? Colors.white10 : Colors.white70,
        suffixIcon: isPass ? IconButton(
          icon: Icon(_obscure ? Icons.visibility_off : Icons.visibility, color: Colors.cyan),
          onPressed: () => setState(() => _obscure = !_obscure),
        ) : null,
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(15), borderSide: BorderSide.none),
      ),
    );
  }

  Widget _socialBtn(String text, IconData icon, bool isDark) {
    return OutlinedButton.icon(
      onPressed: () {},
      icon: Icon(icon, color: isDark ? Colors.white : Colors.black),
      label: Text(text, style: TextStyle(color: isDark ? Colors.white : Colors.black)),
      style: OutlinedButton.styleFrom(
        minimumSize: const Size(double.infinity, 50),
        side: const BorderSide(color: Colors.grey),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      ),
    );
  }
}