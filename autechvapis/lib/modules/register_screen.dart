import 'package:flutter/material.dart';
import '../theme.dart';

class RegisterScreen extends StatefulWidget {
  @override
  _RegisterScreenState createState() => _RegisterScreenState();
}

class _RegisterScreenState extends State<RegisterScreen> {
  // Controladores para capturar los datos
  final TextEditingController _nameCtrl = TextEditingController();
  final TextEditingController _emailCtrl = TextEditingController();
  final TextEditingController _passCtrl = TextEditingController();
  final TextEditingController _confirmPassCtrl = TextEditingController();

  // Estados para ocultar/mostrar contraseñas
  bool _obscurePass = true;
  bool _obscureConfirm = true;

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
            Text(
              "Ingresa tus datos",
              style: TextStyle(
                fontSize: 22, 
                fontWeight: FontWeight.bold, 
                color: isDark ? Colors.white : Colors.black
              ),
            ),
            const SizedBox(height: 30),
            
            // Campos de texto
            _buildField("Nombre", _nameCtrl, isDark),
            const SizedBox(height: 15),
            _buildField("Correo electrónico", _emailCtrl, isDark, keyboardType: TextInputType.emailAddress),
            const SizedBox(height: 15),
            _buildField(
              "Contraseña", 
              _passCtrl, 
              isDark, 
              isPass: true, 
              obscure: _obscurePass,
              onToggle: () => setState(() => _obscurePass = !_obscurePass)
            ),
            const SizedBox(height: 15),
            _buildField(
              "Repetir contraseña", 
              _confirmPassCtrl, 
              isDark, 
              isPass: true, 
              obscure: _obscureConfirm,
              onToggle: () => setState(() => _obscureConfirm = !_obscureConfirm)
            ),
            
            const SizedBox(height: 30),
            
            // Botón Principal
            ElevatedButton(
              onPressed: () {
                // Aquí iría la lógica de registro hacia el backend
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: AutechColors.primaryCyan,
                minimumSize: const Size(double.infinity, 55),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
              ),
              child: const Text(
                "REGISTRARSE", 
                style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16)
              ),
            ),
            
            const SizedBox(height: 30),
            const Row(
              children: [
                Expanded(child: Divider(color: Colors.grey)),
                Padding(padding: EdgeInsets.symmetric(horizontal: 10), child: Text("ó", style: TextStyle(color: Colors.grey))),
                Expanded(child: Divider(color: Colors.grey)),
              ],
            ),
            const SizedBox(height: 30),
            
            // Botones Sociales
            _socialBtn("REGISTRARSE CON GOOGLE", Icons.g_mobiledata, isDark),
            const SizedBox(height: 12),
            _socialBtn("REGISTRARSE CON FACEBOOK", Icons.facebook, isDark),
            
            const SizedBox(height: 30),
            Text(
              "Al registrarte en AUTECH, aceptas nuestros Términos y Condiciones, al igual que la Política de Privacidad.",
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.grey.shade600, fontSize: 11),
            ),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }

  // Widget reutilizable para los campos de entrada
  Widget _buildField(String hint, TextEditingController ctrl, bool isDark, {
    bool isPass = false, 
    bool obscure = false, 
    VoidCallback? onToggle,
    TextInputType keyboardType = TextInputType.text
  }) {
    return TextField(
      controller: ctrl,
      obscureText: obscure,
      keyboardType: keyboardType,
      style: TextStyle(color: isDark ? Colors.white : Colors.black),
      decoration: InputDecoration(
        hintText: hint,
        hintStyle: const TextStyle(color: Colors.grey),
        filled: true,
        fillColor: isDark ? Colors.white10 : Colors.grey.shade200,
        suffixIcon: isPass ? IconButton(
          icon: Icon(obscure ? Icons.visibility_off : Icons.visibility, color: Colors.grey),
          onPressed: onToggle,
        ) : null,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(15), 
          borderSide: BorderSide.none
        ),
      ),
    );
  }

  // Widget reutilizable para botones sociales
  Widget _socialBtn(String text, IconData icon, bool isDark) {
    return OutlinedButton.icon(
      onPressed: () {},
      icon: Icon(icon, color: isDark ? Colors.white : Colors.black, size: 28),
      label: Text(
        text, 
        style: TextStyle(color: isDark ? Colors.white : Colors.black, fontSize: 13, fontWeight: FontWeight.bold)
      ),
      style: OutlinedButton.styleFrom(
        minimumSize: const Size(double.infinity, 55),
        side: const BorderSide(color: Colors.grey),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      ),
    );
  }
}