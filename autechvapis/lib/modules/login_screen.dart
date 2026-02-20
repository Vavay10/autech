import 'package:flutter/material.dart';
import '../theme.dart';
import '../main.dart';

class LoginScreen extends StatefulWidget {
  @override
  _LoginScreenState createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  bool isSwitched = true;

  @override
  Widget build(BuildContext context) {
    bool isDark = Theme.of(context).brightness == Brightness.dark;

    return Scaffold(
      backgroundColor: isDark ? AutechColors.darkBg : AutechColors.lightBg,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(icon: Icon(Icons.arrow_back, color: isDark ? Colors.white : Colors.black), onPressed: () => Navigator.pop(context)),
      ),
      body: Padding(
        padding: const EdgeInsets.all(30.0),
        child: Column(
          children: [
            Text("Ingresa tus datos", style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: isDark ? Colors.white : Colors.black)),
            const SizedBox(height: 30),
            _buildTextField("Correo, teléfono o usuario", isDark),
            const SizedBox(height: 15),
            _buildTextField("Contraseña", isDark, isPassword: true),
            const SizedBox(height: 20),
            // Switch decorativo
            Container(
              padding: EdgeInsets.symmetric(horizontal: 15, vertical: 5),
              decoration: BoxDecoration(color: Colors.blueGrey.withOpacity(0.2), borderRadius: BorderRadius.circular(15)),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text("**************** ******************", style: TextStyle(color: Colors.grey)),
                  Switch(value: isSwitched, onChanged: (v) => setState(() => isSwitched = v), activeColor: AutechColors.primaryCyan),
                ],
              ),
            ),
            const SizedBox(height: 20),
            TextButton(onPressed: () {}, child: Text("OLVIDÉ MI CONTRASEÑA", style: TextStyle(color: AutechColors.primaryCyan, fontWeight: FontWeight.bold))),
            const Spacer(),
            ElevatedButton(
              onPressed: () => Navigator.pushReplacement(context, MaterialPageRoute(builder: (context) => MainContainer())),
              style: ElevatedButton.styleFrom(
                backgroundColor: AutechColors.primaryCyan,
                minimumSize: Size(double.infinity, 55),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
              ),
              child: Text("INICIAR", style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTextField(String hint, bool isDark, {bool isPassword = false}) {
    return TextField(
      obscureText: isPassword,
      decoration: InputDecoration(
        hintText: hint,
        hintStyle: TextStyle(color: Colors.grey),
        filled: true,
        fillColor: isDark ? Colors.white10 : Colors.white,
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(15), borderSide: BorderSide.none),
      ),
    );
  }
}