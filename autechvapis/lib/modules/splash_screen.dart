import 'package:flutter/material.dart';
import 'dart:async';
import 'auth_screen.dart'; // importar login
import '../main.dart'; // Asegúrate de que apunte a donde está MainContainer

class SplashScreen extends StatefulWidget {
  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    // Temporizador de 3 segundos antes de navegar a la principal
   // Dentro del Timer en initState:
Timer(Duration(seconds: 3), () {
  Navigator.pushReplacement(
    context,
    MaterialPageRoute(builder: (context) => AuthScreen()),
  );
});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF222831), // El tono oscuro de tu imagen
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Círculo con el logo (puedes usar un Icon o una imagen)
            Container(
              padding: EdgeInsets.all(20),
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(color: Colors.cyan, width: 4),
                boxShadow: [
                  BoxShadow(
                    color: Colors.cyan.withOpacity(0.5),
                    blurRadius: 20,
                    spreadRadius: 5,
                  ),
                ],
              ),
              child: Icon(
                Icons.file_upload_outlined, // Un icono similar al de tu logo
                size: 80,
                color: Colors.cyan,
              ),
            ),
            SizedBox(height: 40),
            Text(
              "AUTECH",
              style: TextStyle(
                color: Colors.white,
                fontSize: 32,
                fontWeight: FontWeight.bold,
                letterSpacing: 4,
              ),
            ),
          ],
        ),
      ),
    );
  }
}