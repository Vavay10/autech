import 'package:flutter/material.dart';
import 'main_navigation.dart'; // Importa el archivo de navegación
import 'modules/splash_screen.dart'; 

void main() => runApp(MaterialApp(
      debugShowCheckedModeBanner: false,
      themeMode: ThemeMode.dark, 
      darkTheme: ThemeData.dark(),
      theme: ThemeData.light(),
      // El Splash debe navegar hacia MainNavigation al terminar
      home: SplashScreen(), 
    ));