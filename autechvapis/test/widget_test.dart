import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
// Importamos tu SplashScreen para que el test pueda iniciar la app
import 'package:autechvapis/modules/splash_screen.dart';

void main() {
  testWidgets('Prueba de humo: Carga inicial de AUTECH', (WidgetTester tester) async {
    // Envolvemos tu pantalla inicial en un MaterialApp para el entorno de pruebas
    await tester.pumpWidget(MaterialApp(
      home: SplashScreen(),
    ));

    // Verificamos que el SplashScreen se haya renderizado correctamente
    expect(find.byType(SplashScreen), findsOneWidget);
  });
}