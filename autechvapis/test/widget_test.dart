import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:autechvapis/main.dart';
import 'package:autechvapis/modules/splash_screen.dart';

void main() {
  testWidgets('Prueba de carga inicial de AUTECH', (WidgetTester tester) async {
    // 1. Cargamos la aplicación tal cual la tienes en el main.dart
    // Nota: Como en tu main usas MaterialApp directamente, aquí lo replicamos
    await tester.pumpWidget(MaterialApp(
      home: SplashScreen(),
    ));

    // 2. Verificamos que el SplashScreen esté presente
    // (Asegúrate de que en tu SplashScreen haya algún texto o widget identificable)
    expect(find.byType(SplashScreen), findsOneWidget);

    // 3. Si quieres probar el contenedor principal directamente:
    await tester.pumpWidget(MaterialApp(
      home: MainContainer(),
    ));

    // Verificamos que aparezca el título de tu AppBar móvil
    expect(find.text("Suite de Teoría de la Computación"), findsOneWidget);

    // Verificamos que los íconos del NavigationRail (o menú) estén ahí
    expect(find.byIcon(Icons.hub_outlined), findsOneWidget);
    expect(find.byIcon(Icons.layers_outlined), findsOneWidget);
    expect(find.byIcon(Icons.memory_outlined), findsOneWidget);
  });
}