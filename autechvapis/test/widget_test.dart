import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:autechvapis/modules/login_screen.dart';
import 'package:autechvapis/layout/main_layout.dart';
import 'package:autechvapis/theme.dart';

void main() {
  testWidgets('Flujo de Autenticación: Login a MainLayout', (WidgetTester tester) async {
    // 1. Cargamos el LoginScreen dentro de un MaterialApp
    // Usamos un tema base para evitar errores con AutechColors
    await tester.pumpWidget(MaterialApp(
      theme: ThemeData(primaryColor: Colors.cyan),
      home: LoginScreen(),
    ));

    // 2. Verificamos que el título del Login aparezca
    expect(find.text("Ingresa tus datos"), findsOneWidget);

    // 3. Verificamos que los campos de texto existan
    expect(find.byType(TextField), findsNWidgets(2));

    // 4. Simulamos el tap en el botón de INICIAR
    final btnIniciar = find.text("INICIAR");
    expect(btnIniciar, findsOneWidget);
    
    await tester.tap(btnIniciar);
    // pumpAndSettle espera a que todas las animaciones de transición terminen
    await tester.pumpAndSettle();

    // 5. Verificamos que después del login estemos en el MainLayout
    expect(find.byType(MainLayout), findsOneWidget);
  });
}