// lib/modules/forgot_password_screen.dart
import 'package:flutter/material.dart';
import '../theme.dart';

class ForgotPasswordScreen extends StatelessWidget {
  const ForgotPasswordScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(backgroundColor: Colors.transparent, elevation: 0,
        leading: IconButton(icon: const Icon(Icons.arrow_back, color: AppColors.textPrimary), onPressed: () => Navigator.pop(context))),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 28),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 20),
            Container(padding: const EdgeInsets.all(14), decoration: BoxDecoration(color: AppColors.primary.withOpacity(0.1), borderRadius: BorderRadius.circular(14)), child: const Icon(Icons.lock_reset, color: AppColors.primary, size: 28)),
            const SizedBox(height: 20),
            const Text('¿Olvidaste tu contraseña?', style: TextStyle(fontSize: 24, fontWeight: FontWeight.w800)),
            const SizedBox(height: 8),
            const Text('Ingresa tu correo y te enviaremos un enlace de restablecimiento.', style: TextStyle(color: AppColors.textSecondary, fontSize: 14, height: 1.5)),
            const SizedBox(height: 32),
            const TextField(keyboardType: TextInputType.emailAddress, decoration: InputDecoration(labelText: 'Correo electrónico', prefixIcon: Icon(Icons.email_outlined, size: 18))),
            const Spacer(),
            ElevatedButton(
              onPressed: () {},
              style: ElevatedButton.styleFrom(minimumSize: const Size(double.infinity, 52)),
              child: const Text('ENVIAR ENLACE', style: TextStyle(fontWeight: FontWeight.w700, letterSpacing: 1)),
            ),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }
}
