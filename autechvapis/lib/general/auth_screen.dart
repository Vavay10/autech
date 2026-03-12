// lib/modules/auth_screen.dart
import 'package:flutter/material.dart';
import '../theme.dart';
import 'login_screen.dart';
import 'register_screen.dart';

class AuthScreen extends StatelessWidget {
  const AuthScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 32),
          child: Column(
            children: [
              const Spacer(flex: 2),
              // Logo
              Container(
                padding: const EdgeInsets.all(22),
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: LinearGradient(colors: [AppColors.primary.withOpacity(0.15), AppColors.primary.withOpacity(0.05)]),
                  border: Border.all(color: AppColors.primary, width: 2),
                ),
                child: const Icon(Icons.memory_outlined, size: 60, color: AppColors.primary),
              ),
              const SizedBox(height: 24),
              const Text('AUTECH', style: TextStyle(fontSize: 34, fontWeight: FontWeight.w800, letterSpacing: 4, color: AppColors.textPrimary)),
              const SizedBox(height: 8),
              const Text('Tu asistente de Teoría de la Computación', textAlign: TextAlign.center, style: TextStyle(color: AppColors.textSecondary, fontSize: 14)),
              const Spacer(flex: 2),
              // Highlights
              _FeatureRow(icon: Icons.account_tree_outlined, text: 'Construye autómatas gráficamente'),
              const SizedBox(height: 10),
              _FeatureRow(icon: Icons.play_circle_outline, text: 'Simula paso a paso con animaciones'),
              const SizedBox(height: 10),
              _FeatureRow(icon: Icons.school_outlined, text: 'Aprende con rutas de aprendizaje'),
              const Spacer(flex: 2),
              // Buttons
              ElevatedButton(
                onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const LoginScreen())),
                style: ElevatedButton.styleFrom(minimumSize: const Size(double.infinity, 52)),
                child: const Text('INICIAR SESIÓN', style: TextStyle(fontWeight: FontWeight.w700, letterSpacing: 1)),
              ),
              const SizedBox(height: 12),
              OutlinedButton(
                onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const RegisterScreen())),
                style: OutlinedButton.styleFrom(minimumSize: const Size(double.infinity, 52)),
                child: const Text('REGISTRARSE', style: TextStyle(fontWeight: FontWeight.w700, letterSpacing: 1)),
              ),
              const SizedBox(height: 24),
              const Text('Al continuar aceptas nuestros Términos y Política de Privacidad.', textAlign: TextAlign.center, style: TextStyle(color: AppColors.textHint, fontSize: 11)),
              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }
}

class _FeatureRow extends StatelessWidget {
  final IconData icon;
  final String text;
  const _FeatureRow({required this.icon, required this.text});

  @override
  Widget build(BuildContext context) => Row(
    children: [
      Container(padding: const EdgeInsets.all(8), decoration: BoxDecoration(color: AppColors.primary.withOpacity(0.1), borderRadius: BorderRadius.circular(10)), child: Icon(icon, color: AppColors.primary, size: 18)),
      const SizedBox(width: 12),
      Text(text, style: const TextStyle(fontSize: 14, color: AppColors.textPrimary)),
    ],
  );
}
