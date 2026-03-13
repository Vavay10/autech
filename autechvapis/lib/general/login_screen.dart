// lib/modules/login_screen.dart
import 'package:flutter/material.dart';
import '../theme.dart';
import '../layout/main_layout.dart';
import 'forgot_password_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _userCtrl = TextEditingController();
  final _passCtrl = TextEditingController();
  bool _obscure = true;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(backgroundColor: Colors.transparent, elevation: 0,
        leading: IconButton(icon: const Icon(Icons.arrow_back, color: AppColors.textPrimary), onPressed: () => Navigator.pop(context))),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 28),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 12),
            const Text('Iniciar Sesión', style: TextStyle(fontSize: 26, fontWeight: FontWeight.w800, color: AppColors.textPrimary)),
            const SizedBox(height: 4),
            const Text('Bienvenido de vuelta', style: TextStyle(color: AppColors.textSecondary)),
            const SizedBox(height: 32),
            TextField(controller: _userCtrl, decoration: const InputDecoration(labelText: 'Correo o usuario', prefixIcon: Icon(Icons.person_outline, size: 18))),
            const SizedBox(height: 14),
            TextField(
              controller: _passCtrl,
              obscureText: _obscure,
              decoration: InputDecoration(
                labelText: 'Contraseña',
                prefixIcon: const Icon(Icons.lock_outline, size: 18),
                suffixIcon: IconButton(icon: Icon(_obscure ? Icons.visibility_off_outlined : Icons.visibility_outlined, size: 18, color: AppColors.textHint), onPressed: () => setState(() => _obscure = !_obscure)),
              ),
            ),
            const SizedBox(height: 8),
            Align(alignment: Alignment.centerRight,
              child: TextButton(onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const ForgotPasswordScreen())), child: const Text('¿Olvidé mi contraseña?'))),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () => Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => MainLayout())),
              style: ElevatedButton.styleFrom(minimumSize: const Size(double.infinity, 52)),
              child: const Text('INICIAR SESIÓN', style: TextStyle(fontWeight: FontWeight.w700, letterSpacing: 1)),
            ),
            const SizedBox(height: 28),
            Row(children: const [Expanded(child: Divider()), Padding(padding: EdgeInsets.symmetric(horizontal: 12), child: Text('o', style: TextStyle(color: AppColors.textHint))), Expanded(child: Divider())]),
            const SizedBox(height: 16),
            _SocialBtn(icon: Icons.g_mobiledata, label: 'Continuar con Google'),
            const SizedBox(height: 10),
            _SocialBtn(icon: Icons.facebook, label: 'Continuar con Facebook'),
          ],
        ),
      ),
    );
  }
}

class _SocialBtn extends StatelessWidget {
  final IconData icon;
  final String label;
  const _SocialBtn({required this.icon, required this.label});

  @override
  Widget build(BuildContext context) => OutlinedButton.icon(
    onPressed: () {},
    icon: Icon(icon, size: 22),
    label: Text(label, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
    style: OutlinedButton.styleFrom(
      minimumSize: const Size(double.infinity, 50),
      foregroundColor: AppColors.textPrimary,
      side: const BorderSide(color: AppColors.border),
    ),
  );
}
