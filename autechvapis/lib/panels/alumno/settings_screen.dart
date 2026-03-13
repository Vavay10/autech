// lib/modules/settings_screen.dart
import 'package:flutter/material.dart';
import '../../theme.dart';
import '../../general/auth_screen.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  bool _notifications = true;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: CustomScrollView(
        slivers: [
          const SliverAppBar(
            pinned: true,
            backgroundColor: AppColors.surface,
            surfaceTintColor: Colors.transparent,
            title: Text('Perfil y Ajustes', style: TextStyle(color: AppColors.textPrimary, fontWeight: FontWeight.w700)),
          ),
          SliverPadding(
            padding: const EdgeInsets.all(16),
            sliver: SliverList(
              delegate: SliverChildListDelegate([
                // Profile
                Container(
                  padding: const EdgeInsets.all(18),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(colors: [AppColors.primary.withOpacity(0.08), AppColors.primary.withOpacity(0.03)]),
                    borderRadius: BorderRadius.circular(18),
                    border: Border.all(color: AppColors.primary.withOpacity(0.2)),
                  ),
                  child: Row(children: [
                    CircleAvatar(radius: 34, backgroundColor: AppColors.primary.withOpacity(0.15), child: const Icon(Icons.person, size: 34, color: AppColors.primary)),
                    const SizedBox(width: 14),
                    const Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      Text('Iván Estudiante', style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700)),
                      Text('Boleta: 202XXXXXXX', style: TextStyle(color: AppColors.textSecondary, fontSize: 13)),
                      Text('ESCOM - IPN', style: TextStyle(color: AppColors.primary, fontSize: 12, fontWeight: FontWeight.w600)),
                    ]),
                  ]),
                ),
                const SizedBox(height: 20),
                _sectionTitle('Cuenta'),
                _tile(Icons.person_outline, 'Editar Perfil', 'Nombre, correo y foto', () {}),
                _tile(Icons.lock_outline, 'Seguridad', 'Cambiar contraseña', () {}),
                const SizedBox(height: 16),
                _sectionTitle('Aplicación'),
                _switchTile(Icons.notifications_none, 'Notificaciones', 'Alertas de tareas y ejercicios', _notifications, (v) => setState(() => _notifications = v)),
                _tile(Icons.dns_outlined, 'Servidor API', 'Configurar IP del backend', () => _showIPDialog(context)),
                const SizedBox(height: 16),
                _sectionTitle('Acerca de'),
                _tile(Icons.info_outline, 'Versión', '1.0.0', () {}),
                _tile(Icons.help_outline, 'Ayuda y soporte', 'Documentación y contacto', () {}),
                const SizedBox(height: 28),
                OutlinedButton.icon(
                  onPressed: () => Navigator.of(context).pushAndRemoveUntil(MaterialPageRoute(builder: (_) => const AuthScreen()), (_) => false),
                  icon: const Icon(Icons.logout, size: 18),
                  label: const Text('CERRAR SESIÓN', style: TextStyle(fontWeight: FontWeight.w700)),
                  style: OutlinedButton.styleFrom(
                    minimumSize: const Size(double.infinity, 50),
                    side: const BorderSide(color: AppColors.error),
                    foregroundColor: AppColors.error,
                  ),
                ),
                const SizedBox(height: 20),
              ]),
            ),
          ),
        ],
      ),
    );
  }

  Widget _sectionTitle(String t) => Padding(
    padding: const EdgeInsets.only(bottom: 8),
    child: Text(t, style: const TextStyle(color: AppColors.primary, fontWeight: FontWeight.w700, fontSize: 12, letterSpacing: 0.5)),
  );

  Widget _tile(IconData icon, String title, String sub, VoidCallback onTap) => Container(
    margin: const EdgeInsets.only(bottom: 1),
    child: ListTile(
      leading: Container(padding: const EdgeInsets.all(8), decoration: BoxDecoration(color: AppColors.surfaceAlt, borderRadius: BorderRadius.circular(8)), child: Icon(icon, size: 18, color: AppColors.primary)),
      title: Text(title, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600)),
      subtitle: Text(sub, style: const TextStyle(fontSize: 12, color: AppColors.textSecondary)),
      trailing: const Icon(Icons.chevron_right, size: 16, color: AppColors.textHint),
      onTap: onTap,
      contentPadding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
    ),
  );

  Widget _switchTile(IconData icon, String title, String sub, bool val, ValueChanged<bool> onChanged) => ListTile(
    leading: Container(padding: const EdgeInsets.all(8), decoration: BoxDecoration(color: AppColors.surfaceAlt, borderRadius: BorderRadius.circular(8)), child: Icon(icon, size: 18, color: AppColors.primary)),
    title: Text(title, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600)),
    subtitle: Text(sub, style: const TextStyle(fontSize: 12, color: AppColors.textSecondary)),
    trailing: Switch.adaptive(value: val, onChanged: onChanged, activeColor: AppColors.primary),
    contentPadding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
  );

  void _showIPDialog(BuildContext ctx) => showDialog(context: ctx, builder: (_) => AlertDialog(
    title: const Text('Configurar servidor'),
    content: const TextField(decoration: InputDecoration(labelText: 'URL del servidor', hintText: 'http://192.168.1.XX:8000')),
    actions: [
      TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancelar')),
      ElevatedButton(onPressed: () => Navigator.pop(ctx), child: const Text('Guardar')),
    ],
  ));
}
