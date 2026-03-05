import 'package:flutter/material.dart';
import '../theme.dart';

class SettingsScreen extends StatefulWidget {
  @override
  _SettingsScreenState createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  bool _isDarkMode = true;
  bool _notifications = true;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Ajustes y Perfil"),
        centerTitle: true,
      ),
      body: ListView(
        children: [
          // 1. Sección de Perfil
          _buildProfileHeader(),
          
          const Divider(),

          // 2. Configuración de Usuario
          _buildSectionTitle("Cuenta"),
          _buildSettingItem(Icons.person_outline, "Editar Perfil", "Nombre, correo y foto", () {}),
          _buildSettingItem(Icons.lock_outline, "Seguridad", "Cambiar contraseña", () {}),

          const Divider(),

          // 3. Configuración de la App
          _buildSectionTitle("Aplicación"),
          SwitchListTile(
            secondary: const Icon(Icons.dark_mode_outlined, color: Colors.cyan),
            title: const Text("Modo Oscuro"),
            subtitle: const Text("Cambiar el tema de la interfaz"),
            value: _isDarkMode,
            onChanged: (val) => setState(() => _isDarkMode = val),
          ),
          SwitchListTile(
            secondary: const Icon(Icons.notifications_none, color: Colors.cyan),
            title: const Text("Notificaciones"),
            subtitle: const Text("Alertas de tareas y ejercicios"),
            value: _notifications,
            onChanged: (val) => setState(() => _notifications = val),
          ),

          const Divider(),

          // 4. Configuración Técnica (Vital para tu TT)
          _buildSectionTitle("Conectividad"),
          _buildSettingItem(
            Icons.dns_outlined, 
            "Dirección del Servidor", 
            "Actual: http://10.0.2.2:8000", 
            () => _showIPDialog(context)
          ),

          const SizedBox(height: 30),

          // 5. Botón Cerrar Sesión
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.redAccent.withOpacity(0.1),
                side: const BorderSide(color: Colors.redAccent),
                foregroundColor: Colors.redAccent,
                padding: const EdgeInsets.symmetric(vertical: 15),
              ),
              onPressed: () => Navigator.of(context).pushReplacementNamed('/login'),
              child: const Text("CERRAR SESIÓN", style: TextStyle(fontWeight: FontWeight.bold)),
            ),
          ),
          const SizedBox(height: 20),
        ],
      ),
    );
  }

  Widget _buildProfileHeader() {
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Row(
        children: [
          CircleAvatar(
            radius: 40,
            backgroundColor: Colors.cyan.shade100,
            child: const Icon(Icons.person, size: 50, color: Colors.cyan),
          ),
          const SizedBox(width: 20),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: const [
              Text("Iván Estudiante", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              Text("Boleta: 202XXXXXXX", style: TextStyle(color: Colors.grey)),
              Text("ESCOM - IPN", style: TextStyle(fontSize: 12, color: Colors.cyan)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 15, 20, 10),
      child: Text(title, style: const TextStyle(color: Colors.cyan, fontWeight: FontWeight.bold, fontSize: 13)),
    );
  }

  Widget _buildSettingItem(IconData icon, String title, String subtitle, VoidCallback onTap) {
    return ListTile(
      leading: Icon(icon, color: Colors.cyan),
      title: Text(title),
      subtitle: Text(subtitle, style: const TextStyle(fontSize: 12)),
      trailing: const Icon(Icons.arrow_forward_ios, size: 14),
      onTap: onTap,
    );
  }

  void _showIPDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text("Configurar API"),
        content: const TextField(
          decoration: InputDecoration(hintText: "http://192.168.1.XX:8000"),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text("CANCELAR")),
          ElevatedButton(onPressed: () => Navigator.pop(context), child: const Text("GUARDAR")),
        ],
      ),
    );
  }
}