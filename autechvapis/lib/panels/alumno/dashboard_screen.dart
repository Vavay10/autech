// ─── dashboard_screen.dart ────────────────────────────────────────────────────
import 'package:flutter/material.dart';
import '../../theme.dart';

class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: CustomScrollView(
        slivers: [
          SliverAppBar(
            expandedHeight: 160,
            pinned: true,
            backgroundColor: AppColors.surface,
            surfaceTintColor: Colors.transparent,
            flexibleSpace: FlexibleSpaceBar(
              background: Container(
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topLeft, end: Alignment.bottomRight,
                    colors: [Color(0xFFE6FFFA), Color(0xFFEBF8FF)],
                  ),
                ),
                padding: const EdgeInsets.fromLTRB(20, 56, 20, 16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: const [
                    Text('¡Bienvenido de vuelta!', style: TextStyle(color: AppColors.textSecondary, fontSize: 13)),
                    SizedBox(height: 2),
                    Text('Mis Clases', style: TextStyle(color: AppColors.textPrimary, fontSize: 24, fontWeight: FontWeight.w800)),
                  ],
                ),
              ),
            ),
          ),
          SliverPadding(
            padding: const EdgeInsets.all(16),
            sliver: SliverList(
              delegate: SliverChildListDelegate([
                _ClassCard(
                  subject: 'Teoría de la Computación',
                  teacher: 'Prof. Alan Turing',
                  info: '3 Ejercicios activos',
                  color: const Color(0xFF3182CE),
                  progress: 0.65,
                ),
                const SizedBox(height: 12),
                _ClassCard(
                  subject: 'Compiladores',
                  teacher: 'Prof. Grace Hopper',
                  info: 'Tarea pendiente: AFN a AFD',
                  color: const Color(0xFF38A169),
                  progress: 0.40,
                ),
                const SizedBox(height: 24),
                const _SectionHeader(title: 'Acceso rápido', icon: Icons.bolt),
                const SizedBox(height: 12),
                Row(children: const [
                  Expanded(child: _QuickBtn(icon: Icons.code, label: 'Regex', color: Color(0xFF3182CE))),
                  SizedBox(width: 10),
                  Expanded(child: _QuickBtn(icon: Icons.layers, label: 'PDA', color: Color(0xFF38A169))),
                  SizedBox(width: 10),
                  Expanded(child: _QuickBtn(icon: Icons.memory, label: 'Turing', color: Color(0xFF805AD5))),
                ]),
              ]),
            ),
          ),
        ],
      ),
    );
  }
}

class _ClassCard extends StatelessWidget {
  final String subject, teacher, info;
  final Color color;
  final double progress;

  const _ClassCard({required this.subject, required this.teacher, required this.info, required this.color, required this.progress});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 8, offset: const Offset(0, 2))],
      ),
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: color, borderRadius: const BorderRadius.vertical(top: Radius.circular(17)),
            ),
            child: Row(children: [
              Expanded(child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(subject, style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w700)),
                  const SizedBox(height: 2),
                  Text(teacher, style: const TextStyle(color: Colors.white70, fontSize: 12)),
                ],
              )),
              Icon(Icons.school_outlined, color: Colors.white.withOpacity(0.6), size: 36),
            ]),
          ),
          Padding(
            padding: const EdgeInsets.all(14),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(children: [
                  Icon(Icons.info_outline, size: 14, color: AppColors.textSecondary),
                  const SizedBox(width: 6),
                  Expanded(child: Text(info, style: const TextStyle(fontSize: 13, color: AppColors.textSecondary))),
                  Text('${(progress * 100).toInt()}%', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: color)),
                ]),
                const SizedBox(height: 8),
                ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(value: progress, backgroundColor: color.withOpacity(0.1), valueColor: AlwaysStoppedAnimation(color), minHeight: 5),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _SectionHeader extends StatelessWidget {
  final String title;
  final IconData icon;
  const _SectionHeader({required this.title, required this.icon});

  @override
  Widget build(BuildContext context) => Row(children: [
    Icon(icon, size: 16, color: AppColors.primary),
    const SizedBox(width: 6),
    Text(title, style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 15)),
  ]);
}

class _QuickBtn extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  const _QuickBtn({required this.icon, required this.label, required this.color});

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.symmetric(vertical: 16),
    decoration: BoxDecoration(
      color: color.withOpacity(0.08),
      borderRadius: BorderRadius.circular(14),
      border: Border.all(color: color.withOpacity(0.2)),
    ),
    child: Column(children: [
      Icon(icon, color: color, size: 24),
      const SizedBox(height: 6),
      Text(label, style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: color)),
    ]),
  );
}
