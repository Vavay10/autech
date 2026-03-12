import 'package:flutter/material.dart';
import '../../theme.dart';

class StatsScreen extends StatelessWidget {
  const StatsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: CustomScrollView(
        slivers: [
          SliverAppBar(
            expandedHeight: 140,
            pinned: true,
            backgroundColor: AppColors.surface,
            surfaceTintColor: Colors.transparent,
            flexibleSpace: FlexibleSpaceBar(
              title: const Text('Mi Desempeño', style: TextStyle(color: AppColors.textPrimary, fontWeight: FontWeight.w700, fontSize: 18)),
              background: Container(
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [Color(0xFFE6FFFA), Color(0xFFEBF8FF)],
                  ),
                ),
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(20, 50, 20, 0),
                  child: Row(
                    children: const [
                      Icon(Icons.emoji_events, color: AppColors.warning, size: 40),
                      SizedBox(width: 12),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text('¡Buen progreso!', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 16, color: AppColors.textPrimary)),
                          Text('Sigue practicando para mejorar', style: TextStyle(color: AppColors.textSecondary, fontSize: 12)),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),

          SliverPadding(
            padding: const EdgeInsets.all(16),
            sliver: SliverList(
              delegate: SliverChildListDelegate([

                // ── Summary cards
                Row(children: const [
                  Expanded(child: _StatCard(value: '24', label: 'Ejercicios\ncompletados', icon: Icons.check_circle_outline, color: AppColors.success)),
                  SizedBox(width: 10),
                  Expanded(child: _StatCard(value: '78%', label: 'Tasa de\naciertos', icon: Icons.percent, color: AppColors.primary)),
                  SizedBox(width: 10),
                  Expanded(child: _StatCard(value: '5', label: 'Racha\nactual (días)', icon: Icons.local_fire_department, color: AppColors.warning)),
                ]),
                const SizedBox(height: 16),

                // ── Progress by unit
                const _SectionTitle('Progreso por Unidad'),
                const SizedBox(height: 10),
                const _UnitProgress(unit: 'UNIDAD I – Autómatas Finitos', progress: 0.75, color: AppColors.primary),
                const SizedBox(height: 8),
                const _UnitProgress(unit: 'UNIDAD II – Expresiones Regulares', progress: 0.50, color: AppColors.success),
                const SizedBox(height: 8),
                const _UnitProgress(unit: 'UNIDAD III – Autómatas de Pila', progress: 0.30, color: AppColors.warning),
                const SizedBox(height: 8),
                const _UnitProgress(unit: 'UNIDAD IV – Máquinas de Turing', progress: 0.10, color: AppColors.error),
                const SizedBox(height: 20),

                // ── Activity by topic
                const _SectionTitle('Actividad por Tema'),
                const SizedBox(height: 10),
                const _TopicBarChart(),
                const SizedBox(height: 20),

                // ── Recent activity
                const _SectionTitle('Actividad Reciente'),
                const SizedBox(height: 10),
                ..._recentActivity.map((a) => _ActivityTile(
                  title: a['title']!,
                  subtitle: a['subtitle']!,
                  result: a['result']!,
                  icon: _iconForTopic(a['topic']!),
                  color: _colorForResult(a['result']!),
                )),
                const SizedBox(height: 20),

                // ── Achievements
                const _SectionTitle('Logros'),
                const SizedBox(height: 10),
                Wrap(
                  spacing: 10,
                  runSpacing: 10,
                  children: const [
                    _AchievementBadge(icon: Icons.star, label: 'Primera práctica', unlocked: true),
                    _AchievementBadge(icon: Icons.local_fire_department, label: 'Racha de 5 días', unlocked: true),
                    _AchievementBadge(icon: Icons.emoji_events, label: '10 ejercicios', unlocked: true),
                    _AchievementBadge(icon: Icons.psychology, label: 'Maestro DFA', unlocked: false),
                    _AchievementBadge(icon: Icons.memory, label: 'Turing experto', unlocked: false),
                    _AchievementBadge(icon: Icons.school, label: 'Unidad I completa', unlocked: false),
                  ],
                ),
                const SizedBox(height: 30),
              ]),
            ),
          ),
        ],
      ),
    );
  }

  static IconData _iconForTopic(String t) {
    switch (t) {
      case 'regex': return Icons.code;
      case 'pda':   return Icons.layers;
      case 'turing':return Icons.memory;
      default:      return Icons.check;
    }
  }

  static Color _colorForResult(String r) {
    if (r.contains('✅')) return AppColors.success;
    if (r.contains('❌')) return AppColors.error;
    return AppColors.warning;
  }

  static const _recentActivity = [
    {'title': 'AFD – Acepta (a|b)*abb', 'subtitle': 'hace 1 hora', 'result': '✅ Correcto', 'topic': 'regex'},
    {'title': 'PDA – Reconocer aⁿbⁿ', 'subtitle': 'hace 2 horas', 'result': '✅ Correcto', 'topic': 'pda'},
    {'title': 'MT – Sumar en unario', 'subtitle': 'ayer', 'result': '❌ Incorrecto', 'topic': 'turing'},
    {'title': 'AFN → AFD conversión', 'subtitle': 'ayer', 'result': '✅ Correcto', 'topic': 'regex'},
    {'title': 'PDA – Gramática CFG', 'subtitle': 'hace 2 días', 'result': '⏳ Incompleto', 'topic': 'pda'},
  ];
}

// ─── Widgets ─────────────────────────────────────────────────────────────────

class _SectionTitle extends StatelessWidget {
  final String text;
  const _SectionTitle(this.text);

  @override
  Widget build(BuildContext context) => Text(
    text,
    style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 15, color: AppColors.textPrimary),
  );
}

class _StatCard extends StatelessWidget {
  final String value, label;
  final IconData icon;
  final Color color;

  const _StatCard({required this.value, required this.label, required this.icon, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color, size: 22),
          const SizedBox(height: 8),
          Text(value, style: TextStyle(fontSize: 26, fontWeight: FontWeight.w800, color: color)),
          const SizedBox(height: 2),
          Text(label, style: const TextStyle(fontSize: 11, color: AppColors.textSecondary, height: 1.3)),
        ],
      ),
    );
  }
}

class _UnitProgress extends StatelessWidget {
  final String unit;
  final double progress;
  final Color color;

  const _UnitProgress({required this.unit, required this.progress, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(child: Text(unit, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600))),
              Text('${(progress * 100).toInt()}%', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w700, color: color)),
            ],
          ),
          const SizedBox(height: 8),
          ClipRRect(
            borderRadius: BorderRadius.circular(6),
            child: LinearProgressIndicator(
              value: progress,
              backgroundColor: color.withOpacity(0.12),
              valueColor: AlwaysStoppedAnimation(color),
              minHeight: 7,
            ),
          ),
        ],
      ),
    );
  }
}

// ─── Bar chart (simple custom paint) ────────────────────────────────────────

class _TopicBarChart extends StatelessWidget {
  const _TopicBarChart();

  @override
  Widget build(BuildContext context) {
    const data = [
      _BarData('DFA/NFA', 12, AppColors.primary),
      _BarData('Regex', 8, AppColors.success),
      _BarData('PDA', 6, AppColors.warning),
      _BarData('Turing', 2, AppColors.error),
    ];
    final maxV = data.fold(0, (m, d) => d.value > m ? d.value : m);

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: AppColors.surface, borderRadius: BorderRadius.circular(16), border: Border.all(color: AppColors.border)),
      child: Column(
        children: [
          SizedBox(
            height: 100,
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: data.expand((d) => [
                Expanded(child: _Bar(data: d, maxValue: maxV)),
                const SizedBox(width: 8),
              ]).toList()..removeLast(),
            ),
          ),
          const SizedBox(height: 8),
          Row(
            children: data.expand((d) => [
              Expanded(child: Column(children: [
                Container(width: 10, height: 10, decoration: BoxDecoration(color: d.color, shape: BoxShape.circle)),
                const SizedBox(height: 3),
                Text(d.label, style: const TextStyle(fontSize: 10, color: AppColors.textSecondary), textAlign: TextAlign.center),
              ])),
              const SizedBox(width: 8),
            ]).toList()..removeLast(),
          ),
        ],
      ),
    );
  }
}

class _BarData {
  final String label;
  final int value;
  final Color color;
  const _BarData(this.label, this.value, this.color);
}

class _Bar extends StatelessWidget {
  final _BarData data;
  final int maxValue;
  const _Bar({required this.data, required this.maxValue});

  @override
  Widget build(BuildContext context) {
    final frac = maxValue > 0 ? data.value / maxValue : 0.0;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.center,
      mainAxisAlignment: MainAxisAlignment.end,
      children: [
        Text('${data.value}', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: data.color)),
        const SizedBox(height: 4),
        Flexible(
          child: FractionallySizedBox(
            heightFactor: frac.clamp(0.05, 1.0),
            child: Container(
              decoration: BoxDecoration(
                color: data.color,
                borderRadius: const BorderRadius.vertical(top: Radius.circular(4)),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

class _ActivityTile extends StatelessWidget {
  final String title, subtitle, result;
  final IconData icon;
  final Color color;

  const _ActivityTile({required this.title, required this.subtitle, required this.result, required this.icon, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(color: AppColors.surface, borderRadius: BorderRadius.circular(12), border: Border.all(color: AppColors.border)),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(color: color.withOpacity(0.1), borderRadius: BorderRadius.circular(10)),
            child: Icon(icon, color: color, size: 18),
          ),
          const SizedBox(width: 12),
          Expanded(child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
              Text(subtitle, style: const TextStyle(fontSize: 11, color: AppColors.textSecondary)),
            ],
          )),
          Text(result, style: TextStyle(fontSize: 12, color: color, fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }
}

class _AchievementBadge extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool unlocked;

  const _AchievementBadge({required this.icon, required this.label, required this.unlocked});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 90,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: unlocked ? AppColors.nodeActive : AppColors.surfaceAlt,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: unlocked ? AppColors.warning : AppColors.border),
      ),
      child: Column(
        children: [
          Icon(icon, color: unlocked ? AppColors.warning : AppColors.textHint, size: 28),
          const SizedBox(height: 6),
          Text(label, textAlign: TextAlign.center, style: TextStyle(fontSize: 10, color: unlocked ? AppColors.textPrimary : AppColors.textHint, fontWeight: FontWeight.w600, height: 1.3)),
        ],
      ),
    );
  }
}
