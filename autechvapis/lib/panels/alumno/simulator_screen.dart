import 'package:flutter/material.dart';
import '../../theme.dart';
import 'modules/pda_screen.dart';
import 'modules/regex_screen.dart';
import 'modules/turing_screen.dart';

class SimulatorScreen extends StatelessWidget {
  const SimulatorScreen({super.key});

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
              title: const Text(
                'Herramientas',
                style: TextStyle(
                    color: AppColors.textPrimary, fontWeight: FontWeight.w700),
              ),
              background: Container(
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [Color(0xFFE6FFFA), Color(0xFFEBF8FF)],
                  ),
                ),
              ),
            ),
          ),
          SliverPadding(
            padding: const EdgeInsets.all(16),
            sliver: SliverList(
              delegate: SliverChildListDelegate([
                const Text(
                  'Simuladores interactivos de modelos computacionales.\nDiseña, visualiza y prueba autómatas paso a paso.',
                  style: TextStyle(
                      color: AppColors.textSecondary, fontSize: 13, height: 1.5),
                ),
                const SizedBox(height: 20),

                // ── Tool cards ──────────────────────────────────────────
                _ToolCard(
                  title: 'Expresiones Regulares',
                  subtitle:
                      'Construye AFN/AFD gráficamente · Convierte regex en autómata · Anima prueba de cadenas',
                  pixelIcon: _PixelIcons.regex,
                  color: const Color(0xFF3182CE),
                  tags: ['DFA', 'NFA', 'Regex', 'Animación'],
                  onTap: () => Navigator.push(context,
                      MaterialPageRoute(builder: (_) => const RegexScreen())),
                ),
                const SizedBox(height: 12),
                _ToolCard(
                  title: 'Autómata de Pila (PDA)',
                  subtitle:
                      'Define, visualiza y simula un PDA · Convierte a Gramática Libre de Contexto (CFG)',
                  pixelIcon: _PixelIcons.pda,
                  color: const Color(0xFF38A169),
                  tags: ['PDA', 'CFG', 'Pila', 'Simulación'],
                  onTap: () => Navigator.push(context,
                      MaterialPageRoute(builder: (_) => const PdaScreen())),
                ),
                const SizedBox(height: 12),
                _ToolCard(
                  title: 'Máquina de Turing',
                  subtitle:
                      'Simulación completa con cinta animada · Grafo del autómata interactivo · Paso a paso',
                  pixelIcon: _PixelIcons.turing,
                  color: const Color(0xFF805AD5),
                  tags: ['Turing', 'Cinta', 'Paso a paso'],
                  onTap: () => Navigator.push(context,
                      MaterialPageRoute(builder: (_) => const TuringScreen())),
                ),

                const SizedBox(height: 28),

                // ── Quick reference ─────────────────────────────────────
                const _QuickReferenceCard(),
                const SizedBox(height: 20),
              ]),
            ),
          ),
        ],
      ),
    );
  }
}

// ─── Pixel icon definitions ──────────────────────────────────────────────────

class _PixelIcons {
  static const regex = _PixelIcon(
    icon: Icons.code_rounded,
    bgColor: Color(0xFF3182CE),
    badge: 'DFA',
  );
  static const pda = _PixelIcon(
    icon: Icons.layers_rounded,
    bgColor: Color(0xFF38A169),
    badge: 'PDA',
  );
  static const turing = _PixelIcon(
    icon: Icons.memory_rounded,
    bgColor: Color(0xFF805AD5),
    badge: 'MT',
  );
}

class _PixelIcon {
  final IconData icon;
  final Color bgColor;
  final String badge;
  const _PixelIcon({required this.icon, required this.bgColor, required this.badge});
}

// ─── Tool Card ───────────────────────────────────────────────────────────────

class _ToolCard extends StatelessWidget {
  final String title, subtitle;
  final _PixelIcon pixelIcon;
  final Color color;
  final List<String> tags;
  final VoidCallback onTap;

  const _ToolCard({
    required this.title,
    required this.subtitle,
    required this.pixelIcon,
    required this.color,
    required this.tags,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(18),
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: AppColors.border),
            boxShadow: [
              BoxShadow(
                  color: Colors.black.withOpacity(0.04),
                  blurRadius: 8,
                  offset: const Offset(0, 2))
            ],
          ),
          child: Row(
            children: [
              // Pixel-style icon box
              _PixelIconWidget(pixelIcon: pixelIcon),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: const TextStyle(
                          fontSize: 15, fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      subtitle,
                      style: const TextStyle(
                          fontSize: 12,
                          color: AppColors.textSecondary,
                          height: 1.4),
                    ),
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 6,
                      runSpacing: 4,
                      children: tags
                          .map((t) => Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 8, vertical: 3),
                                decoration: BoxDecoration(
                                  color: color.withOpacity(0.08),
                                  borderRadius: BorderRadius.circular(6),
                                ),
                                child: Text(t,
                                    style: TextStyle(
                                        fontSize: 10,
                                        color: color,
                                        fontWeight: FontWeight.w600)),
                              ))
                          .toList(),
                    ),
                  ],
                ),
              ),
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.08),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(Icons.chevron_right, color: color, size: 20),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ─── Pixel icon widget (game-like) ───────────────────────────────────────────

class _PixelIconWidget extends StatelessWidget {
  final _PixelIcon pixelIcon;
  const _PixelIconWidget({required this.pixelIcon});

  @override
  Widget build(BuildContext context) {
    return Stack(
      clipBehavior: Clip.none,
      children: [
        // Main icon container (pixel/retro game style)
        Container(
          width: 62,
          height: 62,
          decoration: BoxDecoration(
            color: pixelIcon.bgColor,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(
              color: pixelIcon.bgColor.withOpacity(0.7),
              width: 2,
            ),
            boxShadow: [
              // Pixel-art style shadow (offset bottom-right)
              BoxShadow(
                color: pixelIcon.bgColor.withOpacity(0.4),
                offset: const Offset(3, 3),
                blurRadius: 0,
              ),
              BoxShadow(
                color: pixelIcon.bgColor.withOpacity(0.2),
                blurRadius: 12,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Center(
            child: Icon(pixelIcon.icon, color: Colors.white, size: 30),
          ),
        ),
        // Badge label
        Positioned(
          bottom: -6,
          right: -6,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
            decoration: BoxDecoration(
              color: pixelIcon.bgColor,
              borderRadius: BorderRadius.circular(6),
              border: Border.all(color: Colors.white, width: 1.5),
            ),
            child: Text(
              pixelIcon.badge,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 9,
                fontWeight: FontWeight.w800,
                letterSpacing: 0.5,
              ),
            ),
          ),
        ),
      ],
    );
  }
}

// ─── Quick Reference Card ─────────────────────────────────────────────────────

class _QuickReferenceCard extends StatelessWidget {
  const _QuickReferenceCard();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.primary.withOpacity(0.05),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.primary.withOpacity(0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(children: [
            Icon(Icons.lightbulb_outline, color: AppColors.primary, size: 16),
            SizedBox(width: 6),
            Text(
              'Referencia rápida',
              style: TextStyle(
                  fontWeight: FontWeight.w700,
                  fontSize: 13,
                  color: AppColors.primary),
            ),
          ]),
          const SizedBox(height: 10),
          ...[
            ('○', 'Estado normal'),
            ('○○', 'Estado de aceptación (doble círculo)'),
            ('→○', 'Estado inicial (flecha entrante)'),
            ('—a→', 'Transición con símbolo a'),
            ('ε', 'Transición vacía (épsilon)'),
          ].map((item) => Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Row(children: [
                  Container(
                    width: 36,
                    child: Text(
                      item.$1,
                      style: const TextStyle(
                          fontFamily: 'monospace',
                          fontWeight: FontWeight.w700,
                          color: AppColors.primary,
                          fontSize: 13),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      item.$2,
                      style: const TextStyle(
                          fontSize: 12, color: AppColors.textSecondary),
                    ),
                  ),
                ]),
              )),
        ],
      ),
    );
  }
}