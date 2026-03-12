// lib/modules/learning_path_screen.dart
import 'package:flutter/material.dart';
import 'unit_nodes_screen.dart';

class LearningPathScreen extends StatefulWidget {
  const LearningPathScreen({super.key});

  @override
  State<LearningPathScreen> createState() => _LearningPathScreenState();
}

class _LearningPathScreenState extends State<LearningPathScreen>
    with TickerProviderStateMixin {
  bool _darkMode = false;
  late AnimationController _pulseCtrl;
  late Animation<double> _pulse;

  // Units data
  static const _units = [
    _UnitData(
      unitNum: 'I',
      title: 'Autómatas Finitos',
      subtitle: 'Lenguajes Regulares',
      icon: Icons.menu_book_rounded,
      color: Color(0xFF3182CE),
      progress: 0.65,
      streak: 5,
      exercises: [
        _ExData('Introducción teórica', Icons.menu_book, true),
        _ExData('Ejercicios AFD/AFN', Icons.extension, true),
        _ExData('Simulador interactivo', Icons.code, false),
        _ExData('Evaluación', Icons.quiz_outlined, false),
      ],
    ),
    _UnitData(
      unitNum: 'II',
      title: 'Autómatas de Pila',
      subtitle: 'Gramáticas Libres de Contexto',
      icon: Icons.layers_rounded,
      color: Color(0xFF38A169),
      progress: 0.30,
      streak: 2,
      exercises: [
        _ExData('Gramáticas CFG', Icons.menu_book, true),
        _ExData('PDA y CFG', Icons.extension, false),
        _ExData('Simulador PDA', Icons.code, false),
        _ExData('Evaluación', Icons.quiz_outlined, false),
      ],
    ),
    _UnitData(
      unitNum: 'III',
      title: 'Máquinas de Turing',
      subtitle: 'Decidibilidad y Computabilidad',
      icon: Icons.memory_rounded,
      color: Color(0xFF805AD5),
      progress: 0.10,
      streak: 0,
      exercises: [
        _ExData('MT básica', Icons.menu_book, false),
        _ExData('Variantes de MT', Icons.extension, false),
        _ExData('Simulador MT', Icons.code, false),
        _ExData('Evaluación', Icons.quiz_outlined, false),
      ],
    ),
  ];

  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(
        vsync: this, duration: const Duration(seconds: 2))
      ..repeat(reverse: true);
    _pulse = Tween<double>(begin: 0.95, end: 1.05).animate(
        CurvedAnimation(parent: _pulseCtrl, curve: Curves.easeInOut));
  }

  @override
  void dispose() {
    _pulseCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor:
          _darkMode ? const Color(0xFF0D1117) : const Color(0xFF87CEEB),
      body: Stack(
        children: [
          // Background
          _buildBackground(),
          // Content
          SafeArea(
            child: Column(
              children: [
                _buildTopBar(),
                Expanded(
                  child: SingleChildScrollView(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 16, vertical: 8),
                    child: Column(
                      children: [
                        for (int i = 0; i < _units.length; i++) ...[
                          _buildUnitSection(i),
                          const SizedBox(height: 24),
                        ],
                        const SizedBox(height: 40),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBackground() {
    if (_darkMode) {
      // Stars background
      return CustomPaint(
        size: Size.infinite,
        painter: _StarsPainter(),
      );
    } else {
      // Clouds background
      return CustomPaint(
        size: Size.infinite,
        painter: _CloudsPainter(),
      );
    }
  }

  Widget _buildTopBar() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
      child: Row(
        children: [
          // Streak
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: Colors.orange.withOpacity(0.2),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: Colors.orange.withOpacity(0.5)),
            ),
            child: const Row(
              children: [
                Text('🔥', style: TextStyle(fontSize: 18)),
                SizedBox(width: 4),
                Text('7',
                    style: TextStyle(
                        color: Colors.orange,
                        fontWeight: FontWeight.w800,
                        fontSize: 16)),
              ],
            ),
          ),
          const Spacer(),
          // Title
          Text(
            'Rutas de Aprendizaje',
            style: TextStyle(
              color: _darkMode ? Colors.white : const Color(0xFF1A365D),
              fontWeight: FontWeight.w800,
              fontSize: 16,
            ),
          ),
          const Spacer(),
          // Trophy + toggle
          GestureDetector(
            onTap: () => setState(() => _darkMode = !_darkMode),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.amber.withOpacity(0.2),
                borderRadius: BorderRadius.circular(20),
                border:
                    Border.all(color: Colors.amber.withOpacity(0.5)),
              ),
              child: Row(
                children: [
                  const Text('🏆', style: TextStyle(fontSize: 18)),
                  const SizedBox(width: 4),
                  Text(
                    _darkMode ? '🌙' : '☀️',
                    style: const TextStyle(fontSize: 14),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildUnitSection(int idx) {
    final unit = _units[idx];
    final isLocked = idx > 0 && _units[idx - 1].progress < 0.5;

    return Column(
      children: [
        // Unit header card
        _buildUnitCard(unit, isLocked),
        const SizedBox(height: 16),
        // Node path
        _buildNodePath(unit, isLocked, idx),
      ],
    );
  }

  Widget _buildUnitCard(_UnitData unit, bool isLocked) {
    return Container(
      decoration: BoxDecoration(
        color: isLocked
            ? Colors.grey.shade700
            : unit.color,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
              color: (isLocked ? Colors.black : unit.color)
                  .withOpacity(0.3),
              blurRadius: 12,
              offset: const Offset(0, 4))
        ],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: isLocked ? null : () => _openUnit(unit),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Container(
                  width: 48,
                  height: 48,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.2),
                    shape: BoxShape.circle,
                  ),
                  child: isLocked
                      ? const Icon(Icons.lock,
                          color: Colors.white70, size: 22)
                      : Icon(unit.icon, color: Colors.white, size: 24),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'UNIDAD ${unit.unitNum}',
                        style: TextStyle(
                          color: Colors.white.withOpacity(0.7),
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 1,
                        ),
                      ),
                      Text(
                        unit.title,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 15,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                      Text(
                        unit.subtitle,
                        style: TextStyle(
                          color: Colors.white.withOpacity(0.75),
                          fontSize: 11,
                        ),
                      ),
                    ],
                  ),
                ),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    // Progress
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        '${(unit.progress * 100).toInt()}%',
                        style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.w800,
                          fontSize: 13,
                        ),
                      ),
                    ),
                    const SizedBox(height: 4),
                    // List icon
                    Icon(Icons.format_list_bulleted,
                        color: Colors.white.withOpacity(0.7), size: 18),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildNodePath(_UnitData unit, bool isLocked, int unitIdx) {
    final nodes = unit.exercises;
    // Alternating left/right zigzag layout (mirroring Image 2)
    final alignments = [
      Alignment.centerLeft,
      Alignment.centerRight,
      Alignment.centerLeft,
      Alignment.centerRight,
    ];

    return Column(
      children: [
        for (int i = 0; i < nodes.length; i++) ...[
          _buildGameNode(
            node: nodes[i],
            alignment: alignments[i % alignments.length],
            isActive: !isLocked && nodes[i].unlocked,
            isCurrent: !isLocked &&
                nodes[i].unlocked &&
                (i == nodes.length - 1 ||
                    !nodes[i + 1].unlocked),
            color: isLocked ? Colors.grey : unit.color,
            pulseAnim: _pulse,
            onTap: () {
              if (!isLocked && nodes[i].unlocked) {
                _openExercise(unit, nodes[i]);
              } else if (isLocked || !nodes[i].unlocked) {
                ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                  content: Text(isLocked
                      ? '🔒 Completa la unidad anterior primero'
                      : '🔒 Completa los ejercicios anteriores'),
                  duration: const Duration(seconds: 2),
                ));
              }
            },
          ),
          if (i < nodes.length - 1)
            _buildConnector(
              fromRight: alignments[i % alignments.length] ==
                  Alignment.centerRight,
              color: isLocked ? Colors.white24 : unit.color,
            ),
        ],
      ],
    );
  }

  Widget _buildGameNode({
    required _ExData node,
    required Alignment alignment,
    required bool isActive,
    required bool isCurrent,
    required Color color,
    required Animation<double> pulseAnim,
    required VoidCallback onTap,
  }) {
    final locked = !isActive;

    Widget nodeWidget = GestureDetector(
      onTap: onTap,
      child: AnimatedBuilder(
        animation: pulseAnim,
        builder: (_, child) => Transform.scale(
          scale: isCurrent ? pulseAnim.value : 1.0,
          child: child,
        ),
        child: Container(
          width: 72,
          height: 72,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: locked
                ? Colors.grey.shade500
                : isCurrent
                    ? color
                    : color.withOpacity(0.85),
            border: Border.all(
              color: locked
                  ? Colors.white24
                  : isCurrent
                      ? Colors.white
                      : Colors.white.withOpacity(0.5),
              width: isCurrent ? 3 : 2,
            ),
            boxShadow: [
              BoxShadow(
                color: locked
                    ? Colors.black26
                    : isCurrent
                        ? color.withOpacity(0.6)
                        : color.withOpacity(0.3),
                blurRadius: isCurrent ? 20 : 8,
                spreadRadius: isCurrent ? 2 : 0,
              ),
            ],
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                locked ? Icons.lock : node.icon,
                color: Colors.white,
                size: 28,
              ),
            ],
          ),
        ),
      ),
    );

    // Label below node
    Widget fullNode = Column(
      children: [
        nodeWidget,
        const SizedBox(height: 6),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
          decoration: BoxDecoration(
            color: locked
                ? Colors.black26
                : color.withOpacity(0.85),
            borderRadius: BorderRadius.circular(10),
          ),
          child: Text(
            node.title,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 10,
              fontWeight: FontWeight.w700,
            ),
            textAlign: TextAlign.center,
            maxLines: 2,
          ),
        ),
      ],
    );

    return Padding(
      padding: EdgeInsets.only(
        left: alignment == Alignment.centerLeft ? 30 : 110,
        right: alignment == Alignment.centerRight ? 30 : 110,
      ),
      child: Align(
        alignment: alignment,
        child: fullNode,
      ),
    );
  }

  Widget _buildConnector(
      {required bool fromRight, required Color color}) {
    return SizedBox(
      height: 50,
      child: CustomPaint(
        size: const Size(double.infinity, 50),
        painter: _ConnectorPainter(fromRight: fromRight, color: color),
      ),
    );
  }

  void _openUnit(_UnitData unit) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => UnitNodesScreen(
          title: 'UNIDAD ${unit.unitNum} – ${unit.title}',
          unitColor: unit.color,
          exercises: unit.exercises
              .map((e) => {'title': e.title, 'icon': e.icon, 'unlocked': e.unlocked})
              .toList(),
        ),
      ),
    );
  }

  void _openExercise(_UnitData unit, _ExData ex) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (_) => _ExerciseSheet(unit: unit, exercise: ex),
    );
  }
}

// ─── Custom painters ──────────────────────────────────────────────────────────

class _StarsPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..color = Colors.white;
    final rand = [
      Offset(50, 80), Offset(120, 40), Offset(200, 100), Offset(280, 30),
      Offset(320, 120), Offset(80, 180), Offset(160, 220), Offset(250, 170),
      Offset(340, 200), Offset(30, 300), Offset(140, 350), Offset(220, 290),
      Offset(300, 380), Offset(70, 450), Offset(180, 420), Offset(260, 470),
      Offset(350, 440), Offset(100, 520), Offset(230, 560), Offset(310, 510),
    ];
    for (final p in rand) {
      canvas.drawCircle(p, 2, paint);
      canvas.drawCircle(p * 0.8 + Offset(size.width * 0.3, size.height * 0.1), 1.5, paint..color = Colors.white70);
    }
    // Draw background gradient via rect
    final gradient = const LinearGradient(
      begin: Alignment.topCenter,
      end: Alignment.bottomCenter,
      colors: [Color(0xFF0D1117), Color(0xFF161B22)],
    ).createShader(Rect.fromLTWH(0, 0, size.width, size.height));
    canvas.drawRect(
      Rect.fromLTWH(0, 0, size.width, size.height),
      Paint()..shader = gradient,
    );
    // Redraw stars on top
    final starPaint = Paint()..color = Colors.white;
    for (final p in rand) {
      if (p.dx < size.width && p.dy < size.height) {
        canvas.drawCircle(p, 2, starPaint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class _CloudsPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    // Sky gradient
    final gradient = const LinearGradient(
      begin: Alignment.topCenter,
      end: Alignment.bottomCenter,
      colors: [Color(0xFF87CEEB), Color(0xFFB0E2FF)],
    ).createShader(Rect.fromLTWH(0, 0, size.width, size.height));
    canvas.drawRect(
      Rect.fromLTWH(0, 0, size.width, size.height),
      Paint()..shader = gradient,
    );
    // Clouds
    _drawCloud(canvas, size.width * 0.1, size.height * 0.12, 60);
    _drawCloud(canvas, size.width * 0.55, size.height * 0.08, 50);
    _drawCloud(canvas, size.width * 0.75, size.height * 0.22, 45);
    _drawCloud(canvas, size.width * 0.25, size.height * 0.35, 55);
    _drawCloud(canvas, size.width * 0.6, size.height * 0.50, 40);
    _drawCloud(canvas, size.width * 0.1, size.height * 0.65, 50);
    _drawCloud(canvas, size.width * 0.8, size.height * 0.75, 45);
  }

  void _drawCloud(Canvas canvas, double x, double y, double r) {
    final paint = Paint()..color = Colors.white.withOpacity(0.7);
    canvas.drawCircle(Offset(x, y), r * 0.6, paint);
    canvas.drawCircle(Offset(x + r * 0.5, y + r * 0.1), r * 0.5, paint);
    canvas.drawCircle(Offset(x - r * 0.5, y + r * 0.1), r * 0.45, paint);
    canvas.drawCircle(Offset(x + r * 0.9, y + r * 0.3), r * 0.4, paint);
    canvas.drawCircle(Offset(x - r * 0.8, y + r * 0.3), r * 0.38, paint);
    canvas.drawRect(
      Rect.fromLTWH(x - r * 0.8, y + r * 0.1, r * 1.7, r * 0.4),
      paint,
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class _ConnectorPainter extends CustomPainter {
  final bool fromRight;
  final Color color;
  const _ConnectorPainter({required this.fromRight, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color.withOpacity(0.6)
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    final dotPaint = Paint()
      ..color = color.withOpacity(0.4)
      ..style = PaintingStyle.fill;

    final startX = fromRight ? size.width * 0.72 : size.width * 0.28;
    final endX = fromRight ? size.width * 0.28 : size.width * 0.72;

    final path = Path()
      ..moveTo(startX, 0)
      ..cubicTo(
        startX,
        size.height * 0.5,
        endX,
        size.height * 0.5,
        endX,
        size.height,
      );

    canvas.drawPath(path, paint);

    // Arrow head
    final arrowPaint = Paint()
      ..color = color.withOpacity(0.6)
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;
    canvas.drawLine(
      Offset(endX - 6, size.height - 10),
      Offset(endX, size.height),
      arrowPaint,
    );
    canvas.drawLine(
      Offset(endX + 6, size.height - 10),
      Offset(endX, size.height),
      arrowPaint,
    );

    // Dots along path
    for (double t = 0.2; t < 1.0; t += 0.25) {
      final x = _cubicBezier(startX, startX, endX, endX, t);
      final y = _cubicBezier(0, size.height * 0.5, size.height * 0.5, size.height, t);
      canvas.drawCircle(Offset(x, y), 3, dotPaint);
    }
  }

  double _cubicBezier(double p0, double p1, double p2, double p3, double t) {
    return (1 - t) * (1 - t) * (1 - t) * p0 +
        3 * (1 - t) * (1 - t) * t * p1 +
        3 * (1 - t) * t * t * p2 +
        t * t * t * p3;
  }

  @override
  bool shouldRepaint(covariant CustomPainter old) => false;
}

// ─── Data models ─────────────────────────────────────────────────────────────

class _UnitData {
  final String unitNum, title, subtitle;
  final IconData icon;
  final Color color;
  final double progress;
  final int streak;
  final List<_ExData> exercises;

  const _UnitData({
    required this.unitNum,
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.color,
    required this.progress,
    required this.streak,
    required this.exercises,
  });
}

class _ExData {
  final String title;
  final IconData icon;
  final bool unlocked;
  const _ExData(this.title, this.icon, this.unlocked);
}

// ─── Exercise bottom sheet ────────────────────────────────────────────────────

class _ExerciseSheet extends StatelessWidget {
  final _UnitData unit;
  final _ExData exercise;

  const _ExerciseSheet({required this.unit, required this.exercise});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 40,
            height: 4,
            decoration: BoxDecoration(
              color: Colors.grey.shade300,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(height: 20),
          Container(
            width: 72,
            height: 72,
            decoration: BoxDecoration(
              color: unit.color,
              shape: BoxShape.circle,
            ),
            child: Icon(exercise.icon, color: Colors.white, size: 36),
          ),
          const SizedBox(height: 16),
          Text(
            exercise.title,
            style: const TextStyle(
                fontSize: 20, fontWeight: FontWeight.w800),
          ),
          const SizedBox(height: 6),
          Text(
            'Unidad ${unit.unitNum} – ${unit.title}',
            style: TextStyle(
                color: unit.color,
                fontWeight: FontWeight.w600,
                fontSize: 13),
          ),
          const SizedBox(height: 20),
          ElevatedButton.icon(
            onPressed: () => Navigator.pop(context),
            icon: const Icon(Icons.play_arrow),
            label: const Text('COMENZAR'),
            style: ElevatedButton.styleFrom(
              backgroundColor: unit.color,
              minimumSize: const Size(double.infinity, 50),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(14)),
            ),
          ),
          const SizedBox(height: 10),
        ],
      ),
    );
  }
}