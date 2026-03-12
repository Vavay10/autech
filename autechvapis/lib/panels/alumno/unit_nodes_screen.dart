// lib/modules/unit_nodes_screen.dart
import 'package:flutter/material.dart';
import '../../theme.dart';

class UnitNodesScreen extends StatefulWidget {
  final String title;
  final Color unitColor;
  final List<Map<String, dynamic>> exercises;

  const UnitNodesScreen({
    super.key,
    required this.title,
    this.unitColor = AppColors.primary,
    this.exercises = const [],
  });

  @override
  State<UnitNodesScreen> createState() => _UnitNodesScreenState();
}

class _UnitNodesScreenState extends State<UnitNodesScreen>
    with TickerProviderStateMixin {
  late AnimationController _pulseCtrl;
  late Animation<double> _pulse;
  bool _darkMode = false;

  // Default exercises if none provided
  List<Map<String, dynamic>> get _nodes {
    if (widget.exercises.isNotEmpty) return widget.exercises;
    return [
      {'title': 'Conceptos Base', 'icon': Icons.menu_book_rounded, 'unlocked': true},
      {'title': 'Ejemplos Resueltos', 'icon': Icons.extension_rounded, 'unlocked': true},
      {'title': 'Simulación', 'icon': Icons.code_rounded, 'unlocked': false},
      {'title': 'Ejercicios', 'icon': Icons.quiz_outlined, 'unlocked': false},
    ];
  }

  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(
        vsync: this, duration: const Duration(seconds: 2))
      ..repeat(reverse: true);
    _pulse = Tween<double>(begin: 0.93, end: 1.07).animate(
        CurvedAnimation(parent: _pulseCtrl, curve: Curves.easeInOut));
  }

  @override
  void dispose() {
    _pulseCtrl.dispose();
    super.dispose();
  }

  // Current node = first unlocked but not yet "completed" (simplified: last unlocked)
  int get _currentNodeIdx {
    int last = 0;
    for (int i = 0; i < _nodes.length; i++) {
      if (_nodes[i]['unlocked'] == true) last = i;
    }
    return last;
  }

  @override
  Widget build(BuildContext context) {
    final color = widget.unitColor;

    return Scaffold(
      backgroundColor:
          _darkMode ? const Color(0xFF0D1117) : const Color(0xFF87CEEB),
      body: Stack(
        children: [
          // Background
          _darkMode ? _StarsBg() : _SkyBg(),
          SafeArea(
            child: Column(
              children: [
                // App Bar
                _buildAppBar(color),
                // Level card
                Padding(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  child: _buildLevelCard(color),
                ),
                // Node path
                Expanded(
                  child: SingleChildScrollView(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 16, vertical: 8),
                    child: Column(
                      children: [
                        for (int i = 0; i < _nodes.length; i++) ...[
                          _buildNode(i, color),
                          if (i < _nodes.length - 1)
                            _buildConnectorLine(i, color),
                        ],
                        const SizedBox(height: 60),
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

  Widget _buildAppBar(Color color) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(8, 4, 16, 0),
      child: Row(
        children: [
          IconButton(
            icon: Container(
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.2),
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.arrow_back,
                  color: Colors.white, size: 20),
            ),
            onPressed: () => Navigator.pop(context),
          ),
          const Spacer(),
          Text(
            widget.title,
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.w800,
              fontSize: 15,
              shadows: [Shadow(color: Colors.black38, blurRadius: 4)],
            ),
          ),
          const Spacer(),
          GestureDetector(
            onTap: () => setState(() => _darkMode = !_darkMode),
            child: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.2),
                shape: BoxShape.circle,
              ),
              child: Text(
                _darkMode ? '☀️' : '🌙',
                style: const TextStyle(fontSize: 16),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLevelCard(Color color) {
    final completed = _nodes.where((n) => n['unlocked'] == true).length;
    return Container(
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
              color: color.withOpacity(0.4),
              blurRadius: 16,
              offset: const Offset(0, 4))
        ],
      ),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      child: Row(
        children: [
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Text(
                  'NIVEL ${completed > 0 ? completed : 1}',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 11,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 1,
                  ),
                ),
              ),
              const SizedBox(height: 4),
              Text(
                widget.title.split('–').last.trim(),
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
          const Spacer(),
          // Progress circle
          SizedBox(
            width: 48,
            height: 48,
            child: Stack(
              alignment: Alignment.center,
              children: [
                CircularProgressIndicator(
                  value: completed / _nodes.length,
                  backgroundColor: Colors.white.withOpacity(0.2),
                  valueColor:
                      const AlwaysStoppedAnimation(Colors.white),
                  strokeWidth: 4,
                ),
                Text(
                  '$completed/${_nodes.length}',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 11,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 12),
          Icon(Icons.format_list_bulleted,
              color: Colors.white.withOpacity(0.8), size: 22),
        ],
      ),
    );
  }

  Widget _buildNode(int idx, Color color) {
    final node = _nodes[idx];
    final unlocked = node['unlocked'] == true;
    final isCurrent = idx == _currentNodeIdx && unlocked;
    final icon = node['icon'] as IconData? ?? Icons.star;
    final title = node['title'] as String? ?? 'Paso ${idx + 1}';

    // Alignment pattern: L, R, L, R
    final alignments = [
      Alignment.centerLeft,
      Alignment.centerRight,
      Alignment.centerLeft,
      Alignment.centerRight,
    ];
    final alignment = alignments[idx % alignments.length];
    final isLeft = alignment == Alignment.centerLeft;

    return Align(
      alignment: alignment,
      child: Padding(
        padding: EdgeInsets.only(
          left: isLeft ? 24 : 96,
          right: isLeft ? 96 : 24,
        ),
        child: Column(
          children: [
            GestureDetector(
              onTap: () {
                if (unlocked) {
                  _showNodeDetail(idx, node, color);
                } else {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text('🔒 Completa los pasos anteriores para desbloquear'),
                      duration: Duration(seconds: 2),
                    ),
                  );
                }
              },
              child: AnimatedBuilder(
                animation: _pulse,
                builder: (_, child) => Transform.scale(
                  scale: isCurrent ? _pulse.value : 1.0,
                  child: child,
                ),
                child: Container(
                  width: 76,
                  height: 76,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: unlocked
                        ? (isCurrent ? color : color.withOpacity(0.8))
                        : Colors.grey.shade500,
                    border: Border.all(
                      color: unlocked
                          ? (isCurrent
                              ? Colors.white
                              : Colors.white.withOpacity(0.4))
                          : Colors.white24,
                      width: isCurrent ? 3.5 : 2,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: unlocked
                            ? (isCurrent
                                ? color.withOpacity(0.7)
                                : color.withOpacity(0.25))
                            : Colors.black26,
                        blurRadius: isCurrent ? 24 : 8,
                        spreadRadius: isCurrent ? 3 : 0,
                      ),
                    ],
                  ),
                  // CENTERED ICON
                  child: Center(
                    child: Icon(
                      unlocked ? icon : Icons.lock_rounded,
                      color: Colors.white,
                      size: 32,
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 8),
            // Label
            Container(
              padding: const EdgeInsets.symmetric(
                  horizontal: 12, vertical: 5),
              decoration: BoxDecoration(
                color: unlocked
                    ? color.withOpacity(0.85)
                    : Colors.black.withOpacity(0.35),
                borderRadius: BorderRadius.circular(12),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.15),
                    blurRadius: 4,
                  ),
                ],
              ),
              child: Text(
                title,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                ),
                textAlign: TextAlign.center,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ),
            if (!unlocked)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text(
                  '🔒 Bloqueado',
                  style: TextStyle(
                    color: Colors.white.withOpacity(0.5),
                    fontSize: 10,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildConnectorLine(int fromIdx, Color color) {
    final fromNode = _nodes[fromIdx];
    final alignments = [
      Alignment.centerLeft,
      Alignment.centerRight,
      Alignment.centerLeft,
      Alignment.centerRight,
    ];
    final fromRight =
        alignments[fromIdx % alignments.length] == Alignment.centerRight;
    final unlocked = fromNode['unlocked'] == true;

    return SizedBox(
      height: 56,
      child: CustomPaint(
        size: const Size(double.infinity, 56),
        painter: _NodeConnectorPainter(
          fromRight: fromRight,
          color: unlocked ? color : Colors.white24,
        ),
      ),
    );
  }

  void _showNodeDetail(int idx, Map<String, dynamic> node, Color color) {
    final title = node['title'] as String? ?? 'Paso ${idx + 1}';
    final icon = node['icon'] as IconData? ?? Icons.star;

    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (_) => Container(
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
            // Centered icon in large circle
            Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                color: color,
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                      color: color.withOpacity(0.3),
                      blurRadius: 16,
                      spreadRadius: 2)
                ],
              ),
              child: Center(
                child: Icon(icon, color: Colors.white, size: 38),
              ),
            ),
            const SizedBox(height: 16),
            Text(
              title,
              style: const TextStyle(
                  fontSize: 20, fontWeight: FontWeight.w800),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 4),
            Text(
              widget.title,
              style: TextStyle(
                  color: color,
                  fontWeight: FontWeight.w600,
                  fontSize: 13),
            ),
            const SizedBox(height: 20),
            // Progress indicator for this node
            _NodeProgressCard(
              stepNum: idx + 1,
              total: _nodes.length,
              color: color,
            ),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: () => Navigator.pop(context),
              icon: const Icon(Icons.play_arrow),
              label: const Text('COMENZAR EJERCICIO'),
              style: ElevatedButton.styleFrom(
                backgroundColor: color,
                minimumSize: const Size(double.infinity, 50),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14)),
              ),
            ),
            const SizedBox(height: 8),
          ],
        ),
      ),
    );
  }
}

// ─── Node Progress Card ───────────────────────────────────────────────────────
class _NodeProgressCard extends StatelessWidget {
  final int stepNum, total;
  final Color color;

  const _NodeProgressCard(
      {required this.stepNum, required this.total, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: color.withOpacity(0.06),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.2)),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _Stat('Paso', '$stepNum / $total', Icons.format_list_numbered, color),
          _Stat('XP', '+50', Icons.star, Colors.amber),
          _Stat('Tiempo', '~10 min', Icons.timer_outlined, AppColors.info),
        ],
      ),
    );
  }
}

class _Stat extends StatelessWidget {
  final String label, value;
  final IconData icon;
  final Color color;
  const _Stat(this.label, this.value, this.icon, this.color);

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Icon(icon, color: color, size: 22),
        const SizedBox(height: 4),
        Text(value,
            style: TextStyle(
                fontWeight: FontWeight.w800, fontSize: 14, color: color)),
        Text(label,
            style: const TextStyle(
                fontSize: 11, color: AppColors.textSecondary)),
      ],
    );
  }
}

// ─── Painters ────────────────────────────────────────────────────────────────

class _NodeConnectorPainter extends CustomPainter {
  final bool fromRight;
  final Color color;
  const _NodeConnectorPainter(
      {required this.fromRight, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    final startX = fromRight ? size.width * 0.7 : size.width * 0.3;
    final endX = fromRight ? size.width * 0.3 : size.width * 0.7;

    final path = Path()
      ..moveTo(startX, 0)
      ..cubicTo(startX, size.height * 0.5, endX, size.height * 0.5,
          endX, size.height);
    canvas.drawPath(path, paint);

    // Small dots
    final dotPaint = Paint()..color = color..style = PaintingStyle.fill;
    for (double t = 0.15; t <= 0.85; t += 0.25) {
      final x = _cubic(startX, startX, endX, endX, t);
      final y = _cubic(0, size.height * 0.5, size.height * 0.5, size.height, t);
      canvas.drawCircle(Offset(x, y), 3.5, dotPaint);
    }
  }

  double _cubic(double p0, double p1, double p2, double p3, double t) =>
      (1 - t) * (1 - t) * (1 - t) * p0 +
      3 * (1 - t) * (1 - t) * t * p1 +
      3 * (1 - t) * t * t * p2 +
      t * t * t * p3;

  @override
  bool shouldRepaint(covariant CustomPainter old) => false;
}

// ─── Background widgets ───────────────────────────────────────────────────────

class _StarsBg extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [Color(0xFF0D1117), Color(0xFF161B22)],
        ),
      ),
      child: CustomPaint(
        painter: _MiniStarsPainter(),
        size: Size.infinite,
      ),
    );
  }
}

class _SkyBg extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [Color(0xFF87CEEB), Color(0xFFB0E2FF)],
        ),
      ),
    );
  }
}

class _MiniStarsPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..color = Colors.white.withOpacity(0.6);
    final positions = [
      const Offset(40, 60), const Offset(120, 30), const Offset(200, 80),
      const Offset(290, 50), const Offset(50, 180), const Offset(160, 140),
      const Offset(260, 200), const Offset(100, 280), const Offset(220, 320),
      const Offset(330, 150), const Offset(70, 400), const Offset(190, 380),
      const Offset(310, 440), const Offset(150, 500), const Offset(280, 520),
    ];
    for (final p in positions) {
      canvas.drawCircle(p, 2, paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}