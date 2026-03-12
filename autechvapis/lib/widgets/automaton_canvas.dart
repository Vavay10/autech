import 'dart:math';
import 'package:flutter/material.dart';
import '../theme.dart';

// ─── Data models ─────────────────────────────────────────────────────────────

class StateNode {
  String id;
  bool isInitial;
  bool isAccepting;

  StateNode({required this.id, this.isInitial = false, this.isAccepting = false});

  StateNode copyWith({String? id, bool? isInitial, bool? isAccepting}) => StateNode(
        id: id ?? this.id,
        isInitial: isInitial ?? this.isInitial,
        isAccepting: isAccepting ?? this.isAccepting,
      );

  Map<String, dynamic> toJson() =>
      {'id': id, 'isInitial': isInitial, 'isAccepting': isAccepting};

  factory StateNode.fromJson(Map<String, dynamic> j) => StateNode(
        id: j['id'] as String,
        isInitial: j['isInitial'] as bool? ?? false,
        isAccepting: j['isAccepting'] as bool? ?? false,
      );
}

class TransitionEdge {
  String from;
  String to;
  String label;

  TransitionEdge({required this.from, required this.to, required this.label});

  Map<String, dynamic> toJson() => {'from': from, 'to': to, 'label': label};

  factory TransitionEdge.fromJson(Map<String, dynamic> j) => TransitionEdge(
        from: j['from'] as String,
        to: j['to'] as String,
        label: j['label'] as String,
      );
}

// ─── AutomatonCanvas widget ───────────────────────────────────────────────────

enum CanvasMode { select, addTransition }

class AutomatonCanvas extends StatefulWidget {
  final List<StateNode> states;
  final List<TransitionEdge> edges;
  final Map<String, Offset>? initialPositions;
  final bool editable;
  final String? animatedState;
  final Set<String>? animatedEdgeKeys;
  final void Function(StateNode added, Offset pos)? onStateAdded;
  final void Function(String id)? onStateDeleted;
  final void Function(StateNode updated)? onStateUpdated;
  final void Function(TransitionEdge added)? onEdgeAdded;
  final void Function(String from, String to)? onEdgeDeleted;
  final void Function(Map<String, Offset> positions)? onPositionsChanged;

  const AutomatonCanvas({
    super.key,
    required this.states,
    required this.edges,
    this.initialPositions,
    this.editable = true,
    this.animatedState,
    this.animatedEdgeKeys,
    this.onStateAdded,
    this.onStateDeleted,
    this.onStateUpdated,
    this.onEdgeAdded,
    this.onEdgeDeleted,
    this.onPositionsChanged,
  });

  @override
  State<AutomatonCanvas> createState() => AutomatonCanvasState();
}

class AutomatonCanvasState extends State<AutomatonCanvas> {
  static const double _R = 28.0;
  static const double _canvasW = 1400.0;
  static const double _canvasH = 900.0;

  final Map<String, Offset> _pos = {};
  CanvasMode _mode = CanvasMode.select;
  String? _edgeSource;

  // FIX: track which state is being dragged via Listener (bypasses gesture arena)
  String? _draggingStateId;

  // ── Zoom / Pan ──────────────────────────────────────────────────────────────
  final TransformationController _tfCtrl = TransformationController();
  double _scale = 1.0;

  Offset _toCanvas(Offset local) {
    final m = Matrix4.inverted(_tfCtrl.value);
    final s = m.storage;
    return Offset(s[0] * local.dx + s[4] * local.dy + s[12],
                  s[1] * local.dx + s[5] * local.dy + s[13]);
  }

  void zoomIn()    => _scaleBy(1.25);
  void zoomOut()   => _scaleBy(0.8);
  void resetZoom() => setState(() { _tfCtrl.value = Matrix4.identity(); _scale = 1.0; });

  void fitToScreen() {
    if (_pos.isEmpty) return;
    final box = context.findRenderObject() as RenderBox?;
    if (box == null) return;
    final sz = box.size;
    final xs = _pos.values.map((p) => p.dx);
    final ys = _pos.values.map((p) => p.dy);
    final minX = xs.reduce(min) - _R - 24;
    final maxX = xs.reduce(max) + _R + 24;
    final minY = ys.reduce(min) - _R - 24;
    final maxY = ys.reduce(max) + _R + 24;
    final cw = maxX - minX, ch = maxY - minY;
    if (cw <= 0 || ch <= 0) return;
    final s = min(sz.width / cw, sz.height / ch).clamp(0.2, 3.0);
    final m = Matrix4.identity()
      ..translate(-minX * s + (sz.width  - cw * s) / 2,
                  -minY * s + (sz.height - ch * s) / 2)
      ..scale(s);
    setState(() { _tfCtrl.value = m; _scale = s; });
  }

  void _scaleBy(double f) {
    final m = _tfCtrl.value.clone()..scale(f);
    final ns = m.getMaxScaleOnAxis();
    if (ns < 0.15 || ns > 5.0) return;
    setState(() { _tfCtrl.value = m; _scale = ns; });
  }

  @override
  void initState() {
    super.initState();
    _syncPositions();
    _tfCtrl.addListener(() {
      final s = _tfCtrl.value.getMaxScaleOnAxis();
      if ((s - _scale).abs() > 0.001) setState(() => _scale = s);
    });
  }

  @override
  void dispose() { _tfCtrl.dispose(); super.dispose(); }

  @override
  void didUpdateWidget(AutomatonCanvas old) {
    super.didUpdateWidget(old);
    for (final s in widget.states) {
      if (!_pos.containsKey(s.id)) {
        _pos[s.id] = _autoPlace();
      }
    }
    _pos.removeWhere((k, _) => !widget.states.any((s) => s.id == k));
  }

  void _syncPositions() {
    if (widget.initialPositions != null) {
      _pos.addAll(widget.initialPositions!);
    }
    for (int i = 0; i < widget.states.length; i++) {
      final s = widget.states[i];
      if (!_pos.containsKey(s.id)) {
        final angle = (2 * pi * i) / max(widget.states.length, 1);
        final cx = _canvasW / 2 + 160 * cos(angle - pi / 2);
        final cy = _canvasH / 2 + 130 * sin(angle - pi / 2);
        _pos[s.id] = Offset(cx, cy);
      }
    }
  }

  /// FIX: _autoPlace now spreads states out so they don't stack.
  Offset _autoPlace() {
    final n = _pos.length;
    // Spiral placement to avoid overlap
    final angle = n * 2.399; // golden angle in radians
    final r = 80.0 + 60.0 * sqrt(n.toDouble());
    final cx = (_canvasW / 2 + r * cos(angle)).clamp(_R + 20, _canvasW - _R - 20);
    final cy = (_canvasH / 2 + r * sin(angle)).clamp(_R + 20, _canvasH - _R - 20);
    return Offset(cx, cy);
  }

  // ─── Public layout API ─────────────────────────────────────────────────────

  void layoutCircle() {
    setState(() {
      final n = widget.states.length;
      for (int i = 0; i < n; i++) {
        final angle = (2 * pi * i) / max(n, 1) - pi / 2;
        _pos[widget.states[i].id] = Offset(
          _canvasW / 2 + 160 * cos(angle),
          _canvasH / 2 + 130 * sin(angle),
        );
      }
    });
    widget.onPositionsChanged?.call(Map.from(_pos));
  }

  void layoutRow() {
    setState(() {
      final n = widget.states.length;
      for (int i = 0; i < n; i++) {
        _pos[widget.states[i].id] = Offset(100.0 + i * 170.0, _canvasH / 2);
      }
    });
    widget.onPositionsChanged?.call(Map.from(_pos));
  }

  // ─── Gesture handlers ──────────────────────────────────────────────────────

  /// FIX: Canvas tap now lives OUTSIDE InteractiveViewer to avoid viewport shifts.
  void _onCanvasTap(TapUpDetails d) {
    if (!widget.editable) return;
    if (_mode == CanvasMode.addTransition) {
      setState(() { _edgeSource = null; _mode = CanvasMode.select; });
      return;
    }
    _showAddStateDialog(_toCanvas(d.localPosition));
  }

  void _onStateTap(String id) {
    if (!widget.editable) return;
    if (_mode == CanvasMode.addTransition) {
      if (_edgeSource == null) {
        setState(() => _edgeSource = id);
      } else {
        _showAddEdgeDialog(_edgeSource!, id);
        setState(() { _edgeSource = null; _mode = CanvasMode.select; });
      }
    }
  }

  // ─── Dialogs ───────────────────────────────────────────────────────────────

  void _showAddStateDialog(Offset pos) {
    final ctrl = TextEditingController(text: 'q${widget.states.length}');
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Nuevo estado'),
        content: TextField(
          controller: ctrl,
          autofocus: true,
          decoration: const InputDecoration(labelText: 'Nombre del estado'),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancelar')),
          ElevatedButton(
            onPressed: () {
              final name = ctrl.text.trim();
              Navigator.pop(context);
              if (name.isEmpty) return;
              final node = StateNode(id: name);
              setState(() => _pos[name] = pos);
              widget.onStateAdded?.call(node, pos);
            },
            child: const Text('Añadir'),
          ),
        ],
      ),
    );
  }

  void _showAddEdgeDialog(String from, String to) {
    final ctrl = TextEditingController();
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: Text('Transición $from → $to'),
        content: TextField(
          controller: ctrl,
          autofocus: true,
          decoration: const InputDecoration(labelText: 'Etiqueta (ej: a, 0,1)'),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancelar')),
          ElevatedButton(
            onPressed: () {
              final lbl = ctrl.text.trim();
              Navigator.pop(context);
              if (lbl.isEmpty) return;
              widget.onEdgeAdded?.call(TransitionEdge(from: from, to: to, label: lbl));
            },
            child: const Text('Añadir'),
          ),
        ],
      ),
    );
  }

  void _showStateMenu(String id) {
    final node = widget.states.firstWhere((s) => s.id == id);
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) => _StateMenuSheet(
        node: node,
        onSetInitial: () {
          Navigator.pop(context);
          widget.onStateUpdated?.call(node.copyWith(isInitial: !node.isInitial));
        },
        onSetAccepting: () {
          Navigator.pop(context);
          widget.onStateUpdated?.call(node.copyWith(isAccepting: !node.isAccepting));
        },
        onDelete: () {
          Navigator.pop(context);
          widget.onStateDeleted?.call(id);
        },
        onDrawEdge: () {
          Navigator.pop(context);
          setState(() { _edgeSource = id; _mode = CanvasMode.addTransition; });
        },
      ),
    );
  }

  // ─── Toolbar ───────────────────────────────────────────────────────────────

  void startDrawingEdge() => setState(() { _mode = CanvasMode.addTransition; _edgeSource = null; });
  void cancelMode() => setState(() { _mode = CanvasMode.select; _edgeSource = null; });

  // ─── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        if (widget.editable) _buildToolbar(),
        Expanded(
          child: Container(
            decoration: BoxDecoration(
              color: AppColors.surfaceAlt,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: AppColors.border),
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: Stack(
                children: [
                  // ── BACKGROUND DOT GRID ─────────────────────────────────
                  Positioned.fill(
                    child: CustomPaint(painter: _GridPainter()),
                  ),

                  // ── CANVAS TAP (outside InteractiveViewer to avoid viewport drift) ──
                  if (widget.editable && _mode != CanvasMode.addTransition)
                    Positioned.fill(
                      child: GestureDetector(
                        behavior: HitTestBehavior.translucent,
                        onTapUp: _onCanvasTap,
                      ),
                    ),

                  // ── InteractiveViewer (zoom + pan only, no state nodes here) ──
                  Positioned.fill(
                    child: InteractiveViewer(
                      transformationController: _tfCtrl,
                      // FIX: disable pan while dragging a state node
                      panEnabled: _draggingStateId == null,
                      scaleEnabled: _draggingStateId == null,
                      minScale: 0.15,
                      maxScale: 5.0,
                      constrained: false,
                      child: SizedBox(
                        width: _canvasW,
                        height: _canvasH,
                        child: Stack(
                          children: [
                            // Arrows only (no state circles here)
                            Positioned.fill(
                              child: CustomPaint(
                                painter: _ArrowsPainter(
                                  states: widget.states,
                                  edges: widget.edges,
                                  positions: _pos,
                                  animatedState: widget.animatedState,
                                  animatedEdgeKeys: widget.animatedEdgeKeys,
                                  edgeSource: _edgeSource,
                                ),
                              ),
                            ),
                            // State nodes – each uses Listener for drag
                            ...widget.states.map((s) => _buildStateNode(s)),
                          ],
                        ),
                      ),
                    ),
                  ),

                  // ── Mode hint banner ────────────────────────────────────
                  if (_mode == CanvasMode.addTransition)
                    Positioned(
                      top: 10, left: 0, right: 0,
                      child: IgnorePointer(
                        child: Center(
                          child: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                            decoration: BoxDecoration(
                              color: AppColors.primary,
                              borderRadius: BorderRadius.circular(20),
                              boxShadow: const [BoxShadow(color: Colors.black26, blurRadius: 6)],
                            ),
                            child: Text(
                              _edgeSource == null
                                  ? 'Toca el estado ORIGEN de la transición'
                                  : 'Ahora toca el estado DESTINO',
                              style: const TextStyle(color: Colors.white, fontSize: 13),
                            ),
                          ),
                        ),
                      ),
                    ),

                  // ── Zoom controls ────────────────────────────────────────
                  Positioned(
                    top: 10, right: 10,
                    child: _ZoomPanel(
                      scale: _scale,
                      onZoomIn: zoomIn,
                      onZoomOut: zoomOut,
                      onFit: fitToScreen,
                      onReset: resetZoom,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildToolbar() {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: Row(
          children: [
            _ToolBtn(
              icon: Icons.add_circle_outline,
              label: 'Estado',
              active: false,
              // FIX: use _autoPlace() instead of hardcoded Offset(200,200)
              onTap: () => _showAddStateDialog(_autoPlace()),
            ),
            const SizedBox(width: 8),
            _ToolBtn(
              icon: Icons.arrow_forward,
              label: 'Transición',
              active: _mode == CanvasMode.addTransition,
              onTap: _mode == CanvasMode.addTransition ? cancelMode : startDrawingEdge,
            ),
            const SizedBox(width: 8),
            _ToolBtn(icon: Icons.account_tree_outlined, label: 'Círculo', active: false, onTap: layoutCircle),
            const SizedBox(width: 8),
            _ToolBtn(icon: Icons.horizontal_rule, label: 'Fila', active: false, onTap: layoutRow),
            if (widget.states.isEmpty)
              const Padding(
                padding: EdgeInsets.only(left: 12),
                child: Text(
                  'Toca el canvas para añadir un estado',
                  style: TextStyle(color: AppColors.textHint, fontSize: 12),
                ),
              ),
          ],
        ),
      ),
    );
  }

  /// FIX: Uses Listener (raw pointer events, bypasses gesture arena) for drag.
  /// This prevents InteractiveViewer from stealing the drag gesture.
  Widget _buildStateNode(StateNode s) {
    final pos = _pos[s.id] ?? const Offset(100, 100);
    final isAnimated = s.id == widget.animatedState;
    final isPending  = s.id == _edgeSource;

    return Positioned(
      left: pos.dx - _R,
      top:  pos.dy - _R,
      width: _R * 2,
      height: _R * 2,
      child: Listener(
        // HitTestBehavior.opaque: this widget fully absorbs the pointer event.
        behavior: HitTestBehavior.opaque,
        onPointerDown: (_) {
          // Only drag in select mode (not while drawing a transition)
          if (widget.editable && _mode == CanvasMode.select) {
            setState(() => _draggingStateId = s.id);
          }
        },
        onPointerMove: (e) {
          if (_draggingStateId == s.id) {
            setState(() {
              final cur = _pos[s.id] ?? const Offset(100, 100);
              _pos[s.id] = Offset(
                (cur.dx + e.delta.dx / _scale).clamp(_R, _canvasW - _R),
                (cur.dy + e.delta.dy / _scale).clamp(_R, _canvasH - _R),
              );
            });
          }
        },
        onPointerUp: (_) {
          if (_draggingStateId == s.id) {
            setState(() => _draggingStateId = null);
            widget.onPositionsChanged?.call(Map.from(_pos));
          }
        },
        onPointerCancel: (_) {
          if (_draggingStateId == s.id) {
            setState(() => _draggingStateId = null);
          }
        },
        child: GestureDetector(
          // Tap handled separately (not drag, so no arena conflict)
          onTap: () {
            if (_mode == CanvasMode.addTransition) {
              _onStateTap(s.id);
            } else if (widget.editable) {
              _showStateMenu(s.id);
            }
          },
          child: CustomPaint(
            painter: _StatePainter(
              node: s,
              isAnimated: isAnimated,
              isPending: isPending,
            ),
            child: Center(
              child: Text(
                s.id,
                style: TextStyle(
                  fontSize: s.id.length > 3 ? 10 : 12,
                  fontWeight: FontWeight.w800,
                  color: Colors.white,
                  shadows: const [Shadow(color: Colors.black38, blurRadius: 2)],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ─── Grid background painter ─────────────────────────────────────────────────

class _GridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = AppColors.border.withOpacity(0.5)
      ..strokeWidth = 1;
    const step = 28.0;
    for (double x = 0; x < size.width; x += step) {
      canvas.drawCircle(Offset(x, 0), 1, paint);
      for (double y = 0; y < size.height; y += step) {
        canvas.drawCircle(Offset(x, y), 1.2, paint..color = AppColors.border.withOpacity(0.35));
      }
    }
  }
  @override
  bool shouldRepaint(_GridPainter _) => false;
}

// ─── State painter ────────────────────────────────────────────────────────────

class _StatePainter extends CustomPainter {
  final StateNode node;
  final bool isAnimated;
  final bool isPending;
  static const double R = 28.0;

  const _StatePainter({required this.node, this.isAnimated = false, this.isPending = false});

  static const Color _colorNormal    = Color(0xFF4CAF50);
  static const Color _colorInitial   = Color(0xFF2E7D32);
  static const Color _colorAccepting = Color(0xFF81C784);
  static const Color _colorActive    = Color(0xFFFFC107);
  static const Color _colorPending   = Color(0xFF00ADB5);

  @override
  void paint(Canvas canvas, Size size) {
    final cx = size.width / 2, cy = size.height / 2;

    Color fill, border;
    if (isAnimated) {
      fill = _colorActive; border = const Color(0xFFFF8F00);
    } else if (isPending) {
      fill = _colorPending; border = AppColors.primaryDark;
    } else if (node.isInitial) {
      fill = _colorInitial; border = const Color(0xFF1B5E20);
    } else if (node.isAccepting) {
      fill = _colorAccepting; border = const Color(0xFF388E3C);
    } else {
      fill = _colorNormal; border = const Color(0xFF2E7D32);
    }

    // Shadow
    canvas.drawCircle(Offset(cx + 2, cy + 3), R, Paint()..color = Colors.black.withOpacity(0.18));
    // Fill
    canvas.drawCircle(Offset(cx, cy), R, Paint()..color = fill);
    // Gloss
    canvas.drawCircle(Offset(cx - R * 0.25, cy - R * 0.3), R * 0.45, Paint()..color = Colors.white.withOpacity(0.18));
    // Border
    canvas.drawCircle(Offset(cx, cy), R, Paint()
      ..color = border
      ..style = PaintingStyle.stroke
      ..strokeWidth = isPending || isAnimated ? 2.8 : 2.0);

    // Double ring for accepting
    if (node.isAccepting) {
      canvas.drawCircle(Offset(cx, cy), R - 5, Paint()
        ..color = Colors.white.withOpacity(0.7)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.8);
    }

    // Initial arrow
    if (node.isInitial) {
      const arrowColor = Color(0xFF1B5E20);
      final paint = Paint()
        ..color = arrowColor
        ..strokeWidth = 2.5
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round;
      canvas.drawLine(Offset(cx - R - 24, cy), Offset(cx - R - 1, cy), paint);
      final path = Path()
        ..moveTo(cx - R, cy)
        ..lineTo(cx - R - 10, cy - 6)
        ..lineTo(cx - R - 10, cy + 6)
        ..close();
      canvas.drawPath(path, Paint()..color = arrowColor);
    }
  }

  @override
  bool shouldRepaint(_StatePainter old) =>
      old.isAnimated != isAnimated || old.isPending != isPending ||
      old.node.isInitial != node.isInitial || old.node.isAccepting != node.isAccepting;
}

// ─── Arrows painter ──────────────────────────────────────────────────────────

class _ArrowsPainter extends CustomPainter {
  final List<StateNode> states;
  final List<TransitionEdge> edges;
  final Map<String, Offset> positions;
  final String? animatedState;
  final Set<String>? animatedEdgeKeys;
  final String? edgeSource;

  static const double R = 28.0;

  const _ArrowsPainter({
    required this.states,
    required this.edges,
    required this.positions,
    this.animatedState,
    this.animatedEdgeKeys,
    this.edgeSource,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final Map<String, List<TransitionEdge>> groups = {};
    for (final e in edges) {
      groups.putIfAbsent('${e.from}→${e.to}', () => []).add(e);
    }
    for (final entry in groups.entries) {
      final parts = entry.key.split('→');
      final from = parts[0], to = parts[1];
      final combinedLabel = entry.value.map((e) => e.label).join(', ');
      final hasBidi = groups.containsKey('$to→$from') && from != to;
      final isActive = animatedEdgeKeys?.contains(entry.key) ?? false;
      _drawEdge(canvas, from, to, combinedLabel, hasBidi: hasBidi, isActive: isActive);
    }
  }

  void _drawEdge(Canvas canvas, String from, String to, String label,
      {bool hasBidi = false, bool isActive = false}) {
    final p1 = positions[from], p2 = positions[to];
    if (p1 == null || p2 == null) return;

    final paint = Paint()
      ..color = isActive ? AppColors.edgeActive : AppColors.edgeColor
      ..strokeWidth = isActive ? 2.4 : 1.8
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    if (from == to) {
      _drawSelfLoop(canvas, p1, label, paint);
    } else if (hasBidi) {
      _drawCurvedArrow(canvas, p1, p2, label, paint, curveDir: 1);
    } else {
      _drawStraightArrow(canvas, p1, p2, label, paint);
    }
  }

  void _drawStraightArrow(Canvas canvas, Offset p1, Offset p2, String label, Paint paint) {
    final dx = p2.dx - p1.dx, dy = p2.dy - p1.dy;
    final dist = sqrt(dx * dx + dy * dy);
    if (dist < 1) return;
    final ux = dx / dist, uy = dy / dist;
    final start = Offset(p1.dx + ux * R, p1.dy + uy * R);
    final end   = Offset(p2.dx - ux * (R + 4), p2.dy - uy * (R + 4));
    canvas.drawLine(start, end, paint);
    _drawArrowHead(canvas, end, atan2(uy, ux), paint.color);
    final mid = Offset((start.dx + end.dx) / 2, (start.dy + end.dy) / 2);
    _drawLabel(canvas, Offset(mid.dx - uy * 16, mid.dy + ux * 16), label);
  }

  void _drawCurvedArrow(Canvas canvas, Offset p1, Offset p2, String label, Paint paint, {int curveDir = 1}) {
    final dx = p2.dx - p1.dx, dy = p2.dy - p1.dy;
    final dist = sqrt(dx * dx + dy * dy);
    if (dist < 1) return;
    final ux = dx / dist, uy = dy / dist;
    final nx = -uy * curveDir, ny = ux * curveDir;
    final ctrl = Offset((p1.dx + p2.dx) / 2 + nx * 50, (p1.dy + p2.dy) / 2 + ny * 50);
    final ang1 = atan2(ctrl.dy - p1.dy, ctrl.dx - p1.dx);
    final ang2 = atan2(p2.dy - ctrl.dy, p2.dx - ctrl.dx);
    final start = Offset(p1.dx + cos(ang1) * R, p1.dy + sin(ang1) * R);
    final end   = Offset(p2.dx - cos(ang2) * (R + 4), p2.dy - sin(ang2) * (R + 4));
    final path  = Path()
      ..moveTo(start.dx, start.dy)
      ..quadraticBezierTo(ctrl.dx, ctrl.dy, end.dx, end.dy);
    canvas.drawPath(path, paint);
    _drawArrowHead(canvas, end, ang2, paint.color);
    _drawLabel(canvas, Offset(ctrl.dx, ctrl.dy - 12), label);
  }

  void _drawSelfLoop(Canvas canvas, Offset p, String label, Paint paint) {
    final top  = Offset(p.dx, p.dy - R - 45);
    final cp1  = Offset(p.dx - 40, p.dy - R - 85);
    final cp2  = Offset(p.dx + 40, p.dy - R - 85);
    final start = Offset(p.dx - R * 0.5, p.dy - R * 0.87);
    final end   = Offset(p.dx + R * 0.5, p.dy - R * 0.87);
    final path  = Path()
      ..moveTo(start.dx, start.dy)
      ..cubicTo(cp1.dx, cp1.dy, cp2.dx, cp2.dy, end.dx, end.dy);
    canvas.drawPath(path, paint);
    _drawArrowHead(canvas, end, pi * 0.25, paint.color);
    _drawLabel(canvas, Offset(top.dx, top.dy - 14), label);
  }

  void _drawArrowHead(Canvas canvas, Offset tip, double angle, Color color) {
    const len = 9.0, spread = 0.4;
    final path = Path()
      ..moveTo(tip.dx, tip.dy)
      ..lineTo(tip.dx - len * cos(angle - spread), tip.dy - len * sin(angle - spread))
      ..lineTo(tip.dx - len * cos(angle + spread), tip.dy - len * sin(angle + spread))
      ..close();
    canvas.drawPath(path, Paint()..color = color);
  }

  void _drawLabel(Canvas canvas, Offset pos, String text) {
    final tp = TextPainter(
      text: TextSpan(text: text, style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: AppColors.textPrimary)),
      textDirection: TextDirection.ltr,
    )..layout();
    final bgRect = Rect.fromCenter(center: pos, width: tp.width + 10, height: tp.height + 6);
    canvas.drawRRect(RRect.fromRectAndRadius(bgRect, const Radius.circular(4)),
        Paint()..color = Colors.white.withOpacity(0.92));
    canvas.drawRRect(RRect.fromRectAndRadius(bgRect, const Radius.circular(4)),
        Paint()..color = AppColors.border..style = PaintingStyle.stroke..strokeWidth = 0.8);
    tp.paint(canvas, pos - Offset(tp.width / 2, tp.height / 2));
  }

  @override
  bool shouldRepaint(_ArrowsPainter old) => true;
}

// ─── Zoom panel ───────────────────────────────────────────────────────────────

class _ZoomPanel extends StatelessWidget {
  final double scale;
  final VoidCallback onZoomIn, onZoomOut, onFit, onReset;
  const _ZoomPanel({required this.scale, required this.onZoomIn,
      required this.onZoomOut, required this.onFit, required this.onReset});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.94),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: AppColors.border),
        boxShadow: const [BoxShadow(color: Colors.black12, blurRadius: 6, offset: Offset(0, 2))],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          _ZBtn(icon: Icons.add, tip: 'Acercar', onTap: onZoomIn),
          _ZDivider(),
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 3, horizontal: 4),
            child: Text('${(scale * 100).round()}%',
                style: const TextStyle(fontSize: 9, fontWeight: FontWeight.w700, color: AppColors.textSecondary)),
          ),
          _ZDivider(),
          _ZBtn(icon: Icons.remove, tip: 'Alejar', onTap: onZoomOut),
          _ZDivider(),
          _ZBtn(icon: Icons.fit_screen, tip: 'Ajustar', onTap: onFit),
          _ZDivider(),
          _ZBtn(icon: Icons.youtube_searched_for, tip: 'Restablecer', onTap: onReset),
        ],
      ),
    );
  }
}

class _ZBtn extends StatelessWidget {
  final IconData icon; final String tip; final VoidCallback onTap;
  const _ZBtn({required this.icon, required this.tip, required this.onTap});
  @override
  Widget build(BuildContext context) => Tooltip(
    message: tip,
    child: InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(8),
      child: Padding(padding: const EdgeInsets.all(7), child: Icon(icon, size: 17, color: AppColors.primary)),
    ),
  );
}

class _ZDivider extends StatelessWidget {
  @override
  Widget build(BuildContext context) => Divider(height: 1, thickness: 1, color: AppColors.border);
}

// ─── Tool button ─────────────────────────────────────────────────────────────

class _ToolBtn extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool active;
  final VoidCallback onTap;

  const _ToolBtn({required this.icon, required this.label, required this.active, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: active ? AppColors.primary : AppColors.surface,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: active ? AppColors.primary : AppColors.border),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 16, color: active ? Colors.white : AppColors.primary),
            const SizedBox(width: 5),
            Text(label, style: TextStyle(fontSize: 12, color: active ? Colors.white : AppColors.textPrimary, fontWeight: FontWeight.w600)),
          ],
        ),
      ),
    );
  }
}

// ─── State menu sheet ─────────────────────────────────────────────────────────

class _StateMenuSheet extends StatelessWidget {
  final StateNode node;
  final VoidCallback onSetInitial, onSetAccepting, onDelete, onDrawEdge;

  const _StateMenuSheet({
    required this.node,
    required this.onSetInitial,
    required this.onSetAccepting,
    required this.onDelete,
    required this.onDrawEdge,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(width: 36, height: 4, decoration: BoxDecoration(color: AppColors.border, borderRadius: BorderRadius.circular(2))),
          const SizedBox(height: 16),
          Text('Estado: ${node.id}', style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700)),
          const SizedBox(height: 16),
          _MenuTile(icon: Icons.arrow_right_alt, label: node.isInitial ? 'Quitar como inicial' : 'Marcar como inicial', color: AppColors.success, onTap: onSetInitial),
          _MenuTile(icon: Icons.radio_button_checked, label: node.isAccepting ? 'Quitar como aceptación' : 'Marcar como aceptación / final', color: AppColors.warning, onTap: onSetAccepting),
          _MenuTile(icon: Icons.arrow_forward, label: 'Dibujar transición desde aquí', color: AppColors.primary, onTap: onDrawEdge),
          _MenuTile(icon: Icons.delete_outline, label: 'Eliminar estado', color: AppColors.error, onTap: onDelete),
        ],
      ),
    );
  }
}

class _MenuTile extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback onTap;

  const _MenuTile({required this.icon, required this.label, required this.color, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return ListTile(
      leading: Icon(icon, color: color),
      title: Text(label, style: const TextStyle(fontSize: 14)),
      onTap: onTap,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      horizontalTitleGap: 8,
    );
  }
}