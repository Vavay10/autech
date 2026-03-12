import 'dart:async';
import 'package:flutter/material.dart';
import '../../../api_service.dart';
import '../../../theme.dart';
import '../../../widgets/automaton_canvas.dart';

// ─── Enums ────────────────────────────────────────────────────────────────────
enum _MenuAction { simulate, getRegex, clear, help }

// ─── Main Screen ─────────────────────────────────────────────────────────────
class RegexScreen extends StatefulWidget {
  const RegexScreen({super.key});
  @override
  State<RegexScreen> createState() => _RegexScreenState();
}

class _RegexScreenState extends State<RegexScreen> with SingleTickerProviderStateMixin {
  late final TabController _tab;

  @override
  void initState() {
    super.initState();
    _tab = TabController(length: 2, vsync: this);
  }

  @override
  void dispose() { _tab.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) => Scaffold(
    backgroundColor: AppColors.bg,
    appBar: AppBar(
      title: const Text('Expresiones Regulares'),
      bottom: TabBar(
        controller: _tab,
        labelColor: AppColors.primary,
        unselectedLabelColor: AppColors.textSecondary,
        indicatorColor: AppColors.primary,
        indicatorWeight: 3,
        tabs: const [
          Tab(icon: Icon(Icons.edit, size: 16), text: 'Construir'),
          Tab(icon: Icon(Icons.auto_awesome, size: 16), text: 'Regex → AFD'),
        ],
      ),
    ),
    body: TabBarView(
      controller: _tab,
      children: const [_BuildAutomatonTab(), _FromRegexTab()],
    ),
  );
}

// ─── Tab 1: Build Automaton Graphically ──────────────────────────────────────
class _BuildAutomatonTab extends StatefulWidget {
  const _BuildAutomatonTab();
  @override
  State<_BuildAutomatonTab> createState() => _BuildAutomatonTabState();
}

class _BuildAutomatonTabState extends State<_BuildAutomatonTab> {
  final _api = ApiService();
  List<StateNode> _states = [];
  List<TransitionEdge> _edges = [];

  // Simulation
  String _testInput = '';
  List<Map<String, dynamic>> _simSteps = [];
  int _simStep = -1;
  bool _simRunning = false;
  Timer? _autoTimer;

  // Results
  String _regex = '';
  bool _loading = false;

  // Panels
  bool _showSimPanel = false;
  bool _showRegexPanel = false;

  // Panel sizes (for resizable split)
  double _graphFlex = 3;
  double _panelFlex = 2;

  final _canvasKey = GlobalKey<AutomatonCanvasState>();

  @override
  void dispose() { _autoTimer?.cancel(); super.dispose(); }

  String? get _animatedState {
    if (_simStep < 0 || _simStep >= _simSteps.length) return null;
    return _simSteps[_simStep]['state'] as String?;
  }

  Set<String>? get _animatedEdges {
    if (_simStep <= 0 || _simStep >= _simSteps.length) return null;
    final prev = _simSteps[_simStep - 1]['state'] as String?;
    final cur = _simSteps[_simStep]['state'] as String?;
    if (prev != null && cur != null) return {'$prev→$cur'};
    return null;
  }

  void _runSimulation() {
    if (_states.isEmpty) { _snack('Primero construye el autómata'); return; }
    if (_testInput.isEmpty) { _snack('Ingresa una cadena a probar'); return; }
    final initial = _states.firstWhere((s) => s.isInitial, orElse: () => _states.first);
    final steps = <Map<String, dynamic>>[];
    var cur = initial.id;
    steps.add({'state': cur, 'consumed': '', 'symbol': ''});

    for (int i = 0; i < _testInput.length; i++) {
      final ch = _testInput[i];
      final edge = _edges.firstWhere(
        (e) => e.from == cur && e.label.split(',').map((l) => l.trim()).contains(ch),
        orElse: () => TransitionEdge(from: '', to: '', label: ''),
      );
      if (edge.from.isEmpty) {
        steps.add({'state': cur, 'consumed': _testInput.substring(0, i + 1), 'symbol': ch, 'rejected': true});
        break;
      }
      cur = edge.to;
      steps.add({'state': cur, 'consumed': _testInput.substring(0, i + 1), 'symbol': ch});
    }
    setState(() { _simSteps = steps; _simStep = 0; _showSimPanel = true; });
  }

  bool get _simAccepted {
    if (_simSteps.isEmpty) return false;
    final last = _simSteps.last;
    if (last['rejected'] == true) return false;
    return _states.firstWhere((s) => s.id == last['state'], orElse: () => StateNode(id: '')).isAccepting;
  }

  void _autoPlay() {
    _autoTimer?.cancel();
    _autoTimer = Timer.periodic(const Duration(milliseconds: 800), (t) {
      if (_simStep < _simSteps.length - 1) {
        setState(() => _simStep++);
      } else {
        t.cancel();
        setState(() => _simRunning = false);
      }
    });
    setState(() => _simRunning = true);
  }

  void _stopSim() { _autoTimer?.cancel(); setState(() => _simRunning = false); }

  Future<void> _getRegex() async {
    if (_states.isEmpty) { _snack('Primero construye el autómata'); return; }
    setState(() { _loading = true; _showRegexPanel = true; });
    try {
      final data = {
        'states': _states.map((s) => s.id).toList(),
        'transitions': <String, Map<String, String>>{},
        'initial': _states.firstWhere((s) => s.isInitial, orElse: () => _states.first).id,
        'accepting': _states.where((s) => s.isAccepting).map((s) => s.id).toList(),
        'alphabet': <String>[],
      };
      // Build transitions map
      final Map<String, Map<String, String>> trans = {};
      for (final e in _edges) {
        trans.putIfAbsent(e.from, () => {})[e.label] = e.to;
      }
      data['transitions'] = trans;
      final result = await _api.automatonToRegex(data);
      setState(() => _regex = result['regex'] as String? ?? '');
    } on ApiException catch (e) {
      setState(() => _regex = 'Error: ${e.message}');
    } finally {
      setState(() => _loading = false);
    }
  }

  void _clearAll() {
    _autoTimer?.cancel();
    setState(() {
      _states = []; _edges = [];
      _simSteps = []; _simStep = -1;
      _regex = ''; _showSimPanel = false; _showRegexPanel = false; _simRunning = false;
    });
  }

  void _snack(String msg) => ScaffoldMessenger.of(context)
      .showSnackBar(SnackBar(content: Text(msg), duration: const Duration(seconds: 2)));

  void _showSimulateDialog() {
    final ctrl = TextEditingController(text: _testInput);
    showDialog<String>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Row(children: [Icon(Icons.play_circle_outline, color: AppColors.primary), SizedBox(width: 8), Text('Simular cadena')]),
        content: TextField(controller: ctrl, autofocus: true,
            decoration: const InputDecoration(labelText: 'Cadena a probar (ej: abb)')),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancelar')),
          ElevatedButton(onPressed: () => Navigator.pop(context, ctrl.text), child: const Text('Simular')),
        ],
      ),
    ).then((v) {
      if (v != null) { setState(() => _testInput = v); _runSimulation(); }
    });
  }

  void _onMenuSelected(_MenuAction action) {
    switch (action) {
      case _MenuAction.simulate: _showSimulateDialog();
      case _MenuAction.getRegex: _getRegex();
      case _MenuAction.clear: _clearAll();
      case _MenuAction.help: _showHelp();
    }
  }

  void _showHelp() => showDialog(
    context: context,
    builder: (_) => AlertDialog(
      title: const Text('Cómo usar'),
      content: const SingleChildScrollView(
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          _HelpRow(icon: Icons.touch_app, text: 'Toca el canvas para crear un estado'),
          _HelpRow(icon: Icons.touch_app, text: 'Toca un estado y luego otro para crear una transición'),
          _HelpRow(icon: Icons.pan_tool, text: 'Mantén presionado un estado para opciones (renombrar, aceptación, etc.)'),
          _HelpRow(icon: Icons.zoom_in, text: 'Usa los controles de zoom (arriba a la derecha)'),
          _HelpRow(icon: Icons.drag_handle, text: 'Arrastra estados para reorganizar el autómata'),
        ]),
      ),
      actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('OK'))],
    ),
  );

  @override
  Widget build(BuildContext context) {
    final hasPanels = _showSimPanel || _showRegexPanel;
    return Column(
      children: [
        // Toolbar
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          color: AppColors.surface,
          child: Row(children: [
            const Icon(Icons.schema, color: AppColors.primary, size: 16),
            const SizedBox(width: 8),
            Text('${_states.length} estado(s) · ${_edges.length} transición(es)',
                style: const TextStyle(fontSize: 12, color: AppColors.textSecondary)),
            const Spacer(),
            PopupMenuButton<_MenuAction>(
              icon: const Icon(Icons.more_vert, size: 20),
              itemBuilder: (_) => [
                const PopupMenuItem(value: _MenuAction.simulate, child: Row(children: [Icon(Icons.play_arrow, size: 16), SizedBox(width: 8), Text('Simular')])),
                const PopupMenuItem(value: _MenuAction.getRegex, child: Row(children: [Icon(Icons.functions, size: 16), SizedBox(width: 8), Text('Obtener Regex')])),
                const PopupMenuDivider(),
                const PopupMenuItem(value: _MenuAction.clear, child: Row(children: [Icon(Icons.delete_outline, size: 16), SizedBox(width: 8), Text('Limpiar todo')])),
                const PopupMenuItem(value: _MenuAction.help, child: Row(children: [Icon(Icons.help_outline, size: 16), SizedBox(width: 8), Text('Ayuda')])),
              ],
              onSelected: _onMenuSelected,
            ),
          ]),
        ),

        // Main area: graph + optional side panels
        Expanded(
          child: hasPanels
              ? Row(
                  children: [
                    // Graph (flexible)
                    Expanded(
                      flex: 2,
                      child: _buildGraph(),
                    ),
                    const VerticalDivider(width: 1),
                    // Side panels
                    SizedBox(
                      width: 280,
                      child: Column(children: [
                        if (_showSimPanel) Flexible(child: _buildSimPanel()),
                        if (_showSimPanel && _showRegexPanel) const Divider(height: 1),
                        if (_showRegexPanel) Flexible(child: _buildRegexPanel()),
                      ]),
                    ),
                  ],
                )
              : _buildGraph(),
        ),
      ],
    );
  }

  Widget _buildGraph() => AutomatonCanvas(
    key: _canvasKey,
    states: _states,
    edges: _edges,
    editable: true,
    animatedState: _animatedState,
    animatedEdgeKeys: _animatedEdges,   // ← era animatedEdges:
    onStateAdded: (node, pos) => setState(() => _states.add(node)),
    onStateDeleted: (id) => setState(() {
      _states.removeWhere((s) => s.id == id);
      _edges.removeWhere((e) => e.from == id || e.to == id);
    }),
    onStateUpdated: (node) => setState(() {
      final i = _states.indexWhere((s) => s.id == node.id);
      if (i != -1) _states[i] = node;
    }),
    onEdgeAdded: (edge) => setState(() => _edges.add(edge)),
    onEdgeDeleted: (from, to) => setState(() =>
      _edges.removeWhere((e) => e.from == from && e.to == to)),
    onPositionsChanged: (_) {},         // ← era onChanged:
  );

  Widget _buildSimPanel() {
    return Container(
      color: AppColors.surface,
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 10, 8, 6),
            child: Row(children: [
              const Icon(Icons.play_circle_outline, color: AppColors.primary, size: 16),
              const SizedBox(width: 6),
              Expanded(
                child: Text(
                  'Simulando: "${_testInput.isEmpty ? "ε" : _testInput}"',
                  style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 12, color: AppColors.primary),
                ),
              ),
              IconButton(
                icon: const Icon(Icons.close, size: 16),
                onPressed: () => setState(() { _showSimPanel = false; _simSteps = []; _simStep = -1; }),
                visualDensity: VisualDensity.compact,
              ),
            ]),
          ),
          // Tape
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: _SimTape(input: _testInput, step: _simStep, steps: _simSteps),
          ),
          const SizedBox(height: 6),
          // Controls
          Row(mainAxisAlignment: MainAxisAlignment.center, children: [
            IconButton(icon: const Icon(Icons.chevron_left, size: 20), onPressed: _simStep > 0 ? () => setState(() => _simStep--) : null),
            GestureDetector(
              onTap: _simRunning ? _stopSim : _autoPlay,
              child: Container(
                padding: const EdgeInsets.all(6),
                decoration: BoxDecoration(color: AppColors.primary, shape: BoxShape.circle),
                child: Icon(_simRunning ? Icons.pause : Icons.play_arrow, color: Colors.white, size: 18),
              ),
            ),
            IconButton(icon: const Icon(Icons.chevron_right, size: 20), onPressed: _simStep < _simSteps.length - 1 ? () => setState(() => _simStep++) : null),
            Text('${_simStep + 1}/${_simSteps.length}', style: const TextStyle(fontSize: 11, color: AppColors.textSecondary)),
            const Spacer(),
            if (_simStep == _simSteps.length - 1)
              _StatusBadge(accepted: _simAccepted),
          ]),
          const SizedBox(height: 4),
        ],
      ),
    );
  }

  Widget _buildRegexPanel() {
    return Container(
      color: AppColors.surface,
      padding: const EdgeInsets.all(12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            const Icon(Icons.functions, color: AppColors.primary, size: 16),
            const SizedBox(width: 6),
            const Text('Expresión Regular', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 12)),
            const Spacer(),
            IconButton(
              icon: const Icon(Icons.close, size: 16),
              onPressed: () => setState(() => _showRegexPanel = false),
              visualDensity: VisualDensity.compact,
            ),
          ]),
          const SizedBox(height: 8),
          if (_loading) const Center(child: CircularProgressIndicator(strokeWidth: 2))
          else Expanded(
            child: Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: AppColors.surfaceAlt,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: AppColors.border),
              ),
              child: SelectableText(
                _regex.isEmpty ? 'Presiona "Obtener Regex" para generar' : _regex,
                style: TextStyle(
                  fontFamily: 'monospace',
                  fontSize: 13,
                  color: _regex.isEmpty ? AppColors.textHint : AppColors.textPrimary,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ─── Tab 2: Regex → DFA ───────────────────────────────────────────────────────
class _FromRegexTab extends StatefulWidget {
  const _FromRegexTab();
  @override
  State<_FromRegexTab> createState() => _FromRegexTabState();
}

class _FromRegexTabState extends State<_FromRegexTab> {
  final _api = ApiService();
  final _regexCtrl = TextEditingController();
  List<StateNode> _states = [];
  List<TransitionEdge> _edges = [];
  bool _loading = false;
  String _error = '';
  String _status = '';
  bool _showGraph = false;

  // Simulate on generated DFA
  String _testInput = '';
  List<Map<String, dynamic>> _simSteps = [];
  int _simStep = -1;
  bool _showSim = false;
  String? _animatedState;

  final _canvasKey = GlobalKey<AutomatonCanvasState>();

  Future<void> _generate() async {
    final regex = _regexCtrl.text.trim();
    if (regex.isEmpty) return;
    setState(() { _loading = true; _error = ''; _status = 'Generando...'; _showGraph = false; });
    try {
      final res = await _api.regexToAutomaton(regex);
      final statesList = (res['states'] as List?)?.map((s) => StateNode.fromJson(s as Map<String, dynamic>)).toList() ?? [];
      final edgesList = (res['edges'] as List?)?.map((e) => TransitionEdge.fromJson(e as Map<String, dynamic>)).toList() ?? [];
      setState(() {
        _states = statesList;
        _edges = edgesList;
        _status = '✅ AFD Mínimo generado (${_states.length} estados)';
        _showGraph = true;
      });
    } on ApiException catch (e) {
      setState(() { _error = e.message; _status = ''; });
    } finally {
      setState(() => _loading = false);
    }
  }

  void _simulate() {
    if (_states.isEmpty) return;
    final initial = _states.firstWhere((s) => s.isInitial, orElse: () => _states.first);
    final steps = <Map<String, dynamic>>[];
    var cur = initial.id;
    steps.add({'state': cur, 'consumed': '', 'symbol': ''});

    for (int i = 0; i < _testInput.length; i++) {
      final ch = _testInput[i];
      final edge = _edges.firstWhere(
        (e) => e.from == cur && e.label.split(',').map((l) => l.trim()).contains(ch),
        orElse: () => TransitionEdge(from: '', to: '', label: ''),
      );
      if (edge.from.isEmpty) {
        steps.add({'state': cur, 'consumed': _testInput.substring(0, i + 1), 'symbol': ch, 'rejected': true});
        break;
      }
      cur = edge.to;
      steps.add({'state': cur, 'consumed': _testInput.substring(0, i + 1), 'symbol': ch});
    }
    setState(() { _simSteps = steps; _simStep = 0; _showSim = true; _animatedState = initial.id; });
  }

  bool get _simAccepted {
    if (_simSteps.isEmpty) return false;
    final last = _simSteps.last;
    if (last['rejected'] == true) return false;
    return _states.firstWhere((s) => s.id == last['state'], orElse: () => StateNode(id: '')).isAccepting;
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // Input row
        Padding(
          padding: const EdgeInsets.fromLTRB(12, 12, 12, 8),
          child: Row(children: [
            Expanded(
              child: TextField(
                controller: _regexCtrl,
                onSubmitted: (_) => _generate(),
                decoration: const InputDecoration(
                  labelText: 'Expresión regular',
                  hintText: 'a(b|c)*  ó  (ab)+',
                  prefixIcon: Icon(Icons.functions, size: 18),
                  isDense: true,
                ),
              ),
            ),
            const SizedBox(width: 8),
            FilledButton.icon(
              onPressed: _loading ? null : _generate,
              icon: _loading
                  ? const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                  : const Icon(Icons.auto_awesome, size: 16),
              label: const Text('Generar AFD'),
            ),
          ]),
        ),

        if (_status.isNotEmpty)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: Row(children: [
              Text(_status,
                style: TextStyle(
                  fontSize: 12, fontWeight: FontWeight.w600,
                  color: _status.startsWith('✅') ? AppColors.success : AppColors.textSecondary,
                )),
              if (_showGraph) ...[
                const Spacer(),
                TextButton.icon(
                  icon: const Icon(Icons.play_arrow, size: 14),
                  label: const Text('Simular', style: TextStyle(fontSize: 12)),
                  onPressed: () => _showSimDialog(),
                ),
              ],
            ]),
          ),

        if (_error.isNotEmpty)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
            child: _ErrorBanner(_error),
          ),

        // Graph (expandable)
        if (_showGraph)
          Expanded(
            child: Column(
              children: [
                if (_showSim)
                  Container(
                    color: AppColors.surface,
                    padding: const EdgeInsets.fromLTRB(12, 8, 12, 8),
                    child: Column(children: [
                      _SimTape(input: _testInput, step: _simStep, steps: _simSteps),
                      const SizedBox(height: 6),
                      Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                        IconButton(icon: const Icon(Icons.chevron_left, size: 20), onPressed: _simStep > 0 ? () => setState(() { _simStep--; _animatedState = _simSteps[_simStep]['state'] as String?; }) : null),
                        IconButton(icon: const Icon(Icons.chevron_right, size: 20), onPressed: _simStep < _simSteps.length - 1 ? () => setState(() { _simStep++; _animatedState = _simSteps[_simStep]['state'] as String?; }) : null),
                        Text('${_simStep + 1}/${_simSteps.length}', style: const TextStyle(fontSize: 11)),
                        const SizedBox(width: 12),
                        if (_simStep == _simSteps.length - 1) _StatusBadge(accepted: _simAccepted),
                        const Spacer(),
                        TextButton(onPressed: () => setState(() => _showSim = false), child: const Text('Cerrar')),
                      ]),
                    ]),
                  ),
                Expanded(
                  child: Container(
                    margin: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: AppColors.border),
                    ),
                    child: AutomatonCanvas(
                      key: _canvasKey,
                      states: _states,
                      edges: _edges,
                      editable: false,
                      animatedState: _animatedState,
                    ),
                  ),
                ),
              ],
            ),
          )
        else
          Expanded(
            child: Center(
              child: Column(mainAxisSize: MainAxisSize.min, children: [
                Icon(Icons.auto_awesome, size: 48, color: Colors.grey.shade300),
                const SizedBox(height: 12),
                Text('Ingresa una expresión regular y presiona "Generar AFD"',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.grey.shade500)),
                const SizedBox(height: 20),
                _ExamplesChips(onSelect: (r) {
                  _regexCtrl.text = r;
                  _generate();
                }),
              ]),
            ),
          ),
      ],
    );
  }

  void _showSimDialog() {
    final ctrl = TextEditingController(text: _testInput);
    showDialog<String>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Simular cadena'),
        content: TextField(controller: ctrl, autofocus: true,
            decoration: const InputDecoration(labelText: 'Cadena a probar')),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancelar')),
          ElevatedButton(onPressed: () => Navigator.pop(context, ctrl.text), child: const Text('Simular')),
        ],
      ),
    ).then((v) {
      if (v != null) { setState(() => _testInput = v); _simulate(); }
    });
  }
}

// ─── Example Chips ────────────────────────────────────────────────────────────
class _ExamplesChips extends StatelessWidget {
  final void Function(String) onSelect;
  const _ExamplesChips({required this.onSelect});

  static const _examples = ['a*b', '(a|b)*', 'a(b|c)*', '(ab)+', 'a?b*c'];

  @override
  Widget build(BuildContext context) => Wrap(
    spacing: 8,
    children: _examples.map((r) => ActionChip(
      label: Text(r, style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
      onPressed: () => onSelect(r),
    )).toList(),
  );
}

// ─── Shared Widgets ───────────────────────────────────────────────────────────

class _SimTape extends StatelessWidget {
  final String input;
  final int step;
  final List<Map<String, dynamic>> steps;
  const _SimTape({required this.input, required this.step, required this.steps});

  @override
  Widget build(BuildContext context) {
    final consumed = step >= 0 && step < steps.length
        ? (steps[step]['consumed'] as String? ?? '') : '';
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: Row(
        children: [
          for (int i = 0; i < input.length; i++)
            _TapeCell(char: input[i], isConsumed: i < consumed.length, isCurrent: i == consumed.length - 1 && consumed.isNotEmpty),
          if (input.isEmpty)
            Container(
              width: 40, height: 36,
              alignment: Alignment.center,
              decoration: BoxDecoration(border: Border.all(color: AppColors.border), borderRadius: BorderRadius.circular(6)),
              child: const Text('ε', style: TextStyle(fontStyle: FontStyle.italic, fontSize: 18)),
            ),
        ],
      ),
    );
  }
}

class _TapeCell extends StatelessWidget {
  final String char;
  final bool isConsumed, isCurrent;
  const _TapeCell({required this.char, required this.isConsumed, required this.isCurrent});

  @override
  Widget build(BuildContext context) => AnimatedContainer(
    duration: const Duration(milliseconds: 250),
    width: 36, height: 36,
    margin: const EdgeInsets.only(right: 2),
    alignment: Alignment.center,
    decoration: BoxDecoration(
      color: isCurrent ? AppColors.primary.withOpacity(0.15) : isConsumed ? AppColors.surfaceAlt : AppColors.surface,
      border: Border.all(color: isCurrent ? AppColors.primary : AppColors.border, width: isCurrent ? 2 : 1),
      borderRadius: BorderRadius.circular(6),
    ),
    child: Text(char, style: TextStyle(fontWeight: isCurrent ? FontWeight.w700 : FontWeight.w500,
        color: isConsumed ? AppColors.textHint : AppColors.textPrimary, fontSize: 15)),
  );
}

class _StatusBadge extends StatelessWidget {
  final bool accepted;
  const _StatusBadge({required this.accepted});

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
    decoration: BoxDecoration(
      color: (accepted ? AppColors.success : AppColors.error).withOpacity(0.1),
      borderRadius: BorderRadius.circular(16),
      border: Border.all(color: accepted ? AppColors.success : AppColors.error),
    ),
    child: Row(mainAxisSize: MainAxisSize.min, children: [
      Icon(accepted ? Icons.check_circle : Icons.cancel, color: accepted ? AppColors.success : AppColors.error, size: 13),
      const SizedBox(width: 4),
      Text(accepted ? 'ACEPTADA' : 'RECHAZADA',
          style: TextStyle(color: accepted ? AppColors.success : AppColors.error, fontWeight: FontWeight.w700, fontSize: 11)),
    ]),
  );
}

class _ErrorBanner extends StatelessWidget {
  final String message;
  const _ErrorBanner(this.message);

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.all(10),
    decoration: BoxDecoration(color: AppColors.error.withOpacity(0.08), borderRadius: BorderRadius.circular(8),
        border: Border.all(color: AppColors.error.withOpacity(0.4))),
    child: Row(children: [
      const Icon(Icons.error_outline, color: AppColors.error, size: 16),
      const SizedBox(width: 8),
      Expanded(child: Text(message, style: const TextStyle(color: AppColors.error, fontSize: 12))),
    ]),
  );
}

class _HelpRow extends StatelessWidget {
  final IconData icon;
  final String text;
  const _HelpRow({required this.icon, required this.text});

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.only(bottom: 10),
    child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Icon(icon, size: 18, color: AppColors.primary),
      const SizedBox(width: 10),
      Expanded(child: Text(text, style: const TextStyle(fontSize: 13, height: 1.4))),
    ]),
  );
}