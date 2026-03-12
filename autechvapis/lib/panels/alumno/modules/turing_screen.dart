import 'dart:async';
import 'package:flutter/material.dart';
import '../../../api_service.dart';
import '../../../theme.dart';
import '../../../widgets/automaton_canvas.dart';

// ─── Turing Screen ───────────────────────────────────────────────────────────
class TuringScreen extends StatefulWidget {
  const TuringScreen({super.key});
  @override
  State<TuringScreen> createState() => _TuringScreenState();
}

class _TuringScreenState extends State<TuringScreen> with SingleTickerProviderStateMixin {
  final _api = ApiService();
  late final TabController _tab;

  // Form
  final _statesCtrl = TextEditingController(text: 'q0,q1,q2,q3,qf');
  final _initCtrl   = TextEditingController(text: 'q0');
  final _acceptCtrl = TextEditingController(text: 'qf');
  final _transCtrl  = TextEditingController(
    text: 'q0,a->q1,X,R\nq0,Y->q3,Y,R\nq0,_->qf,_,S\n'
        'q1,a->q1,a,R\nq1,Y->q1,Y,R\nq1,b->q2,Y,L\n'
        'q2,a->q2,a,L\nq2,Y->q2,Y,L\nq2,X->q0,X,R\n'
        'q3,Y->q3,Y,R\nq3,_->qf,_,S',
  );
  final _tapeCtrl = TextEditingController(text: 'aabb');

  // Simulation
  List<Map<String, dynamic>> _steps = [];
  int _step = -1;
  bool _running = false;
  bool _loading = false;
  String _error = '';
  Timer? _autoTimer;

  // Graph
  List<StateNode> _gStates = [];
  List<TransitionEdge> _gEdges = [];
  String? _highlightState;

  // Layout
  bool _formExpanded = true;
  final _canvasKey = GlobalKey<AutomatonCanvasState>();

  @override
  void initState() {
    super.initState();
    _tab = TabController(length: 3, vsync: this);
    _buildGraph();
  }

  @override
  void dispose() {
    _tab.dispose();
    _autoTimer?.cancel();
    _api.dispose();
    super.dispose();
  }

  // ── Graph ──────────────────────────────────────────────────────────────────

  Future<void> _buildGraph() async {
    try {
      final res = await _api.turingGraph(_definition);
      _parseGraph(res);
    } catch (_) {
      _buildGraphLocally();
    }
  }

  void _parseGraph(Map<String, dynamic> data) {
    final states = (data['states'] as List?)?.map((s) => StateNode.fromJson(s as Map<String, dynamic>)).toList() ?? [];
    final edges = (data['edges'] as List?)?.map((e) => TransitionEdge.fromJson(e as Map<String, dynamic>)).toList() ?? [];
    setState(() { _gStates = states; _gEdges = edges; });
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _canvasKey.currentState?.fitToScreen();
    });
  }

  void _buildGraphLocally() {
    final states = _statesCtrl.text.split(',').map((s) => s.trim()).where((s) => s.isNotEmpty).toList();
    final init = _initCtrl.text.trim();
    final accepts = _acceptCtrl.text.split(',').map((s) => s.trim()).toSet();
    final stateNodes = states.map((s) => StateNode(id: s, isInitial: s == init, isAccepting: accepts.contains(s))).toList();

    // Parse transitions for edges
    final grouped = <String, List<String>>{};
    for (final raw in _transCtrl.text.split('\n')) {
      final line = raw.trim();
      if (line.isEmpty || !line.contains('->')) continue;
      final parts = line.split('->');
      final left = parts[0].trim().split(',');
      final right = parts[1].trim().split(',');
      if (left.length < 2 || right.length < 3) continue;
      final from = left[0].trim(), read = left[1].trim();
      final to = right[0].trim(), write = right[1].trim(), dir = right[2].trim();
      final key = '$from→$to';
      grouped.putIfAbsent(key, () => []).add('$read→$write,$dir');
    }

    final edges = grouped.entries.map((e) {
      final parts = e.key.split('→');
      return TransitionEdge(from: parts[0], to: parts[1], label: e.value.join('\n'));
    }).toList();

    setState(() { _gStates = stateNodes; _gEdges = edges; });
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _canvasKey.currentState?.fitToScreen();
    });
  }

  // ── Simulation ─────────────────────────────────────────────────────────────

  Map<String, dynamic> get _definition => {
    'states': _statesCtrl.text,
    'initial': _initCtrl.text,
    'acceptStates': _acceptCtrl.text,
    'transitions': _transCtrl.text,
  };

  Future<void> _startSim() async {
    setState(() { _loading = true; _error = ''; });
    try {
      final res = await _api.turingSimulate(
        _definition,
        _tapeCtrl.text,
        headPos: 0,
        maxSteps: 500,
      );
      final steps = (res['steps'] as List?)?.cast<Map<String, dynamic>>() ?? [];
      setState(() {
        _steps = steps;
        _step = 0;
        _running = false;
        _highlightState = steps.isNotEmpty ? steps[0]['state'] as String? : null;
      });
    } on ApiException catch (_) {
      _startLocalSim();
    } finally {
      setState(() => _loading = false);
    }
  }

  void _startLocalSim() {
    // Use local Turing simulator (same as original turing_screen.dart)
    final rules = _LocalTuringSimulator.parseRules(_transCtrl.text);
    final simulator = _LocalTuringSimulator(
      rules: rules,
      initialState: _initCtrl.text.trim(),
      acceptStates: _acceptCtrl.text.split(',').map((s) => s.trim()).toSet(),
      allStates: _statesCtrl.text.split(',').map((s) => s.trim()).toSet(),
    );
    final steps = simulator.simulate(_tapeCtrl.text);
    setState(() {
      _steps = steps.map((s) => {
        'step': s.step,
        'state': s.state,
        'tape': s.tape,
        'headPos': s.headPos,
        'message': s.message,
        'isAccepted': s.isAccepted,
        'isRejected': s.isRejected,
      }).toList();
      _step = 0;
      _highlightState = _steps.isNotEmpty ? _steps[0]['state'] as String? : null;
    });
  }

  void _goToStep(int idx) {
    if (idx < 0 || idx >= _steps.length) return;
    setState(() {
      _step = idx;
      _highlightState = _steps[idx]['state'] as String?;
    });
  }

  void _autoPlay() {
    _autoTimer?.cancel();
    _autoTimer = Timer.periodic(const Duration(milliseconds: 600), (t) {
      if (_step < _steps.length - 1) {
        _goToStep(_step + 1);
      } else {
        t.cancel();
        setState(() => _running = false);
      }
    });
    setState(() => _running = true);
  }

  void _stopAuto() { _autoTimer?.cancel(); setState(() => _running = false); }

  void _loadExample(String states, String init, String accepts, String trans, String tape) {
    _statesCtrl.text = states;
    _initCtrl.text = init;
    _acceptCtrl.text = accepts;
    _transCtrl.text = trans;
    _tapeCtrl.text = tape;
    _steps = [];
    _step = -1;
    _buildGraph();
  }

  // ── Build ──────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(
        title: const Text('Máquina de Turing'),
        actions: [
          PopupMenuButton<String>(
            icon: const Icon(Icons.list_alt),
            tooltip: 'Cargar ejemplo',
            onSelected: _onExampleSelected,
            itemBuilder: (_) => const [
              PopupMenuItem(value: 'anbn', child: Text('aⁿbⁿ (aabb)')),
              PopupMenuItem(value: 'suma', child: Text('Suma Unaria (11_111)')),
              PopupMenuItem(value: 'busy', child: Text('Busy Beaver 3-state')),
            ],
          ),
        ],
        bottom: TabBar(
          controller: _tab,
          labelColor: AppColors.primary,
          unselectedLabelColor: AppColors.textSecondary,
          indicatorColor: AppColors.primary,
          tabs: const [
            Tab(icon: Icon(Icons.edit, size: 16), text: 'Definición'),
            Tab(icon: Icon(Icons.play_arrow, size: 16), text: 'Simulación'),
            Tab(icon: Icon(Icons.schema, size: 16), text: 'Grafo'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tab,
        children: [
          _buildDefineTab(),
          _buildSimTab(),
          _buildGraphTab(),
        ],
      ),
    );
  }

  // ── Tab 1: Define ─────────────────────────────────────────────────────────

  Widget _buildDefineTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Instructions card
          _InfoCard(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: const [
              Text('Formato de transiciones:', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 12)),
              SizedBox(height: 4),
              Text('estado,leer -> nuevo_estado,escribir,dirección (L/R/S)\nUsa "_" para el símbolo blanco.',
                  style: TextStyle(fontSize: 11, fontFamily: 'monospace', height: 1.5, color: AppColors.textSecondary)),
            ]),
          ),
          const SizedBox(height: 10),
          Row(children: [
            Expanded(child: _field(_statesCtrl, 'Estados (Q)', 'q0,q1,qf')),
            const SizedBox(width: 8),
            Expanded(child: _field(_initCtrl, 'Estado inicial', 'q0')),
            const SizedBox(width: 8),
            Expanded(child: _field(_acceptCtrl, 'Estados aceptación', 'qf')),
          ]),
          const SizedBox(height: 10),
          TextField(
            controller: _transCtrl,
            minLines: 6,
            maxLines: 12,
            style: const TextStyle(fontSize: 11, fontFamily: 'monospace'),
            decoration: const InputDecoration(
              labelText: 'Transiciones',
              hintText: 'q0,a -> q1,X,R\nq1,b -> q2,Y,L\n...',
              isDense: true,
            ),
          ),
          const SizedBox(height: 10),
          _field(_tapeCtrl, 'Cinta inicial', 'aabb'),
          if (_error.isNotEmpty)
            Padding(padding: const EdgeInsets.only(top: 8), child: _ErrorCard(_error)),
          const SizedBox(height: 12),
          FilledButton.icon(
            onPressed: _loading ? null : () async { await _buildGraph(); await _startSim(); _tab.animateTo(1); },
            icon: _loading
                ? const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                : const Icon(Icons.play_circle, size: 16),
            label: const Text('Iniciar Simulación'),
          ),
        ],
      ),
    );
  }

  // ── Tab 2: Simulation ─────────────────────────────────────────────────────

  Widget _buildSimTab() {
    if (_steps.isEmpty) {
      return Center(child: Column(mainAxisSize: MainAxisSize.min, children: [
        Icon(Icons.play_circle_outline, size: 48, color: Colors.grey.shade300),
        const SizedBox(height: 12),
        const Text('Configura la MT y presiona "Iniciar Simulación"', style: TextStyle(color: AppColors.textHint)),
        const SizedBox(height: 16),
        OutlinedButton(onPressed: () => _tab.animateTo(0), child: const Text('Ir a Definición')),
      ]));
    }

    return LayoutBuilder(builder: (context, constraints) {
      final isLandscape = constraints.maxWidth > constraints.maxHeight && constraints.maxWidth > 500;
      final content = _buildSimContent();
      if (isLandscape) {
        return Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Expanded(flex: 3, child: content),
        ]);
      }
      return content;
    });
  }

  Widget _buildSimContent() {
    final cur = _steps[_step];
    final tape = (cur['tape'] as List?)?.cast<String>() ?? [];
    final headPos = cur['headPos'] as int? ?? 0;
    final state = cur['state'] as String? ?? '';
    final msg = cur['message'] as String? ?? '';
    final isAcc = cur['isAccepted'] == true;
    final isRej = cur['isRejected'] == true;

    return Column(
      children: [
        // State & status bar
        Container(
          color: isAcc ? AppColors.success.withOpacity(0.1) : isRej ? AppColors.error.withOpacity(0.1) : AppColors.surface,
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
          child: Row(children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: isAcc ? AppColors.success : isRej ? AppColors.error : AppColors.primary,
                borderRadius: BorderRadius.circular(16),
              ),
              child: Text(state, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w800, fontSize: 13)),
            ),
            const SizedBox(width: 10),
            Expanded(child: Text(msg, style: TextStyle(
              fontSize: 11, fontFamily: 'monospace',
              color: isAcc ? AppColors.success : isRej ? AppColors.error : AppColors.textPrimary,
            ))),
          ]),
        ),

        // Tape visualization
        Container(
          color: AppColors.surfaceAlt,
          padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 8),
          child: SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: [
                for (int i = 0; i < tape.length; i++)
                  _TapeCell(
                    char: tape[i],
                    isHead: i == headPos,
                    isPast: i < headPos,
                  ),
              ],
            ),
          ),
        ),

        // Navigation controls
        Container(
          color: AppColors.surface,
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
          child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
            IconButton(icon: const Icon(Icons.first_page, size: 20), onPressed: _step > 0 ? () => _goToStep(0) : null),
            IconButton(icon: const Icon(Icons.chevron_left, size: 20), onPressed: _step > 0 ? () => _goToStep(_step - 1) : null),
            const SizedBox(width: 6),
            GestureDetector(
              onTap: _running ? _stopAuto : _autoPlay,
              child: Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(color: AppColors.primary, shape: BoxShape.circle),
                child: Icon(_running ? Icons.pause : Icons.play_arrow, color: Colors.white, size: 22),
              ),
            ),
            const SizedBox(width: 6),
            IconButton(icon: const Icon(Icons.chevron_right, size: 20), onPressed: _step < _steps.length - 1 ? () => _goToStep(_step + 1) : null),
            IconButton(icon: const Icon(Icons.last_page, size: 20), onPressed: _step < _steps.length - 1 ? () => _goToStep(_steps.length - 1) : null),
            const SizedBox(width: 8),
            Text('${_step + 1}/${_steps.length}', style: const TextStyle(fontSize: 12, color: AppColors.textSecondary, fontWeight: FontWeight.w600)),
          ]),
        ),

        // Trace list
        Expanded(
          child: ListView.builder(
            itemCount: _steps.length,
            padding: const EdgeInsets.symmetric(vertical: 4),
            itemBuilder: (_, i) {
              final s = _steps[i];
              final isCur = i == _step;
              final acc = s['isAccepted'] == true;
              final rej = s['isRejected'] == true;
              return AnimatedContainer(
                duration: const Duration(milliseconds: 150),
                margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: isCur ? AppColors.primary.withOpacity(0.08) : Colors.transparent,
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(color: isCur ? AppColors.primary.withOpacity(0.4) : Colors.transparent),
                ),
                child: Row(children: [
                  Text('${i + 1}', style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700, color: isCur ? AppColors.primary : AppColors.textHint)),
                  const SizedBox(width: 6),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
                    decoration: BoxDecoration(
                      color: isCur ? AppColors.primary : AppColors.surfaceAlt,
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(s['state'] as String? ?? '', style: TextStyle(
                      fontSize: 9, fontWeight: FontWeight.w800,
                      color: isCur ? Colors.white : AppColors.textSecondary,
                    )),
                  ),
                  const SizedBox(width: 6),
                  Expanded(
                    child: Text(
                      s['message'] as String? ?? '',
                      style: TextStyle(
                        fontFamily: 'monospace', fontSize: 10,
                        color: acc ? AppColors.success : rej ? AppColors.error : AppColors.textSecondary,
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                ]),
              );
            },
          ),
        ),
      ],
    );
  }

  // ── Tab 3: Graph ──────────────────────────────────────────────────────────

  Widget _buildGraphTab() {
    return LayoutBuilder(builder: (context, constraints) {
      final legend = Padding(
        padding: const EdgeInsets.fromLTRB(12, 10, 12, 6),
        child: SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: Row(children: [
            _LegendDot(color: Colors.blue.shade100, label: 'Inicial'),
            const SizedBox(width: 12),
            _LegendDot(color: Colors.orange.shade100, label: 'Aceptación'),
            const SizedBox(width: 12),
            _LegendDot(color: Colors.yellow.shade100, label: 'Actual'),
            const SizedBox(width: 12),
            _LegendDot(color: Colors.blueGrey.shade50, label: 'Normal'),
            const SizedBox(width: 12),
            const Text('Arrastra estados para moverlos',
              style: TextStyle(fontSize: 11, color: AppColors.textHint, fontStyle: FontStyle.italic)),
          ]),
        ),
      );
      final canvas = Container(
        margin: const EdgeInsets.fromLTRB(8, 0, 8, 8),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppColors.border),
        ),
        child: _gStates.isEmpty
            ? Center(child: Column(mainAxisSize: MainAxisSize.min, children: [
                Icon(Icons.schema_outlined, size: 40, color: Colors.grey.shade300),
                const SizedBox(height: 8),
                const Text('El grafo aparece al iniciar la simulación',
                    style: TextStyle(color: AppColors.textHint)),
              ]))
            : AutomatonCanvas(
                key: _canvasKey,
                states: _gStates,
                edges: _gEdges,
                editable: false,
                draggable: true,
                animatedState: _highlightState,
                onPositionsChanged: (_) {},
              ),
      );
      return Column(children: [legend, Expanded(child: canvas)]);
    });
  }

  void _onExampleSelected(String key) {
    switch (key) {
      case 'anbn':
        _loadExample(
          'q0,q1,q2,q3,qf', 'q0', 'qf',
          'q0,a->q1,X,R\nq0,Y->q3,Y,R\nq0,_->qf,_,S\n'
          'q1,a->q1,a,R\nq1,Y->q1,Y,R\nq1,b->q2,Y,L\n'
          'q2,a->q2,a,L\nq2,Y->q2,Y,L\nq2,X->q0,X,R\n'
          'q3,Y->q3,Y,R\nq3,_->qf,_,S',
          'aabb',
        );
      case 'suma':
        _loadExample(
          'q0,q1,qf', 'q0', 'qf',
          'q0,1->q0,1,R\nq0,_->q1,1,R\nq1,1->q1,1,R\nq1,_->qf,_,L',
          '11_111',
        );
      case 'busy':
        _loadExample(
          'A,B,C,H', 'A', 'H',
          'A,_->B,1,R\nA,1->C,1,L\nB,_->C,1,R\nB,1->B,1,R\nC,_->H,1,S\nC,1->A,1,L',
          '_',
        );
    }
  }

  Widget _field(TextEditingController ctrl, String label, String hint) => TextField(
    controller: ctrl, style: const TextStyle(fontSize: 12),
    decoration: InputDecoration(labelText: label, hintText: hint, isDense: true),
  );
}

// ─── Tape Cell Widget ─────────────────────────────────────────────────────────

class _TapeCell extends StatelessWidget {
  final String char;
  final bool isHead;
  final bool isPast;
  const _TapeCell({required this.char, required this.isHead, required this.isPast});

  @override
  Widget build(BuildContext context) => Column(
    children: [
      if (isHead)
        const Icon(Icons.arrow_drop_down, color: AppColors.primary, size: 22)
      else
        const SizedBox(height: 22),
      AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        width: 40, height: 42,
        margin: const EdgeInsets.symmetric(horizontal: 2),
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: isHead ? AppColors.primary.withOpacity(0.15) : isPast ? AppColors.surfaceAlt : Colors.white,
          border: Border.all(color: isHead ? AppColors.primary : AppColors.border, width: isHead ? 2.5 : 1),
          borderRadius: BorderRadius.circular(6),
        ),
        child: Text(
          char == '_' ? '□' : char,
          style: TextStyle(
            fontFamily: 'monospace',
            fontSize: 16,
            fontWeight: isHead ? FontWeight.w800 : FontWeight.w500,
            color: isHead ? AppColors.primary : AppColors.textPrimary,
          ),
        ),
      ),
    ],
  );
}

// ─── Shared Widgets ───────────────────────────────────────────────────────────

class _LegendDot extends StatelessWidget {
  final Color color;
  final String label;
  const _LegendDot({required this.color, required this.label});

  @override
  Widget build(BuildContext context) => Row(children: [
    Container(width: 14, height: 14, decoration: BoxDecoration(color: color, shape: BoxShape.circle,
        border: Border.all(color: Colors.blueGrey.shade300))),
    const SizedBox(width: 4),
    Text(label, style: const TextStyle(fontSize: 11, color: AppColors.textSecondary)),
  ]);
}

class _InfoCard extends StatelessWidget {
  final Widget child;
  const _InfoCard({required this.child});

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.all(12),
    decoration: BoxDecoration(
      color: AppColors.primary.withOpacity(0.05),
      borderRadius: BorderRadius.circular(8),
      border: Border.all(color: AppColors.primary.withOpacity(0.2)),
    ),
    child: child,
  );
}

class _ErrorCard extends StatelessWidget {
  final String message;
  const _ErrorCard(this.message);

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.all(12),
    decoration: BoxDecoration(
      color: AppColors.error.withOpacity(0.08),
      borderRadius: BorderRadius.circular(8),
      border: Border.all(color: AppColors.error.withOpacity(0.4)),
    ),
    child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
      const Icon(Icons.error_outline, color: AppColors.error, size: 16),
      const SizedBox(width: 8),
      Expanded(child: Text(message, style: const TextStyle(color: AppColors.error, fontSize: 12, height: 1.4))),
    ]),
  );
}

// ─── Local Turing Simulator (fallback) ───────────────────────────────────────

class _TuringStep {
  final int step;
  final List<String> tape;
  final int headPos;
  final String state;
  final String message;
  final bool isAccepted, isRejected;
  const _TuringStep({required this.step, required this.tape, required this.headPos, required this.state, required this.message, this.isAccepted = false, this.isRejected = false});
}

class _LocalTuringSimulator {
  final List<Map<String, dynamic>> rules;
  final String initialState;
  final Set<String> acceptStates;
  final Set<String> allStates;
  static const String blank = '_';
  static const int maxSteps = 300;

  const _LocalTuringSimulator({required this.rules, required this.initialState, required this.acceptStates, required this.allStates});

  static List<Map<String, dynamic>> parseRules(String text) {
    final rules = <Map<String, dynamic>>[];
    for (final raw in text.split('\n')) {
      final line = raw.trim();
      if (line.isEmpty || line.startsWith('#')) continue;
      final arrowIdx = line.indexOf('->');
      if (arrowIdx < 0) continue;
      final left = line.substring(0, arrowIdx).trim().split(',');
      final right = line.substring(arrowIdx + 2).trim().split(',');
      if (left.length < 2 || right.length < 3) continue;
      rules.add({'from': left[0].trim(), 'read': left[1].trim(), 'to': right[0].trim(), 'write': right[1].trim(), 'dir': right[2].trim().toUpperCase()});
    }
    return rules;
  }

  List<_TuringStep> simulate(String input) {
    final tape = input.isEmpty ? [blank] : input.split('').toList();
    int headPos = 0;
    String state = initialState;
    final steps = <_TuringStep>[];
    steps.add(_TuringStep(step: 0, tape: List.from(tape), headPos: headPos, state: state, message: 'Inicio: estado=$state'));

    for (int i = 0; i < maxSteps; i++) {
      if (acceptStates.contains(state)) {
        steps.add(_TuringStep(step: i + 1, tape: List.from(tape), headPos: headPos, state: state, message: '✅ Cadena ACEPTADA en estado $state', isAccepted: true));
        return steps;
      }

      while (headPos < 0) { tape.insert(0, blank); headPos = 0; }
      while (headPos >= tape.length) tape.add(blank);

      final read = tape[headPos];
      Map<String, dynamic>? rule;
      for (final r in rules) {
        if (r['from'] == state && r['read'] == read) { rule = r; break; }
      }

      if (rule == null) {
        steps.add(_TuringStep(step: i + 1, tape: List.from(tape), headPos: headPos, state: state, message: '❌ Sin transición para ($state, $read) — RECHAZADA', isRejected: true));
        return steps;
      }

      final prevState = state;
      tape[headPos] = rule['write'] as String;
      state = rule['to'] as String;
      final prevHead = headPos;
      if (rule['dir'] == 'R') headPos++;
      else if (rule['dir'] == 'L') headPos--;
      while (headPos >= tape.length) tape.add(blank);
      if (headPos < 0) { tape.insert(0, blank); headPos = 0; }

      steps.add(_TuringStep(step: i + 1, tape: List.from(tape), headPos: headPos, state: state,
          message: '($prevState,${rule['read']})→($state,${rule['write']},${rule['dir']})  cabeza: $prevHead→$headPos'));

      if (acceptStates.contains(state)) {
        steps.add(_TuringStep(step: i + 2, tape: List.from(tape), headPos: headPos, state: state, message: '✅ Cadena ACEPTADA en estado $state', isAccepted: true));
        return steps;
      }
    }

    steps.add(_TuringStep(step: maxSteps + 1, tape: List.from(tape), headPos: headPos, state: state, message: '⚠️ Límite de $maxSteps pasos alcanzado', isRejected: true));
    return steps;
  }
}