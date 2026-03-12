import 'dart:async';
import 'package:flutter/material.dart';
import '../../../api_service.dart';
import '../../../theme.dart';
import '../../../widgets/automaton_canvas.dart';

// ─── PDA Screen ──────────────────────────────────────────────────────────────

class PdaScreen extends StatefulWidget {
  const PdaScreen({super.key});

  @override
  State<PdaScreen> createState() => _PdaScreenState();
}

class _PdaScreenState extends State<PdaScreen> with SingleTickerProviderStateMixin {
  late final TabController _tab;
  final _api = ApiService();

  // Form
  final _statesCtrl   = TextEditingController(text: 'q0,q1,q2');
  final _inputCtrl    = TextEditingController(text: 'a,b');
  final _stackCtrl    = TextEditingController(text: 'A,Z');
  final _initCtrl     = TextEditingController(text: 'q0');
  final _initSymCtrl  = TextEditingController(text: 'Z');
  final _acceptCtrl   = TextEditingController(text: 'q2');
  final _transCtrl    = TextEditingController(text:
      'q0,a,Z -> q0,AZ\nq0,a,A -> q0,AA\nq0,b,A -> q1,ε\nq1,b,A -> q1,ε\nq1,ε,Z -> q2,Z');
  final _simInputCtrl = TextEditingController();

  // State
  bool _loading = false;
  String _error = '';
  String _validMsg = '';
  String _cfgText = '';
  List<Map<String, dynamic>> _simSteps = [];
  bool _accepted = false;
  bool _rejected = false;
  int _simStep = -1;
  bool _simRunning = false;
  Timer? _autoTimer;

  // Graph
  List<StateNode> _states = [];
  List<TransitionEdge> _edges = [];

  // Layout
  bool _formExpanded = true;
  bool _showSim = false;
  bool _showCfg = false;
  final _canvasKey = GlobalKey<AutomatonCanvasState>();

  @override
  void initState() {
    super.initState();
    _tab = TabController(length: 3, vsync: this);
  }

  @override
  void dispose() {
    _tab.dispose();
    _autoTimer?.cancel();
    _api.dispose();
    super.dispose();
  }

  // ── Helpers ────────────────────────────────────────────────────────────────

  Map<String, dynamic> get _definition => {
    'states': _statesCtrl.text,
    'inputAlphabet': _inputCtrl.text,
    'stackAlphabet': _stackCtrl.text,
    'startState': _initCtrl.text,
    'startSymbol': _initSymCtrl.text,
    'acceptStates': _acceptCtrl.text,
    'transitions': _transCtrl.text,
  };

  void _parseGraphJson(Map<String, dynamic> graph) {
    final statesList = (graph['states'] as List?)
        ?.map((s) => StateNode.fromJson(s as Map<String, dynamic>))
        .toList() ?? [];
    final edgesList = (graph['edges'] as List?)
        ?.map((e) => TransitionEdge.fromJson(e as Map<String, dynamic>))
        .toList() ?? [];
    setState(() { _states = statesList; _edges = edgesList; });
  }

  Future<void> _validate() async {
    setState(() { _loading = true; _error = ''; _validMsg = ''; });
    try {
      final res = await _api.validatePda(_definition);
      _parseGraphJson(res['graph'] as Map<String, dynamic>);
      setState(() { _validMsg = '✅ PDA válido'; _formExpanded = false; });
    } on ApiException catch (e) {
      setState(() => _error = e.message);
    } finally {
      setState(() => _loading = false);
    }
  }

  Future<void> _simulate() async {
    final input = _simInputCtrl.text;
    setState(() { _loading = true; _error = ''; _showSim = true; });
    try {
      final res = await _api.simulatePda(_definition, input);
      final steps = (res['trace'] as List?)?.cast<String>() ?? [];
      setState(() {
        _simSteps = steps.asMap().entries.map((e) => {'step': e.key, 'text': e.value}).toList();
        _accepted = res['accepted'] == true;
        _rejected = res['accepted'] != true;
        _simStep = 0;
      });
    } on ApiException catch (e) {
      // Fallback to local simulator
      _runLocalSim(input);
    } finally {
      setState(() => _loading = false);
    }
  }

  /// Local PDA simulation (same logic as pda_screen.dart original)
  void _runLocalSim(String input) {
    final rules = _parseLocalRules(_transCtrl.text);
    final simulator = _LocalPdaSimulator(
      rules: rules,
      initialState: _initCtrl.text.trim(),
      initialStackSym: _initSymCtrl.text.trim(),
      acceptStates: _acceptCtrl.text.split(',').map((s) => s.trim()).toSet(),
    );
    final steps = simulator.simulate(input);
    final last = steps.isNotEmpty ? steps.last : null;
    setState(() {
      _simSteps = steps.asMap().entries.map((e) => {'step': e.key, 'text': e.value['action']}).toList();
      _accepted = last != null && last['action'].toString().startsWith('✅');
      _rejected = !_accepted;
      _simStep = 0;
    });
  }

  List<Map<String, String>> _parseLocalRules(String text) {
    final rules = <Map<String, String>>[];
    for (final raw in text.split('\n')) {
      final line = raw.trim();
      if (line.isEmpty || line.startsWith('#')) continue;
      final arrowIdx = line.indexOf('->');
      if (arrowIdx < 0) continue;
      final left = line.substring(0, arrowIdx).trim().split(',');
      final right = line.substring(arrowIdx + 2).trim().split(',');
      if (left.length < 3 || right.isEmpty) continue;
      rules.add({
        'from': left[0].trim(),
        'inputSym': left[1].trim().replaceAll('ε', ''),
        'stackTop': left[2].trim(),
        'to': right[0].trim(),
        'push': right.length > 1 ? right.sublist(1).join(',').trim().replaceAll('ε', '') : '',
      });
    }
    return rules;
  }

  Future<void> _toCfg() async {
    setState(() { _loading = true; _showCfg = true; _cfgText = ''; });
    try {
      final res = await _api.pdaToCfg(_definition);
      setState(() => _cfgText = res['cfg'] as String? ?? '');
    } on ApiException catch (e) {
      setState(() => _cfgText = 'Error: ${e.message}');
    } finally {
      setState(() => _loading = false);
    }
  }

  void _loadExample() {
    _statesCtrl.text = 'q0,q1,q2';
    _inputCtrl.text = 'a,b';
    _stackCtrl.text = 'A,Z';
    _initCtrl.text = 'q0';
    _initSymCtrl.text = 'Z';
    _acceptCtrl.text = 'q2';
    _transCtrl.text = 'q0,a,Z -> q0,AZ\nq0,a,A -> q0,AA\nq0,b,A -> q1,ε\nq1,b,A -> q1,ε\nq1,ε,Z -> q2,Z';
    _validate();
  }

  void _autoPlay() {
    _autoTimer?.cancel();
    _autoTimer = Timer.periodic(const Duration(milliseconds: 700), (t) {
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

  // ── Build ──────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(
        title: const Text('Autómata de Pila (PDA)'),
        actions: [
          TextButton.icon(
            onPressed: _loadExample,
            icon: const Icon(Icons.science_outlined, size: 16),
            label: const Text('Ejemplo'),
          ),
        ],
        bottom: TabBar(
          controller: _tab,
          labelColor: AppColors.primary,
          unselectedLabelColor: AppColors.textSecondary,
          indicatorColor: AppColors.primary,
          tabs: const [
            Tab(icon: Icon(Icons.edit, size: 16), text: 'Definir'),
            Tab(icon: Icon(Icons.play_arrow, size: 16), text: 'Simular'),
            Tab(icon: Icon(Icons.transform, size: 16), text: 'CFG'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tab,
        children: [
          _buildDefineTab(),
          _buildSimTab(),
          _buildCfgTab(),
        ],
      ),
    );
  }

  // ── Tab 1: Define + Graph ─────────────────────────────────────────────────

  Widget _buildDefineTab() {
    return Column(
      children: [
        // Collapsible form
        AnimatedSize(
          duration: const Duration(milliseconds: 250),
          child: _formExpanded ? _buildForm() : _buildFormCollapsed(),
        ),
        // Graph area — takes remaining space
        Expanded(child: _buildGraphArea()),
      ],
    );
  }

  Widget _buildFormCollapsed() {
    return GestureDetector(
      onTap: () => setState(() => _formExpanded = true),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        color: AppColors.surface,
        child: Row(
          children: [
            const Icon(Icons.edit, size: 16, color: AppColors.primary),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                _validMsg.isEmpty ? 'Definición del PDA (toca para editar)' : _validMsg,
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  color: _validMsg.isNotEmpty ? AppColors.success : AppColors.textPrimary,
                  fontSize: 13,
                ),
              ),
            ),
            const Icon(Icons.expand_more, size: 18, color: AppColors.textHint),
          ],
        ),
      ),
    );
  }

  Widget _buildForm() {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(12, 10, 12, 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            Expanded(child: _field(_statesCtrl, 'Estados (Q)', 'q0,q1,q2')),
            const SizedBox(width: 8),
            Expanded(child: _field(_inputCtrl, 'Alfabeto Σ', 'a,b')),
          ]),
          const SizedBox(height: 8),
          Row(children: [
            Expanded(child: _field(_stackCtrl, 'Alfabeto pila Γ', 'A,Z')),
            const SizedBox(width: 8),
            Expanded(child: _field(_initCtrl, 'Estado inicial', 'q0')),
            const SizedBox(width: 8),
            Expanded(child: _field(_initSymCtrl, 'Símbolo pila Z₀', 'Z')),
          ]),
          const SizedBox(height: 8),
          _field(_acceptCtrl, 'Estados aceptación F', 'q2'),
          const SizedBox(height: 8),
          TextField(
            controller: _transCtrl,
            minLines: 3,
            maxLines: 6,
            style: const TextStyle(fontSize: 12, fontFamily: 'monospace'),
            decoration: InputDecoration(
              labelText: 'Transiciones δ  (q,a,Z -> q\',push)',
              hintText: 'q0,a,Z -> q0,AZ\nq0,b,A -> q1,ε',
              isDense: true,
            ),
          ),
          if (_error.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: _ErrorBanner(_error),
            ),
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 10),
            child: Row(children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: _loading ? null : _validate,
                  icon: _loading
                      ? const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                      : const Icon(Icons.check_circle_outline, size: 16),
                  label: const Text('Validar y Visualizar'),
                ),
              ),
              const SizedBox(width: 8),
              IconButton.outlined(
                icon: const Icon(Icons.expand_less, size: 18),
                onPressed: _states.isNotEmpty ? () => setState(() => _formExpanded = false) : null,
                tooltip: 'Minimizar formulario',
              ),
            ]),
          ),
        ],
      ),
    );
  }

  Widget _buildGraphArea() {
    if (_states.isEmpty) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.schema_outlined, size: 48, color: Colors.grey.shade300),
            const SizedBox(height: 12),
            Text('Valida el PDA para ver el autómata',
                style: TextStyle(color: Colors.grey.shade500)),
          ],
        ),
      );
    }
    return Container(
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
      ),
    );
  }

  // ── Tab 2: Simulate ───────────────────────────────────────────────────────

  Widget _buildSimTab() {
    return Column(
      children: [
        // Input row
        Padding(
          padding: const EdgeInsets.fromLTRB(12, 12, 12, 0),
          child: Row(children: [
            Expanded(
              child: TextField(
                controller: _simInputCtrl,
                decoration: const InputDecoration(
                  labelText: 'Cadena a simular',
                  hintText: 'aabb',
                  prefixIcon: Icon(Icons.input, size: 18),
                  isDense: true,
                ),
              ),
            ),
            const SizedBox(width: 8),
            FilledButton.icon(
              onPressed: _loading ? null : _simulate,
              icon: const Icon(Icons.play_arrow, size: 18),
              label: const Text('Simular'),
            ),
          ]),
        ),

        if (_simSteps.isNotEmpty) ...[
          // Result badge
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 10, 12, 0),
            child: _ResultBadge(accepted: _accepted),
          ),
          // Navigation controls
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            child: Row(children: [
              IconButton(icon: const Icon(Icons.first_page), onPressed: _simStep > 0 ? () => setState(() => _simStep = 0) : null),
              IconButton(icon: const Icon(Icons.chevron_left), onPressed: _simStep > 0 ? () => setState(() => _simStep--) : null),
              GestureDetector(
                onTap: _simRunning ? _stopSim : _autoPlay,
                child: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(color: AppColors.primary, shape: BoxShape.circle),
                  child: Icon(_simRunning ? Icons.pause : Icons.play_arrow, color: Colors.white, size: 20),
                ),
              ),
              IconButton(icon: const Icon(Icons.chevron_right), onPressed: _simStep < _simSteps.length - 1 ? () => setState(() => _simStep++) : null),
              IconButton(icon: const Icon(Icons.last_page), onPressed: _simStep < _simSteps.length - 1 ? () => setState(() { _simStep = _simSteps.length - 1; }) : null),
              const SizedBox(width: 8),
              Text('${_simStep + 1}/${_simSteps.length}',
                  style: const TextStyle(fontSize: 12, color: AppColors.textSecondary)),
            ]),
          ),
          // Trace list
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              itemCount: _simSteps.length,
              itemBuilder: (_, i) {
                final isCur = i == _simStep;
                final text = _simSteps[i]['text'] as String? ?? '';
                return AnimatedContainer(
                  duration: const Duration(milliseconds: 150),
                  margin: const EdgeInsets.only(bottom: 3),
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: isCur ? AppColors.primary.withOpacity(0.08) : AppColors.surfaceAlt,
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(color: isCur ? AppColors.primary.withOpacity(0.4) : Colors.transparent),
                  ),
                  child: Row(children: [
                    Container(
                      width: 20, height: 20,
                      alignment: Alignment.center,
                      decoration: BoxDecoration(
                        color: isCur ? AppColors.primary : Colors.transparent,
                        shape: BoxShape.circle,
                      ),
                      child: Text('${i + 1}', style: TextStyle(fontSize: 9, fontWeight: FontWeight.w700, color: isCur ? Colors.white : AppColors.textHint)),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        text,
                        style: TextStyle(
                          fontFamily: 'monospace',
                          fontSize: 11,
                          color: text.startsWith('✅') ? AppColors.success : text.startsWith('❌') ? AppColors.error : AppColors.textPrimary,
                        ),
                      ),
                    ),
                  ]),
                );
              },
            ),
          ),
        ] else
          const Expanded(child: Center(child: Text('Ingresa una cadena y presiona Simular', style: TextStyle(color: AppColors.textHint)))),
      ],
    );
  }

  // ── Tab 3: CFG ────────────────────────────────────────────────────────────

  Widget _buildCfgTab() {
    return Padding(
      padding: const EdgeInsets.all(12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          FilledButton.icon(
            onPressed: _loading ? null : _toCfg,
            icon: const Icon(Icons.transform, size: 16),
            label: const Text('Convertir a CFG'),
          ),
          const SizedBox(height: 12),
          Expanded(
            child: _cfgText.isEmpty
                ? Center(child: Column(mainAxisSize: MainAxisSize.min, children: [
                    Icon(Icons.functions, size: 40, color: Colors.grey.shade300),
                    const SizedBox(height: 10),
                    const Text('Presiona "Convertir a CFG"', style: TextStyle(color: AppColors.textHint)),
                  ]))
                : Container(
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: AppColors.surface,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: AppColors.border),
                    ),
                    child: SingleChildScrollView(
                      child: SelectableText(
                        _cfgText,
                        style: const TextStyle(fontFamily: 'monospace', fontSize: 11, height: 1.6),
                      ),
                    ),
                  ),
          ),
        ],
      ),
    );
  }

  Widget _field(TextEditingController ctrl, String label, String hint) => TextField(
        controller: ctrl,
        style: const TextStyle(fontSize: 12),
        decoration: InputDecoration(labelText: label, hintText: hint, isDense: true),
      );
}

// ─── Local PDA Simulator (fallback) ──────────────────────────────────────────

class _LocalPdaSimulator {
  final List<Map<String, String>> rules;
  final String initialState;
  final String initialStackSym;
  final Set<String> acceptStates;

  _LocalPdaSimulator({
    required this.rules,
    required this.initialState,
    required this.initialStackSym,
    required this.acceptStates,
  });

  List<Map<String, dynamic>> simulate(String input) {
    const maxSteps = 400;
    final initial = _cfg(initialState, 0, [initialStackSym], [
      {'state': initialState, 'action': 'Inicio: estado=$initialState, entrada=$input, pila=[$initialStackSym]'}
    ]);
    final queue = <Map<String, dynamic>>[initial];
    final visited = <String>{};

    while (queue.isNotEmpty && queue.length < maxSteps) {
      final cfg = queue.removeLast();
      final state = cfg['state'] as String;
      final pos = cfg['pos'] as int;
      final stack = List<String>.from(cfg['stack'] as List);
      final steps = List<Map<String, dynamic>>.from(cfg['steps'] as List);

      final key = '$state|$pos|${stack.join(",")}';
      if (visited.contains(key)) continue;
      visited.add(key);

      if (acceptStates.contains(state) && pos >= input.length) {
        return [...steps, {'state': state, 'action': '✅ Cadena ACEPTADA en estado $state'}];
      }

      final top = stack.isEmpty ? null : stack.last;
      if (top == null) continue;

      for (final rule in rules) {
        if (rule['from'] != state) continue;
        if (rule['stackTop'] != top) continue;

        final inputSym = rule['inputSym']!;
        final isEps = inputSym.isEmpty;
        if (!isEps && (pos >= input.length || input[pos] != inputSym)) continue;

        final newStack = List<String>.from(stack)..removeLast();
        final push = rule['push']!;
        if (push.isNotEmpty) {
          for (int k = push.length - 1; k >= 0; k--) newStack.add(push[k]);
        }

        final newPos = isEps ? pos : pos + 1;
        final to = rule['to']!;
        final label = isEps
            ? 'ε-trans: ($state,ε,$top) → ($to,${push.isEmpty ? "ε" : push})'
            : 'Trans: ($state,${input[pos]},$top) → ($to,${push.isEmpty ? "ε" : push})';

        final step = {'state': to, 'action': label};
        queue.add(_cfg(to, newPos, newStack, [...steps, step]));
      }
    }

    return [{'state': initialState, 'action': '❌ Cadena RECHAZADA'}];
  }

  Map<String, dynamic> _cfg(String state, int pos, List<String> stack, List steps) =>
      {'state': state, 'pos': pos, 'stack': stack, 'steps': steps};
}

// ─── Shared Widgets ───────────────────────────────────────────────────────────

class _ResultBadge extends StatelessWidget {
  final bool accepted;
  const _ResultBadge({required this.accepted});

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: (accepted ? AppColors.success : AppColors.error).withOpacity(0.1),
          borderRadius: BorderRadius.circular(24),
          border: Border.all(color: accepted ? AppColors.success : AppColors.error),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Icon(accepted ? Icons.check_circle : Icons.cancel,
              color: accepted ? AppColors.success : AppColors.error, size: 16),
          const SizedBox(width: 8),
          Text(
            accepted ? 'CADENA ACEPTADA' : 'CADENA RECHAZADA',
            style: TextStyle(
              color: accepted ? AppColors.success : AppColors.error,
              fontWeight: FontWeight.w700,
              fontSize: 13,
            ),
          ),
        ]),
      );
}

class _ErrorBanner extends StatelessWidget {
  final String message;
  const _ErrorBanner(this.message);

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.all(10),
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