import 'dart:convert';
import 'package:http/http.dart' as http;

const String _baseUrl = 'http://10.0.2.2:8000';

class ApiService {
  final _client = http.Client();
  Map<String, String> get _headers => {'Content-Type': 'application/json'};

  // ── PDA ─────────────────────────────────────────────────────────────────────

  /// Valida el PDA y retorna el grafo para dibujar.
  /// Respuesta: { "graph": { "states": [...], "edges": [...] }, "valid": true }
  Future<Map<String, dynamic>> validatePda(Map<String, dynamic> definition) =>
      _post('/pda/validate', _pdaBody(definition));

  /// Simula el PDA sobre [inputString].
  /// Respuesta: { "accepted": bool, "trace": [...], "steps": int }
  Future<Map<String, dynamic>> simulatePda(
    Map<String, dynamic> definition,
    String inputString,
  ) =>
      _post('/pda/simulate', {
        ..._pdaBody(definition),
        'input_string': inputString,
      });

  /// Convierte PDA → CFG.
  /// Respuesta: { "cfg": "<texto>" }
  Future<Map<String, dynamic>> pdaToCfg(Map<String, dynamic> definition) =>
      _post('/pda/to-cfg', _pdaBody(definition));

  /// Construye el cuerpo estándar para todas las peticiones de PDA.
  Map<String, dynamic> _pdaBody(Map<String, dynamic> d) => {
        'states': d['states'] ?? '',
        'input_alpha': d['inputAlphabet'] ?? d['input_alpha'] ?? '',
        'stack_alpha': d['stackAlphabet'] ?? d['stack_alpha'] ?? '',
        'start_state': d['startState'] ?? d['start_state'] ?? '',
        'start_symbol': d['startSymbol'] ?? d['start_symbol'] ?? '',
        'accept_states': d['acceptStates'] ?? d['accept_states'] ?? '',
        'transitions': d['transitions'] ?? '',
      };

  // ── Regex ────────────────────────────────────────────────────────────────────

  /// Convierte regex → DFA mínimo.
  /// Respuesta: { "states": [...], "edges": [...], "alphabet": [...] }
  Future<Map<String, dynamic>> regexToAutomaton(String regex) async {
    try {
      final uri = Uri.parse('$_baseUrl/regex/to-automaton')
          .replace(queryParameters: {'exp': regex});
      final resp = await _client
          .get(uri, headers: _headers)
          .timeout(const Duration(seconds: 30));
      final decoded =
          json.decode(utf8.decode(resp.bodyBytes)) as Map<String, dynamic>;
      if (resp.statusCode >= 400) {
        throw ApiException(
            decoded['detail']?.toString() ?? 'Error del servidor');
      }
      return decoded;
    } on ApiException {
      rethrow;
    } catch (e) {
      throw ApiException('No se pudo conectar al servidor: $e');
    }
  }

  /// Autómata → Expresión Regular (eliminación de estados).
  /// [automaton] debe tener: states, transitions, initial, accepting, alphabet.
  /// Respuesta: { "regex": "<expresión>" }
  Future<Map<String, dynamic>> automatonToRegex(
    Map<String, dynamic> automaton,
  ) =>
      _post('/regex/automaton-to-regex', {
        'states': automaton['states'] ?? [],
        'transitions': automaton['transitions'] ?? {},
        'initial': automaton['initial'] ?? '',
        'accepting': automaton['accepting'] ?? [],
        'alphabet': automaton['alphabet'] ?? [],
      });

  /// Operaciones de lenguaje sobre autómatas generados desde regex.
  /// [operation]: "union" | "intersection" | "kleene" | "complement"
  /// Respuesta: { "states": [...], "edges": [...], "alphabet": [...] }
  Future<Map<String, dynamic>> performOperation(
    Map<String, dynamic> payload,
  ) =>
      _post('/regex/operation', {
        'operation': payload['operation'] ?? '',
        'regex1': payload['regex1'] ?? '',
        if (payload['regex2'] != null) 'regex2': payload['regex2'],
      });

  // ── Turing ───────────────────────────────────────────────────────────────────

  /// Construye el grafo de la MT para AutomatonCanvas.
  /// Respuesta: { "states": [...], "edges": [...] }
  Future<Map<String, dynamic>> turingGraph(
    Map<String, dynamic> definition,
  ) =>
      _post('/turing/graph', _turingBody(definition, tape: '', headPos: 0));

  /// Simula la MT paso a paso.
  /// Respuesta: { "steps": [...], "result": "ACCEPTED"|"REJECTED"|"TIMEOUT" }
  Future<Map<String, dynamic>> turingSimulate(
    Map<String, dynamic> definition,
    String tape, {
    int headPos = 0,
    int maxSteps = 500,
  }) =>
      _post('/turing/simulate',
          _turingBody(definition, tape: tape, headPos: headPos, maxSteps: maxSteps));

  /// Construye el cuerpo estándar para todas las peticiones de Turing.
  Map<String, dynamic> _turingBody(
    Map<String, dynamic> d, {
    required String tape,
    required int headPos,
    int maxSteps = 500,
  }) =>
      {
        'states': d['states'] ?? '',
        'initial': d['initial'] ?? '',
        'accepts': d['acceptStates'] ?? d['accepts'] ?? '',
        'transitions': d['transitions'] ?? '',
        'cinta': tape,
        'head_pos': headPos,
        'max_steps': maxSteps,
      };

  // ── Interno ──────────────────────────────────────────────────────────────────

  Future<Map<String, dynamic>> _post(
    String path,
    Map<String, dynamic> body,
  ) async {
    try {
      final resp = await _client
          .post(
            Uri.parse('$_baseUrl$path'),
            headers: _headers,
            body: json.encode(body),
          )
          .timeout(const Duration(seconds: 30));

      final decoded =
          json.decode(utf8.decode(resp.bodyBytes)) as Map<String, dynamic>;
      if (resp.statusCode >= 400) {
        throw ApiException(
            decoded['detail']?.toString() ?? 'Error del servidor');
      }
      return decoded;
    } on ApiException {
      rethrow;
    } catch (e) {
      throw ApiException('No se pudo conectar al servidor: $e');
    }
  }

  void dispose() => _client.close();
}

class ApiException implements Exception {
  final String message;
  const ApiException(this.message);
  @override
  String toString() => message;
}