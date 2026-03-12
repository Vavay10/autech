import 'dart:convert';
import 'package:http/http.dart' as http;

const String _baseUrl = 'http://10.0.2.2:8000';

class ApiService {
  final _client = http.Client();
  Map<String, String> get _headers => {'Content-Type': 'application/json'};

  // ── PDA ─────────────────────────────────────────────────────────────────────

  /// Valida el PDA y construye el grafo para dibujar.
  /// El backend no tiene /pda/validate, así que construimos el grafo localmente
  /// y lanzamos ApiException para que la pantalla use su fallback local.
  Future<Map<String, dynamic>> validatePda(Map<String, dynamic> definition) {
    // No existe este endpoint en el backend → la PdaScreen tiene fallback local.
    return Future.error(const ApiException('Endpoint no disponible: use simulación local'));
  }

  /// Simula el PDA. El backend no tiene /pda/simulate → fallback local.
  Future<Map<String, dynamic>> simulatePda(
      Map<String, dynamic> definition, String inputString) {
    return Future.error(const ApiException('Endpoint no disponible: use simulación local'));
  }

  /// Convierte PDA → CFG. Envía los campos planos que espera PDARequest.
  Future<Map<String, dynamic>> pdaToCfg(Map<String, dynamic> definition) =>
      _post('/pda/to-cfg', {
        'states':       definition['states']        ?? '',
        'input_alpha':  definition['inputAlphabet'] ?? definition['input_alpha'] ?? '',
        'stack_alpha':  definition['stackAlphabet'] ?? definition['stack_alpha'] ?? '',
        'start_state':  definition['startState']    ?? definition['start_state'] ?? '',
        'start_symbol': definition['startSymbol']   ?? definition['start_symbol'] ?? '',
        'accept_states':definition['acceptStates']  ?? definition['accept_states'] ?? '',
        'transitions':  definition['transitions']   ?? '',
      });

  // ── Regex ────────────────────────────────────────────────────────────────────

  /// Convierte regex → DFA mínimo.
  /// Es un GET con query param ?exp=<regex> (NO es POST).
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
        throw ApiException(decoded['detail']?.toString() ?? 'Error del servidor');
      }
      return decoded;
    } on ApiException {
      rethrow;
    } catch (e) {
      throw ApiException('No se pudo conectar al servidor: $e');
    }
  }

  /// Autómata → Regex. El backend aún no tiene este endpoint.
  Future<Map<String, dynamic>> automatonToRegex(
      Map<String, dynamic> automaton) {
    return Future.error(
        const ApiException('automatonToRegex aún no implementado en el backend'));
  }

  /// Operaciones de lenguaje (unión, intersección, etc.). Pendiente en backend.
  Future<Map<String, dynamic>> performOperation(
      Map<String, dynamic> payload) {
    return Future.error(
        const ApiException('performOperation aún no implementado en el backend'));
  }

  // ── Turing ───────────────────────────────────────────────────────────────────

  /// Construye el grafo de la MT para dibujar en AutomatonCanvas.
  /// Traduce los campos del frontend (acceptStates) al formato del backend (accepts).
  Future<Map<String, dynamic>> turingGraph(
      Map<String, dynamic> definition) =>
      _post('/turing/graph', {
        'states':      definition['states']      ?? '',
        'initial':     definition['initial']     ?? '',
        'accepts':     definition['acceptStates'] ?? definition['accepts'] ?? '',
        'transitions': definition['transitions'] ?? '',
        'cinta':       '',       // no se necesita para el grafo
        'head_pos':    0,
      });

  /// Simula la MT paso a paso.
  /// Retorna {"steps": [...], "result": "ACCEPTED"|"REJECTED"|"TIMEOUT"}.
  Future<Map<String, dynamic>> turingSimulate(
      Map<String, dynamic> definition,
      String tape, {
      int headPos = 0,
      int maxSteps = 500,
  }) =>
      _post('/turing/simulate', {
        'states':      definition['states']      ?? '',
        'initial':     definition['initial']     ?? '',
        'accepts':     definition['acceptStates'] ?? definition['accepts'] ?? '',
        'transitions': definition['transitions'] ?? '',
        'cinta':       tape,
        'head_pos':    headPos,
        'max_steps':   maxSteps,
      });

  // ── Interno ──────────────────────────────────────────────────────────────────

  Future<Map<String, dynamic>> _post(
      String path, Map<String, dynamic> body) async {
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
        throw ApiException(decoded['detail']?.toString() ?? 'Error del servidor');
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