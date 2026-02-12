import 'package:http/http.dart' as http;
import 'dart:convert';

class ApiService {
  final String baseUrl = "http://127.0.0.1:8000";

  Future<Map<String, dynamic>> postPDA(Map<String, String> data) async {
    final response = await http.post(
      Uri.parse('$baseUrl/pda/to-cfg'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(data),
    );
    if (response.statusCode == 200) return jsonDecode(response.body);
    throw Exception(jsonDecode(response.body)['detail']);
  }

  Future<String> getRegexImage(String exp) async {
    final response = await http.get(Uri.parse('$baseUrl/regex/to-image?exp=${Uri.encodeComponent(exp)}'));
    if (response.statusCode == 200) return jsonDecode(response.body)['image_base64'];
    throw Exception("Error en el servidor");
  }
  Future<Map<String, dynamic>> simularTuring(Map<String, dynamic> data) async {
  final response = await http.post(
    Uri.parse('$baseUrl/turing/simulate'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode(data),
  );
  if (response.statusCode == 200) return jsonDecode(response.body);
  throw Exception("Error en simulación");
}
}