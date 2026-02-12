import 'package:flutter/material.dart';
import 'dart:convert';
import '../api_service.dart';

class RegexScreen extends StatefulWidget {
  @override
  _RegexScreenState createState() => _RegexScreenState();
}

class _RegexScreenState extends State<RegexScreen> {
  final TextEditingController _controller = TextEditingController(text: "(a|b)*abb");
  String? _imageB64;
  bool _loading = false;

  void _generar() async {
    setState(() => _loading = true);
    try {
      final img = await ApiService().getRegexImage(_controller.text);
      setState(() => _imageB64 = img);
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Error: $e")));
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Visualizador de Autómatas (Regex)")),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Column(
          children: [
            TextField(
              controller: _controller,
              decoration: InputDecoration(labelText: "Expresión Regular", border: OutlineInputBorder()),
            ),
            SizedBox(height: 10),
            ElevatedButton(
              onPressed: _loading ? null : _generar, 
              child: _loading ? CircularProgressIndicator() : Text("Generar Imagen")
            ),
            Divider(height: 40),
            if (_imageB64 != null)
              Container(
                decoration: BoxDecoration(border: Border.all(color: Colors.grey)),
                child: Image.memory(base64Decode(_imageB64!), fit: BoxFit.contain),
              )
            else
              Text("Introduce una expresión para ver el autómata"),
          ],
        ),
      ),
    );
  }
}