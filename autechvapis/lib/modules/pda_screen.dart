import 'package:flutter/material.dart';
import '../api_service.dart';

class PdaScreen extends StatefulWidget {
  @override
  _PdaScreenState createState() => _PdaScreenState();
}

class _PdaScreenState extends State<PdaScreen> {
  final TextEditingController qCtrl = TextEditingController(text: "q0,q1");
  final TextEditingController transCtrl = TextEditingController(text: "q0,a,Z -> q1,Z");
  String result = "";

  void _enviar() async {
    try {
      final res = await ApiService().postPDA({
        "states": qCtrl.text, "input_alpha": "a,b", "stack_alpha": "Z,A",
        "start_state": "q0", "start_symbol": "Z", "accept_states": "q1",
        "transitions": transCtrl.text,
      });
      setState(() => result = res['cfg']);
    } catch (e) {
      setState(() => result = "Error: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Módulo AP")),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Column(children: [
          TextField(controller: qCtrl, decoration: InputDecoration(labelText: "Estados (Q)")),
          TextField(controller: transCtrl, decoration: InputDecoration(labelText: "Transiciones"), maxLines: 3),
          ElevatedButton(onPressed: _enviar, child: Text("Convertir")),
          Divider(),
          SelectableText(result),
        ]),
      ),
    );
  }
}