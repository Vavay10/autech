import 'package:flutter/material.dart';
import '../api_service.dart';

class TuringScreen extends StatefulWidget {
  @override
  _TuringScreenState createState() => _TuringScreenState();
}

class _TuringScreenState extends State<TuringScreen> {
  final TextEditingController transCtrl = TextEditingController(text: "q0,a->q0,a,R\nq0,_->qf,_,S");
  List<dynamic> pasos = [];
  String status = "";

  void _ejecutar() async {
    try {
      final res = await ApiService().simularTuring({
        "states": "q0,qf", "initial": "q0", "accepts": "qf",
        "transitions": transCtrl.text, "cinta": "aaa", "head_pos": 0
      });
      setState(() {
        pasos = res['pasos'];
        status = res['resultado'];
      });
    } catch (e) {
      setState(() => status = "Error: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Simulador Máquina de Turing")),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Column(children: [
          TextField(controller: transCtrl, maxLines: 5, decoration: InputDecoration(labelText: "Transiciones (q,a->p,b,M)", border: OutlineInputBorder())),
          SizedBox(height: 10),
          ElevatedButton(onPressed: _ejecutar, child: Text("Simular")),
          Text("Resultado: $status", style: TextStyle(fontWeight: FontWeight.bold)),
          Divider(),
          if (pasos.isNotEmpty) 
            Column(children: pasos.map((p) => ListTile(
              title: Text("Paso ${p['paso_num']}: Estado ${p['estado']}"),
              subtitle: Text("Cinta: ${p['cinta'].join('')}"),
            )).toList()),
        ]),
      ),
    );
  }
}