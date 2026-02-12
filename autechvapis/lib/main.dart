import 'package:flutter/material.dart';
import 'modules/pda_screen.dart';
import 'modules/regex_screen.dart';
import 'modules/turing_screen.dart';

void main() => runApp(MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: Colors.blue,
      ),
      home: MainContainer(),
    ));

class MainContainer extends StatefulWidget {
  @override
  _MainContainerState createState() => _MainContainerState();
}

class _MainContainerState extends State<MainContainer> {
  int _selectedIndex = 0;

  // Lista de tus módulos importados
  final List<Widget> _pages = [
    RegexScreen(),  // Índice 0
    PdaScreen(),    // Índice 1
    TuringScreen(), // Índice 2
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Suite de Teoría de la Computación"),
        backgroundColor: Colors.blue.shade50,
      ),
      body: Row(
        children: [
          // Este es el menú lateral que siempre estará visible
          NavigationRail(
            selectedIndex: _selectedIndex,
            onDestinationSelected: (int index) {
              setState(() {
                _selectedIndex = index;
              });
            },
            labelType: NavigationRailLabelType.all,
            destinations: const [
              NavigationRailDestination(
                icon: Icon(Icons.hub_outlined),
                selectedIcon: Icon(Icons.hub),
                label: Text('AF (Regex)'),
              ),
              NavigationRailDestination(
                icon: Icon(Icons.layers_outlined),
                selectedIcon: Icon(Icons.layers),
                label: Text('AP (Pila)'),
              ),
              NavigationRailDestination(
                icon: Icon(Icons.memory_outlined),
                selectedIcon: Icon(Icons.memory),
                label: Text('MT (Turing)'),
              ),
            ],
          ),
          const VerticalDivider(thickness: 1, width: 1),
          // Aquí se muestra la pantalla seleccionada
          Expanded(
            child: _pages[_selectedIndex],
          ),
        ],
      ),
    );
  }
}