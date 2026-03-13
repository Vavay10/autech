import 'package:flutter/material.dart';
import 'package:autechvapis/panels/alumno/dashboard_screen.dart';
import 'package:autechvapis/panels/alumno/simulator_screen.dart';
import 'package:autechvapis/panels/alumno/learning_path_screen.dart';
import 'package:autechvapis/panels/alumno/stats_screen.dart';
import 'package:autechvapis/panels/alumno/settings_screen.dart';
import 'package:autechvapis/theme.dart';

class MainLayout extends StatefulWidget {
  const MainLayout({super.key});

  @override
  State<MainLayout> createState() => _MainLayoutState();
}

class _MainLayoutState extends State<MainLayout> {
  int _index = 0;

  static const _pages = [
    DashboardScreen(),
    SimulatorScreen(),
    LearningPathScreen(),
    StatsScreen(),
    SettingsScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(index: _index, children: _pages),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          color: AppColors.surface,
          border: const Border(top: BorderSide(color: AppColors.border)),
          boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.05), blurRadius: 10, offset: const Offset(0, -3))],
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 6),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _NavItem(icon: Icons.home_outlined, selectedIcon: Icons.home, label: 'Inicio', index: 0, current: _index, onTap: _setIndex),
                _NavItem(icon: Icons.memory_outlined, selectedIcon: Icons.memory, label: 'Simulador', index: 1, current: _index, onTap: _setIndex),
                _NavItem(icon: Icons.psychology_outlined, selectedIcon: Icons.psychology, label: 'Aprender', index: 2, current: _index, onTap: _setIndex),
                _NavItem(icon: Icons.bar_chart_outlined, selectedIcon: Icons.bar_chart, label: 'Stats', index: 3, current: _index, onTap: _setIndex),
                _NavItem(icon: Icons.person_outline, selectedIcon: Icons.person, label: 'Perfil', index: 4, current: _index, onTap: _setIndex),
              ],
            ),
          ),
        ),
      ),
    );
  }

  void _setIndex(int i) => setState(() => _index = i);
}

class _NavItem extends StatelessWidget {
  final IconData icon, selectedIcon;
  final String label;
  final int index, current;
  final void Function(int) onTap;

  const _NavItem({
    required this.icon, required this.selectedIcon, required this.label,
    required this.index, required this.current, required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final selected = index == current;
    return GestureDetector(
      onTap: () => onTap(index),
      behavior: HitTestBehavior.opaque,
      child: SizedBox(
        width: 64,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
              decoration: BoxDecoration(
                color: selected ? AppColors.primary.withValues(alpha: 0.12) : Colors.transparent,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Icon(
                selected ? selectedIcon : icon,
                color: selected ? AppColors.primary : AppColors.textHint,
                size: 22,
              ),
            ),
            const SizedBox(height: 2),
            Text(
              label,
              style: TextStyle(
                fontSize: 10,
                color: selected ? AppColors.primary : AppColors.textHint,
                fontWeight: selected ? FontWeight.w700 : FontWeight.normal,
              ),
            ),
          ],
        ),
      ),
    );
  }
}