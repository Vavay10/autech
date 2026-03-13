import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'theme.dart';
import 'general/splash_screen.dart';
import 'panels/alumno/dashboard_screen.dart';
import 'panels/alumno/learning_path_screen.dart';
import 'panels/alumno/simulator_screen.dart';
import 'panels/alumno/stats_screen.dart';
import 'panels/alumno/settings_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.dark,
  ));
  runApp(const AuTechApp());
}

class AuTechApp extends StatelessWidget {
  const AuTechApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AUTECH',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light,
      home: const SplashScreen(),
    );
  }
}

// ─── Main Scaffold ────────────────────────────────────────────────────────────

class MainScaffold extends StatefulWidget {
  const MainScaffold({super.key});

  @override
  State<MainScaffold> createState() => _MainScaffoldState();
}

class _MainScaffoldState extends State<MainScaffold> {
  int _idx = 0;

  // ✅ Fix: declarar explícitamente como List<Widget> para evitar List<dynamic>
  static const List<Widget> _screens = <Widget>[
    DashboardScreen(),
    LearningPathScreen(),
    SimulatorScreen(),
    StatsScreen(),
    SettingsScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _idx,
        children: _screens,
      ),
      bottomNavigationBar: _GameBottomNav(
        currentIndex: _idx,
        onTap: (i) => setState(() => _idx = i),
      ),
    );
  }
}

// ─── Game-style Bottom Navigation ────────────────────────────────────────────

class _GameBottomNav extends StatelessWidget {
  final int currentIndex;
  final ValueChanged<int> onTap;

  const _GameBottomNav({required this.currentIndex, required this.onTap});

  static const List<_NavItem> _items = <_NavItem>[
    _NavItem(label: 'Inicio',   icon: Icons.home_rounded),
    _NavItem(label: 'Aprender', icon: Icons.school_rounded),
    _NavItem(label: 'Simular',  icon: Icons.memory_rounded),
    _NavItem(label: 'Progreso', icon: Icons.bar_chart_rounded),
    _NavItem(label: 'Perfil',   icon: Icons.person_rounded),
  ];

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.surface,
        border: Border(top: BorderSide(color: AppColors.border)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 12,
            offset: const Offset(0, -3),
          ),
        ],
      ),
      child: SafeArea(
        child: SizedBox(
          height: 64,
          child: Row(
            children: List<Widget>.generate(_items.length, (i) {
              final item = _items[i];
              final selected = currentIndex == i;
              return Expanded(
                child: GestureDetector(
                  onTap: () => onTap(i),
                  behavior: HitTestBehavior.opaque,
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      AnimatedContainer(
                        duration: const Duration(milliseconds: 200),
                        padding: const EdgeInsets.all(5),
                        decoration: BoxDecoration(
                          color: selected
                              ? AppColors.primary.withOpacity(0.12)
                              : Colors.transparent,
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: AnimatedScale(
                          scale: selected ? 1.12 : 1.0,
                          duration: const Duration(milliseconds: 200),
                          child: Container(
                            width: 30,
                            height: 30,
                            decoration: BoxDecoration(
                              color: selected
                                  ? AppColors.primary
                                  : AppColors.surfaceAlt,
                              borderRadius: BorderRadius.circular(8),
                              border: Border.all(
                                color: selected
                                    ? AppColors.primaryDark
                                    : AppColors.border,
                                width: 1.5,
                              ),
                            ),
                            child: Center(
                              child: Icon(
                                item.icon,
                                size: 16,
                                color:
                                    selected ? Colors.white : AppColors.textHint,
                              ),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(height: 2),
                      AnimatedDefaultTextStyle(
                        duration: const Duration(milliseconds: 200),
                        style: TextStyle(
                          fontSize: 10,
                          fontWeight: selected
                              ? FontWeight.w700
                              : FontWeight.w500,
                          color:
                              selected ? AppColors.primary : AppColors.textHint,
                        ),
                        child: Text(item.label),
                      ),
                    ],
                  ),
                ),
              );
            }),
          ),
        ),
      ),
    );
  }
}

class _NavItem {
  final String label;
  final IconData icon;
  const _NavItem({required this.label, required this.icon});
}