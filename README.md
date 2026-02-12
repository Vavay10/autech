# Autech - Suite de Teoría de la Computación

**Autech** es una aplicación educativa integral diseñada para experimentar con conceptos fundamentales de la Teoría de la Computación. Combina la potencia de **Python** para el procesamiento algorítmico y **Flutter** para una interfaz de usuario moderna.

## 📦 Estructura del Proyecto
* `/autechvapis`: Cliente frontend desarrollado en **Flutter**.
* `/backend`: Lógica computacional en **Python** y servidor de API.
  * `api_server.py`: Servidor FastAPI (Punto de entrada).
  * `AP.py`, `expre.py`, `turing3.py`: Módulos de lógica.

---

## 🚀 Guía de Instalación y Ejecución

### 1. Requisitos Previos
* **Flutter SDK** (Versión estable).
* **Python 3.10+**
* **Git**

### 2. Configuración del Backend (Python)
Es necesario tener el servidor activo para que la aplicación de Flutter pueda procesar los datos.

```bash
# Entrar a la carpeta del backend
cd backend

# Instalar dependencias necesarias
pip install fastapi uvicorn matplotlib networkx flet

# Ejecutar el servidor
python api_server.py

Proyecto desarrollado para el apoyo a los estudiantes de ToC - Instituto Politécnico Nacional (IPN) para nuestra titulación.