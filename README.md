# 💸 Ahorro ODE

Simulador de ahorro familiar utilizando **ecuaciones diferenciales ordinarias (EDO)** de primer y segundo orden.  
Construido con **Streamlit**, permite a cualquier persona proyectar escenarios financieros de manera sencilla.

## 📌 Funcionalidad

- **Modelos de primer orden** (`growth` y `decay`):
  - `growth`: $S' = +rS + A$
  - `decay`: $S' = -rS + A$ (equilibrio en $A/r$ si $r>0$)
- **Modelos de segundo orden (LC2):**
  - Respuesta a **cambio de meta** (escalón).
  - **Estacionalidad sinusoidal** (meses caros/baratos).
- Descarga de resultados en **CSV** y gráficas en **PNG**.
- Cálculo de métricas clave: tiempo de establecimiento (±5%), sobreimpulso, amplitud estacional, desfase.

## ⚙️ Instalación local

1. Clona este repositorio:

   ```bash
   git clone https://github.com/tu-usuario/ahorro-ode.git
   cd ahorro-ode
   ```

2. (Opcional) Crea un entorno virtual:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Instala dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Ejecuta la app:

   ```bash
   streamlit run app_streamlit.py
   ```

5. Abre en el navegador:

   ```
   http://localhost:8501
   ```

## 🚀 Despliegue en Streamlit Community Cloud

1. Sube los archivos a un repo de GitHub (`app_streamlit.py`, `ode_finance.py`, `requirements.txt`, `README.md`).
2. Ve a [Streamlit Community Cloud](https://share.streamlit.io).
3. Crea una nueva app y selecciona:

   - **Repositorio:** tu repo
   - **Archivo principal:** `app_streamlit.py`
   - **Branch:** main

4. Haz clic en **Deploy**.
5. Obtendrás una URL pública que podrás compartir.

## 📂 Estructura del proyecto

```
ahorro-ode/
├── app_streamlit.py   # Interfaz Streamlit
├── ode_finance.py     # Motor matemático (modelos EDO)
├── requirements.txt   # Dependencias
└── README.md          # Documentación
```

## 📖 Ejemplo rápido

### Primer orden (decay)

- Ahorro inicial: 0
- Aporte neto anual: 10,000,000 COP
- r = 0.12 (12%)
- Horizonte: 10 años

### Segundo orden (meta escalón)

- Meta: 60,000,000 COP
- ζ = 0.7 (amortiguación moderada)
- ωₙ = 1.0 (ajuste rápido)

## 📚 Referencias (para fines académicos)

- Mesa, F. (2012). _Ecuaciones diferenciales ordinarias: una introducción_. Ecoe Ediciones.
- Zill, D. G. (2015). _Ecuaciones diferenciales con aplicaciones de modelado_ (10a. ed.). Cengage Learning.
- Agud Albesa, L., Pla Ferrando, L., & Boix García, M. (2020). _Funciones de varias variables y ecuaciones diferenciales: ejercicios resueltos analíticamente y con Matlab_. Editorial UPV.

## 👤 Autor

Proyecto académico desarrollado Por Juan Camilo García Martín como parte del curso de **Ecuaciones Diferenciales**, Universidad Compensar.
