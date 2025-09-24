"""
Aplicación Streamlit para simulación de modelos de ahorro con EDO.
Versión mejorada con mejor UX y manejo de errores.
"""

import streamlit as st
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
from ode_finance import (
    first_order_closed_form,
    lc2_step_analytic,
    lc2_sin_analytic,
    settling_time,
    compute_overshoot,
    compute_rise_time,
    ModelType,
    DampingRegime,
)

# Configuración de la página
st.set_page_config(
    page_title="Ahorro ODE",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configuración de matplotlib para mejor calidad
plt.style.use("default")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10

# -------------------- Cabecera --------------------
st.title("💸 Ahorro ODE: Simulador de Ahorro Familiar con EDO")

st.markdown(
    """
    **¿Qué hace esta herramienta?**  
    Te permite **proyectar tu ahorro** con modelos matemáticos sencillos. No necesitas saber cálculo:
    solo ingresa tus datos y observa **cómo cambia el saldo con el tiempo**.
    
    **Cómo usarla (3 pasos):**
    1. Elige una pestaña según tu caso: **Primer orden** (aportes constantes), **Escalón** (cambié la meta),
       o **Estacionalidad** (meses caros/baratos).
    2. Completa los campos con tus cifras. Cada campo tiene una **ayuda** (pasa el cursor por el ⓘ).
    3. Pulsa **Simular**. Puedes **descargar** la tabla (CSV) y el gráfico (PNG).
    """
)

st.caption(
    "Modelos: primer orden (growth/decay), segundo orden (LC2: escalón de meta y estacionalidad). Exporta CSV/PNG."
)

# Panel lateral con explicación corta (no técnica)
with st.sidebar:
    st.header("📊 Guía Rápida")
    st.markdown(
        """
        ### Parámetros Principales:
        - **S(0):** lo que tienes hoy ahorrado
        - **S'(0):** velocidad inicial de cambio (normalmente 0)
        - **Meta S★:** cuánto quieres alcanzar
        - **ωₙ:** qué tan **rápido** corriges rumbo (0.5–2 por año)
        - **ζ:** qué tanto **evitas rebotar** u oscilar (≈1 es ideal)
        - **F:** tamaño del **pico** de gasto/ingreso estacional
        - **Ω:** **frecuencia** de ese ciclo (anual ≈ 2π)
        """
    )

    st.info(
        "💡 **Consejo:** Si ves oscilaciones, sube ζ hacia 1; si la respuesta es muy lenta, aumenta ωₙ."
    )

    st.markdown("---")
    st.markdown("### 🎯 Valores Típicos")
    st.markdown(
        """
        - **ωₙ:** 0.5-2.0 (anual)
        - **ζ:** 0.7-1.2 (estable)
        - **r:** 0.05-0.15 (5%-15%)
        - **Ω:** 2π (anual), 4π (semestral)
        """
    )


# -------------------- Funciones auxiliares --------------------
def format_currency(value):
    """Formatear valores monetarios."""
    return f"${value:,.0f}"


def create_download_section(fig, data_dict, base_filename):
    """Crear sección de descarga unificada."""
    col1, col2 = st.columns(2)

    with col1:
        # CSV download
        df = pd.DataFrame(data_dict)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📄 Descargar CSV",
            data=csv,
            file_name=f"{base_filename}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        # PNG download
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        st.download_button(
            "🖼️ Descargar PNG",
            data=buf.getvalue(),
            file_name=f"{base_filename}.png",
            mime="image/png",
            use_container_width=True,
        )


def validate_inputs(**kwargs):
    """Validar inputs comunes."""
    errors = []

    if "tmax" in kwargs and kwargs["tmax"] <= 0:
        errors.append("El tiempo final debe ser positivo")

    if "dt" in kwargs and kwargs["dt"] <= 0:
        errors.append("El paso temporal debe ser positivo")

    if "wn" in kwargs and kwargs["wn"] <= 0:
        errors.append("La frecuencia natural (ωₙ) debe ser positiva")

    if "zeta" in kwargs and kwargs["zeta"] < 0:
        errors.append("El factor de amortiguación (ζ) debe ser no negativo")

    return errors


# -------------------- Pestañas --------------------
tab1, tab2, tab3 = st.tabs(
    ["🔄 Primer Orden", "🎯 LC2: Escalón de Meta", "🌊 LC2: Estacionalidad"]
)

# ---------- PRIMER ORDEN ----------
with tab1:
    st.header("🔄 Primer Orden (Aportes Constantes)")

    st.markdown(
        """
        Usa este modelo si **aportas una cantidad constante cada año** y quieres ver
        cómo crece tu ahorro con el tiempo. Es el más simple y directo.
        """
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Parámetros del Modelo")

        model = st.selectbox(
            "Tipo de modelo",
            [ModelType.GROWTH.value, ModelType.DECAY.value],
            index=1,
            help=(
                "• **Growth**: S' = +r·S + A (crece más cuando ya tienes más ahorro)\n"
                "• **Decay**: S' = -r·S + A (se acerca a un equilibrio A/r; útil para aportes netos fijos)"
            ),
        )

        col1a, col1b = st.columns(2)
        with col1a:
            S0 = st.number_input(
                "Ahorro inicial S(0) [COP]",
                value=0.0,
                step=1e6,
                format="%.0f",
                help="¿Cuánto tienes hoy ahorrado? Ejemplo: 3,000,000",
            )

            r = st.number_input(
                "Tasa r [1/año]",
                value=0.12,
                step=0.01,
                format="%.4f",
                help="Velocidad de cambio del modelo. Valores típicos: 0.05–0.20 (5%-20% anual)",
            )

        with col1b:
            A = st.number_input(
                "Aporte neto anual A [COP/año]",
                value=1e7,
                step=1e6,
                format="%.0f",
                help="Aportes menos retiros al año. Ejemplo: 10,000,000",
            )

            tmax = st.number_input(
                "Horizonte temporal [años]",
                value=10.0,
                step=0.5,
                format="%.1f",
                help="Durante cuántos años quieres simular",
            )

        dt = st.number_input(
            "Paso temporal dt",
            value=0.01,
            step=0.01,
            format="%.3f",
            help="Precisión de la simulación. Valor más pequeño = curva más suave",
        )

    with col2:
        st.subheader("Ecuaciones de Referencia")
        st.latex(
            r"""
        \begin{cases}
        \text{Growth:} & S' = +r \cdot S + A \\
        \text{Decay:} & S' = -r \cdot S + A
        \end{cases}
        """
        )

        if model == ModelType.DECAY.value and r > 0:
            equilibrium = A / r
            st.info(f"Equilibrio: {format_currency(equilibrium)}")

    if st.button("🚀 Simular (Primer Orden)", use_container_width=True):
        # Validar inputs
        errors = validate_inputs(tmax=tmax, dt=dt)

        if errors:
            for error in errors:
                st.error(error)
        else:
            try:
                with st.spinner("Calculando..."):
                    t = np.arange(0.0, tmax + 1e-12, dt, dtype=float)
                    S = first_order_closed_form(model, t, S0, r, A)

                # Crear gráfico
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(t, S, linewidth=2, color="#1f77b4")
                ax.set_xlabel("Tiempo [años]")
                ax.set_ylabel("Saldo S(t) [COP]")
                ax.set_title(f"Modelo de Primer Orden ({model.title()})")
                ax.grid(True, alpha=0.3)
                ax.ticklabel_format(style="plain", axis="y")

                # Formatear eje Y
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M")
                )

                st.pyplot(fig, use_container_width=True)

                # Mostrar estadísticas
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("Valor Inicial", format_currency(S[0]))
                with col_stats2:
                    st.metric("Valor Final", format_currency(S[-1]))
                with col_stats3:
                    growth_rate = (
                        ((S[-1] - S[0]) / S[0] * 100) if S[0] != 0 else float("inf")
                    )
                    st.metric("Crecimiento Total", f"{growth_rate:.1f}%")

                # Sección de descarga
                st.subheader("📥 Descargar Resultados")
                create_download_section(fig, {"t": t, "S": S}, "primer_orden")

            except Exception as e:
                st.error(f"Error en la simulación: {str(e)}")

# ---------- LC2: ESCALÓN ----------
with tab2:
    st.header("🎯 Segundo Orden (LC2) — Cambié la Meta de Ahorro")

    st.markdown(
        """
        Usa este modelo si **fijaste una nueva meta** (ej: 60 millones) y quieres ver
        **cómo tu ahorro se ajusta** hacia esa meta. Este modelo describe **inercia** (hábitos)
        y permite evitar **rebotes/oscilaciones** ajustando correctamente los parámetros.
        """
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Parámetros del Sistema")

        col1a, col1b = st.columns(2)
        with col1a:
            S0_step = st.number_input(
                "Ahorro inicial S(0) [COP]",
                value=0.0,
                step=1e6,
                format="%.0f",
                key="s0_step",
                help="¿Cuánto tienes hoy ahorrado?",
            )

            V0 = st.number_input(
                "Ritmo inicial S'(0) [COP/año]",
                value=0.0,
                step=1e6,
                format="%.0f",
                help="Velocidad inicial de cambio (normalmente 0)",
            )

            Sstar = st.number_input(
                "Nueva meta S★ [COP]",
                value=6e7,
                step=1e6,
                format="%.0f",
                help="Nivel objetivo que quieres alcanzar",
            )

        with col1b:
            wn = st.number_input(
                "Velocidad de ajuste ωₙ [1/año]",
                value=1.0,
                step=0.1,
                format="%.2f",
                help="Qué tan rápido corriges el rumbo (0.5–2.0 son valores típicos)",
            )

            zeta = st.number_input(
                "Factor de amortiguación ζ",
                value=0.7,
                step=0.05,
                format="%.2f",
                help="Controla las oscilaciones. ζ≈1 = rápido sin rebote; ζ<1 = puede oscilar",
            )

            tmax_step = st.number_input(
                "Horizonte temporal [años]",
                value=6.0,
                step=0.5,
                format="%.1f",
                key="tmax_step",
                help="Durante cuántos años simular el ajuste",
            )

        dt_step = st.number_input(
            "Paso temporal dt",
            value=0.005,
            step=0.005,
            format="%.3f",
            key="dt_step",
            help="Precisión de la simulación",
        )

    with col2:
        st.subheader("Modelo Matemático")
        st.latex(
            r"""
        S'' + 2\zeta \omega_n S' + \omega_n^2 S = \omega_n^2 S^*
        """
        )

        # Mostrar régimen de amortiguación
        if zeta < 1:
            regime = "Subamortiguado"
            color = "🟡"
        elif zeta == 1:
            regime = "Críticamente amortiguado"
            color = "🟢"
        else:
            regime = "Sobreamortiguado"
            color = "🔵"

        st.info(f"{color} **Régimen:** {regime}")

        # Métricas teóricas
        if zeta < 1 and wn > 0:
            tp = np.pi / (wn * np.sqrt(1 - zeta**2))
            Mp = np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))
            ts = 3 / (zeta * wn) if zeta > 0 else float("inf")

            st.markdown("**Métricas teóricas:**")
            st.text(f"• Tiempo al pico: {tp:.2f} años")
            st.text(f"• Sobreimpulso: {Mp*100:.1f}%")
            st.text(f"• Tiempo estab.: {ts:.2f} años")

    if st.button("🎯 Simular (LC2: Escalón)", use_container_width=True):
        # Validar inputs
        errors = validate_inputs(tmax=tmax_step, dt=dt_step, wn=wn, zeta=zeta)

        if errors:
            for error in errors:
                st.error(error)
        else:
            try:
                with st.spinner("Calculando respuesta del sistema..."):
                    t = np.arange(0.0, tmax_step + 1e-12, dt_step, dtype=float)
                    S = lc2_step_analytic(t, S0_step, V0, Sstar, wn, zeta)
                    ts = settling_time(t, S, Sstar, tol=0.05)
                    overshoot = compute_overshoot(S, Sstar)
                    rise_time = compute_rise_time(t, S, Sstar)

                # Crear gráfico
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(t, S, label="S(t)", linewidth=2, color="#1f77b4")
                ax.axhline(
                    Sstar, linestyle="--", label="S★ (meta)", color="red", alpha=0.7
                )

                # Bandas de tolerancia
                tol_band = 0.05 * abs(Sstar)
                ax.fill_between(
                    t,
                    Sstar - tol_band,
                    Sstar + tol_band,
                    alpha=0.2,
                    color="red",
                    label="±5% tolerancia",
                )

                ax.set_xlabel("Tiempo [años]")
                ax.set_ylabel("Saldo S(t) [COP]")
                ax.set_title("LC2 — Ajuste hacia la Meta")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M")
                )

                st.pyplot(fig, use_container_width=True)

                # Mostrar métricas
                col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                with col_met1:
                    st.metric(
                        "Tiempo de establecimiento",
                        f"{ts:.2f} años" if not np.isnan(ts) else "N/A",
                    )
                with col_met2:
                    st.metric(
                        "Sobreimpulso",
                        f"{overshoot:.1f}%" if not np.isnan(overshoot) else "N/A",
                    )
                with col_met3:
                    st.metric(
                        "Tiempo de subida",
                        f"{rise_time:.2f} años" if not np.isnan(rise_time) else "N/A",
                    )
                with col_met4:
                    st.metric("Valor final", format_currency(S[-1]))

                # Sección de descarga
                st.subheader("📥 Descargar Resultados")
                create_download_section(fig, {"t": t, "S": S}, "lc2_escalon")

            except Exception as e:
                st.error(f"Error en la simulación: {str(e)}")

# ---------- LC2: ESTACIONALIDAD ----------
with tab3:
    st.header("🌊 Segundo Orden (LC2) — Estacionalidad")

    st.markdown(
        """
        Usa este modelo si tus aportes/gastos **cambian en ciclos** (ej: meses costosos cada año).
        Podrás estimar la **ondulación** alrededor de la meta y su **desfase** temporal.
        """
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Parámetros del Sistema")

        col1a, col1b = st.columns(2)
        with col1a:
            S0_sin = st.number_input(
                "Ahorro inicial S(0) [COP]",
                value=0.0,
                step=1e6,
                format="%.0f",
                key="s0_sin",
                help="¿Cuánto tienes hoy ahorrado?",
            )

            V0_sin = st.number_input(
                "Ritmo inicial S'(0) [COP/año]",
                value=0.0,
                step=1e6,
                format="%.0f",
                key="v0_sin",
                help="Velocidad inicial (normalmente 0)",
            )

            Sstar_sin = st.number_input(
                "Meta S★ [COP]",
                value=6e7,
                step=1e6,
                format="%.0f",
                key="sstar_sin",
                help="Nivel objetivo promedio",
            )

            wn_sin = st.number_input(
                "Velocidad de ajuste ωₙ [1/año]",
                value=1.0,
                step=0.1,
                format="%.2f",
                key="wn_sin",
                help="Qué tan rápido corriges rumbo",
            )

        with col1b:
            zeta_sin = st.number_input(
                "Factor de amortiguación ζ",
                value=0.7,
                step=0.05,
                format="%.2f",
                key="zeta_sin",
                help="Controla oscilaciones",
            )

            F = st.number_input(
                "Amplitud estacional F [COP]",
                value=1e7,
                step=1e6,
                format="%.0f",
                help="Magnitud del pico estacional",
            )

            # Selectbox para frecuencia común
            freq_option = st.selectbox(
                "Frecuencia del ciclo",
                [
                    "Personalizada",
                    "Anual (2π)",
                    "Semestral (4π)",
                    "Trimestral (8π)",
                    "Mensual (24π)",
                ],
                index=1,
            )

            if freq_option == "Personalizada":
                Omega = st.number_input(
                    "Ω [rad/año]",
                    value=2 * np.pi,
                    step=0.1,
                    format="%.3f",
                    help="Frecuencia personalizada",
                )
            else:
                freq_map = {
                    "Anual (2π)": 2 * np.pi,
                    "Semestral (4π)": 4 * np.pi,
                    "Trimestral (8π)": 8 * np.pi,
                    "Mensual (24π)": 24 * np.pi,
                }
                Omega = freq_map[freq_option]
                st.info(f"Ω = {Omega:.3f} rad/año")

            tmax_sin = st.number_input(
                "Horizonte temporal [años]",
                value=6.0,
                step=0.5,
                format="%.1f",
                key="tmax_sin",
            )

        dt_sin = st.number_input(
            "Paso temporal dt", value=0.005, step=0.005, format="%.3f", key="dt_sin"
        )

    with col2:
        st.subheader("Modelo Matemático")
        st.latex(
            r"""
        S'' + 2\zeta \omega_n S' + \omega_n^2 S = \omega_n^2 S^* + F\sin(\Omega t)
        """
        )

        st.subheader("Respuesta de Frecuencia")
        if wn_sin > 0:
            # Calcular respuesta teórica
            denom = np.sqrt(
                (wn_sin**2 - Omega**2) ** 2 + (2 * zeta_sin * wn_sin * Omega) ** 2
            )
            Ahat_theory = F / denom if denom > 0 else float("inf")
            phi_theory = np.arctan2(2 * zeta_sin * wn_sin * Omega, wn_sin**2 - Omega**2)

            st.info(f"**Amplitud teórica:** {format_currency(Ahat_theory)}")
            st.info(f"**Fase teórica:** {phi_theory:.3f} rad")

            # Indicador de resonancia
            if abs(Omega - wn_sin) < 0.1:
                st.warning("⚠️ Cerca de resonancia (Ω ≈ ωₙ)")

    if st.button("🌊 Simular (LC2: Estacionalidad)", use_container_width=True):
        # Validar inputs
        errors = validate_inputs(tmax=tmax_sin, dt=dt_sin, wn=wn_sin, zeta=zeta_sin)
        if Omega < 0:
            errors.append("La frecuencia Ω debe ser no negativa")

        if errors:
            for error in errors:
                st.error(error)
        else:
            try:
                with st.spinner("Calculando respuesta estacional..."):
                    t = np.arange(0.0, tmax_sin + 1e-12, dt_sin, dtype=float)
                    S, Ahat, phi = lc2_sin_analytic(
                        t, S0_sin, V0_sin, Sstar_sin, wn_sin, zeta_sin, F, Omega
                    )

                # Crear gráfico
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

                # Gráfico principal
                ax1.plot(t, S, label="S(t)", linewidth=2, color="#1f77b4")
                ax1.axhline(
                    Sstar_sin,
                    linestyle="--",
                    label="S★ (meta promedio)",
                    color="red",
                    alpha=0.7,
                )
                ax1.set_ylabel("Saldo S(t) [COP]")
                ax1.set_title("LC2 — Respuesta con Estacionalidad")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M")
                )

                # Gráfico de la entrada
                forcing = F * np.sin(Omega * t)
                ax2.plot(
                    t, forcing, label="Entrada: F·sin(Ωt)", color="orange", alpha=0.7
                )
                ax2.set_xlabel("Tiempo [años]")
                ax2.set_ylabel("Fuerza [COP]")
                ax2.set_title("Señal de Entrada Estacional")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M")
                )

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

                # Mostrar métricas de la respuesta
                col_freq1, col_freq2, col_freq3 = st.columns(3)
                with col_freq1:
                    st.metric("Amplitud de oscilación", format_currency(Ahat))
                with col_freq2:
                    st.metric("Desfase", f"{phi:.3f} rad")
                with col_freq3:
                    phase_months = (phi / (2 * np.pi)) * 12 if Omega > 0 else 0
                    st.metric("Desfase temporal", f"{phase_months:.1f} meses")

                # Análisis de la respuesta
                st.subheader("📊 Análisis de la Respuesta")
                max_deviation = np.max(np.abs(S - Sstar_sin))
                avg_value = np.mean(S)

                col_anal1, col_anal2, col_anal3 = st.columns(3)
                with col_anal1:
                    st.metric("Desviación máxima", format_currency(max_deviation))
                with col_anal2:
                    st.metric("Valor promedio", format_currency(avg_value))
                with col_anal3:
                    deviation_pct = (
                        (max_deviation / abs(Sstar_sin)) * 100 if Sstar_sin != 0 else 0
                    )
                    st.metric("Desviación relativa", f"{deviation_pct:.1f}%")

                # Sección de descarga
                st.subheader("📥 Descargar Resultados")
                data_dict = {"t": t, "S": S, "Entrada": F * np.sin(Omega * t)}
                create_download_section(fig, data_dict, "lc2_estacional")

            except Exception as e:
                st.error(f"Error en la simulación: {str(e)}")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    📚 Herramienta educativa para modelado de ahorro con ecuaciones diferenciales<br>
    Desarrollada con Streamlit • Datos procesados localmente
    </div>
    """,
    unsafe_allow_html=True,
)
