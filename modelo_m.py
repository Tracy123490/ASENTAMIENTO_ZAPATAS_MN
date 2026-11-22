"""Modelo interactivo para estimar el asentamiento de una zapata."""

import math
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

if __name__ == "__main__" and get_script_run_ctx() is None:
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(Path(__file__).resolve()),
            ]
        )
    )

st.set_page_config(page_title="Asentamiento de zapata", page_icon="📐", layout="wide")
st.title("Modelo interactivo de asentamiento")
st.caption(
    "Estimación del asentamiento mediante integración numérica compuesta "
    "(difusión de esfuerzos 2:1)."
)

with st.sidebar:
    st.header("Datos de la zapata")
    ancho = st.number_input("Ancho B (m)", min_value=0.01, value=2.0, step=0.1)
    largo = st.number_input("Largo L (m)", min_value=0.01, value=2.0, step=0.1)
    carga = st.number_input("Carga vertical P (kN)", min_value=0.01, value=400.0, step=10.0)
    tolerancia = st.number_input(
        "Tolerancia de error (%)", min_value=0.0001, value=0.1, step=0.1, format="%.4f"
    )
    niveles = st.slider("Niveles de refinamiento", min_value=2, max_value=7, value=5)

st.subheader("Estratos del suelo")
st.write(
    "Ingresa el espesor y el módulo de deformación de cada estrato. "
    "Puedes agregar o quitar filas con los controles de la tabla."
)
estratos_default = pd.DataFrame(
    {
        "Espesor (m)": [0.8, 1.2, 1.2, 0.8],
        "Es (kPa)": [8000.0, 12000.0, 18000.0, 25000.0],
    }
)
estratos = st.data_editor(
    estratos_default,
    num_rows="dynamic",
    hide_index=True,
    width="stretch",
    column_config={
        "Espesor (m)": st.column_config.NumberColumn(
            "Espesor (m)", min_value=0.001, step=0.1, format="%.3f"
        ),
        "Es (kPa)": st.column_config.NumberColumn(
            "Es (kPa)", min_value=0.001, step=100.0, format="%.2f"
        ),
    },
    key="estratos",
)

if estratos.empty:
    st.error("Agrega al menos un estrato.")
    st.stop()

espesores = pd.to_numeric(estratos["Espesor (m)"], errors="coerce").to_numpy(dtype=float)
modulos = pd.to_numeric(estratos["Es (kPa)"], errors="coerce").to_numpy(dtype=float)
if (
    not np.all(np.isfinite(espesores))
    or not np.all(np.isfinite(modulos))
    or np.any(espesores <= 0)
    or np.any(modulos <= 0)
):
    st.error("Cada estrato debe tener un espesor y un Es numéricos, ambos mayores que cero.")
    st.stop()

limites = np.concatenate(([0.0], np.cumsum(espesores)))
numero_estratos = len(espesores)
profundidad = float(limites[-1])
presion = carga / (ancho * largo)


def esfuerzo(z):
    """Esfuerzo vertical según la difusión 2:1, en kPa."""
    return carga / ((ancho + z) * (largo + z))


def integrando(z, indice_estrato):
    return esfuerzo(z) / modulos[indice_estrato]


def trapecio(funcion, a, b, n):
    h = (b - a) / n
    valores = np.asarray([funcion(a + i * h) for i in range(n + 1)])
    return h * (valores[0] / 2 + valores[1:-1].sum() + valores[-1] / 2)


def simpson(funcion, a, b, n):
    if n % 2:
        raise ValueError("Simpson 1/3 requiere un número par de subintervalos.")
    h = (b - a) / n
    valores = np.asarray([funcion(a + i * h) for i in range(n + 1)])
    return h / 3 * (
        valores[0]
        + valores[-1]
        + 4 * valores[1:-1:2].sum()
        + 2 * valores[2:-1:2].sum()
    )


def primitiva(z):
    diferencia = ancho - largo
    if abs(diferencia) <= 1e-10 * max(ancho, largo):
        return -1.0 / (ancho + z)
    return -math.log1p(diferencia / (largo + z)) / diferencia


def referencia_analitica():
    total = 0.0
    for k in range(numero_estratos):
        total += (
            carga
            / modulos[k]
            * (primitiva(limites[k + 1]) - primitiva(limites[k]))
        )
    return total


def integral_por_malla_unica(regla, n):
    def funcion(z):
        indice = min(
            int(np.searchsorted(limites[1:], z, side="right")),
            numero_estratos - 1,
        )
        return integrando(z, indice)

    return regla(funcion, 0.0, profundidad, n)


def integral_por_estratos(regla, n_por_estrato):
    return sum(
        regla(
            lambda z, k=k: integrando(z, k),
            limites[k],
            limites[k + 1],
            n_por_estrato,
        )
        for k in range(numero_estratos)
    )


referencia = referencia_analitica()
n_por_estrato_lista = [2 * (2**i) for i in range(niveles)]
resultados = []

for n_por_estrato in n_por_estrato_lista:
    n_total = n_por_estrato * numero_estratos
    calculos = {
        "Trapecio (malla única)": integral_por_malla_unica(trapecio, n_total),
        "Simpson 1/3 (malla única)": integral_por_malla_unica(simpson, n_total),
        "Trapecio (por estratos)": integral_por_estratos(trapecio, n_por_estrato),
        "Simpson 1/3 (por estratos)": integral_por_estratos(simpson, n_por_estrato),
    }
    for metodo, asentamiento in calculos.items():
        resultados.append(
            {
                "Método": metodo,
                "n total": n_total,
                "Asentamiento (mm)": asentamiento * 1000,
                "Error relativo (%)": abs(asentamiento - referencia)
                / abs(referencia)
                * 100,
            }
        )

tabla = pd.DataFrame(resultados)
tabla["Variación (%)"] = tabla.groupby("Método")["Asentamiento (mm)"].transform(
    lambda serie: serie.pct_change().abs() * 100
)

col1, col2, col3 = st.columns(3)
col1.metric("Presión de contacto", f"{presion:.2f} kPa")
col2.metric("Profundidad analizada", f"{profundidad:.2f} m")
col3.metric("Referencia analítica", f"{referencia * 1000:.4f} mm")

st.subheader("Resultados numéricos")
st.dataframe(
    tabla.style.format(
        {
            "Asentamiento (mm)": "{:.6f}",
            "Error relativo (%)": "{:.6f}",
            "Variación (%)": "{:.6f}",
        },
        na_rep="—",
    ),
    width="stretch",
    hide_index=True,
)

st.subheader("Gráficas de convergencia")
fig, ejes = plt.subplots(1, 2, figsize=(13, 4.5))
colores = {
    "Trapecio (malla única)": "#2563eb",
    "Simpson 1/3 (malla única)": "#f97316",
    "Trapecio (por estratos)": "#16a34a",
    "Simpson 1/3 (por estratos)": "#dc2626",
}

for metodo, datos in tabla.groupby("Método", sort=False):
    ejes[0].plot(
        datos["n total"],
        datos["Asentamiento (mm)"],
        marker="o",
        label=metodo,
        color=colores[metodo],
    )
    ejes[1].plot(
        datos["n total"],
        datos["Error relativo (%)"].clip(lower=1e-14),
        marker="o",
        label=metodo,
        color=colores[metodo],
    )

ejes[0].axhline(
    referencia * 1000, color="black", linestyle=":", label="Referencia analítica"
)
ejes[0].set_xscale("log", base=2)
ejes[0].set(
    xlabel="Subintervalos totales (n)",
    ylabel="Asentamiento (mm)",
    title="Asentamiento calculado",
)
ejes[1].axhline(
    tolerancia, color="red", linestyle=":", label=f"Tolerancia ({tolerancia:g} %)"
)
ejes[1].set_xscale("log", base=2)
ejes[1].set_yscale("log")
ejes[1].set(
    xlabel="Subintervalos totales (n)",
    ylabel="Error relativo (%)",
    title="Error respecto a la referencia",
)
for eje in ejes:
    eje.grid(True, alpha=0.3)
    eje.legend(fontsize=8)
fig.tight_layout()
st.pyplot(fig, width="stretch")
plt.close(fig)

st.subheader("Interpretación automática")
ultimos_resultados = tabla.groupby("Método", sort=False).tail(1)
mejor_metodo = ultimos_resultados.sort_values("Error relativo (%)").iloc[0]["Método"]
mejor_fila = ultimos_resultados[ultimos_resultados["Método"] == mejor_metodo].iloc[0]

if mejor_fila["Error relativo (%)"] <= tolerancia:
    st.success(
        f"Con el mayor refinamiento, **{mejor_metodo}** queda dentro de la "
        f"tolerancia: error relativo de {mejor_fila['Error relativo (%)']:.4g} %."
    )
else:
    st.warning(
        f"El menor error observado corresponde a **{mejor_metodo}**, pero con "
        f"el refinamiento actual es {mejor_fila['Error relativo (%)']:.4g} %, "
        f"mayor que la tolerancia de {tolerancia:g} %. Aumenta los niveles de refinamiento."
    )

st.markdown(
    f"""
- **Gráfica de asentamiento:** al aumentar *n*, las curvas deberían acercarse
  a la línea punteada de referencia ({referencia * 1000:.4f} mm). Una curva
  que se estabiliza indica que el resultado cambia poco al refinar la malla.
- **Gráfica de error:** valores menores significan mayor precisión; la línea
  roja marca la tolerancia ingresada ({tolerancia:g} %). Si el error desciende
  y queda debajo de esa línea, el método satisface el criterio de error.
- **Comparación de enfoques:** la malla única puede atravesar límites entre
  estratos, donde cambia Es. La integración por estratos respeta esos límites
  y suele producir una convergencia más regular.
- **Resultado de este caso:** el método más cercano a la referencia en el
  refinamiento máximo es **{mejor_metodo}**, con {mejor_fila['Asentamiento (mm)']:.6f} mm.
"""
)

with st.expander("Supuestos y alcance del modelo"):
    st.write(
        "Se usa la difusión de esfuerzos 2:1 y la relación de deformación "
        "ε(z) = Δσ(z)/Es(z), integradas desde la base de la zapata hasta el "
        "último estrato. La referencia analítica integra exactamente cada "
        "estrato con Es constante. Es una herramienta didáctica de estimación; "
        "no sustituye la evaluación geotécnica ni el diseño profesional."
    )