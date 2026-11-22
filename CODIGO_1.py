# =====================================================================
# Asentamiento de una zapata cuadrada sobre suelo fino
# Integración numérica: regla del Trapecio y Simpson 1/3 (compuestas)
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. DATOS DE ENTRADA ----------
def pedir(texto, defecto):
    """Pide un dato por teclado; si se pulsa Enter, conserva el valor por defecto."""
    r = input(f"{texto} [{defecto}]: ").strip()
    return float(r) if r else defecto

def leer_datos(interactivo=False):
    d = {"B": 2.0, "L": 2.0,                         # dimensiones de la zapata (m)
         "P": 400.0,                                 # carga vertical (kN)
         "TOL": 0.1,                                 # tolerancia relativa (%)
         "limites": [0.0, 0.8, 2.0, 3.2, 4.0],       # límites de estratos (m)
         "Es": [8000.0, 12000.0, 18000.0, 25000.0],  # módulo de deformación (kPa)
         "n_lista": [4, 8, 16, 32, 64]}              # subdivisiones a evaluar
    if interactivo:
        for clave, texto in (("B", "Ancho B (m)"), ("L", "Largo L (m)"),
                             ("P", "Carga P (kN)"), ("TOL", "Tolerancia (%)")):
            d[clave] = pedir(texto, d[clave])
    return d

D = leer_datos(interactivo=False)        # usar True para ingresar datos por teclado
B, L, P, TOL = D["B"], D["L"], D["P"], D["TOL"]
lim, Es_k, N_LISTA = D["limites"], D["Es"], D["n_lista"]
H = lim[-1]                              # profundidad de análisis (m)
q = P / (B * L)                          # presión media de contacto (kPa)

# ---------- 2. MODELO MATEMÁTICO ----------
def delta_sigma(z):
    """Incremento de esfuerzo vertical (kPa) con difusión 2:1."""
    return q * B * L / ((B + z) * (L + z))

def modulo_Es(z):
    """Es(z) en kPa; en un límite entre estratos se toma el estrato inferior."""
    for k in range(len(Es_k)):
        if z < lim[k + 1] - 1e-12:
            return Es_k[k]
    return Es_k[-1]

def f(z):
    """Deformación unitaria e(z) = delta_sigma(z) / Es(z), adimensional."""
    return delta_sigma(z) / modulo_Es(z)

def f_estrato(k):
    """Integrando del estrato k, con Es constante."""
    return lambda z: delta_sigma(z) / Es_k[k]

def solucion_analitica():
    """Valor de referencia (m): integral exacta estrato por estrato."""
    prim = (lambda z: -1 / (B + z)) if abs(B - L) < 1e-12 else \
           (lambda z: np.log((L + z) / (B + z)) / (L - B))
    return [q * B * L / Es_k[k] * (prim(lim[k + 1]) - prim(lim[k]))
            for k in range(len(Es_k))]

# ---------- 3. MÉTODOS NUMÉRICOS ----------
def trapecio(fun, a, b, n):
    """Regla del Trapecio compuesta con n subintervalos."""
    h = (b - a) / n
    y = np.array([fun(a + i * h) for i in range(n + 1)])
    return h / 2 * (y[0] + 2 * y[1:-1].sum() + y[-1])

def simpson(fun, a, b, n):
    """Regla de Simpson 1/3 compuesta (n debe ser par)."""
    if n % 2 != 0:
        raise ValueError("Simpson 1/3 requiere un número par de subintervalos")
    h = (b - a) / n
    y = np.array([fun(a + i * h) for i in range(n + 1)])
    return h / 3 * (y[0] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum() + y[-1])

def malla_unica(regla, n):
    """Enfoque A: una sola malla de n subintervalos sobre [0, H]."""
    return regla(f, 0.0, H, n)

def por_estratos(regla, n):
    """Enfoque B: se integra cada estrato con n/4 subintervalos y se suma."""
    m = n // len(Es_k)
    return sum(regla(f_estrato(k), lim[k], lim[k + 1], m) for k in range(len(Es_k)))

# ---------- 4. ERRORES Y CONVERGENCIA ----------
def error_abs(S, S_ref):
    return abs(S - S_ref)

def error_rel(S, S_ref):
    return abs(S - S_ref) / abs(S_ref) * 100

def variacion(S_n, S_previo):
    """Cambio relativo (%) entre dos refinamientos consecutivos."""
    return abs(S_n - S_previo) / abs(S_n) * 100

def orden_observado(e_grueso, e_fino):
    """Orden de convergencia p cuando n se duplica: p = log2(e_grueso / e_fino)."""
    return np.log2(e_grueso / e_fino)
# ---------- 5. EJECUCIÓN Y RESULTADOS ----------
S_ref = sum(solucion_analitica())                      # m
print(f"q = {q:.2f} kPa | H = {H:.2f} m | S_ref = {S_ref*1000:.4f} mm\n")

casos = {"Trapecio (A)": (malla_unica, trapecio), "Simpson 1/3 (A)": (malla_unica, simpson),
         "Trapecio (B)": (por_estratos, trapecio), "Simpson 1/3 (B)": (por_estratos, simpson)}
res = {}
for nombre, (esquema, regla) in casos.items():
    res[nombre] = {}
    print(nombre)
    print(f"{'n':>4} {'S (mm)':>10} {"Ea (mm)":>10} {'Er (%)':>10} {'Var (%)':>10}")
    previo = None
    for n in N_LISTA:
        if regla is simpson and esquema is por_estratos and (n // 4) % 2 != 0:
            print(f"{n:>4}   Simpson no aplicable (n/4 impar)")
            continue
        S = esquema(regla, n)
        ea, er = error_abs(S, S_ref) * 1000, error_rel(S, S_ref)
        var = variacion(S, previo) if previo is not None else float("nan")
        ok = "  <- cumple TOL" if (previo is not None and var < TOL and er < TOL) else ""
        print(f"{n:>4} {S*1000:>10.4f} {ea:>10.5f} {er:>10.4f} {var:>10.4f}{ok}")
        res[nombre][n] = (S * 1000, er)
        previo = S
    print()

# ---------- 6. GRÁFICOS ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
estilos = {"Trapecio (A)": "o--", "Simpson 1/3 (A)": "s--",
           "Trapecio (B)": "o-", "Simpson 1/3 (B)": "s-"}
for nombre, datos in res.items():
    ns = list(datos.keys())
    ax[0].plot(ns, [datos[n][0] for n in ns], estilos[nombre], label=nombre)
    ax[1].plot(ns, [datos[n][1] for n in ns], estilos[nombre], label=nombre)
ax[0].axhline(S_ref * 1000, color="k", ls=":", label="Referencia")
ax[1].axhline(TOL, color="r", ls=":", label=f"Tolerancia {TOL} %")
ax[0].set(xscale="log", xlabel="n (subintervalos)", ylabel="Asentamiento S (mm)",
          title="Asentamiento calculado")
ax[1].set(xscale="log", yscale="log", xlabel="n (subintervalos)",
          ylabel="Error relativo (%)", title="Error respecto a la referencia")
for a in ax:
    a.set_xticks(N_LISTA); a.set_xticklabels(N_LISTA)
    a.xaxis.set_minor_formatter(plt.NullFormatter())
    a.grid(alpha=0.3); a.legend()
plt.tight_layout()
plt.savefig("convergencia.png", dpi=200)
plt.show()