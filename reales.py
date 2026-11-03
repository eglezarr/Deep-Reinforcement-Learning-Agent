"""
Fase 6: DQN con series de precios reales.

Mismo diseño que la fase 8 de la parte 1 (taller_trader.ipynb), para comparar directamente con Q-Learning y Sarsa:
- Mercados: EUR/USD, S&P 500 con dividendos y Energía (10 petroleras y el Brent), copia local en datos_reales/.
- Bloques de 252 sesiones sin solapar; ~70% más antiguo para entrenar, 15% validación, 15% test, cortes por fechas.
- Walk-forward: cada bloque evaluado se juega con un agente entrenado con todas las sesiones anteriores, salvo un
  embargo de 50. Los episodios de entrenamiento son ventanas de 252 sesiones con inicio (y serie) al azar.
- Entorno del profesor sin cambios (comisión 0.1%); solo cambia la observación que recibe la red.

El DQN es el de dqn.py (misma red, memoria, red objetivo, Huber, ε-greedy). Cambia el origen de los episodios
y la entrada de la red, que puede tener más variables: sin discretizar, el estado puede crecer sin que se
disparen los estados, que es lo que no podía hacer la tabla.

Las funciones están en un módulo porque se ejecutan en procesos aparte (spawn).
"""
import functools
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from entorno import SimpleTradingEnv
from protocolo import politica_aleatoria, politica_buy_and_hold
from dqn import CONFIG_DQN, LIMITE_Q, N_ACCIONES, RedQ, Memoria, epsilon_episodio, copia_pesos

CARPETA_DATOS = Path(__file__).resolve().parent / "datos_reales"
FECHA_FIN = "2026-10-01"
ARCHIVOS = {"EUR/USD": "eurusd.csv", "S&P 500": "sp500tr.csv", "Energía": "energia.csv"}

L_BLOQUE = 252     # sesiones por bloque (≈ 1 año)
HISTORIA = 49      # sesiones previas que necesita la variable más larga (tendencia de 50 precios)
EMBARGO = 50       # sesiones sin usar entre el final del entrenamiento y el bloque evaluado
FRAC_VAL = FRAC_TEST = 0.15
SEMILLA_TEST = 20_000  # semillas del agente aleatorio (las de la parte 1)
LIMITE_Z = 10.0    # recorte de las variables normalizadas (crisis de 2008 y 2020 en la volatilidad)

# Variables de entrada de la red (además de la posición). Todas usan solo precios hasta la sesión actual.
VARIANTES = {
    "profesor": ["tend6"],
    "ampliado": ["tend6", "tend50", "r0", "r1", "r2", "r3", "r4", "vol20"],
}

N_EPISODIOS = 1000
CONFIG_REAL = {"frac_exploracion": 0.5}
LRS = (1e-5, 1e-4)
GAMMAS = (0.5, 0.9, 0.99)
REPS = (0, 1, 2)


# --- Datos y bloques (como en la parte 1) ---
@functools.lru_cache(maxsize=None)
def datos():
    """Precios de cierre por mercado: mismo final y solo las fechas con dato en todas sus series."""
    salida = {}
    for m, archivo in ARCHIVOS.items():
        d = pd.read_csv(CARPETA_DATOS / archivo, index_col=0, parse_dates=True)
        salida[m] = d[d.index < FECHA_FIN].dropna()
    return salida


def precios(mercado):
    return datos()[mercado].to_numpy()


def bloques(n_sesiones):
    """(inicio, fin) de los bloques consecutivos de L_BLOQUE sesiones, alineados con la última sesión."""
    n = (n_sesiones - 1 - HISTORIA) // L_BLOQUE
    ultima = n_sesiones - 1
    return [(ultima - (n - k) * L_BLOQUE, ultima - (n - k - 1) * L_BLOQUE) for k in range(n)]


@functools.lru_cache(maxsize=None)
def particion():
    """Bloques de validación y de test de cada mercado."""
    val, test = {}, {}
    for m, d in datos().items():
        b = bloques(len(d))
        n_test, n_val = round(FRAC_TEST * len(b)), round(FRAC_VAL * len(b))
        test[m] = b[-n_test:]
        val[m] = b[-n_test - n_val:-n_test]
    return val, test


# --- Variables de estado ---
def tendencia(serie, n_precios):
    """Fórmula de _get_obs del entorno: último precio frente a la media de los n últimos."""
    media = pd.Series(serie).rolling(n_precios).mean().to_numpy()
    return np.clip((serie - media) / media, -1.0, 1.0)


def variables_serie(serie):
    """Todas las variables candidatas en cada sesión de una serie (columnas en el orden de COLUMNAS)."""
    r = np.concatenate([[np.nan], np.diff(np.log(serie))])
    retardos = [np.concatenate([[np.nan] * k, r[:len(r) - k]]) for k in range(5)]  # r0 = rentab. log. de hoy
    vol = pd.Series(r).rolling(20).std().to_numpy()
    return np.column_stack([tendencia(serie, 6), tendencia(serie, 50), *retardos, vol])


COLUMNAS = ["tend6", "tend50", "r0", "r1", "r2", "r3", "r4", "vol20"]


@functools.lru_cache(maxsize=None)
def variables(mercado):
    """Matriz series x sesiones x variables del mercado."""
    p = precios(mercado)
    return np.stack([variables_serie(p[:, j]) for j in range(p.shape[1])])


def normalizacion(mercado, variante, fin):
    """
    Media y desviación de cada variable, y desviación del retorno por paso (escala de la recompensa),
    con las sesiones de entrenamiento (antes de fin) de todas las series del mercado.
    """
    cols = [COLUMNAS.index(c) for c in VARIANTES[variante]]
    x = variables(mercado)[:, HISTORIA:fin, :][:, :, cols].reshape(-1, len(cols))
    p = precios(mercado)[HISTORIA:fin]
    retornos = (p[1:] / p[:-1] - 1).ravel()
    return {"cols": cols, "media": x.mean(axis=0), "desv": x.std(axis=0), "desv_retorno": float(retornos.std())}


class EntornoRealDRL(SimpleTradingEnv):
    """
    Entorno del profesor; solo cambia la observación: [variables normalizadas de la sesión, posición].
    prices y x empiezan HISTORIA sesiones antes de la primera decisión.
    """

    def __init__(self, prices, x):
        super().__init__(prices)
        self.x = x

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = HISTORIA
        return self._get_obs(), {}

    def _get_obs(self):
        return np.append(self.x[self.current_step], self.position).astype(np.float32)


def entorno(mercado, j, inicio, fin, norm):
    """Entorno para operar en la serie j de la sesión inicio a la sesión fin."""
    serie = precios(mercado)[inicio - HISTORIA: fin + 1, j]
    x = variables(mercado)[j, inicio - HISTORIA: fin + 1][:, norm["cols"]]
    x = np.clip((x - norm["media"]) / norm["desv"], -LIMITE_Z, LIMITE_Z)
    return EntornoRealDRL(serie, np.nan_to_num(x).astype(np.float32))


# --- Entrenamiento ---
def entrenar_dqn_real(mercado, variante, fin, config=None, n_episodios=N_EPISODIOS, repeticion=0):
    """
    Mismo bucle que dqn.entrenar_dqn; cada episodio es una ventana de L_BLOQUE sesiones con inicio y serie al azar,
    entera antes de fin. Devuelve el modelo final, sin validación durante el entrenamiento.
    """
    cfg = {**CONFIG_DQN, **CONFIG_REAL, **(config or {})}
    torch.set_num_threads(1)
    torch.manual_seed(repeticion)
    rng = np.random.default_rng(repeticion)

    norm = normalizacion(mercado, variante, fin)
    n_entradas = len(norm["cols"]) + 1
    red, red_objetivo = RedQ(cfg["n_ocultas"], n_entradas), RedQ(cfg["n_ocultas"], n_entradas)
    red_objetivo.load_state_dict(red.state_dict())
    optimizador = torch.optim.Adam(red.parameters(), lr=cfg["lr"])
    perdida_huber = nn.HuberLoss()
    memoria = Memoria(cfg["tam_memoria"], n_entradas)

    n_series = precios(mercado).shape[1]
    ganancias = []
    n_pasos = n_actualizaciones = 0
    divergio = False
    t0 = time.perf_counter()

    for ep in range(n_episodios):
        j = int(rng.integers(n_series))
        inicio = int(rng.integers(HISTORIA, fin - L_BLOQUE))
        env = entorno(mercado, j, inicio, inicio + L_BLOQUE, norm)
        epsilon = epsilon_episodio(ep, n_episodios, cfg["eps_inicial"], cfg["eps_final"], cfg["frac_exploracion"])

        s, info = env.reset()
        terminated = False
        while not terminated:
            if rng.random() < epsilon:
                a = int(rng.integers(N_ACCIONES))
            else:
                with torch.no_grad():
                    a = int(torch.argmax(red(torch.from_numpy(s))))
            s2, reward, terminated, truncated, info = env.step(a)
            memoria.guardar(s, a, reward / norm["desv_retorno"], s2, float(terminated))
            s = s2
            n_pasos += 1

            if memoria.n >= cfg["inicio_aprendizaje"] and n_pasos % cfg["cada_pasos"] == 0:
                sb, ab, rb, s2b, finb = memoria.muestra(cfg["tam_lote"], rng)
                with torch.no_grad():
                    objetivo = rb + cfg["gamma"] * (1 - finb) * red_objetivo(s2b).max(dim=1).values
                q = red(sb).gather(1, ab.unsqueeze(1)).squeeze(1)
                perdida = perdida_huber(q, objetivo)
                optimizador.zero_grad()
                perdida.backward()
                nn.utils.clip_grad_norm_(red.parameters(), cfg["max_norma_grad"])
                optimizador.step()
                n_actualizaciones += 1
                if n_actualizaciones % cfg["sinc_objetivo"] == 0:
                    red_objetivo.load_state_dict(red.state_dict())
                if not np.isfinite(perdida.item()) or q.detach().abs().mean().item() > LIMITE_Q:
                    divergio = True
                    break
        if divergio:
            break
        ganancias.append(np.log(env.balance / env.initial_balance))

    return {"pesos": copia_pesos(red), "norm": norm, "config": cfg, "n_entradas": n_entradas,
            "ganancias": np.array(ganancias), "tiempo": time.perf_counter() - t0, "divergio": divergio}


def red_desde_resultado(r):
    red = RedQ(r["config"]["n_ocultas"], r["n_entradas"])
    red.load_state_dict(r["pesos"])
    red.eval()
    return red


def politica_red(red):
    def politica(obs, env):
        with torch.no_grad():
            return int(torch.argmax(red(torch.from_numpy(obs))))
    return politica


# --- Evaluación ---
def jugar(env, politica, semilla=None):
    obs, info = env.reset()
    if semilla is not None:
        env.action_space.seed(semilla)
    balances, operaciones, invertido = [env.balance], 0, 0
    terminated = False
    while not terminated:
        action = politica(obs, env)
        operaciones += (action == 2 and env.position == 0) or (action == 0 and env.position == 1)
        obs, reward, terminated, truncated, info = env.step(action)
        invertido += env.position
        balances.append(env.balance)
    return {"balances": np.array(balances), "operaciones": int(operaciones),
            "pct_invertido": 100 * invertido / (len(balances) - 1)}


def evaluar_bloque(politica, mercado, bloque, norm, semilla=None):
    """Juega el bloque en cada serie del mercado, con el buy & hold de referencia en el mismo entorno."""
    salida = []
    for j in range(precios(mercado).shape[1]):
        r = jugar(entorno(mercado, j, *bloque, norm), politica, semilla)
        r["bh_final"] = jugar(entorno(mercado, j, *bloque, norm), politica_buy_and_hold)["balances"][-1]
        salida.append(r)
    return salida


def referencias(conjunto):
    """Aleatorio y buy & hold en los bloques de 'val' o 'test' (como en la parte 1)."""
    bloques_mercado = particion()[0 if conjunto == "val" else 1]
    ref = {}
    for m in ARCHIVOS:
        norm = normalizacion(m, "profesor", bloques_mercado[m][0][0])  # la observación no influye en ellos
        ref[(m, "Aleatorio")] = [[evaluar_bloque(politica_aleatoria, m, b, norm, semilla=SEMILLA_TEST + k)
                                  for k, b in enumerate(bloques_mercado[m])]]
        ref[(m, "Buy & hold")] = [[evaluar_bloque(politica_buy_and_hold, m, b, norm) for b in bloques_mercado[m]]]
    return ref


# --- Walk-forward ---
def nombre_tarea(conjunto, mercado, variante, lr, gamma, rep, bloque):
    m = {"EUR/USD": "eurusd", "S&P 500": "sp500", "Energía": "energia"}[mercado]
    return f"{conjunto}_{m}_{variante}_lr{lr:g}_g{gamma:g}_r{rep}_b{bloque[0]}.pkl"


def tarea(conjunto, mercado, variante, lr, gamma, rep, bloque, carpeta, n_episodios=N_EPISODIOS):
    """Entrena con lo anterior al bloque (menos el embargo) y lo juega. Si ya está en disco, lo carga."""
    ruta = Path(carpeta) / nombre_tarea(conjunto, mercado, variante, lr, gamma, rep, bloque)
    if ruta.exists():
        with open(ruta, "rb") as f:
            return pickle.load(f)
    r = entrenar_dqn_real(mercado, variante, bloque[0] - EMBARGO, {"lr": lr, "gamma": gamma},
                          n_episodios=n_episodios, repeticion=rep)
    red = red_desde_resultado(r)
    salida = {"series": evaluar_bloque(politica_red(red), mercado, bloque, r["norm"]),
              "tiempo": r["tiempo"], "divergio": r["divergio"], "ganancias": r["ganancias"]}
    if variante == "profesor":
        salida["pesos"], salida["norm"] = r["pesos"], r["norm"]  # para leer la política aprendida
    with open(ruta, "wb") as f:
        pickle.dump(salida, f)
    return salida


def tareas(conjunto, combinaciones, reps=REPS):
    """combinaciones: lista de (mercado, variante, lr, gamma) -> lista de argumentos de tarea (sin la carpeta)."""
    bloques_mercado = particion()[0 if conjunto == "val" else 1]
    return [(conjunto, m, v, lr, g, rep, b) for (m, v, lr, g) in combinaciones for rep in reps
            for b in bloques_mercado[m]]


def combinaciones_validacion():
    """Rejilla de validación: γ y α para cada mercado y variante (γ por fuera, como en la parte 1)."""
    return [(m, v, lr, g) for m in ARCHIVOS for v in VARIANTES for g in GAMMAS for lr in LRS]


def elegir(val):
    """
    Para cada (mercado, variante), la combinación con mayor rentabilidad anual media en validación.
    Con empate (p. ej., ninguna opera) se queda la primera: el γ y el α más pequeños.
    """
    elegidas = {}
    for (m, v, lr, g), res in val.items():
        rent = resumir(res)["rent_anual_%"]
        if (m, v) not in elegidas or rent > elegidas[(m, v)]["rent"] + 1e-12:
            elegidas[(m, v)] = {"lr": lr, "gamma": g, "rent": rent}
    return elegidas


def agrupar(lista_tareas, resultados):
    """{(mercado, variante, lr, gamma): repeticiones -> bloques en orden -> series}, como en la parte 1."""
    salida = {}
    for (conjunto, m, v, lr, g, rep, b), r in zip(lista_tareas, resultados):
        salida.setdefault((m, v, lr, g), {}).setdefault(rep, []).append(r["series"])
    return {c: list(reps.values()) for c, reps in salida.items()}


# --- Métricas (las de la parte 1) ---
def resumir(res):
    """Métricas de cada repetición (sobre todos los bloques y series) y su media."""
    filas = []
    for rep in res:
        finales = np.array([[s["balances"][-1] for s in b] for b in rep])  # bloques x series
        bh = np.array([[s["bh_final"] for s in b] for b in rep])
        sharpe, caida = [], []
        for j in range(finales.shape[1]):
            r = np.concatenate([np.diff(np.log(b[j]["balances"])) for b in rep])
            sharpe.append(r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0)
            capital = np.exp(np.concatenate([[0.0], np.cumsum(r)]))
            caida.append(100 * np.max(1 - capital / np.maximum.accumulate(capital)))
        filas.append({
            "rent_anual_%": 100 * (np.exp(np.log(finales / 1000).mean()) - 1),
            "pct_gana_bh": 100 * np.mean(finales > bh),
            "sharpe": np.mean(sharpe),
            "caida_max_%": np.mean(caida),
            "operaciones": np.mean([[s["operaciones"] for s in b] for b in rep]),
            "pct_invertido": np.mean([[s["pct_invertido"] for s in b] for b in rep]),
        })
    df = pd.DataFrame(filas)
    return {**df.mean().to_dict(), "rent_anual_std_reps": df["rent_anual_%"].std(ddof=0)}


def curva_capital(res):
    """Capital encadenando los bloques: media geométrica sobre repeticiones y series."""
    log_capital = [np.concatenate([[0.0], np.cumsum(np.concatenate([np.diff(np.log(b[j]["balances"])) for b in rep]))])
                   for rep in res for j in range(len(rep[0]))]
    return 1000 * np.exp(np.mean(log_capital, axis=0))
