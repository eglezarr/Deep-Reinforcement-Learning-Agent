"""
Q-Learning y Sarsa tabulares de la parte 1 (taller_trader.ipynb), copiados sin cambios, para comparar las tablas con las
redes en la fase 7.

La parte 1 no guardó sus tablas Q, pero su entrenamiento es determinista (semillas fijas): con el mismo código, las
mismas configuraciones y las mismas repeticiones se obtienen exactamente las mismas tablas. El notebook lo comprueba
con los resultados de test publicados en la parte 1.
"""
import functools
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from entorno import SimpleTradingEnv
from protocolo import (SEMILLA_TRAIN, SEMILLA_VAL, SEMILLA_DISC, N_DISC, evaluar, generar_conjunto, generar_serie,
                       tendencias_serie)

N_ACCIONES = 3
N_EP_FINAL = 5000
CADA_FINAL = 100
REPS_FINAL = range(10, 15)

# Configuración elegida en la fase 3 de la parte 1 (tabla_mejor)
MEJOR = {
    ("ciclo", "qlearning"): {"alpha": 0.01, "gamma": 0.5, "frac_exploracion": 0.5, "discretizacion": "cuantiles 20"},
    ("ciclo", "sarsa"): {"alpha": 0.01, "gamma": 0.5, "frac_exploracion": 0.8, "discretizacion": "cuantiles 10"},
    ("browniano", "qlearning"): {"alpha": 0.01, "gamma": 0.5, "frac_exploracion": 0.2, "discretizacion": "uniforme 10"},
    ("browniano", "sarsa"): {"alpha": 0.001, "gamma": 0.0, "frac_exploracion": 0.2, "discretizacion": "uniforme 10"},
    ("reversion", "qlearning"): {"alpha": 0.003, "gamma": 0.5, "frac_exploracion": 0.8, "discretizacion": "cuantiles 10"},
    ("reversion", "sarsa"): {"alpha": 0.001, "gamma": 0.5, "frac_exploracion": 0.2, "discretizacion": "cuantiles 10"},
    ("regimenes", "qlearning"): {"alpha": 0.001, "gamma": 0.99, "frac_exploracion": 0.2, "discretizacion": "cuantiles 10"},
    ("regimenes", "sarsa"): {"alpha": 0.001, "gamma": 0.0, "frac_exploracion": 0.2, "discretizacion": "cuantiles 20"},
}


# --- Discretización ---
def calcular_cortes(tendencias, n_intervalos=10):
    """Cortes interiores (n_intervalos - 1) en los cuantiles de la tendencia."""
    return np.quantile(tendencias, np.linspace(0, 1, n_intervalos + 1)[1:-1])


@functools.lru_cache(maxsize=None)
def tendencias_disc(nombre_generador):
    return np.concatenate([tendencias_serie(p) for p in generar_conjunto(nombre_generador, SEMILLA_DISC, N_DISC)])


def cortes_config(nombre_generador, discretizacion):
    """'cuantiles n': cortes en los cuantiles de la tendencia; 'uniforme n': n intervalos iguales en [-1, 1]."""
    tipo, n = discretizacion.split()
    n = int(n)
    if tipo == "uniforme":
        return np.linspace(-1, 1, n + 1)[1:-1]
    return calcular_cortes(tendencias_disc(nombre_generador), n)


def discretizar(obs, cortes):
    intervalo = int(np.searchsorted(cortes, obs[0], side="right"))
    posicion = int(obs[1])
    return posicion * (len(cortes) + 1) + intervalo


# --- Agente ---
def elegir_accion(Q, estado, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(N_ACCIONES))
    q = Q[estado]
    return int(rng.choice(np.flatnonzero(q == q.max())))


def epsilon_episodio(episodio, n_episodios, eps_inicial, eps_final, frac_exploracion):
    n_decaimiento = max(1, int(frac_exploracion * n_episodios))
    frac = min(1.0, episodio / n_decaimiento)
    return eps_inicial + frac * (eps_final - eps_inicial)


def acciones_efectivas(Q):
    """Acción voraz de cada estado: 0 = vende, 1 = mantiene la posición, 2 = compra."""
    a = Q.argmax(1)
    n = len(a) // 2
    ef = np.ones_like(a)
    ef[:n][a[:n] == 2] = 2
    ef[n:][a[n:] == 0] = 0
    return ef


def politica_voraz(Q, cortes):
    def politica(obs, env):
        return int(np.argmax(Q[discretizar(obs, cortes)]))
    return politica


def entrenar(metodo, nombre_generador, cortes, n_episodios=1000, alpha=0.05, gamma=0.9,
             eps_inicial=1.0, eps_final=0.01, frac_exploracion=0.8, repeticion=0, series_val=None, cada=100):
    assert metodo in ("qlearning", "sarsa")
    assert n_episodios <= SEMILLA_VAL - SEMILLA_TRAIN

    rng = np.random.default_rng(repeticion)
    Q = np.zeros((2 * (len(cortes) + 1), N_ACCIONES))
    recompensas = []
    curva = []
    politica_anterior = None
    tiempo_eval = 0.0
    t0 = time.perf_counter()

    for ep in range(n_episodios):
        prices = generar_serie(nombre_generador, SEMILLA_TRAIN + 1_000_000 * repeticion + ep)
        env = SimpleTradingEnv(prices)
        epsilon = epsilon_episodio(ep, n_episodios, eps_inicial, eps_final, frac_exploracion)

        obs, info = env.reset()
        s = discretizar(obs, cortes)
        a = elegir_accion(Q, s, epsilon, rng)
        terminated = False
        total = 0.0

        while not terminated:
            obs, reward, terminated, truncated, info = env.step(a)
            s2 = discretizar(obs, cortes)
            a2 = elegir_accion(Q, s2, epsilon, rng)

            if terminated:
                objetivo = reward
            elif metodo == "qlearning":
                objetivo = reward + gamma * Q[s2].max()
            else:
                objetivo = reward + gamma * Q[s2, a2]

            Q[s, a] += alpha * (objetivo - Q[s, a])
            s, a = s2, a2
            total += reward

        recompensas.append(total)

        if series_val is not None and (ep + 1) % cada == 0:
            t_eval = time.perf_counter()
            res = evaluar(politica_voraz(Q, cortes), series_val)
            politica = acciones_efectivas(Q)
            cambios = np.nan if politica_anterior is None else int((politica != politica_anterior).sum())
            politica_anterior = politica
            curva.append({"episodio": ep + 1, "epsilon": epsilon, "cambios_politica": cambios,
                          **{k: v for k, v in res.items() if k != "finales"}})
            tiempo_eval += time.perf_counter() - t_eval

    return {"Q": Q, "recompensas": np.array(recompensas), "curva": pd.DataFrame(curva),
            "tiempo": time.perf_counter() - t0 - tiempo_eval}


def entrenar_final(nombre_generador, metodo, repeticion, ruta):
    """Entrenamiento final de la parte 1 (5 000 episodios, validación cada 100); si ya está en disco, lo carga."""
    ruta = Path(ruta)
    if ruta.exists():
        with open(ruta, "rb") as f:
            return pickle.load(f)
    config = MEJOR[(nombre_generador, metodo)]
    r = entrenar(metodo, nombre_generador, cortes_config(nombre_generador, config["discretizacion"]),
                 n_episodios=N_EP_FINAL, alpha=config["alpha"], gamma=config["gamma"],
                 frac_exploracion=config["frac_exploracion"], repeticion=repeticion,
                 series_val=generar_conjunto(nombre_generador, SEMILLA_VAL, 50), cada=CADA_FINAL)
    with open(ruta, "wb") as f:
        pickle.dump(r, f)
    return r
