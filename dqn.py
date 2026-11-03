"""
DQN para el entorno de trading de la plantilla.

Una red neuronal estima Q(s, a) a partir del estado continuo (tendencia normalizada, posición).
Aprende de transiciones guardadas en memoria (experience replay), calcula los objetivos con una
red objetivo que se copia cada cierto número de actualizaciones, usa la pérdida de Huber y explora
con ε-greedy con el mismo decaimiento lineal de la parte 1.
"""
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from entorno import SimpleTradingEnv
from protocolo import SEMILLA_TRAIN, SEMILLA_VAL, evaluar, generar_serie, normalizar_obs

N_ACCIONES = 3     # 0 = vender, 1 = mantener, 2 = comprar
LIMITE_Q = 1e4     # si el |Q| medio supera este valor, la red se ha descontrolado

# Valores por defecto. Los marcados con (*) se ajustan con Optuna; el resto son fijos.
CONFIG_DQN = {
    "lr": 1e-3,                   # (*) tasa de aprendizaje de Adam (α)
    "gamma": 0.5,                 # (*) factor de descuento
    "frac_exploracion": 0.5,      # (*) fracción de los episodios en la que ε baja de eps_inicial a eps_final
    "eps_inicial": 1.0,
    "eps_final": 0.01,
    "n_ocultas": 64,              # neuronas en cada una de las 2 capas ocultas
    "tam_lote": 64,               # transiciones por actualización
    "tam_memoria": 100_000,       # capacidad de la memoria (~200 episodios)
    "inicio_aprendizaje": 1_000,  # transiciones guardadas antes de la primera actualización
    "cada_pasos": 4,              # una actualización de la red cada 4 pasos del entorno
    "sinc_objetivo": 500,         # actualizaciones entre dos copias de la red en la red objetivo
    "max_norma_grad": 10.0,       # recorte de la norma del gradiente
}


class RedQ(nn.Module):
    """Perceptrón con 2 capas ocultas: entra el estado (2 valores) y sale Q(s, a) para las 3 acciones."""

    def __init__(self, n_ocultas=64, n_entradas=2):
        super().__init__()
        self.capas = nn.Sequential(
            nn.Linear(n_entradas, n_ocultas), nn.ReLU(),
            nn.Linear(n_ocultas, n_ocultas), nn.ReLU(),
            nn.Linear(n_ocultas, N_ACCIONES),  # salida lineal: es un problema de regresión
        )

    def forward(self, x):
        return self.capas(x)


class Memoria:
    """Memoria circular de transiciones (s, a, r, s', fin); al llenarse sobrescribe las más antiguas."""

    def __init__(self, capacidad, n_entradas=2):
        self.s = np.zeros((capacidad, n_entradas), dtype=np.float32)
        self.a = np.zeros(capacidad, dtype=np.int64)
        self.r = np.zeros(capacidad, dtype=np.float32)
        self.s2 = np.zeros((capacidad, n_entradas), dtype=np.float32)
        self.fin = np.zeros(capacidad, dtype=np.float32)
        self.capacidad, self.n, self.i = capacidad, 0, 0

    def guardar(self, s, a, r, s2, fin):
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i], self.fin[self.i] = s, a, r, s2, fin
        self.i = (self.i + 1) % self.capacidad
        self.n = min(self.n + 1, self.capacidad)

    def muestra(self, tam, rng):
        """Lote de transiciones elegidas al azar entre todas las guardadas."""
        idx = rng.integers(0, self.n, size=tam)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]), torch.from_numpy(self.r[idx]),
                torch.from_numpy(self.s2[idx]), torch.from_numpy(self.fin[idx]))


def epsilon_episodio(episodio, n_episodios, eps_inicial, eps_final, frac_exploracion):
    """ε baja linealmente durante frac_exploracion de los episodios y después se queda en eps_final (parte 1)."""
    n_decaimiento = max(1, int(frac_exploracion * n_episodios))
    frac = min(1.0, episodio / n_decaimiento)
    return eps_inicial + frac * (eps_final - eps_inicial)


def politica_dqn(red, norm):
    """Política voraz (sin exploración), con la interfaz politica(obs, env) del protocolo."""
    def politica(obs, env):
        with torch.no_grad():
            q = red(torch.from_numpy(normalizar_obs(obs, norm["media"], norm["desv"])))
        return int(torch.argmax(q))
    return politica


# Tendencia normalizada en la que se lee la política aprendida (cubre el 99.7% de las observaciones)
REJILLA_Z = np.linspace(-3, 3, 121, dtype=np.float32)


def ventajas_en_rejilla(red, rejilla=REJILLA_Z):
    """
    Para cada valor de la tendencia normalizada de la rejilla:
    - sin activo: Q(comprar) - max(Q(vender), Q(mantener)); si es > 0, compra
    - con activo: Q(vender) - max(Q(mantener), Q(comprar)); si es > 0, vende
    (sin activo, vender equivale a mantener; con activo, comprar equivale a mantener)
    """
    sin_activo = torch.from_numpy(np.stack([rejilla, np.zeros_like(rejilla)], axis=1))
    con_activo = torch.from_numpy(np.stack([rejilla, np.ones_like(rejilla)], axis=1))
    with torch.no_grad():
        q0, q1 = red(sin_activo).numpy(), red(con_activo).numpy()
    return q0[:, 2] - q0[:, :2].max(axis=1), q1[:, 0] - q1[:, 1:].max(axis=1)


def copia_pesos(red):
    return {k: v.detach().clone() for k, v in red.state_dict().items()}


def red_desde_pesos(pesos, n_ocultas=64):
    red = RedQ(n_ocultas)
    red.load_state_dict(pesos)
    red.eval()
    return red


def entrenar_dqn(nombre_generador, norm, config=None, n_episodios=1000, repeticion=0,
                 series_val=None, cada=50, callback=None):
    """
    Entrena un DQN con una serie nueva del generador en cada episodio.

    norm: parámetros del generador (media y desviación de la tendencia, desviación del retorno por paso).
    Cada `cada` episodios, si se pasan series de validación, evalúa la política voraz, guarda una fila de la
    curva de aprendizaje y llama a callback(curva, ganancias) para seguir el entrenamiento en vivo.
    Si la pérdida deja de ser finita o el |Q| medio se dispara, el entrenamiento se detiene (divergio=True).
    """
    cfg = {**CONFIG_DQN, **(config or {})}
    assert n_episodios <= SEMILLA_VAL - SEMILLA_TRAIN  # las series de entrenamiento no pisan las de validación

    torch.set_num_threads(1)  # red pequeña: un hilo es lo más rápido y hace el resultado reproducible
    torch.manual_seed(repeticion)
    rng = np.random.default_rng(repeticion)

    red, red_objetivo = RedQ(cfg["n_ocultas"]), RedQ(cfg["n_ocultas"])
    red_objetivo.load_state_dict(red.state_dict())
    optimizador = torch.optim.Adam(red.parameters(), lr=cfg["lr"])
    perdida_huber = nn.HuberLoss()
    memoria = Memoria(cfg["tam_memoria"])

    ganancias, curva = [], []                 # rentabilidad log. de cada episodio de entrenamiento; filas de seguimiento
    perdidas_periodo, q_periodo = [], []      # se vacían en cada evaluación
    mejor = {"ret_log_medio": -np.inf, "pesos": None, "episodio": None}
    signos_previos = None
    n_pasos = n_actualizaciones = 0
    divergio = False
    tiempo_eval = 0.0
    t0 = time.perf_counter()

    for ep in range(n_episodios):
        prices = generar_serie(nombre_generador, SEMILLA_TRAIN + 1_000_000 * repeticion + ep)
        env = SimpleTradingEnv(prices)
        epsilon = epsilon_episodio(ep, n_episodios, cfg["eps_inicial"], cfg["eps_final"], cfg["frac_exploracion"])

        obs, info = env.reset()
        s = normalizar_obs(obs, norm["media"], norm["desv"])
        terminated = False

        while not terminated:
            # 1. Acción ε-greedy
            if rng.random() < epsilon:
                a = int(rng.integers(N_ACCIONES))
            else:
                with torch.no_grad():
                    a = int(torch.argmax(red(torch.from_numpy(s))))

            # 2. Paso del entorno y transición a la memoria. La recompensa se expresa en unidades de la
            #    volatilidad típica del mercado: es un cambio de escala que no altera la política óptima
            obs, reward, terminated, truncated, info = env.step(a)
            s2 = normalizar_obs(obs, norm["media"], norm["desv"])
            memoria.guardar(s, a, reward / norm["desv_retorno"], s2, float(terminated))
            s = s2
            n_pasos += 1

            # 3. Actualización con un lote al azar: objetivo r + γ max_a' Q_objetivo(s', a') (solo r si termina)
            if memoria.n >= cfg["inicio_aprendizaje"] and n_pasos % cfg["cada_pasos"] == 0:
                sb, ab, rb, s2b, finb = memoria.muestra(cfg["tam_lote"], rng)
                with torch.no_grad():
                    objetivo = rb + cfg["gamma"] * (1 - finb) * red_objetivo(s2b).max(dim=1).values
                q = red(sb).gather(1, ab.unsqueeze(1)).squeeze(1)   # Q de la acción realizada
                perdida = perdida_huber(q, objetivo)

                optimizador.zero_grad()
                perdida.backward()
                nn.utils.clip_grad_norm_(red.parameters(), cfg["max_norma_grad"])
                optimizador.step()

                n_actualizaciones += 1
                if n_actualizaciones % cfg["sinc_objetivo"] == 0:
                    red_objetivo.load_state_dict(red.state_dict())

                perdidas_periodo.append(perdida.item())
                q_periodo.append(q.detach().abs().mean().item())
                if not np.isfinite(perdidas_periodo[-1]) or q_periodo[-1] > LIMITE_Q:
                    divergio = True
                    break

        if divergio:
            break
        ganancias.append(np.log(env.balance / env.initial_balance))

        # 4. Seguimiento: política voraz en validación cada `cada` episodios
        if series_val is not None and (ep + 1) % cada == 0:
            t_eval = time.perf_counter()
            tiempo_entrenando = t_eval - t0 - tiempo_eval
            red.eval()
            res = evaluar(politica_dqn(red, norm), series_val)
            ventaja_compra, ventaja_venta = ventajas_en_rejilla(red)
            red.train()

            # % de puntos de la rejilla en que cambia la acción efectiva respecto a la evaluación anterior
            signos = np.concatenate([ventaja_compra > 0, ventaja_venta > 0])
            cambios = np.nan if signos_previos is None else 100 * np.mean(signos != signos_previos)
            signos_previos = signos

            curva.append({
                "episodio": ep + 1, "epsilon": epsilon,
                "ganancia_entren": np.mean(ganancias[-cada:]),
                **{k: v for k, v in res.items() if k != "finales"},
                "perdida": np.mean(perdidas_periodo) if perdidas_periodo else np.nan,
                "q_medio": np.mean(q_periodo) if q_periodo else np.nan,
                "cambios_politica_%": cambios,
                "tiempo_s": tiempo_entrenando,
            })
            perdidas_periodo, q_periodo = [], []

            if res["ret_log_medio"] > mejor["ret_log_medio"]:
                mejor = {"ret_log_medio": res["ret_log_medio"], "pesos": copia_pesos(red), "episodio": ep + 1}

            tiempo_eval += time.perf_counter() - t_eval
            if callback is not None:
                callback(pd.DataFrame(curva), np.array(ganancias))

    return {
        "generador": nombre_generador, "config": cfg, "repeticion": repeticion, "n_episodios": n_episodios,
        "pesos": copia_pesos(red),                      # modelo final (el que va a test)
        "mejor_pesos": mejor["pesos"], "mejor_episodio": mejor["episodio"],  # mejor en validación (diagnóstico)
        "curva": pd.DataFrame(curva), "ganancias": np.array(ganancias),
        "tiempo": time.perf_counter() - t0 - tiempo_eval,  # segundos de entrenamiento, sin las evaluaciones
        "divergio": divergio,
    }


def guardar(resultado, ruta):
    with open(ruta, "wb") as f:
        pickle.dump(resultado, f)


def cargar(ruta):
    with open(ruta, "rb") as f:
        return pickle.load(f)
