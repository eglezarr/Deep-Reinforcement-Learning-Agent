"""
A3C (Asynchronous Advantage Actor-Critic) para el entorno de trading de la plantilla.

Una red con dos salidas sobre unas capas ocultas comunes: la política (actor) y el valor del estado (crítico).
Varios trabajadores, cada uno en su propio proceso, juegan episodios con una copia local de la red; cada t_max
pasos calculan el gradiente con el retorno a n pasos y lo aplican sobre la red global compartida sin esperar a
los demás. El proceso principal reparte los episodios (cada uno con una serie nueva) y, cada `cada` episodios,
detiene a los trabajadores y evalúa la red global en validación.

Las funciones de los trabajadores están en este módulo porque los procesos se crean con "spawn" y solo pueden
ejecutar código importable.
"""
import queue
import threading
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from entorno import SimpleTradingEnv
from protocolo import SEMILLA_TRAIN, SEMILLA_VAL, evaluar_lote, generar_serie, normalizar_obs
from dqn import REJILLA_Z, copia_pesos

N_ACCIONES = 3     # 0 = vender, 1 = mantener, 2 = comprar
LIMITE_V = 1e4     # si el |V| medio supera este valor, la red se ha descontrolado

# Varios entrenamientos pueden lanzarse desde hilos del mismo proceso (ajuste): la creación de la red global con su
# semilla se hace de uno en uno para que los pesos iniciales dependan solo de la repetición
_BLOQUEO_INICIALIZACION = threading.Lock()

# Valores por defecto. Los marcados con (*) se ajustan con Optuna; el resto son fijos.
CONFIG_A3C = {
    "lr": 3e-4,               # (*) tasa de aprendizaje de Adam (α)
    "gamma": 0.9,             # (*) factor de descuento
    "beta": 0.01,             # (*) peso de la entropía de la política (exploración)
    "t_max": 20,              # (*) pasos por actualización: retorno a n pasos
    "n_trabajadores": 4,      # procesos que entrenan en paralelo
    "n_ocultas": 64,          # neuronas en cada una de las 2 capas ocultas comunes
    "coef_critico": 0.5,      # peso de la pérdida del crítico en la pérdida total
    "max_norma_grad": 10.0,   # recorte de la norma del gradiente
}


class RedActorCritico(nn.Module):
    """Capas ocultas comunes (2 → 64 → 64) con dos salidas: logits de la política y V(s)."""

    def __init__(self, n_ocultas=64, n_entradas=2):
        super().__init__()
        self.comun = nn.Sequential(
            nn.Linear(n_entradas, n_ocultas), nn.ReLU(),
            nn.Linear(n_ocultas, n_ocultas), nn.ReLU(),
        )
        self.actor = nn.Linear(n_ocultas, N_ACCIONES)
        self.critico = nn.Linear(n_ocultas, 1)
        with torch.no_grad():
            self.actor.weight.mul_(0.01)  # política inicial casi uniforme: máxima exploración al empezar
            self.actor.bias.zero_()

    def forward(self, x):
        h = self.comun(x)
        return self.actor(h), self.critico(h).squeeze(-1)


def red_desde_pesos(pesos, n_ocultas=64, n_entradas=2):
    red = RedActorCritico(n_ocultas, n_entradas)
    red.load_state_dict(pesos)
    red.eval()
    return red


# --- Política voraz y lectura de la política en una rejilla de tendencias ---
def acciones_voraces(red, norm):
    """Acción más probable para un lote de observaciones (interfaz de evaluar_lote)."""
    def acciones(obs):
        x = torch.from_numpy(np.stack([normalizar_obs(o, norm["media"], norm["desv"]) for o in obs]))
        with torch.no_grad():
            logits, _ = red(x)
        return logits.argmax(dim=1).numpy()
    return acciones


def politica_en_rejilla(red, rejilla=REJILLA_Z):
    """
    Para cada valor de la tendencia normalizada de la rejilla: probabilidad de comprar sin activo y de vender
    con activo, y si la acción más probable cambia de posición (comprar sin activo / vender con activo).
    """
    sin_activo = torch.from_numpy(np.stack([rejilla, np.zeros_like(rejilla)], axis=1))
    con_activo = torch.from_numpy(np.stack([rejilla, np.ones_like(rejilla)], axis=1))
    with torch.no_grad():
        l0, _ = red(sin_activo)
        l1, _ = red(con_activo)
    return (torch.softmax(l0, dim=1)[:, 2].numpy(), torch.softmax(l1, dim=1)[:, 0].numpy(),
            (l0.argmax(dim=1) == 2).numpy(), (l1.argmax(dim=1) == 0).numpy())


# --- Trabajador ---
def _adam_compartido(red_global, estado_adam, lr):
    """Adam sobre la red global con sus momentos en memoria compartida: todos los trabajadores usan los mismos."""
    optimizador = torch.optim.Adam(red_global.parameters(), lr=lr, foreach=False)
    for p, (paso, m1, m2) in zip(red_global.parameters(), estado_adam):
        optimizador.state[p] = {"step": paso, "exp_avg": m1, "exp_avg_sq": m2}
    return optimizador


def _episodio(red, red_global, optimizador, cfg, prices, norm, rng, env=None, entrada=None):
    """
    Juega un episodio con la política estocástica y actualiza la red global cada t_max pasos.
    Por defecto, una serie sintética con el entorno de la plantilla y la tendencia normalizada (fases 3 a 5).
    env y entrada permiten otro entorno y otra transformación de la observación (fase 7, datos reales);
    la recompensa se divide siempre por norm["desv_retorno"].
    """
    if env is None:
        env = SimpleTradingEnv(prices)
    if entrada is None:
        def entrada(o):
            return normalizar_obs(o, norm["media"], norm["desv"])
    obs, info = env.reset()
    terminado = False
    perdidas, entropias, valores = [], [], []

    while not terminado:
        # 1. Copia local de la red global
        red.load_state_dict(red_global.state_dict())

        # 2. Hasta t_max pasos, eligiendo cada acción al azar según las probabilidades de la política
        estados, acciones, recompensas = [], [], []
        for _ in range(cfg["t_max"]):
            s = entrada(obs)
            with torch.no_grad():
                logits, _ = red(torch.from_numpy(s))
            p = torch.softmax(logits, dim=0).numpy().astype(np.float64)
            a = min(int(np.searchsorted(np.cumsum(p), rng.random() * p.sum())), N_ACCIONES - 1)
            obs, reward, terminado, truncated, info = env.step(a)
            estados.append(s)
            acciones.append(a)
            recompensas.append(reward / norm["desv_retorno"])  # misma escala de la recompensa que el DQN
            if terminado:
                break

        # 3. Retorno a n pasos, de atrás hacia delante: parte de V del último estado (0 si el episodio termina)
        with torch.no_grad():
            R = 0.0 if terminado else float(red(torch.from_numpy(entrada(obs)))[1])
        retornos = np.empty(len(recompensas), dtype=np.float32)
        for i in range(len(recompensas) - 1, -1, -1):
            R = recompensas[i] + cfg["gamma"] * R
            retornos[i] = R

        # 4. Pérdida: actor (log π de la acción por su ventaja), crítico (Huber) y entropía (exploración)
        logits, v = red(torch.from_numpy(np.stack(estados)))
        log_pi = torch.log_softmax(logits, dim=1)
        objetivo = torch.from_numpy(retornos)
        ventaja = (objetivo - v).detach()
        perdida_actor = -(log_pi[torch.arange(len(acciones)), torch.tensor(acciones)] * ventaja).mean()
        perdida_critico = F.huber_loss(v, objetivo)
        entropia = -(log_pi.exp() * log_pi).sum(dim=1).mean()
        perdida = perdida_actor + cfg["coef_critico"] * perdida_critico - cfg["beta"] * entropia

        # 5. Gradiente con la copia local, aplicado sobre la red global sin esperar a los demás trabajadores
        red.zero_grad()
        perdida.backward()
        nn.utils.clip_grad_norm_(red.parameters(), cfg["max_norma_grad"])
        for p_global, p_local in zip(red_global.parameters(), red.parameters()):
            p_global.grad = p_local.grad
        optimizador.step()

        perdidas.append(perdida_critico.item())
        entropias.append(entropia.item())
        valores.append(v.detach().abs().mean().item())
        if not np.isfinite(perdidas[-1]) or valores[-1] > LIMITE_V:
            motivo = "pérdida no finita" if not np.isfinite(perdidas[-1]) else f"|V| medio > {LIMITE_V:.0e}"
            return {"divergio": True, "motivo": motivo}

    return {"divergio": False, "ganancia": np.log(env.balance / env.initial_balance),
            "perdida": np.mean(perdidas), "entropia": np.mean(entropias), "v_medio": np.mean(valores)}


def _trabajador(rango, red_global, estado_adam, siguiente, limite, parar, bloqueo, cola,
                cfg, nombre_generador, norm, repeticion, n_episodios):
    """Toma episodios del contador común hasta agotarlos; se detiene en cada control (limite)."""
    try:
        torch.set_num_threads(1)
        rng = np.random.default_rng([repeticion, rango])  # semilla distinta en cada trabajador y repetición
        red = RedActorCritico(cfg["n_ocultas"])
        optimizador = _adam_compartido(red_global, estado_adam, cfg["lr"])
        cola.put(("listo", rango, None))

        while True:
            with bloqueo:
                if parar.value or siguiente.value >= n_episodios:
                    break
                episodio = None
                if siguiente.value < limite.value:
                    episodio = siguiente.value
                    siguiente.value += 1
            if episodio is None:  # control en curso: espera a que el proceso principal termine la evaluación
                time.sleep(0.001)
                continue
            prices = generar_serie(nombre_generador, SEMILLA_TRAIN + 1_000_000 * repeticion + episodio)
            cola.put(("episodio", episodio, _episodio(red, red_global, optimizador, cfg, prices, norm, rng)))
    except Exception:
        cola.put(("error", rango, traceback.format_exc()))


# --- Entrenamiento (proceso principal) ---
def entrenar_a3c(nombre_generador, norm, config=None, n_episodios=1000, repeticion=0,
                 series_val=None, cada=50, callback=None):
    """
    Entrena A3C con una serie nueva del generador en cada episodio (las mismas series que el DQN con la misma
    repetición). Cada `cada` episodios, si se pasan series de validación, los trabajadores se detienen, se evalúa
    la política voraz de la red global, se guarda una fila de la curva y se llama a callback(curva, ganancias).
    Si la pérdida deja de ser finita o el |V| medio se dispara, el entrenamiento se detiene (divergio=True).
    """
    cfg = {**CONFIG_A3C, **(config or {})}
    assert n_episodios <= SEMILLA_VAL - SEMILLA_TRAIN  # las series de entrenamiento no pisan las de validación

    torch.set_num_threads(1)
    with _BLOQUEO_INICIALIZACION:
        torch.manual_seed(repeticion)  # pesos iniciales de la red global
        red_global = RedActorCritico(cfg["n_ocultas"])
    red_global.share_memory()
    estado_adam = [(torch.zeros(()).share_memory_(), torch.zeros_like(p.data).share_memory_(),
                    torch.zeros_like(p.data).share_memory_()) for p in red_global.parameters()]

    ctx = mp.get_context("spawn")
    siguiente = ctx.Value("i", 0, lock=False)   # próximo episodio por repartir
    limite = ctx.Value("i", cada if series_val is not None else n_episodios, lock=False)  # primer episodio no permitido
    parar = ctx.Value("b", 0, lock=False)
    bloqueo, cola = ctx.Lock(), ctx.Queue()

    ganancias = np.full(n_episodios, np.nan)
    curva, bloque = [], {"perdida": [], "entropia": [], "v_medio": []}
    mejor = {"ret_log_medio": -np.inf, "pesos": None, "episodio": None}
    decisiones_previas = None
    terminados = listos = 0
    divergio, motivo = False, None
    tiempo_eval, t_listos = 0.0, None

    t0 = time.perf_counter()
    trabajadores = [ctx.Process(target=_trabajador, daemon=True,
                                args=(k, red_global, estado_adam, siguiente, limite, parar, bloqueo, cola,
                                      cfg, nombre_generador, norm, repeticion, n_episodios))
                    for k in range(cfg["n_trabajadores"])]
    try:
        for t in trabajadores:
            t.start()

        while terminados < n_episodios:
            try:
                tipo, origen, contenido = cola.get(timeout=1)
            except queue.Empty:
                if not any(t.is_alive() for t in trabajadores):
                    raise RuntimeError("Los trabajadores han terminado antes de completar los episodios")
                continue
            if tipo == "error":
                raise RuntimeError(f"Error en el trabajador {origen}:\n{contenido}")
            if tipo == "listo":
                listos += 1
                if listos == len(trabajadores):
                    t_listos = time.perf_counter()
                continue
            if contenido["divergio"]:
                divergio, motivo = True, f"{contenido['motivo']} (episodio {origen})"
                break

            ganancias[origen] = contenido["ganancia"]
            for k in bloque:
                bloque[k].append(contenido[k])
            terminados += 1

            # Control: todos los episodios anteriores a `limite` han terminado y nadie puede empezar otro
            if series_val is not None and terminados == limite.value:
                t_eval = time.perf_counter()
                res = evaluar_lote(acciones_voraces(red_global, norm), series_val)
                p_compra, p_venta, compra, venta = politica_en_rejilla(red_global)
                decisiones = np.concatenate([compra, venta])
                cambios = np.nan if decisiones_previas is None else 100 * np.mean(decisiones != decisiones_previas)
                decisiones_previas = decisiones

                curva.append({
                    "episodio": terminados, "beta": cfg["beta"],
                    "ganancia_entren": np.mean(ganancias[terminados - cada:terminados]),
                    **{k: v for k, v in res.items() if k != "finales"},
                    "perdida": np.mean(bloque["perdida"]), "entropia": np.mean(bloque["entropia"]),
                    "v_medio": np.mean(bloque["v_medio"]), "cambios_politica_%": cambios,
                    "tiempo_s": t_eval - t0 - tiempo_eval,
                })
                bloque = {k: [] for k in bloque}
                if res["ret_log_medio"] > mejor["ret_log_medio"]:
                    mejor = {"ret_log_medio": res["ret_log_medio"], "pesos": copia_pesos(red_global),
                             "episodio": terminados}
                if callback is not None:
                    callback(pd.DataFrame(curva), ganancias[:terminados])
                tiempo_eval += time.perf_counter() - t_eval
                with bloqueo:
                    limite.value = min(limite.value + cada, n_episodios)
    finally:
        parar.value = 1
        for t in trabajadores:
            t.join(timeout=10)
            if t.is_alive():
                t.terminate()

    return {
        "generador": nombre_generador, "config": cfg, "repeticion": repeticion, "n_episodios": n_episodios,
        "pesos": copia_pesos(red_global),                                    # modelo final (el que va a test)
        "mejor_pesos": mejor["pesos"], "mejor_episodio": mejor["episodio"],  # mejor en validación (diagnóstico)
        "curva": pd.DataFrame(curva), "ganancias": ganancias[:terminados],
        "tiempo": time.perf_counter() - t0 - tiempo_eval,                   # segundos, con el arranque y sin evaluaciones
        "tiempo_arranque": (t_listos - t0) if t_listos else np.nan,         # hasta que todos los trabajadores están listos
        "divergio": divergio, "motivo_divergencia": motivo,
    }
