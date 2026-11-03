"""
Protocolo experimental reutilizado de la parte 1 del taller.

Contiene los conjuntos de series con semillas disjuntas, la ejecución de un episodio,
los baselines y las métricas de evaluación. Es el mismo código de la parte 1, de modo que
los resultados de DQN y A3C se pueden comparar directamente con los de Q-Learning y Sarsa.
También incluye la normalización de la tendencia que reciben las redes.
"""
import numpy as np

from entorno import (SimpleTradingEnv, generar_precios, generar_precios_browniano,
                     generar_precios_reversion_media, generar_precios_regimenes)

# --- Configuración general (idéntica a la parte 1) ---
N_STEPS = 500

GENERADORES = {
    "ciclo": generar_precios,
    "browniano": generar_precios_browniano,
    "reversion": generar_precios_reversion_media,
    "regimenes": generar_precios_regimenes,
}

# Rangos de semillas disjuntos para cada conjunto
SEMILLA_TRAIN = 0
SEMILLA_VAL = 10_000
SEMILLA_TEST = 20_000
SEMILLA_DISC = 30_000

N_VAL = 50
N_TEST = 200
N_DISC = 100


# --- Generación de series reproducibles ---
def generar_serie(nombre_generador, semilla, n_steps=N_STEPS):
    """
    Genera una serie de precios reproducible con una de las funciones propuestas.
    Las funciones usan el estado global de np.random, así que se fija la semilla justo antes de llamarlas.
    """
    np.random.seed(semilla)
    return GENERADORES[nombre_generador](n_steps=n_steps)


def generar_conjunto(nombre_generador, semilla_inicial, n_series):
    return [generar_serie(nombre_generador, semilla_inicial + i) for i in range(n_series)]


# --- Ejecución de un episodio con una política cualquiera ---
def jugar_episodio(prices, politica, semilla=None):
    """
    Juega un episodio completo. `politica(obs, env)` devuelve la acción.
    Devuelve el balance tras cada paso, las acciones elegidas y la posición antes de cada acción.
    """
    env = SimpleTradingEnv(prices)
    obs, info = env.reset()
    if semilla is not None:
        env.action_space.seed(semilla)
    terminated = False

    balances = [env.balance]
    actions_taken = []
    positions = []

    while not terminated:
        action = politica(obs, env)
        positions.append(env.position)
        actions_taken.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        balances.append(env.balance)

    return np.array(balances), np.array(actions_taken), np.array(positions)


# --- Baselines ---
def politica_aleatoria(obs, env):
    return env.action_space.sample()


def politica_buy_and_hold(obs, env):
    return 2 if env.position == 0 else 1


# --- Evaluación sobre un conjunto de series ---
def evaluar(politica, lista_series, semilla=None):
    """
    Métricas de una política sobre una lista de series (las mismas que en la parte 1):
    balance final medio, desviación y mediana; rentabilidad logarítmica media, mean(log(balance_final / 1000));
    % de series en que supera a buy & hold; operaciones por episodio y % de pasos con el activo.
    """
    episodios = []
    for i, prices in enumerate(lista_series):
        semilla_ep = None if semilla is None else semilla + i
        balances, acciones, posiciones = jugar_episodio(prices, politica, semilla_ep)
        episodios.append(resumen_episodio(prices, balances[-1], acciones, posiciones))
    return metricas(episodios)


def evaluar_lote(acciones_lote, lista_series):
    """
    Mismas métricas que evaluar, para una política determinista, jugando todas las series a la vez:
    en cada paso, acciones_lote(obs) recibe las observaciones de las series en juego (matriz n x 2)
    y devuelve sus acciones. Permite evaluar una red con una sola pasada por paso para todas las series.
    """
    entornos = [SimpleTradingEnv(prices) for prices in lista_series]
    obs = np.stack([env.reset()[0] for env in entornos])
    en_juego = np.ones(len(entornos), dtype=bool)
    acciones = [[] for _ in entornos]
    posiciones = [[] for _ in entornos]

    while en_juego.any():
        indices = np.flatnonzero(en_juego)
        elegidas = acciones_lote(obs[indices])
        for i, a in zip(indices, elegidas):
            env = entornos[i]
            posiciones[i].append(env.position)
            acciones[i].append(int(a))
            obs[i], reward, terminated, truncated, info = env.step(int(a))
            en_juego[i] = not terminated

    return metricas([resumen_episodio(prices, env.balance, np.array(acc), np.array(pos))
                     for prices, env, acc, pos in zip(lista_series, entornos, acciones, posiciones)])


def resumen_episodio(prices, balance_final, acciones, posiciones):
    """Balance final, si supera a buy & hold, operaciones ejecutadas y fracción de pasos con el activo."""
    # Buy & hold de referencia, misma fórmula que en la plantilla
    bh_shares = (1000.0 * 0.999) / prices[5]
    bh_final = bh_shares * prices[-1]

    # Compras y ventas que realmente se ejecutan
    compra = (acciones == 2) & (posiciones == 0)
    venta = (acciones == 0) & (posiciones == 1)

    # Posición mantenida durante cada paso (después de ejecutar la acción)
    pos_paso = posiciones.copy()
    pos_paso[compra] = 1
    pos_paso[venta] = 0

    return balance_final, balance_final > bh_final, compra.sum() + venta.sum(), pos_paso.mean()


def metricas(episodios):
    """Métricas de un conjunto de episodios a partir de sus resúmenes (resumen_episodio)."""
    finales, gana_bh, n_ops, invertido = (np.array(x) for x in zip(*episodios))
    return {
        "balance_medio": finales.mean(),
        "balance_std": finales.std(),
        "balance_mediana": np.median(finales),
        "ret_log_medio": np.mean(np.log(finales / 1000.0)),
        "pct_gana_bh": 100 * np.mean(gana_bh),
        "operaciones": np.mean(n_ops),
        "pct_invertido": 100 * np.mean(invertido),
        "finales": finales,
    }


# --- Tendencia y su normalización ---
def tendencias_serie(prices):
    """Tendencia observada en cada paso de un episodio, tal como la calcula el entorno."""
    env = SimpleTradingEnv(prices)
    obs, info = env.reset()
    terminated = False
    tendencias = [obs[0]]
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(1)  # la tendencia no depende de las acciones
        tendencias.append(obs[0])
    return np.array(tendencias)


def parametros_normalizacion(tendencias):
    """Media y desviación típica de la tendencia, calculadas con las series de discretización."""
    return {"media": float(np.mean(tendencias)), "desv": float(np.std(tendencias))}


def normalizar_obs(obs, media, desv):
    """Entrada de la red: tendencia normalizada y posición sin cambios."""
    return np.array([(obs[0] - media) / desv, obs[1]], dtype=np.float32)


def desviacion_retorno(series):
    """
    Desviación típica del retorno por paso, (p_t - p_t-1) / p_t-1, en los pasos que juega el agente
    (desde el paso 5). Es la recompensa de un paso con el activo, sin comisión.
    """
    retornos = np.concatenate([p[6:] / p[5:-1] - 1 for p in series])
    return float(np.std(retornos))


def normalizacion(nombre_generador):
    """Todos los parámetros de normalización del generador, calculados con sus series de discretización."""
    series = generar_conjunto(nombre_generador, SEMILLA_DISC, N_DISC)
    tendencias = np.concatenate([tendencias_serie(p) for p in series])
    return {**parametros_normalizacion(tendencias), "desv_retorno": desviacion_retorno(series)}
