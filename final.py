"""
Fase 5: entrenamientos largos con las configuraciones elegidas, episodios hasta converger, tiempo por episodio y test.

Mismo protocolo que la parte 1 (5 semillas nuevas, 5 000 episodios, validación cada 100), para comparar directamente
con Q-Learning y Sarsa. Las funciones de entrenamiento están en un módulo porque se ejecutan en procesos aparte.
"""
from pathlib import Path

import numpy as np

from protocolo import evaluar, evaluar_lote
from dqn import entrenar_dqn, guardar, cargar, politica_dqn, red_desde_pesos as red_dqn_desde_pesos
from a3c import entrenar_a3c, acciones_voraces, red_desde_pesos as red_a3c_desde_pesos
from ajuste import N_EPISODIOS_AJUSTE, datos_mercado

N_EPISODIOS_FINAL = 5000
CADA_FINAL = 100              # episodios entre controles en validación
REPS_FINAL = (10, 11, 12, 13, 14)
VENTANA_SUAVIZADO = 5         # controles de la media móvil del criterio de convergencia
FRACCION_MEJORA = 0.9         # fracción de la mejora total que define la convergencia
N_EPISODIOS_TIEMPO = 500      # episodios de la medición del tiempo por episodio


def config_final_dqn(config):
    """
    La exploración se mantiene en episodios absolutos: ε llega a 0.01 en el mismo episodio que en el ajuste.
    Con la fracción original, en 5 000 episodios la exploración duraría 5 veces más que la ajustada.
    """
    return {**config, "frac_exploracion": config["frac_exploracion"] * N_EPISODIOS_AJUSTE / N_EPISODIOS_FINAL}


def entrenar_final(algoritmo, nombre_generador, config, repeticion, ruta):
    """Entrenamiento largo con validación cada CADA_FINAL episodios; si ya está en disco, lo carga."""
    ruta = Path(ruta)
    if ruta.exists():
        return cargar(ruta)
    series_val, norm = datos_mercado(nombre_generador)
    entrenar = entrenar_dqn if algoritmo == "dqn" else entrenar_a3c
    r = entrenar(nombre_generador, norm, config, n_episodios=N_EPISODIOS_FINAL, repeticion=repeticion,
                 series_val=series_val, cada=CADA_FINAL)
    guardar(r, ruta)
    return r


def episodio_convergencia(curva, ventana=VENTANA_SUAVIZADO, fraccion=FRACCION_MEJORA):
    """
    Primer episodio a partir del cual la curva de validación suavizada (media móvil de `ventana` controles) ya no
    baja del `fraccion` de su mejora total, del primer al último punto suavizado (criterio de la parte 1).
    Si la curva no mejora, no hay nada que esperar: converge en el primer control.
    """
    suavizada = curva["ret_log_medio"].rolling(ventana, min_periods=1).mean().to_numpy()
    episodios = curva["episodio"].to_numpy()
    mejora = suavizada[-1] - suavizada[0]
    if mejora <= 0:
        return int(episodios[0])
    por_debajo = np.flatnonzero(suavizada < suavizada[0] + fraccion * mejora)
    return int(episodios[por_debajo[-1] + 1]) if len(por_debajo) else int(episodios[0])


def medir_tiempo(algoritmo, nombre_generador, config, n_episodios=N_EPISODIOS_TIEMPO, repeticion=99):
    """
    Milisegundos por episodio de entrenamiento, con un solo entrenamiento en marcha y sin evaluaciones.
    DQN: ε en su valor final desde el principio, el coste de los episodios ya explorados, que son la mayoría.
    A3C: sin el arranque de los procesos, que se devuelve aparte.
    """
    _, norm = datos_mercado(nombre_generador)
    if algoritmo == "dqn":
        r = entrenar_dqn(nombre_generador, norm, {**config, "frac_exploracion": 0.0}, n_episodios=n_episodios,
                         repeticion=repeticion)
        return {"ms_por_episodio": 1000 * r["tiempo"] / n_episodios, "arranque_s": 0.0}
    r = entrenar_a3c(nombre_generador, norm, config, n_episodios=n_episodios, repeticion=repeticion)
    return {"ms_por_episodio": 1000 * (r["tiempo"] - r["tiempo_arranque"]) / n_episodios,
            "arranque_s": r["tiempo_arranque"]}


def evaluar_test(algoritmo, resultado, nombre_generador, series):
    """Política voraz del modelo final sobre las series de test, con la misma evaluación que en validación."""
    _, norm = datos_mercado(nombre_generador)
    if algoritmo == "dqn":
        red = red_dqn_desde_pesos(resultado["pesos"], resultado["config"]["n_ocultas"])
        return evaluar(politica_dqn(red, norm), series)
    red = red_a3c_desde_pesos(resultado["pesos"], resultado["config"]["n_ocultas"])
    return evaluar_lote(acciones_voraces(red, norm), series)
