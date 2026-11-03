"""
Búsqueda de hiperparámetros con Optuna.

Cada estudio (uno por mercado y algoritmo) se guarda en un fichero de registro en disco. Varios procesos o
hilos pueden añadir pruebas al mismo estudio a la vez, y si la búsqueda se interrumpe, al relanzarla continúa
donde se quedó: solo cuentan las pruebas terminadas.
"""
import functools
import warnings
from pathlib import Path

import optuna

from protocolo import SEMILLA_VAL, N_VAL, generar_conjunto, normalizacion
from dqn import entrenar_dqn, guardar, cargar
from a3c import entrenar_a3c

N_EPISODIOS_AJUSTE = 1000   # episodios de cada prueba
CADA_AJUSTE = 50            # episodios entre controles en validación
N_CONTROLES = 5             # controles que puntúan: los 5 últimos (episodios 800 a 1000)
PUNTUACION_DIVERGENCIA = -1.0  # peor que cualquier resultado de los cuatro mercados


@functools.lru_cache(maxsize=None)
def datos_mercado(nombre_generador):
    """Series de validación y normalización del generador; se calculan una sola vez en cada proceso."""
    return generar_conjunto(nombre_generador, SEMILLA_VAL, N_VAL), normalizacion(nombre_generador)


def puntuacion(curva, n=N_CONTROLES):
    """Rentabilidad logarítmica media en validación en los n últimos controles."""
    return float(curva["ret_log_medio"].iloc[-n:].mean()) if len(curva) else PUNTUACION_DIVERGENCIA


def puntuacion_resultado(resultado):
    """Puntuación de un entrenamiento: la peor posible si divergió, aunque llegara a evaluarse antes."""
    return PUNTUACION_DIVERGENCIA if resultado["divergio"] else puntuacion(resultado["curva"])


def config_dqn(trial):
    """Espacio de búsqueda: α en escala log., γ a través del horizonte 1/(1 - γ), fracción de exploración."""
    horizonte = trial.suggest_float("horizonte", 2, 100, log=True)
    return {
        "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "gamma": 1 - 1 / horizonte,
        "frac_exploracion": trial.suggest_float("frac_exploracion", 0.1, 0.8),
    }


def objetivo_dqn(trial, nombre_generador):
    series_val, norm = datos_mercado(nombre_generador)
    r = entrenar_dqn(nombre_generador, norm, config_dqn(trial), n_episodios=N_EPISODIOS_AJUSTE,
                     repeticion=0, series_val=series_val, cada=CADA_AJUSTE)
    curva = r["curva"]
    # Información adicional de la prueba, para el análisis posterior
    trial.set_user_attr("curva", curva["ret_log_medio"].round(4).tolist())
    trial.set_user_attr("operaciones_final", float(curva["operaciones"].iloc[-1]) if len(curva) else None)
    trial.set_user_attr("tiempo_s", r["tiempo"])
    trial.set_user_attr("divergio", r["divergio"])
    return puntuacion(curva)


def almacen(ruta):
    """Almacén de Optuna en un fichero de registro, apto para varios procesos a la vez."""
    return optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(ruta)))


def terminadas(estudio):
    return estudio.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))


def limpiar_interrumpidas(estudio):
    """Marca como fallidas las pruebas que quedaron a medias si se interrumpió una ejecución anterior."""
    for t in estudio.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        estudio.tell(t.number, state=optuna.trial.TrialState.FAIL)


def prueba_dqn(nombre_generador, ruta_almacen, semilla):
    """Una prueba (un entrenamiento): Optuna propone la configuración y registra su puntuación en el almacén."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    # TPE multivariante: modela las interacciones entre hiperparámetros. constant_liar evita que dos
    # procesos que trabajan a la vez en el mismo estudio propongan la misma combinación.
    muestreador = optuna.samplers.TPESampler(multivariate=True, constant_liar=True, seed=semilla)
    estudio = optuna.load_study(study_name=f"dqn_{nombre_generador}", storage=almacen(ruta_almacen),
                                sampler=muestreador)
    estudio.optimize(lambda t: objetivo_dqn(t, nombre_generador), n_trials=1)


def confirmar_dqn(nombre_generador, config, repeticion, ruta):
    """Entrena una configuración con otra semilla (otras series de entrenamiento); si ya está en disco, la carga."""
    ruta = Path(ruta)
    if ruta.exists():
        return cargar(ruta)
    series_val, norm = datos_mercado(nombre_generador)
    r = entrenar_dqn(nombre_generador, norm, config, n_episodios=N_EPISODIOS_AJUSTE, repeticion=repeticion,
                     series_val=series_val, cada=CADA_AJUSTE)
    guardar(r, ruta)
    return r


# --- A3C ---
def config_a3c(trial):
    """Espacio de búsqueda de A3C: α y β en escala log., γ a través del horizonte 1/(1 - γ), t_max (5 o 20)."""
    horizonte = trial.suggest_float("horizonte", 2, 100, log=True)
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "gamma": 1 - 1 / horizonte,
        "beta": trial.suggest_float("beta", 1e-4, 1e-1, log=True),
        "t_max": trial.suggest_categorical("t_max", [5, 20]),
    }


def objetivo_a3c(trial, nombre_generador):
    series_val, norm = datos_mercado(nombre_generador)
    r = entrenar_a3c(nombre_generador, norm, config_a3c(trial), n_episodios=N_EPISODIOS_AJUSTE,
                     repeticion=0, series_val=series_val, cada=CADA_AJUSTE)
    curva = r["curva"]
    # Información adicional de la prueba, para el análisis posterior. Si diverge antes del primer control
    # (cada 50 episodios), la curva está vacía
    trial.set_user_attr("curva", curva["ret_log_medio"].round(4).tolist() if len(curva) else [])
    trial.set_user_attr("operaciones_final", float(curva["operaciones"].iloc[-1]) if len(curva) else None)
    trial.set_user_attr("entropia_final", float(curva["entropia"].iloc[-1]) if len(curva) else None)
    trial.set_user_attr("tiempo_s", r["tiempo"])
    trial.set_user_attr("divergio", r["divergio"])
    trial.set_user_attr("motivo_divergencia", r["motivo_divergencia"])
    return puntuacion_resultado(r)


def prueba_a3c(nombre_generador, ruta_almacen, semilla):
    """
    Una prueba de A3C. Se ejecuta en un hilo del proceso principal y no en un proceso aparte, porque cada
    entrenamiento de A3C crea sus propios procesos trabajadores.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    muestreador = optuna.samplers.TPESampler(multivariate=True, constant_liar=True, seed=semilla)
    estudio = optuna.load_study(study_name=f"a3c_{nombre_generador}", storage=almacen(ruta_almacen),
                                sampler=muestreador)
    estudio.optimize(lambda t: objetivo_a3c(t, nombre_generador), n_trials=1)


def confirmar_a3c(nombre_generador, config, repeticion, ruta):
    """Entrena A3C con otra semilla (otras series de entrenamiento); si ya está en disco, lo carga."""
    ruta = Path(ruta)
    if ruta.exists():
        return cargar(ruta)
    series_val, norm = datos_mercado(nombre_generador)
    r = entrenar_a3c(nombre_generador, norm, config, n_episodios=N_EPISODIOS_AJUSTE, repeticion=repeticion,
                     series_val=series_val, cada=CADA_AJUSTE)
    guardar(r, ruta)
    return r
