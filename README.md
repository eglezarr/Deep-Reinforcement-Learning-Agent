# Deep Reinforcement Learning Agent

Agentes de aprendizaje por refuerzo profundo, **DQN** y **A3C** (PyTorch), que aprenden a comprar, mantener o vender un activo en cuatro mercados sintéticos: **ciclo**, **browniano**, **reversión a la media** y **regímenes**. El trabajo continúa un taller previo con Q-Learning y Sarsa tabulares y sigue su mismo protocolo, de modo que los resultados se comparan directamente.

Taller de Deep Reinforcement Learning del Máster MIAX.

## Planteamiento

- **Entorno:** `SimpleTradingEnv` del enunciado. Acciones: vender, mantener o comprar; comisión del 0.1% por operación; episodios de 500 pasos.
- **Estado:** tendencia de los últimos 6 precios, normalizada en cada mercado, y posición (con o sin activo). Sin discretizar: las redes admiten entradas continuas.
- **Recompensa:** la del entorno, escalada por la volatilidad de cada mercado; no cambia la política óptima.
- **Protocolo:** series de entrenamiento, validación (50) y test (200) con semillas disjuntas. Métrica principal: rentabilidad logarítmica media.

## Fases

| Fase | Contenido |
|---|---|
| 0 | Entorno, protocolo y baselines (aleatorio y *buy & hold*) |
| 1 | DQN: memoria de repeticiones, red objetivo y pérdida de Huber |
| 2 | Ajuste del DQN con Optuna (α, γ y exploración), confirmación con semillas nuevas y diagnóstico de la sobreestimación de Q |
| 3 | A3C asíncrono: 4 procesos, Adam compartido, retorno a n pasos y entropía |
| 4 | Ajuste de A3C con Optuna (α, γ, β y t_max) |
| 5 | Convergencia (5 semillas × 5 000 episodios), coste por episodio y test |

## Resultados en test

Rentabilidad logarítmica media en 200 series nuevas. DQN y A3C: media de 5 semillas.

| Mercado | Buy & hold | Q-Learning | Sarsa | DQN | A3C |
|---|---|---|---|---|---|
| Ciclo | 3.33 | 6.24 | 6.25 | 6.09 | **6.27** |
| Browniano | −0.06 | 0.00 | 0.00 | −0.01 | −0.02 |
| Reversión | −0.01 | 0.14 | **0.23** | 0.17 | 0.16 |
| Regímenes | −0.18 | 0.02 | 0.00 | −0.01 | 0.00 |

- El DQN aprende rápido: llega a su techo en menos de 1 000 episodios. A3C aprende más despacio, pero sigue mejorando y con 5 000 episodios lo alcanza y lo supera en ciclo.
- Con un estado de dos variables, las redes no superan a la tabla: evitan discretizar, pero no añaden información.
- Coste por episodio: DQN 49 ms, A3C 9 ms con 4 procesos, tabla ~14 ms.

## Estructura

| Fichero | Contenido |
|---|---|
| `Taller2_DRL.ipynb` | Trabajo completo por fases, con las decisiones justificadas |
| `entorno.py` | Entorno y generadores de precios del enunciado |
| `protocolo.py` | Series, baselines, normalización y evaluación |
| `dqn.py`, `a3c.py` | Agentes |
| `ajuste.py` | Búsqueda de hiperparámetros con Optuna y confirmación |
| `final.py` | Entrenamientos finales, convergencia, tiempos y test |
| `resultados/` | Modelos entrenados y búsquedas: el notebook los carga sin reentrenar |

## Uso

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Abrir `Taller2_DRL.ipynb` con el kernel de `.venv` y ejecutar todo. Con la carpeta `resultados/`, los entrenamientos se cargan de disco; para repetir uno, basta con borrar su fichero.
