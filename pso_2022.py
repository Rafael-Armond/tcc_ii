"""
Otimização por Enxame de Partículas para o método de 2022.
"""

from __future__ import annotations
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

from GA_2022 import (
    DATA_DIR,
    BOUNDS_2022,
    clamp_gene,
    evaluate_candidate_fitness_2022,
    init_population_2022,
    _auto_seed,
)


def _project_bounds(x: List[float]) -> List[float]:
    """Projeção nos limites e tratamento do gene inteiro."""
    out = []
    for i, val in enumerate(x):
        out.append(clamp_gene(i, val))
    return out


def _rand_in_bounds(rng: random.Random) -> List[float]:
    g_lo, g_hi = BOUNDS_2022[0]
    a_lo, a_hi = BOUNDS_2022[1]
    b_lo, b_hi = BOUNDS_2022[2]
    d_lo, d_hi = BOUNDS_2022[3]
    return [
        rng.uniform(g_lo, g_hi),
        rng.uniform(a_lo, a_hi),
        rng.randint(int(b_lo), int(b_hi)),
        rng.uniform(d_lo, d_hi),
    ]


def _range_vector() -> List[float]:
    """Amplitude de cada dimensão para definir Vmax."""
    return [
        BOUNDS_2022[0][1] - BOUNDS_2022[0][0],
        BOUNDS_2022[1][1] - BOUNDS_2022[1][0],
        float(BOUNDS_2022[2][1] - BOUNDS_2022[2][0]),
        BOUNDS_2022[3][1] - BOUNDS_2022[3][0],
    ]


def _clamp_velocity(v: List[float], vmax: List[float]) -> List[float]:
    return [max(-vm, min(vm, vi)) for vi, vm in zip(v, vmax)]


# -------------------------------
# Estruturas de log
# -------------------------------
@dataclass
class Particle:
    x: List[float]
    v: List[float]
    pbest_x: List[float]
    pbest_fit: float


@dataclass
class PSOLogEntry:
    it: int
    gbest_x: List[float]
    gbest_fit: float
    gbest_metrics: Dict[str, Any]
    mean_fit: float


# -------------------------------
# PSO
# -------------------------------
def run_pso_2022(
    data_dir,
    swarm_size: int = 36,
    iterations: int = 60,
    subset_pos: int = 20,
    subset_neg: int = 20,
    repeats: int = 3,
    base_seed: Optional[int] = None,
    use_constriction: bool = False,
    w_start: float = 0.9,
    w_end: float = 0.4,
    c1: float = 2.0,
    c2: float = 2.0,
    chi: float = 0.729,
    c1_chi: float = 1.49445,
    c2_chi: float = 1.49445,
    vmax_frac: float = 0.15,
    topology: str = "gbest",  # "gbest" ou "ring"
    ring_k: int = 3,  # vizinhos para lbest (ring)
    patience: int = 10,
    accuracy_goal: Optional[float] = None,
    stagnation_kick: bool = True,
    stagnation_iters: int = 8,
    kick_scale: float = 0.05,
    include_seeds: bool = True,
) -> Tuple[List[float], float, Dict[str, Any], List[PSOLogEntry], List[Particle]]:
    """
    Retorna:
      - gbest_x: melhor vetor de parâmetros [gamma, alfa, beta, beta_diff]
      - gbest_fit: fitness do gbest
      - gbest_metrics: métricas agregadas do gbest
      - history: lista PSOLogEntry por iteração
      - swarm: estado final das partículas do enxame
    """
    if base_seed is None:
        base_seed = _auto_seed()
    rng = random.Random(base_seed)

    # Vmax por dimensão
    ranges = _range_vector()
    vmax = [vmax_frac * r for r in ranges]

    # Inicialização de partículas
    seeds = init_population_2022(
        pop_size=min(swarm_size, 12), seed=base_seed, include_seeds=include_seeds
    )

    swarm: List[Particle] = []
    while len(swarm) < swarm_size:
        if seeds:
            x0 = seeds.pop(0)
        else:
            x0 = _rand_in_bounds(rng)
        v0 = [rng.uniform(-0.1 * r, 0.1 * r) for r in ranges]
        x0 = _project_bounds(x0)

        fit0, _ = evaluate_candidate_fitness_2022(
            x0,
            data_dir,
            subset_pos=subset_pos,
            subset_neg=subset_neg,
            repeats=repeats,
            base_seed=base_seed + len(swarm) * 17,
            verbose=False,
        )
        swarm.append(Particle(x=x0[:], v=v0[:], pbest_x=x0[:], pbest_fit=fit0))

    # Inicializa gbest
    gbest_idx = max(range(len(swarm)), key=lambda i: swarm[i].pbest_fit)
    gbest_x = swarm[gbest_idx].pbest_x[:]
    gbest_fit = swarm[gbest_idx].pbest_fit
    _, gbest_metrics = evaluate_candidate_fitness_2022(
        gbest_x,
        data_dir,
        subset_pos=subset_pos,
        subset_neg=subset_neg,
        repeats=repeats,
        base_seed=base_seed + 99991,
        verbose=False,
    )

    history: List[PSOLogEntry] = []
    no_improve = 0
    no_improve_acc = 0
    best_acc = gbest_metrics.get("accuracy", 0.0)

    def nbest_for(i: int) -> List[float]:
        if topology == "ring":
            idxs = [(i + d) % len(swarm) for d in range(-ring_k, ring_k + 1)]
            j = max(idxs, key=lambda k: swarm[k].pbest_fit)
            return swarm[j].pbest_x
        else:
            return gbest_x

    for it in range(1, iterations + 1):
        t0 = time.time()

        # Ajuste de parâmetros de dinâmica
        if use_constriction:
            _c1, _c2 = c1_chi, c2_chi
            _chi = chi
            _w = 1.0
        else:
            # inércia decrescente linear
            frac = (iterations - it) / max(1, iterations - 1)
            _w = w_end + (w_start - w_end) * max(0.0, min(1.0, frac))
            _c1, _c2 = c1, c2
            _chi = 1.0

        # Atualização de partículas
        fits = []
        for i, p in enumerate(swarm):
            # velocidade
            nbest = nbest_for(i)
            r1 = rng.random()
            r2 = rng.random()
            new_v = [
                _chi
                * (
                    (_w * p.v[d])
                    + _c1 * r1 * (p.pbest_x[d] - p.x[d])
                    + _c2 * r2 * (nbest[d] - p.x[d])
                )
                for d in range(4)
            ]
            new_v = _clamp_velocity(new_v, vmax)

            # posição
            new_x = [p.x[d] + new_v[d] for d in range(4)]
            new_x = _project_bounds(new_x)

            # “kick” de diversidade em partículas estagnadas
            if stagnation_kick and (it % stagnation_iters == 0):
                if p.pbest_fit < (gbest_fit - 1e-12):
                    for d in range(4):
                        new_v[d] += rng.uniform(-kick_scale, kick_scale) * ranges[d]
                    new_v = _clamp_velocity(new_v, vmax)
                    new_x = _project_bounds([p.x[d] + new_v[d] for d in range(4)])

            # avaliação
            fit, _ = evaluate_candidate_fitness_2022(
                new_x,
                data_dir,
                subset_pos=subset_pos,
                subset_neg=subset_neg,
                repeats=repeats,
                base_seed=base_seed + it * 10007 + i * 37,
                verbose=False,
            )

            # atualiza estado da partícula
            p.x = new_x
            p.v = new_v
            # atualiza pbest se ganhou por margem mínima
            if fit > p.pbest_fit + 1e-6:
                p.pbest_x = new_x[:]
                p.pbest_fit = fit
            fits.append(p.pbest_fit)

        # atualiza gbest
        best_idx = max(range(len(swarm)), key=lambda k: swarm[k].pbest_fit)
        candidate_x = swarm[best_idx].pbest_x[:]
        candidate_fit = swarm[best_idx].pbest_fit

        _, candidate_metrics = evaluate_candidate_fitness_2022(
            candidate_x,
            data_dir,
            subset_pos=subset_pos,
            subset_neg=subset_neg,
            repeats=repeats,
            base_seed=base_seed + it * 5 + 123,
            verbose=False,
        )

        improved = candidate_fit > gbest_fit + 1e-6
        acc_improved = candidate_metrics.get("accuracy", 0.0) > (best_acc + 1e-12)

        if improved:
            gbest_x = candidate_x[:]
            gbest_fit = candidate_fit
            gbest_metrics = candidate_metrics
            no_improve = 0
        else:
            no_improve += 1

        if acc_improved:
            best_acc = candidate_metrics.get("accuracy", 0.0)
            no_improve_acc = 0
        else:
            no_improve_acc += 1

        mean_fit = sum(fits) / len(fits)
        history.append(
            PSOLogEntry(
                it=it,
                gbest_x=gbest_x[:],
                gbest_fit=gbest_fit,
                gbest_metrics=gbest_metrics,
                mean_fit=mean_fit,
            )
        )

        if (
            accuracy_goal is not None
            and gbest_metrics.get("accuracy", 0.0) >= accuracy_goal
        ):
            break

        # ausência de melhora de fitness
        if no_improve >= patience:
            break

        if no_improve_acc >= max(2, patience // 2):
            break

        _elapsed = time.time() - t0
        # print(f"[IT {it:03d}] fit={gbest_fit:.4f} acc={gbest_metrics.get('accuracy',0.0):.3f} TP={gbest_metrics.get('TP')} FP={gbest_metrics.get('FP')} FN={gbest_metrics.get('FN')} ({_elapsed:.2f}s)")

    return gbest_x, gbest_fit, gbest_metrics, history, swarm


# =========== Execução ===========
best_x, best_fit, best_metrics, hist, swarm = run_pso_2022(
    DATA_DIR,
    swarm_size=22,
    iterations=16,
    subset_pos=8,
    subset_neg=8,
    repeats=2,
    base_seed=42,
    use_constriction=False,
    w_start=0.9,
    w_end=0.4,
    c1=2.0,
    c2=2.0,
    vmax_frac=0.15,
    topology="gbest",  # ou "ring"
    ring_k=3,
    patience=6,
    accuracy_goal=0.9,
    stagnation_kick=True,
    stagnation_iters=3,
    kick_scale=0.03,
    include_seeds=True,
)

print("\n=== PSO — MELHOR SOLUÇÃO ===")
print("params   :", best_x)  # [gamma, alfa, beta, beta_diff]
print("fitness  :", round(best_fit, 4))
print("metrics  :", best_metrics)


# from GA_2022 import evaluate_candidate_fitness_2022 as eval_fit

# fit_paper, m_paper = eval_fit(
#     [0.0040, 1.05, 40, 1.50],
#     DATA_DIR,
#     subset_pos=20,
#     subset_neg=20,
#     repeats=2,
#     base_seed=_auto_seed() + 13,
#     verbose=False,
# )
# print("\n=== INDIVÍDUO DO ARTIGO (referência) ===")
# print("fitness  :", round(fit_paper, 4))
# print("metrics  :", m_paper)
