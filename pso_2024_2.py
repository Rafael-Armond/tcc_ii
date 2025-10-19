from __future__ import annotations
import time, json, random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

# ===== Importa utilidades do seu GA_2024 =====
from GA_2024 import (
    DATA_DIR,
    evaluate_candidate_fitness_2024,
    init_population_2024,
    clamp_gene_2024,
    BOUNDS_2024,
    _auto_seed,
)

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results_2024_PSO_weighted"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- Utilitários ----------------
def _project_bounds_2024(x: List[float]) -> List[float]:
    return [clamp_gene_2024(i, float(x[i])) for i in range(2)]


def _range_vector_2024() -> List[float]:
    return [
        float(BOUNDS_2024[0][1] - BOUNDS_2024[0][0]),
        float(BOUNDS_2024[1][1] - BOUNDS_2024[1][0]),
    ]


def _clamp_velocity(v: List[float], vmax: List[float]) -> List[float]:
    return [max(-vm, min(vm, vi)) for vi, vm in zip(v, vmax)]


def _rand_in_bounds(rng: random.Random) -> List[float]:
    return [rng.uniform(*BOUNDS_2024[0]), rng.uniform(*BOUNDS_2024[1])]


# ---------------- Estruturas ----------------
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


# --------- Fitness: accuracy com punição FP/FN ----------
def weighted_fitness_from_metrics(
    metrics: Dict[str, Any],
    w_tp: float = 0.20,
    w_fp: float = 0.45,
    w_fn: float = 0.90,
    k_acc: float = 0.60,
) -> float:
    """
    Fitness alinhado à accuracy:
      score = k_acc*accuracy + (1-k_acc) * [ +w_tp*TP - w_fp*FP - w_fn*FN ] / total
    - Penaliza FN > FP (w_fn > w_fp), e ainda reforça TP.
    - Normaliza por total para manter a escala.
    """
    tp = float(metrics.get("TP", 0))
    fp = float(metrics.get("FP", 0))
    fn = float(metrics.get("FN", 0))
    tn = float(metrics.get("TN", 0))
    total = max(1.0, tp + fp + fn + tn)
    acc = float(metrics.get("accuracy", 0.0))
    shaped = (w_tp * tp - w_fp * fp - w_fn * fn) / total
    return k_acc * acc + (1.0 - k_acc) * shaped


# ---------------- PSO ----------------
def run_pso_2024(
    data_dir: Path,
    swarm_size: int = 36,
    iterations: int = 100,
    subset_pos: int = 8,
    subset_neg: int = 8,
    repeats: int = 2,
    recheck_best: bool = True,
    recheck_repeats: int = 3,
    base_seed: Optional[int] = None,
    use_constriction: bool = False,
    w_start: float = 0.95,
    w_end: float = 0.35,
    c1: float = 1.6,
    c2: float = 2.2,
    chi: float = 0.729,
    c1_chi: float = 1.49445,
    c2_chi: float = 1.49445,
    vmax_frac: float = 0.12,
    topology: str = "gbest",
    ring_k: int = 3,
    patience: int = 20,
    stagnation_kick: bool = True,
    stagnation_iters: int = 6,
    kick_scale: float = 0.05,
    include_seeds: bool = False,
    switch_on_stagnation: bool = True,
    switch_after_no_improve: int = 30,
    switch_to_topology: str = "ring",
    switch_ring_k: int = 3,
    escalate_after_iter: Optional[int] = 60,
    escalate_subset_pos: int = 12,
    escalate_subset_neg: int = 12,
    escalate_repeats: int = 3,
    w_tp: float = 0.20,
    w_fp: float = 0.45,
    w_fn: float = 0.90,
    k_acc: float = 0.60,
) -> Tuple[List[float], float, Dict[str, Any], List[PSOLogEntry], List[Particle]]:
    if base_seed is None:
        base_seed = _auto_seed()
    rng = random.Random(base_seed)

    ranges = _range_vector_2024()
    vmax = [vmax_frac * r for r in ranges]

    seeds = init_population_2024(
        pop_size=min(swarm_size, 6),
        seed=base_seed,
        include_seeds=include_seeds,
    )

    swarm: List[Particle] = []
    while len(swarm) < swarm_size:
        if seeds:
            x0 = list(seeds.pop(0))
        else:
            x0 = _rand_in_bounds(rng)
        v0 = [rng.uniform(-0.1 * r, 0.1 * r) for r in ranges]
        x0 = _project_bounds_2024(x0)

        _, m0 = evaluate_candidate_fitness_2024(
            x0,
            data_dir,
            subset_pos=subset_pos,
            subset_neg=subset_neg,
            seed=base_seed + len(swarm) * 101,
            verbose_every=0,
        )
        fit0 = weighted_fitness_from_metrics(
            m0, w_tp=w_tp, w_fp=w_fp, w_fn=w_fn, k_acc=k_acc
        )
        swarm.append(Particle(x=x0[:], v=v0[:], pbest_x=x0[:], pbest_fit=fit0))

    g_idx = max(range(len(swarm)), key=lambda i: swarm[i].pbest_fit)
    gbest_x = swarm[g_idx].pbest_x[:]
    _, gbest_metrics = evaluate_candidate_fitness_2024(
        gbest_x,
        data_dir,
        subset_pos=subset_pos,
        subset_neg=subset_neg,
        seed=base_seed + 9999,
        verbose_every=0,
    )
    if recheck_best:
        for j in range(1, recheck_repeats):
            _, gbest_metrics2 = evaluate_candidate_fitness_2024(
                gbest_x,
                data_dir,
                subset_pos=subset_pos,
                subset_neg=subset_neg,
                seed=base_seed + 9999 + j,
                verbose_every=0,
            )

            for k in ("TP", "FP", "FN", "TN"):
                gbest_metrics[k] = gbest_metrics.get(k, 0) + gbest_metrics2.get(k, 0)
            gbest_metrics["accuracy"] = (
                gbest_metrics["accuracy"] + gbest_metrics2["accuracy"]
            ) / 2.0

        for k in ("TP", "FP", "FN", "TN"):
            gbest_metrics[k] = gbest_metrics[k] / float(recheck_repeats)
    gbest_fit = weighted_fitness_from_metrics(
        gbest_metrics, w_tp=w_tp, w_fp=w_fp, w_fn=w_fn, k_acc=k_acc
    )

    history: List[PSOLogEntry] = []
    no_improve = 0
    did_switch = False

    def nbest_for(i: int) -> List[float]:
        if topology == "ring":
            idxs = [(i + d) % len(swarm) for d in range(-ring_k, ring_k + 1)]
            j = max(idxs, key=lambda k: swarm[k].pbest_fit)
            return swarm[j].pbest_x
        return gbest_x

    for it in range(1, iterations + 1):
        cur_subset_pos = subset_pos
        cur_subset_neg = subset_neg
        cur_repeats = repeats
        if escalate_after_iter is not None and it > escalate_after_iter:
            cur_subset_pos = escalate_subset_pos
            cur_subset_neg = escalate_subset_neg
            cur_repeats = max(repeats, escalate_repeats)

        if use_constriction:
            _c1, _c2 = c1_chi, c2_chi
            _chi = chi
            _w = 1.0
        else:
            frac = (iterations - it) / max(1, iterations - 1)
            _w = w_end + (w_start - w_end) * max(0.0, min(1.0, frac))
            _c1, _c2 = c1, c2
            _chi = 1.0

        fits = []
        for i, p in enumerate(swarm):
            nbest = nbest_for(i)
            r1 = rng.random()
            r2 = rng.random()

            new_v = [
                _chi
                * (
                    _w * p.v[d]
                    + _c1 * r1 * (p.pbest_x[d] - p.x[d])
                    + _c2 * r2 * (nbest[d] - p.x[d])
                )
                for d in range(2)
            ]
            new_v = _clamp_velocity(new_v, vmax)
            new_x = _project_bounds_2024([p.x[d] + new_v[d] for d in range(2)])

            if (
                stagnation_kick
                and (it % stagnation_iters == 0)
                and (p.pbest_fit < gbest_fit - 1e-12)
            ):
                for d in range(2):
                    new_v[d] += rng.uniform(-kick_scale, kick_scale) * ranges[d]
                new_v = _clamp_velocity(new_v, vmax)
                new_x = _project_bounds_2024([p.x[d] + new_v[d] for d in range(2)])

            agg_metrics = None
            for rep in range(cur_repeats):
                _, m = evaluate_candidate_fitness_2024(
                    new_x,
                    data_dir,
                    subset_pos=cur_subset_pos,
                    subset_neg=cur_subset_neg,
                    seed=base_seed + it * 1009 + i * 37 + rep,
                    verbose_every=0,
                )
                if agg_metrics is None:
                    agg_metrics = m
                else:
                    for k in ("TP", "FP", "FN", "TN"):
                        agg_metrics[k] += m.get(k, 0)
                    agg_metrics["accuracy"] = (
                        agg_metrics["accuracy"] + m["accuracy"]
                    ) / 2.0

            for k in ("TP", "FP", "FN", "TN"):
                agg_metrics[k] = agg_metrics[k] / float(cur_repeats)

            fit = weighted_fitness_from_metrics(
                agg_metrics, w_tp=w_tp, w_fp=w_fp, w_fn=w_fn, k_acc=k_acc
            )

            p.x = new_x
            p.v = new_v
            if fit > p.pbest_fit + 1e-6:
                p.pbest_x = new_x[:]
                p.pbest_fit = fit
            fits.append(p.pbest_fit)

        b_idx = max(range(len(swarm)), key=lambda k: swarm[k].pbest_fit)
        cand_x = swarm[b_idx].pbest_x[:]

        val_metrics = None
        val_repeats = max(cur_repeats, recheck_repeats if recheck_best else cur_repeats)
        for rep in range(val_repeats):
            _, m = evaluate_candidate_fitness_2024(
                cand_x,
                data_dir,
                subset_pos=cur_subset_pos,
                subset_neg=cur_subset_neg,
                seed=base_seed + it * 5 + 123 + rep,
                verbose_every=0,
            )
            if val_metrics is None:
                val_metrics = m
            else:
                for k in ("TP", "FP", "FN", "TN"):
                    val_metrics[k] += m.get(k, 0)
                val_metrics["accuracy"] = (
                    val_metrics["accuracy"] + m["accuracy"]
                ) / 2.0
        for k in ("TP", "FP", "FN", "TN"):
            val_metrics[k] = val_metrics[k] / float(val_repeats)

        cand_fit = weighted_fitness_from_metrics(
            val_metrics, w_tp=w_tp, w_fp=w_fp, w_fn=w_fn, k_acc=k_acc
        )

        improved = cand_fit > (gbest_fit + 1e-6)
        if improved:
            gbest_x = cand_x[:]
            gbest_fit = cand_fit
            gbest_metrics = val_metrics
            no_improve = 0
        else:
            no_improve += 1

        if (
            switch_on_stagnation
            and not did_switch
            and no_improve >= switch_after_no_improve
        ):
            topology = switch_to_topology
            if topology == "ring":
                ring_k = switch_ring_k
            vmax = [vm * 0.9 for vm in vmax]
            for p in swarm:
                for d, r in enumerate(ranges):
                    p.v[d] += rng.uniform(-0.02, 0.02) * r
            did_switch = True
            no_improve = 0

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

    return gbest_x, gbest_fit, gbest_metrics, history, swarm


# --------------- Execução ---------------
if __name__ == "__main__":
    best_x, best_fit, best_metrics, hist, swarm = run_pso_2024(
        DATA_DIR,
        swarm_size=100,
        iterations=15,
        subset_pos=10,
        subset_neg=10,
        repeats=2,
        recheck_best=False,
        recheck_repeats=1,
        base_seed=None,
        use_constriction=False,
        w_start=0.95,
        w_end=0.35,
        c1=1.6,
        c2=2.2,
        vmax_frac=0.12,
        topology="gbest",
        ring_k=3,
        patience=22,
        stagnation_kick=True,
        stagnation_iters=6,
        kick_scale=0.05,
        include_seeds=True,
        switch_on_stagnation=True,
        switch_after_no_improve=20,
        switch_to_topology="ring",
        switch_ring_k=3,
        escalate_after_iter=60,
        escalate_subset_pos=12,
        escalate_subset_neg=12,
        escalate_repeats=3,
        w_tp=0.20,
        w_fp=0.50,
        w_fn=1.00,
        k_acc=0.60,
    )

    print("\n=== PSO 2024 — MELHOR SOLUÇÃO ===")
    print("params   :", best_x, "  # [alpha, zeta]")
    print("fitness  :", round(best_fit, 6))
    print("metrics  :", best_metrics)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "best_ind": best_x,
        "best_fit": best_fit,
        "best_metrics": best_metrics,
        "history": [
            {
                "it": e.it,
                "gbest_x": e.gbest_x,
                "gbest_fit": e.gbest_fit,
                "gbest_metrics": e.gbest_metrics,
                "mean_fit": e.mean_fit,
            }
            for e in hist
        ],
    }
    out_path = RESULTS_DIR / f"pso2024_weighted_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] salvo: {out_path}")
