# Otimização de parâmetros (alpha, zeta) do método 2024 por GA

from __future__ import annotations
import os, time, math, random, json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from hifdm_optimization.OpenPL4.openPL4 import readPL4, convertType

# Método 2024 — hifdm(Ia, Ib, Ic, amostras, parametros=[alfa,zeta])
from hifdm_optimization.MetodoNunes2024.hifdm import hifdm as hifdm_2024_impl


# ===============================
# Configurações de dados/paths
# ===============================
DATA_DIR = Path(r"D:\TCCII\Dados\hifdm_optimization\data\sinais_para_otimizar_v2")
RESULTS_DIR = HERE / "results_2024_GA"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =======================
# Leitura de arquivos PL4
# =======================
def openpl4(path: str) -> Dict[str, Any]:
    """
    Retorna um dicionário com canais e metadados:
      - chaves tipo "I-bran:FROM-TO" (cada coluna original)
      - 'time', 'fs', 'f0'
    """
    dfHEAD, data, misc = readPL4(path)
    dfHEAD = convertType(dfHEAD)

    out: Dict[str, Any] = {
        "__dfHEAD__": dfHEAD,
        "__data__": data,
        "__meta__": {**misc, "filename": str(path)},
    }
    # col 0 = tempo, col 1.. = canais
    for idx, row in dfHEAD.iterrows():
        key = f"{row['TYPE']}:{row['FROM']}-{row['TO']}"
        out[key] = data[:, idx + 1]
    out["time"] = data[:, 0]
    out["fs"] = 1.0 / misc["deltat"]  # Hz
    out["f0"] = 60.0  # Hz
    return out


# ============================
# Varredura do dataset (rótulo)
# ============================
def iter_dataset_pl4(data_dir: Path) -> List[Tuple[Path, int]]:
    """
    Retorna lista [(caminho, label)], label=1 para HIF (grupos FAI*) e 0 para não-HIF (NFAI).
    """
    data_dir = Path(data_dir)
    positives = []
    for sub in ["FAI", "FAI_com_forno", "FAI_com_gd", "FAI_retificador"]:
        d = data_dir / sub
        if d.exists():
            positives += list(d.rglob("*.pl4"))
    negatives = list((data_dir / "NFAI").rglob("*.pl4"))
    return [(p, 1) for p in positives] + [(p, 0) for p in negatives]


def _extract_triplet_bases(sig: Dict[str, Any]) -> List[str]:
    bases = []
    for k in list(sig.keys()):
        if isinstance(k, str) and k.startswith("I-bran:"):
            try:
                after = k.split("I-bran:")[1]
                FROM, TO = after.split("-")
                base_from = FROM[:-1] if FROM and FROM[-1].isalpha() else FROM
                base_to = TO[:-1] if TO and TO[-1].isalpha() else TO
                base = f"{base_from}-{base_to}"
                if base not in bases:
                    bases.append(base)
            except Exception:
                pass
    return bases


def _inject_triplet(sig: Dict[str, Any], base: str) -> bool:
    f, t = base.split("-")
    kA, kB, kC = f"I-bran:{f}A-{t}A", f"I-bran:{f}B-{t}B", f"I-bran:{f}C-{t}C"
    IA, IB, IC = sig.get(kA), sig.get(kB), sig.get(kC)
    if IA is None or IB is None or IC is None:
        return False
    sig["IA"], sig["IB"], sig["IC"] = IA, IB, IC
    sig["IN"] = IA + IB + IC
    return True


def _samples_per_cycle_from_time(sig: Dict[str, Any]) -> int:
    t = np.asarray(sig.get("time", []), dtype=float)
    if t.size >= 2:
        dt = float(np.median(np.diff(t)))
        fs = 1.0 / dt if dt > 0 else float(sig.get("fs", 0.0))
    else:
        fs = float(sig.get("fs", 0.0))
    f0 = float(sig.get("f0", 60.0))
    oc = int(round(fs / f0)) if (fs > 0 and f0 > 0) else 128
    return max(16, oc)


def run_hifdm_2024(
    sig: Dict[str, Any], params: List[float], verbose: bool = False, top_k: int = 5
) -> Tuple[int, int, Dict[str, Any]]:
    one_cycle = _samples_per_cycle_from_time(sig)

    # rankeia bases pelo RMS do neutro nos 2 primeiros ciclos e pega top_k
    scored = []
    for base in _extract_triplet_bases(sig):
        tmp = dict(sig)
        if not _inject_triplet(tmp, base):
            continue
        IN = np.asarray(tmp["IN"], dtype=float)
        if IN.size < 2 * one_cycle:
            continue
        w = IN[: 2 * one_cycle] - IN[: 2 * one_cycle].mean()
        rms = float(np.sqrt(np.mean(w * w)))
        scored.append((rms, base))
    if not scored:
        raise RuntimeError("Não encontrei nenhum triplet completo A/B/C.")
    scored.sort(reverse=True)
    candidates = [b for _, b in scored[: max(1, top_k)]]

    # testa todos os candidatos e pega o “melhor”
    best_trip, best_t, best_base = 0, 10**9, None
    for base in candidates:
        tmp = dict(sig)
        _inject_triplet(tmp, base)
        IA, IB, IC = (
            np.asarray(tmp["IA"], float),
            np.asarray(tmp["IB"], float),
            np.asarray(tmp["IC"], float),
        )
        out = hifdm_2024_impl(IA, IB, IC, one_cycle, params)
        if not (isinstance(out, tuple) and len(out) >= 2):
            continue
        trip, t_cycles = int(out[0]), int(out[1])
        if trip and t_cycles < best_t:
            best_trip, best_t, best_base = 1, t_cycles, base
    if best_base is None:
        # ninguém trippou -> retorne o melhor RMS para fins de debug
        _, base0 = scored[0]
        return (
            0,
            int(len(sig.get("time", [])) // one_cycle),
            {"base": base0, "one_cycle": one_cycle},
        )
    return best_trip, best_t, {"base": best_base, "one_cycle": one_cycle}


# =======================
# Métricas e agregadores
# =======================
def _metrics_from_hits(hits: List[Tuple[int, int, int]]) -> Dict[str, Any]:
    """
    hits: lista de (label, trip, t_cycles)
    """
    TP = TN = FP = FN = 0
    det_times = []
    for label, trip, t in hits:
        if trip:
            if label == 1:
                TP += 1
                det_times.append(t)
            else:
                FP += 1
        else:
            if label == 1:
                FN += 1
            else:
                TN += 1
    tot = max(1, TP + TN + FP + FN)
    acc = (TP + TN) / tot
    tmean = (sum(det_times) / len(det_times)) if det_times else None
    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "accuracy": acc,
        "tmean_cycles": tmean,
    }


def objective_from_metrics(
    m: Dict[str, Any],
    w_fn: float = 2.5,
    w_fp: float = 3.0,
    w_time: float = 0.05,
    w_tp_reward: float = 2.2,  # bônus por TP
) -> float:
    TP, TN, FP, FN = m["TP"], m["TN"], m["FP"], m["FN"]
    tot_pos = max(1, TP + FN)
    tot_neg = max(1, TN + FP)
    tp_rate = TP / tot_pos
    fn_rate = FN / tot_pos
    fp_rate = FP / tot_neg
    tmean = m["tmean_cycles"] if m["tmean_cycles"] is not None else 9.0

    cost = (
        (1.3 * w_fn * fn_rate)
        + (1.3 * w_fp * fp_rate)
        + (w_time * (tmean / 9.0))
        - (w_tp_reward * tp_rate)
    )
    return cost


# ==========================================================
# Avaliação de um candidato [alfa, zeta] em subset do dataset
# ==========================================================
def evaluate_candidate_fitness_2024(
    params: List[float],
    data_dir: Path,
    subset_pos: int = 10,
    subset_neg: int = 10,
    seed: int = 0,
    verbose_every: int = 0,
) -> Tuple[float, Dict[str, Any]]:
    rng = random.Random(seed)
    all_items = iter_dataset_pl4(data_dir)
    pos = [p for p, y in all_items if y == 1]
    neg = [p for p, y in iter_dataset_pl4(data_dir) if y == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    if subset_pos is not None:
        pos = pos[:subset_pos]
    if subset_neg is not None:
        neg = neg[:subset_neg]
    batch = [(p, 1) for p in pos] + [(p, 0) for p in neg]
    rng.shuffle(batch)

    hits = []
    errors = 0
    t0 = time.time()

    for i, (path, label) in enumerate(batch, 1):
        try:
            sig = openpl4(str(path))
            trip, t_cycles, meta = run_hifdm_2024(sig, params, verbose=False)
            hits.append((label, trip, t_cycles))
            if verbose_every and (i % verbose_every == 0):
                print(
                    f"[{i}/{len(batch)}] {path.name} label={label} -> trip={trip} t={t_cycles} base={meta['base']}"
                )
        except Exception as e:
            errors += 1
            if verbose_every:
                print(f"[ERR] {path.name}: {e}")

    m = _metrics_from_hits(hits)
    m.update(
        {
            "n_eval": len(batch),
            "n_used": len(hits),
            "n_errors": errors,
            "elapsed_sec": time.time() - t0,
        }
    )
    cost = objective_from_metrics(m)
    fitness = 1.0 / (1e-9 + cost)  # fitness maior é melhor
    m["cost"] = cost
    m["fitness"] = fitness
    return fitness, m


# ==========================================
# Avaliar população (com repetição para média)
# ==========================================
def evaluate_population_fitness_2024(
    population: List[List[float]],
    data_dir: Path,
    subset_pos: int = 10,
    subset_neg: int = 10,
    repeats: int = 2,
    base_seed: int = 0,
) -> List[Tuple[List[float], float, Dict[str, Any]]]:
    scored = []
    for idx, ind in enumerate(population, 1):
        fits = []
        agg = {
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0,
            "accuracy": 0.0,
            "tmean_cycles": 0.0,
            "n_eval": 0,
            "n_used": 0,
            "n_errors": 0,
            "elapsed_sec": 0.0,
        }
        tmeans = []
        for r in range(repeats):
            fit, m = evaluate_candidate_fitness_2024(
                ind,
                data_dir,
                subset_pos=subset_pos,
                subset_neg=subset_neg,
                seed=base_seed + (idx * 1009) + r * 131,
                verbose_every=0,
            )
            fits.append(fit)
            for k in ["TP", "TN", "FP", "FN", "n_eval", "n_used", "n_errors"]:
                agg[k] += m[k]
            agg["elapsed_sec"] += m["elapsed_sec"]
            if m["tmean_cycles"] is not None:
                tmeans.append(m["tmean_cycles"])
        # médias
        R = max(1, repeats)
        agg["accuracy"] = (agg["TP"] + agg["TN"]) / max(
            1, (agg["TP"] + agg["TN"] + agg["FP"] + agg["FN"])
        )
        agg["tmean_cycles"] = (sum(tmeans) / len(tmeans)) if tmeans else None
        agg["elapsed_sec"] /= R
        mean_fit = sum(fits) / R
        scored.append((ind, mean_fit, agg))
    # maior fitness primeiro
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ===========================
# Espaço de busca e operadores
# ===========================
BOUNDS_2024 = {
    0: (0.000001, 40.0),  # alpha: fator no limiar de energia da fundamental
    1: (0.000001, 100.0),  # zeta : fator no limiar de rugosidade do 3º harmônico
}

RESOLUTION = {
    0: 0.000001,
    1: 0.000001,
}


def clamp_gene_2024(idx: int, val: float) -> float:
    lo, hi = BOUNDS_2024[idx]
    return float(min(hi, max(lo, val)))


def init_population_2024(
    pop_size=12, seed: Optional[int] = 123, include_seeds=True
) -> List[List[float]]:
    rng = random.Random(seed)
    pop = []
    if include_seeds:
        pop.append([1.03, 7.0])  # artigo
        pop.append([1.10, 2.00])
        pop.append([1.50, 4.00])
    while len(pop) < pop_size:
        a = rng.uniform(*BOUNDS_2024[0])
        z = rng.uniform(*BOUNDS_2024[1])
        pop.append([a, z])
    return pop


def tournament_select(
    scored_pop: List[Tuple[List[float], float, Dict[str, Any]]],
    k: int = 3,
    rng: Optional[random.Random] = None,
) -> List[float]:
    rng = rng or random
    contestants = rng.sample(scored_pop, k=min(k, len(scored_pop)))
    contestants.sort(key=lambda x: x[1], reverse=True)
    return contestants[0][0][:]


def _levels_and_bits(idx: int) -> tuple[int, int]:
    lo, hi = BOUNDS_2024[idx]
    step = RESOLUTION[idx]
    n_levels = int(math.floor((hi - lo) / step)) + 1
    bits = max(1, math.ceil(math.log2(n_levels)))
    return n_levels, bits


def _bitlayout() -> list[tuple[int, int]]:
    layout = []
    for i in range(2):
        _, b = _levels_and_bits(i)
        layout.append((i, b))
    return layout


def _encode_gene(idx: int, val: float) -> str:
    lo, hi = BOUNDS_2024[idx]
    step = RESOLUTION[idx]
    n_levels, bits = _levels_and_bits(idx)

    v = clamp_gene_2024(idx, val)
    code = int(round((v - lo) / step))
    code = max(0, min(code, n_levels - 1))
    return format(code, f"0{bits}b")


def _encode_chromosome(p: list[float]) -> tuple[str, list[tuple[int, int]]]:
    layout = _bitlayout()
    parts = []
    for idx, _bits in layout:
        parts.append(_encode_gene(idx, p[idx]))
    return "".join(parts), layout


def _decode_gene(idx: int, bits_str: str) -> float:
    lo, hi = BOUNDS_2024[idx]
    step = RESOLUTION[idx]
    code = int(bits_str, 2)
    v = lo + code * step
    return clamp_gene_2024(idx, v)


def _decode_chromosome(bitstr: str, layout: list[tuple[int, int]]) -> list[float]:
    vals = [0.0] * len(layout)
    pos = 0
    for idx, w in layout:
        slice_bits = bitstr[pos : pos + w]
        vals[idx] = _decode_gene(idx, slice_bits)
        pos += w
    return vals


def crossover_one_point(
    p1: list[float], p2: list[float], rng: Optional[random.Random] = None
) -> tuple[list[float], list[float]]:
    rng = rng or random
    if len(p1) != 2 or len(p2) != 2:
        raise ValueError("Indivíduos devem ter 2 genes.")

    # Codifica pais para bitstrings
    b1, layout = _encode_chromosome(p1)
    b2, _ = _encode_chromosome(p2)
    nbits = len(b1)
    if nbits != len(b2):
        raise RuntimeError("Bitstrings com larguras diferentes.")
    if nbits < 2:
        raise RuntimeError("Bitstring muito curta para crossover.")

    cx = rng.randint(1, nbits - 1)

    # Recombina
    c1_bits = b1[:cx] + b2[cx:]
    c2_bits = b2[:cx] + b1[cx:]

    # Decodifica e aplica clamp
    c1 = _decode_chromosome(c1_bits, layout)
    c2 = _decode_chromosome(c2_bits, layout)
    return c1, c2


def mutate_2024(
    ind: list[float],
    pm: float = 0.25,
    rng: Optional[random.Random] = None,
) -> list[float]:
    rng = rng or random
    out = ind[:]

    if rng.random() >= pm:
        return out

    # escolhe 1 gene para mutar
    gene_idx = rng.randrange(2)

    # codifica gene selecionado em bits
    bits_str = _encode_gene(gene_idx, out[gene_idx])
    if not bits_str:
        return out

    # escolhe 1 posição de bit para flipar
    pos = rng.randrange(len(bits_str))

    # flip de 1 bit
    flipped_bit = "1" if bits_str[pos] == "0" else "0"
    new_bits = bits_str[:pos] + flipped_bit + bits_str[pos + 1 :]

    # decodifica de volta e aplica clamp
    out[gene_idx] = _decode_gene(gene_idx, new_bits)

    return out


# =================
# Loop do GA (2024)
# =================
@dataclass
class GALogEntry:
    gen: int
    best_ind: List[float]
    best_fit: float
    best_metrics: Dict[str, Any]
    mean_fit: float


def _auto_seed() -> int:
    return int.from_bytes(os.urandom(8), "little") ^ (time.time_ns() & 0xFFFFFFFF)


def run_ga_2024(
    data_dir: Path,
    pop_size: int = 16,
    generations: int = 8,
    subset_pos: int = 12,
    subset_neg: int = 12,
    repeats: int = 2,
    base_seed: Optional[int] = 42,
    tournament_k: int = 3,
    cx_prob: float = 0.9,
    mut_prob: float = 0.25,
    elitism: int = 2,
    patience: int = 4,  # early stop se não melhorar
) -> Tuple[
    List[float],
    float,
    Dict[str, Any],
    List[GALogEntry],
    List[Tuple[List[float], float, Dict[str, Any]]],
]:

    if base_seed is None:
        base_seed = _auto_seed()
    rng = random.Random(base_seed)

    # 1) População inicial
    population = init_population_2024(
        pop_size=pop_size, seed=base_seed, include_seeds=True
    )

    history: List[GALogEntry] = []
    best_global = (None, -1.0, None)  # (ind, fitness, metrics)
    no_improve = 0

    for gen in range(1, generations + 1):
        # 2) Avaliar população
        scored = evaluate_population_fitness_2024(
            population,
            data_dir,
            subset_pos=subset_pos,
            subset_neg=subset_neg,
            repeats=repeats,
            base_seed=base_seed + gen * 131,
        )
        best_ind, best_fit, best_metrics = scored[0]
        mean_fit = sum(f for _, f, _ in scored) / len(scored)
        history.append(GALogEntry(gen, best_ind, best_fit, best_metrics, mean_fit))

        # 3) Early stop
        if best_fit > best_global[1] + 1e-9:
            best_global = (best_ind, best_fit, best_metrics)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        # 4) Elitismo
        elites = [ind for ind, _, _ in scored[:elitism]]

        # 5) Geração de filhos
        new_pop: List[List[float]] = []
        new_pop.extend(elites)
        while len(new_pop) < pop_size:
            p1 = tournament_select(scored, k=tournament_k, rng=rng)
            p2 = tournament_select(scored, k=tournament_k, rng=rng)
            if rng.random() < cx_prob:
                c1, c2 = crossover_one_point(p1, p2, rng=rng)
            else:
                c1, c2 = p1[:], p2[:]
            c1 = mutate_2024(c1, pm=mut_prob, rng=rng)
            c2 = mutate_2024(c2, pm=mut_prob, rng=rng)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        population = new_pop
        print("Uma geração concluída.")
        print(
            "Melhor atual:",
            best_ind,
            "fit:",
            round(best_fit, 6),
            "acc:",
            round(best_metrics["accuracy"], 4),
        )

    # 6) Reavalia ranking final
    final_scored = evaluate_population_fitness_2024(
        population,
        data_dir,
        subset_pos=subset_pos,
        subset_neg=subset_neg,
        repeats=repeats,
        base_seed=base_seed + 999,
    )
    best_ind, best_fit, best_metrics = final_scored[0]
    return best_ind, best_fit, best_metrics, history, final_scored


# =========
# Execução
# =========
if __name__ == "__main__":

    best_ind, best_fit, best_metrics, history, final_scored = run_ga_2024(
        DATA_DIR,
        pop_size=4,
        generations=2,
        subset_pos=8,
        subset_neg=8,
        repeats=2,
        base_seed=42,
        cx_prob=0.65,
        mut_prob=0.15,
        elitism=1,
        patience=3,
    )

    # Avaliar parâmetros do artigo
    # scored = evaluate_population_fitness_2024(
    #     [[1.03, 7.0]],
    #     DATA_DIR,
    #     subset_pos=20,
    #     subset_neg=20,
    #     repeats=1,
    #     base_seed=42,
    # )
    # best_ind, best_fit, best_metrics = scored[0]
    print("\n=== MÉTODO 2024 — MELHOR INDIVÍDUO ===")
    print("params   :", best_ind, "  # [alpha, zeta]")
    print("fitness  :", round(best_fit, 6))
    print("metrics  :", best_metrics)

    # Salva histórico/resultado em JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "best_ind": best_ind,
        "best_fit": best_fit,
        "best_metrics": best_metrics,
        "final_scored": [
            {"ind": ind, "fit": fit, "metrics": m} for ind, fit, m in final_scored
        ],
    }
    out_path = RESULTS_DIR / f"ga2024_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] salvo: {out_path}")
