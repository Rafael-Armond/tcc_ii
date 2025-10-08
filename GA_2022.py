# -*- coding: utf-8 -*-
"""
@author: rafael
"""

import os
import time
import random
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from pathlib import Path
from typing import Optional
import numpy as np

# ------------------------------------------------------------------------------
# Ajuste de PATH para importar o pacote do zip hifdm_optimization
# ------------------------------------------------------------------------------
HERE = Path(os.getcwd())  # D:\TCCII\Dados\tcc_rafael
ROOT = HERE.parent  # D:\TCCII\Dados
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

# Pasta Utilities (para o módulo st.py)
UTILS_DIR = ROOT / "hifdm_optimization" / "Utilities"
if str(UTILS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(UTILS_DIR))

# Importações internas do projeto
from hifdm_optimization.MetodoNunes2022.hifdm import hifdm as hifdm_2022_impl

# from hifdm_optimization.MetodoNunes2024.hifdm import hifdm as hifdm_2024_impl
from hifdm_optimization.OpenPL4.openPL4 import readPL4, convertType

# ------------------------------------------------------------------------------
# Configurações gerais
# ------------------------------------------------------------------------------
DATA_DIR = Path(r"D:\TCCII\Dados\hifdm_optimization\data\sinais_para_otimizar_v2")
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

"""
Estrutura de Algoritmo Genético (GA)
BEGIN 
    INITIALIZE population
    EVALUATE each candidate in population
    WHILE termination criteria not met DO
        SELECT parents from population
        RECOMBINE parents to produce offspring
        MUTATE offspring
        EVALUATE each candidate in offspring
        SELECT individuals for next generation from population and offspring
    END WHILE
END
"""


def _auto_seed():
    return int.from_bytes(os.urandom(8), "little") ^ int(time.time_ns() & 0xFFFFFFFF)


def _build_args_hifdm_2022(
    sig: Dict[str, Any], params: List[float]
) -> Tuple[Tuple, Dict]:
    # >>> Usar o time[] para estimar corretamente as amostras por ciclo
    one_cycle = 128  # samples_per_cycle_from_time(sig)

    # Escolha automática do melhor canal de corrente de fase
    ch_key, sinal_1d = _choose_best_channel_for_2022(sig, one_cycle)

    # Remoção de média (ajuda com offset/DC)
    sinal_1d = sinal_1d - np.mean(sinal_1d)

    args = (sinal_1d,)
    kwargs = {"janela": one_cycle, "parametros": params}
    return args, kwargs, ch_key, one_cycle


# ----------------
# Leitura de PL4
# ----------------
def openpl4(path: str) -> Dict[str, Any]:
    dfHEAD, data, meta = readPL4(path)
    dfHEAD = convertType(dfHEAD)
    out = {
        "__dfHEAD__": dfHEAD,
        "__data__": data,
        "__meta__": {**meta, "filename": str(path)},
    }
    for idx, row in dfHEAD.iterrows():
        key = f"{row['TYPE']}:{row['FROM']}-{row['TO']}"
        out[key] = data[:, idx + 1]
    out["time"] = data[:, 0]
    out["fs"] = 1.0 / meta["deltat"]
    out["f0"] = 60.0
    return out


# -----------------------------------------------------------
# Descoberta de canais e seleção automática (método 2022)
# -----------------------------------------------------------
def _list_phase_currents(sig: Dict[str, Any]) -> List[str]:
    """
    Lista chaves de corrente de fase do tipo:
      - 'I-bran:BUSxA-MEDyA' / ...B / ...C
      - Fallback: chaves 'IA','IB','IC' se existirem
    """
    keys = []
    for k in sig.keys():
        if not isinstance(k, str):
            continue
        if k.startswith("I-bran:") and (
            k.endswith("A") or k.endswith("B") or k.endswith("C")
        ):
            # filtro: parece corrente (não 'V-node', não '2:TACS', etc.)
            if k.split(":")[0] == "I-bran":
                keys.append(k)
    if not keys:
        for alt in ["IA", "IB", "IC"]:
            if alt in sig:
                keys.append(alt)
    return keys


def _choose_best_channel_for_2022(
    sig: Dict[str, Any], one_cycle: int
) -> Tuple[str, np.ndarray]:
    """
    Critério simples e rápido: escolhe o canal com maior razão de energia (último ciclo / primeiro ciclo).
    Ideia: um HIF genuíno tende a alterar (não linearizar) o espectro/energia ao longo dos ciclos.
    """
    cand_keys = _list_phase_currents(sig)
    if not cand_keys:
        raise RuntimeError(
            "Não encontrei canais de corrente de fase para o método 2022."
        )

    def energy(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(np.sum(x * x))

    best_key, best_score = None, -1.0
    for k in cand_keys:
        v = np.asarray(sig[k], dtype=float)
        if len(v) < 2 * one_cycle:
            continue
        first = v[:one_cycle]
        last = v[-one_cycle:]
        e1 = energy(first) + 1e-12
        e2 = energy(last) + 1e-12
        score = e2 / e1
        if score > best_score:
            best_key, best_score = k, score

    if best_key is None:
        # Fallback: pega o primeiro candidato
        best_key = cand_keys[0]
    return best_key, np.asarray(sig[best_key], dtype=float)


def run_hifdm_2022(
    sig: Dict[str, Any], params: List[float], verbose: bool = False
) -> Tuple[int, int, Dict[str, Any]]:
    """
    Chama o hifdm 2022 corretamente: retorna (trip, time, meta)
    """
    args, kwargs, ch_key, one_cycle = _build_args_hifdm_2022(sig, params)
    if verbose:
        print(f"[2022] canal='{ch_key}'  one_cycle={one_cycle}  params={params}")
    out = hifdm_2022_impl(
        *args, **kwargs
    )  # (trip, time) — assinatura do código original (paper 2022) :contentReference[oaicite:1]{index=1}
    if verbose:
        print(f"[2022] retorno hifdm: {out}")
    if not (isinstance(out, tuple) and len(out) >= 2):
        raise RuntimeError(f"Retorno inesperado do hifdm_2022: {out}")
    trip, t = int(out[0]), int(out[1])
    return trip, t, {"channel": ch_key, "one_cycle": one_cycle}


# ---------------------------------------------
# Dataset (assume estrutura do seu repositório)
# ---------------------------------------------
def iter_dataset_pl4(data_dir: Path) -> List[Tuple[Path, int]]:
    data_dir = Path(data_dir)
    positives = []
    for sub in ["FAI", "FAI_com_forno", "FAI_com_gd", "FAI_retificador"]:
        d = data_dir / sub
        if d.exists():
            positives += list(d.rglob("*.pl4"))
    negatives = list((data_dir / "NFAI").glob("*.pl4"))
    return [(p, 1) for p in positives] + [(p, 0) for p in negatives]


# ----------------------------------------------------------
# Avaliação
# ----------------------------------------------------------
def evaluate_params_2022_on_dataset(
    params: List[float],
    data_dir: Path,
    max_pos: int | None = 10,
    max_neg: int | None = 10,
    seed: int = 0,
    verbose_every: int = 0,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    all_items = iter_dataset_pl4(data_dir)
    pos = [p for p, y in all_items if y == 1]
    neg = [p for p, y in iter_dataset_pl4(data_dir) if y == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    if max_pos is not None:
        pos = pos[:max_pos]
    if max_neg is not None:
        neg = neg[:max_neg]
    batch = [(p, 1) for p in pos] + [(p, 0) for p in neg]
    rng.shuffle(batch)

    TP = TN = FP = FN = 0
    det_times = []
    used = errors = 0
    t0 = time.time()

    for i, (path, label) in enumerate(batch, 1):
        try:
            sig = openpl4(str(path))
            trip, tc, meta = run_hifdm_2022(sig, params, verbose=False)
            used += 1
            if trip:
                if label == 1:
                    TP += 1
                    det_times.append(tc)
                else:
                    FP += 1
            else:
                if label == 1:
                    FN += 1
                else:
                    TN += 1

            if verbose_every and (i % verbose_every == 0):
                print(
                    f"[{i}/{len(batch)}] {path.name}  label={label}  trip={trip}  t={tc}  ch={meta['channel']}"
                )

        except Exception as e:
            errors += 1
            if verbose_every:
                print(f"[ERR] {path.name}: {e}")

    elapsed = time.time() - t0
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
        "n_eval": tot,
        "n_used": used,
        "n_errors": errors,
        "elapsed_sec": elapsed,
    }


# ===============================
# FITNESS para o GA (método 2022)
# ===============================


# 1) Definição do "custo" (menor é melhor) a partir das métricas
# função suave e limitada (0,1]; custo=0 → fitness=1
def cost_from_metrics(
    m: Dict[str, Any],
    w_fn: float = 3.0,  # penaliza FN mais forte (perder HIF é pior)
    w_fp: float = 1.5,  # penaliza FP (nunca queremos alarmes falsos)
    w_time: float = 0.03,  # penaliza tempo médio de detecção (fraco)
    max_cycles_norm: int = 300,  # normaliza tmean (~5 s @60Hz ≈ 300 ciclos)
) -> float:
    TP, TN, FP, FN = m["TP"], m["TN"], m["FP"], m["FN"]
    tot_pos = max(1, TP + FN)
    tot_neg = max(1, TN + FP)

    fn_rate = FN / tot_pos
    fp_rate = FP / tot_neg

    # se não houve detecções, use um default conservador
    tmean = m.get("tmean_cycles", None)
    if tmean is None:
        tmean = max_cycles_norm

    time_term = min(1.0, float(tmean) / float(max_cycles_norm))
    cost = (w_fn * fn_rate) + (w_fp * fp_rate) + (w_time * time_term)
    return float(cost)


# 2) Converte custo → fitness (maior é melhor)
# função suave e limitada (0,1]; custo=0 → fitness=1
def fitness_from_cost(cost: float) -> float:
    return 1.0 / (1.0 + max(0.0, cost))


# 3) Avalia um indivíduo (parâmetros) com 1..N repetições para reduzir variância
def evaluate_candidate_fitness_2022(
    params: List[float],
    data_dir: Path,
    subset_pos: Optional[int] = 10,
    subset_neg: Optional[int] = 10,
    repeats: int = 2,  # repetições com seeds diferentes para estabilizar
    base_seed: int = 0,
    verbose: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    Retorna (fitness_médio, metrics_agregadas)
    """
    assert repeats >= 1
    agg = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "tmean_cycles": 0.0, "tmean_count": 0}
    costs = []

    for r in range(repeats):
        seed = base_seed + (r * 9973)
        m = evaluate_params_2022_on_dataset(
            params,
            data_dir,
            max_pos=subset_pos,
            max_neg=subset_neg,
            seed=seed,
            verbose_every=0,
        )
        c = cost_from_metrics(m)
        costs.append(c)

        # agrega contagens
        for k in ["TP", "TN", "FP", "FN"]:
            agg[k] += int(m.get(k, 0))

        # agrega tempos de detecção (média das médias, ponderada por nº de detecções)
        if m.get("tmean_cycles") is not None and m["TP"] > 0:
            agg["tmean_cycles"] += float(m["tmean_cycles"]) * float(m["TP"])
            agg["tmean_count"] += int(m["TP"])

        if verbose:
            print(f"[rep {r+1}/{repeats}] cost={c:.4f}  metrics={m}")

    # custo médio nas repetições
    mean_cost = sum(costs) / len(costs)
    fitness = fitness_from_cost(mean_cost)

    # fecha métricas agregadas
    agg_tot = max(1, agg["TP"] + agg["TN"] + agg["FP"] + agg["FN"])
    metrics_agg = {
        "TP": agg["TP"],
        "TN": agg["TN"],
        "FP": agg["FP"],
        "FN": agg["FN"],
        "accuracy": (agg["TP"] + agg["TN"]) / agg_tot,
        "tmean_cycles": (
            (agg["tmean_cycles"] / agg["tmean_count"])
            if agg["tmean_count"] > 0
            else None
        ),
        "repeats": repeats,
        "mean_cost": mean_cost,
        "fitness": fitness,
    }
    return fitness, metrics_agg


# 4) Avalia a população inteira e imprime ranking
def evaluate_population_fitness_2022(
    population: List[List[float]],
    data_dir: Path,
    subset_pos: Optional[int] = 10,
    subset_neg: Optional[int] = 10,
    repeats: int = 2,
    base_seed: int = 0,
) -> List[Tuple[List[float], float, Dict[str, Any]]]:
    """
    Retorna lista de tuplas (individuo, fitness, metrics_agg), ordenada por fitness decrescente.
    """
    results = []
    for i, ind in enumerate(population, 1):
        fit, m_agg = evaluate_candidate_fitness_2022(
            ind,
            data_dir,
            subset_pos=subset_pos,
            subset_neg=subset_neg,
            repeats=repeats,
            base_seed=base_seed + i * 13,
            verbose=False,
        )
        results.append((ind, fit, m_agg))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ==========================================
# GA COMPLETO para otimizar o método 2022
# ==========================================

# ---------- Espaço de busca ----------
BOUNDS_2022 = {
    0: (1e-9, 1),  # gamma
    1: (1e-3, 10),  # alfa
    2: (1, 300),  # beta (inteiro)
    3: (1e-3, 80),  # beta_diff
}

RESOLUTION = {
    0: 1e-6,  # gamma
    1: 1e-3,  # alfa
    2: 1.0,  # beta (inteiro)
    3: 1e-3,  # beta_diff
}


def clamp_gene(idx: int, val: float):
    lo, hi = BOUNDS_2022[idx]
    if idx == 2:  # beta inteiro
        return int(min(hi, max(lo, round(val))))
    return float(min(hi, max(lo, val)))


# ---------- População inicial ----------
# *
def init_population_2022(
    pop_size=10,
    seed: int = 123,
    include_seeds: bool = True,
) -> list[list[float]]:
    rng = random.Random(seed)
    pop = []

    if include_seeds:
        pop.append([0.0040, 1.05, 40, 1.50])
        pop.append([0.0035, 1.04, 30, 1.30])
        pop.append([0.003765718335758652, 1.0252561383191456, 30, 1.05])
        pop.append([0.005419895645744238, 1.027009354023173, 26, 1.0909437959596677])
        pop.append([0.005419895645744238, 1.027009354023173, 55, 2.2822170682854335])

    while len(pop) < pop_size:
        gamma = rng.uniform(*BOUNDS_2022[0])
        alfa = rng.uniform(*BOUNDS_2022[1])
        beta = rng.randint(int(BOUNDS_2022[2][0]), int(BOUNDS_2022[2][1]))
        bdiff = rng.uniform(*BOUNDS_2022[3])
        pop.append([gamma, alfa, int(beta), bdiff])

    return pop


# ---------- Seleção ----------
def tournament_select(
    scored_pop: list[tuple[list[float], float, dict]],
    rng: Optional[random.Random] = None,
    k: int = 3,
) -> list[float]:
    """Seleciona 1 pai via torneio entre k indivíduos aleatórios."""
    rng = rng or random
    contestants = rng.sample(scored_pop, k=min(k, len(scored_pop)))
    # scored_pop.sort(key=lambda x: x[1], reverse=True) # ordenado por fitness
    scored_pop.sort(key=lambda x: x[2]["accuracy"], reverse=True)  # ordena por accuracy
    return contestants[0][0][:]


# --- Cálculo de níveis/bitwidth por gene ---
def _levels_and_bits(idx: int) -> tuple[int, int]:
    lo, hi = BOUNDS_2022[idx]
    step = RESOLUTION[idx]
    # número de níveis discretos que cabem no intervalo com esse passo
    n_levels = int(math.floor((hi - lo) / step)) + 1
    bits = max(1, math.ceil(math.log2(n_levels)))
    return n_levels, bits


# --- Codificação/Decodificação de gene <-> binário (com largura fixa) ---
def _encode_gene(idx: int, val: float) -> str:
    lo, hi = BOUNDS_2022[idx]
    step = RESOLUTION[idx]
    n_levels, bits = _levels_and_bits(idx)

    v = clamp_gene(idx, val)  # garante bounds (e beta inteiro)
    code = int(round((v - lo) / step))
    code = max(0, min(code, n_levels - 1))  # saturação
    return format(code, f"0{bits}b")  # largura fixa


def _decode_gene(idx: int, bits_str: str) -> float:
    lo, hi = BOUNDS_2022[idx]
    step = RESOLUTION[idx]
    code = int(bits_str, 2)
    v = lo + code * step
    return clamp_gene(idx, v)  # reforça bounds (e beta inteiro)


# --- Helpers para cromossomo inteiro ---
def _bitlayout() -> list[tuple[int, int]]:
    """
    Retorna uma lista com (idx_gene, largura_em_bits) na ordem 0..3.
    """
    layout = []
    for i in range(4):
        _, b = _levels_and_bits(i)
        layout.append((i, b))
    return layout


def _encode_chromosome(p: list[float]) -> tuple[str, list[tuple[int, int]]]:
    """
    Concatena os bits de todos os genes na ordem 0..3.
    Retorna (bitstring_total, layout) para posterior decodificação.
    """
    layout = _bitlayout()
    parts = []
    for idx, _bits in layout:
        parts.append(_encode_gene(idx, p[idx]))
    return "".join(parts), layout


def _decode_chromosome(bitstr: str, layout: list[tuple[int, int]]) -> list[float]:
    """
    Faz o slicing conforme as larguras do layout e decodifica cada gene.
    """
    vals = [0.0] * len(layout)
    pos = 0
    for idx, w in layout:
        slice_bits = bitstr[pos : pos + w]
        vals[idx] = _decode_gene(idx, slice_bits)
        pos += w
    return vals


# ---------- Crossover em 1 ponto sobre a fita binária ----------
def crossover_one_point(
    p1: list[float], p2: list[float], rng: Optional[random.Random] = None
) -> tuple[list[float], list[float]]:
    """
    Single-point crossover no binário concatenado de 4 genes.
    - Cada gene é quantizado com RESOLUTION e mapeado para binário (largura fixa).
    - A fita inteira (concatenação dos 4 genes) é cortada em 1 ponto aleatório.
    - Filhos são decodificados de volta e 'clamped' (beta permanece inteiro).
    """
    rng = rng or random
    if len(p1) != 4 or len(p2) != 4:
        raise ValueError("Indivíduos devem ter 4 genes.")

    # Codifica pais para bitstrings
    b1, layout = _encode_chromosome(p1)
    b2, _ = _encode_chromosome(p2)
    nbits = len(b1)
    if nbits != len(b2):
        raise RuntimeError(
            "Bitstrings com larguras diferentes (verifique RESOLUTION/BOUNDS)."
        )
    if nbits < 2:
        raise RuntimeError("Bitstring muito curta para crossover.")

    # Ponto de corte entre 1..nbits-1 (evita filhos idênticos aos pais)
    cx = rng.randint(1, nbits - 1)

    # Recombina
    c1_bits = b1[:cx] + b2[cx:]
    c2_bits = b2[:cx] + b1[cx:]

    # Decodifica e aplica clamp (já garantido dentro dos helpers)
    c1 = _decode_chromosome(c1_bits, layout)
    c2 = _decode_chromosome(c2_bits, layout)
    return c1, c2


# ---------- Mutação (bit flip em 1 gene) ----------
def mutate_2022(
    ind: list[float],
    pm: float = 0.25,
    rng: Optional[random.Random] = None,
) -> list[float]:
    """
    Mutação por 'bit flip' na representação binária (coerente com o crossover binário):
    - Com probabilidade pm, realiza mutação; caso contrário retorna o indivíduo igual.
    - Se mutar: escolhe exatamente 1 gene (entre 0..3), codifica para binário
      (largura fixa conforme RESOLUTION/BOUNDS), escolhe 1 posição de bit aleatória
      e inverte ('0'->'1' ou '1'->'0'). Depois decodifica e aplica clamp.
    - Trata inteiros e decimais:
        * gene 2 (beta) permanece inteiro (via clamp_gene).
        * genes float são quantizados pela RESOLUTION correspondente.
    """
    rng = rng or random
    out = ind[:]  # cópia

    # decide se muta
    if rng.random() >= pm:
        return out

    # escolhe 1 gene para mutar
    gene_idx = rng.randrange(4)

    # codifica gene selecionado em bits (largura fixa)
    bits_str = _encode_gene(gene_idx, out[gene_idx])
    if not bits_str:
        # fallback de segurança; se por algum motivo vier vazio, não muta
        return out

    # escolhe 1 posição de bit para flipar
    pos = rng.randrange(len(bits_str))

    # flip de 1 bit
    flipped_bit = "1" if bits_str[pos] == "0" else "0"
    new_bits = bits_str[:pos] + flipped_bit + bits_str[pos + 1 :]

    # decodifica de volta e aplica clamp (gene 2 continua inteiro)
    out[gene_idx] = _decode_gene(gene_idx, new_bits)

    return out


# ---------- GA Loop ----------
@dataclass
class GALogEntry:
    gen: int
    best_ind: list[float]
    best_fit: float
    best_metrics: dict
    mean_fit: float


def run_ga_2022(
    data_dir: Path,
    pop_size: int = 10,
    generations: int = 12,
    subset_pos: int = 8,
    subset_neg: int = 8,
    repeats: int = 2,
    base_seed: int | None = 42,
    cx_prob: float = 0.9,
    mut_prob: float = 0.25,
    elitism: int = 2,
    patience: int = 5,
) -> tuple[
    list[float], float, dict, list[GALogEntry], list[tuple[list[float], float, dict]]
]:

    if base_seed is None:
        base_seed = _auto_seed()

    rng = random.Random(base_seed)

    # 1) População inicial
    population = init_population_2022(
        pop_size=pop_size, seed=base_seed, include_seeds=True
    )

    history: list[GALogEntry] = []
    best_global = (None, -1.0, None)  # (ind, fitness, metrics)
    no_improve = 0

    ACCURACY_STOP = 0.90

    for gen in range(1, generations + 1):
        # 2) Avaliar população
        scored = evaluate_population_fitness_2022(
            population,
            data_dir,
            subset_pos=subset_pos,
            subset_neg=subset_neg,
            repeats=repeats,
            base_seed=base_seed + gen * 131,
        )
        # scored: list[(ind, fit, metrics_agg)] ordenada por fit desc
        best_ind, best_fit, best_metrics = scored[0]
        mean_fit = sum(f for _, f, _ in scored) / len(scored)
        history.append(GALogEntry(gen, best_ind, best_fit, best_metrics, mean_fit))

        # parada por acurácia
        acc_candidates = [
            t for t in scored if t[2].get("accuracy", 0.0) >= ACCURACY_STOP
        ]
        if acc_candidates:
            # escolhe o com MAIOR acurácia; em empate, maior fitness
            best_acc_ind, best_acc_fit, best_acc_metrics = max(
                acc_candidates, key=lambda t: (t[2]["accuracy"], t[1])
            )
            return best_acc_ind, best_acc_fit, best_acc_metrics, history, scored

        # print(f"\n[GEN {gen:02d}] best_fit={best_fit:.4f}  best_ind={best_ind}  acc={best_metrics['accuracy']:.3f}  TP={best_metrics['TP']} FP={best_metrics['FP']} FN={best_metrics['FN']}  mean_fit={mean_fit:.4f}")

        # 3) Early stopping (opcional)
        if best_fit > (best_global[1] + 1e-6):
            best_global = (best_ind, best_fit, best_metrics)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                # print(f"[EARLY-STOP] Sem melhora por {patience} gerações.")
                break

        # 4) Elitismo
        elites = [ind for ind, _, _ in scored[:elitism]]

        # 5) Nova população via seleção + crossover + mutação
        new_pop: list[list[float]] = []
        new_pop.extend(elites)

        # gera filhos até completar população
        while len(new_pop) < pop_size:
            p1 = tournament_select(scored, rng=rng, k=3)
            # tenta garantir diversidade do par
            for _ in range(5):
                p2 = tournament_select(scored, rng=rng, k=3)
                if p2 != p1:
                    break
            if rng.random() < cx_prob:
                c1, c2 = crossover_one_point(p1, p2, rng=rng)
            else:
                c1, c2 = p1[:], p2[:]
            # mutação
            c1 = mutate_2022(c1, pm=mut_prob, rng=rng)
            c2 = mutate_2022(c2, pm=mut_prob, rng=rng)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop

    # resultado final
    # reavalia a população final para retornar um ranking final
    final_scored = evaluate_population_fitness_2022(
        population,
        data_dir,
        subset_pos=subset_pos,
        subset_neg=subset_neg,
        repeats=repeats,
        base_seed=base_seed + 999,
    )
    best_ind, best_fit, best_metrics = final_scored[0]
    return best_ind, best_fit, best_metrics, history, final_scored


# ===== Executar o GA ===== #
best_ind, best_fit, best_metrics, history, final_scored = run_ga_2022(
    DATA_DIR,
    pop_size=20,
    generations=200,
    subset_pos=20,
    subset_neg=20,
    repeats=3,
    base_seed=None,
    cx_prob=0.8,
    mut_prob=0.1,
    elitism=3,
    patience=2,
)


# ===== Resultado ===== #
# print("\n=== MELHOR INDIVÍDUO ===")
# print("params   :", best_ind)  # [gamma, alfa, beta, beta_diff]
# print("fitness  :", round(best_fit, 4))
# print("metrics  :", best_metrics)

# Avaliação do indivíduo específico do artigo de 2022
paper_avaliation_candidate = evaluate_candidate_fitness_2022(
    [0.0040, 1.05, 40, 1.50],
    DATA_DIR,
    subset_pos=20,
    subset_neg=20,
    repeats=2,
    base_seed=_auto_seed() + 1 * 13,
    verbose=False,
)

# ===== Resultado ===== #
print("\n=== INDIVÍDUO DO ARTIGO ===")
print("result   :", paper_avaliation_candidate)  # [gamma, alfa, beta, beta_diff]
# print("fitness  :", round(best_fit, 4))
# print("metrics  :", best_metrics)
