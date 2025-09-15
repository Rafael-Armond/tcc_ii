# main4.py
import os
import json
import time
import math
import random
import inspect
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Any
from pathlib import Path
from datetime import datetime

import numpy as np

# ------------------------------------------------------------------------------
# Ajuste de PATH para importar o pacote do zip hifdm_optimization
# ------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent  # D:\TCCII\Dados\tcc_rafael
ROOT = HERE.parent  # D:\TCCII\Dados
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

# Pasta Utilities (para o módulo st.py)
UTILS_DIR = ROOT / "hifdm_optimization" / "Utilities"
if str(UTILS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(UTILS_DIR))

# Importações do seu projeto
from hifdm_optimization.MetodoNunes2022.hifdm import hifdm as hifdm_2022_impl
from hifdm_optimization.MetodoNunes2024.hifdm import hifdm as hifdm_2024_impl
from hifdm_optimization.OpenPL4.openPL4 import readPL4, convertType

# ST utilitário (o código original usa um "shim" com .st)
from hifdm_optimization.Utilities import st as _st


# ------------------------------------------------------------------------------
# Configurações gerais
# ------------------------------------------------------------------------------
DATA_DIR = Path(r"D:\TCCII\Dados\hifdm_optimization\data\sinais_para_otimizar_v2")
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 123
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)


# ------------------------------------------------------------------------------
# Leitura PL4 (helper)
# ------------------------------------------------------------------------------
def openpl4(path: str) -> Dict[str, Any]:
    dfHEAD, data, miscData = readPL4(path)
    dfHEAD = convertType(dfHEAD)
    out = {}
    out["__dfHEAD__"] = dfHEAD
    out["__data__"] = data
    out["__meta__"] = {**miscData, "filename": str(path)}
    for idx, row in dfHEAD.iterrows():
        key = f"{row['TYPE']}:{row['FROM']}-{row['TO']}"
        out[key] = data[:, idx + 1]
    out["time"] = data[:, 0]
    out["fs"] = 1.0 / miscData["deltat"]
    out["f0"] = 60.0
    return out


# ------------------------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------------------------
def iter_dataset_pl4(data_dir: Path) -> List[Tuple[Path, int]]:
    """
    Retorna lista [(caminho_pl4, label)], label=1 para HIF (FAI*) e 0 para não-HIF (NFAI).
    """
    data_dir = Path(data_dir)
    positives = []
    for sub in ["FAI", "FAI_com_forno", "FAI_com_gd", "FAI_retificador"]:
        d = data_dir / sub
        if d.exists():
            positives += list(d.rglob("*.pl4"))
    negatives = list((data_dir / "NFAI").glob("*.pl4"))
    items = [(p, 1) for p in positives] + [(p, 0) for p in negatives]
    return items


# ------------------------------------------------------------------------------
# Descoberta de triplets do tipo "BUSx-MEDy" a partir das chaves I-bran:FROM-TO
# ------------------------------------------------------------------------------
def extract_triplet_bases(sig: Dict[str, Any], max_items: int | None = 5) -> List[str]:
    """
    Varre as chaves 'I-bran:FROM-TO' e retorna bases únicas 'BUSx-MEDy' (sem a fase A/B/C).
    Ex.: 'I-bran:BUS10A-MED4A' -> base 'BUS10-MED4'
    """
    bases = []
    for k in list(sig.keys()):
        if not isinstance(k, str):
            continue
        if not k.startswith("I-bran:"):
            continue
        try:
            after = k.split("I-bran:")[1]
            FROM, TO = after.split("-")
            base_from = FROM[:-1] if FROM and FROM[-1].isalpha() else FROM
            base_to = TO[:-1] if TO and TO[-1].isalpha() else TO
            base = f"{base_from}-{base_to}"
            if base not in bases:
                bases.append(base)
        except Exception:
            continue
    if max_items is not None:
        bases = bases[:max_items]
    return bases


def inject_triplet_phases(sig: Dict[str, Any], base: str) -> Dict[str, Any]:
    """
    Dado 'BUSx-MEDy', injeta IA/IB/IC no dicionário a partir de:
      I-bran:BUSxA-MEDyA, I-bran:BUSxB-MEDyB, I-bran:BUSxC-MEDyC
    """
    from_id, to_id = base.split("-")
    keyA = f"I-bran:{from_id}A-{to_id}A"
    keyB = f"I-bran:{from_id}B-{to_id}B"
    keyC = f"I-bran:{from_id}C-{to_id}C"
    IA = sig.get(keyA)
    IB = sig.get(keyB)
    IC = sig.get(keyC)
    if IA is not None:
        sig["IA"] = IA
    if IB is not None:
        sig["IB"] = IB
    if IC is not None:
        sig["IC"] = IC
    if "IN" not in sig and IA is not None and IB is not None and IC is not None:
        sig["IN"] = IA + IB + IC
    return sig


# ------------------------------------------------------------------------------
# Adaptadores de chamada aos detectores
# ------------------------------------------------------------------------------
@dataclass
class DetectResult:
    detected: bool
    det_cycles: float | None  # ciclos até detecção (se disponível)
    meta: Dict[str, Any]


def _build_hifdm_kwargs(
    impl: Callable, signal_dict: Dict[str, Any], parametros: List[float]
) -> tuple[tuple, dict]:
    """
    Retorna (args, kwargs) corretos p/ assinatura:
      - 2022: hifdm(sinal_1d, janela, parametros[, show])
      - 2024: hifdm(Ia, Ib, Ic, amostras, parametros)
    """
    import numpy as _np

    sig = inspect.signature(impl)
    params = list(sig.parameters.values())
    names = [p.name for p in params]

    fs = float(signal_dict.get("fs", 0.0))
    f0 = float(signal_dict.get("f0", 60.0))
    one_cycle = int(round(fs / f0)) if (fs > 0 and f0 > 0) else 0
    if one_cycle <= 0:
        raise ValueError(
            f"Não foi possível calcular amostras por ciclo (fs={fs}, f0={f0})."
        )

    IA = signal_dict.get("IA")
    IB = signal_dict.get("IB")
    IC = signal_dict.get("IC")

    # ====== 2022: hifdm(sinal, janela, parametros) — precisa de vetor 1D ======
    if names and names[0] == "sinal":
        candidates = [(IA, "IA"), (IB, "IB"), (IC, "IC")]
        candidates = [(v, n) for v, n in candidates if v is not None]
        if not candidates:
            raise ValueError("Faltam IA/IB/IC para chamar o método 2022.")
        rms_vals = [
            (float(_np.sqrt(_np.mean(_np.square(c)))), n, c) for c, n in candidates
        ]
        rms_vals.sort(reverse=True)
        sinal_1d = rms_vals[0][2]
        args = [sinal_1d]
        kwargs = {"janela": one_cycle, "parametros": parametros}
        if "show" in names:
            kwargs["show"] = False
        return tuple(args), kwargs

    # ====== 2024 original: hifdm(Ia, Ib, Ic, amostras, parametros) ======
    if names and names[0] == "Ia":
        if IA is None or IB is None or IC is None:
            raise ValueError("IA/IB/IC ausentes para chamar hifdm 2024.")
        args = [IA, IB, IC, one_cycle, parametros]
        kwargs = {}
        return tuple(args), kwargs

    # Fallback genérico (se houver variações)
    kwargs = {}
    if "parametros" in names:
        kwargs["parametros"] = parametros
    if "janela" in names:
        kwargs["janela"] = one_cycle
    if "amostras" in names:
        kwargs["amostras"] = one_cycle
    if "Ia" in names and IA is not None:
        kwargs["Ia"] = IA
    if "Ib" in names and IB is not None:
        kwargs["Ib"] = IB
    if "Ic" in names and IC is not None:
        kwargs["Ic"] = IC
    if "show" in names:
        kwargs["show"] = False
    return tuple(), kwargs


def _call_hifdm_impl(
    impl: Callable, signal_dict: Dict[str, Any], parametros: List[float]
) -> DetectResult:
    try:
        args, kwargs = _build_hifdm_kwargs(impl, signal_dict, parametros)
        out = impl(*args, **kwargs)

        if isinstance(out, tuple) and len(out) >= 1:
            detected = bool(out[0])
            det_cycles = float(out[1]) if len(out) > 1 and out[1] is not None else None
            return DetectResult(detected, det_cycles, {"raw": out})

        if isinstance(out, dict):
            detected = bool(
                out.get("detected")
                or out.get("is_hif")
                or out.get("resultado")
                or out.get("trip", False)
            )
            det_cycles = (
                out.get("cycles") or out.get("n_ciclos") or out.get("tempo_ciclos")
            )
            det_cycles = float(det_cycles) if det_cycles is not None else None
            return DetectResult(detected, det_cycles, out)

        return DetectResult(bool(out), None, {"raw": out})

    except Exception as e:
        meta = signal_dict.get("__meta__", {})
        fname = meta.get("filename") or signal_dict.get("source") or "?"
        print(f"[ERR] {impl.__name__} on {fname}: {e}")
        return DetectResult(False, None, {"error": str(e)})


# ------------------------------------------------------------------------------
# WRAPPER 2024 com 5 parâmetros [alpha, zeta, eta, C, N]
# ------------------------------------------------------------------------------
def hifdm_2024_config(
    Ia, Ib, Ic, amostras: int, parametros: List[float]
) -> Tuple[int, int]:
    """
    Implementa o método 2024 permitindo otimizar:
      parametros = [alfa, zeta, eta, C, N]
        - alfa: fator do limiar de energia da fundamental
        - zeta: fator do limiar de rugosidade (3º harmônico)
        - eta: janela (em ciclos) para cálculo da rugosidade
        - C: ciclos de espera após ruptura antes de iniciar confirmação de HIF
        - N: confirmações necessárias para trip
    Retorna (trip, ciclos_processados)
    """
    import numpy as _np
    from statistics import median, stdev

    alfa = float(parametros[0])  # ~[1.01..2.5]
    zeta = float(parametros[1])  # ~[1.01..3.0]
    eta = int(parametros[2])  # janela rugosidade (ciclos) ~[6..18]
    C = int(parametros[3])  # espera pós-ruptura ~[3..20]
    N = int(parametros[4])  # confirmações ~[3..12]

    # buffers de tamanho N (confirmações) como no original
    E1 = [0.0 for _ in range(N)]
    E3 = [0.0 for _ in range(N)]
    GAMMA = [0.0 for _ in range(N)]

    gamma_ene = 0.01
    gamma_r = 0.01

    Ia = _np.array(Ia)
    Ib = _np.array(Ib)
    Ic = _np.array(Ic)
    In = Ia + Ib + Ic

    def energia(espectro):
        return float(np_sum_abs2(espectro))

    # rugosidade a partir das últimas 'eta' energias do 3º harmônico
    def rugosidade(energias: List[float]) -> float:
        n = len(energias)
        if n < 2:
            return 1e-2
        acc = 0.0
        for i in range(1, n):
            d = energias[i] - energias[i - 1]
            acc += d * d
        return max(acc / n, 1e-2)

    tau_1 = 0
    tau_2 = 0
    cont_rupt = 0
    detect_rupt = False
    trip = 0
    ciclo = 0

    # histórico deslizante para rugosidade (eta amostras de E3)
    hist_E3: List[float] = []

    # varre janelas de 1 ciclo
    for ciclo in range(0, len(In) - amostras, amostras):
        # ST do neutro nesta janela
        espectro = _st(In[ciclo : ciclo + amostras], 2)
        fund = espectro[1]
        h3 = espectro[3]

        e1 = energia(fund)
        e3 = energia(h3)

        # atualiza históricos
        E1 = [e1] + E1[:-1]
        E3 = [e3] + E3[:-1]
        hist_E3.append(e3)
        if len(hist_E3) > eta:
            hist_E3.pop(0)

        # rugosidade do 3º harmônico nos últimos 'eta' ciclos
        R = rugosidade(hist_E3)

        # --- Atualização de limiares (igual à lógica do paper/código, mas com α, ζ variáveis)
        if e1 <= max(E1):
            med = median(E1)
            sd = stdev(E1) if len(set(E1)) > 1 else 0.0
            gamma_ene = alfa * (med + sd)

        if len(GAMMA) > 0:
            if R <= max(GAMMA) + (stdev(GAMMA) if len(set(GAMMA)) > 1 else 0.0):
                GAMMA = [R] + GAMMA[:-1]
                sdg = stdev(GAMMA) if len(set(GAMMA)) > 1 else 0.0
                gamma_r = zeta * (max(GAMMA) + sdg)

        # --- Detecção de ruptura (aumento de energia fundamental sustentado)
        if e1 >= gamma_ene:
            tau_1 += 1
        else:
            tau_1 = 0

        if tau_1 >= N:
            detect_rupt = True

        if detect_rupt:
            cont_rupt += 1

        # --- Após aguardar C ciclos, confirma HIF usando rugosidade do 3º
        if cont_rupt >= C:
            if R >= gamma_r:
                tau_2 += 1
            else:
                tau_2 = max(tau_2 - 1, 0)

            if tau_2 >= N:
                trip = 1
                break

    return trip, ciclo // amostras


# helper rápido (evita import re-de-FFT dentro do laço)
def np_sum_abs2(x) -> float:
    x = np.asarray(x)
    return float(np.sum(np.abs(x) ** 2))


# ------------------------------------------------------------------------------
# Avaliação em lote (dataset)
# ------------------------------------------------------------------------------
def evaluate_params_on_dataset(
    method: str,
    parametros: List[float],
    data_dir: Path,
    max_pos: int | None = None,
    max_neg: int | None = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executa o detector em um subconjunto (ou no total) do dataset e produz métricas.
    """
    rng = random.Random(seed)
    all_items = iter_dataset_pl4(data_dir)

    pos = [(p, y) for p, y in all_items if y == 1]
    neg = [(p, y) for p, y in all_items if y == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    if max_pos is not None:
        pos = pos[:max_pos]
    if max_neg is not None:
        neg = neg[:max_neg]

    batch = pos + neg
    rng.shuffle(batch)

    TP = FP = TN = FN = 0
    det_times = []
    errors = 0
    used = 0
    t0 = time.time()

    for path, label in batch:
        sig = openpl4(str(path))

        # Normalização mínima: se não houver neutro explícito, compute IN = IA+IB+IC quando possível
        IA = sig.get("IA")
        IB = sig.get("IB")
        IC = sig.get("IC")
        if IA is None or IB is None or IC is None:
            bases = extract_triplet_bases(sig, max_items=1)
            if bases:
                inject_triplet_phases(sig, bases[0])
                IA, IB, IC = sig.get("IA"), sig.get("IB"), sig.get("IC")
        if (
            sig.get("IN") is None
            and IA is not None
            and IB is not None
            and IC is not None
        ):
            sig["IN"] = IA + IB + IC

        fs = float(sig.get("fs", 0.0))
        f0 = float(sig.get("f0", 60.0))
        one_cycle = int(round(fs / f0)) if (fs > 0 and f0 > 0) else 0
        if one_cycle <= 0:
            errors += 1
            continue

        if method == "2024":
            # parametros = [alpha, zeta, eta, C, N]
            if len(parametros) >= 5:
                C = int(parametros[3])
                N = int(parametros[4])
                n_cycles_total = int(
                    len(sig.get("IN", IA if IA is not None else [])) // one_cycle
                )
                if n_cycles_total < (C + N + 5):
                    continue

        if method == "2022":
            res = _call_hifdm_impl(hifdm_2022_impl, sig, parametros)
        elif method == "2024":
            if IA is None or IB is None or IC is None:
                errors += 1
                continue
            trip, cyc = hifdm_2024_config(IA, IB, IC, one_cycle, parametros)
            res = DetectResult(bool(trip), float(cyc), {"raw": (trip, cyc)})
        else:
            raise ValueError("method deve ser '2022' ou '2024'")

        used += 1
        detected = res.detected
        if "error" in res.meta:
            errors += 1

        if label == 1 and detected:
            TP += 1
            if res.det_cycles is not None:
                det_times.append(res.det_cycles)
        elif label == 1 and not detected:
            FN += 1
        elif label == 0 and detected:
            FP += 1
        else:
            TN += 1

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
        "n_errors": errors,
        "n_eval": tot,
        "n_used": used,
        "elapsed_sec": elapsed,
    }


# ------------------------------------------------------------------------------
# Função-objetivo
# ------------------------------------------------------------------------------
def objective_from_metrics(
    m: Dict[str, Any], w_fn: float = 2.0, w_fp: float = 1.0, w_time: float = 0.05
) -> float:
    """
    Penaliza mais forte os falsos negativos (perder HIF), depois falsos positivos,
    e levemente o tempo médio de detecção (em ciclos).
    """
    TP, TN, FP, FN = m["TP"], m["TN"], m["FP"], m["FN"]
    tot_pos = max(1, TP + FN)
    tot_neg = max(1, TN + FP)

    fn_rate = FN / tot_pos
    fp_rate = FP / tot_neg
    tmean = m["tmean_cycles"] if m["tmean_cycles"] is not None else 9.0

    return (w_fn * fn_rate) + (w_fp * fp_rate) + (w_time * (tmean / 9.0))


# ------------------------------------------------------------------------------
# Espaços de busca (domínios discretizados)
# ------------------------------------------------------------------------------
def default_domain_2022() -> List[List[Any]]:
    # δ (gamma), α, β, β_diff
    delta_vals = np.round(np.linspace(0.0005, 0.01, 20), 6).tolist()
    alpha_vals = np.round(np.linspace(1.01, 1.20, 20), 3).tolist()
    beta_vals = [int(x) for x in np.linspace(10, 60, 15)]
    beta_diff_vals = np.round(np.linspace(1.05, 2.50, 30), 3).tolist()
    return [delta_vals, alpha_vals, beta_vals, beta_diff_vals]


def default_domain_2024() -> List[List[Any]]:
    """
    5 parâmetros no wrapper: [alpha, zeta, eta, C, N]
    Ranges seguros p/ sinais curtos (~9-20 ciclos) e longos:
      - alpha  ∈ [1.01 .. 2.50]   (passo ~0.1)
      - zeta   ∈ [1.01 .. 3.00]   (passo ~0.1)
      - eta    ∈ {6..18}          (ciclos)
      - C      ∈ {3..20}          (ciclos)
      - N      ∈ {3..12}          (ciclos)
    """
    alpha_vals = np.round(np.linspace(1.01, 2.50, 16), 3).tolist()
    zeta_vals = np.round(np.linspace(1.01, 3.00, 16), 3).tolist()
    eta_vals = list(range(6, 19))
    C_vals = list(range(3, 21))
    N_vals = list(range(3, 13))
    return [alpha_vals, zeta_vals, eta_vals, C_vals, N_vals]


# ------------------------------------------------------------------------------
# ACO
# ------------------------------------------------------------------------------
class DiscreteAntColony:
    def __init__(
        self,
        domains: List[List[Any]],
        alpha=1.0,
        beta=2.0,
        rho=0.3,
        Q=1.0,
        ants=24,
        seed=0,
    ):
        """
        domains: lista de listas com os valores possíveis de cada parâmetro
        alpha: influência do feromônio, beta: influência da heurística
        rho: evaporação (0<rho<1), Q: quantidade de feromônio depositado
        ants: número de formigas por iteração
        """
        self.domains = domains
        self.P = len(domains)
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.ants = ants
        self.rng = random.Random(seed)
        self.tau = [[1.0 for _ in dom] for dom in domains]  # feromônios

    def _roulette(self, weights: List[float]) -> int:
        s = sum(weights)
        if s <= 0.0 or not math.isfinite(s):
            return self.rng.randrange(len(weights))
        r = self.rng.random() * s
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                return i
        return len(weights) - 1

    def iterate(
        self, eval_cost: Callable[[List[Any]], float], greedy_k: int = 3
    ) -> Tuple[List[Any], float]:
        """
        Constrói 'ants' soluções, atualiza feromônio e retorna a melhor (params, custo).
        Heurística local: amostra alguns candidatos (greedy_k) por parâmetro.
        """
        solutions = []
        best = (None, float("inf"))

        # Construção das soluções
        for _ in range(self.ants):
            choice_idx = []
            for p, dom in enumerate(self.domains):
                cand_idx = list(range(len(dom)))
                self.rng.shuffle(cand_idx)
                cand_idx = cand_idx[: max(1, min(greedy_k, len(cand_idx)))]

                heur = []
                for k in cand_idx:
                    trial = []
                    for j in range(self.P):
                        if j < len(choice_idx):
                            trial.append(self.domains[j][choice_idx[j]])
                        elif j == p:
                            trial.append(dom[k])
                        else:
                            trial.append(self.domains[j][0])  # preenchimento barato
                    c = eval_cost(trial)
                    heur.append(1.0 / (1e-9 + c))

                if len(heur) == 0:
                    local_weights = [
                        self.tau[p][i] ** self.alpha for i in range(len(dom))
                    ]
                else:
                    baseline = sum(heur) / len(heur)
                    heuristics = [baseline] * len(dom)
                    for k_s, h_s in zip(cand_idx, heur):
                        heuristics[k_s] = h_s
                    local_weights = [
                        (self.tau[p][i] ** self.alpha) * (heuristics[i] ** self.beta)
                        for i in range(len(dom))
                    ]

                idx = self._roulette(local_weights)
                choice_idx.append(idx)

            params = [self.domains[p][choice_idx[p]] for p in range(self.P)]
            cost = eval_cost(params)
            solutions.append((params, cost))
            if cost < best[1]:
                best = (params, cost)

        # Evaporação
        for p in range(self.P):
            for i in range(len(self.domains[p])):
                self.tau[p][i] *= 1.0 - self.rho

        # Depósito nos melhores (elitismo leve: top 3)
        solutions.sort(key=lambda x: x[1])
        top = solutions[: max(1, min(3, len(solutions)))]
        for params, cost in top:
            for p in range(self.P):
                i = self.domains[p].index(params[p])
                self.tau[p][i] += self.Q / (1e-9 + cost)

        return best


# ------------------------------------------------------------------------------
# Otimizador principal com ACO
# ------------------------------------------------------------------------------
def optimize_params_with_aco(
    method: str,
    data_dir: Path,
    n_iters: int = 25,
    ants: int = 24,
    seed: int = 0,
    subset_pos: int | None = 30,
    subset_neg: int | None = 30,
    weights: Tuple[float, float, float] = (2.0, 1.0, 0.05),
    custom_domains: List[List[Any]] | None = None,
    aco_alpha: float = 1.0,
    aco_beta: float = 2.0,
    aco_rho: float = 0.3,
    aco_Q: float = 1.0,
) -> Dict[str, Any]:
    """
    method: '2022' ou '2024'
    Retorna {'best_params','best_cost','best_metrics','history'}
    """
    if method == "2022":
        domains = custom_domains or default_domain_2022()
    elif method == "2024":
        domains = custom_domains or default_domain_2024()
    else:
        raise ValueError("method deve ser '2022' ou '2024'")

    def eval_cost(params: List[Any]) -> float:
        m = evaluate_params_on_dataset(
            method, params, data_dir, max_pos=subset_pos, max_neg=subset_neg, seed=seed
        )
        return objective_from_metrics(m, *weights)

    colony = DiscreteAntColony(
        domains,
        alpha=aco_alpha,
        beta=aco_beta,
        rho=aco_rho,
        Q=aco_Q,
        ants=ants,
        seed=seed,
    )

    history = []
    global_best = (None, float("inf"), None)

    for it in range(1, n_iters + 1):
        best_params, best_cost = colony.iterate(eval_cost)
        best_metrics = evaluate_params_on_dataset(
            method,
            best_params,
            data_dir,
            max_pos=subset_pos,
            max_neg=subset_neg,
            seed=seed + 1234,
        )
        history.append(
            {
                "iter": it,
                "params": best_params,
                "cost": best_cost,
                "metrics": best_metrics,
            }
        )
        if best_cost < global_best[1]:
            global_best = (best_params, best_cost, best_metrics)

        print(
            f"[ACO] {method}  it {it:02d}/{n_iters}  "
            f"cost={best_cost:.4f}  params={best_params}  "
            f"ACC={best_metrics['accuracy']:.3f}  "
            f"TP={best_metrics['TP']}, FP={best_metrics['FP']}, FN={best_metrics['FN']}  "
            f"tmean={best_metrics['tmean_cycles']}  errors={best_metrics['n_errors']}  "
            f"used={best_metrics.get('n_used')}"
        )

    return {
        "best_params": global_best[0],
        "best_cost": global_best[1],
        "best_metrics": global_best[2],
        "history": history,
    }


# ------------------------------------------------------------------------------
# Runner / salvamento de resultados
# ------------------------------------------------------------------------------
def save_json(obj: Dict[str, Any], prefix: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{prefix}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[OK] salvo: {path}")


if __name__ == "__main__":
    common = dict(
        data_dir=DATA_DIR,
        n_iters=25,
        ants=24,
        seed=0,
        subset_pos=40,
        subset_neg=40,
        weights=(2.0, 1.0, 0.05),
        aco_alpha=1.0,
        aco_beta=2.0,
        aco_rho=0.3,
        aco_Q=1.0,
    )

    print("\n=== Otimizando método 2022 ===")
    res_2022 = optimize_params_with_aco(method="2022", **common)
    save_json(res_2022, "aco_2022")

    print("\n=== Otimizando método 2024 ===")
    common["seed"] = 1
    res_2024 = optimize_params_with_aco(method="2024", **common)
    save_json(res_2024, "aco_2024")

    print("\n--- Resumo ---")
    print(
        "2022  best_params:",
        res_2022["best_params"],
        "best_metrics:",
        res_2022["best_metrics"],
    )
    print(
        "2024  best_params:",
        res_2024["best_params"],
        "best_metrics:",
        res_2024["best_metrics"],
    )
