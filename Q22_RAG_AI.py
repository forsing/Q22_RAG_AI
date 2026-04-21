#!/usr/bin/env python3
"""
Q22 RAG — tehnika: Quantum Retrieval-Augmented Generation (QRAG)
(čisto kvantno: SWAP-test retrieval + aux-weighted superpozicija top-K dokumenata).

Koncept (kvantni analog klasičnog RAG-a „pretraži → uslovi generisanje"):
  1) Korpus:  CEO CSV podeljen u D „dokumenata" (chunk-ovi ~ N/D redova).
              Svaki dokument d je amplitude-encoding svog freq_vector-a → |ψ_d⟩.
  2) Query:   |ψ_q⟩ = amplitude-encoding freq_vector-a poslednjih L redova
              (aktuelni „prompt" / kontekst).
  3) Retrieval (čisto kvantno): za svaki dokument d, realni SWAP-test kolo
              meri s_d = |⟨ψ_q | ψ_d⟩|² preko 2·nq+1 qubit-a.
  4) Top-K:   K = 2^m dokumenata sa najvišim s_d (deterministička stable-sort).
  5) Augmented state (aux-registar sa NEJEDNAKIM težinama, NE Hadamard-uniform):
              aux pripremljen preko StatePreparation u Σ_i √(s_i/Σs) · |i⟩_aux.
              Za i = 0..K-1: multi-ctrl StatePreparation(|ψ_{d_i}⟩) sa ctrl_state=i.
              Rezultat: |Ψ⟩ = Σ_i √(s_i/Σs) · |i⟩_aux ⊗ |ψ_{d_i}⟩_state.
  6) Readout: marginalizacija aux-a → p[k] = Σ_i (s_i/Σs)|ψ_{d_i}[k]|²
              → bias_39 → TOP-7 = NEXT.

Razlika u odnosu na slične fajlove:
  Q10 (QSAN/Attention): svi B blokova doprinose ponderisano u attention-agregaciji,
                        bez top-K selekcije i bez aux-registra (agregacija je klasična).
  Q13 (QCW):            log-spaced tail-prozori sa UNIFORMNOM Hadamard superpozicijom aux-a.
  Q20 (QPTM):           4 fiksna semantička template-a sa UNIFORMNOM Hadamard superpozicijom aux-a.
  QRAG:                 query-zavisna NON-UNIFORMNA aux-superpozicija, težine izvedene
                        iz realnog kvantnog SWAP-test retrieval-a — pravi RAG pattern.

Sve deterministički: seed=39; ceo CSV se deli u dokumente (pravilo 10).
Deterministička grid-optimizacija (nq, D, m, L) po cos(bias_39, freq_csv).

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39

GRID_NQ = (5, 6)
GRID_D = (8, 16)
GRID_M = (1, 2, 3)
GRID_L = (50, 200, 1000)


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


def amp_from_freq(f: np.ndarray, nq: int) -> np.ndarray:
    dim = 2 ** nq
    edges = np.linspace(0, N_MAX, dim + 1, dtype=int)
    amp = np.array(
        [float(f[edges[i] : edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0 for i in range(dim)],
        dtype=np.float64,
    )
    amp = np.maximum(amp, 0.0)
    n2 = float(np.linalg.norm(amp))
    if n2 < 1e-18:
        amp = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    else:
        amp = amp / n2
    return amp


# =========================
# Korpus dokumenata (D contiguous chunk-ova CELOG CSV-a)
# =========================
def document_amps(H: np.ndarray, nq: int, D: int) -> List[np.ndarray]:
    n = H.shape[0]
    edges = np.linspace(0, n, int(D) + 1, dtype=int)
    amps: List[np.ndarray] = []
    for d in range(int(D)):
        lo, hi = int(edges[d]), int(edges[d + 1])
        if hi <= lo:
            amps.append(amp_from_freq(np.zeros(N_MAX), nq))
        else:
            amps.append(amp_from_freq(freq_vector(H[lo:hi]), nq))
    return amps


def query_amp(H: np.ndarray, nq: int, L: int) -> np.ndarray:
    L_eff = max(N_NUMBERS, min(H.shape[0], int(L)))
    return amp_from_freq(freq_vector(H[-L_eff:]), nq)


# =========================
# SWAP-test retrieval — realni kvantni overlap |⟨ψ_q|ψ_d⟩|²
# =========================
def swap_test_overlap_sq(nq: int, amp_q: np.ndarray, amp_d: np.ndarray) -> float:
    """2·nq+1 qubit-a: 1 ancilla + 2 nq-registra; P(ancilla=0) = 1/2 + 1/2·|⟨q|d⟩|²."""
    qc = QuantumCircuit(2 * nq + 1)
    qc.append(StatePreparation(amp_q.tolist()), range(1, nq + 1))
    qc.append(StatePreparation(amp_d.tolist()), range(nq + 1, 2 * nq + 1))
    qc.h(0)
    for i in range(nq):
        qc.cswap(0, 1 + i, nq + 1 + i)
    qc.h(0)

    sv = Statevector(qc)
    probs = np.abs(sv.data) ** 2
    dim = 2 ** (2 * nq + 1)
    p0 = 0.0
    for idx in range(dim):
        if (idx & 1) == 0:
            p0 += float(probs[idx])
    val = 2.0 * p0 - 1.0
    return max(0.0, float(val))


# =========================
# QRAG augmented state: aux (nejednake težine) + multi-ctrl SP po top-K
# =========================
def build_qrag_state(
    nq: int, m: int, top_amps: List[np.ndarray], top_scores: List[float]
) -> Statevector:
    K = 2 ** m
    assert len(top_amps) == K and len(top_scores) == K

    scores = np.maximum(np.asarray(top_scores, dtype=np.float64), 0.0)
    s_sum = float(scores.sum())
    if s_sum < 1e-18:
        aux_vec = np.ones(K, dtype=np.float64) / np.sqrt(K)
    else:
        probs = scores / s_sum
        aux_vec = np.sqrt(probs)
        n2 = float(np.linalg.norm(aux_vec))
        aux_vec = aux_vec / n2 if n2 > 0 else np.ones(K) / np.sqrt(K)

    state = QuantumRegister(nq, name="s")
    aux = QuantumRegister(m, name="a")
    qc = QuantumCircuit(state, aux)

    qc.append(StatePreparation(aux_vec.tolist()), aux)

    for i in range(K):
        sp = StatePreparation(top_amps[i].tolist())
        sp_ctrl = sp.control(num_ctrl_qubits=m, ctrl_state=i)
        qc.append(sp_ctrl, list(aux) + list(state))

    return Statevector(qc)


def qrag_state_probs(H: np.ndarray, nq: int, D: int, m: int, L: int) -> np.ndarray:
    K = 2 ** m
    if K > int(D):
        raise ValueError(f"K={K} > D={D}")

    amps = document_amps(H, nq, D)
    amp_q = query_amp(H, nq, L)

    scores = [swap_test_overlap_sq(nq, amp_q, a) for a in amps]
    order = np.argsort(-np.array(scores, dtype=np.float64), kind="stable")
    top_idx = order[:K]
    top_amps = [amps[int(j)] for j in top_idx]
    top_scores = [float(scores[int(j)]) for j in top_idx]

    sv = build_qrag_state(nq, m, top_amps, top_scores)
    p = np.abs(sv.data) ** 2

    dim_s = 2 ** nq
    dim_a = 2 ** m
    mat = p.reshape(dim_a, dim_s)
    p_s = mat.sum(axis=0)
    s_tot = float(p_s.sum())
    return p_s / s_tot if s_tot > 0 else p_s


# =========================
# Readout
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    return tuple(sorted(int(o + 1) for o in order[:k]))


# =========================
# Determ. grid-optimizacija (nq, D, m, L) — uslov K = 2^m ≤ D
# =========================
def optimize_hparams(H: np.ndarray):
    f_csv = freq_vector(H)
    s_tot = float(f_csv.sum())
    f_csv_n = f_csv / s_tot if s_tot > 0 else np.ones(N_MAX) / N_MAX
    best = None
    for nq in GRID_NQ:
        for D in GRID_D:
            for m in GRID_M:
                if (2 ** m) > int(D):
                    continue
                for L in GRID_L:
                    try:
                        p = qrag_state_probs(H, nq, int(D), int(m), int(L))
                        bi = bias_39(p)
                        score = cosine(bi, f_csv_n)
                    except Exception:
                        continue
                    key = (score, nq, int(m), int(D), -int(L))
                    if best is None or key > best[0]:
                        best = (
                            key,
                            dict(nq=nq, D=int(D), m=int(m), L=int(L), score=float(score)),
                        )
    return best[1] if best else None


def main() -> int:
    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q22 RAG (QRAG — SWAP-test retrieval + aux-weighted top-K): CSV:", CSV_PATH)
    print("redova:", H.shape[0], "| seed:", SEED)

    best = optimize_hparams(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    K_best = 2 ** best["m"]
    print(
        "BEST hparam:",
        "nq=", best["nq"],
        "| D (dokumenata):", best["D"],
        "| m (aux):", best["m"],
        "| K = 2^m (retrieved):", K_best,
        "| L (query window):", best["L"],
        "| cos(bias, freq_csv):", round(float(best["score"]), 6),
    )

    nq_best = int(best["nq"])
    D_best = int(best["D"])
    m_best = int(best["m"])
    L_best = int(best["L"])

    amps = document_amps(H, nq_best, D_best)
    amp_q = query_amp(H, nq_best, L_best)
    scores = [swap_test_overlap_sq(nq_best, amp_q, a) for a in amps]
    order = np.argsort(-np.array(scores), kind="stable")
    top_idx = order[:K_best]
    s_sum = float(max(1e-18, float(sum(max(0.0, scores[int(j)]) for j in top_idx))))
    print("--- retrieved top-K (chunk_idx : SWAP-score : težina) ---")
    for rank, j in enumerate(top_idx):
        s_j = max(0.0, float(scores[int(j)]))
        w_j = s_j / s_sum
        print(f"  rank {rank:d}  chunk={int(j):3d}  s={s_j:.6f}  w={w_j:.6f}")

    p = qrag_state_probs(H, nq_best, D_best, m_best, L_best)
    pred = pick_next_combination(p)
    print("--- glavna predikcija (QRAG augmented state) ---")
    print("predikcija NEXT:", pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q22 RAG (QRAG — SWAP-test retrieval + aux-weighted top-K): CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39
BEST hparam: nq= 5 | D (dokumenata): 8 | m (aux): 3 | K = 2^m (retrieved): 8 | L (query window): 200 | cos(bias, freq_csv): 0.90035
--- retrieved top-K (chunk_idx : SWAP-score : težina) ---
  rank 0  chunk=  2  s=0.988875  w=0.125389
  rank 1  chunk=  5  s=0.987861  w=0.125261
  rank 2  chunk=  4  s=0.987570  w=0.125224
  rank 3  chunk=  7  s=0.985730  w=0.124990
  rank 4  chunk=  3  s=0.985274  w=0.124932
  rank 5  chunk=  1  s=0.985030  w=0.124902
  rank 6  chunk=  6  s=0.984000  w=0.124771
  rank 7  chunk=  0  s=0.982111  w=0.124531
--- glavna predikcija (QRAG augmented state) ---
predikcija NEXT: (7, 19, 22, 24, 27, 28, 31)
"""



"""
Q22_RAG_AI.py — tehnika: Quantum Retrieval-Augmented Generation (QRAG).

Koncept:
Pretraži → uslovi generisanje. Ceo CSV je korpus podeljen u D contiguous dokumenata.
Query je freq_vector poslednjih L redova. Za svaki dokument kvantni SWAP-test
meri overlap s_d = |⟨ψ_q|ψ_d⟩|². Top-K dokumenata po s_d služi kao „retrieved context".
Augmented stanje je NON-UNIFORMNA aux-superpozicija: težine su baš retrieval-skorovi.

Kolo:
SWAP-test (2·nq+1 qubit-a) po dokumentu: H + cswap-ovi + H → P(ancilla=0) daje overlap².
QRAG state (nq + m qubit-a):
  StatePreparation aux-a u Σ_i √(s_i/Σs)·|i⟩ (NE Hadamard).
  Za i = 0..K-1: multi-ctrl StatePreparation(|ψ_{d_i}⟩) sa ctrl_state=i.
  Marginala aux-a → p[k] = Σ_i (s_i/Σs)|ψ_{d_i}[k]|² → bias_39 → TOP-7 = NEXT.

Tehnike:
Amplitude encoding po dokumentu (StatePreparation).
Realni kvantni SWAP-test za retrieval (kao u Q10, ali za ceo korpus).
NON-UNIFORMNA aux-priprema preko StatePreparation (query-dependent težine).
Multi-controlled state prep sa ctrl_state (ekskluzivno po indeksu top-dokumenta).
Egzaktni Statevector (bez uzorkovanja).
Deterministička grid-optimizacija (nq, D, m, L).

Prednosti:
Direktan kvantni analog RAG-a: query-zavisni top-K retrieval + uslovno generisanje.
Non-uniformne aux-težine razlikuju QRAG od svih dosadašnjih aux-superpozicija
(Q13 i Q20 koriste UNIFORMNU Hadamard superpoziciju).
Ceo CSV učestvuje (pravilo 10: korpus = ceo CSV; query = tail-L kao „prompt").
Čisto kvantno: bez klasičnog treninga, bez softmax-a, bez hibrida.

Nedostaci:
D je ograničeno (multi-ctrl SP i 2·nq+1 SWAP-testovi) — ovde D ≤ 16, K ≤ 8.
Retrieval koristi contiguous chunk-ove (deterministička partitura), druge šeme
(npr. preklapajući prozori) davale bi drugi top-K.
mod-39 readout meša stanja (dim 2^nq ≠ 39).
"""
