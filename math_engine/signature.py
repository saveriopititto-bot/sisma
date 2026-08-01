"""
math_engine.signature
=====================

Firma del cammino (path signature) applicata alle serie sismiche a due canali.

Costruzione
-----------
Dati due canali osservati sulla stessa griglia temporale, si consideri il
cammino poligonale nel piano

    P_k = (x_k, y_k),    x_k = magnitudo equivalente,  y_k = canale di tasso.

Il triangolo generato dall'origine e da due eventi consecutivi ha area con segno

    a_k = 1/2 * (x_k * y_{k+1} - x_{k+1} * y_k)

cioe' meta' del prodotto vettoriale. Dove la covarianza di Pearson usa il
prodotto scalare, e restituisce cos(theta), qui compare il prodotto vettoriale,
che restituisce sin(theta): sono la parte simmetrica e quella antisimmetrica
della stessa forma bilineare.

La somma delle aree lungo il cammino, ancorata al punto iniziale, e' l'area di
Levy

    A = 1/2 * sum_k [ (x_k - x_0) dy_k - (y_k - y_0) dx_k ]

che coincide con il livello 2 antisimmetrico della firma del cammino.

Perche' e' la statistica giusta
-------------------------------
Il livello 1 della firma sono gli incrementi totali, che telescopizzano: per
ogni cammino valgono x_K - x_0 e y_K - y_0, indipendentemente dal percorso. E'
esattamente la degenerazione che rendeva non informativa la somma cumulata
delle aree triangolari nella prima versione del modello.

Il livello 2 antisimmetrico non telescopizza: si annulla sui cammini rettilinei
e cresce con l'area racchiusa dai cicli. Misura quindi il **ritardo relativo**
fra i due canali. Se l'energia guida il tasso, il cammino ruota in un verso; se
il tasso guida l'energia, nel verso opposto.

Proprieta'
----------
- invariante per traslazione (l'ancoraggio a P_0 la rende tale)
- invariante per rotazione del piano, cambia segno per riflessione
- nulla su ogni cammino rettilineo, in particolare su ogni cammino monotono
  in cui un canale e' funzione affine dell'altro
- ricorsiva: la firma di una concatenazione e' il prodotto tensoriale delle
  firme (identita' di Chen), il che permette il calcolo incrementale

Riferimenti
-----------
K.-T. Chen, Integration of paths, geometric invariants and a generalized
Baker-Hausdorff formula, Ann. of Math. 65 (1957).
T. Lyons, Differential equations driven by rough signals, Rev. Mat. Iberoam. 14 (1998).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "segment_signature", "chen_product", "path_signature",
    "levy_area", "rolling_levy_area", "signature_features",
    "SignatureSummary", "summarize_signature",
]


# ---------------------------------------------------------------------------
# Firma
# ---------------------------------------------------------------------------

def segment_signature(delta: np.ndarray, depth: int) -> list[np.ndarray]:
    """
    Firma di un segmento rettilineo di incremento `delta`.

    Per un segmento la firma e' l'esponenziale tensoriale dell'incremento:
    il livello k vale delta^(tensor k) / k!.
    """
    levels, term = [], np.array(1.0)
    factorial = 1.0
    for k in range(1, depth + 1):
        term = np.multiply.outer(term, delta)
        factorial *= k
        levels.append(term / factorial)
    return levels


def chen_product(a: list[np.ndarray], b: list[np.ndarray], depth: int) -> list[np.ndarray]:
    """
    Identita' di Chen: firma della concatenazione di due cammini.

        S(a * b)^{i_1..i_k} = sum_{j=0}^{k} S(a)^{i_1..i_j} S(b)^{i_{j+1}..i_k}

    E' questa la ricorsione che permette di aggiornare la firma un segmento
    alla volta, senza ricalcolare gli integrali iterati da capo.
    """
    out = []
    for k in range(1, depth + 1):
        acc = a[k - 1] + b[k - 1]
        for j in range(1, k):
            acc = acc + np.multiply.outer(a[j - 1], b[k - j - 1])
        out.append(acc)
    return out


def path_signature(path: np.ndarray, depth: int = 2) -> list[np.ndarray]:
    """
    Firma troncata al livello `depth` di un cammino poligonale.

    `path` ha forma (n_punti, dimensione). Restituisce la lista dei livelli
    1..depth; il livello k e' un tensore di forma (dim,)*k.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path deve avere forma (n_punti, dimensione)")
    if len(path) < 2:
        dim = path.shape[1] if path.ndim == 2 else 1
        return [np.zeros((dim,) * k) for k in range(1, depth + 1)]

    deltas = np.diff(path, axis=0)
    sig = segment_signature(deltas[0], depth)
    for delta in deltas[1:]:
        sig = chen_product(sig, segment_signature(delta, depth), depth)
    return sig


def levy_area(path: np.ndarray, i: int = 0, j: int = 1) -> float:
    """
    Area di Levy fra le componenti i e j, calcolata in forma chiusa.

    Equivale alla parte antisimmetrica del livello 2 della firma, ma si ottiene
    direttamente dalla formula dell'area con segno dei triangoli generati dal
    punto iniziale e da coppie di punti consecutivi.
    """
    path = np.asarray(path, dtype=float)
    if len(path) < 3:
        return 0.0
    x = path[:, i] - path[0, i]
    y = path[:, j] - path[0, j]
    return float(0.5 * np.sum(x[:-1] * np.diff(y) - y[:-1] * np.diff(x)))


def rolling_levy_area(
    path: np.ndarray,
    window: int,
    i: int = 0,
    j: int = 1,
    normalize: bool = True,
) -> np.ndarray:
    """
    Area di Levy su finestra scorrevole, allineata a destra.

    L'elemento k usa i punti [k-window+1, k]. Restituisce NaN dove la finestra
    non e' piena.

    Con `normalize=True` l'area viene divisa per il prodotto delle deviazioni
    standard delle due componenti nella finestra, ottenendo una quantita'
    adimensionale confrontabile fra finestre a diversa ampiezza di
    oscillazione. E' l'analogo antisimmetrico della normalizzazione che
    trasforma la covarianza nell'indice di Pearson.
    """
    path = np.asarray(path, dtype=float)
    n = len(path)
    out = np.full(n, np.nan)
    if window < 3:
        raise ValueError("la finestra deve contenere almeno 3 punti")

    for k in range(window - 1, n):
        chunk = path[k - window + 1: k + 1]
        if not np.isfinite(chunk).all():
            continue
        area = levy_area(chunk, i, j)
        if normalize:
            sx, sy = chunk[:, i].std(), chunk[:, j].std()
            scale = sx * sy * (window - 1)
            area = area / scale if scale > 1e-12 else np.nan
        out[k] = area
    return out


# ---------------------------------------------------------------------------
# Sintesi e diagnostica
# ---------------------------------------------------------------------------

@dataclass
class SignatureSummary:
    """Livelli 1 e 2 della firma, con la decomposizione simmetrica."""
    increments: np.ndarray          # livello 1: telescopizza
    level2: np.ndarray              # livello 2 completo
    symmetric: np.ndarray           # parte simmetrica: determinata dal livello 1
    levy: float                     # parte antisimmetrica: informazione nuova
    cos_theta: float                # coseno fra gli incrementi (Pearson del cammino)

    @property
    def telescopes(self) -> bool:
        """Verifica che la parte simmetrica sia ridondante rispetto al livello 1."""
        expected = np.outer(self.increments, self.increments) / 2.0
        return bool(np.allclose(self.symmetric, expected, atol=1e-9))


def summarize_signature(path: np.ndarray) -> SignatureSummary:
    """
    Calcola la firma di livello 2 e ne separa le due componenti.

    La parte simmetrica soddisfa S^{ij} + S^{ji} = S^i S^j: e' interamente
    determinata dal livello 1, quindi non porta informazione oltre gli estremi.
    Tutta l'informazione nuova del livello 2 sta nella parte antisimmetrica,
    cioe' nell'area di Levy.
    """
    path = np.asarray(path, dtype=float)
    sig = path_signature(path, depth=2)
    s1, s2 = sig[0], sig[1]

    sym = 0.5 * (s2 + s2.T)
    asym = 0.5 * (s2 - s2.T)

    norm = np.linalg.norm(s1)
    deltas = np.diff(path, axis=0)
    nx, ny = np.linalg.norm(deltas[:, 0]), np.linalg.norm(deltas[:, 1])
    cos_t = float(deltas[:, 0] @ deltas[:, 1] / (nx * ny)) if nx * ny > 1e-12 else np.nan

    return SignatureSummary(
        increments=s1,
        level2=s2,
        symmetric=sym,
        levy=float(asym[0, 1]) if norm >= 0 else 0.0,
        cos_theta=cos_t,
    )


def signature_features(
    path: np.ndarray,
    window: int,
    depth: int = 2,
) -> dict[str, np.ndarray]:
    """
    Estrae le componenti della firma su finestra scorrevole, pronte per essere
    usate come predittori.

    Restituisce, per ciascun istante:
      dx, dy       incrementi sulla finestra (livello 1)
      levy         area di Levy grezza (livello 2 antisimmetrico)
      levy_norm    area normalizzata, adimensionale
      cos          coseno fra gli incrementi passo-passo (parte simmetrica)

    Il livello 1 e il coseno riproducono l'informazione gia' disponibile; solo
    `levy` aggiunge qualcosa. Sono restituiti insieme proprio per permettere di
    verificare, in fase di validazione, se l'area apporti guadagno reale.
    """
    path = np.asarray(path, dtype=float)
    n = len(path)
    feats = {k: np.full(n, np.nan) for k in ("dx", "dy", "levy", "levy_norm", "cos")}

    for k in range(window - 1, n):
        chunk = path[k - window + 1: k + 1]
        if not np.isfinite(chunk).all():
            continue
        s = summarize_signature(chunk)
        feats["dx"][k] = s.increments[0]
        feats["dy"][k] = s.increments[1]
        feats["levy"][k] = s.levy
        feats["cos"][k] = s.cos_theta
        d = np.diff(chunk, axis=0)
        scale = chunk[:, 0].std() * chunk[:, 1].std() * (window - 1)
        feats["levy_norm"][k] = s.levy / scale if scale > 1e-12 else np.nan

    return feats
