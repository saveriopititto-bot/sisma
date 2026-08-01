"""
Valutazione onesta della firma del cammino come predittore.

Il protocollo e' costruito per poter FALLIRE. Ogni confronto e' fuori campione
e contro una baseline esplicita; senza queste due cose qualunque insieme di
predittori sembra funzionare.
"""
import sys
import numpy as np
import pandas as pd

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from math_engine import Binning, Config, Region, run, synthetic_catalog
from math_engine.signature import rolling_levy_area, signature_features

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

BIN_DAYS = 5
WINDOW = 8

# ---------------------------------------------------------------- dati
cat = synthetic_catalog(mc=2.0)
cfg = Config(region=Region(42.5, 43.5, 12.0, 13.5), mc=2.0,
             start=pd.Timestamp("2020-12-01"), target_year=2024,
             binning=Binning("uniform", BIN_DAYS), window=7)
res = run(cfg, catalog=cat)
b = res.binned

path = np.stack([b["m_eq"].to_numpy(), b["r"].to_numpy()], axis=1)
print(f"bin da {BIN_DAYS} giorni: {len(b)}   eventi: {len(cat)}")

# ------------------------------------------- orientazione del ciclo sismico
ms = b.index[b["m_max"].fillna(0).argmax()]
loc = b.index.get_loc(ms)
seq = path[loc - 2: loc + 10]
from math_engine.signature import levy_area
print(f"\nsequenza mainshock (inizio {ms:%Y-%m-%d}):")
print(f"  area di Levy sul ciclo = {levy_area(seq):+.4f}")
print(f"  segno negativo = l'energia guida il tasso (mainshock poi repliche)")

quiete = path[20:32]
print(f"  area su un tratto di fondo    = {levy_area(quiete):+.4f}")

# ---------------------------------------------------------------- feature
feats = signature_features(path, window=WINDOW)
lev = rolling_levy_area(path, WINDOW, normalize=True)

X = pd.DataFrame({
    "dx": feats["dx"], "dy": feats["dy"],
    "cos": feats["cos"], "levy": feats["levy_norm"],
}, index=b.index)

# bersaglio: il bin SUCCESSIVO supera la mediana mobile di 0.5 magnitudo
med = b["m_eq"].rolling(24, min_periods=8).median()
y = ((b["m_eq"].shift(-1) - med) > 0.5).astype(int)

data = X.join(y.rename("target")).dropna()
print(f"\ncampioni utilizzabili: {len(data)}   eventi positivi: {int(data['target'].sum())} "
      f"({data['target'].mean():.1%})")

# ------------------------------------------------- split temporale, no shuffle
cut = int(len(data) * 0.6)
tr, te = data.iloc[:cut], data.iloc[cut:]
print(f"train: {len(tr)}   test: {len(te)}   positivi nel test: {int(te['target'].sum())}")

if te["target"].nunique() < 2:
    print("\nIl test set non contiene entrambe le classi: valutazione impossibile.")
    sys.exit(0)

base_rate = tr["target"].mean()
brier_base = brier_score_loss(te["target"], np.full(len(te), base_rate))
print(f"\nbaseline (frequenza di base {base_rate:.3f}): Brier = {brier_base:.4f}")

sets = {
    "solo livello 1 (dx, dy)": ["dx", "dy"],
    "livello 1 + coseno":      ["dx", "dy", "cos"],
    "livello 1 + area Levy":   ["dx", "dy", "levy"],
    "tutto":                   ["dx", "dy", "cos", "levy"],
}

print(f"\n{'insieme di predittori':<26} {'Brier':>8} {'skill':>8} {'AUC':>7}")
print("-" * 53)
for name, cols in sets.items():
    m = LogisticRegression(max_iter=2000, C=1.0)
    m.fit(tr[cols], tr["target"])
    p = m.predict_proba(te[cols])[:, 1]
    br = brier_score_loss(te["target"], p)
    skill = 1 - br / brier_base
    try:
        auc = roc_auc_score(te["target"], p)
    except ValueError:
        auc = float("nan")
    print(f"{name:<26} {br:>8.4f} {skill:>+8.3f} {auc:>7.3f}")

print("\nskill > 0 significa meglio della baseline, skill < 0 peggio.")
