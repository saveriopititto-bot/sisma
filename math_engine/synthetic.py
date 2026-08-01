"""
math_engine.synthetic
=====================

Generatore di cataloghi sismici sintetici.

Serve a due scopi: eseguire i test senza toccare la rete, e permettere una
demo dell'applicazione quando FDSNWS non e' raggiungibile. Il catalogo e'
costruito da una legge frequenza-magnitudo troncata, con una sequenza
mainshock-repliche che segue la legge di Omori e alcuni mesi di quiescenza
totale che innescano la censura.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["synthetic_catalog"]


def synthetic_catalog(
    seed: int = 20260801,
    start: str = "2020-12-01",
    n_months: int = 40,
    mc: float = 2.0,
    b_value: float = 1.0,
    rate: int = 14,
    quiet_months: tuple[int, ...] = (8, 9),
    mainshock_month: int = 20,
    mainshock_mag: float = 5.6,
    n_aftershocks: int = 420,
) -> pd.DataFrame:
    """
    Restituisce un catalogo con le stesse colonne di `fetch_catalog`.

    `quiet_months` elenca gli indici dei mesi lasciati privi di eventi, cosi'
    da esercitare il percorso della censura. `mainshock_month` colloca una
    scossa forte seguita da repliche con decadimento di Omori, che esercita il
    percorso della concentrazione energetica.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range(start, periods=n_months, freq="MS")

    def magnitudes(size: int, mmax: float = 6.5) -> np.ndarray:
        u = rng.uniform(0, 1, size)
        cap = 1 - 10 ** (-b_value * (mmax - mc))
        return mc - np.log10(1 - u * cap) / b_value

    frames = []
    for k, month in enumerate(months):
        days = (month + pd.offsets.MonthBegin(1) - month).days
        size = rng.poisson(0 if k in quiet_months else rate)
        if size:
            frames.append(pd.DataFrame({
                "Time": month + pd.to_timedelta(rng.uniform(0, days, size), unit="D"),
                "Magnitude": magnitudes(size),
                "MagType": "ML",
            }))

    origin = months[mainshock_month] + pd.Timedelta(days=11)
    frames.append(pd.DataFrame({
        "Time": [origin], "Magnitude": [mainshock_mag], "MagType": ["Mw"],
    }))

    # Omori: n(t) proporzionale a (t + c)^-p, campionato per inversione.
    c, p = 0.05, 1.1
    u = rng.uniform(0, 1, n_aftershocks)
    delays = ((1 - u) ** (1 / (1 - p)) - 1) * c
    delays = delays[delays < 75]
    frames.append(pd.DataFrame({
        "Time": origin + pd.to_timedelta(delays, unit="D"),
        "Magnitude": magnitudes(len(delays), mmax=mainshock_mag - 0.7),
        "MagType": "ML",
    }))

    catalog = pd.concat(frames).sort_values("Time").reset_index(drop=True)
    catalog = catalog[catalog["Magnitude"] >= mc].reset_index(drop=True)
    catalog["EventID"] = np.arange(len(catalog))
    catalog["Latitude"] = 43.0 + rng.normal(0, 0.05, len(catalog))
    catalog["Longitude"] = 12.7 + rng.normal(0, 0.05, len(catalog))
    catalog["EventLocationName"] = "Catalogo sintetico"

    return catalog
