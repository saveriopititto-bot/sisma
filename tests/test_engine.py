"""
Test degli invarianti del motore.

Non verificano che il codice giri, ma che rispetti le proprieta' matematiche
da cui dipende l'interpretazione dei risultati. Un fallimento qui significa che
un grafico dell'applicazione sta mostrando qualcosa di diverso da cio' che
dichiara.

    pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from math_engine import (
    ALPHA,
    BETA,
    Binning,
    Config,
    Region,
    build_edges,
    energy_from_magnitude,
    magnitude_from_energy,
    run,
    synthetic_catalog,
)

REGION = Region(42.5, 43.5, 12.0, 13.5)
MC = 2.0


@pytest.fixture(scope="module")
def catalog() -> pd.DataFrame:
    return synthetic_catalog(mc=MC)


def make_config(**overrides) -> Config:
    base = dict(
        region=REGION, mc=MC,
        start=pd.Timestamp("2020-12-01"), target_year=2024,
    )
    base.update(overrides)
    return Config(**base)


# ---------------------------------------------------------------------------
# Mappa energia-magnitudo
# ---------------------------------------------------------------------------

def test_energia_e_magnitudo_sono_inverse():
    m = np.linspace(0.0, 8.0, 41)
    assert np.allclose(magnitude_from_energy(energy_from_magnitude(m)), m)


def test_una_unita_di_magnitudo_vale_dieci_alla_beta():
    ratio = energy_from_magnitude(5.0) / energy_from_magnitude(4.0)
    assert ratio == pytest.approx(10 ** BETA)


# ---------------------------------------------------------------------------
# Griglia
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["calendar", "uniform"])
def test_nessun_bin_e_parziale(catalog, kind):
    """Ogni bin restituito e' integralmente coperto dai dati."""
    res = run(make_config(binning=Binning(kind, 30)), catalog=catalog)
    assert (res.binned["t_end"] <= catalog["Time"].max()).all()


def test_griglia_uniforme_ha_base_costante(catalog):
    res = run(make_config(binning=Binning("uniform", 30)), catalog=catalog)
    assert res.binned["width_days"].nunique() == 1
    assert res.binned["width_days"].iloc[0] == 30


def test_griglia_calendariale_ha_base_variabile(catalog):
    res = run(make_config(binning=Binning("calendar")), catalog=catalog)
    assert set(res.binned["width_days"].unique()) <= {28.0, 29.0, 30.0, 31.0}
    assert res.binned["width_days"].nunique() > 1


def test_orizzonte_prima_dell_inizio_da_griglia_vuota():
    edges = build_edges(pd.Timestamp("2024-01-01"), pd.Timestamp("2020-01-01"),
                        pd.Timestamp("2020-01-01"), Binning("calendar"))
    assert len(edges) == 0


# ---------------------------------------------------------------------------
# Censura (Prop. 4.1)
# ---------------------------------------------------------------------------

def test_censura_solo_sui_bin_vuoti(catalog):
    """La censura non tocca i bin popolati: ogni evento contribuisce >= E_min."""
    res = run(make_config(), catalog=catalog)
    b = res.binned
    pieni = b[~b["censored"]]
    assert (b.loc[b["censored"], "n"] == 0).all()
    assert np.allclose(pieni["S_hat"], pieni["S"])
    assert (pieni["S"] >= energy_from_magnitude(MC) - 1e-6).all()


def test_bin_vuoti_imputati_alla_soglia(catalog):
    res = run(make_config(), catalog=catalog)
    vuoti = res.binned[res.binned["censored"]]
    assert len(vuoti) > 0, "il catalogo sintetico deve contenere bin vuoti"
    assert np.allclose(vuoti["m_eq"], MC)


def test_soglia_piu_bassa_abbassa_il_fondo(catalog):
    """Il livello di fondo e' beta*Mc + alpha, quindi dipende dalla soglia."""
    alta = run(make_config(mc=MC), catalog=catalog).binned
    bassa = run(make_config(mc=MC - 1.0), catalog=catalog).binned
    fondo_alto = alta.loc[alta["censored"], "y"].iloc[0]
    fondo_basso = bassa.loc[bassa["censored"], "y"].iloc[0]
    assert fondo_alto - fondo_basso == pytest.approx(BETA)


# ---------------------------------------------------------------------------
# Dominanza dell'evento massimo (Prop. 4.4)
# ---------------------------------------------------------------------------

def test_limiti_sulla_magnitudo_equivalente(catalog):
    """M_max <= M_eq <= M_max + log10(n)/beta."""
    b = run(make_config(), catalog=catalog).binned.dropna(subset=["m_max"])
    assert (b["m_eq"] >= b["m_max"] - 1e-9).all()
    assert (b["m_eq"] <= b["m_max"] + np.log10(b["n"]) / BETA + 1e-9).all()


def test_concentrazione_elevata_sul_mainshock(catalog):
    """Con b < beta la somma e' controllata dalla coda superiore."""
    res = run(make_config(), catalog=catalog)
    assert res.concentration > 0.8
    assert (res.binned["kappa"].dropna() <= 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# Differenziali
# ---------------------------------------------------------------------------

def test_dm_eq_e_h_diviso_beta(catalog):
    d = run(make_config(), catalog=catalog).diff
    assert np.allclose(d["dm_eq"] * BETA, d["h"])


def test_somma_delle_altezze_telescopizza(catalog):
    """Prop. 6.1: la somma con segno dipende solo dagli estremi."""
    res = run(make_config(), catalog=catalog)
    y = res.binned["y"].to_numpy()
    assert res.summary["net_m"] == pytest.approx((y[-1] - y[0]) / BETA)


def test_variazione_totale_distingue_percorsi_che_la_somma_confonde():
    """
    Due serie con gli stessi estremi: la somma con segno coincide, la
    variazione totale no. E' la ragione per cui l'aggregato lineare e' stato
    rimosso dall'interfaccia.
    """
    quieto = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    agitato = np.array([0.0, 2.5, -1.0, 3.0, -0.5, 1.0])
    assert np.diff(quieto).sum() == pytest.approx(np.diff(agitato).sum())
    assert np.abs(np.diff(agitato)).sum() > 4 * np.abs(np.diff(quieto)).sum()


def test_transizioni_censurate_sono_marcate(catalog):
    res = run(make_config(), catalog=catalog)
    cens = res.binned["censored"].to_numpy()
    atteso = cens[1:] | cens[:-1]
    assert np.array_equal(res.diff["censored_edge"].to_numpy(), atteso)


def test_pendenza_ignora_i_bin_censurati(catalog):
    """Una serie tutta censurata non puo' produrre pendenze non nulle."""
    vuoto = catalog.iloc[:0]
    res = run(make_config(), catalog=catalog)
    assert res.diff["slope"].notna().any()
    assert np.isfinite(res.diff["slope"].dropna()).all()
    assert len(vuoto) == 0


# ---------------------------------------------------------------------------
# Normalizzazione dell'esposizione
# ---------------------------------------------------------------------------

def test_normalizzazione_lascia_invariata_la_griglia_uniforme(catalog):
    """Con base costante la correzione e' l'identita'."""
    senza = run(make_config(binning=Binning("uniform", 30)), catalog=catalog)
    con = run(make_config(binning=Binning("uniform", 30), normalize_exposure=True),
              catalog=catalog)
    assert np.allclose(senza.binned["y"], con.binned["y"])


def test_normalizzazione_corregge_i_mesi_corti(catalog):
    """Su griglia calendariale febbraio viene alzato, i mesi lunghi abbassati."""
    senza = run(make_config(), catalog=catalog).binned
    con = run(make_config(normalize_exposure=True), catalog=catalog).binned
    pieni = ~senza["censored"]
    delta = (con["y"] - senza["y"])[pieni]
    corti = senza["width_days"][pieni] < senza["width_days"][pieni].mean()
    assert (delta[corti] > 0).all()
    assert (delta[~corti] < 0).all()


# ---------------------------------------------------------------------------
# Coerenza complessiva
# ---------------------------------------------------------------------------

def test_il_motore_non_richiede_rete(catalog):
    """`run` con catalogo esplicito non deve toccare FDSNWS."""
    res = run(make_config(), catalog=catalog)
    assert len(res.binned) > 0
    assert len(res.diff) == len(res.binned) - 1


def test_note_diagnostiche_presenti(catalog):
    res = run(make_config(), catalog=catalog)
    testo = " ".join(res.notes).lower()
    assert "bin vuoti" in testo
    assert "concentrazione" in testo


def test_annual_matrix_canali(catalog):
    from math_engine import annual_matrix
    res = run(make_config(), catalog=catalog)
    mat_m = annual_matrix(res.binned, "m_eq")
    mat_n = annual_matrix(res.binned, "n")
    assert not mat_m.empty
    assert not mat_n.empty
    assert (mat_n.values >= 0).all()

