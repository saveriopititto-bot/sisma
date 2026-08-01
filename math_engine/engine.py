"""
math_engine.engine
==================

Motore di calcolo per l'analisi differenziale del rilascio di energia sismica.

Il modulo e' puro: nessuna dipendenza da Streamlit, nessun effetto collaterale
sull'interfaccia. Solleva eccezioni e restituisce diagnostiche come dati, cosi'
da restare eseguibile da uno script, da un notebook o dai test.

Rispetto alla versione precedente del modello sono cadute due statistiche:

  - `area = base_days * h / 2`, perche' il fattore geometrico introduceva una
    modulazione stagionale dell'8% senza aggiungere informazione;
  - la somma cumulata delle aree, perche' telescopizza: coincide con la
    differenza fra il primo e l'ultimo bin, indipendentemente dal percorso.

La statistica primaria e' ora l'altezza differenziale in unita' di magnitudo
equivalente, `dm_eq = (y_k - y_{k-1}) / BETA`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import requests

__all__ = [
    "ALPHA", "BETA", "MAG_TYPE_NOTE",
    "CatalogError", "EmptyCatalogError",
    "Region", "Binning", "Config", "AnalysisResult",
    "energy_from_magnitude", "magnitude_from_energy",
    "fetch_catalog", "filter_magnitude_types", "build_edges",
    "aggregate", "differentiate", "summarize", "annual_matrix", "run",
]

# ---------------------------------------------------------------------------
# Relazione energia-magnitudo di Gutenberg-Richter (1956):
#
#       log10(E) = BETA * M + ALPHA,      E in joule
#
# BETA e ALPHA non sono i parametri b e a della legge frequenza-magnitudo
# log10 N(>=M) = a - b*M, che e' una legge distinta con lo stesso nome.
# ---------------------------------------------------------------------------
ALPHA: float = 4.8
BETA: float = 1.5

FDSNWS_URL = "https://webservices.ingv.it/fdsnws/event/1/query"

MAG_TYPE_NOTE = (
    "La relazione log10(E) = 1.5*M + 4.8 e' calibrata su magnitudo delle onde "
    "superficiali (Ms). Il catalogo regionale INGV e' prevalentemente ML/Mw: "
    "il bias risultante non e' modellato."
)


class CatalogError(RuntimeError):
    """Errore nel reperimento o nella decodifica del catalogo."""


class EmptyCatalogError(CatalogError):
    """Il servizio ha risposto correttamente ma senza eventi utilizzabili."""


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Region:
    """Rettangolo geografico di interrogazione."""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def __post_init__(self) -> None:
        if self.min_lat > self.max_lat:
            object.__setattr__(self, "min_lat", min(self.min_lat, self.max_lat))
            object.__setattr__(self, "max_lat", max(self.min_lat, self.max_lat))
        if self.min_lon > self.max_lon:
            object.__setattr__(self, "min_lon", min(self.min_lon, self.max_lon))
            object.__setattr__(self, "max_lon", max(self.min_lon, self.max_lon))

    def as_params(self) -> dict[str, float]:
        return {
            "minlat": self.min_lat, "maxlat": self.max_lat,
            "minlon": self.min_lon, "maxlon": self.max_lon,
        }


@dataclass(frozen=True)
class Binning:
    """
    Partizione dell'asse temporale.

    kind="calendar": bin allineati ai mesi solari, ampiezza 28-31 giorni.
    kind="uniform":  bin di ampiezza fissa `width_days`, ancorati all'inizio.

    Con la griglia calendariale l'esposizione varia del +-5% fra mesi; si veda
    `Config.normalize_exposure`.
    """
    kind: Literal["calendar", "uniform"] = "calendar"
    width_days: int = 30

    def __post_init__(self) -> None:
        if self.kind not in ("calendar", "uniform"):
            raise ValueError(f"binning sconosciuto: {self.kind!r}")
        if self.width_days < 1:
            raise ValueError("width_days deve essere >= 1")


@dataclass(frozen=True)
class Config:
    """Parametri completi di un'analisi."""
    region: Region
    mc: float                       # soglia di completezza, coerente col fetch
    start: pd.Timestamp             # inizio della griglia
    target_year: int                # ultimo anno solare richiesto
    binning: Binning = field(default_factory=Binning)
    mag_types: tuple[str, ...] | None = None
    window: int = 7                 # bin per la derivata locale
    normalize_exposure: bool = False
    timeout: int = 30

    @property
    def e_min(self) -> float:
        """Pavimento energetico: energia di un evento alla soglia."""
        return float(energy_from_magnitude(self.mc))

    @property
    def requested_end(self) -> pd.Timestamp:
        """Estremo destro richiesto: fine dell'anno bersaglio."""
        return pd.Timestamp(year=self.target_year + 1, month=1, day=1)


# ---------------------------------------------------------------------------
# Mappa energia-magnitudo
# ---------------------------------------------------------------------------

def energy_from_magnitude(m):
    """E(M) = 10^(BETA*M + ALPHA), in joule. Vettorializzata."""
    return np.power(10.0, BETA * np.asarray(m, dtype=float) + ALPHA)


def magnitude_from_energy(e):
    """Magnitudo equivalente: inversa di `energy_from_magnitude`."""
    return (np.log10(np.asarray(e, dtype=float)) - ALPHA) / BETA


# ---------------------------------------------------------------------------
# Reperimento del catalogo
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS = ("EventID", "Time", "Latitude", "Longitude", "Magnitude")


def fetch_catalog(
    region: Region,
    start: pd.Timestamp,
    end: pd.Timestamp,
    mc: float,
    timeout: int = 30,
    url: str = FDSNWS_URL,
) -> pd.DataFrame:
    """
    Interroga il web service FDSNWS dell'INGV e restituisce il catalogo.

    Solleva `EmptyCatalogError` se la finestra non contiene eventi utilizzabili,
    `CatalogError` per errori di rete o di formato. Non stampa nulla.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    params = {
        "starttime": start_ts.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end_ts.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmag": float(mc),
        "format": "text",
        **region.as_params(),
    }

    headers = {
        "User-Agent": "SISMA-Sismologia-Analitica/1.0 (INGV FDSNWS Client)",
        "Accept": "text/plain, */*",
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        raise CatalogError(f"Errore di connessione verso il servizio INGV FDSNWS: {exc}") from exc

    if response.status_code == 204 or not response.text.strip():
        raise EmptyCatalogError(
            f"Nessun evento sismico trovato dall'INGV per l'area selezionata "
            f"[{region.min_lat:.2f}–{region.max_lat:.2f}°N, {region.min_lon:.2f}–{region.max_lon:.2f}°E], "
            f"periodo {start_ts:%Y-%m-%d} → {end_ts:%Y-%m-%d} con M >= {mc}. "
            "Prova ad abbassare la soglia M_c o ad ampliare il bounding box."
        )

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        raise CatalogError(f"Servizio INGV FDSNWS ha risposto con codice HTTP {response.status_code}") from exc

    try:
        df = pd.read_csv(StringIO(response.text), sep="|")
    except Exception as exc:
        raise CatalogError(f"Risposta del catalogo INGV non decodificabile: {exc}") from exc

    # L'header del formato text e' "#EventID|Time|...": pulizia spazi e cancelletto.
    df.columns = df.columns.str.strip().str.lstrip("#")

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise CatalogError(f"Colonne essenziali assenti nel catalogo INGV: {missing}")

    # Rimozione duplicati
    if "EventID" in df.columns:
        df = df.drop_duplicates(subset=["EventID"])

    # Conversione date in UTC naive
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce", utc=True).dt.tz_localize(None)
    df["Magnitude"] = pd.to_numeric(df["Magnitude"], errors="coerce")
    df = df.dropna(subset=["Time", "Magnitude"])

    if df.empty:
        raise EmptyCatalogError("Tutti gli eventi scaricati sono stati scartati in fase di parsing")

    # Filtro sul tipo evento se presente (esclude esplosioni di cava se esplicite)
    if "EventType" in df.columns:
        non_seismic = df["EventType"].astype(str).str.lower().isin(["quarry blast", "explosion", "other event"])
        df = df[~non_seismic]

    df = df[df["Magnitude"] >= mc]
    if df.empty:
        raise EmptyCatalogError(f"Nessun evento sismico con magnitudo M >= {mc}")

    return df.sort_values("Time").reset_index(drop=True)


def filter_magnitude_types(
    catalog: pd.DataFrame,
    allowed: Sequence[str] | None,
) -> pd.DataFrame:
    """
    Restringe il catalogo ai tipi di magnitudo indicati, es. ("ML", "Mw").

    Filtrare rende omogenea la scala ma puo' rimuovere l'evento massimo di un
    bin, spostando y_k in modo non trascurabile. `allowed=None` non filtra.
    """
    if allowed is None or "MagType" not in catalog.columns:
        return catalog
    mask = catalog["MagType"].astype(str).str.strip().isin(allowed)
    return catalog.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Griglia temporale
# ---------------------------------------------------------------------------

def build_edges(
    start: pd.Timestamp,
    requested_end: pd.Timestamp,
    coverage_end: pd.Timestamp,
    binning: Binning,
) -> pd.DatetimeIndex:
    """
    Bordi dei bin, scartando l'ultimo bin non integralmente coperto dai dati.

    Criterio unico: si generano i bordi oltre l'orizzonte, poi si tengono solo
    quelli non successivi a

        min(fine richiesta, fine dei dati, oggi)

    Un bin [e[i], e[i+1]) sopravvive se e[i+1] rispetta il vincolo, quindi
    nessun bin restituito e' parziale. Il criterio vale identico per griglia
    calendariale e uniforme.
    """
    start = pd.Timestamp(start).normalize()
    horizon = min(pd.Timestamp(requested_end),
                  pd.Timestamp(coverage_end),
                  pd.Timestamp.today().normalize())

    if horizon <= start:
        return pd.DatetimeIndex([], name="edge")

    if binning.kind == "calendar":
        start = start.replace(day=1)
        edges = pd.date_range(start, horizon + pd.offsets.MonthBegin(2), freq="MS")
    else:
        step = pd.Timedelta(days=binning.width_days)
        n = int(np.ceil((horizon - start) / step)) + 2
        edges = pd.date_range(start, periods=n + 1, freq=step)

    valid = edges[edges <= horizon]
    if len(valid) < 2:
        return pd.DatetimeIndex([], name="edge")
    return pd.DatetimeIndex(valid, name="edge")


# ---------------------------------------------------------------------------
# Aggregazione
# ---------------------------------------------------------------------------

def aggregate(
    catalog: pd.DataFrame,
    edges: pd.DatetimeIndex,
    mc: float,
    normalize_exposure: bool = False,
) -> pd.DataFrame:
    """
    Aggrega il catalogo sui bin definiti da `edges`.

    Colonne restituite, indicizzate sul bordo sinistro del bin:

      t_end        bordo destro, escluso
      width_days   ampiezza del bin in giorni
      n            numero di eventi
      censored     True se n == 0, energia imputata al pavimento
      m_max        magnitudo massima osservata, NaN se il bin e' vuoto
      S            energia grezza [J]
      S_hat        energia censurata dal basso [J]
      y            log10(S_hat) [dex]
      m_eq         magnitudo equivalente, (y - ALPHA) / BETA
      kappa        E_max / S, concentrazione sull'evento massimo
      r            log10(1 + n), canale di tasso

    La censura `S_hat = max(S, E_min)` agisce solo sui bin vuoti: ogni evento
    soddisfa M >= mc e contribuisce quindi almeno E_min. Equivale a imputare
    un evento singolo di magnitudo mc.

    Con `normalize_exposure=True` l'energia viene riportata a un bin di
    lunghezza media prima della censura (S * b_medio / b_k), correggendo il
    fatto che un febbraio raccoglie tre giorni di sismicita' in meno di un
    gennaio. La correzione e' esatta per un tasso energetico costante e
    approssimata quando la somma e' dominata dall'evento massimo.
    """
    if len(edges) < 2:
        return pd.DataFrame()

    left, right = edges[:-1], edges[1:]
    n_bins = len(left)
    width = (right - left).days.to_numpy().astype(float)

    # Ricerca binaria su interi: pd.cut fallisce quando i bordi e la colonna
    # Time hanno risoluzioni datetime64 diverse (ns contro us), evenienza
    # normale mescolando date_range e dati esterni.
    t = catalog["Time"].to_numpy(dtype="datetime64[ns]").astype("int64")
    e = edges.to_numpy(dtype="datetime64[ns]").astype("int64")
    slot = np.searchsorted(e, t, side="right") - 1
    inside = (slot >= 0) & (slot < n_bins)

    slot = slot[inside]
    mag = catalog["Magnitude"].to_numpy(dtype=float)[inside]
    energy = energy_from_magnitude(mag)

    n = np.bincount(slot, minlength=n_bins)
    S = np.bincount(slot, weights=energy, minlength=n_bins)

    m_max = np.full(n_bins, np.nan)
    e_max = np.full(n_bins, np.nan)
    if slot.size:
        order = np.lexsort((energy, slot))            # per bin, energia crescente
        s_sorted = slot[order]
        last = np.r_[s_sorted[1:] != s_sorted[:-1], True]
        m_max[s_sorted[last]] = mag[order][last]
        e_max[s_sorted[last]] = energy[order][last]

    S_eff = S * (width.mean() / width) if normalize_exposure else S

    # Modello di bias: Censura degli intervalli vuoti (Proposizione 5.1)
    e_min = float(energy_from_magnitude(mc))
    censored = n == 0
    S_hat = np.where(censored, e_min, S_eff)
    y = np.log10(S_hat)

    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = np.where(S > 0, e_max / S, np.nan)

    return pd.DataFrame(
        {
            "t_end": right,
            "width_days": width,
            "n": n,
            "censored": censored,
            "m_max": m_max,
            "S": S,
            "S_hat": S_hat,
            "y": y,
            "m_eq": (y - ALPHA) / BETA,
            "kappa": kappa,
            "r": np.log10(1.0 + n),
        },
        index=pd.DatetimeIndex(left, name="t_start"),
    )


# ---------------------------------------------------------------------------
# Differenziazione
# ---------------------------------------------------------------------------

def _local_slope(
    t_days: np.ndarray,
    y: np.ndarray,
    usable: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Pendenza locale di y, minimi quadrati su finestra centrata di `window` bin.

    I bin censurati sono esclusi dal fit: appoggiati sul pavimento
    produrrebbero pendenze fittizie. NaN dove i punti utili sono meno di tre.
    """
    n = len(y)
    half = window // 2
    out = np.full(n, np.nan)

    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        sel = usable[lo:hi]
        if sel.sum() < 3:
            continue
        x = t_days[lo:hi][sel]
        if np.ptp(x) == 0:
            continue
        out[i] = np.polyfit(x - x.mean(), y[lo:hi][sel], 1)[0]

    return out


def differentiate(binned: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Statistiche differenziali sulle transizioni fra bin consecutivi.

    Colonne restituite, indicizzate sul bordo sinistro del bin di arrivo:

      h              altezza y_k - y_{k-1} [dex]
      dm_eq          h / BETA, in unita' di magnitudo equivalente
      dr             variazione del canale di tasso
      censored_edge  True se uno dei due bin e' censurato
      slope          pendenza locale [dex/giorno]
      slope_m_yr     pendenza in magnitudo equivalente per anno

    Segno: dm_eq > 0 significa che il bin ha rilasciato piu' energia del
    precedente (intensificazione), dm_eq < 0 il contrario (attenuazione). Il
    catalogo misura energia rilasciata, non deformazione accumulata: la
    seconda non e' osservabile dai sismogrammi.
    """
    if binned.empty or len(binned) < 2:
        return pd.DataFrame()

    y = binned["y"].to_numpy()
    r = binned["r"].to_numpy()
    cens = binned["censored"].to_numpy()
    starts = binned.index
    t_days = (starts - starts[0]).days.to_numpy().astype(float)

    slope = _local_slope(t_days, y, ~cens, window)
    h = np.diff(y)

    return pd.DataFrame(
        {
            "h": h,
            "dm_eq": h / BETA,
            "dr": np.diff(r),
            "censored_edge": cens[1:] | cens[:-1],
            "slope": slope[1:],
            "slope_m_yr": slope[1:] / BETA * 365.25,
        },
        index=starts[1:],
    )


# ---------------------------------------------------------------------------
# Aggregati
# ---------------------------------------------------------------------------

def summarize(diff: pd.DataFrame, exclude_censored: bool = False) -> dict[str, float]:
    """
    Aggregati sulle transizioni, in unita' di magnitudo equivalente.

    Tutti gli aggregati riportati sotto `total_variation`, `positive_part`,
    `negative_part` e `rms` sono non lineari in h, quindi sensibili al percorso.

    `net_m` e' invece lineare e telescopizza: vale y_K - y_0 diviso BETA, cioe'
    dipende unicamente dal primo e dall'ultimo bin. Viene restituito con quel
    nome esplicito perche' e' comunque una quantita' legittima, purche' letta
    per quello che e': una differenza fra due estremi, non un bilancio.
    """
    if diff.empty:
        return {}

    d = diff.loc[~diff["censored_edge"]] if exclude_censored else diff
    if d.empty:
        return {}

    dm = d["dm_eq"].to_numpy()

    return {
        # --- non lineari: informativi sul percorso ---
        "total_variation": float(np.abs(dm).sum()),
        "positive_part": float(dm[dm > 0].sum()),
        "negative_part": float(dm[dm < 0].sum()),
        "rms": float(np.sqrt(np.mean(dm ** 2))),
        "max_jump": float(dm.max()),
        "min_jump": float(dm.min()),
        # --- lineare: dipende solo dagli estremi ---
        "net_m": float(dm.sum()),
        # --- diagnostica ---
        "n_transitions": int(len(d)),
        "n_censored_edges": int(diff["censored_edge"].sum()),
    }


def annual_matrix(binned: pd.DataFrame, column: str = "m_eq") -> pd.DataFrame:
    """
    Riorganizza una colonna in matrice anno x mese.

    Sostituisce l'overlay cumulato della versione precedente: confrontare un
    anno con gli altri e' una domanda legittima, ma va posta sulla grandezza
    osservata (la magnitudo equivalente mensile) e non sulla sua somma
    cumulata, che telescopizza e riproduce la grandezza stessa riscalata.
    """
    if binned.empty:
        return pd.DataFrame()
    df = binned.reset_index()
    return df.pivot_table(
        index=df["t_start"].dt.year,
        columns=df["t_start"].dt.month,
        values=column,
        aggfunc="mean",
    ).rename_axis(index="anno", columns="mese")


# ---------------------------------------------------------------------------
# Orchestrazione
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """Esito completo di un'analisi: dati piu' diagnostica."""
    catalog: pd.DataFrame
    binned: pd.DataFrame
    diff: pd.DataFrame
    summary: dict[str, float]
    config: Config
    mag_type_mix: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        if self.binned.empty:
            return None
        return self.binned.index[0], self.binned["t_end"].iloc[-1]

    @property
    def concentration(self) -> float:
        """
        Frazione dell'energia totale dovuta ai soli eventi massimi di ciascun
        bin. Pesata sull'energia e non mediana: la mediana e' dominata dai bin
        tranquilli, mentre la questione riguarda quanto la serie sia guidata
        dai bin forti.
        """
        if self.binned.empty:
            return float("nan")
        S = self.binned["S"].to_numpy()
        kappa = self.binned["kappa"].to_numpy()
        mask = np.isfinite(kappa) & (S > 0)
        if not mask.any():
            return float("nan")
        return float(np.sum(kappa[mask] * S[mask]) / np.sum(S[mask]))


def run(config: Config, catalog: pd.DataFrame | None = None) -> AnalysisResult:
    """
    Esegue l'analisi completa.

    Se `catalog` e' fornito si salta il fetch, il che rende la pipeline
    testabile su cataloghi sintetici senza toccare la rete.
    """
    notes: list[str] = []

    if catalog is None:
        catalog = fetch_catalog(
            region=config.region,
            start=config.start,
            end=config.requested_end,
            mc=config.mc,
            timeout=config.timeout,
        )

    mix: dict[str, int] = {}
    if "MagType" in catalog.columns:
        mix = catalog["MagType"].astype(str).str.strip().value_counts().to_dict()
        if len(mix) > 1:
            notes.append(f"Tipi di magnitudo eterogenei {sorted(mix)}. {MAG_TYPE_NOTE}")

    if config.mag_types is not None:
        before = len(catalog)
        catalog = filter_magnitude_types(catalog, config.mag_types)
        if catalog.empty:
            raise EmptyCatalogError(f"nessun evento con MagType in {config.mag_types}")
        notes.append(f"Filtro sul tipo di magnitudo: {before} -> {len(catalog)} eventi.")

    edges = build_edges(config.start, config.requested_end,
                        catalog["Time"].max(), config.binning)

    if len(edges) < 3:
        notes.append("Finestra insufficiente: servono almeno due bin completi.")
        return AnalysisResult(catalog, pd.DataFrame(), pd.DataFrame(), {},
                              config, mix, notes)

    binned = aggregate(catalog, edges, config.mc, config.normalize_exposure)
    diff = differentiate(binned, window=config.window)
    summary = summarize(diff)

    n_cens = int(binned["censored"].sum())
    if n_cens:
        notes.append(
            f"{n_cens} bin vuoti su {len(binned)}: energia imputata al pavimento "
            f"E_min = E({config.mc}). Le altezze adiacenti misurano una distanza "
            "assoluta dalla soglia, non una variazione fra due stati osservati."
        )

    kappa_w = _weighted_concentration(binned)
    if np.isfinite(kappa_w) and kappa_w > 0.8:
        notes.append(
            f"Concentrazione {kappa_w:.1%}: la serie energetica traccia di fatto "
            "la magnitudo massima del bin. Per il tasso di attivita' si legga il "
            "canale dei conteggi."
        )

    if config.binning.kind == "calendar" and not config.normalize_exposure:
        notes.append(
            "Griglia calendariale senza normalizzazione dell'esposizione: un "
            "febbraio raccoglie il 10% di giorni in meno di un gennaio, con un "
            "bias sistematico di circa 0,03 unita' di magnitudo equivalente."
        )

    return AnalysisResult(catalog, binned, diff, summary, config, mix, notes)


def _weighted_concentration(binned: pd.DataFrame) -> float:
    if binned.empty:
        return float("nan")
    S = binned["S"].to_numpy()
    kappa = binned["kappa"].to_numpy()
    mask = np.isfinite(kappa) & (S > 0)
    if not mask.any():
        return float("nan")
    return float(np.sum(kappa[mask] * S[mask]) / np.sum(S[mask]))
