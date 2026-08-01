"""
app.py
======

Punto di ingresso Streamlit.

Il file non contiene matematica ne' grafica: raccoglie gli input, invoca il
motore e distribuisce il risultato ai componenti. La cache vive qui perche' e'
una preoccupazione dell'interfaccia, non del calcolo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from math_engine import (
    CatalogError,
    Config,
    EmptyCatalogError,
    Region,
    fetch_catalog,
    run,
    synthetic_catalog,
)
from ui.components import (
    render_annual_comparison,
    render_channels,
    render_diagnostics,
    render_differentials,
    render_events,
    render_sidebar,
    render_summary,
)

st.set_page_config(page_title="Sismologia Computazionale Analitica", layout="wide")


def load_css(path: str) -> None:
    file = Path(path)
    if not file.exists():
        return
    st.markdown(f"<style>{file.read_text()}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner="Interrogazione del web service FDSNWS in corso...")
def cached_fetch(
    min_lat: float, max_lat: float, min_lon: float, max_lon: float,
    start: pd.Timestamp, end: pd.Timestamp, mc: float, timeout: int,
) -> pd.DataFrame:
    """
    Wrapper cacheabile di `fetch_catalog`.

    La firma usa solo tipi primitivi: st.cache_data calcola la chiave dagli
    argomenti, e passare una dataclass funzionerebbe solo finche' resta
    hashable, dipendenza fragile che conviene non introdurre.
    """
    return fetch_catalog(
        region=Region(min_lat, max_lat, min_lon, max_lon),
        start=start, end=end, mc=mc, timeout=timeout,
    )


def analyze(config: Config, synthetic: bool):
    """Esegue l'analisi traducendo le eccezioni in messaggi. None se fallisce."""
    try:
        if synthetic:
            catalog = synthetic_catalog(mc=config.mc)
            st.warning(
                "Analisi su catalogo sintetico: i dati sono generati localmente "
                "e non rappresentano sismicita' reale.", icon="⚠️",
            )
        else:
            catalog = cached_fetch(
                config.region.min_lat, config.region.max_lat,
                config.region.min_lon, config.region.max_lon,
                config.start, config.requested_end, config.mc, config.timeout,
            )
        return run(config, catalog=catalog)
    except EmptyCatalogError as exc:
        st.warning(f"Nessun evento utilizzabile: {exc}")
    except CatalogError as exc:
        st.error(f"Impossibile ottenere il catalogo: {exc}")
    return None


def main() -> None:
    load_css("ui/style.css")

    st.title("Sismologia Computazionale Analitica")
    st.markdown(
        "Analisi differenziale del **rilascio** di energia sismica a partire dai "
        "cataloghi INGV. Il modello confronta un anno di verifica con "
        "l'inviluppo degli anni di riferimento, su due canali indipendenti: "
        "l'energia rilasciata e il tasso di attivita'."
    )

    config, execute, synthetic = render_sidebar()

    if execute:
        result = analyze(config, synthetic)
        if result is not None and not result.binned.empty:
            st.session_state["result"] = result
        elif result is not None:
            st.warning("Finestra troppo corta: nessun bin completo da analizzare.")

    result = st.session_state.get("result")
    if result is None:
        st.info(
            "Imposta i parametri nella barra laterale e avvia l'analisi. "
            "Senza connessione all'INGV puoi spuntare *Usa catalogo sintetico* "
            "per esplorare l'interfaccia.", icon="👈",
        )
        return

    render_diagnostics(result)
    render_summary(result)
    render_events(result)
    render_channels(result)
    render_differentials(result)
    render_annual_comparison(result)

    with st.expander("Dati per bin"):
        st.dataframe(result.binned, use_container_width=True)
    with st.expander("Dati per transizione"):
        st.dataframe(result.diff, use_container_width=True)


if __name__ == "__main__":
    main()
