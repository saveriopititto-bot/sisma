"""
math_engine
===========

Motore di calcolo per l'analisi differenziale del rilascio di energia sismica.
Puro: nessuna dipendenza da Streamlit, eseguibile e testabile in isolamento.
"""

from math_engine.engine import (
    ALPHA,
    BETA,
    MAG_TYPE_NOTE,
    AnalysisResult,
    Binning,
    CatalogError,
    Config,
    EmptyCatalogError,
    Region,
    aggregate,
    annual_matrix,
    build_edges,
    differentiate,
    energy_from_magnitude,
    fetch_catalog,
    filter_magnitude_types,
    magnitude_from_energy,
    run,
    summarize,
)
from math_engine.signature import (
    SignatureSummary,
    chen_product,
    levy_area,
    path_signature,
    rolling_levy_area,
    segment_signature,
    signature_features,
    summarize_signature,
)
from math_engine.synthetic import synthetic_catalog

__all__ = [
    "ALPHA", "BETA", "MAG_TYPE_NOTE",
    "AnalysisResult", "Binning", "CatalogError", "Config",
    "EmptyCatalogError", "Region",
    "aggregate", "annual_matrix", "build_edges", "differentiate",
    "energy_from_magnitude", "fetch_catalog", "filter_magnitude_types",
    "magnitude_from_energy", "run", "summarize", "synthetic_catalog",
    "SignatureSummary", "chen_product", "levy_area", "path_signature",
    "rolling_levy_area", "segment_signature", "signature_features",
    "summarize_signature",
]
