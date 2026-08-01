"""
ui.components
=============

Livello di presentazione: input, diagnostica e grafici Plotly.

Non contiene matematica. Ogni grandezza mostrata viene da `math_engine`, che
resta eseguibile senza Streamlit.

Convenzione di segno, unica in tutta l'applicazione:

    dm_eq > 0  ->  intensificazione: il bin ha rilasciato piu' del precedente
    dm_eq < 0  ->  attenuazione:     il bin ha rilasciato meno del precedente

Il catalogo misura energia rilasciata. La deformazione elastica accumulata e'
caricata dal moto tettonico e non e' osservabile dai sismogrammi: un bin
quieto e un bin di caricamento sono indistinguibili. Per questo nell'interfaccia
non compaiono le parole "accumulo" e "rilascio" come lettura del segno.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from math_engine import ALPHA, BETA, AnalysisResult, Binning, Config, Region, annual_matrix

__all__ = [
    "render_sidebar", "render_diagnostics", "render_summary",
    "render_events", "render_channels", "render_differentials",
    "render_annual_comparison",
]

POS = "#2E7D32"      # intensificazione
NEG = "#C62828"      # attenuazione
ACC = "#1F5C8B"
GREY = "rgba(0,0,0,0.45)"

FAULTS = {
    "Manuale (coordinate libere)": None,
    "Faglia di Gubbio (ITCS001)": Region(43.20, 43.45, 12.45, 12.75),
    "Faglia del Monte Vettore": Region(42.70, 42.95, 13.10, 13.40),
    "Faglia di Paganica (L'Aquila)": Region(42.25, 42.45, 13.30, 13.60),
}


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[Config, bool, bool]:
    """
    Raccoglie i parametri. Restituisce (config, esegui, usa_dati_sintetici).
    """
    st.sidebar.header("Area sismogenetica")

    name = st.sidebar.selectbox("Faglia catalogata", list(FAULTS))
    preset = FAULTS[name]
    manual = preset is None

    st.sidebar.markdown("**Bounding box**")
    min_lat = st.sidebar.number_input("Lat min", value=42.80 if manual else preset.min_lat,
                                      format="%.2f", disabled=not manual)
    max_lat = st.sidebar.number_input("Lat max", value=43.50 if manual else preset.max_lat,
                                      format="%.2f", disabled=not manual)
    min_lon = st.sidebar.number_input("Lon min", value=12.50 if manual else preset.min_lon,
                                      format="%.2f", disabled=not manual)
    max_lon = st.sidebar.number_input("Lon max", value=13.50 if manual else preset.max_lon,
                                      format="%.2f", disabled=not manual)

    st.sidebar.divider()
    st.sidebar.header("Finestra temporale")

    mc = st.sidebar.number_input(
        "Soglia di completezza M_c", value=1.5, step=0.1, format="%.1f",
        help="Applicata lato servizio. Determina anche il pavimento energetico "
             "dei bin vuoti, quindi analisi con M_c diverse non sono confrontabili.",
    )
    year_zero = st.sidebar.number_input("Anno centrale", 1900, 2100, 2014, 1)
    span = st.sidebar.slider("Semiampiezza [anni]", 1, 15, 5)
    window_start, window_end = year_zero - span, year_zero + span
    target_year = window_end + 1

    st.sidebar.caption(
        f"Riferimento {window_start}–{window_end} · centro {year_zero} · "
        f"verifica {target_year}"
    )

    st.sidebar.divider()
    st.sidebar.header("Griglia e stima")

    kind = st.sidebar.radio(
        "Partizione temporale", ["calendar", "uniform"],
        format_func=lambda k: {"calendar": "Mesi solari",
                               "uniform": "Ampiezza fissa"}[k],
        help="I mesi solari durano da 28 a 31 giorni. Con ampiezza fissa "
             "l'esposizione e' costante per costruzione.",
    )
    width = st.sidebar.number_input("Ampiezza [giorni]", 7, 90, 30,
                                    disabled=(kind == "calendar"))
    normalize = st.sidebar.checkbox(
        "Normalizza l'esposizione", value=(kind == "calendar"),
        help="Riporta l'energia a un bin di lunghezza media. Corregge il fatto "
             "che febbraio raccoglie il 10% di giorni in meno di gennaio.",
    )
    smoothing = st.sidebar.slider("Finestra per la derivata locale [bin]", 3, 15, 7, 2)

    st.sidebar.divider()
    synthetic = st.sidebar.checkbox(
        "Usa catalogo sintetico", value=False,
        help="Esegue l'analisi su dati generati localmente, senza interrogare "
             "l'INGV. Utile per esplorare l'interfaccia offline.",
    )
    execute = st.sidebar.button("Esegui analisi", type="primary")

    config = Config(
        region=Region(min_lat, max_lat, min_lon, max_lon),
        mc=float(mc),
        start=pd.Timestamp(year=int(window_start) - 1, month=12, day=1),
        target_year=int(target_year),
        binning=Binning(kind=kind, width_days=int(width)),
        window=int(smoothing),
        normalize_exposure=bool(normalize),
    )
    return config, execute, synthetic


# ---------------------------------------------------------------------------
# Diagnostica
# ---------------------------------------------------------------------------

def render_diagnostics(result: AnalysisResult) -> None:
    """Copertura effettiva, note del motore, composizione delle magnitudo."""
    coverage = result.coverage
    if coverage:
        a, b = coverage
        st.caption(
            f"Copertura {a:%Y-%m-%d} → {b:%Y-%m-%d} · {len(result.binned)} bin · "
            f"{len(result.catalog)} eventi · M ≥ {result.config.mc}"
        )

    for note in result.notes:
        st.info(note, icon="ℹ️")

    if len(result.mag_type_mix) > 1:
        with st.expander("Composizione per tipo di magnitudo"):
            st.dataframe(
                pd.Series(result.mag_type_mix, name="eventi").sort_values(ascending=False),
                use_container_width=True,
            )


def render_summary(result: AnalysisResult) -> None:
    """Metriche di sintesi, separando gli aggregati non lineari da quelli lineari."""
    s = result.summary
    if not s:
        return

    st.subheader("Sintesi delle Metriche Non Lineari", divider="gray")

    tot_var = s.get("total_variation", 0.0)
    pos_part = s.get("positive_part", 0.0)
    rms_val = s.get("rms", 0.0)
    max_j = s.get("max_jump", 0.0)
    net_m_val = s.get("net_m", 0.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Variazione totale (V)", f"{tot_var:.2f} M",
              help="Somma dei valori assoluti delle altezze differenziali. "
                   "Non lineare, quindi distingue percorsi diversi anche a "
                   "parita' di estremi.")
    c2.metric("Intensificazione accumulata (V+)", f"{pos_part:+.2f} M",
              help="Somma delle altezze differenziali positive (solo intensificazioni).")
    c3.metric("RMS delle variazioni (ρ)", f"{rms_val:.2f} M",
              help="Radice media quadratica delle variazioni differenziali.")
    c4.metric("Salto massimo", f"{max_j:+.2f} M",
              help="Massima intensificazione fra due bin consecutivi.")

    with st.expander("Differenza fra gli estremi (lineare) e concentrazione"):
        cc1, cc2 = st.columns(2)
        cc1.metric("Concentrazione", f"{result.concentration:.1%}",
                   help="Frazione dell'energia totale dovuta ai soli eventi massimi "
                        "di ciascun bin.")
        cc2.metric("Δ magnitudo equivalente, primo → ultimo bin", f"{net_m_val:+.2f} M",
                   help="La somma con segno telescopizza: vale la differenza "
                        "fra l'ultimo e il primo bin.")


# ---------------------------------------------------------------------------
# Grafici
# ---------------------------------------------------------------------------

def _target_marker(fig: go.Figure, target_year: int) -> None:
    fig.add_vline(
        x=pd.Timestamp(f"{target_year}-01-01"), line_dash="dash",
        line_color=GREY, line_width=2,
        annotation_text="  inizio anno di verifica", annotation_position="top right",
    )


def render_events(result: AnalysisResult) -> None:
    """Eventi registrati. L'area del marcatore e' proporzionale a E^(1/3)."""
    cat = result.catalog
    if cat.empty:
        return

    st.subheader("Sismicita' registrata", divider="gray")

    # size ~ 10^(BETA*M/3): l'area cresce come la radice cubica dell'energia.
    # Usare M^2, come nella versione precedente, sotto-rappresentava di molto
    # gli eventi forti rispetto al loro peso energetico effettivo.
    size = 4 + 26 * (10 ** (BETA * (cat["Magnitude"] - cat["Magnitude"].max()) / 3))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cat["Time"], y=cat["Magnitude"], mode="markers",
        marker=dict(size=size, color=cat["Magnitude"], colorscale="Viridis",
                    showscale=True, colorbar=dict(title="M"), line=dict(width=0)),
        customdata=np.stack([cat["Magnitude"]], axis=-1),
        hovertemplate="M %{customdata[0]:.1f}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
        name="eventi",
    ))
    _target_marker(fig, result.config.target_year)
    fig.update_layout(template="plotly_white", showlegend=False, height=380,
                      xaxis_title="tempo", yaxis_title="magnitudo",
                      margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)


def render_channels(result: AnalysisResult) -> None:
    """
    I due canali indipendenti: energia e tasso.

    Sono separati perche' misurano cose diverse. L'energia mensile e' quasi
    interamente dovuta all'evento piu' forte, quindi il canale energetico e'
    cieco a una sciamatura di microeventi, che il canale dei conteggi vede.
    """
    b = result.binned
    if b.empty:
        return

    st.subheader("Canale energetico", divider="gray")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=b.index, y=b["m_eq"], mode="lines+markers", name="M equivalente",
        line=dict(color=ACC, width=2), marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=b.index, y=b["m_max"], mode="markers", name="M massima osservata",
        marker=dict(size=5, color=POS, symbol="diamond-open"),
    ))
    fig.add_hline(y=result.config.mc, line_dash="dot", line_color=NEG,
                  annotation_text=f"  soglia M_c = {result.config.mc}",
                  annotation_position="bottom right")

    cens = b[b["censored"]]
    if not cens.empty:
        fig.add_trace(go.Scatter(
            x=cens.index, y=cens["m_eq"], mode="markers", name="bin vuoti (imputati)",
            marker=dict(size=11, color="rgba(0,0,0,0)",
                        line=dict(color=NEG, width=1.6)),
        ))

    _target_marker(fig, result.config.target_year)
    fig.update_layout(template="plotly_white", height=360,
                      xaxis_title="tempo", yaxis_title="magnitudo equivalente",
                      margin=dict(l=0, r=0, t=30, b=0),
                      legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Le due serie quasi coincidono: e' la dominanza dell'evento massimo. "
        "Poiche' l'energia cresce come 10^(1.5·M) mentre gli eventi si "
        "diradano come 10^(−M), la somma e' controllata dalla scossa piu' forte."
    )

    st.subheader("Canale di tasso", divider="gray")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=b.index, y=b["n"], marker_color=ACC, name="eventi"))
    _target_marker(fig2, result.config.target_year)
    fig2.update_layout(template="plotly_white", height=260, showlegend=False,
                       xaxis_title="tempo", yaxis_title="eventi per bin",
                       margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Informazione ortogonale alla precedente: uno sciame di microeventi "
        "e' visibile qui e invisibile nel canale energetico."
    )


def render_differentials(result: AnalysisResult) -> None:
    """Altezze differenziali, filtro di pendenza locale e modello di bias."""
    d = result.diff
    if d.empty:
        return

    win = result.config.window
    st.subheader("Analisi differenziale e filtro di riduzione del rumore", divider="gray")
    st.markdown(
        "Confronto tra la **differenza a un passo** ($\\Delta M_{\\mathrm{eq}}$) e il "
        f"**filtro di pendenza locale su finestra scorrevole** ($w={win}$) "
        "per l'attenuazione del rumore stocastico."
    )

    colors = np.where(d["dm_eq"] > 0, POS, NEG)
    opacity = np.where(d["censored_edge"], 0.35, 0.9)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d.index, y=d["dm_eq"],
        marker=dict(color=colors, opacity=opacity),
        customdata=np.stack([d["censored_edge"]], axis=-1),
        hovertemplate=("%{x|%Y-%m}<br>Δ %{y:+.2f} M"
                       "<br>bin vuoto adiacente: %{customdata[0]}<extra></extra>"),
        name="Differenza a un passo (ΔM_eq)",
    ))

    # Overlay del filtro di pendenza locale
    if d["slope_m_yr"].notna().any():
        slope_per_bin = d["slope"] / BETA
        fig.add_trace(go.Scatter(
            x=d.index, y=slope_per_bin, mode="lines",
            line=dict(color=ACC, width=3),
            name=f"Filtro pendenza locale (finestra w={win})",
        ))

    _target_marker(fig, result.config.target_year)
    fig.update_layout(template="plotly_white", height=380,
                      xaxis_title="tempo",
                      yaxis_title="Variazione per intervallo [M]",
                      margin=dict(l=0, r=0, t=30, b=0),
                      legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Verde: intensificazione (Δ > 0). Rosso: attenuazione (Δ < 0). "
        "Le barre sbiadite coinvolgono un bin vuoto censurato dal modello di bias. "
        f"La linea blu rappresenta il **filtro di pendenza locale** su finestra $w={win}$, "
        "che esclude i bin censurati per ridurre il rumore stocastico."
    )

    if d["slope_m_yr"].notna().any():
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=d.index, y=d["slope_m_yr"], mode="lines",
            line=dict(color=ACC, width=2), name="pendenza annua",
        ))
        fig3.add_hline(y=0, line_color=GREY, line_width=1)
        _target_marker(fig3, result.config.target_year)
        fig3.update_layout(
            template="plotly_white", height=260, showlegend=False,
            xaxis_title="tempo",
            yaxis_title=f"M equivalente / anno (finestra {win} bin)",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Modello di bias e note metodologiche"):
        st.markdown(
            r"""
- **Modello di bias della censura (Proposizione 5.1)**: nei bin vuoti l'energia è imputata al pavimento $\hat S_k = \max(S_k, E(M_c)) = E(M_c)$. Questo evita $\log 0$ ma colloca i bin vuoti al livello di fondo $M_c$. Le transizioni da/verso bin vuoti misurano la distanza dalla soglia di completezza e non una variazione sismica fisica.
- **Filtro su finestra scorrevole (Sezione 7.3)**: la regressione ai minimi quadrati su $w$ bin riduce il rumore stocastico di singolo evento ed esclude dal fit i bin censurati per evitare pendenze fittizie.
- **Bias di esposizione**: nella partizione calendariale i mesi durano da 28 a 31 giorni (febbraio raccoglie il 10% in meno di giorni rispetto a gennaio). L'opzione *Normalizza l'esposizione* rimuove questo bias.
- **Bias della magnitudo**: la relazione di Gutenberg-Richter $10^{1.5M+4.8}$ è tarata su $M_s$, mentre il catalogo INGV usa prevalentemente $M_L$ e $M_w$.
"""
        )


def render_annual_comparison(result: AnalysisResult) -> None:
    """
    Confronto dell'anno di verifica con l'inviluppo degli anni di riferimento.

    Mostra il rilevamento dell'anno di verifica rispetto alla previsione del
    modello (mediana e inviluppo storico 10–90%) su due canali:
    1. Canale Energetico (Magnitudo equivalente)
    2. Canale di Tasso (Numero di eventi per mese)
    """
    matrix = annual_matrix(result.binned, "m_eq")
    if matrix.empty or len(matrix) < 2:
        return

    st.subheader("Confronto annuale e previsione del modello", divider="gray")

    target = result.config.target_year
    reference = matrix.drop(index=target, errors="ignore")
    if reference.empty:
        return

    months = list(matrix.columns)
    labels = ["gen", "feb", "mar", "apr", "mag", "giu",
              "lug", "ago", "set", "ott", "nov", "dic"]
    x = [labels[m - 1] for m in months]

    # --- Grafico 1: Canale Energetico ---
    st.markdown("##### Canale energetico (Magnitudo equivalente)")
    lo = reference.quantile(0.10)
    hi = reference.quantile(0.90)
    med = reference.median()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(31,92,139,0.15)",
        name="inviluppo 10–90% anni di riferimento", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=med, mode="lines+markers", name="previsione del modello (mediana di riferimento)",
        line=dict(color=ACC, width=3, dash="dashdot"),
    ))

    if target in matrix.index:
        fig.add_trace(go.Scatter(
            x=x, y=matrix.loc[target], mode="lines+markers",
            name=f"rilevamento dell'anno {target}",
            line=dict(color=NEG, width=3),
        ))

    fig.update_layout(
        template="plotly_white", height=350, hovermode="x unified",
        xaxis_title="mese", yaxis_title="magnitudo equivalente",
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.14, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Grafico 2: Canale di Tasso ---
    matrix_n = annual_matrix(result.binned, "n")
    if not matrix_n.empty and len(matrix_n) >= 2:
        reference_n = matrix_n.drop(index=target, errors="ignore")
        if not reference_n.empty:
            st.markdown("##### Canale di tasso (Numero di eventi per mese)")
            lo_n = reference_n.quantile(0.10)
            hi_n = reference_n.quantile(0.90)
            med_n = reference_n.median()

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=x, y=hi_n, mode="lines", line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig2.add_trace(go.Scatter(
                x=x, y=lo_n, mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor="rgba(31,92,139,0.15)",
                name="inviluppo 10–90% anni di riferimento", hoverinfo="skip",
            ))
            fig2.add_trace(go.Scatter(
                x=x, y=med_n, mode="lines+markers", name="previsione del modello (mediana di riferimento)",
                line=dict(color=ACC, width=3, dash="dashdot"),
            ))

            if target in matrix_n.index:
                fig2.add_trace(go.Scatter(
                    x=x, y=matrix_n.loc[target], mode="lines+markers",
                    name=f"rilevamento dell'anno {target}",
                    line=dict(color=NEG, width=3),
                ))

            fig2.update_layout(
                template="plotly_white", height=350, hovermode="x unified",
                xaxis_title="mese", yaxis_title="eventi per bin",
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", y=1.14, x=0),
            )
            st.plotly_chart(fig2, use_container_width=True)

    min_yr = reference.index.min()
    max_yr = reference.index.max()
    st.caption(
        f"Ogni mese del **rilevamento dell'anno di verifica {target}** e' confrontato "
        f"con la **previsione del modello** (mediana e inviluppo 10–90% degli anni di riferimento "
        f"{min_yr}–{max_yr}) sui due canali: energia rilasciata e tasso di attivita'."
    )

    with st.expander("Matrice anno × mese (Magnitudo equivalente)"):
        st.dataframe(matrix.round(2), use_container_width=True)
    if not matrix_n.empty:
        with st.expander("Matrice anno × mese (Numero di eventi)"):
            st.dataframe(matrix_n.round(0), use_container_width=True)
