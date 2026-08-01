"""
Pagina di documentazione interattiva del modello.

Riscritta per eliminare la contraddizione di segno presente nella versione
precedente, dove la legenda del grafico dichiarava "Rilascio > 0" mentre
questa pagina affermava l'opposto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from math_engine import ALPHA, BETA

st.set_page_config(page_title="Come funziona il modello", page_icon="🔍", layout="wide")

st.title("🔍 Come funziona il modello")
st.markdown(
    "Questa pagina espone la matematica del modello e i suoi limiti. Se hai "
    "appena eseguito un'analisi, le tabelle mostrano i tuoi dati reali."
)

result = st.session_state.get("result")

tabs = st.tabs([
    "1 · Dati grezzi",
    "2 · Energia",
    "3 · Censura",
    "4 · Analisi differenziale",
    "5 · Limiti",
])

# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("Dati grezzi")
    st.markdown(
        "Gli eventi vengono scaricati dal web service **FDSNWS** dell'INGV, "
        "filtrati per rettangolo geografico, finestra temporale e soglia di "
        "magnitudo $M_c$."
    )
    st.markdown(
        "La soglia non e' solo un filtro di comodo: garantisce che ogni evento "
        "del catalogo soddisfi $M_i \\ge M_c$, ipotesi da cui dipende il "
        "comportamento della censura descritto nella scheda 3."
    )
    if result is not None:
        st.dataframe(result.catalog.head(100), use_container_width=True)
    else:
        st.info("Nessun dato in memoria. Esegui un'analisi nella pagina principale.")

# ---------------------------------------------------------------------------
with tabs[1]:
    st.header("Dalla magnitudo all'energia")
    st.latex(r"\log_{10} E = \beta M + \alpha, \qquad \beta = 1.5,\ \alpha = 4.8")
    st.markdown(
        "Relazione energia–magnitudo di Gutenberg e Richter (1956), con $E$ in "
        "joule. Ogni unita' di magnitudo moltiplica l'energia per "
        "$10^{1.5} \\approx 31.6$."
    )
    st.warning(
        "Da non confondere con la legge frequenza–magnitudo "
        "$\\log_{10} N(\\ge M) = a - bM$, che porta lo stesso nome ma e' una "
        "legge distinta. I due parametri $a, b$ non hanno nulla a che vedere "
        "con $\\alpha, \\beta$.",
        icon="⚠️",
    )

    m = np.linspace(1.0, 7.0, 200)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m, y=10 ** (BETA * m + ALPHA), mode="lines",
                             line=dict(color="#1F5C8B", width=3), name="E(M)"))
    fig.update_layout(template="plotly_white", height=340, showlegend=False,
                      xaxis_title="magnitudo M", yaxis_title="energia [J]",
                      yaxis_type="log", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "L'energia di un bin e' la somma su tutti i suoi eventi. Poiche' "
        "l'energia cresce come $10^{1.5M}$ mentre il numero di eventi decresce "
        "come $10^{-M}$, la somma e' dominata dalla scossa piu' forte: la serie "
        "energetica e', con ottima approssimazione, la serie della magnitudo "
        "massima del bin."
    )
    if result is not None and not result.binned.empty:
        st.metric("Concentrazione sull'evento massimo",
                  f"{result.concentration:.1%}",
                  help="Frazione dell'energia totale dovuta ai soli eventi "
                       "massimi di ciascun bin.")
        st.dataframe(
            result.binned[["n", "m_max", "m_eq", "kappa"]].head(50).round(3),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
with tabs[2]:
    st.header("Censura dei bin vuoti")
    st.markdown(
        "Un bin senza eventi ha energia nulla, e $\\log_{10} 0$ non esiste. "
        "Si applica quindi un pavimento:"
    )
    st.latex(r"\hat S_k = \max\{S_k,\ E_{\min}\}, \qquad E_{\min} = E(M_c)")
    st.markdown(
        "**Il pavimento agisce solo sui bin vuoti.** Se un bin contiene almeno "
        "un evento, quell'evento soddisfa $M \\ge M_c$ e da solo contribuisce "
        "gia' $E_{\\min}$: la somma e' quindi sempre sopra il pavimento. "
        "L'operazione equivale a imputare, nei bin vuoti, un singolo evento di "
        "magnitudo esattamente $M_c$."
    )
    st.error(
        "Conseguenza pratica: il livello di fondo della serie e' $M_c$. "
        "Abbassare la soglia di una unita' abbassa il fondo di una unita' e "
        "amplifica di altrettanto ogni escursione che parte da un bin vuoto. "
        "**Analisi condotte con soglie diverse non sono confrontabili.**",
        icon="🚫",
    )
    if result is not None and not result.binned.empty:
        n_cens = int(result.binned["censored"].sum())
        st.metric("Bin censurati", f"{n_cens} su {len(result.binned)}")

# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("Analisi differenziale")
    st.markdown("Detta $y_k = \\log_{10}\\hat S_k$ la serie log-energetica, si calcola")
    st.latex(r"h_k = y_k - y_{k-1}, \qquad \Delta M^{\mathrm{eq}}_k = \frac{h_k}{\beta}")
    st.markdown(
        "L'altezza viene espressa in unita' di **magnitudo equivalente**, "
        "leggibile direttamente: $\\Delta M^{\\mathrm{eq}} = +1$ significa che il "
        "bin ha rilasciato tanta energia quanto ne rilascerebbe un evento di una "
        "unita' di magnitudo superiore rispetto al bin precedente."
    )

    st.subheader("Convenzione di segno")
    col1, col2 = st.columns(2)
    col1.success("**Δ > 0 — intensificazione**\n\nIl bin ha rilasciato "
                 "**piu'** energia del precedente.")
    col2.error("**Δ < 0 — attenuazione**\n\nIl bin ha rilasciato "
               "**meno** energia del precedente.")

    st.info(
        "Il catalogo misura energia **rilasciata**. La deformazione elastica "
        "**accumulata** e' caricata dal moto tettonico, e' pressoche' costante "
        "nel tempo e non e' osservabile dai sismogrammi: un mese di quiete e un "
        "mese di caricamento producono lo stesso dato, cioe' nessun evento. "
        "Il segno di $\\Delta M^{\\mathrm{eq}}$ non va quindi letto come "
        "accumulo o scarico di energia elastica.",
        icon="ℹ️",
    )

    st.subheader("Perche' non si cumulano le altezze")
    st.markdown(
        "Sommare le altezze lungo il periodo produce una quantita' telescopica:"
    )
    st.latex(r"\sum_{k=1}^{K} h_k = y_K - y_0")
    st.markdown(
        "Tutti i termini intermedi si cancellano. La somma dipende quindi "
        "**solo dal primo e dall'ultimo bin**, e vale identica per un periodo "
        "tranquillo e per uno attraversato da una sequenza sismica maggiore, "
        "purche' gli estremi coincidano. Per questo l'applicazione riporta "
        "aggregati **non lineari** — variazione totale, parte positiva, salto "
        "massimo — che al contrario distinguono i percorsi."
    )

    # Dimostrazione: due percorsi con gli stessi estremi.
    rng = np.random.default_rng(7)
    k = np.arange(15)
    lin = np.linspace(0.0, 1.8, 15)
    quiet = lin + np.r_[0, rng.normal(0, 0.08, 13), 0]
    agitated = lin.copy()
    agitated[3] += 1.6
    agitated[4] += 1.1
    agitated[9] -= 0.5

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=quiet, mode="lines+markers", name="percorso quieto",
                             line=dict(color="#1F5C8B", width=2.5)))
    fig.add_trace(go.Scatter(x=k, y=agitated, mode="lines+markers", name="percorso agitato",
                             line=dict(color="#C62828", width=2.5)))
    fig.add_trace(go.Scatter(x=[0, 14], y=[lin[0], lin[-1]], mode="markers",
                             name="estremi comuni",
                             marker=dict(size=13, color="#B7791F", symbol="circle-open",
                                         line=dict(width=3))))
    fig.update_layout(template="plotly_white", height=340, hovermode="x unified",
                      xaxis_title="bin", yaxis_title="magnitudo equivalente",
                      margin=dict(l=0, r=0, t=20, b=0),
                      legend=dict(orientation="h", y=1.15, x=0))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        pd.DataFrame({
            "somma con segno (lineare)": [np.diff(quiet).sum(), np.diff(agitated).sum()],
            "variazione totale (non lineare)": [np.abs(np.diff(quiet)).sum(),
                                                np.abs(np.diff(agitated)).sum()],
        }, index=["percorso quieto", "percorso agitato"]).round(3),
        use_container_width=True,
    )
    st.caption(
        "La colonna di sinistra e' identica per costruzione e non distingue i "
        "due regimi. Quella di destra li separa."
    )

    if result is not None and not result.diff.empty:
        st.subheader("I tuoi dati")
        st.dataframe(result.diff.round(4), use_container_width=True)

# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("Limiti del modello")
    st.markdown(
        """
Vale la pena tenerli presenti prima di trarre conclusioni.

**Il modello non prevede terremoti.** Descrive come e' variato il rilascio
energetico osservato. Non esiste in questo impianto alcun meccanismo predittivo,
e nessuna delle grandezze calcolate ha valore prognostico dimostrato.

**La deformazione accumulata non e' osservabile.** Il catalogo registra energia
rilasciata. Il caricamento tettonico e' invisibile ai sismogrammi.

**La serie energetica e' quasi la serie della magnitudo massima.** Una sciamatura
di migliaia di microeventi sparisce accanto a una singola scossa piu' forte. Per
questo l'applicazione affianca il canale dei conteggi.

**L'aggregazione mensile e' un filtro passa-basso.** Le sequenze
mainshock-repliche si sviluppano su ore e giorni, e vengono collassate in uno o
due bin. Il decadimento di Omori, che e' la principale struttura temporale dei
cataloghi, resta invisibile a questa risoluzione.

**Le magnitudo non sono omogenee.** La relazione energia-magnitudo e' calibrata
su $M_s$, mentre il catalogo INGV e' prevalentemente $M_L$ e $M_w$. Il bias
risultante non e' modellato.

**La soglia condiziona i risultati.** Il livello di fondo coincide con $M_c$:
analisi con soglie diverse non sono confrontabili fra loro.
        """
    )
