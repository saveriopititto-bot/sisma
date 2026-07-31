import streamlit as st

st.set_page_config(page_title="Sismologia Computazionale Analitica", layout="wide")

# Inject Custom CSS dalla cartella UI
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception:
        pass

load_css("ui/style.css")

# Import Moduli
from ui.components import (
    render_sidebar, 
    render_metrics, 
    render_scatter, 
    render_bar_chart, 
    render_overlay
)
from math_engine.engine import MathEngine

# --- TITOLO APP ---
st.title("Sismologia Computazionale Analitica")
st.markdown("""
Analisi dell'accumulo e rilascio di energia tettonica tramite **Triangoli Locali Differenziali**.
Il modello confronta 3 anni di riferimento, distanziati di **5 anni prima e 5 anni dopo**
l'Anno Zero centrale, con l'Anno di Verifica successivo, per validare l'accumulo di energia.
""")

# --- SIDEBAR E INPUTS ---
inputs = render_sidebar()

# --- ESECUZIONE MAIN ENGINE ---
if inputs['execute']:
    # Il fetch copre da dicembre dell'anno precedente triennium_start fino a target_year
    fetch_start = f"{inputs['triennium_start'] - 1}-12-01T00:00:00"
    fetch_end = f"{inputs['target_year']}-12-31T23:59:59"

    # 1. Fetching Dati
    df_raw = MathEngine.fetch_ingv_data(
        fetch_start, fetch_end,
        inputs['min_lat'], inputs['max_lat'],
        inputs['min_lon'], inputs['max_lon'],
        inputs['min_mag']
    )
    
    # 2. Elaborazione & Routing UI
    if df_raw is not None and not df_raw.empty:
        df_events, df_tri = MathEngine.process_data(
            df_raw,
            triennium_start=inputs['triennium_start'],
            triennium_end=inputs['triennium_end'],
            target_year=inputs['target_year'],
            min_mag=inputs['min_mag'],
            data_end=fetch_end
        )
        
        # Rendering delle metriche
        render_metrics(df_tri, inputs['triennium_start'], inputs['year_zero'], inputs['triennium_end'], inputs['target_year'])
        
        # Rendering dei 3 grafici Plotly
        render_scatter(df_events, inputs['target_year'])
        render_bar_chart(df_tri, inputs['target_year'])
        render_overlay(df_tri, inputs['triennium_start'], inputs['year_zero'], inputs['triennium_end'], inputs['target_year'])
        
    else:
        st.warning("Nessun dato trovato per i parametri selezionati o i terremoti registrati non superano la magnitudo minima indicata.")
