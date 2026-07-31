import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import plotly.graph_objects as go

st.set_page_config(page_title="Sismologia Computazionale Analitica", layout="wide")

st.title("Sismologia Computazionale Analitica")
st.markdown("""
Analisi dell'accumulo e rilascio di energia tettonica tramite **Triangoli Locali Differenziali**.
Il modello analizza i 3 anni precedenti a un potenziale evento per validare l'accumulo di energia.
""")

# --- SIDEBAR ---
st.sidebar.header("Input Geografici (Bounding Box)")
min_lat = st.sidebar.number_input("Min Latitudine", value=42.80, format="%.2f")
max_lat = st.sidebar.number_input("Max Latitudine", value=43.50, format="%.2f")
min_lon = st.sidebar.number_input("Min Longitudine", value=12.50, format="%.2f")
max_lon = st.sidebar.number_input("Max Longitudine", value=13.50, format="%.2f")
min_mag = st.sidebar.number_input("Magnitudo Minima", value=1.5, format="%.1f")

st.sidebar.header("Input Temporale")
year_zero = st.sidebar.number_input("Anno Zero Centrale", value=2014, min_value=1900, max_value=2100, step=1)

# Logica Temporale
triennium_start = year_zero - 1
triennium_end = year_zero + 1
target_year = year_zero + 2

st.sidebar.markdown(f"""
**Triennio di Analisi (Accumulo):** {triennium_start}, {year_zero}, {triennium_end}  
**Anno di Verifica (Target):** {target_year}
""")

@st.cache_data(show_spinner="Download dati da INGV FDSNWS in corso...")
def fetch_ingv_data(start_time, end_time, min_lat, max_lat, min_lon, max_lon, min_mag):
    url = "https://webservices.ingv.it/fdsnws/event/1/query"
    params = {
        "starttime": start_time,
        "endtime": end_time,
        "minlat": min_lat,
        "maxlat": max_lat,
        "minlon": min_lon,
        "maxlon": max_lon,
        "minmag": min_mag,
        "format": "text"
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        # Il formato di testo di INGV utilizza il separatore pipe '|'
        df = pd.read_csv(StringIO(response.text), sep='|')
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Errore di rete durante il download dei dati da INGV: {e}")
        return None
    except Exception as e:
        st.error(f"Errore durante l'elaborazione dei dati: {e}")
        return None

# Fetch data dal Dicembre dell'anno precedente al triennio per avere il delta_t per il primo mese (Gennaio triennium_start)
fetch_start = f"{triennium_start - 1}-12-01T00:00:00"
fetch_end = f"{target_year}-12-31T23:59:59"

if st.sidebar.button("Esegui Analisi", type="primary"):
    df_raw = fetch_ingv_data(fetch_start, fetch_end, min_lat, max_lat, min_lon, max_lon, min_mag)
    
    if df_raw is not None and not df_raw.empty:
        # Preprocessing
        df = df_raw.copy()
        # Converte la colonna temporale e imposta index
        df['Time'] = pd.to_datetime(df['Time'])
        
        # 2. Calcolo Energia (Gutenberg-Richter)
        df['Energy_J'] = 10**(1.5 * df['Magnitude'] + 4.8)
        
        # 3. Resampling Mensile
        df.set_index('Time', inplace=True)
        # Raggruppa sommando per inizio del mese ('MS')
        monthly_energy = df['Energy_J'].resample('MS').sum()
        
        # Assicuriamoci che tutti i mesi nel range di analisi siano presenti, anche se senza terremoti
        all_months = pd.date_range(start=f"{triennium_start - 1}-12-01", end=f"{target_year}-12-31", freq='MS')
        monthly_energy = monthly_energy.reindex(all_months, fill_value=0.0)
        
        # 4. Scala Logaritmica e Pavimento
        floor_energy = 10**(1.5 * min_mag + 4.8)
        monthly_energy = monthly_energy.clip(lower=floor_energy)
        log_energy = np.log10(monthly_energy)
        
        # 5. Triangoli Locali Differenziali
        df_tri = log_energy.to_frame(name='log10_E')
        df_tri['t'] = df_tri.index
        df_tri['prev_t'] = df_tri['t'].shift(1)
        df_tri['prev_log10_E'] = df_tri['log10_E'].shift(1)
        
        # Rimuoviamo il primo mese (Dicembre pre-triennio) usato solo per calcolare il delta base del primo Gennaio
        df_tri = df_tri.dropna().copy()
        
        # Calcolo Base e Altezza
        df_tri['base_days'] = (df_tri['t'] - df_tri['prev_t']).dt.days
        df_tri['height'] = df_tri['log10_E'] - df_tri['prev_log10_E']
        
        # Calcolo Area (con segno: > 0 Rilascio, < 0 Accumulo)
        df_tri['area'] = (df_tri['base_days'] * df_tri['height']) / 2.0
        
        # Annotazioni per raggruppamenti annuali
        df_tri['Year'] = df_tri['t'].dt.year
        df_tri['Month'] = df_tri['t'].dt.month
        
        # Separazione Triennio / Anno Target
        df_triennio = df_tri[df_tri['Year'].isin([triennium_start, year_zero, triennium_end])]
        df_target = df_tri[df_tri['Year'] == target_year]
        
        # --- METRICHE ---
        st.subheader("Metriche di Bilancio Energetico", divider='gray')
        
        accumulo_cumulato_triennio = df_triennio['area'].sum()
        media_annuale_accumulo = accumulo_cumulato_triennio / 3.0
        bilancio_target = df_target['area'].sum() if not df_target.empty else 0.0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Bilancio Cumulato Triennio", f"{accumulo_cumulato_triennio:.2f}", 
                    help="Somma totale delle aree (Rilascio e Accumulo) durante i 3 anni di analisi.")
        col2.metric("Media Annuale Triennio", f"{media_annuale_accumulo:.2f}")
        col3.metric(f"Bilancio Anno Target ({target_year})", f"{bilancio_target:.2f}", 
                    delta=f"{(bilancio_target - media_annuale_accumulo):.2f} vs Media", 
                    delta_color="off")
        
        # --- GRAFICI ---
        st.subheader("Visualizzazione Analitica", divider='gray')
        
        # GRAFICO 1: Serie Storica (Barre)
        fig1 = go.Figure()
        
        colors = ['#2ca02c' if val > 0 else '#d62728' for val in df_tri['area']]
        
        fig1.add_trace(go.Bar(
            x=df_tri['t'],
            y=df_tri['area'],
            marker_color=colors,
            name='Area Triangolo (Rilascio > 0, Accumulo < 0)'
        ))
        
        # Linea verticale tratteggiata per separare Triennio da Target
        separator_date = pd.to_datetime(f"{target_year}-01-01")
        fig1.add_vline(x=separator_date, line_dash="dash", line_color="rgba(0,0,0,0.7)", line_width=2,
                       annotation_text="  Inizio Anno di Verifica", annotation_position="top right")
        
        fig1.update_layout(
            title="Area Triangoli Locali Differenziali (Serie Storica)",
            xaxis_title="Tempo",
            yaxis_title="Area (giorni * Δlog10(E))",
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        
        # GRAFICO 2: Overlay Cumulato Gen-Dic
        fig2 = go.Figure()
        
        month_names = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
        
        # Linee per i 3 anni del Triennio di Analisi
        for year in [triennium_start, year_zero, triennium_end]:
            df_year = df_tri[df_tri['Year'] == year]
            if not df_year.empty:
                cum_area = df_year['area'].cumsum().values
                fig2.add_trace(go.Scatter(
                    x=month_names[:len(cum_area)],
                    y=cum_area,
                    mode='lines+markers',
                    name=f"Accumulo {year}",
                    line=dict(width=2, dash='dash', color='gray'),
                    opacity=0.6
                ))
                
        # Linea per l'Anno di Verifica (Target)
        if not df_target.empty:
            cum_area_target = df_target['area'].cumsum().values
            fig2.add_trace(go.Scatter(
                x=month_names[:len(cum_area_target)],
                y=cum_area_target,
                mode='lines+markers',
                name=f"Target ({target_year})",
                line=dict(width=4, color='red')
            ))
            
        fig2.update_layout(
            title="Overlay Cumulato Mensile delle Aree (Gen-Dic)",
            xaxis_title="Mese",
            yaxis_title="Area Cumulata",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig2, use_container_width=True)
        
    else:
        st.warning("Nessun dato trovato per i parametri selezionati o i terremoti registrati non superano la magnitudo minima indicata.")
