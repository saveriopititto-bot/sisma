import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def render_sidebar():
    st.sidebar.header("Selezione Area Sismogenetica")
    
    # 1. Mini-catalogo delle faglie (Bounding Box approssimativi per l'estrazione)
    faglie_db = {
        "Manuale (Usa coordinate libere)": None,
        "Faglia di Gubbio (ITCS001)": {"min_lat": 43.20, "max_lat": 43.45, "min_lon": 12.45, "max_lon": 12.75},
        "Faglia del Monte Vettore": {"min_lat": 42.70, "max_lat": 42.95, "min_lon": 13.10, "max_lon": 13.40},
        "Faglia di Paganica (L'Aquila)": {"min_lat": 42.25, "max_lat": 42.45, "min_lon": 13.30, "max_lon": 13.60}
    }
    
    # 2. Selezione tramite nome
    faglia_scelta = st.sidebar.selectbox(
        "Scegli una faglia catalogata:", 
        list(faglie_db.keys())
    )
    
    # 3. Logica di auto-compilazione
    is_manual = faglia_scelta == "Manuale (Usa coordinate libere)"
    box = faglie_db[faglia_scelta]
    
    st.sidebar.markdown("**Bounding Box (Lat/Lon):**")
    
    # Se è manuale usiamo i default originali e permettiamo la modifica, 
    # altrimenti usiamo i valori del catalogo e disabilitiamo l'input.
    min_lat = st.sidebar.number_input("Min Latitudine", 
                                      value=42.80 if is_manual else box["min_lat"], 
                                      format="%.2f", disabled=not is_manual)
    max_lat = st.sidebar.number_input("Max Latitudine", 
                                      value=43.50 if is_manual else box["max_lat"], 
                                      format="%.2f", disabled=not is_manual)
    min_lon = st.sidebar.number_input("Min Longitudine", 
                                      value=12.50 if is_manual else box["min_lon"], 
                                      format="%.2f", disabled=not is_manual)
    max_lon = st.sidebar.number_input("Max Longitudine", 
                                      value=13.50 if is_manual else box["max_lon"], 
                                      format="%.2f", disabled=not is_manual)
    
    st.sidebar.divider()
    
    min_mag = st.sidebar.number_input("Magnitudo Minima", value=1.5, format="%.1f")
    
    st.sidebar.header("Input Temporale")
    year_zero = st.sidebar.number_input("Anno Zero Centrale", value=2014, min_value=1900, max_value=2100, step=1)
    triennium_start = year_zero - 5
    triennium_end = year_zero + 5
    target_year = year_zero + 6
    
    st.sidebar.markdown(f"""
    **Periodo di Analisi (11 anni, N-5 ➔ N+5):** {triennium_start} - {triennium_end}  
    **Anno Zero Centrale (N):** {year_zero}  
    **Anno di Verifica (Target):** {target_year}
    """)
    
    execute = st.sidebar.button("Esegui Analisi", type="primary")
    
    return {
        "min_lat": min_lat, "max_lat": max_lat,
        "min_lon": min_lon, "max_lon": max_lon,
        "min_mag": min_mag, "year_zero": year_zero,
        "triennium_start": triennium_start, "triennium_end": triennium_end,
        "target_year": target_year, "execute": execute
    }

def render_metrics(df_tri, triennium_start, year_zero, triennium_end, target_year):
    st.subheader("Metriche di Bilancio Energetico", divider='gray')
    
    # Seleziona gli 11 anni di analisi (da N-5 a N+5 inclusi)
    df_periodo = df_tri[(df_tri['Year'] >= triennium_start) & (df_tri['Year'] <= triennium_end)]
    df_target = df_tri[df_tri['Year'] == target_year]
    
    accumulo_cumulato_periodo = df_periodo['area'].sum()
    num_years = triennium_end - triennium_start + 1  # 11 anni
    media_annuale_accumulo = accumulo_cumulato_periodo / float(num_years) if num_years > 0 else 0.0
    bilancio_target = df_target['area'].sum() if not df_target.empty else 0.0
    
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Bilancio Cumulato ({num_years} Anni)", f"{accumulo_cumulato_periodo:.2f}", 
                help=f"Somma totale delle aree (Rilascio e Accumulo) durante gli {num_years} anni di analisi ({triennium_start}-{triennium_end}).")
    col2.metric(f"Media Annuale ({num_years} Anni)", f"{media_annuale_accumulo:.2f}")
    col3.metric(f"Bilancio Anno Target ({target_year})", f"{bilancio_target:.2f}", 
                delta=f"{(bilancio_target - media_annuale_accumulo):.2f} vs Media", 
                delta_color="off")

def render_scatter(df_events, target_year):
    st.subheader("Visualizzazione Analitica", divider='gray')
    fig0 = go.Figure()
    fig0.add_trace(go.Scatter(
        x=df_events.index,
        y=df_events['Magnitude'],
        mode='markers',
        marker=dict(
            size=df_events['Magnitude'] ** 2,
            color=df_events['Magnitude'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Magnitudo")
        ),
        text="Mag: " + df_events['Magnitude'].astype(str) + "<br>" + df_events.index.strftime('%Y-%m-%d %H:%M'),
        hoverinfo='text',
        name='Sismi Registrati'
    ))
    
    separator_date = pd.to_datetime(f"{target_year}-01-01")
    fig0.add_vline(x=separator_date, line_dash="dash", line_color="rgba(0,0,0,0.7)", line_width=2,
                   annotation_text="  Inizio Anno di Verifica", annotation_position="top right")
    
    fig0.update_layout(
        title="Andamento Reale Sismicità (Eventi Registrati vs Tempo)",
        xaxis_title="Tempo",
        yaxis_title="Magnitudo",
        template="plotly_white",
        showlegend=False
    )
    st.plotly_chart(fig0, use_container_width=True)

def render_bar_chart(df_tri, target_year):
    fig1 = go.Figure()
    colors = ['#2ca02c' if val > 0 else '#d62728' for val in df_tri['area']]
    
    fig1.add_trace(go.Bar(
        x=df_tri['t'],
        y=df_tri['area'],
        marker_color=colors,
        name='Area Triangolo (Rilascio > 0, Accumulo < 0)'
    ))
    
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

def render_overlay(df_tri, triennium_start, year_zero, triennium_end, target_year):
    fig2 = go.Figure()
    month_names = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
    cum_areas_periodo = []
    
    # Linee Accumulo per tutti gli 11 anni del periodo (da triennium_start N-5 a triennium_end N+5)
    for year in range(triennium_start, triennium_end + 1):
        df_year = df_tri[df_tri['Year'] == year]
        if not df_year.empty:
            cum_area = df_year['area'].cumsum().values
            cum_areas_periodo.append(cum_area)
            is_year_zero = (year == year_zero)
            fig2.add_trace(go.Scatter(
                x=month_names[:len(cum_area)],
                y=cum_area,
                mode='lines',
                name=f"Accumulo {year}" + (" (Anno Zero)" if is_year_zero else ""),
                line=dict(width=3 if is_year_zero else 1.5, dash='solid' if is_year_zero else 'dash'),
                opacity=0.85 if is_year_zero else 0.35
            ))
            
    # Andamento Previsto (Media degli 11 Anni)
    if cum_areas_periodo:
        max_len = max(len(arr) for arr in cum_areas_periodo)
        padded = np.full((len(cum_areas_periodo), max_len), np.nan)
        for i, arr in enumerate(cum_areas_periodo):
            padded[i, :len(arr)] = arr
        mean_cum_area = np.nanmean(padded, axis=0)
        fig2.add_trace(go.Scatter(
            x=month_names[:len(mean_cum_area)],
            y=mean_cum_area,
            mode='lines+markers',
            name='Andamento Previsto (Media 11 Anni)',
            line=dict(width=4, color='royalblue', dash='dashdot')
        ))
            
    # Andamento Reale Target
    df_target = df_tri[df_tri['Year'] == target_year]
    if not df_target.empty:
        cum_area_target = df_target['area'].cumsum().values
        fig2.add_trace(go.Scatter(
            x=month_names[:len(cum_area_target)],
            y=cum_area_target,
            mode='lines+markers',
            name=f"Andamento Reale Target ({target_year})",
            line=dict(width=4, color='red')
        ))
        
    fig2.update_layout(
        title=f"Overlay Cumulato Mensile delle Aree ({triennium_start}-{triennium_end} vs Target {target_year})",
        xaxis_title="Mese",
        yaxis_title="Area Cumulata",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig2, use_container_width=True)
