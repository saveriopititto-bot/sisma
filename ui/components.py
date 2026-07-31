import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def render_sidebar():
    st.sidebar.header("Input Geografici (Bounding Box)")
    min_lat = st.sidebar.number_input("Min Latitudine", value=42.80, format="%.2f")
    max_lat = st.sidebar.number_input("Max Latitudine", value=43.50, format="%.2f")
    min_lon = st.sidebar.number_input("Min Longitudine", value=12.50, format="%.2f")
    max_lon = st.sidebar.number_input("Max Longitudine", value=13.50, format="%.2f")
    min_mag = st.sidebar.number_input("Magnitudo Minima", value=1.5, format="%.1f")

    st.sidebar.header("Input Temporale")
    year_zero = st.sidebar.number_input("Anno Zero Centrale", value=2014, min_value=1900, max_value=2100, step=1)

    triennium_start = year_zero - 5
    triennium_end = year_zero + 5
    target_year = year_zero + 6

    st.sidebar.markdown(f"""
    **Anni di Riferimento (-5 / 0 / +5):** {triennium_start}, {year_zero}, {triennium_end}  
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
    
    df_triennio = df_tri[df_tri['Year'].isin([triennium_start, year_zero, triennium_end])]
    df_target = df_tri[df_tri['Year'] == target_year]
    
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
    cum_areas_triennio = []
    
    # Linee Accumulo
    for year in [triennium_start, year_zero, triennium_end]:
        df_year = df_tri[df_tri['Year'] == year]
        if not df_year.empty:
            cum_area = df_year['area'].cumsum().values
            cum_areas_triennio.append(cum_area)
            fig2.add_trace(go.Scatter(
                x=month_names[:len(cum_area)],
                y=cum_area,
                mode='lines',
                name=f"Accumulo {year}",
                line=dict(width=2, dash='dash', color='gray'),
                opacity=0.4
            ))
            
    # Andamento Previsto (Media)
    if cum_areas_triennio:
        max_len = max(len(arr) for arr in cum_areas_triennio)
        padded = np.full((len(cum_areas_triennio), max_len), np.nan)
        for i, arr in enumerate(cum_areas_triennio):
            padded[i, :len(arr)] = arr
        mean_cum_area = np.nanmean(padded, axis=0)
        fig2.add_trace(go.Scatter(
            x=month_names[:len(mean_cum_area)],
            y=mean_cum_area,
            mode='lines+markers',
            name='Andamento Previsto (Modello)',
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
        title="Overlay Cumulato Mensile delle Aree (Gen-Dic)",
        xaxis_title="Mese",
        yaxis_title="Area Cumulata",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig2, use_container_width=True)
