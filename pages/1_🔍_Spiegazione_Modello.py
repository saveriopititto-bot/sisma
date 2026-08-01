import streamlit as st
import pandas as pd

st.set_page_config(page_title="Spiegazione Modello", page_icon="🔍", layout="wide")

st.title("🔍 Come funziona il Modello Analitico")

st.markdown("""
Questa pagina ti permette di ispezionare "sotto il cofano" la matematica del modello di Sismologia Computazionale Analitica. 
Se hai appena eseguito un'analisi nella pagina principale, qui sotto vedrai esattamente come i dati grezzi sono stati trasformati in energia e triangoli differenziali.
""")

tab1, tab2, tab3 = st.tabs(["1. Dati Grezzi (INGV)", "2. Energia & Resampling", "3. Triangoli Differenziali"])

with tab1:
    st.header("1. Dati Grezzi (Sismicità Reale)")
    st.markdown("Il modello scarica gli eventi sismici dal web service FDSNWS dell'INGV filtrando per le coordinate geografiche, la magnitudo minima e il tempo.")
    if 'df_raw' in st.session_state and st.session_state['df_raw'] is not None:
        st.dataframe(st.session_state['df_raw'].head(100))
    else:
        st.info("Nessun dato in memoria. Vai nella pagina principale e clicca 'Esegui Analisi' per popolare queste tabelle.")

with tab2:
    st.header("2. Calcolo dell'Energia & Resampling")
    st.markdown("""
    L'energia (in Joule) viene stimata a partire dalla magnitudo locale sfruttando la formula di **Gutenberg-Richter**:
    
    $$ \\log_{10}(E) = 1.5 \\cdot M + 4.8 $$
    
    Dopodiché, l'energia rilasciata da ogni singolo terremoto viene aggregata (sommata) per mese solare. Viene applicata una correzione "floor" (pavimento) corrispondente all'energia minima di un evento per impedire cadute a zero assoluto logaritmico.
    """)
    if 'df_events' in st.session_state and st.session_state['df_events'] is not None:
        st.dataframe(st.session_state['df_events'][['Magnitude', 'Energy_J']].head(100))

with tab3:
    st.header("3. Triangoli Locali Differenziali")
    st.markdown("""
    L'energia mensile cumulativa viene trasformata in scala logaritmica. Il modello calcola le differenze tra mesi consecutivi per formare la geometria differenziale:
    
    - **Base ($b$)**: i giorni trascorsi tra la rilevazione corrente e quella precedente (sempre ~30).
    - **Altezza ($h$)**: la differenza energetica (in scala log10).
    - **Area ($A$)**: calcolata come $\\frac{b \\cdot h}{2}$, quantifica in maniera vettoriale l'accumulo o il rilascio cinetico in quell'intervallo di tempo.
    
    Se l'Area è **positiva**, si sta accumulando energia. Se è **negativa**, l'energia viene rilasciata.
    """)
    if 'df_tri' in st.session_state and st.session_state['df_tri'] is not None:
        st.dataframe(st.session_state['df_tri'])
