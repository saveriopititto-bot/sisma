# Sismologia Computazionale Analitica

Applicazione web sviluppata in **Python** con **Streamlit** per l'analisi dell'accumulo e del rilascio di energia tettonica tramite il modello matematico dei **Triangoli Locali Differenziali**.

L'app interroga il web service FDSNWS dell'INGV per estrarre i cataloghi sismici, calcola l'energia rilasciata basandosi sulla magnitudo (Gutenberg-Richter), e modella l'accumulo elastico nel tempo. L'obiettivo dell'applicativo è validare e studiare l'energia in un "Triennio di Analisi" (accumulo) confrontandola con un "Anno di Verifica" (target).

## Funzionalità Principali
- **Integrazione API INGV**: Download automatico tramite web service FDSNWS, filtrato per data, magnitudo e Bounding Box geografico.
- **Calcolo Energetico**: Resampling mensile dell'energia sismica in scala logaritmica.
- **Modello Matematico**: Computazione dell'area dei Triangoli Locali Differenziali (Rilascio > 0, Accumulo < 0).
- **Dashboard Interattiva**: Visualizzazioni avanzate con Plotly (serie storiche ed overlay cumulate annuali).

## Requisiti

Assicurati di avere Python 3.8 o superiore installato sul tuo sistema.
Le dipendenze sono elencate nel file `requirements.txt` e includono `streamlit`, `pandas`, `numpy`, `requests` e `plotly`.

## Installazione

1. Clona la repository o scarica i file in una cartella locale.
2. Crea un ambiente virtuale (consigliato):
   ```bash
   python -m venv venv
   # Su Windows:
   venv\Scripts\activate
   # Su Mac/Linux:
   source venv/bin/activate
   ```
3. Installa le librerie necessarie:
   ```bash
   pip install -r requirements.txt
   ```

## Esecuzione

Avvia l'applicazione digitando nel terminale:

```bash
streamlit run app.py
```

L'app si aprirà automaticamente nel tuo browser all'indirizzo `http://localhost:8501`.
