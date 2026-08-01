# Sismologia Computazionale Analitica

Applicazione **Streamlit** per l'analisi differenziale del **rilascio** di energia
sismica a partire dai cataloghi INGV.

L'app interroga il web service FDSNWS, converte le magnitudo in energia tramite
la relazione di Gutenberg–Richter, aggrega su griglia temporale e ne studia le
variazioni fra intervalli consecutivi. Il confronto fra un anno di verifica e
l'inviluppo degli anni di riferimento avviene su due canali indipendenti:
l'energia rilasciata e il tasso di attività.

> **Cosa il modello non fa.** Non prevede terremoti e non misura la
> deformazione elastica accumulata. Il catalogo registra energia *rilasciata*;
> il caricamento tettonico è invisibile ai sismogrammi, perché un mese di quiete
> e un mese di caricamento producono lo stesso dato, cioè nessun evento. Le
> grandezze calcolate descrivono la storia osservata, non uno stato latente.

## Modello

Detta $S_k$ l'energia rilasciata nel bin $k$ e $M_c$ la soglia di completezza:

| Grandezza | Definizione | Note |
|---|---|---|
| Energia | $E(M) = 10^{1.5M + 4.8}$ [J] | Gutenberg–Richter 1956 |
| Censura | $\hat S_k = \max\{S_k, E(M_c)\}$ | Agisce **solo** sui bin vuoti |
| Serie log | $y_k = \log_{10}\hat S_k$ | |
| Magnitudo equivalente | $M_k^{\mathrm{eq}} = (y_k - 4.8)/1.5$ | Lettura naturale di $y_k$ |
| Altezza differenziale | $\Delta M^{\mathrm{eq}}_k = (y_k - y_{k-1})/1.5$ | **Statistica primaria** |
| Canale di tasso | $r_k = \log_{10}(1 + n_k)$ | Ortogonale all'energia |

**Convenzione di segno**, unica in tutta l'applicazione:

- $\Delta M^{\mathrm{eq}} > 0$ → **intensificazione**: il bin ha rilasciato più energia del precedente
- $\Delta M^{\mathrm{eq}} < 0$ → **attenuazione**: ne ha rilasciata meno

### Perché non si cumula

Sommare le altezze produce una quantità telescopica:

$$\sum_{k=1}^{K} h_k = y_K - y_0$$

I termini intermedi si cancellano: la somma dipende solo dal primo e dall'ultimo
bin, e vale identica per un periodo tranquillo e per uno attraversato da una
sequenza sismica maggiore, purché gli estremi coincidano. L'applicazione riporta
quindi aggregati **non lineari** — variazione totale, parte positiva, salto
massimo — che al contrario distinguono i percorsi.

### Perché due canali

L'energia cresce come $10^{1.5M}$ mentre gli eventi si diradano come $10^{-M}$:
la somma è dominata dalla scossa più forte, e la serie energetica coincide
quasi con la serie della magnitudo massima. Uno sciame di migliaia di
microeventi è invisibile in quel canale. Il canale dei conteggi lo vede.

## Struttura

```
app.py                          punto di ingresso, cache e orchestrazione
math_engine/
    engine.py                   motore puro: nessuna dipendenza da Streamlit
    synthetic.py                generatore di cataloghi per test e demo offline
ui/
    components.py               input, diagnostica, grafici Plotly
    style.css
pages/
    1_🔍_Spiegazione_Modello.py  documentazione interattiva del modello
tests/
    test_engine.py              invarianti matematici
```

Il motore è deliberatamente separato dall'interfaccia: solleva eccezioni invece
di chiamare `st.error`, e restituisce le diagnostiche come dati. È quindi
utilizzabile da uno script, da un notebook o da un job, e testabile senza
avviare Streamlit.

```python
from math_engine import Config, Region, Binning, run

config = Config(
    region=Region(43.20, 43.45, 12.45, 12.75),
    mc=1.5,
    start=pd.Timestamp("2013-12-01"),
    target_year=2020,
    binning=Binning("uniform", 30),
    normalize_exposure=True,
)
result = run(config)
print(result.summary["total_variation"])
```

## Installazione

Richiede Python 3.10 o superiore.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Esecuzione

```bash
streamlit run app.py
```

L'app si apre su `http://localhost:8501`. Senza connessione all'INGV, la casella
*Usa catalogo sintetico* nella barra laterale genera dati localmente per
esplorare l'interfaccia.

## Test

```bash
pytest
```

I test verificano proprietà matematiche, non solo che il codice giri: la censura
tocca esclusivamente i bin vuoti, i limiti $M^{\max} \le M^{\mathrm{eq}} \le
M^{\max} + \log_{10}n / 1.5$ sono rispettati, nessun bin restituito è
parzialmente coperto dai dati, la somma con segno telescopizza. Un fallimento
significa che un grafico sta mostrando qualcosa di diverso da ciò che dichiara.

## Limiti

- **Nessun valore predittivo.** Il modello è descrittivo.
- **Risoluzione temporale.** L'aggregazione mensile è un filtro passa-basso:
  le sequenze mainshock–repliche si sviluppano su ore e giorni e vengono
  collassate in uno o due bin. Il decadimento di Omori resta invisibile.
- **Magnitudo non omogenee.** La relazione energia–magnitudo è calibrata su
  $M_s$; il catalogo INGV è prevalentemente $M_L$ e $M_w$. Il bias non è
  modellato.
- **Dipendenza dalla soglia.** Il livello di fondo coincide con $M_c$: analisi
  condotte con soglie diverse non sono confrontabili.
- **Esposizione variabile.** Con griglia calendariale febbraio raccoglie il 10%
  di giorni in meno di gennaio. L'opzione *Normalizza l'esposizione* corregge il
  bias; la griglia a passo fisso lo elimina alla radice.

## Riferimenti

- B. Gutenberg, C. F. Richter, *Magnitude and energy of earthquakes*, Annali di Geofisica 9 (1956)
- B. Gutenberg, C. F. Richter, *Frequency of earthquakes in California*, BSSA 34 (1944)
- T. Utsu, Y. Ogata, R. Matsu'ura, *The centenary of the Omori formula*, J. Phys. Earth 43 (1995)
- [FDSN Web Service Specifications](https://www.fdsn.org/webservices/)
