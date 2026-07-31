import pandas as pd
import numpy as np
import requests
from io import StringIO
import streamlit as st


class MathEngine:
    # Costanti Gutenberg-Richter: log10(E) = 1.5*M + 4.8
    GR_B = 1.5
    GR_A = 4.8

    @staticmethod
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
            "format": "text",
        }
        try:
            response = requests.get(url, params=params, timeout=30)

            # FDSNWS restituisce 204 (No Content) se non ci sono eventi
            if response.status_code == 204 or not response.text.strip():
                st.warning("Nessun evento trovato per i parametri specificati.")
                return None

            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text), sep='|')

            # FIX #2: l'header del formato text e' "#EventID|Time|..."
            # -> la prima colonna arriva come "#EventID". Normalizziamo.
            df.columns = df.columns.str.strip().str.lstrip('#')

            return df
        except requests.exceptions.RequestException as e:
            st.error(f"Errore di rete durante il download dei dati da INGV: {e}")
            return None
        except Exception as e:
            st.error(f"Errore durante l'elaborazione dei dati: {e}")
            return None

    @staticmethod
    def process_data(df_raw, target_year, min_mag,
                     years_before=5, years_after=5, data_end=None):
        """
        Finestra di analisi: [target_year - years_before, target_year + years_after]
        (default -5 / 0 / +5).

        data_end: (opzionale) fine effettiva della copertura dati (l'end_time
        usato nel fetch). Serve a non generare mesi oltre i dati disponibili.
        """
        df = df_raw.copy()
        df['Time'] = pd.to_datetime(df['Time'])

        # --- Finestra temporale estesa -5 / +5 ---
        window_start = pd.Timestamp(year=target_year - years_before, month=1, day=1)
        window_end = pd.Timestamp(year=target_year + years_after, month=12, day=31,
                                  hour=23, minute=59, second=59)
        df = df[(df['Time'] >= window_start) & (df['Time'] <= window_end)]

        # Energia di Gutenberg-Richter
        df['Energy_J'] = 10 ** (MathEngine.GR_B * df['Magnitude'] + MathEngine.GR_A)

        # Copia per lo scatter degli eventi registrati
        df_events = df.copy()
        df_events.set_index('Time', inplace=True)

        # Resampling mensile
        df.set_index('Time', inplace=True)
        monthly_energy = df['Energy_J'].resample('MS').sum()

        # --- FIX #1: niente mesi futuri fantasma ---
        # L'ultimo mese valido e' il piu' restrittivo tra:
        #   a) dicembre di (target_year + years_after)
        #   b) l'ultimo mese SOLARE COMPLETO rispetto a oggi
        #   c) l'ultimo mese completo coperto dai dati (se data_end e' fornito)
        current_month_start = pd.Timestamp.today().normalize().replace(day=1)
        last_complete_month = current_month_start - pd.offsets.MonthBegin(1)

        end_month = min(pd.Timestamp(year=target_year + years_after, month=12, day=1),
                        last_complete_month)

        if data_end is not None:
            data_end = pd.Timestamp(data_end)
            data_end_month_start = data_end.normalize().replace(day=1)
            # Se data_end non chiude il mese, quel mese e' incompleto -> escluso
            if data_end < (data_end_month_start + pd.offsets.MonthBegin(1) - pd.Timedelta(seconds=1)):
                data_end_month_start -= pd.offsets.MonthBegin(1)
            end_month = min(end_month, data_end_month_start)

        # Un mese in piu' all'inizio per avere il "prev" del primo triangolo
        all_months = pd.date_range(start=window_start - pd.offsets.MonthBegin(1),
                                   end=end_month, freq='MS')

        if len(all_months) < 2:
            st.warning("Finestra temporale insufficiente per calcolare i triangoli.")
            return df_events, pd.DataFrame()

        monthly_energy = monthly_energy.reindex(all_months, fill_value=0.0)

        # Pavimento e scala logaritmica
        floor_energy = 10 ** (MathEngine.GR_B * min_mag + MathEngine.GR_A)
        monthly_energy = monthly_energy.clip(lower=floor_energy)
        log_energy = np.log10(monthly_energy)

        # Triangoli Locali Differenziali
        df_tri = log_energy.to_frame(name='log10_E')
        df_tri['t'] = df_tri.index
        df_tri['prev_t'] = df_tri['t'].shift(1)
        df_tri['prev_log10_E'] = df_tri['log10_E'].shift(1)

        df_tri = df_tri.dropna().copy()

        df_tri['base_days'] = (df_tri['t'] - df_tri['prev_t']).dt.days
        df_tri['height'] = df_tri['log10_E'] - df_tri['prev_log10_E']
        df_tri['area'] = (df_tri['base_days'] * df_tri['height']) / 2.0

        df_tri['Year'] = df_tri['t'].dt.year
        df_tri['Month'] = df_tri['t'].dt.month

        return df_events, df_tri
