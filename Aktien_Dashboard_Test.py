# === BENÖTIGTE BIBLIOTHEKEN ===
import yfinance as yf
import pandas as pd
import numpy as np  # Wird für komplexere Berechnungen benötigt

# === 1. KONFIGURATION ===
# Ändere hier die Ticker, den Zeitraum und die Parameter für die Indikatoren
TICKERS = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL', 'AMZN']  # Amazon als neuen Ticker hinzugefügt
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'

# Parameter für technische Indikatoren
MA_WINDOWS = [12, 21, 50, 100, 200]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2
VOLATILITY_WINDOW = 20
VOLATILITY_HISTORY = 252  # Vergleichsperiode für Volatilität (ca. 1 Handelsjahr)


# === 2. FUNKTIONEN ===

def get_ticker_info(ticker_symbol):
    """Ruft allgemeine Informationen für einen Ticker von yfinance ab."""
    print(f"ℹ️  Sammle Infos für {ticker_symbol}...")
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        # Wähle die gewünschten Informationen aus und gib sie als Dictionary zurück
        return {
            'Ticker': ticker_symbol,
            'Name': info.get('longName', 'N/A'),
            'Sektor': info.get('sector', 'N/A'),
            'Branche': info.get('industry', 'N/A'),
            'Land': info.get('country', 'N/A'),
            'Website': info.get('website', 'N/A')
        }
    except Exception as e:
        print(f"⚠️ Konnte keine Infos für {ticker_symbol} abrufen: {e}")
        return {
            'Ticker': ticker_symbol,
            'Name': 'N/A', 'Sektor': 'N/A', 'Branche': 'N/A', 'Land': 'N/A', 'Website': 'N/A'
        }


def calculate_indicators(df):
    """Berechnet alle technischen Indikatoren für den gegebenen DataFrame."""

    # Gleitende Durchschnitte (dynamisch aus Konfiguration)
    for window in MA_WINDOWS:
        df[f'MA{window}'] = df['Close'].rolling(window=window).mean()

    # Tägliche Rendite und Volatilität
    df['Return'] = df['Close'].pct_change()

    # --- KORRIGIERTE VOLATILITÄTS-BERECHNUNG ---
    # 1. Berechne die rollierende Standardabweichung der täglichen Renditen
    rolling_std = df['Return'].rolling(window=VOLATILITY_WINDOW).std()
    # 2. Annualisiere sie (multipliziere mit der Wurzel aus der Anzahl der Handelstage im Jahr)
    # Dies ist robuster und finanziell die gängigere Methode.
    df['Volatility'] = rolling_std * np.sqrt(252)
    # --- ENDE DER KORREKTUR ---

    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    ema_fast = df['Close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bänder
    df['Bollinger_Mid'] = df['Close'].rolling(window=BOLLINGER_WINDOW).mean()
    df['Bollinger_Std'] = df['Close'].rolling(window=BOLLINGER_WINDOW).std()
    df['Bollinger_Upper'] = df['Bollinger_Mid'] + (df['Bollinger_Std'] * BOLLINGER_STD)
    df['Bollinger_Lower'] = df['Bollinger_Mid'] - (df['Bollinger_Std'] * BOLLINGER_STD)

    # Normalisierter Kurs für Performance-Vergleich
    df['Norm_Close'] = df['Close'] / df['Close'].iloc[0] * 100

    return df


def classify_signals(df):
    """Klassifiziert Signale und Marktsentiment basierend auf Indikatoren."""

    # Golden Cross / Death Cross Signal (mit NaN-Behandlung)
    df['Trend_Signal'] = np.nan
    df.loc[df['MA50'] > df['MA200'], 'Trend_Signal'] = 'Bullish (Golden Cross)'
    df.loc[df['MA50'] < df['MA200'], 'Trend_Signal'] = 'Bearish (Death Cross)'

    # RSI Status
    df['RSI_Status'] = 'Neutral'
    # Wichtig: Prüfe, ob RSI überhaupt existiert, bevor du darauf zugreifst
    if 'RSI' in df.columns:
        df.loc[df['RSI'] > 70, 'RSI_Status'] = 'Überkauft'
        df.loc[df['RSI'] < 30, 'RSI_Status'] = 'Überverkauft'

    # --- ROBUSTE SENTIMENT-BERECHNUNG ---
    # Standard-Sentiment ist neutral
    df['Sentiment'] = 'Seitwärts / Neutral'

    # Berechne die Quantile nur, wenn die Volatilitätsspalte existiert
    if 'Volatility' in df.columns:
        vol_q75 = df['Volatility'].rolling(window=VOLATILITY_HISTORY, min_periods=1).quantile(0.75)
        vol_q25 = df['Volatility'].rolling(window=VOLATILITY_HISTORY, min_periods=1).quantile(0.25)

        # Definiere Bedingungen für das Sentiment
        # WICHTIG: Die Vergleiche werden nur durchgeführt, wo die Werte gültig sind (kein NaN)
        conditions = [
            (df['Trend_Signal'] == 'Bullish (Golden Cross)') & (df['Volatility'] < vol_q25),
            (df['Trend_Signal'] == 'Bullish (Golden Cross)') & (df['Volatility'] > vol_q75),
            (df['Trend_Signal'] == 'Bearish (Death Cross)') & (df['Volatility'] > vol_q75),
            (df['Trend_Signal'] == 'Bearish (Death Cross)') & (df['Volatility'] < vol_q25)
        ]
        choices = [
            'Stabiler Aufwärtstrend',
            'Volatiler Aufwärtstrend',
            'Panischer Abwärtstrend',
            'Schwacher Abwärtstrend'
        ]

        # Wende np.select an, aber nur auf die Zeilen, wo die nötigen Daten vorhanden sind.
        # Wir erstellen eine Maske, die prüft, ob alle für die Bedingung benötigten Spalten einen gültigen Wert haben.
        mask = df['Trend_Signal'].notna() & df['Volatility'].notna() & vol_q25.notna() & vol_q75.notna()

        # Führe np.select nur für die gültigen Zeilen aus und weise das Ergebnis zu
        df.loc[mask, 'Sentiment'] = np.select(
            [c[mask] for c in conditions],
            choices,
            default='Seitwärts / Neutral'
        )

    return df


# === 3. HAUPTSKRIPT (DATENVERARBEITUNG) ===
all_data = []
all_info = []

print("🚀 Starte Aktien-Daten-Pipeline...")

for ticker in TICKERS:
    # 1. Statische Infos abrufen
    info_data = get_ticker_info(ticker)
    all_info.append(info_data)

    # 2. Historische Kursdaten abrufen
    print(f"⬇️  Lade Kursdaten für {ticker}...")
    try:
        yf_ticker = yf.Ticker(ticker)
        # Wir verwenden wieder .history() und auto_adjust=True für saubere Daten ohne 'Adj Close'
        data = yf_ticker.history(start=START_DATE, end=END_DATE, auto_adjust=True)
    except Exception as e:
        print(f"Fehler beim Download für {ticker}: {e}")
        continue

    # --- START: NEUE, KUGELSICHERE DATENVALIDIERUNG & BEREINIGUNG ---

    # Schritt A: Prüfen, ob wir überhaupt einen gültigen DataFrame haben
    if not isinstance(data, pd.DataFrame) or data.empty:
        print(f"⚠️ Keine gültigen DataFrame-Daten für {ticker} empfangen – überspringe.")
        continue

    # Schritt B: Index zurücksetzen, um 'Date' als Spalte zu haben. Das stabilisiert viele Operationen.
    data.reset_index(inplace=True)

    # Schritt C: Spalten definieren, die zwingend numerisch sein müssen
    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']

    for col in cols_to_numeric:
        # Prüfen, ob die Spalte überhaupt existiert
        if col in data.columns:
            try:
                # Versuch der Konvertierung
                data[col] = pd.to_numeric(data[col], errors='coerce')
            except TypeError:
                # DIESER BLOCK WIRD NUR AUSGEFÜHRT, WENN DER FEHLER ERNEUT AUFTRITT
                print("\n" + "=" * 60)
                print(f"!!! FATALER FEHLER BEI DER DATENKONVERTIERUNG !!!")
                print(f"TICKER: {ticker}, SPALTE: '{col}'")
                print(f"Der Datentyp des 'data'-Objekts ist: {type(data)}")
                print(f"Die Spalten des 'data'-Objekts sind: {data.columns.to_list()}")
                print(f"Der Datentyp der problematischen Spalte '{col}' ist: {type(data[col])}")
                print("Die ersten 5 Zeilen der Spalte sehen so aus:")
                print(data[col].head())
                print("=" * 60 + "\n")
                # Stoppe das Skript, da die Daten nicht verarbeitet werden können
                raise
        else:
            print(f"Hinweis: Spalte '{col}' wurde in den Daten für {ticker} nicht gefunden.")

    # Schritt D: Zeilen entfernen, in denen 'Close' nach der Bereinigung ungültig (NaN) ist
    original_rows = len(data)
    data.dropna(subset=['Close'], inplace=True)
    if len(data) < original_rows:
        print(f"🧹 {original_rows - len(data)} fehlerhafte Zeile(n) für {ticker} entfernt.")

    # --- ENDE: DATENVALIDIERUNG & BEREINIGUNG ---

    # 3. Indikatoren berechnen (Rest des Codes bleibt gleich)
    print(f"📈 Berechne Indikatoren für {ticker}...")
    data = calculate_indicators(data)

    # 4. Signale klassifizieren
    print(f"🧠 Klassifiziere Signale für {ticker}...")
    data = classify_signals(data)

    # 5. Daten für Zusammenführung vorbereiten
    data['Ticker'] = ticker
    # Der Index wurde bereits zurückgesetzt, daher ist data.reset_index() hier nicht mehr nötig
    all_data.append(data)

    print(f"✅ {ticker} erfolgreich verarbeitet.\n")

# === 4. DATEN KOMBINIEREN & SPEICHERN ===
print("💾 Kombiniere alle Daten und speichere Dateien...")

# Erzeuge finale DataFrames
final_df = pd.concat(all_data, ignore_index=True)
info_df = pd.DataFrame(all_info)

# Bereinige Spalten, die nur für Zwischenberechnungen nötig waren
final_df.drop(columns=['Bollinger_Std'], inplace=True, errors='ignore')

# Speichere die Ergebnisse als CSV-Dateien
final_df.to_csv("aktien_dashboard_enhanced.csv", index=False, float_format="%.4f")
info_df.to_csv("ticker_info_dynamic.csv", index=False)

print("\n🎉 FERTIG! Folgende Dateien wurden erstellt:")
print("- aktien_dashboard_enhanced.csv (Haupt-DataFrame mit allen Indikatoren)")
print("- ticker_info_dynamic.csv (Dynamisch abgerufene Firmen-Infos)")

# Zeige eine Vorschau der finalen Daten
print("\n🔎 Vorschau der erweiterten Daten:")
print(final_df[['Date', 'Ticker', 'Close', 'RSI', 'RSI_Status', 'Trend_Signal', 'Sentiment']].tail())

print("\n🔎 Vorschau der Ticker-Infos:")
print(info_df.head())