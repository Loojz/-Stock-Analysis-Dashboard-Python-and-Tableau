import yfinance as yf
import pandas as pd
import numpy as np

TICKERS = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL', 'AMZN']
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'

MA_WINDOWS = [12, 21, 50, 100, 200]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2
VOLATILITY_WINDOW = 20
VOLATILITY_HISTORY = 252

def get_ticker_info(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        return {
            'Ticker': ticker_symbol,
            'Name': info.get('longName', 'N/A'),
            'Sektor': info.get('sector', 'N/A'),
            'Branche': info.get('industry', 'N/A'),
            'Land': info.get('country', 'N/A'),
            'Website': info.get('website', 'N/A')
        }
    except Exception as e:
        print(f"Konnte keine Infos für {ticker_symbol} abrufen: {e}")
        return {
            'Ticker': ticker_symbol,
            'Name': 'N/A', 'Sektor': 'N/A', 'Branche': 'N/A', 'Land': 'N/A', 'Website': 'N/A'
        }


def calculate_indicators(df):
    for window in MA_WINDOWS:
        df[f'MA{window}'] = df['Close'].rolling(window=window).mean()

    df['Return'] = df['Close'].pct_change()

    rolling_std = df['Return'].rolling(window=VOLATILITY_WINDOW).std()

    df['Volatility'] = rolling_std * np.sqrt(252)

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema_fast = df['Close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    df['Bollinger_Mid'] = df['Close'].rolling(window=BOLLINGER_WINDOW).mean()
    df['Bollinger_Std'] = df['Close'].rolling(window=BOLLINGER_WINDOW).std()
    df['Bollinger_Upper'] = df['Bollinger_Mid'] + (df['Bollinger_Std'] * BOLLINGER_STD)
    df['Bollinger_Lower'] = df['Bollinger_Mid'] - (df['Bollinger_Std'] * BOLLINGER_STD)

    df['Norm_Close'] = df['Close'] / df['Close'].iloc[0] * 100

    return df


def classify_signals(df):

    df['Trend_Signal'] = np.nan
    df.loc[df['MA50'] > df['MA200'], 'Trend_Signal'] = 'Bullish (Golden Cross)'
    df.loc[df['MA50'] < df['MA200'], 'Trend_Signal'] = 'Bearish (Death Cross)'

    df['RSI_Status'] = 'Neutral'
    if 'RSI' in df.columns:
        df.loc[df['RSI'] > 70, 'RSI_Status'] = 'Überkauft'
        df.loc[df['RSI'] < 30, 'RSI_Status'] = 'Überverkauft'

    df['Sentiment'] = 'Seitwärts / Neutral'

    if 'Volatility' in df.columns:
        vol_q75 = df['Volatility'].rolling(window=VOLATILITY_HISTORY, min_periods=1).quantile(0.75)
        vol_q25 = df['Volatility'].rolling(window=VOLATILITY_HISTORY, min_periods=1).quantile(0.25)

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

        mask = df['Trend_Signal'].notna() & df['Volatility'].notna() & vol_q25.notna() & vol_q75.notna()

        df.loc[mask, 'Sentiment'] = np.select(
            [c[mask] for c in conditions],
            choices,
            default='Seitwärts / Neutral'
        )

    return df

all_data = []
all_info = []

for ticker in TICKERS:
    info_data = get_ticker_info(ticker)
    all_info.append(info_data)

    try:
        yf_ticker = yf.Ticker(ticker)
        data = yf_ticker.history(start=START_DATE, end=END_DATE, auto_adjust=True)
    except Exception as e:
        print(f"Fehler beim Download für {ticker}: {e}")
        continue

    if not isinstance(data, pd.DataFrame) or data.empty:
        print(f"Keine gültigen DataFrame-Daten für {ticker} empfangen – überspringe.")
        continue

    data.reset_index(inplace=True)

    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']

    for col in cols_to_numeric:
        if col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            except TypeError:
                print("\n" + "=" * 60)
                print(f"!!! FATALER FEHLER BEI DER DATENKONVERTIERUNG !!!")
                print(f"TICKER: {ticker}, SPALTE: '{col}'")
                print(f"Der Datentyp des 'data'-Objekts ist: {type(data)}")
                print(f"Die Spalten des 'data'-Objekts sind: {data.columns.to_list()}")
                print(f"Der Datentyp der problematischen Spalte '{col}' ist: {type(data[col])}")
                print("Die ersten 5 Zeilen der Spalte sehen so aus:")
                print(data[col].head())
                print("=" * 60 + "\n")
                raise
        else:
            print(f"Hinweis: Spalte '{col}' wurde in den Daten für {ticker} nicht gefunden.")

    original_rows = len(data)
    data.dropna(subset=['Close'], inplace=True)
    if len(data) < original_rows:
        print(f"{original_rows - len(data)} fehlerhafte Zeile(n) für {ticker} entfernt.")


    data = calculate_indicators(data)

    data = classify_signals(data)

    data['Ticker'] = ticker
    all_data.append(data)


final_df = pd.concat(all_data, ignore_index=True)
info_df = pd.DataFrame(all_info)

final_df.drop(columns=['Bollinger_Std'], inplace=True, errors='ignore')

final_df.to_csv("aktien_dashboard_enhanced.csv", index=False, float_format="%.4f")
info_df.to_csv("ticker_info_dynamic.csv", index=False)

print(final_df[['Date', 'Ticker', 'Close', 'RSI', 'RSI_Status', 'Trend_Signal', 'Sentiment']].tail())

print(info_df.head())