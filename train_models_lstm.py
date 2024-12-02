import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


atr = 3
rsi =3
roc = 3
sma10 = 5
sma50 = 10

# Função para calcular o RSI
def calculate_rsi(data, window=3):
    delta = data.diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Função para calcular o ROC
def calculate_roc(data, window=3):
    return ((data - data.shift(window)) / data.shift(window)) * 100


# Função para calcular o ATR
def calculate_atr(high, low, close, window=3):
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    return true_range.rolling(window=window).mean()


# Carregar os dados
file_path = "sp500_nasdaq100_data.xlsx"  # Substituir pelo caminho do seu arquivo
df = pd.read_excel(file_path)

# Ordenar por data para manter a sequência temporal
df = df.sort_values(by=["Ticker", "Date"])

# Criar uma pasta para salvar os modelos
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

# Lista para salvar as métricas de cada modelo
performance_metrics = []

# Loop para treinar e salvar o modelo para cada ticker
sequence_length = 4
h5_files = []

for ticker in df['Ticker'].unique():
    print(f"Treinando o modelo para o ticker {ticker}...")

    # Filtrar os dados do ticker
    ticker_data = df[df['Ticker'] == ticker].copy()

    # Calcular indicadores técnicos
    ticker_data['RSI'] = calculate_rsi(ticker_data['Close'], window=rsi)
    ticker_data['SMA10'] = ticker_data['Close'].rolling(window=sma10).mean()
    ticker_data['SMA50'] = ticker_data['Close'].rolling(window=sma50).mean()
    ticker_data['ROC10'] = calculate_roc(ticker_data['Close'], window=roc)
    ticker_data['ATR'] = calculate_atr(ticker_data['High'], ticker_data['Low'], ticker_data['Close'], window=atr)

    # Substituir NaN gerados pelos cálculos
    ticker_data.fillna(0, inplace=True)
    print(ticker)

    # Normalizar os dados
    scaler = MinMaxScaler()
    columns_to_scale = ['Open', 'Close', 'High', 'Low', 'Volume', 'RSI', 'SMA10', 'SMA50', 'ROC10', 'ATR']
    ticker_data[columns_to_scale] = scaler.fit_transform(ticker_data[columns_to_scale])

    # Criar sequências temporais
    data = ticker_data[columns_to_scale].values
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, 1])  # Apenas o preço de fechamento (Close)
    X, y = np.array(X), np.array(y)

    # Dividir os dados em treino e teste
    if len(X) == 0 or len(y) == 0:  # Ignorar tickers com dados insuficientes
        print(f"Dados insuficientes para o ticker {ticker}. Pulando...")
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Criar o modelo LSTM
    model = Sequential([
        LSTM(300, return_sequences=True, input_shape=(sequence_length, X.shape[2])),
        LSTM(300),
        Dense(1)  # Prever apenas o preço de fechamento (Close)
    ])

    # Compilar o modelo
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Treinar o modelo
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test)

    # Reverter a normalização para calcular as métricas no domínio original
    y_test_original = scaler.inverse_transform(
        np.hstack((np.zeros((len(y_test), len(columns_to_scale) - 1)), y_test.reshape(-1, 1))))[:, -1]
    y_pred_original = scaler.inverse_transform(np.hstack((np.zeros((len(y_pred), len(columns_to_scale) - 1)), y_pred)))[
                      :, -1]

    # Calcular as métricas de desempenho
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mse = mean_squared_error(y_test_original, y_pred_original)
    mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100

    performance_metrics.append({
        "Ticker": ticker,
        "MAE": mae,
        "MSE": mse,
        "MAPE (%)": mape
    })

    print(f"Desempenho para {ticker} - MAE: {mae:.2f}, MSE: {mse:.2f}, MAPE: {mape:.2f}%")

    # Salvar o modelo para o ticker
    h5_file = os.path.join(output_dir, f"lstm_{ticker}.h5")
    model.save(h5_file)
    h5_files.append(h5_file)
    print(f"Modelo salvo como {h5_file}")

# ******************************************************************#
# break # Remove aqui para pegar todos os Tickers

# Compactar todos os arquivos .h5 em um arquivo .zip
zip_filename = "models.zip"
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for h5_file in h5_files:
        zipf.write(h5_file, os.path.basename(h5_file))

# Salvar as métricas de desempenho em um arquivo CSV
metrics_df = pd.DataFrame(performance_metrics)
metrics_df.to_excel("performance_metrics.xlsx", index=False)
print("Métricas de desempenho salvas em performance_metrics.xlsx")
