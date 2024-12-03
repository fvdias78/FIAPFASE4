from flask import Flask, request, jsonify
import zipfile
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime
import os
os.environ["FLASK_SKIP_DOTENV"] = "1"
app = Flask(__name__)

atr = 3
rsi =3
roc = 3
sma10 = 5
sma50 = 10

# Função para carregar o modelo do arquivo ZIP
def load_model_from_zip(ticker, zip_path="models.zip"):
    model_filename = f"lstm_{ticker}.h5"
    with zipfile.ZipFile(zip_path, 'r') as z:
        if model_filename in z.namelist():
            z.extract(model_filename)  # Extrai o modelo temporariamente
            model = load_model(model_filename)  # Carrega o modelo
            os.remove(model_filename)  # Remove o modelo extraído após carregar
            return model
        else:
            raise FileNotFoundError(f"Modelo para o ticker {ticker} não encontrado no arquivo {zip_path}.")


# Função para calcular o RSI
def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Função para calcular o ROC
def calculate_roc(data, window=14):
    return ((data - data.shift(window)) / data.shift(window)) * 100


# Função para calcular o ATR
def calculate_atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    return true_range.rolling(window=window).mean()


# Função para obter os dados históricos do Yahoo Finance
def get_data_from_yfinance(ticker, start_date="2000-01-01"):
    print('GetDataYfinance')
    ytk = yf.Ticker(ticker)
    df = ytk.history(start=start_date, interval="1d")
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].dt.tz_localize(None)  # Remover timezone

    # Calcular indicadores técnicos
    df['RSI'] = calculate_rsi(df['Close'], window=rsi)
    df['SMA10'] = df['Close'].rolling(window=sma10).mean()
    df['SMA50'] = df['Close'].rolling(window=sma50).mean()
    df['ROC10'] = calculate_roc(df['Close'], window=roc)
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], window=atr)

    # Substituir valores NaN gerados pelos cálculos
    df.fillna(0, inplace=True)
    return df


# Função para normalizar os dados
def normalize_data(df):
    scaler = MinMaxScaler()
    columns_to_scale = ['Open', 'Close', 'High', 'Low', 'Volume', 'RSI', 'SMA10', 'SMA50', 'ROC10', 'ATR']
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    return df, scaler


# Função para obter a sequência necessária para o modelo
def get_data_for_ticker(ticker, date=None, sequence_length=4):
    # Obter os dados históricos do ticker
    df = get_data_from_yfinance(ticker)

    # Normalizar os dados
    df, scaler = normalize_data(df)

    # Converter a data fornecida para datetime
    if date:
        date = pd.to_datetime(date)

    if date:
        # Obter o índice da data selecionada
        index = df[df['Date'] == date].index
        if index.empty:
            raise ValueError(f"Data {date} não encontrada para o ticker {ticker}.")

        # Verificar se há dados suficientes antes da data para criar a sequência
        index = index[0]
        if index < sequence_length:
            raise ValueError(f"Dados insuficientes antes da data {date} para criar a sequência.")

        # Obter a sequência de entrada
        sequence = df.iloc[index - sequence_length:index][
            ['Open', 'Close', 'High', 'Low', 'Volume', 'RSI', 'SMA10', 'SMA50', 'ROC10', 'ATR']].values
    else:
        # Usar os dados mais recentes
        if len(df) < sequence_length:
            raise ValueError(f"Dados insuficientes para o ticker {ticker}.")

        sequence = df.iloc[-sequence_length:][
            ['Open', 'Close', 'High', 'Low', 'Volume', 'RSI', 'SMA10', 'SMA50', 'ROC10', 'ATR']].values

    # Expandir a dimensão para a entrada do modelo (batch_size=1)
    sequence = np.expand_dims(sequence, axis=0)
    return sequence, scaler


# Rota para prever o próximo preço
@app.route('/predict', methods=['POST'])
def predict_next_price():
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').upper()
        date = data.get('date')  # Data opcional

        if not ticker:
            return jsonify({'error': 'O campo "ticker" é obrigatório.'}), 400

        # Converter a data para datetime, se fornecida
        date_obj = None
        if date:
            date_obj = datetime.datetime.strptime(date, '%Y-%m-%d')

        # Carregar o modelo do ticker
        model = load_model_from_zip(ticker)

        # Obter a sequência de dados
        sequence, scaler = get_data_for_ticker(ticker, date_obj)

        # Fazer a previsão
        predicted_scaled = model.predict(sequence)[0][0]

        # Reverter a normalização para o preço original
        dummy_data = np.zeros((1, len(sequence[0][0])))
        dummy_data[0, 1] = predicted_scaled  # A posição 1 corresponde a 'Close'
        predicted_price = scaler.inverse_transform(dummy_data)[0][1]

        # Calcular a data D+1
        if date_obj:
            next_date = date_obj + datetime.timedelta(days=1)
        else:
            # Obter a última data dos dados históricos
            df = get_data_from_yfinance(ticker)
            last_date = df['Date'].iloc[-1]
            next_date = last_date + datetime.timedelta(days=1)

        return jsonify({
            'ticker': ticker,
            'date': str(next_date.date()),  # Retornar a data D+1
            'predicted_price': round(predicted_price, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Executar a API
if __name__ == '__main__':
    app.run(port=8080, host='localhost', debug=True)
