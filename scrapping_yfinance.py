import pandas as pd
import yfinance as yf

# URLs para obter os tickers
url_nasdaq100 = "https://en.wikipedia.org/wiki/NASDAQ-100"
url_sp500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

atr = 3
rsi =3
roc = 3
sma10 = 5
sma50 = 10

# Função para obter tickers do NASDAQ-100
def get_nasdaq100_tickers():
    nasdaq100_tables = pd.read_html(url_nasdaq100)
    df_nasdaq100 = nasdaq100_tables[0]  # Geralmente, a tabela relevante é a primeira
    return df_nasdaq100['Symbol'].tolist()


# Função para obter os 50 mais valiosos do S&P 500
def get_sp500_tickers(top=50):
    tables = pd.read_html(url_sp500)
    df_sp500 = tables[0]  # Geralmente, a tabela relevante é a primeira

    # Adicionar uma coluna para Market Cap
    df_sp500['Market Cap'] = None

    # Obter o Market Cap de cada empresa usando yfinance
    for index, row in df_sp500.iterrows():
        try:
            ticker = row['Symbol']
            stock = yf.Ticker(ticker)
            market_cap = stock.info.get('marketCap', 0)  # Obter o Market Cap (0 se não disponível)
            df_sp500.at[index, 'Market Cap'] = market_cap
        except Exception as e:
            print(f"Erro ao buscar dados para {row['Symbol']}: {e}")

    # Converter Market Cap para float
    df_sp500['Market Cap'] = pd.to_numeric(df_sp500['Market Cap'], errors='coerce')

    # Classificar pelo Market Cap em ordem decrescente
    top_x = df_sp500.sort_values(by='Market Cap', ascending=False).head(top)

    return top_x['Symbol'].tolist()


# Função para baixar dados das ações
def download_stock_data(tickers, start_date):
    data_list = []
    for ticker in tickers:
        try:
            newdf = pd.DataFrame()
            ytk = yf.Ticker(ticker)
            df = ytk.history(start=start_date, interval="1d")
            df.reset_index(inplace=True)

            newdf['Date'] = df['Date'].dt.tz_localize(None)
            newdf['Ticker'] = ticker
            newdf['Open'] = df['Open']
            newdf['Close'] = df['Close']
            newdf['High'] = df['High']
            newdf['Low'] = df['Low']
            newdf['Volume'] = df['Volume']
            data_list.append(newdf)
        except Exception as e:
            print(f"Erro ao baixar dados para {ticker}: {e}")
    return data_list


# Função principal
def main():
    # Obter tickers
    # tickers_nasdaq100 = get_nasdaq100_tickers()
    tickers_sp500 = get_sp500_tickers(5) #Aqui coloquei somente os TOP 5

    # Combinar tickers de ambas as listas
    all_tickers = tickers_sp500  # + tickers_nasdaq100

    # Baixar dados
    print("Baixando dados das ações...")
    data_list = download_stock_data(all_tickers, start_date="2010-05-01")

    # Concatenar todos os dados em um único DataFrame
    final_df = pd.concat(data_list, ignore_index=True)

    # Salvar como Excel
    output_file = "sp500_nasdaq100_data.xlsx"
    final_df.to_excel(output_file, index=False)
    print(f"Dados salvos no arquivo: {output_file}")


# Executar o script
if __name__ == "__main__":
    main()
