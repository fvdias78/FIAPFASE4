"# FIAPFASE4" 

#Modelo de predição do SP500 usando LSTM

Artefatos:

- scrapping_yfinance.py

	Realiza a importação de dados do yfinance, obtém os dados SP500 e salva em um arquivo excel.

- train_models_lstm.py
	
	Faz o enriquecimento dos dados, com indicadores técnicos de ações como média movel, RSI, ROC e Treina um modelo por Ticker (Ação) exemplo: terá um modelo treinado e salvo em .h5 para AAPL (Apple) lstm_aapl.h5, para Nvda (Nvidia) lstm_nvda.h5 e assim por diante. Os modelos treinados serão salvo em .zip em models.zip

- spApiPredict.py
	
	é API que realiza a inferência do modelo, ao invocar a previsão de um Ticker (AAPL), verifica se existe o modelo em models.zip, e traz o resultado.
	Verifica o desempenho em info na console cmd


Modo de uso:
	
	python spApiPredict.py

	Para realizar o teste com uma data anterior basta colocar o campo date: no body do Json:

	curl --location 'http://localhost:8080/predict' \
	     --header 'Content-Type: application/json' \
             --data '{
  		"ticker": "AAPL",
  		"date": "2024-11-25"
	}'

	Para realizar a previsão de amanhã, basta remover o campo date:
	
	curl --location 'http://localhost:8080/predict' \
	     --header 'Content-Type: application/json' \
             --data '{
  		"ticker": "AAPL"
	}'
	
Desempenho:	

	para verificar o desempenho será informado no console cmd, aonde foi executado o python terá um log.info informando  o desempenho da request.

Considerações:

	Não foi utilizado banco de dados para armazenamento do modelo, poderar ser otimizado para isso

