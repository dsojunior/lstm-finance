# 📈 API de Predição de Ativos Financeiros (Deep Learning / LSTM)
Tech Challenge 4 FIAP - Pós Graduação em Machine Learning Engineering

## 📚 Sobre o Projeto

Este projeto consiste em uma API RESTful projetada para realizar inferências de valores de ativos financeiros na bolsa de valores utilizando Redes Neurais Artificiais Recorrentes, especificamente a arquitetura **LSTM (Long Short-Term Memory)**.

O objetivo da construção é demonstrar o ciclo completo de MLOps, desde o treinamento e exportação dos pesos matemáticos até a conteinerização e deploy em nuvem, servindo o modelo através de uma arquitetura escalável e observável. A solução inclui integrações com a biblioteca `yfinance` para captação automatizada de dados históricos, permitindo predições precisas com base em janelas deslizantes de 60 dias de valores.

## 🎯 Objetivos

- Disponibilizar um modelo de Deep Learning (LSTM) em produção através de uma API Flask.
- Automatizar a formatação e captura de dados reais do mercado financeiro integrando com a biblioteca yfinance (Yahoo Finance).
- Fornecer endpoints de previsão de curtíssimo prazo (no nosso cenário, predizer o próximo dia de valores) e previsões estendidas baseadas em auto-regressão (no nosso caso, previsões de até 30 dias).
- Garantir que seja possível uma futura integração de observabilidade da aplicação com Prometheus.
- Assegurar a portabilidade, isolamento de hardware e fácil escalabilidade  utilizando conteinerização com Docker.

## 🧠 Treinamento e Extensibilidade

Para fins do projeto bimestral do Tech Challenge, a API foi configurada por padrão com os pesos treinados apenas da empresa **Nintendo (NTDOY)**. Entretanto, a arquitetura da API foi desenhada para ser totalmente agnóstica ao ativo.

### O Notebook de Modelagem
Na pasta `/notebooks`, encontra-se o arquivo `desenvolvimento_lstm.ipynb`. Este notebook contém o pipeline completo de:
1. **Data Discovery:** Coleta de dados históricos via biblioteca yfinance.
2. **Feature Engineering:** Normalização técnica e criação de janelas temporais de 60 dias.
3. **Model Design:** Arquitetura da rede neural recorrente (LSTM).
4. **Validation:** Cálculo de métricas de erro (MAE, MAPE e RMSE).

### Como Adaptar para Outras Empresas
Para expandir o uso dessa aplicação para ações / ativos de outras empresas como Apple (AAPL), Google (GOOGL) ou Petrobras (PETR4.SA), basta:
1. Executar o arquivo `obter_dados.py` na pasta `scripts`, indicando qual ativo será feito a obtenção de dados.
2. Trabalhar no arquivo `desenvolvimento_lstm.ipynb`, alterando a variável `acao_empresa` no início do notebook de treinamento.
2. Executar o treinamento para que a rede aprenda os padrões de volatilidade específicos desse novo ativo. Se necessário, altere os hiperparâmetros para um melhor desempenho.
3. Exportar o novo arquivo `.keras` e o novo `.save` (scaler) para a pasta `/model`.
4. A API detectará o novo modelo automaticamente através das rotas dinâmicas, permitindo consultas imediatas via URL.

## 🚀 Como Reproduzir

### Execução Local (Python)

1. **Clone o repositório:**
  ```bash
  git clone https://github.com/seu-usuario/lstm-finance.git
  cd lstm-finance
  ```

2. **Instale as dependências:**
  ```bash
  pip install -r requirements.txt
  ```

3. **Inicie a API, executando o arquivo na pasta 'api':**
  ```bash
  python app.py
  ```

### Execução via Docker (recomendado)
1. **Construa a imagem da aplicação:**
  ```bash
  docker build -t lstm-predictor .
  ```

2. **Inicie o contêiner mapeando a porta 5001:**
  ```bash
  docker run -p 5001:5001 lstm-predictor
  ```

A API estará acessível em `http://localhost:5001/apidocs/`

## 🛠️ Principais Tecnologias

- Python (Backend e Modelagem)
- TensorFlow / Keras (Motor de Deep Learning)
- Flask (Framework Web RESTful)
- Flasgger / Swagger (Documentação interativa OpenAPI)
- yfinance (Integração de dados financeiros)
- Prometheus (Telemetria e Observabilidade)
- Docker (Conteinerização)

## 🔗 Endpoints da API

- `GET ​/`  
  Health Check da aplicação. Redireciona automaticamente para a interface do Swagger.

- `GET /obter_valores_acao/{ticker}?data_inicio={data}&data_fim={data}`  
  Comunica-se com o Yahoo Finance e obtém o histórico exato dos últimos 60 dias úteis de fechamento do ativo, já formatado para a predição.

- `GET /gerar_dados_teste?preco_base={valor}`  
  Gera dados sintéticos de 60 dias utilizando a técnica de Random Walk, com o intuito de testes de carga / QA sem consumir APIs externas.

- `POST /predict/{acao_empresa}`  
  Rota principal da predição. Recebe um JSON de 60 dias de histórico e realiza a predição do próximo dia útil utilizando a rede LSTM correspondente ao ativo.

- `POST /predict_recursive/{acao_empresa}/{dias_futuros}`  
  Rota de predição usando de mais de uma data. Utiliza auto-regressão geométrica e janelas dinâmicas para prever `X` dias no futuro.

- `GET /metrics`  
  Ponto de coleta de telemetria nativa do Prometheus. Expõe métricas de uso de memória, latência de inferência (tempo de resposta do modelo) e contagem de erros (SLIs).

## 📦 Estrutura dos Dados e Artefatos

O projeto utiliza **Lazy Loading** com cache em memória para os modelos e escalonadores, garantindo alta performance para múltiplas requisições. Para adicionar uma nova empresa à API, após o treinamento e exportação do modelo, basta incluir novos arquivos na pasta /model:

- `modelo_{acao_empresa}_lstm.keras`: O modelo neural treinado salvo no formato Keras.
- `scaler_{acao_empresa}.save`: O transformador (ex: MinMaxScaler) utilizado para normalizar as entradas entre 0 e 1,  para o cálculo tensorial não se perder nas ordens de grandeza.

A API espera como entrada, além da indicação do ativo para seleção do modelo, um JSON contendo a matriz histórica:

  ```JSON
  {
  "precos_historicos": [15.2, 15.5, 15.4, ...] // Exatamente 60 valores float
  }
  ```

## 🌐 Link do Deploy

O projeto está disponível em produção, conteinerizado através do **Render**, no seguinte endereço:

- Swagger UI: [https://lstm-finance.onrender.com/apidocs/](https://lstm-finance.onrender.com/apidocs/)
- Contrato OpenAPI: [https://lstm-finance.onrender.com/apispec_1.json](https://lstm-finance.onrender.com/apispec_1.json)
- Monitoramento Prometheus: [https://lstm-finance.onrender.com/metrics](https://lstm-finance.onrender.com/apispec_1.json)

*Obs: Como o deploy foi feito no Render com o plano gratuito, há uma espécie de "hibernação" em APIs que não são muito chamadas, o que provavelmente acontecerá no momento do teste com as rotas acima. Quando a API está "hibernando", ela demora em média 40 segundos para "voltar a vida" na primeira chamada (é exibido uma tela do próprio Render como se estivesse sendo feito um deploy, conforme abaixo). Após isso, as chamadas voltam a funcionar de forma consistente na sequencia*

![Imagem da API saindo da hibernação - Plano gratuito do Render](docs/Saida%20hibernação%20render%20plano%20gratuito.png)

## 🎥 Vídeo de Demonstração

Assista a uma explicação geral do funcionamento do projeto, bem como demosntração de chamadas realizadas em produção:

[https://www.youtube.com/watch?v=ja6G0jjo7mk](https://www.youtube.com/watch?v=ja6G0jjo7mk)

## 🗂️ Decisões Arquiteturais e Observações

- **Previsão de 1 dia vs Previsão de X dias**: Usualmente, o mercado financeiro comporta-se como um ambiente imprevisível. A rota padrão da API prevê apenas 1 passo no futuro para garantir uma margem de erro (MAPE) controlada (~3%). A rota recursiva que prevê mais dias foi incluída para fins analíticos, mas há um risco matemático de degradação e propagação de erros, pois se utiliza de dados simulados como dados "reais".
- **Otimização de Infraestrutura**: O `Dockerfile` foi estruturado para forçar a simulação da plataforma linux/amd64 (garantindo compatibilidade na nuvem, mesmo quando construído em processadores do tipo ARM/Apple Silicon, que foi o meu caso) e utiliza a versão `tensorflow-cpu` para otimização de custo/memória em ambientes Serverless/PaaS nos planos gratuitos.
- As variavéis, docstrings, comentários e tratativas de erro, a fim de auxiliar a explicação e avaliação do trabalho, foram escritos em português
