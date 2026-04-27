from flask import Flask, request, jsonify, redirect
from flasgger import Swagger
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import yfinance as yf
import random
from prometheus_flask_exporter import PrometheusMetrics

# Inicialização do app Flask, do Swagger e dos logs do Prometheus
app = Flask(__name__)

metrics = PrometheusMetrics(app, path='/metrics')
metrics.info('app_info', 'API Predição de Ativos', version='1.0.0')
# Porta 5099 para o Prometheus, para evitar conflito com a porta da API (5001), entretanto removendo por conta do plano gratuito do Render não suportar
#metrics.start_http_server(5099) 

app.config['SWAGGER'] = {
    'ui_params': {
        'defaultModelsExpandDepth': -1
    }
}

swagger = Swagger(app, template={
    "info": {
        "title": "API Predição de Ativos - utilizando modelo LSTM",
        "description": "API para predição de ativos financeiros utilizando Deep Learning (LSTM).",
        "version": "1.0.0",
    }
})

# Configuração de caminhos baseada na estrutura do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'model')

# Cache em memória para evitar IO excessivo de disco
modelos_cache = {}
scalers_cache = {}

def carregar_recursos(acao_empresa):
    """
    Gerencia o carregamento sob demanda dos arquivos de modelo e scaler.
    """
    acao_empresa = acao_empresa.upper()
    
    # Verifica se os artefatos já estão no cache de memória
    if acao_empresa in modelos_cache and acao_empresa in scalers_cache:
        return modelos_cache[acao_empresa], scalers_cache[acao_empresa]
    
    # Define os nomes de arquivos esperados seguindo o padrão
    modelo_path = os.path.join(MODELS_DIR, f'modelo_{acao_empresa}.keras')
    scaler_path = os.path.join(MODELS_DIR, f'scaler_{acao_empresa}.save')
    
    # Validação de existência física dos arquivos
    if not os.path.exists(modelo_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Não foi encontrado nenhum modelos treinado para o ativo {acao_empresa}")
    
    # Carregamento e atualização do cache
    modelos_cache[acao_empresa] = load_model(modelo_path)
    scalers_cache[acao_empresa] = joblib.load(scaler_path)
    
    return modelos_cache[acao_empresa], scalers_cache[acao_empresa]

@app.route('/', methods=['GET'])
def health_check():
    """
    Rota raiz para o Health Check do Render. Redireciona para a documentação.
    """
    return redirect('/apidocs/')

@app.route('/predict/<acao_empresa>', methods=['POST'])
def predict(acao_empresa):
    """
    Realiza a predicao do proximo valor de fechamento para um ativo especifico.
    ---
    tags:
      - Predição de valores de ativos
    parameters:
      - name: acao_empresa
        in: path
        type: string
        required: true
        description: O codigo da empresa na bolsa (exemplo NTDOY, AAPL, DIS)
      - name: body
        in: body
        required: true
        schema:
          id: DadosEntrada
          required:
            - precos_historicos
          properties:
            precos_historicos:
              type: array
              description: Lista com os ultimos 60 precos de fechamento
              items:
                type: number
    responses:
      200:
        description: Predicao gerada com sucesso
        schema:
          properties:
            acao_empresa:
              type: string
            predição_próximo_preço:
              type: number
      404:
        description: Modelo ou Scaler do ativo nao encontrado
      400:
        description: Erro no formato dos dados de entrada
    """
    try:
        dados = request.get_json()
        precos_historicos = dados.get('precos_historicos', [])
        
        # Validação do requisito de 60 dias
        if len(precos_historicos) != 60:
            return jsonify({
                'erro': f'Formato inválido. São necessários 60 dias de histórico. Recebidos: {len(precos_historicos)}'
            }), 400

        # Carregamento dinâmico dos artefatos (ou recuperação do cache)
        modelo_lstm, scaler = carregar_recursos(acao_empresa)
        
        # Processamento e Normalização
        precos_array = np.array(precos_historicos).reshape(-1, 1)
        precos_scaled = scaler.transform(precos_array)
        
        # Reshape para formato (Amostras, Time Steps, Features)
        X_input = np.reshape(precos_scaled, (1, 60, 1))
        
        # Inferência
        predicao_scaled = modelo_lstm.predict(X_input)
        
        # Desnormalização para valor real
        predicao_real = scaler.inverse_transform(predicao_scaled)
        resultado = round(float(predicao_real[0][0]), 2)
        
        return jsonify({
            'acao_empresa': acao_empresa.upper(),
            'predição_próximo_preço': resultado,
        }), 200

    except FileNotFoundError as e:
        return jsonify({'erro': str(e)}), 404
    except Exception as e:
        return jsonify({'erro': f'Erro interno no processamento: {str(e)}'}), 500
    
@app.route('/obter_valores_acao/<acao_empresa>', methods=['GET'])
def obter_valores_acao(acao_empresa):
    """
    Busca o histórico de preços utilizando a biblioteca yfinance e formata para a entrada da rota /predict.
    Os parametros de data_inicio e data_fim são opcionais, o esperado é que o usuário forneça um intervalo que contenha pelo menos 60 dias úteis. 
    Caso seja fornecido um intervalo que contenha mais de 60 dias de dados do ativo, a função irá retornar os ultimos 60 dias de valores.
    Caso não sejam fornecidos os parâmetros de data, a função irá buscar os últimos 3 meses de dados para garantir uma margem de segurança em relação aos dias úteis.
    ---
    tags:
      - Geração de dados para teste
    parameters:
      - name: acao_empresa
        in: path
        type: string
        required: true
        description: "O codigo da empresa na bolsa (exemplo NTDOY, AAPL, DIS)"
      - name: data_inicio
        in: query
        type: string
        required: false
        description: "Data de início (Formato YYYY-MM-DD). Opcional."
      - name: data_fim
        in: query
        type: string
        required: false
        description: "Data de fim (Formato YYYY-MM-DD). Opcional."
    responses:
      200:
        description: Payload gerado com sucesso para uso no endpoint /predict
      400:
        description: Quantidade insuficiente de dias úteis encontrados
      404:
        description: Ação não encontrada através da biblioteca yfinance
    """
    # Capturando as datas passadas na URL (Query Parameters)
    data_inicio = request.args.get('data_inicio')
    data_fim = request.args.get('data_fim')
    
    try:
        ativo = yf.Ticker(acao_empresa.upper())
        
        # Se as datas foram fornecidas, busca no intervalo exato.
        # Caso contrário, busca os últimos 3 meses para garantir folga de dias úteis.
        if data_inicio and data_fim:
            hist = ativo.history(start=data_inicio, end=data_fim)
        else:
            hist = ativo.history(period="3mo")
            
        if hist.empty:
            return jsonify({'erro': f'Nenhum dado encontrado para o ativo {acao_empresa.upper()}'}), 404
            
        # Extrai a coluna de fechamento do valor do ativo (coluna 'Close') e converte para uma lista simples
        precos = hist['Close'].tolist()
        
        # Validação para permitir estritamente 60 dias de valores histórico
        if len(precos) < 60:
            return jsonify({
                'erro': f'O intervalo retornou apenas {len(precos)} dias úteis. O modelo exige no mínimo 60 dias.'
            }), 400
            
        # Pega exatamente os últimos 60 dias da lista encontrada
        ultimos_60_dias = precos[-60:]
        
        # Arredonda para 2 casas decimais para manter a consistência financeira
        ultimos_60_dias = [round(preco, 2) for preco in ultimos_60_dias]
        
        # Retorna a estrutura JSON exata que o endpoint /predict exige
        return jsonify({
            "precos_historicos": ultimos_60_dias
        }), 200
        
    except Exception as e:
        return jsonify({'erro': f'Erro na comunicação com o Yahoo Finance: {str(e)}'}), 500
    
import random

@app.route('/gerar_dados_teste', methods=['GET'])
def gerar_dados_teste():
    """
    Gera 60 valores simulados aleatórios com comportamento de bolsa para testes.
    É utilizado uma estratégia de Passeio Aleatório (Random Walk) com flutuação leve (-3% a +3%), para garantir que os dados simulados tenham um comportamento mais realista as flutuações de preços de ações.
    Se não for passado nenhum valor, será assumido um valor padrão de 15.00
    ---
    tags:
      - Geração de dados para teste
    parameters:
      - name: preco_base
        in: query
        type: number
        required: false
        description: "Valor inicial da acao para iniciar a simulacao (Padrao: 15.00)"
    responses:
      200:
        description: "Payload simulado gerado com sucesso para uso no endpoint /predict"
        schema:
          properties:
            precos_historicos:
              type: array
              items:
                type: number
      400:
        description: "O parametro fornecido nao e um numero valido"
    """
    try:
        # Captura o parâmetro da URL. Se não for passado, assume 15.00. 
        preco_param = request.args.get('preco_base', default=15.00, type=float)
        
        preco_atual = preco_param
        precos_simulados = []
        
        for _ in range(60):
            # Estratégia do Passeio Aleatório (Random Walk) com flutuação leve (-3% a +3%)
            variacao_diaria = random.uniform(-0.03, 0.03)
            preco_atual = preco_atual * (1 + variacao_diaria)
            
            # Adiciona à lista arredondando para 2 casas decimais
            precos_simulados.append(round(preco_atual, 2))
            
        return jsonify({
            "precos_historicos": precos_simulados
        }), 200

    except ValueError:
        return jsonify({'erro': 'O parâmetro preco_base deve ser um número válido.'}), 400
    except Exception as e:
        return jsonify({'erro': f'Erro ao gerar os dados de teste: {str(e)}'}), 500

@app.route('/predict_recursive/<acao_empresa>/<int:dias_futuros>', methods=['POST'])
def predict_recursive(acao_empresa, dias_futuros = 5):
    """
    Realiza a predicao de X dias no futuro usando recursividade (Auto-regressao).
    Usar com cuidado esse endpoint, pois a qualidade das previsões tende a se degradar quanto mais dias no futuro forem solicitados, devido ao acúmulo de erros.
    ---
    tags:
      - Predição de valores de ativos
    parameters:
      - name: acao_empresa
        in: path
        type: string
        required: true
        description: "Codigo da empresa na bolsa"
      - name: dias_futuros
        in: path
        type: integer
        required: true
        description: "Quantidade de dias no futuro para prever (Limite de 30 dias, por padrão 5 dias)"
      - name: body
        in: body
        required: true
        schema:
          id: DadosEntradaRecursiva
          required:
            - precos_historicos
          properties:
            precos_historicos:
              type: array
              description: "Lista com 60 precos de fechamento"
              items:
                type: number
    responses:
      200:
        description: "Lista de predições gerada com sucesso"
      400:
        description: "Erro de validacao"
    """
    try:
        dados = request.get_json()
        precos_historicos = dados.get('precos_historicos', [])
        
        if len(precos_historicos) != 60:
            return jsonify({
                'erro': f'Formato invalido. Sao necessarios 60 dias de historico. Recebidos: {len(precos_historicos)}'
            }), 400
            
        if dias_futuros <= 0 or dias_futuros > 30:
            return jsonify({'erro': 'O limite seguro para dias_futuros é entre 1 e 30 dias. Mais do que isso há grandes chances do numero nao fazer sentido'}), 400

        modelo_lstm, scaler = carregar_recursos(acao_empresa)
        
        # Prepara a janela inicial normalizada
        precos_array = np.array(precos_historicos).reshape(-1, 1)
        janela_atual_scaled = scaler.transform(precos_array)
        
        # Lista vazia para ir guardando as previsoes na escala 0 a 1
        previsoes_futuras_scaled = []
        
        # Auto-regressão
        for _ in range(dias_futuros):
            # Formata a janela atual para o tensor 3 dimensoes exigido
            X_input = np.reshape(janela_atual_scaled, (1, 60, 1))
            
            # Faz a previsao do proximo dia
            proxima_predicao = modelo_lstm.predict(X_input)
            
            # Guarda a previsao na lista de resultados
            previsoes_futuras_scaled.append(proxima_predicao[0, 0])
            
            # Janela de datas deslizante: Remove o dia mais antigo (indice 0) e adiciona a nova previsao no final
            janela_atual_scaled = np.append(janela_atual_scaled[1:], proxima_predicao, axis=0)
            
        # onverte a lista de previsões (0 a 1) de volta para Dolares
        previsoes_array_scaled = np.array(previsoes_futuras_scaled).reshape(-1, 1)
        previsoes_reais = scaler.inverse_transform(previsoes_array_scaled)
        
        # Arredonda todos os valores previstos e formata como lista
        resultados_finais = [round(float(valor[0]), 2) for valor in previsoes_reais]
        
        return jsonify({
            'acao_empresa': acao_empresa.upper(),
            'dias_previstos': dias_futuros,
            'predicao_precos': resultados_finais
        }), 200

    except FileNotFoundError as e:
        return jsonify({'erro': str(e)}), 404
    except Exception as e:
        return jsonify({'erro': f'Erro interno: {str(e)}'}), 500

if __name__ == '__main__':
    # Porta 5001 para evitar conflito com porta ja usada no macOS
    app.run(debug=False, host='0.0.0.0', port=5001)