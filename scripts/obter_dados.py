import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def obterDados(empresa, data_inicio, data_fim, path_saida):
    """
    Baixa os dados históricos de um empresa via yfinance e salva em CSV.
    """
    print(f"Iniciando download dos dados {empresa} de {data_inicio} até {data_fim}...")
    
    try:
        # Download dos dados
        df = yf.download(empresa, start=data_inicio, end=data_fim)
        
        if df.empty:
            print(f"AVISO: Nenhum dado encontrado para {empresa}.")
            return
        
        # Remove a segunda linha do cabeçalho, caso exista
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        df.reset_index(inplace=True)
        
        # Montando o caminho e o nome do arquivo
        arquivo = f"dados_historicos_{empresa}.csv"
        path_arquivo = os.path.join(path_saida, arquivo)
        
        # Salvando em formato CSV na pasta data/
        df.to_csv(path_arquivo, index=False)
        print(f"Sucesso! Dados salvos em: {path_arquivo}")
        print(f"Registros obtidos: {len(df)}\n")
        
    except Exception as e:
        print(f"Erro processando dados de {empresa}: {e}")

if __name__ == "__main__":
    # Configurações
    EMPRESA = 'NTDOY' #Escolhido indice para download (Nintendo)
    DATA_INICIO = '2012-01-01'
    DATA_FIM = datetime.today().strftime('%Y-%m-%d') 
    
    # Gerenciamento de caminhos
    path_script = os.path.dirname(os.path.abspath(__file__))
    path_raiz = os.path.dirname(path_script)
    pasta_data = os.path.join(path_raiz, 'data')
    os.makedirs(pasta_data, exist_ok=True)
    
    # --- EXECUÇÃO ---
    print(f"Iniciando processo...")
    print("-" * 50)

    obterDados(EMPRESA, DATA_INICIO, DATA_FIM, pasta_data)

    print("-" * 50)
    print(f"Processo concluído.")