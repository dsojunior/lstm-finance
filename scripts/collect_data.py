import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def getAndSaveData(ticker, date_start, date_end, path_output):
    """
    Baixa os dados históricos de um ticker via yfinance e salva em CSV de forma limpa.
    """
    print(f"Starting download of ticker {ticker} between {date_start} and {date_end}...")
    
    try:
        # Download dos dados
        df = yf.download(ticker, start=date_start, end=date_end)
        
        if df.empty:
            print(f"WARNING: No data returned for ticker {ticker}.")
            return
        
        # Remove a segunda linha do cabeçalho, caso exista
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        df.reset_index(inplace=True)
        
        # Montando o caminho e o nome do arquivo
        file_name = f"historical_data_{ticker}.csv"
        path_file = os.path.join(path_output, file_name)
        
        # Salvando em formato CSV na pasta data/
        df.to_csv(path_file, index=False)
        print(f"SUCCESS! Data saved to: {path_file}")
        print(f"Total records obtained: {len(df)}\n")
        
    except Exception as e:
        print(f"Error processing ticker {ticker}: {e}")
if __name__ == "__main__":
    # Configurações
    TICKER = 'NTDOY' #Escolhido indice para download (Nintendo)
    START_DATE = '2012-01-01'
    END_DATE = datetime.today().strftime('%Y-%m-%d') 
    
    # Gerenciamento de caminhos
    path_script = os.path.dirname(os.path.abspath(__file__))
    path_root = os.path.dirname(path_script)
    folder_data = os.path.join(path_root, 'data')
    os.makedirs(folder_data, exist_ok=True)
    
    # --- EXECUÇÃO ---
    print(f"Starting process...")
    print("-" * 50)

    getAndSaveData(TICKER, START_DATE, END_DATE, folder_data)

    print("-" * 50)
    print(f"Process completed.")