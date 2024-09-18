from utils import *
import pandas as pd
import os
import numpy as np

def main():
    files = os.listdir('../upload')
    files = [f for f in files if f.endswith('.csv')]
    
    if len(files) == 0:
        print('No files to process')
        return

    df_dict = {}
    
    for file in files:
        name = file.split('.')[0]
        print('Processing:', name)
        
        try:
            df = pd.read_csv(f'../upload/{file}', encoding='utf-8-sig')
            df = preprocess_check(name, df)
            df_dict[name] = df
        except Exception as e:
            print(f'Failed to process {file}: {e}')

    for key, df in df_dict.items():
        server = 'JPC00228406.jp.sony.com'
        database = 'ELCONDOR_SG_DEV_NEW_DB'
        # database = 'EC_SG_DEV_TEST'
        username = 'sa'
        password = 'ElCondor2.0!$'
        
        try:
            save_to_sql_server(df, key, server, database, username, password, if_exists='append')
            print(f'Data saved successfully for: {key}')
        except Exception as e:
            print(f'Failed to save data for {key}: {e}')

if __name__ == "__main__":
    main()
