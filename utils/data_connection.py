from sqlalchemy import create_engine
import pandas as pd
import time
a = 1
db_data = {}
db2_data = {}
psicontainer = {'test':None}


connection_info = { 'server': '',
                    'database':'' ,
                    'username':'' ,
                    'password':'',
                    'driver':'ODBC Driver 17 for SQL Server',
                    'sales_company':''
                   }


global_progress = {
    'CH_PSI_ACT': {'processed': 0, 'total': 0},  # 仮の値
    'CH_PSI_FCT': {'processed': 0, 'total': 0},  # 仮の値
    'SC_PSI_FCT': {'processed': 0, 'total': 0},  # 仮の値
    'SC_PSI_LIVE': {'processed': 0, 'total': 0},  # 仮の値
    'SEIHAN_SETTING': {'processed': 0, 'total': 0},  # 仮の値
    'SELLIN_PARAM': {'processed': 0, 'total': 0},  # 仮の値
    'PRODUCT_MAP': {'processed': 0, 'total': 0},  # 仮の値
    'DMD_FCT_ACCOUNT_SPLIT': {'processed': 0, 'total': 0},  # 仮の値
    'FIXED_SELLIN_ADJ': {'processed': 0, 'total': 0},  # 仮の値
}

def load_data(connection_string, table_filters=None, callback=None, scn_filter=[], sku_filter=[], cat_filter=[], cty_filter=[], stt_year=2023):
    """
    Pull and load data from the database into global variable db_data
    """
    global db_data
    print("Loading data...")
    db_data = query_data_global_2(connection_string, table_filters=table_filters, callback=callback, scn_filter=scn_filter, sku_filter=sku_filter, cat_filter=cat_filter, cty_filter=cty_filter, stt_year=2023)
    print("Data loaded.")
    print(db_data.keys())
    print('Number of tables loaded:', len(db_data))

    # for key, df in db_data.items():
    #     print(f"Table: {key}, df.dtypes: {df.dtypes}")
        
    # total_mem = 0
    # for key, df in db_data.items():
    #     mem = df.memory_usage(deep=True, index=True).sum()
    #     total_mem += mem
    #     print(f'Memory usage for table {key}: {mem / 1024**2:.2f} MB')
    
    # print(f"Total memory usage: {total_mem / 1024**2:.2f} MB")
    
    # # loop through and print duplicates
    # for key, df in db_data.items():
    #     if df.duplicated().sum() > 0:
    #     print(f"Table: {key}, Number of duplicates: {df.duplicated().sum()}")
        # print("\n")

def query_data_global(connection_string, table_filters=None,callback=None,year_filter=None):
    """
    Create connection to the database and query all tables
    
    Returns: dict of DataFrames
    """
    data_dict = {}
    engine = create_engine(connection_string)
    try:
        table_queries = {
            "time_w": "SELECT * FROM TIME_W",
            "time_m": "SELECT * FROM TIME_M",
            "cat_map": "SELECT * FROM CAT_MAP",
            "cat_map2": "SELECT * FROM CAT_MAP2",
            "sub_cat_map": "SELECT * FROM SUB_CAT_MAP",
            "scenario_map": "SELECT * FROM SCENARIO_MAP",
            "account_map": "SELECT * FROM ACCOUNT_MAP",
            "fct_account_map": "SELECT * FROM FCT_ACCOUNT_MAP",
            "cat_account_map": "SELECT * FROM CAT_ACCOUNT_MAP",
            "sc_map": "SELECT * FROM SC_MAP",
            "product_map": "SELECT * FROM PRODUCT_MAP",
            "fct_type": "SELECT * FROM FCT_TYPE",
            "dmd_type": "SELECT * FROM DMD_TYPE",
            
            #### ACT ####
            "ch_psi_act": "SELECT * FROM CH_PSI_ACT",
            "financial_act": "SELECT * FROM FINANCIAL_ACT",
            "sc_psi_live": "SELECT * FROM SC_PSI_LIVE",
            #### SI ####
            "sellin_param": "SELECT * FROM SELLIN_PARAM",
            "supply_adj": "SELECT * FROM SUPPLY_ADJ",
            "sellin_adj": "SELECT * FROM SELLIN_ADJ",
            "seihan_param": "SELECT * FROM SEIHAN_PARAM",
            #### PRICING ####
            "price_structure": "SELECT * FROM PRICE_STRUCTURE",
            "mp_cost": "SELECT * FROM MP_COST",
            "exchange_rate": "SELECT * FROM EXCHANGE_RATE",
            #### DMD ####
            "demand_split": "SELECT * FROM DEMAND_SPLIT",
            "pricing": "SELECT * FROM PRICING",
            "event": "SELECT * FROM EVENT",
            "seasonality": "SELECT * FROM SEASONALITY",
            "factor": "SELECT * FROM FACTOR",
            "factor_acct": "SELECT * FROM FACTOR_ACCT",
            "seasonality_acct": "SELECT * FROM SEASONALITY_ACCT",
            "dmd_fct": "SELECT * FROM DMD_FCT",
            "dmd_param_fct_window": "SELECT * FROM DMD_PARAM_FCT_WINDOW",
            #### FCT ####
            "sc_psi_fct": "SELECT * FROM SC_PSI_FCT",
            "ch_psi_fct": "SELECT * FROM CH_PSI_FCT",
            "fct_flag": "SELECT * FROM FCT_FLAG",
        }
        # table_filters takes in list of where clauses
        if table_filters:
            for index, (table_name, query) in enumerate(table_queries.items(), start=1):
                if callback is not None:
                    callback(index/len(table_queries))
                print(f'# Loading table: {table_name}...',)
                tm = time.time()
                with engine.connect() as connection:
                    if table_name in ('sellin_param', 'supply_adj', 'sellin_adj', 'seihan_param', 'price_structure', 'mp_cost', 'exchange_rate', 'demand_split', 'pricing', 'event', 'seasonality', 'factor', 'factor_acct', 'seasonality_acct', 'dmd_fct', 
                                      'sc_psi_fct', 'ch_psi_fct', 'fct_flag'):
                        data_dict[table_name] = pd.read_sql_query(query + ' WHERE ' + ' OR '.join([f"SCENARIO = '{filter_}'" for filter_ in table_filters]), connection)
                        if 'WEEK_DATE' in data_dict[table_name].columns:
                            data_dict[table_name]['WEEK_DATE'] = pd.to_datetime(data_dict[table_name]['WEEK_DATE'])
                    
                    elif table_name == 'ch_psi_act' or table_name == 'sc_psi_live' or table_name == 'financial_act':
                        data_dict[table_name] = pd.read_sql_query(query + ' WHERE SEIHAN_YEAR>=' + year_filter, connection)
                        if 'WEEK_DATE' in data_dict[table_name].columns:
                            data_dict[table_name]['WEEK_DATE'] = pd.to_datetime(data_dict[table_name]['WEEK_DATE'])
                    
                    else:
                        data_dict[table_name] = pd.read_sql_query(query, connection)
                        if 'WEEK_DATE' in data_dict[table_name].columns:
                            data_dict[table_name]['WEEK_DATE'] = pd.to_datetime(data_dict[table_name]['WEEK_DATE'])
                
                print(f' - {time.time()-tm} secs',)
        else:
            with engine.connect() as connection:
                for table_name, query in table_queries.items():
                    print(f'# Loading table: {table_name}...',)
                    tm = time.time()
                    if table_name == 'ch_psi_act' or table_name == 'sc_psi_live' or table_name == 'financial_act':
                        data_dict[table_name] = pd.read_sql_query(query+ ' WHERE SEIHAN_YEAR>=' + year_filter, connection)
                        if 'WEEK_DATE' in data_dict[table_name].columns:
                            data_dict[table_name]['WEEK_DATE'] = pd.to_datetime(data_dict[table_name]['WEEK_DATE'])
                    else:
                        data_dict[table_name] = pd.read_sql_query(query, connection)
                        if 'WEEK_DATE' in data_dict[table_name].columns:
                            data_dict[table_name]['WEEK_DATE'] = pd.to_datetime(data_dict[table_name]['WEEK_DATE'])
                        if 'SELLIN_QTY_ORI' in data_dict[table_name].columns:
                            data_dict[table_name]['SELLIN_QTY_ORI'] = data_dict[table_name]['SELLIN_QTY_ORI'].astype(float)
                    print(f' - {time.time()-tm} secs',)
    finally:
        engine.dispose()
        
    return data_dict



def query_data_global_2(connection_string, table_filters=None,callback=None, scn_filter=[], sku_filter=[], cat_filter=[], cty_filter=[], stt_year=2023):
    """
    Create connection to the database and query all tables
    
    Returns: dict of DataFrames
    """
    data_dict = {}
    engine = create_engine(connection_string)
    try:
        tvf_list = {
            "time_w": "F_TIME_W",
            "time_m": "F_TIME_M",
            "cat_map": "F_CAT_MAP",
            "cat_map2": "F_CAT_MAP2",
            "sub_cat_map": "F_SUB_CAT_MAP",
            "scenario_map": "F_SCENARIO_MAP",
            "account_map": "F_ACCOUNT_MAP",
            "fct_account_map": "F_FCT_ACCOUNT_MAP",
            "cat_account_map": "F_CAT_ACCOUNT_MAP",
            "sc_map": "F_SC_MAP",
            "product_map": "F_PRODUCT_MAP",
            "fct_type": "F_FCT_TYPE",
            "dmd_type": "F_DMD_TYPE",
            
            #### ACT ####
            "ch_psi_act": "F_CH_PSI_ACT",
            "financial_act": "F_FINANCIAL_ACT",
            "sc_psi_live": "F_SC_PSI_LIVE",
            #### SI ####
            "sellin_param": "F_SELLIN_PARAM",
            "supply_adj": "F_SUPPLY_ADJ",
            "sellin_adj": "F_SELLIN_ADJ",
            "seihan_param": "F_SEIHAN_PARAM",
            #### PRICING ####
            "price_structure": "F_PRICE_STRUCTURE",
            "mp_cost": "F_MP_COST",
            "exchange_rate": "F_EXCHANGE_RATE",
            #### DMD ####
            "demand_split": "F_DEMAND_SPLIT",
            "pricing": "F_PRICING",
            "event": "F_EVENT",
            "seasonality": "F_SEASONALITY",
            "factor": "F_FACTOR",
            "factor_acct": "F_FACTOR_ACCT",
            "seasonality_acct": "F_SEASONALITY_ACCT",
            "dmd_fct": "F_DMD_FCT",
            "dmd_param_fct_window": "F_DMD_PARAM_FCT_WINDOW",
            #### FCT ####
            "sc_psi_fct": "F_SC_PSI_FCT",
            "ch_psi_fct": "F_CH_PSI_FCT",
            "fct_flag": "F_FCT_FLAG",
        }
        # table_filters takes in list of where clauses

        for index, (table_name, tvf) in enumerate(tvf_list.items(), start=1):
            if callback is not None:
                callback(index/len(tvf_list))
            print(f'# Loading table: {table_name}...',)
            tm = time.time()
            with engine.connect() as connection:
                # tvf_name('FY24 5F,FY24 6F,LIVE', '88087910,02383910', 'IE,TV', 2024)
                data_dict[table_name] = pd.read_sql_query(
                                            "SELECT * FROM " + tvf + f"('{','.join(scn_filter)}', "
                                                                     f"'{','.join(sku_filter)}', "
                                                                     f"'{','.join(cat_filter)}', "
                                                                     f"'{','.join(cty_filter)}', "
                                                                     f"{stt_year})",
                                            connection
                                        )
                #print(data_dict[table_name])

            print(f' - {time.time()-tm:.4f} secs',)

    finally:
        engine.dispose()
        
    return data_dict



def get_unique_scenario():
    global connection_info
    connection_string = f"mssql+pyodbc://{connection_info['username']}:{connection_info['password']}@{connection_info['server']}/{connection_info['database']}?driver={connection_info['driver']}"
    engine = create_engine(connection_string)
    try:
        with engine.connect() as connection:
            query = 'SELECT DISTINCT SCENARIO FROM SCENARIO_MAP'
            df = pd.read_sql_query(query, connection)
            
    except Exception as e:
        print(e)
    
    finally:
        engine.dispose()
        
    return df['SCENARIO'].tolist()

def get_database_names():
    global connection_info
    connection_string = f"mssql+pyodbc://{connection_info['username']}:{connection_info['password']}@{connection_info['server']}/{connection_info['database']}?driver={connection_info['driver']}"
    engine = create_engine(connection_string)
    try:
        with engine.connect() as connection:
            query = "SELECT name FROM master.dbo.sysdatabases WHERE name LIKE '%ELCONDOR%'"
            df = pd.read_sql_query(query, connection)
            
    except Exception as e:
        print(e)
    
    finally:
        engine.dispose()
        
    return df['name'].tolist()

def get_db_data():
    global db_data
    return db_data


def preprocess_data():
    global db2_data


    ## SCENARIO_MAP
    db2_data['scenario_map'] = db_data['scenario_map'].copy()

    ## ACCOUNT_MAP
    db2_data['account_map'] = db_data['account_map'].copy()
    db2_data['product_map'] = db_data['product_map'][['CATEGORY_NAME','SUB_CATEGORY','KATABAN','SEGMENT_2','MODEL_NAME','SYSTEM_MODEL_CODE','PREDECESSOR', 'SUCCESSOR']].copy().rename(columns={'SYSTEM_MODEL_CODE':'MODEL_CODE'})
    db2_data['product_map'] = db2_data['product_map'].copy()


    ## CH_PSI_ACT
    act_gp_cols = ['CATEGORY_NAME', 'MODEL_CODE', 'MODEL_NAME', 'FSMC',
            'SALES_COMPANY', 'FCT_ACCOUNT', 'ACCOUNT_GROUP', 
            'SEIHAN_YEAR','SEIHAN_FISCAL_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK', 'WEEK_DATE']
   
    db2_data['ch_psi_act'] = db_data['ch_psi_act'].copy()\
                                                  .query("SEIHAN_YEAR>=2023 " )\
                                                  .merge(db_data['account_map'][['ACCOUNT_CODE', 'FCT_ACCOUNT','ACCOUNT_GROUP','FSMC','SALES_COMPANY']], on=['ACCOUNT_CODE'], how='left')\
                                                  .merge(db_data['time_w'][['SEIHAN_FISCAL_YEAR','WEEK_DATE']], on=['WEEK_DATE'], how='left')\
                                                  .groupby(act_gp_cols).agg({
                                                        'SELLIN_QTY': 'sum',
                                                        'SELLOUT_QTY': 'sum',
                                                        'CH_STOCK_OH': 'sum',
                                                        #'DISPLAY_INV_QTY': 'sum',
                                                        #'CH_STOCK_TTL': 'sum',
                                                        #'RANGING': 'sum',  
                                                        'SELLIN_QTY_ADJ': 'sum',
                                                        'SELLOUT_QTY_ADJ': 'sum',
                                                        'CH_STOCK_OH_ADJ': 'sum',
                                                        #'SELLABLE_INV_QTY_ADJ': 'sum',
                                                        #'DISPLAY_INV_QTY_ADJ': 'sum'
                                                    }).reset_index()
    db2_data['ch_psi_act'] ['WEEK_DATE'] = pd.to_datetime(db2_data['ch_psi_act']['WEEK_DATE'])
    db2_data['ch_psi_act'] ['SELLIN_QTY'] = db2_data['ch_psi_act'] ['SELLIN_QTY_ADJ']
    db2_data['ch_psi_act'] ['SELLOUT_QTY'] = db2_data['ch_psi_act'] ['SELLOUT_QTY_ADJ']
    db2_data['ch_psi_act'] ['CH_STOCK_OH'] = db2_data['ch_psi_act'] ['CH_STOCK_OH_ADJ']


    ## SC_PSI_LIVE
    db2_data['sc_psi_live'] = db_data['sc_psi_live'].query("SEIHAN_YEAR>=2023 ").copy()
    db2_data['sc_psi_live'] ['WEEK_DATE'] = pd.to_datetime(db2_data['sc_psi_live']['WEEK_DATE'])


    ## CH_PSI_FCT
    db2_data['ch_psi_fct'] = db_data['ch_psi_fct'].copy()\
                                                  .merge(db2_data['product_map'][['CATEGORY_NAME','SUB_CATEGORY','KATABAN','SEGMENT_2','MODEL_NAME','MODEL_CODE','PREDECESSOR', 'SUCCESSOR']], on=['MODEL_CODE'], how='left')\
                                                  .merge(db_data['fct_account_map'][['FSMC','SALES_COMPANY','ACCOUNT_GROUP','FCT_ACCOUNT']], on=['FCT_ACCOUNT'], how='left') \
                                                  .merge(db_data['time_w'][['SEIHAN_FISCAL_YEAR','WEEK_DATE']], on=['WEEK_DATE'], how='left')\
                                                  .merge(db_data['sellin_param'][['SCENARIO','FCT_TYPE','MODEL_CODE','FCT_ACCOUNT','WEEK_DATE','WOS_TARGET','MINSTOCK_TARGET','RANGING_TARGET']], on=['SCENARIO','FCT_TYPE','MODEL_CODE','FCT_ACCOUNT','WEEK_DATE'], how='left')\
                                                  .merge(db_data['sellin_adj'][['SCENARIO','FCT_TYPE','MODEL_CODE','FCT_ACCOUNT','WEEK_DATE','SELL_IN_ADJ']], on=['SCENARIO','FCT_TYPE','MODEL_CODE','FCT_ACCOUNT','WEEK_DATE'], how='left')
    db2_data['ch_psi_fct'] ['WEEK_DATE'] = pd.to_datetime(db2_data['ch_psi_fct'] ['WEEK_DATE'])
    db2_data['ch_psi_fct'] = db2_data['ch_psi_fct'].query(f"SCENARIO in {tuple(db_data['scenario_map']['SCENARIO'].tolist())} ")
    ## SC_PSI_FCT
    db2_data['sc_psi_fct'] = db_data['sc_psi_fct'].copy()\
                                                  .merge(db2_data['product_map'][['CATEGORY_NAME','SUB_CATEGORY','KATABAN','SEGMENT_2','MODEL_NAME','MODEL_CODE','PREDECESSOR', 'SUCCESSOR']], on=['MODEL_CODE'], how='left') \
                                                  .merge(db_data['time_w'][['SEIHAN_FISCAL_YEAR','WEEK_DATE']], on=['WEEK_DATE'], how='left')\
                                                  .merge(db_data['sellin_param'].query("FCT_ACCOUNT=='_TOTAL_COUNTRY'")[['SCENARIO','FCT_TYPE','MODEL_CODE','WEEK_DATE','WOS_TARGET']], on=['SCENARIO','FCT_TYPE','MODEL_CODE','WEEK_DATE'], how='left')\
                                                  .merge(db_data['sellin_adj'].query("FCT_ACCOUNT=='_TOTAL_COUNTRY'")[['SCENARIO','FCT_TYPE','MODEL_CODE','WEEK_DATE','SELL_IN_ADJ']], on=['SCENARIO','FCT_TYPE','MODEL_CODE','WEEK_DATE'], how='left')
    db2_data['sc_psi_fct']['WEEK_DATE'] = pd.to_datetime(db2_data['sc_psi_fct'] ['WEEK_DATE'])
    db2_data['sc_psi_fct']['CAP'] = db2_data['sc_psi_fct']['PO_ETA_QTY']
    db2_data['sc_psi_fct'] = db2_data['sc_psi_fct'].query(f"SCENARIO in {tuple(db_data['scenario_map']['SCENARIO'].tolist())} ")
    #sc_prm = db_data['sellin_param'].groupby(['SCENARIO', 'FCT_TYPE', 'MODEL_CODE', 'COUNTRY',  'WEEK_DATE']).agg({'WOS_TARGET':'max'}).reset_index()
    #db2_data['sc_psi_fct'] = db_data['sc_psi_fct'].merge(sc_prm[['SCENARIO', 'FCT_TYPE', 'MODEL_CODE', 'COUNTRY',  'WEEK_DATE','WOS_TARGET']], on=['SCENARIO', 'FCT_TYPE', 'MODEL_CODE', 'COUNTRY',  'WEEK_DATE'], how='left') 
    #db2_data['sc_psi_fct']['OH_WOS_TARGET'].fillna(4)
    #db2_data['sc_psi_fct']['WOS_TARGET']=4

    
    ## SEIHAN_PARAMS
    db2_data['seihan_param'] = db_data['seihan_param'].copy()

    ## SELLIN_PARAMS
    db2_data['sellin_param'] = db_data['sellin_param'].copy()
    db2_data['sellin_param']['WEEK_DATE'] = pd.to_datetime(db2_data['sellin_param'] ['WEEK_DATE'])

    ## SUPPLY_ADJ
    db2_data['supply_adj'] = db_data['supply_adj'].copy()
    db2_data['supply_adj'] ['WEEK_DATE'] = pd.to_datetime(db2_data['supply_adj'] ['WEEK_DATE'])

    ## SELLIN_ADJ
    db2_data['sellin_adj'] = db_data['sellin_adj'].copy()

    ## DEMAND
    db2_data['demand_split'] = db_data['demand_split'].copy()
    db2_data['demand_split']['WEEK_DATE'] = pd.to_datetime(db2_data['demand_split'] ['WEEK_DATE'])
    db_data['demand_split']['WEEK_DATE'] = pd.to_datetime(db_data['demand_split'] ['WEEK_DATE'])


    #========== SPECIAL TABLES ==========
    ## NS_MP 
    db2_data['mp_ns'] = db_data['price_structure'].copy().merge(db_data['mp_cost'][['SCENARIO','MODEL_CODE','SEIHAN_YEAR','SEIHAN_MONTH_NAME','MP_COST_LC']],
                                                        left_on =['SCENARIO','MODEL_CODE','SEIHAN_YEAR','SEIHAN_MONTH_NAME'],
                                                        right_on=['SCENARIO','MODEL_CODE','SEIHAN_YEAR','SEIHAN_MONTH_NAME'],
                                                        how='left')
