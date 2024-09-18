from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import base64
import datetime
import io
import pandas as pd
import numpy as np

from sqlalchemy import create_engine
import sqlalchemy
import urllib
import pyodbc
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

def save_to_sql_server(df, table_name, server, database, username, password, if_exists='append', chunksize=20000):
    """
    Save a DataFrame to a specified SQL Server table.
    """
    connection_string = f"DRIVER=ODBC Driver 17 for SQL Server;SERVER={server};DATABASE={database};UID={username};PWD={password}"
    params = urllib.parse.quote_plus(connection_string)
    
    # Create the SQLAlchemy engine using the connection string
    engine = sqlalchemy.create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}",
        fast_executemany=True  # Enable fast_executemany for performance
    )

    with engine.begin() as connection:
        if len(df) > chunksize:
            num_chunks = len(df) // chunksize + (1 if len(df) % chunksize else 0)
            for i in range(num_chunks):
                print(f"Processing chunk {i+1}/{num_chunks}")
                chunk = df[i*chunksize:(i+1)*chunksize]
                chunk = chunk.where(pd.notnull(chunk), None)
                
                try:
                    chunk.to_sql(table_name, connection, if_exists=if_exists, index=False, method=None)
                except Exception as e:
                    print(f'Error: {e}')
                    connection.rollback()
                    raise
        else:
            df = df.where(pd.notnull(df), None)
            try:
                df.to_sql(table_name, connection, if_exists=if_exists, index=False, method=None)
            except Exception as e:
                print(f'Error: {e}')
                connection.rollback()
                raise

def delete_records(server, database, username, password, table_name, where_conditions):
    """
    Delete records from a specified SQL Server table based on given conditions.

    :param server: The server name
    :param database: The database name.
    :param username: The username for authentication.
    :param password: The password for authentication.
    :param table_name: The name of the table to delete records from.
    :param where_conditions: The WHERE conditions for deleting records. Must be a valid SQL WHERE clause.
    """
    try:
        connection_string = f"DRIVER=ODBC Driver 17 for SQL Server;SERVER={server};DATABASE={database};UID={username};PWD={password}"
        
        # Establish connection
        with pyodbc.connect(connection_string) as conn:
            cursor = conn.cursor()
            
            # SQL DELETE statement
            sql = f"DELETE FROM {table_name} WHERE {where_conditions};"
            
            # Execute the SQL statement
            cursor.execute(sql)
            conn.commit()  # Commit the transaction
            
            print(f"Records deleted successfully from {table_name}.")
            
    except Exception as e:
        conn.rollback()  # Rollback the transaction if an error occurs
        print(f"An error occurred: {e}")

def preprocess_check(table_name, df):
    """
    Takes in a DataFrame and preprocesses it based on the table name. Ensures that columns needed are present and have the correct data types. Else raises an error.

    :param table_name: The name of the table to preprocess the DataFrame for.
    :param df: The DataFrame to preprocess.

    :return: The preprocessed DataFrame.
    """

    def to_str_or_none(x):
        return None if pd.isnull(x) else str(x)
    
    def model_code_check(x):
        """
        Model code 8 digit check
        """
        if isinstance(x, str):
            if x.isdigit() and len(x) != 8:
                return x.zfill(8)
        elif isinstance(x, (int, np.integer)):
            if len(str(x)) != 8:
                return str(int(x)).zfill(8)
        return x
    
    def convert_date(date):
        try:
            # Check if the date is an integer, Excel serial date
            serial_date = int(date)
            return pd.to_datetime(serial_date, origin='1899-12-30', unit='D')
        except ValueError:
            # If it's not an integer, try parsing it as a standard date string
            try:
                return pd.to_datetime(date, format='%m/%d/%Y')
            except ValueError:
                # If both conversions fail, raise an error
                raise ValueError(f"Unable to parse date: {date} \nPlease ensure the date is in the format 'MM/DD/YYYY'")
        
    ############################################################################# new db tables #############################################################################
    if table_name.lower() == 'time_w':
        
        columns = [
            'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK', 'WEEK_DATE', 'SEIHAN_FISCAL_YEAR', 'MONTH_DATE'
        ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        if df.duplicated().any():
            sample_duplicates = df[df.duplicated()].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
            
        try:
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_MONTH_NAME'] = df['SEIHAN_MONTH_NAME'].apply(to_str_or_none)
            df['SEIHAN_WEEK'] = df['SEIHAN_WEEK'].astype('Int64')
            df['WEEK_DATE'] = pd.to_datetime(df['WEEK_DATE'])
            df['WEEK_DATE'] = df['WEEK_DATE'].dt.strftime('%Y/%m/%d')
            df['SEIHAN_FISCAL_YEAR'] = df['SEIHAN_FISCAL_YEAR'].astype('Int64')
            df['MONTH_DATE'] = pd.to_datetime(df['MONTH_DATE'])
            df['MONTH_DATE'] = df['MONTH_DATE'].dt.strftime('%Y/%m/%d')
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")

    if table_name.lower() == 'time_m':
        
        columns = [
            'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME', 'CY_MONTH_NUM', 'SEIHAN_FISCAL_YEAR', 'FISCAL_MONTH_NAME', 'FY_MONTH_NUM', 'MONTH_DATE', 'REP_WEEK'
        ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        if df.duplicated().any():
            sample_duplicates = df[df.duplicated()].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_MONTH_NAME'] = df['SEIHAN_MONTH_NAME'].apply(to_str_or_none)
            df['CY_MONTH_NUM'] = df['CY_MONTH_NUM'].astype('Int64')
            df['SEIHAN_FISCAL_YEAR'] = df['SEIHAN_FISCAL_YEAR'].astype('Int64')
            df['FISCAL_MONTH_NAME'] = df['FISCAL_MONTH_NAME'].apply(to_str_or_none)
            df['FY_MONTH_NUM'] = df['FY_MONTH_NUM'].astype('Int64')
            df['MONTH_DATE'] = pd.to_datetime(df['MONTH_DATE'])
            df['MONTH_DATE'] = df['MONTH_DATE'].dt.strftime('%Y/%m/%d')
            df['REP_WEEK'] = df['REP_WEEK'].astype('Int64')
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")

    if table_name.lower() == 'scenario_map':
        
        columns = [
            'SCENARIO', 'ACT_WEEK_SO', 'ACT_WEEK_SI'
        ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        if df.duplicated().any():
            sample_duplicates = df[df.duplicated()].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        try:
            df['SCENARIO'] = df['SCENARIO'].apply(to_str_or_none)
            df['ACT_WEEK_SO'] = df['ACT_WEEK_SO'].astype('Int64')
            df['ACT_WEEK_SI'] = df['ACT_WEEK_SI'].astype('Int64')
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")

    if table_name.lower() == 'account_map':
        
        columns = [
            'FSMC', 'SALES_COMPANY', 'COUNTRY', 'ACCOUNT_CODE', 'ACCOUNT_NAME', 'FCT_ACCOUNT', 'ACCOUNT_GROUP', 'CH1', 'CH2', 'CH3'
        ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        if df.duplicated().any():
            sample_duplicates = df[df.duplicated()].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['FSMC'] = df['FSMC'].apply(to_str_or_none)
            df['SALES_COMPANY'] = df['SALES_COMPANY'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['ACCOUNT_CODE'] = df['ACCOUNT_CODE'].apply(to_str_or_none)
            df['ACCOUNT_NAME'] = df['ACCOUNT_NAME'].apply(to_str_or_none)
            df['FCT_ACCOUNT'] = df['FCT_ACCOUNT'].apply(to_str_or_none)
            df['ACCOUNT_GROUP'] = df['ACCOUNT_GROUP'].apply(to_str_or_none)
            df['CH1'] = df['CH1'].apply(to_str_or_none)
            df['CH2'] = df['CH2'].apply(to_str_or_none)
            df['CH3'] = df['CH3'].apply(to_str_or_none)
            
            return df

        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")

    if table_name.lower() == 'cat_account_map':
            
        columns = [
        'COUNTRY', 'FCT_ACCOUNT', 'CATEGORY_NAME', 'KORO_ACCOUNT'
        ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        if df.duplicated().any():
            sample_duplicates = df[df.duplicated()].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['FCT_ACCOUNT'] = df['FCT_ACCOUNT'].apply(to_str_or_none)
            df['CATEGORY_NAME'] = df['CATEGORY_NAME'].apply(to_str_or_none)
            df['KORO_ACCOUNT'] = df['KORO_ACCOUNT'].apply(to_str_or_none)
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")
            
    if table_name.lower() == 'product_map':

            columns = [
                'BIZ_GROUP', 'CATEGORY_CODE', 'CATEGORY_NAME', 'SUB_CATEGORY', 'KATABAN', 'SEGMENT_1', 'SEGMENT_2', 'SEGMENT_3', 'REPORT_GROUP', 'MODEL_CODE', 'MODEL_NAME', 'PREDECESSOR', 'PRE_NAME', 'SUCCESSOR', 'SUCC_NAME', 'REFERENCE', 'REF_NAME',
                'SYSTEM_CATEGORY_NAME', 'SYSTEM_MODEL_CODE', 'SYSTEM_MODEL_NAME'
            ]

            missing_columns = [col for col in columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
            df = df[columns]
            
            if df.duplicated().any():
                sample_duplicates = df[df.duplicated()].head(5)
                raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))

            try:
                df['MODEL_CODE'] = df['MODEL_CODE'].apply(to_str_or_none)
                df['MODEL_CODE'] = df['MODEL_CODE'].apply(lambda x: x.zfill(8) if x.isdigit() and len(x) != 8 else x)
                
                df['SYSTEM_MODEL_CODE'] = df['SYSTEM_MODEL_CODE'].apply(to_str_or_none)
                df['SYSTEM_MODEL_CODE'] = df['SYSTEM_MODEL_CODE'].apply(lambda x: x.zfill(8) if x.isdigit() and len(x) != 8 else x)
                
                df['BIZ_GROUP'] = df['BIZ_GROUP'].apply(to_str_or_none)
                df['CATEGORY_CODE'] = df['CATEGORY_CODE'].apply(to_str_or_none)
                df['CATEGORY_NAME'] = df['CATEGORY_NAME'].apply(to_str_or_none)
                df['SUB_CATEGORY'] = df['SUB_CATEGORY'].apply(to_str_or_none)
                df['KATABAN'] = df['KATABAN'].apply(to_str_or_none)
                df['SEGMENT_1'] = df['SEGMENT_1'].apply(to_str_or_none)
                df['SEGMENT_2'] = df['SEGMENT_2'].apply(to_str_or_none)
                df['SEGMENT_3'] = df['SEGMENT_3'].apply(to_str_or_none)
                df['REPORT_GROUP'] = df['REPORT_GROUP'].apply(to_str_or_none)
                df['MODEL_NAME'] = df['MODEL_NAME'].apply(to_str_or_none)
                df['PREDECESSOR'] = df['PREDECESSOR'].astype('Int64')
                df['PREDECESSOR'] = df['PREDECESSOR'].apply(model_code_check)
                df['PRE_NAME'] = df['PRE_NAME'].apply(to_str_or_none)
                df['SUCCESSOR'] = df['SUCCESSOR'].astype('Int64')
                df['SUCCESSOR'] = df['SUCCESSOR'].apply(model_code_check)
                df['SUCC_NAME'] = df['SUCC_NAME'].apply(to_str_or_none)
                df['REFERENCE'] = df['REFERENCE'].astype('Int64')
                df['REFERENCE'] = df['REFERENCE'].apply(model_code_check)
                df['REF_NAME'] = df['REF_NAME'].apply(to_str_or_none)
                df['SYSTEM_CATEGORY_NAME'] = df['SYSTEM_CATEGORY_NAME'].apply(to_str_or_none)
                df['SYSTEM_MODEL_NAME'] = df['SYSTEM_MODEL_NAME'].apply(to_str_or_none)
                
                return df

            except Exception as e:
                print('Error: ', e)
                raise ValueError(f"Error while preprocessing: {e}")

    if table_name.lower() == 'ch_psi_act':
        
        columns = [
            'CATEGORY_NAME', 'MODEL_CODE', 'MODEL_NAME', 'COUNTRY', 'ACCOUNT_CODE', 'ACCOUNT_NAME', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK', 'WEEK_DATE',
            'SELLIN_QTY', 'SELLOUT_QTY', 'CH_STOCK_OH', 'DISPLAY_INV_QTY', 'CH_STOCK_TTL', 'RANGING', 'SELLIN_QTY_ADJ', 'SELLOUT_QTY_ADJ', 'CH_STOCK_OH_ADJ', 'DISPLAY_INV_QTY_ADJ', 'REFRESH_DATE', 'SYSTEM_CATEGORY_NAME', 'SYSTEM_MODEL_CODE', 'SYSTEM_MODEL_NAME'
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        keys = ['ACCOUNT_CODE', 'SEIHAN_YEAR', 'SEIHAN_WEEK', 'SYSTEM_MODEL_CODE']
        if df.duplicated(subset=keys).any():
            sample_duplicates = df[df.duplicated(subset=keys)].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(model_code_check)
            df['CATEGORY_NAME'] = df['CATEGORY_NAME'].apply(to_str_or_none)
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(to_str_or_none)
            df['MODEL_NAME'] = df['MODEL_NAME'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['ACCOUNT_CODE'] = df['ACCOUNT_CODE'].apply(to_str_or_none)
            df['ACCOUNT_NAME'] = df['ACCOUNT_NAME'].apply(to_str_or_none)
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_MONTH_NAME'] = df['SEIHAN_MONTH_NAME'].apply(to_str_or_none)
            df['SEIHAN_WEEK'] = df['SEIHAN_WEEK'].astype('Int64')
            df['WEEK_DATE'] = pd.to_datetime(df['WEEK_DATE'], format='%m/%d/%Y') 
            df['WEEK_DATE'] = df['WEEK_DATE'].dt.strftime('%Y/%m/%d')
            df['SELLIN_QTY'] = df['SELLIN_QTY'].astype('Int64')
            df['SELLOUT_QTY'] = df['SELLOUT_QTY'].astype('Int64')
            df['CH_STOCK_OH'] = df['CH_STOCK_OH'].astype('Int64')
            df['DISPLAY_INV_QTY'] = df['DISPLAY_INV_QTY'].astype('Int64')
            df['CH_STOCK_TTL'] = df['CH_STOCK_TTL'].astype('Int64')
            df['RANGING'] = df['RANGING'].astype('Int64')
            df['SELLIN_QTY_ADJ'] = df['SELLIN_QTY_ADJ'].astype('Int64')
            df['SELLOUT_QTY_ADJ'] = df['SELLOUT_QTY_ADJ'].astype('Int64')
            df['CH_STOCK_OH_ADJ'] = df['CH_STOCK_OH_ADJ'].astype('Int64')
            df['DISPLAY_INV_QTY_ADJ'] = df['DISPLAY_INV_QTY_ADJ'].astype('Int64')
            df['REFRESH_DATE'] = pd.to_datetime(df['REFRESH_DATE'], format='%m/%d/%Y')
            df['REFRESH_DATE'] = df['REFRESH_DATE'].dt.strftime('%Y/%m/%d')
            df['SYSTEM_CATEGORY_NAME'] = df['SYSTEM_CATEGORY_NAME'].apply(to_str_or_none)
            df['SYSTEM_MODEL_CODE'] = df['SYSTEM_MODEL_CODE'].apply(model_code_check)
            df['SYSTEM_MODEL_CODE'] = df['SYSTEM_MODEL_CODE'].apply(to_str_or_none)
            df['SYSTEM_MODEL_NAME'] = df['SYSTEM_MODEL_NAME'].apply(to_str_or_none)
            
            return df
        
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")

    if table_name.lower() == 'financial_act':
        
        columns = [
            'COUNTRY', 'CATEGORY_NAME', 'MODEL_CODE', 'MODEL_NAME', 'SYSTEM_CATEGORY_NAME', 'SYSTEM_MODEL_CODE', 'SYSTEM_MODEL_NAME', 'ACCOUNT_CODE', 'ACCOUNT_NAME', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK', 'WEEK_DATE',
            'GROSS_SALES_LC', 'NET_SALES_LC', 'MP_AMOUNT_LC', 'GROSS_SALES_RC', 'NET_SALES_RC', 'MP_AMOUNT_RC', 'REFRESH_DATE'
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        keys = ['SYSTEM_MODEL_CODE', 'ACCOUNT_CODE', 'SEIHAN_YEAR', 'SEIHAN_WEEK']
        if df.duplicated(subset=keys).any():
            sample_duplicates = df[df.duplicated(subset=keys)].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['CATEGORY_NAME'] = df['CATEGORY_NAME'].apply(to_str_or_none)
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(model_code_check)
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(to_str_or_none)
            df['MODEL_NAME'] = df['MODEL_NAME'].apply(to_str_or_none)
            df['SYSTEM_CATEGORY_NAME'] = df['SYSTEM_CATEGORY_NAME'].apply(to_str_or_none)
            df['SYSTEM_MODEL_CODE'] = df['SYSTEM_MODEL_CODE'].apply(model_code_check)
            df['SYSTEM_MODEL_CODE'] = df['SYSTEM_MODEL_CODE'].apply(to_str_or_none)
            df['SYSTEM_MODEL_NAME'] = df['SYSTEM_MODEL_NAME'].apply(to_str_or_none)
            df['ACCOUNT_CODE'] = df['ACCOUNT_CODE'].apply(to_str_or_none)
            df['ACCOUNT_NAME'] = df['ACCOUNT_NAME'].apply(to_str_or_none)
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_MONTH_NAME'] = df['SEIHAN_MONTH_NAME'].apply(to_str_or_none)
            df['SEIHAN_WEEK'] = df['SEIHAN_WEEK'].astype('Int64')
            df['WEEK_DATE'] = pd.to_datetime(df['WEEK_DATE'], format='%m/%d/%Y')
            df['WEEK_DATE'] = df['WEEK_DATE'].dt.strftime('%Y/%m/%d')
            df['GROSS_SALES_LC'] = df['GROSS_SALES_LC'].astype(float)
            df['NET_SALES_LC'] = df['NET_SALES_LC'].astype(float)
            df['MP_AMOUNT_LC'] = df['MP_AMOUNT_LC'].astype(float)
            df['GROSS_SALES_RC'] = df['GROSS_SALES_RC'].astype(float)
            df['NET_SALES_RC'] = df['NET_SALES_RC'].astype(float)
            df['MP_AMOUNT_RC'] = df['MP_AMOUNT_RC'].astype(float)
            df['REFRESH_DATE'] = pd.to_datetime(df['REFRESH_DATE'], format='%m/%d/%Y')
            df['REFRESH_DATE'] = df['REFRESH_DATE'].dt.strftime('%Y/%m/%d')
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")

    if table_name.lower() == 'sc_psi_live':
        
        columns = [
            'CATEGORY_NAME', 'MODEL_CODE', 'MODEL_NAME', 'SALES_COMPANY', 'COUNTRY', 'WAREHOUSE', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK', 'WEEK_DATE', 'PO_ETD_QTY', 'PO_INTRANSIT_QTY', 'PO_ETA_QTY',
            'SC_STOCK_OH', 'DTC_ONLINE_INV_QTY', 'DTC_OFFLINE_INV_QTY', 'SC_STOCK_OH_NA', 'SC_STOCK_TR', 'REFRESH_DATE'
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        keys = ['MODEL_CODE', 'COUNTRY', 'SEIHAN_YEAR', 'SEIHAN_WEEK']
        if df.duplicated(subset=keys).any():
            sample_duplicates = df[df.duplicated(subset=keys)].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['CATEGORY_NAME'] = df['CATEGORY_NAME'].apply(to_str_or_none)
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(model_code_check)
            df['MODEL_NAME'] = df['MODEL_NAME'].apply(to_str_or_none)
            df['SALES_COMPANY'] = df['SALES_COMPANY'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['WAREHOUSE'] = df['WAREHOUSE'].apply(to_str_or_none)
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_MONTH_NAME'] = df['SEIHAN_MONTH_NAME'].apply(to_str_or_none)
            df['SEIHAN_WEEK'] = df['SEIHAN_WEEK'].astype('Int64')
            df['WEEK_DATE'] = pd.to_datetime(df['WEEK_DATE'], format='%m/%d/%Y')
            df['WEEK_DATE'] = df['WEEK_DATE'].dt.strftime('%Y/%m/%d')
            df['PO_ETD_QTY'] = df['PO_ETD_QTY'].astype('Float64')
            df['PO_INTRANSIT_QTY'] = df['PO_INTRANSIT_QTY'].astype('Float64')
            df['PO_ETA_QTY'] = df['PO_ETA_QTY'].astype('Float64')
            df['SC_STOCK_OH'] = df['SC_STOCK_OH'].astype('Float64')
            df['DTC_ONLINE_INV_QTY'] = df['DTC_ONLINE_INV_QTY'].astype('Float64')
            df['DTC_OFFLINE_INV_QTY'] = df['DTC_OFFLINE_INV_QTY'].astype('Float64')
            df['SC_STOCK_OH_NA'] = df['SC_STOCK_OH_NA'].astype('Float64')
            df['SC_STOCK_TR'] = df['SC_STOCK_TR'].astype('Float64')
            df['REFRESH_DATE'] = pd.to_datetime(df['REFRESH_DATE'], format='%m/%d/%Y')
            df['REFRESH_DATE'] = df['REFRESH_DATE'].dt.strftime('%Y/%m/%d')
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")

    if table_name.lower() == 'sellin_param':
        
        columns = [
            'SCENARIO', 'FCT_TYPE', 'MODEL_CODE', 'COUNTRY', 'FCT_ACCOUNT', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK', 'WEEK_DATE', 'WOS_TARGET', 'MINSTOCK_TARGET', 
            'RANGING_TARGET', 
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        keys = ['SCENARIO', 'FCT_TYPE', 'MODEL_CODE', 'COUNTRY', 'FCT_ACCOUNT', 'SEIHAN_YEAR', 'SEIHAN_WEEK']
        if df.duplicated(subset=keys).any():
            sample_duplicates = df[df.duplicated(subset=keys)].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(model_code_check)
            df['SCENARIO'] = df['SCENARIO'].apply(to_str_or_none)
            df['FCT_TYPE'] = df['FCT_TYPE'].apply(to_str_or_none)
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_MONTH_NAME'] = df['SEIHAN_MONTH_NAME'].apply(to_str_or_none)
            df['SEIHAN_WEEK'] = df['SEIHAN_WEEK'].astype(int)
            df['WEEK_DATE'] = pd.to_datetime(df['WEEK_DATE'], format='%m/%d/%Y')
            df['WEEK_DATE'] = df['WEEK_DATE'].dt.strftime('%Y/%m/%d')
            df['WOS_TARGET'] = df['WOS_TARGET'].astype('Float64')
            df['MINSTOCK_TARGET'] = df['MINSTOCK_TARGET'].astype('Float64')
            df['RANGING_TARGET'] = df['RANGING_TARGET'].astype('Float64')
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")

    if table_name.lower() == 'supply_adj':
        
        columns = [
            'SCENARIO', 'FCT_TYPE', 'MODEL_CODE', 'SALES_COMPANY', 'COUNTRY', 'WAREHOUSE', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK', 'WEEK_DATE', 'PO_ETD_QTY', 'PO_ETA_QTY'
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        keys = ['SCENARIO', 'FCT_TYPE', 'MODEL_CODE', 'COUNTRY', 'SEIHAN_YEAR', 'SEIHAN_WEEK']
        if df.duplicated(subset=keys).any():
            sample_duplicates = df[df.duplicated(subset=keys)].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(model_code_check)
            df['SCENARIO'] = df['SCENARIO'].apply(to_str_or_none)
            df['FCT_TYPE'] = df['FCT_TYPE'].apply(to_str_or_none)
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(to_str_or_none)
            df['SALES_COMPANY'] = df['SALES_COMPANY'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['WAREHOUSE'] = df['WAREHOUSE'].apply(to_str_or_none)
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_MONTH_NAME'] = df['SEIHAN_MONTH_NAME'].apply(to_str_or_none)
            df['SEIHAN_WEEK'] = df['SEIHAN_WEEK'].astype('Int64')
            df['WEEK_DATE'] = df['WEEK_DATE'].apply(convert_date)
            df['WEEK_DATE'] = df['WEEK_DATE'].dt.strftime('%Y/%m/%d')
            df['PO_ETD_QTY'] = df['PO_ETD_QTY'].astype('Int64')
            df['PO_ETA_QTY'] = df['PO_ETA_QTY'].astype('Int64')
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")

    if table_name.lower() == 'sellin_adj':
        
        columns = [
            'SCENARIO', 'FCT_TYPE', 'COUNTRY', 'MODEL_CODE', 'FCT_ACCOUNT', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK', 'WEEK_DATE', 'SELL_IN_ADJ', 'CONFIRMED'
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        keys = ['SCENARIO', 'FCT_TYPE', 'MODEL_CODE', 'FCT_ACCOUNT', 'SEIHAN_YEAR', 'SEIHAN_WEEK']
        if df.duplicated(subset=keys).any():
            sample_duplicates = df[df.duplicated(subset=keys)].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(model_code_check)
            df['SCENARIO'] = df['SCENARIO'].apply(to_str_or_none)
            df['FCT_TYPE'] = df['FCT_TYPE'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(to_str_or_none)
            df['FCT_ACCOUNT'] = df['FCT_ACCOUNT'].apply(to_str_or_none)
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_MONTH_NAME'] = df['SEIHAN_MONTH_NAME'].apply(to_str_or_none)
            df['SEIHAN_WEEK'] = df['SEIHAN_WEEK'].astype('Int64')
            df['WEEK_DATE'] = pd.to_datetime(df['WEEK_DATE'], format='%d/%m/%Y')
            df['WEEK_DATE'] = df['WEEK_DATE'].dt.strftime('%Y/%m/%d')
            df['SELL_IN_ADJ'] = df['SELL_IN_ADJ'].astype('Int64')
            df['CONFIRMED'] = df['CONFIRMED'].astype('Int64')
            
            return df
        
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")
    
    if table_name.lower() == 'price_structure':
        
        columns = [
            'SCENARIO', 'MODEL_CODE', 'COUNTRY', 'FCT_ACCOUNT', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME', 'MONTH_DATE', 'GSP_LC', 'NSP_LC', 'GROSS_NET_PCT'
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        keys = ['SCENARIO', 'MODEL_CODE', 'FCT_ACCOUNT', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME']
        if df.duplicated(subset=keys).any():
            sample_duplicates = df[df.duplicated(subset=keys)].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(model_code_check)
            df['SCENARIO'] = df['SCENARIO'].apply(to_str_or_none)
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['FCT_ACCOUNT'] = df['FCT_ACCOUNT'].apply(to_str_or_none)
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_MONTH_NAME'] = df['SEIHAN_MONTH_NAME'].apply(to_str_or_none)
            df['MONTH_DATE'] = pd.to_datetime(df['MONTH_DATE'])
            df['MONTH_DATE'] = df['MONTH_DATE'].dt.strftime('%Y/%m/%d')
            df['GSP_LC'] = df['GSP_LC'].astype(float)
            df['NSP_LC'] = df['NSP_LC'].astype(float)
            df['GROSS_NET_PCT'] = df['GROSS_NET_PCT'].astype(float)
            
            return df
        
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")
            
    if table_name.lower() == 'mp_cost':

        columns = [
            'SCENARIO', 'MODEL_CODE', 'COUNTRY', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME', 'MONTH_DATE', 'MP_COST_LC', 'MP_COST_RC', 'COGS_LC', 'COGS_RC', 'REFRESH_DATE'
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        keys = ['SCENARIO', 'MODEL_CODE', 'COUNTRY', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME']
        if df.duplicated(subset=keys).any():
            sample_duplicates = df[df.duplicated(subset=keys)].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(model_code_check)
            df['SCENARIO'] = df['SCENARIO'].apply(to_str_or_none)
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_MONTH_NAME'] = df['SEIHAN_MONTH_NAME'].apply(to_str_or_none)
            df['MONTH_DATE'] = pd.to_datetime(df['MONTH_DATE'], format='%m/%d/%Y')
            df['MONTH_DATE'] = df['MONTH_DATE'].dt.strftime('%Y/%m/%d')
            df['MP_COST_LC'] = df['MP_COST_LC'].astype('Float64')
            df['MP_COST_RC'] = df['MP_COST_RC'].astype('Float64')
            df['COGS_LC'] = df['COGS_LC'].astype('Float64')
            df['COGS_RC'] = df['COGS_RC'].astype('Float64')
            df['REFRESH_DATE'] = pd.to_datetime(df['REFRESH_DATE'], format='%m/%d/%Y')
            df['REFRESH_DATE'] = df['REFRESH_DATE'].dt.strftime('%Y/%m/%d')
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")
            
    if table_name.lower() == 'exchange_rate':
        
        columns = [
            'SCENARIO', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME', 'MONTH_DATE', 'CURRENCY', 'JPY_RC', 'JPY_LC', 'LC_RC', 'RC_LC', 'REFRESH_DATE'
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        keys = ['SCENARIO', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME', 'CURRENCY']
        if df.duplicated(subset=keys).any():
            sample_duplicates = df[df.duplicated(subset=keys)].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['SCENARIO'] = df['SCENARIO'].apply(to_str_or_none)
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_MONTH_NAME'] = df['SEIHAN_MONTH_NAME'].apply(to_str_or_none)
            df['MONTH_DATE'] = pd.to_datetime(df['MONTH_DATE'], format='%m/%d/%Y')
            df['MONTH_DATE'] = df['MONTH_DATE'].dt.strftime('%Y/%m/%d')
            df['CURRENCY'] = df['CURRENCY'].apply(to_str_or_none)
            df['JPY_RC'] = df['JPY_RC'].astype(float)
            df['JPY_LC'] = df['JPY_LC'].astype(float)
            df['LC_RC'] = df['LC_RC'].astype(float)
            df['RC_LC'] = df['RC_LC'].astype(float)
            df['REFRESH_DATE'] = pd.to_datetime(df['REFRESH_DATE'], format='%m/%d/%Y')
            df['REFRESH_DATE'] = df['REFRESH_DATE'].dt.strftime('%Y/%m/%d')
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")
    
    if table_name.lower() == 'demand_split':
        
        columns = [
            'SCENARIO', 'ACT_WEEK_SO', 'DMD_TYPE', 'SALES_COMPANY', 'COUNTRY', 'FCT_ACCOUNT', 'ACCOUNT_GROUP', 'CATEGORY_NAME', 'MODEL_CODE', 'MODEL_NAME', 'SEIHAN_YEAR', 'SEIHAN_FISCAL_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK', 'ACT_FCT',
            'WEEK_DATE', 'SELLOUT_QTY1', 'SELLABLE_INV_QTY1', 'SELLOUT_QTY2', 'SELLABLE_INV_QTY2', 'SELLOUT_QTY', 'SELLABLE_INV_QTY', 'PAST4WK', 'WOS', 'SEASONALITY', 'PROMO_IMPACT_PCT', 'EVENT_IMPACT_PCT', 'TREND_ADJ_PCT', 'RANGING_IMPACT_PCT', 'RR_UPPER', 'RR_LOWER', 'SHORTAGE_IMPACT_PCT', 'NORM_RR', 'BASE_RR', 'ADJ_RR', 'ADJ2_RR',
            'SEASONALITY_IMPACT', 'PROMOTION_IMPACT', 'EVENT_IMPACT', 'RANGING_IMPACT', 'TREND_ADJ_IMPACT', 'SHORTAGE_IMPACT', 'NATURAL_DEMAND_ACCOUNT', 'ACCOUNT_WEIGHT', 'NATURAL_DEMAND_SPLIT'
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        keys = ['SCENARIO', 'ACT_WEEK_SO', 'DMD_TYPE', 'FCT_ACCOUNT', 'MODEL_CODE', 'SEIHAN_YEAR', 'SEIHAN_WEEK']
        if df.duplicated(subset=keys).any():
            sample_duplicates = df[df.duplicated(subset=keys)].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
            
        try:
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(model_code_check)
            df['SCENARIO'] = df['SCENARIO'].apply(to_str_or_none)
            df['ACT_WEEK_SO'] = df['ACT_WEEK_SO'].astype(int)
            df['DMD_TYPE'] = df['DMD_TYPE'].apply(to_str_or_none)
            df['SALES_COMPANY'] = df['SALES_COMPANY'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['FCT_ACCOUNT'] = df['FCT_ACCOUNT'].apply(to_str_or_none)
            df['ACCOUNT_GROUP'] = df['ACCOUNT_GROUP'].apply(to_str_or_none)
            df['CATEGORY_NAME'] = df['CATEGORY_NAME'].apply(to_str_or_none)
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(to_str_or_none)
            df['MODEL_NAME'] = df['MODEL_NAME'].apply(to_str_or_none)
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_FISCAL_YEAR'] = df['SEIHAN_FISCAL_YEAR'].astype('Int64')
            df['SEIHAN_MONTH_NAME'] = df['SEIHAN_MONTH_NAME'].apply(to_str_or_none)
            df['SEIHAN_WEEK'] = df['SEIHAN_WEEK'].astype('Int64')
            df['ACT_FCT'] = df['ACT_FCT'].astype('Int64')
            df['WEEK_DATE'] = pd.to_datetime(df['WEEK_DATE'], format='%d/%m/%Y')
            df['WEEK_DATE'] = df['WEEK_DATE'].dt.strftime('%Y/%m/%d')
            df['SELLOUT_QTY1'] = df['SELLOUT_QTY1'].astype('Float64')
            df['SELLABLE_INV_QTY1'] = df['SELLABLE_INV_QTY1'].astype('Float64')
            df['SELLOUT_QTY2'] = df['SELLOUT_QTY2'].astype('Float64')
            df['SELLABLE_INV_QTY2'] = df['SELLABLE_INV_QTY2'].astype('Float64')
            df['SELLOUT_QTY'] = df['SELLOUT_QTY'].astype('Float64')
            df['SELLABLE_INV_QTY'] = df['SELLABLE_INV_QTY'].astype('Float64')
            df['PAST4WK'] = df['PAST4WK'].astype('Float64')
            df['WOS'] = df['WOS'].astype(str)
            df['SEASONALITY'] = df['SEASONALITY'].astype('Float64')
            df['PROMO_IMPACT_PCT'] = df['PROMO_IMPACT_PCT'].astype('Float64')
            df['EVENT_IMPACT_PCT'] = df['EVENT_IMPACT_PCT'].astype('Float64')
            df['TREND_ADJ_PCT'] = df['TREND_ADJ_PCT'].astype('Float64')
            df['RANGING_IMPACT_PCT'] = df['RANGING_IMPACT_PCT'].astype('Float64')
            df['RR_UPPER'] = df['RR_UPPER'].astype('Float64')
            df['RR_LOWER'] = df['RR_LOWER'].astype('Float64')
            df['SHORTAGE_IMPACT_PCT'] = df['SHORTAGE_IMPACT_PCT'].astype('Float64')
            df['NORM_RR'] = df['NORM_RR'].astype('Float64')
            df['BASE_RR'] = df['BASE_RR'].astype('Float64')
            df['ADJ_RR'] = df['ADJ_RR'].astype('Float64')
            df['ADJ2_RR'] = df['ADJ2_RR'].astype('Float64')
            df['SEASONALITY_IMPACT'] = df['SEASONALITY_IMPACT'].astype('Float64')
            df['PROMOTION_IMPACT'] = df['PROMOTION_IMPACT'].astype('Float64')
            df['EVENT_IMPACT'] = df['EVENT_IMPACT'].astype('Float64')
            df['RANGING_IMPACT'] = df['RANGING_IMPACT'].astype('Float64')
            df['TREND_ADJ_IMPACT'] = df['TREND_ADJ_IMPACT'].astype('Float64')
            df['SHORTAGE_IMPACT'] = df['SHORTAGE_IMPACT'].astype('Float64')
            df['NATURAL_DEMAND_ACCOUNT'] = df['NATURAL_DEMAND_ACCOUNT'].astype('Float64')
            df['ACCOUNT_WEIGHT'] = df['ACCOUNT_WEIGHT'].astype('Float64')
            df['NATURAL_DEMAND_SPLIT'] = df['NATURAL_DEMAND_SPLIT'].astype('Float64')
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")
            
    if table_name.lower() == 'sc_psi_fct':
        
        columns = [
            'SCENARIO', 'FCT_TYPE', 'MODEL_CODE', 'COUNTRY', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK', 'WEEK_DATE',
            'PO_ETA_OSI', 'PO_ETA_CAP', 'PO_ETA_QTY', 'SC_STOCK_OH', 'SELLIN_QTY', 'SELLOUT_QTY', 'CH_STOCK_OH'
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        keys = ['SCENARIO', 'FCT_TYPE', 'MODEL_CODE', 'COUNTRY', 'SEIHAN_YEAR', 'SEIHAN_WEEK']
        if df.duplicated(subset=keys).any():
            sample_duplicates = df[df.duplicated(subset=keys)].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(model_code_check)
            df['SCENARIO'] = df['SCENARIO'].apply(to_str_or_none)
            df['FCT_TYPE'] = df['FCT_TYPE'].apply(to_str_or_none)
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_MONTH_NAME'] = df['SEIHAN_MONTH_NAME'].apply(to_str_or_none)
            df['SEIHAN_WEEK'] = df['SEIHAN_WEEK'].astype('Int64')
            df['WEEK_DATE'] = pd.to_datetime(df['WEEK_DATE'], format='%m/%d/%Y')
            df['WEEK_DATE'] = df['WEEK_DATE'].dt.strftime('%Y/%m/%d')
            df['PO_ETA_OSI'] = df['PO_ETA_OSI'].astype(float)
            df['PO_ETA_CAP'] = df['PO_ETA_CAP'].astype(float)
            df['PO_ETA_QTY'] = df['PO_ETA_QTY'].astype(float)
            df['SC_STOCK_OH'] = df['SC_STOCK_OH'].astype(float)
            df['SELLIN_QTY'] = df['SELLIN_QTY'].astype(float)
            df['SELLOUT_QTY'] = df['SELLOUT_QTY'].astype(float)
            df['CH_STOCK_OH'] = df['CH_STOCK_OH'].astype(float)
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")
    
    if table_name.lower() == 'ch_psi_fct':
        
        columns = [
            'SCENARIO', 'FCT_TYPE', 'MODEL_CODE', 'COUNTRY', 'FCT_ACCOUNT', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK', 'WEEK_DATE', 
            'SELLIN_QTY', 'SELLOUT_QTY', 'CH_STOCK_OH', 'DISPLAY_INV_QTY', 'NATURAL_DEMAND', 'SELLIN_QTY_ORI', 'CH_STOCK_OH_ORI', 'GROSS_SALES_LC', 'NET_SALES_LC', 'MP_AMOUNT_LC'
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        keys = ['SCENARIO', 'FCT_TYPE', 'MODEL_CODE', 'FCT_ACCOUNT', 'SEIHAN_YEAR', 'SEIHAN_WEEK']
        if df.duplicated(subset=keys).any():
            sample_duplicates = df[df.duplicated(subset=keys)].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(model_code_check)
            df['SCENARIO'] = df['SCENARIO'].apply(to_str_or_none)
            df['FCT_TYPE'] = df['FCT_TYPE'].apply(to_str_or_none)
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['FCT_ACCOUNT'] = df['FCT_ACCOUNT'].apply(to_str_or_none)
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_MONTH_NAME'] = df['SEIHAN_MONTH_NAME'].apply(to_str_or_none)
            df['SEIHAN_WEEK'] = df['SEIHAN_WEEK'].astype('Int64')
            df['WEEK_DATE'] = pd.to_datetime(df['WEEK_DATE'], format='%m/%d/%Y')
            df['WEEK_DATE'] = df['WEEK_DATE'].dt.strftime('%Y/%m/%d')
            df['SELLIN_QTY'] = df['SELLIN_QTY'].astype(float)
            df['SELLOUT_QTY'] = df['SELLOUT_QTY'].astype(float)
            df['CH_STOCK_OH'] = df['CH_STOCK_OH'].astype(float)
            df['DISPLAY_INV_QTY'] = df['DISPLAY_INV_QTY'].astype(float)
            df['NATURAL_DEMAND'] = df['NATURAL_DEMAND'].astype(float)
            df['SELLIN_QTY_ORI'] = df['SELLIN_QTY_ORI'].astype(float)
            df['CH_STOCK_OH_ORI'] = df['CH_STOCK_OH_ORI'].astype(float)
            df['GROSS_SALES_LC'] = df['GROSS_SALES_LC'].astype(float)
            df['NET_SALES_LC'] = df['NET_SALES_LC'].astype(float)
            df['MP_AMOUNT_LC'] = df['MP_AMOUNT_LC'].astype(float)
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")

    if table_name.lower() == 'pricing':
        
        columns = [
            'SCENARIO', 'ACT_WEEK_SO', 'DMD_TYPE', 'CATEGORY_NAME', 'MODEL_CODE', 'MODEL_NAME', 'SEIHAN_YEAR', 'SEIHAN_FISCAL_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK', 'SALES_COMPANY', 'COUNTRY', 'CURRENCY', 'SRP', 'DISCOUNT', 'DISCOUNT_PCT',
            'PROMO_PRICE', 'DEALER_SUPPORT', 'STREET_PRICE', 'GROSS_NET_PCT', 'NS_AFT_PROMO', 'MP_COST_LC', 'MP_AMT_LC', 'MP_PCT', 'MP_PCT_BASE', 'BASE_DISCOUNT', 'DISCOUNT_PCT_OVERBASE_PROMO', 'SELLOUT_INCREMENT_PCT', 'PROMO_VS_PW',
            'DEMAND_IMPACT_PCT', 'WEEK_DATE'
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        keys = ['SCENARIO', 'ACT_WEEK_SO', 'DMD_TYPE', 'MODEL_CODE', 'SEIHAN_YEAR', 'SEIHAN_WEEK', 'COUNTRY']
        if df.duplicated(subset=keys).any():
            sample_duplicates = df[df.duplicated(subset=keys)].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(model_code_check)
            df['SCENARIO'] = df['SCENARIO'].apply(to_str_or_none)
            df['ACT_WEEK_SO'] = df['ACT_WEEK_SO'].apply(to_str_or_none)
            df['DMD_TYPE'] = df['DMD_TYPE'].apply(to_str_or_none)
            df['CATEGORY_NAME'] = df['CATEGORY_NAME'].apply(to_str_or_none)
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(to_str_or_none)
            df['MODEL_NAME'] = df['MODEL_NAME'].apply(to_str_or_none)
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_FISCAL_YEAR'] = df['SEIHAN_FISCAL_YEAR'].astype('Int64')
            df['SEIHAN_MONTH_NAME'] = df['SEIHAN_MONTH_NAME'].apply(to_str_or_none)
            df['SEIHAN_WEEK'] = df['SEIHAN_WEEK'].astype('Int64')
            df['SALES_COMPANY'] = df['SALES_COMPANY'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['CURRENCY'] = df['CURRENCY'].apply(to_str_or_none)
            df['SRP'] = df['SRP'].astype(float)
            df['DISCOUNT'] = df['DISCOUNT'].astype(float)
            df['DISCOUNT_PCT'] = df['DISCOUNT_PCT'].astype(float)
            df['PROMO_PRICE'] = df['PROMO_PRICE'].astype(float)
            df['DEALER_SUPPORT'] = df['DEALER_SUPPORT'].astype(float)
            df['STREET_PRICE'] = df['STREET_PRICE'].astype(float)
            df['GROSS_NET_PCT'] = df['GROSS_NET_PCT'].astype(float)
            df['NS_AFT_PROMO'] = df['NS_AFT_PROMO'].astype(float)
            df['MP_COST_LC'] = df['MP_COST_LC'].astype(float)
            df['MP_AMT_LC'] = df['MP_AMT_LC'].astype(float)
            df['MP_PCT'] = df['MP_PCT'].astype(float)
            df['MP_PCT_BASE'] = df['MP_PCT_BASE'].astype(float)
            df['BASE_DISCOUNT'] = df['BASE_DISCOUNT'].astype(float)
            df['DISCOUNT_PCT_OVERBASE_PROMO'] = df['DISCOUNT_PCT_OVERBASE_PROMO'].astype(float)
            df['SELLOUT_INCREMENT_PCT'] = df['SELLOUT_INCREMENT_PCT'].astype(float)
            df['PROMO_VS_PW'] = df['PROMO_VS_PW'].astype(float)
            df['DEMAND_IMPACT_PCT'] = df['DEMAND_IMPACT_PCT'].astype(float)
            df['WEEK_DATE'] = pd.to_datetime(df['WEEK_DATE'], format='%m/%d/%Y')
            df['WEEK_DATE'] = df['WEEK_DATE'].dt.strftime('%Y/%m/%d')
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")
            
    if table_name.lower() == 'event':
        
        columns = [
            'SCENARIO', 'ACT_WEEK_SO', 'DMD_TYPE', 'SALES_COMPANY', 'COUNTRY', 'SEIHAN_YEAR', 'SEIHAN_FISCAL_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK', 'CATEGORY_NAME', 'MODEL_CODE', 'MODEL_NAME', 'ACCOUNT_GROUP', 'FCT_ACCOUNT', 'PROMO_EVENT',
            'DEMAND_IMPACT_PCT', 'WEEK_DATE'
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        keys = ['SCENARIO', 'ACT_WEEK_SO', 'DMD_TYPE', 'COUNTRY', 'SEIHAN_YEAR', 'SEIHAN_WEEK', 'CATEGORY_NAME']
        if df.duplicated(subset=keys).any():
            sample_duplicates = df[df.duplicated(subset=keys)].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(model_code_check)
            df['SCENARIO'] = df['SCENARIO'].apply(to_str_or_none)
            df['ACT_WEEK_SO'] = df['ACT_WEEK_SO'].astype(int)
            df['DMD_TYPE'] = df['DMD_TYPE'].apply(to_str_or_none)
            df['SALES_COMPANY'] = df['SALES_COMPANY'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_FISCAL_YEAR'] = df['SEIHAN_FISCAL_YEAR'].astype('Int64')
            df['SEIHAN_MONTH_NAME'] = df['SEIHAN_MONTH_NAME'].apply(to_str_or_none)
            df['SEIHAN_WEEK'] = df['SEIHAN_WEEK'].astype('Int64')
            df['CATEGORY_NAME'] = df['CATEGORY_NAME'].apply(to_str_or_none)
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(to_str_or_none)
            df['MODEL_NAME'] = df['MODEL_NAME'].apply(to_str_or_none)
            df['ACCOUNT_GROUP'] = df['ACCOUNT_GROUP'].apply(to_str_or_none)
            df['FCT_ACCOUNT'] = df['FCT_ACCOUNT'].apply(to_str_or_none)
            df['PROMO_EVENT'] = df['PROMO_EVENT'].apply(to_str_or_none)
            df['DEMAND_IMPACT_PCT'] = df['DEMAND_IMPACT_PCT'].astype(float)
            df['WEEK_DATE'] = pd.to_datetime(df['WEEK_DATE'], format='%m/%d/%Y')
            df['WEEK_DATE'] = df['WEEK_DATE'].dt.strftime('%Y/%m/%d')
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")

    if table_name.lower() == 'seasonality':
        
        columns = [
            'SCENARIO', 'ACT_WEEK_SO', 'DMD_TYPE', 'CATEGORY_NAME', 'SALES_COMPANY', 'COUNTRY', 'SUB_CATEGORY', 'MODEL_CODE', 'MODEL_NAME', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME', 'MONTH_DATE', 'SEASONALITY'
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        keys = ['SCENARIO', 'ACT_WEEK_SO', 'DMD_TYPE', 'CATEGORY_NAME', 'COUNTRY', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME']
        if df.duplicated(subset=keys).any():
            sample_duplicates = df[df.duplicated(subset=keys)].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(model_code_check)
            df['SCENARIO'] = df['SCENARIO'].apply(to_str_or_none)
            df['ACT_WEEK_SO'] = df['ACT_WEEK_SO'].apply(to_str_or_none)
            df['DMD_TYPE'] = df['DMD_TYPE'].apply(to_str_or_none)
            df['CATEGORY_NAME'] = df['CATEGORY_NAME'].apply(to_str_or_none)
            df['SALES_COMPANY'] = df['SALES_COMPANY'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['SUB_CATEGORY'] = df['SUB_CATEGORY'].apply(to_str_or_none)
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(to_str_or_none)
            df['MODEL_NAME'] = df['MODEL_NAME'].apply(to_str_or_none)
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_MONTH_NAME'] = df['SEIHAN_MONTH_NAME'].apply(to_str_or_none)
            df['MONTH_DATE'] = pd.to_datetime(df['MONTH_DATE'], format='%d/%m/%Y')
            df['MONTH_DATE'] = df['MONTH_DATE'].dt.strftime('%Y/%m/%d')
            df['SEASONALITY'] = df['SEASONALITY'].astype(float)
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")

    if table_name.lower() == 'factor':
        
        columns = [
            'SCENARIO', 'ACT_WEEK_SO', 'DMD_TYPE', 'CATEGORY_NAME', 'SALES_COMPANY', 'COUNTRY', 'MODEL_CODE', 'MODEL_NAME', 'SEIHAN_YEAR', 'SEIHAN_WEEK', 'PROMO_IMPACT_PCT', 'EVENT_IMPACT_PCT', 'RR_UPPER', 'RR_LOWER', 'TREND_ADJ_PCT', 'RANGING_IMPACT_PCT', 'WEEK_DATE'
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        keys = ['SCENARIO', 'ACT_WEEK_SO', 'DMD_TYPE', 'COUNTRY', 'MODEL_CODE', 'SEIHAN_YEAR', 'SEIHAN_WEEK']
        if df.duplicated(subset=keys).any():
            sample_duplicates = df[df.duplicated(subset=keys)].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(model_code_check)
            df['SCENARIO'] = df['SCENARIO'].apply(to_str_or_none)
            df['ACT_WEEK_SO'] = df['ACT_WEEK_SO'].apply(to_str_or_none)
            df['DMD_TYPE'] = df['DMD_TYPE'].apply(to_str_or_none)
            df['CATEGORY_NAME'] = df['CATEGORY_NAME'].apply(to_str_or_none)
            df['SALES_COMPANY'] = df['SALES_COMPANY'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(to_str_or_none)
            df['MODEL_NAME'] = df['MODEL_NAME'].apply(to_str_or_none)
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_WEEK'] = df['SEIHAN_WEEK'].astype('Int64')
            df['PROMO_IMPACT_PCT'] = df['PROMO_IMPACT_PCT'].astype(float)
            df['EVENT_IMPACT_PCT'] = df['EVENT_IMPACT_PCT'].astype(float)
            df['RR_UPPER'] = df['RR_UPPER'].astype(float)
            df['RR_LOWER'] = df['RR_LOWER'].astype(float)
            df['TREND_ADJ_PCT'] = df['TREND_ADJ_PCT'].astype(float)
            df['RANGING_IMPACT_PCT'] = df['RANGING_IMPACT_PCT'].astype(float)
            df['WEEK_DATE'] = pd.to_datetime(df['WEEK_DATE'], format='%m/%d/%Y')
            df['WEEK_DATE'] = df['WEEK_DATE'].dt.strftime('%Y/%m/%d')
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")

    if table_name.lower() == 'factor_acct':
        
        columns = [
            'SCENARIO', 'ACT_WEEK_SO', 'DMD_TYPE', 'CATEGORY_NAME', 'SALES_COMPANY', 'COUNTRY', 'MODEL_CODE', 'MODEL_NAME', 'SEIHAN_YEAR', 'SEIHAN_WEEK', 'FCT_ACCOUNT', 'PROMO_IMPACT_PCT', 'EVENT_IMPACT_PCT', 'RR_UPPER', 'RR_LOWER', 'TREND_ADJ_PCT', 'RANGING_IMPACT_PCT',
            'WEEK_DATE'
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        keys = ['SCENARIO', 'ACT_WEEK_SO', 'DMD_TYPE', 'COUNTRY', 'MODEL_CODE', 'SEIHAN_YEAR', 'SEIHAN_WEEK', 'FCT_ACCOUNT']
        if df.duplicated(subset=keys).any():
            sample_duplicates = df[df.duplicated(subset=keys)].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(model_code_check)
            df['SCENARIO'] = df['SCENARIO'].apply(to_str_or_none)
            df['ACT_WEEK_SO'] = df['ACT_WEEK_SO'].astype(int)
            df['DMD_TYPE'] = df['DMD_TYPE'].apply(to_str_or_none)
            df['CATEGORY_NAME'] = df['CATEGORY_NAME'].apply(to_str_or_none)
            df['SALES_COMPANY'] = df['SALES_COMPANY'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(to_str_or_none)
            df['MODEL_NAME'] = df['MODEL_NAME'].apply(to_str_or_none)
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_WEEK'] = df['SEIHAN_WEEK'].astype('Int64')
            df['FCT_ACCOUNT'] = df['FCT_ACCOUNT'].apply(to_str_or_none)
            df['PROMO_IMPACT_PCT'] = df['PROMO_IMPACT_PCT'].astype(float)
            df['EVENT_IMPACT_PCT'] = df['EVENT_IMPACT_PCT'].astype(float)
            df['RR_UPPER'] = df['RR_UPPER'].astype(float)
            df['RR_LOWER'] = df['RR_LOWER'].astype(float)
            df['TREND_ADJ_PCT'] = df['TREND_ADJ_PCT'].astype(float)
            df['RANGING_IMPACT_PCT'] = df['RANGING_IMPACT_PCT'].astype(float)
            df['WEEK_DATE'] = pd.to_datetime(df['WEEK_DATE'], format='%m/%d/%Y')
            df['WEEK_DATE'] = df['WEEK_DATE'].dt.strftime('%Y/%m/%d')
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")

    if table_name.lower() == 'seasonality_acct':
        
        columns = [
            'SCENARIO', 'ACT_WEEK_SO', 'DMD_TYPE', 'SALES_COMPANY', 'COUNTRY', 'CATEGORY_NAME', 'FCT_ACCOUNT', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME', 'MONTH_DATE', 'SEASONALITY' 
                    ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        keys = ['SCENARIO', 'ACT_WEEK_SO', 'DMD_TYPE', 'CATEGORY_NAME', 'FCT_ACCOUNT', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME']
        if df.duplicated(subset=keys).any():
            sample_duplicates = df[df.duplicated(subset=keys)].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['SCENARIO'] = df['SCENARIO'].apply(to_str_or_none)
            df['ACT_WEEK_SO'] = df['ACT_WEEK_SO'].astype(int)
            df['DMD_TYPE'] = df['DMD_TYPE'].apply(to_str_or_none)
            df['SALES_COMPANY'] = df['SALES_COMPANY'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['CATEGORY_NAME'] = df['CATEGORY_NAME'].apply(to_str_or_none)
            df['FCT_ACCOUNT'] = df['FCT_ACCOUNT'].apply(to_str_or_none)
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_MONTH_NAME'] = df['SEIHAN_MONTH_NAME'].apply(to_str_or_none)
            df['MONTH_DATE'] = pd.to_datetime(df['MONTH_DATE'], format='%d/%m/%Y')
            df['MONTH_DATE'] = df['MONTH_DATE'].dt.strftime('%Y/%m/%d')
            df['SEASONALITY'] = df['SEASONALITY'].astype(float)
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")

    if table_name.lower() == 'dmd_fct':
        
        columns = [
            'SCENARIO', 'ACT_WEEK_SO', 'DMD_TYPE', 'SALES_COMPANY', 'COUNTRY', 'CATEGORY_NAME', 'MODEL_CODE', 'MODEL_NAME', 'SEIHAN_YEAR', 'SEIHAN_FISCAL_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK', 'ACT_FCT', 'WEEK_DATE', 'SELLOUT_QTY1',
            'SELLABLE_INV_QTY1', 'SELLOUT_QTY2', 'SELLABLE_INV_QTY2', 'SELLOUT_QTY', 'SELLABLE_INV_QTY', 'PAST4WK', 'WOS', 'SEASONALITY', 'PROMO_IMPACT_PCT', 'EVENT_IMPACT_PCT', 'TREND_ADJ_PCT', 'RANGING_IMPACT_PCT', 'RR_UPPER',
            'RR_LOWER', 'SHORTAGE_IMPACT_PCT', 'NORM_RR', 'BASE_RR', 'ADJ_RR', 'ADJ2_RR', 'SEASONALITY_IMPACT', 'PROMOTION_IMPACT', 'EVENT_IMPACT', 'RANGING_IMPACT', 'TREND_ADJ_IMPACT', 'SHORTAGE_IMPACT', 'NATURAL_DEMAND'
        ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        keys = ['SCENARIO', 'ACT_WEEK_SO', 'DMD_TYPE', 'COUNTRY', 'MODEL_CODE', 'SEIHAN_YEAR', 'SEIHAN_WEEK']
        if df.duplicated(subset=keys).any():
            sample_duplicates = df[df.duplicated(subset=keys)].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(model_code_check)
            df['SCENARIO'] = df['SCENARIO'].apply(to_str_or_none)
            df['ACT_WEEK_SO'] = df['ACT_WEEK_SO'].astype(int)
            df['DMD_TYPE'] = df['DMD_TYPE'].apply(to_str_or_none)
            df['SALES_COMPANY'] = df['SALES_COMPANY'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['CATEGORY_NAME'] = df['CATEGORY_NAME'].apply(to_str_or_none)
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(to_str_or_none)
            df['MODEL_NAME'] = df['MODEL_NAME'].apply(to_str_or_none)
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_FISCAL_YEAR'] = df['SEIHAN_FISCAL_YEAR'].astype('Int64')
            df['SEIHAN_MONTH_NAME'] = df['SEIHAN_MONTH_NAME'].apply(to_str_or_none)
            df['SEIHAN_WEEK'] = df['SEIHAN_WEEK'].astype('Int64')
            df['ACT_FCT'] = df['ACT_FCT'].astype('Int64')
            df['WEEK_DATE'] = pd.to_datetime(df['WEEK_DATE'], format='%m/%d/%Y')
            df['WEEK_DATE'] = df['WEEK_DATE'].dt.strftime('%Y/%m/%d')
            df['SELLOUT_QTY1'] = df['SELLOUT_QTY1'].astype('Float64')
            df['SELLABLE_INV_QTY1'] = df['SELLABLE_INV_QTY1'].astype('Float64')
            df['SELLOUT_QTY2'] = df['SELLOUT_QTY2'].astype('Float64')
            df['SELLABLE_INV_QTY2'] = df['SELLABLE_INV_QTY2'].astype('Float64')
            df['SELLOUT_QTY'] = df['SELLOUT_QTY'].astype('Float64')
            df['SELLABLE_INV_QTY'] = df['SELLABLE_INV_QTY'].astype('Float64')
            df['PAST4WK'] = df['PAST4WK'].astype('Float64')
            df['WOS'] = df['WOS'].astype(str)
            df['SEASONALITY'] = df['SEASONALITY'].astype('Float64')
            df['PROMO_IMPACT_PCT'] = df['PROMO_IMPACT_PCT'].astype('Float64')
            df['EVENT_IMPACT_PCT'] = df['EVENT_IMPACT_PCT'].astype('Float64')
            df['TREND_ADJ_PCT'] = df['TREND_ADJ_PCT'].astype('Float64')
            df['RANGING_IMPACT_PCT'] = df['RANGING_IMPACT_PCT'].astype('Float64')
            df['RR_UPPER'] = df['RR_UPPER'].astype('Float64')
            df['RR_LOWER'] = df['RR_LOWER'].astype('Float64')
            df['SHORTAGE_IMPACT_PCT'] = df['SHORTAGE_IMPACT_PCT'].astype('Float64')
            df['NORM_RR'] = df['NORM_RR'].astype('Float64')
            df['BASE_RR'] = df['BASE_RR'].astype('Float64')
            df['ADJ_RR'] = df['ADJ_RR'].astype('Float64')
            df['ADJ2_RR'] = df['ADJ2_RR'].astype('Float64')
            df['SEASONALITY_IMPACT'] = df['SEASONALITY_IMPACT'].astype('Float64')
            df['PROMOTION_IMPACT'] = df['PROMOTION_IMPACT'].astype('Float64')
            df['EVENT_IMPACT'] = df['EVENT_IMPACT'].astype('Float64')
            df['RANGING_IMPACT'] = df['RANGING_IMPACT'].astype('Float64')
            df['TREND_ADJ_IMPACT'] = df['TREND_ADJ_IMPACT'].astype('Float64')
            df['SHORTAGE_IMPACT'] = df['SHORTAGE_IMPACT'].astype('Float64')
            df['NATURAL_DEMAND'] = df['NATURAL_DEMAND'].astype('Float64')
            
            return df
        
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")

    if table_name.lower() == 'dmd_param_fct_window':
        
        columns = [
            'SEIHAN_YEAR', 'SEIHAN_FISCAL_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK', 'ACT_FCT'
        ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        if df.duplicated().any():
            sample_duplicates = df[df.duplicated()].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['SEIHAN_YEAR'] = df['SEIHAN_YEAR'].astype('Int64')
            df['SEIHAN_FISCAL_YEAR'] = df['SEIHAN_FISCAL_YEAR'].astype('Int64')
            df['SEIHAN_MONTH_NAME'] = df['SEIHAN_MONTH_NAME'].apply(to_str_or_none)
            df['SEIHAN_WEEK'] = df['SEIHAN_WEEK'].astype('Int64')
            df['ACT_FCT'] = df['ACT_FCT'].astype('Int64')
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")

    if table_name.lower() == 'fct_flag':
            
        columns = [
            'SCENARIO', 'COUNTRY', 'FCT_ACCOUNT', 'CATEGORY_NAME', 'MODEL_CODE', 'MODEL_NAME', 'FCT_ACCOUNT_FLAG', 'FCT_MODEL', 'FCT_FLAG', 'DMD_TYPE', 'SELLIN_OVERWRITE'
        ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        keys = ['SCENARIO', 'FCT_ACCOUNT', 'MODEL_CODE']
        if df.duplicated(subset=keys).any():
            sample_duplicates = df[df.duplicated(subset=keys)].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(model_code_check)
            df['SCENARIO'] = df['SCENARIO'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['FCT_ACCOUNT'] = df['FCT_ACCOUNT'].apply(to_str_or_none)
            df['CATEGORY_NAME'] = df['CATEGORY_NAME'].apply(to_str_or_none)
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(to_str_or_none)
            df['MODEL_NAME'] = df['MODEL_NAME'].apply(to_str_or_none)
            df['FCT_ACCOUNT_FLAG'] = df['FCT_ACCOUNT_FLAG'].apply(to_str_or_none)
            df['FCT_MODEL'] = df['FCT_MODEL'].apply(to_str_or_none)
            df['FCT_FLAG'] = df['FCT_FLAG'].apply(to_str_or_none)
            df['DMD_TYPE'] = df['DMD_TYPE'].apply(to_str_or_none)
            df['SELLIN_OVERWRITE'] = df['SELLIN_OVERWRITE'].apply(to_str_or_none)
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")
                
    if table_name.lower() == 'fct_type':
            
        columns = [
            'FCT_TYPE'
            ]
            
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        if df.duplicated().any():
            sample_duplicates = df[df.duplicated()].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['FCT_TYPE'] = df['FCT_TYPE'].apply(to_str_or_none)
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")
            
    if table_name.lower() == 'dmd_type':
            
        columns = [
            'DMD_TYPE'
            ]
            
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        if df.duplicated().any():
            sample_duplicates = df[df.duplicated()].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['DMD_TYPE'] = df['DMD_TYPE'].apply(to_str_or_none)
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")
                    
    if table_name.lower() == 'cat_map':
        
        columns = [
                'BIZ_GROUP', 'CATEGORY_CODE', 'SYSTEM_CATEGORY_NAME', 'CATEGORY_NAME', 'DUMMY_MODEL_CODE', 'DUMMY_MODEL_NAME'
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        if df.duplicated().any():
            sample_duplicates = df[df.duplicated()].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['BIZ_GROUP'] = df['BIZ_GROUP'].apply(to_str_or_none)
            df['CATEGORY_CODE'] = df['CATEGORY_CODE'].apply(to_str_or_none)
            df['SYSTEM_CATEGORY_NAME'] = df['SYSTEM_CATEGORY_NAME'].apply(to_str_or_none)
            df['CATEGORY_NAME'] = df['CATEGORY_NAME'].apply(to_str_or_none)
            df['DUMMY_MODEL_CODE'] = df['DUMMY_MODEL_CODE'].apply(to_str_or_none)
            df['DUMMY_MODEL_NAME'] = df['DUMMY_MODEL_NAME'].apply(to_str_or_none)
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")
        
    if table_name.lower() == 'cat_map2':
        
        columns = [
                'CATEGORY_NAME'
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        if df.duplicated().any():
            sample_duplicates = df[df.duplicated()].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['CATEGORY_NAME'] = df['CATEGORY_NAME'].apply(to_str_or_none)
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")

    if table_name.lower() == 'sub_cat_map':
        
        columns = [
                'CATEGORY_NAME', 'SUB_CATEGORY'
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        if df.duplicated().any():
            sample_duplicates = df[df.duplicated()].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['CATEGORY_NAME'] = df['CATEGORY_NAME'].apply(to_str_or_none)
            df['SUB_CATEGORY'] = df['SUB_CATEGORY'].apply(to_str_or_none)
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")

    if table_name.lower() == 'fct_account_map':
        
        columns = [
                'FSMC', 'SALES_COMPANY', 'COUNTRY', 'FCT_ACCOUNT', 'ACCOUNT_GROUP', 'CH1', 'CH2', 'CH3'
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        if df.duplicated().any():
            sample_duplicates = df[df.duplicated()].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['FSMC'] = df['FSMC'].apply(to_str_or_none)
            df['SALES_COMPANY'] = df['SALES_COMPANY'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['FCT_ACCOUNT'] = df['FCT_ACCOUNT'].apply(to_str_or_none)
            df['ACCOUNT_GROUP'] = df['ACCOUNT_GROUP'].apply(to_str_or_none)
            df['CH1'] = df['CH1'].apply(to_str_or_none)
            df['CH2'] = df['CH2'].apply(to_str_or_none)
            df['CH3'] = df['CH3'].apply(to_str_or_none)
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")

    if table_name.lower() == 'sc_map':
        
        columns = [
                'FSMC', 'SALES_COMPANY', 'COUNTRY', 'LC', 'RC'
            ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        if df.duplicated().any():
            sample_duplicates = df[df.duplicated()].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['FSMC'] = df['FSMC'].apply(to_str_or_none)
            df['SALES_COMPANY'] = df['SALES_COMPANY'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['LC'] = df['LC'].apply(to_str_or_none)
            df['RC'] = df['RC'].apply(to_str_or_none)
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")
            
    if table_name.lower() == 'seihan_param':
        
        columns = [
                'SCENARIO', 'FCT_TYPE', 'MODEL_CODE', 'COUNTRY', 'FACTORY_LOCATION', 'TR_TYPE', 'WAREHOUSE', 'PO_FROZEN_PERIOD', 'FACTORY_LEAD_TIME', 'ETA_FROZEN_PERIOD', 'ALLOCATION_YEAR',
                'ALLOCATION_WEEK', 'EOL_SEIHAN_YEAR', 'EOL_SEIHAN_WEEK', 'EOL_QTY', 'LAST_SHIP_SEIHAN_YEAR', 'LAST_SHIP_SEIHAN_WEEK', 'LAST_SHIP_QTY', 'MCQ', 'SELLIN_LEAD_TIME'
            ]
        
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)} in table {table_name}")
        df = df[columns]
        
        keys = ['SCENARIO', 'FCT_TYPE', 'MODEL_CODE', 'COUNTRY']
        if df.duplicated(subset=keys).any():
            sample_duplicates = df[df.duplicated(subset=keys)].head(5)
            raise ValueError(f"Duplicated records found in table {table_name}", sample_duplicates.to_dict(orient='records'))
        
        try:
            df['SCENARIO'] = df['SCENARIO'].apply(to_str_or_none)
            df['FCT_TYPE'] = df['FCT_TYPE'].apply(to_str_or_none)
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(model_code_check)
            df['MODEL_CODE'] = df['MODEL_CODE'].apply(to_str_or_none)
            df['COUNTRY'] = df['COUNTRY'].apply(to_str_or_none)
            df['FACTORY_LOCATION'] = df['FACTORY_LOCATION'].apply(to_str_or_none)
            df['TR_TYPE'] = df['TR_TYPE'].apply(to_str_or_none)
            df['WAREHOUSE'] = df['WAREHOUSE'].apply(to_str_or_none)
            df['PO_FROZEN_PERIOD'] = df['PO_FROZEN_PERIOD'].astype('Int64')
            df['FACTORY_LEAD_TIME'] = df['FACTORY_LEAD_TIME'].astype('Int64')
            df['ETA_FROZEN_PERIOD'] = df['ETA_FROZEN_PERIOD'].astype('Int64')
            df['ALLOCATION_YEAR'] = df['ALLOCATION_YEAR'].astype('Int64')
            df['ALLOCATION_WEEK'] = df['ALLOCATION_WEEK'].astype('Int64')
            df['EOL_SEIHAN_YEAR'] = df['EOL_SEIHAN_YEAR'].astype('Int64')
            df['EOL_SEIHAN_WEEK'] = df['EOL_SEIHAN_WEEK'].astype('Int64')
            df['EOL_QTY'] = df['EOL_QTY'].astype('Int64')
            df['LAST_SHIP_SEIHAN_YEAR'] = df['LAST_SHIP_SEIHAN_YEAR'].astype('Int64')
            df['LAST_SHIP_SEIHAN_WEEK'] = df['LAST_SHIP_SEIHAN_WEEK'].astype('Int64')
            df['LAST_SHIP_QTY'] = df['LAST_SHIP_QTY'].astype('Int64')
            df['MCQ'] = df['MCQ'].astype('Int64')
            df['SELLIN_LEAD_TIME'] = df['SELLIN_LEAD_TIME'].astype('Int64')
            
            return df
            
        except Exception as e:
            print('Error: ', e)
            raise ValueError(f"Error while preprocessing: {e}")

def scheduled_db_update(server, database, username, password, condition, global_df):
    """
    Schedule job to update the database with the latest data
    """
    logging.basicConfig(filename='db_update.log', level=logging.INFO, 
                        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    log = logging.getLogger(__name__)

    try:
        log.info("Starting database update")
        test = global_df['time_w'].copy()
        test = test.head(5)
        
        server = 'JPC00228406.jp.sony.com'
        database = 'ELCONDOR_SG_DEV_NEW_DB'
        username = 'sa'
        password = 'ElCondor2.0!$'
        table_name = 'time_w_test'
        
        save_to_sql_server(test, 'time_w_test', server, database, username, password, 'append')
        
        log.info("Database update completed")
        
        # for table_name, df in global_df.items():
        #     df = preprocess_check(table_name, df)
        #     delete_records(table_name, server, database, username, password, condition)
        #     save_to_sql_server(df, table_name, server, database, username, password, 'append')
        #     log.info(f"Scheduled update has been executed successfully. Saved to SQL Server for table: {table_name}")
        
        # dmd_fct_test = pd.read_csv('test.csv')
        # server = 'JPC00228406.jp.sony.com'
        # database = 'ELCONDOR_SG_DEV_NEW_DB'
        # username = 'sa'
        # password = 'ElCondor2.0!$'
        # df = preprocess_check('DMD_FCT_MODEL_TESTING', dmd_fct_test)
        # save_to_sql_server(df, 'DMD_FCT_MODEL_TESTING', server, database, username, password, 'append')
        # log.info("Scheduled update has been executed successfully. Saved to SQL Server for table DMD_FCT_MODEL_TESTING")
        # global src_df
        # for i in src_df:
        #     save_to_sql_server(i, i)
        #     log.info("Scheduled update has been executed successfully. Saved to SQL Server for table ", i)
    except Exception as e:
        log.error(f"Error during the scheduled update for table: {table_name}", exc_info=True)

def update_scenario_act_week(server, database, username, password, global_df):
    """
    Scheduled job to update act_week_so and act_week_si in scenario_map table
    """
    
    logging.basicConfig(filename='db_update_act.log', level=logging.INFO, 
                        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    log = logging.getLogger(__name__)

    try:
        log.info("Starting database update")
        
        today = datetime.now().strftime('%A')
        current_year = datetime.now().year
        # today = 'Thursday'
        
        if today == 'Thursday':
            # live_scenario['ACT_WEEK_SO'] = live_scenario['ACT_WEEK_SO'] + 1
            global_df['scenario_map']['ACT_WEEK_SO'] = global_df['scenario_map']['ACT_WEEK_SO'].astype(int)            
            global_df['factor_acct']['ACT_WEEK_SO'] = global_df['factor_acct']['ACT_WEEK_SO'].astype(int)
            global_df['seasonality_acct']['ACT_WEEK_SO'] = global_df['seasonality_acct']['ACT_WEEK_SO'].astype(int)
            global_df['factor']['ACT_WEEK_SO'] = global_df['factor']['ACT_WEEK_SO'].astype(int)
            global_df['pricing']['ACT_WEEK_SO'] = global_df['pricing']['ACT_WEEK_SO'].astype(int)
            global_df['seasonality']['ACT_WEEK_SO'] = global_df['seasonality']['ACT_WEEK_SO'].astype(int)
            global_df['event']['ACT_WEEK_SO'] = global_df['event']['ACT_WEEK_SO'].astype(int)
            
            if int(str(global_df['scenario_map'].loc[global_df['scenario_map']['SCENARIO'] == 'LIVE', 'ACT_WEEK_SO'] + 1)[::-2]) > \
                global_df['time_w'][global_df['time_w']['SEIHAN_YEAR'] == current_year].max()['SEIHAN_WEEK']:
                
                # take the first 4 digits and +1 , then last 2 digits are 01
                current_year = str(current_year + 1)
                current_year += '01'
                global_df['scenario_map'].loc[global_df['scenario_map']['SCENARIO'] == 'LIVE', 'ACT_WEEK_SO'] = current_year
                global_df['factor_acct'].loc[global_df['factor_acct']['SCENARIO'] == 'LIVE', 'ACT_WEEK_SO'] = current_year
                global_df['seasonality_acct'].loc[global_df['seasonality_acct']['SCENARIO'] == 'LIVE', 'ACT_WEEK_SO'] = current_year
                global_df['factor'].loc[global_df['factor']['SCENARIO'] == 'LIVE', 'ACT_WEEK_SO'] = current_year
                global_df['pricing'].loc[global_df['pricing']['SCENARIO'] == 'LIVE', 'ACT_WEEK_SO'] = current_year
                global_df['seasonality'].loc[global_df['seasonality']['SCENARIO'] == 'LIVE', 'ACT_WEEK_SO'] = current_year
                global_df['event'].loc[global_df['event']['SCENARIO'] == 'LIVE', 'ACT_WEEK_SO'] = current_year
            else:
                global_df['scenario_map'].loc[global_df['scenario_map']['SCENARIO'] == 'LIVE', 'ACT_WEEK_SO'] += 1
                global_df['factor_acct'].loc[global_df['factor_acct']['SCENARIO'] == 'LIVE', 'ACT_WEEK_SO'] += 1
                global_df['seasonality_acct'].loc[global_df['seasonality_acct']['SCENARIO'] == 'LIVE', 'ACT_WEEK_SO'] += 1
                global_df['factor'].loc[global_df['factor']['SCENARIO'] == 'LIVE', 'ACT_WEEK_SO'] += 1
                global_df['pricing'].loc[global_df['pricing']['SCENARIO'] == 'LIVE', 'ACT_WEEK_SO'] += 1
                global_df['seasonality'].loc[global_df['seasonality']['SCENARIO'] == 'LIVE', 'ACT_WEEK_SO'] += 1
                global_df['event'].loc[global_df['event']['SCENARIO'] == 'LIVE', 'ACT_WEEK_SO'] += 1

            log.info("Updating ACT_WEEK_SO")
            
        if today == 'Monday':
            global_df['scenario_map']['ACT_WEEK_SI'] = global_df['scenario_map']['ACT_WEEK_SI'].astype(int)
            
            if int(str(global_df['scenario_map'].loc[global_df['scenario_map']['SCENARIO'] == 'LIVE', 'ACT_WEEK_SI'] + 1)[::-2]) > \
                global_df['time_w'][global_df['time_w']['SEIHAN_YEAR'] == current_year].max()['SEIHAN_WEEK']:
                    
                    current_year = str(current_year + 1)
                    current_year += '01'
                    global_df['scenario_map'].loc[global_df['scenario_map']['SCENARIO'] == 'LIVE', 'ACT_WEEK_SI'] = current_year
            else:
                global_df['scenario_map'].loc[global_df['scenario_map']['SCENARIO'] == 'LIVE', 'ACT_WEEK_SI'] += 1
            
            log.info("Updating ACT_WEEK_SI")
        
        log.info("Database update completed for " + today)
        # if today in ['Thursday', 'Monday']:
        #     # where_conditions = f"SCENARIO = 'LIVE'"
        #     # delete_records(server, database, username, password, table_name=table_name, where_conditions=where_conditions)
        #     # save_to_sql_server(live_scenario, table_name, server, database, username, password, 'append')
            
        #     # also update global_df , removing the old scenario_map and adding the new one
        #     global_df['scenario_map'] = global_df['scenario_map'][global_df['scenario_map']['SCENARIO'] != 'LIVE']
        #     global_df['scenario_map'] = pd.concat([global_df['scenario_map'], live_scenario], ignore_index=True)
                        
            # log.info("Database update completed for " + today)
        
    except Exception as e:
        log.error(f"Error during the scheduled update: {e}", exc_info=True)
        
def product_account_check(df, table_name, product_map, account_map):
    """
    If dataframe contains model_code or account_code, check if existing in product_map or account_map
    
    If not, raise an error
    """
    
    model_code = 'MODEL_CODE'
    account_code = 'ACCOUNT_CODE'
    
    if table_name == 'product_map' or table_name == 'account_map':
        return    
    else:
        if model_code in df.columns:
            missing_models = [model for model in df[model_code].unique() if model not in product_map['MODEL_CODE'].unique()]
            if missing_models:
                raise ValueError(f"Model codes not found in product map: {', '.join(missing_models)}")
        
        if account_code in df.columns:
            missing_accounts = [account for account in df[account_code].unique() if account not in account_map['FCT_ACCOUNT'].unique()]
            if missing_accounts:
                raise ValueError(f"Account codes not found in account map: {', '.join(missing_accounts)}")