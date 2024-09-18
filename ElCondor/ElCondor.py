import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
from datetime import  date
import time
from datetime import timedelta

import pyodbc
#from fbprophet import Prophet
import pandas as pd

from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import statsmodels.api as sm
import threading
from tqdm import tqdm
from utils.data_connection import get_db_data
import os
import copy

'''
General issues to be solved:
- Make an detect and alert scheme when we set up values and inventory becoemes negative
- Scheme for unrolled status (dependency calculation status?)
'''

def date_from_cy_wk(year, week):
    return date.fromisocalendar(year, week,1)
vec_date_from_cy_wk = np.vectorize(date_from_cy_wk, otypes=[np.datetime64])


def date_from_fy_wk(year, week):
    specialdate =  (week ==14 and date.fromisocalendar(year, 14,2).month == 3) # 年度の切り替わりWK14が3月の場合、特殊ケースとして次年度に含める
    month = date.fromisocalendar(year, week,2).month
    result =date.fromisocalendar(year, week,2) if  (month >=4 and week != 1) or specialdate else date.fromisocalendar(year+1, week,2) #WK1が12月の場合、CYと同じルールを使う
    return result
vec_date_from_fy_wk = np.vectorize(date_from_fy_wk, otypes=[np.datetime64])

#===
# Test 


import datetime
 
def date_to_cy_mth_wk(date_obj):
    # Convert the date string to a datetime object
    #date_obj = datetime.datetime.strptime(date_str, '%Y/%m/%d')
   
    # Get the week number
    week_number = date_obj.isocalendar()[1]
   
    # Determine the month based on the majority of days in the week
    next_week = date_obj + datetime.timedelta(days=7)
    if next_week.month != date_obj.month:
        month_name = next_week.strftime("%B")
        calendar_year = next_week.year
    else:
        month_name = date_obj.strftime("%B")
        calendar_year = date_obj.year 
   
    return calendar_year, week_number, month_name
#wk, mth = date_to_cy_mth_wk('2024/1/29')
 
def date_to_cy_fy_mth_wk(date_obj):
    # Convert the date string to a datetime object
    #date_obj = datetime.datetime.strptime(date_str, '%Y/%m/%d')
   
    # Get the week number
    week_number = date_obj.isocalendar()[1]
   
    # Determine the month based on the majority of days in the week
    next_week = date_obj + datetime.timedelta(days=7)
    if next_week.month != date_obj.month:
        month_name = next_week.strftime("%b").upper()
        calendar_year = next_week.year
    else:
        month_name = date_obj.strftime("%b").upper()
        calendar_year = date_obj.year 
    
    fiscal_year = calendar_year - 1 if month_name in ['JAN', 'FEB', 'MAR'] else calendar_year

    return calendar_year, fiscal_year, week_number, month_name
#wk, mth = date_to_cy_mth_wk('2024/1/29')

class PSI:
    def __init__(self, start_date = None, end_date = None, br_list=['Branch A', 'Branch B', 'Branch C'], rt_name = 'Root', predecessor = None, product_code=None):

        # Get current date
        _today = pd.to_datetime("today")
        # Generate time array (set current monday if self = None)
        self.start_date = _today - pd.to_timedelta(_today.weekday(), unit='D') if start_date is None else pd.to_datetime(start_date)
        # Set ACT date
        self.act_date = self.start_date
        # Create branch list
        self.br_list = br_list
        # Generate branch dataframe
        self.end_date = self.start_date + pd.DateOffset(weeks=52) if end_date is None else pd.to_datetime(end_date)

        self._rootname = rt_name

        self.nsp = None
        self.mpcost = None
        self._free_alloc = 999999

        self.df_br = pd.DataFrame( index=pd.MultiIndex.from_product([self.br_list, pd.date_range(start=self.start_date, end=self.end_date, freq='W-MON')], 
                                names=['node','date']), 
                                columns=['P', 'S', 'I','demand','canib_loss','tgt_wos','std_ds','min_stock','prev_inv','RR','ADJ','week','month','year','fy', 'pre_S', 'pre_PI','io_delta','tgt_stock'])
        # Generate root dataframe
        self.df_rt = pd.DataFrame( index=pd.MultiIndex.from_product([[rt_name], pd.date_range(start=self.start_date, end=self.end_date, freq='W-MON')], 
                                names=['node','date']), 
                                columns=['INCOMING', 'S', 'I','demand','std_ds','prev_inv','RR','ADJ','EOL','EOL_QTY','FROZEN','CAP','OSI','week','month','year','fy'])
                
        #self.df = self.df_br.fillna(0)

        
        self.df_br[['P', 'S', 'I','demand','canib_loss','prev_inv','RR','ADJ','std_ds','min_stock','tgt_wos', 'pre_S', 'pre_PI','io_delta','tgt_stock']]=0.0
        self.df_rt[['INCOMING', 'S', 'I','demand','std_ds','prev_inv','RR','EOL','EOL_QTY','CAP','OSI']]=0

        # 列ごとのデータ型を辞書で指定
        dtype_dict = {
            'INCOMING': 'float64',
            'S': 'float64',
            'I': 'float64',
            'demand': 'float64',
            'std_ds': 'float64',
            'prev_inv': 'float64',
            'RR': 'float64',
            'ADJ': 'object',
            'EOL': 'object',
            'EOL_QTY': 'float64',
            'FROZEN': 'object',
            'CAP': 'float64',
            'OSI': 'float64',
            'week': 'object',
            'month': 'object',
            'year': 'object',
            'fy': 'object'
        }
        
        self.df_rt = self.df_rt.astype(dtype_dict)

        dtype_dict = {
            'P': 'float64',
            'S': 'float64',
            'I': 'float64',
            'demand': 'float64',
            'canib_loss': 'float64',
            'tgt_wos': 'float64',
            'std_ds': 'float64',
            'min_stock': 'float64',
            'prev_inv': 'float64',
            'RR': 'float64',
            'ADJ': 'float64',
            'week': 'object',
            'month': 'object',
            'year': 'object',
            'fy': 'object',
            'pre_S': 'float64',
            'pre_PI': 'float64'
        }

        # データフレームにデータ型を適用
        self.df_br = self.df_br.astype(dtype_dict)


        # Auto Generate year and week columns
        #self.df_br['week'] = [idx[1].isocalendar().week for idx in self.df_br.index]
        #self.df_br['year'] = [idx[1].year for idx in self.df_br.index]
        #self.df_rt['week'] = [idx[1].isocalendar().week for idx in self.df_rt.index]
        #self.df_rt['year'] = [idx[1].year for idx in self.df_rt.index]

        self.df_br[['year', 'fy','week','month']] = [date_to_cy_fy_mth_wk(idx[1]) for idx in self.df_br.index]
        self.df_rt[['year', 'fy','week','month']] = [date_to_cy_fy_mth_wk(idx[1]) for idx in self.df_rt.index]

        # Set Precesessor product 
        self.code = product_code
        self.predecessor = predecessor 
        if self.predecessor is not None:
            # METHOD 1 ==
            #self.psi[product_code].df_br['canib_loss'] = self.psi[product_code].df_br['demand'].sub(self.psi[predecessor_dict[product_code]].df_br['S'],fill_value = 0)#
            #self.psi[product_code].df_br['canib_loss'] = self.psi[product_code].df_br['canib_loss'] .apply(lambda x: max(x, 0)) # negatives to -> zero
            # METHOD 2 ==
            self.df_br['pre_S'] = self.predecessor.df_br['S'].fillna(0)  
            self.df_br['canib_loss'] = (self.df_br['demand'] - self.df_br['pre_S']).fillna(0).clip(lower=0)

            self.df_br['pre_PI'] = self.predecessor.df_br['prev_inv'].fillna(0) + self.predecessor.df_br['P'].fillna(0)
        
        self._cache_fx_x=None
        self._cache_fx_r=None
        self._cache_fx_y=None
        self._cache_achvrate=None
        self._cache_df_curr=None
        self._cache_df_curr_hi=None

    '''
    @property
    def df_br(self):
        return self.df_br.loc[:, self._df.columns != 'RR']
    
    @df_br.setter
    def df_br(self, value):
        self.df_br = value
    '''

    def update_value(self, new_value):
        with self.lock:
            self.act_date = new_value

    def __repr__(self):
        #result = self.df_br[self.df_br['month'].isin(['DEC','JAN'])].head(60)
        #result.loc[:,['P','I','I']].stack().unstack(level=1)
        #result.reorder_levels([1, 0])
        return   str(self.print())


    @classmethod
    def from_weeks(cls, start_date, num_weeks, br_list=['Branch A'], rt_name = 'Root', predecessor = None, product_code=None):
        # Set and calculate start end info and recall conttructor
        # <to do: use kwargs to transpass to _init_> 
        start_date = pd.to_datetime(start_date)
        end_date = start_date + pd.Timedelta(weeks=num_weeks - 1)
        return cls(start_date, end_date, br_list, rt_name, predecessor, product_code)

    def calc_tgt_wos(self):
        # Calculate the target wos based on (std_ds, min_stock, and RR values)
        self.df_br['RR'] = self.df_br['RR'].replace(0, 1e-8)
        condition = self.df_br['min_stock'] / self.df_br['RR'] == 100000000
        self.df_br['tgt_wos'] = np.where(condition, 0.01, self.df_br['min_stock'] / self.df_br['RR'])
        # Calculate MAX
        self.df_br['tgt_wos'] = np.maximum(self.df_br['tgt_wos'], self.df_br['std_ds'])
        self.df_br['tgt_wos'] = self.df_br['tgt_wos'].replace(0, 1e-8)

    def calc_RR(self):
        # Calculate 'RR' as the rolling mean of the next 4 weeks of sales for each store
        self.df_br['RR'] = self.df_br.groupby(level=0,group_keys=False)['demand'].apply(
            lambda x: x.rolling(window=4, min_periods=1).mean().shift(-4)+0.00000001
        )
        self.df_br['RR']= self.df_br['RR'].fillna(value=0)
        self.calc_tgt_wos()


    def copy_from(self, psi_class):
        if isinstance(psi_class, PSI):
            self.df_br.update(psi_class.df_br)
            self.df_rt.update(psi_class.df_rt)
        else:
            print("ERROR: Please input a PSI class instance")


    def update(self, _df_br = None, _df_rt = None, _calc_rr = True):
        if _df_br is not None:
            # <to-do: add a dataframe validation here> 
            if _df_br.index.duplicated().sum():
                a = 1
            self.df_br.update(_df_br)

        if _df_rt is not None:
            # <to-do: add a dataframe validation here> 
            self.df_rt.update(_df_rt)

        if _calc_rr is not None:
            # <to-do: add a dataframe validation here> 
            self.calc_RR()

    def update_data(self, data_df,store_list=['A'], tgt_df = 'br'):
        #TBD: 入れるDataframeの形が特殊？
        '''
        issues to solve:
        - if data frame is not in same structure/format, or not multi index?
        - if data start is earlier, trim or giveup?
        - if data has store info but store is determined in params?
        - if there is no data index? > error handlerc
        - if the date format (start week) is not aligned auto-correct or giveup?
        - Current flow is considering single index (only date as level=0)
        - If ACT date is set, it should only use until that date.
        - Error handle when [P] [S] [I] items were not found or missing
        '''
        # Create a data copy
        local_df = data_df.copy()

        # Calculate start and end date
        data_start_date = local_df.index.get_level_values(0).min()
        print("STT:",data_start_date)
        data_end_date = local_df.index.get_level_values(0).max()
        print("END:", data_end_date)

        if data_start_date < self.start_date:
            raise ValueError(f"Data start date ({data_start_date.date()}) is earlier than PSI start date ({self.start_date.date()}).")
        if data_end_date > self.end_date:
            print(local_df.index.get_level_values(0))
            # Trim dates after end_date //  index level=0 : date
            local_df = local_df.loc[local_df.index.get_level_values(0) <= self.end_date]
            data_end_date = self.end_date
            print(f"Data is cut to the PSI end date ({data_end_date.date()}).")

        # Reset index:  before: ['date'] after: ['node','date'] 
        local_df.index = pd.MultiIndex.from_product([store_list, local_df.index.to_list()])

        if tgt_df == 'br':
            self.df_br.update(local_df)
            self.act_date = data_end_date
            print(f"Branch data is updated until {data_end_date.date()}.")
        elif tgt_df == 'tr':
            self.df_rt.update(local_df)
            self.act_date = data_end_date
            print(f"Root data is updated until {data_end_date.date()}.")
        else:
            print(f"Error: target dataframe tgt_df is invalid: {tgt_df}.")

    def sort_by_date(self):
        # Sord data by date
        self.df = self.df_br.sort_index()

    def roll_pi(self):
        '''
        Issues to be solved:
        - Roll branch or root? selection
        - Need to set the standard ds (wos) by week and by account < actually fix 4 weeks 
        - DS base is next sell-out, need to consider future n weeks sell-out average
        - RR utilizaton (RR as a field, then RR calculation should be n determinable)
        '''
        # Roll each branch P (auto demand calculation) and I 
        for name, group in  self.df_br.groupby(level=0): # loop by node level
            df=group
            def cus(ser):
                win = ser.index.tolist()
                # Calculate sell-in for each date
                self.df_br.loc[win[1],'P']=max([0, -self.df_br.loc[win[0],'I']+ self.df_br.loc[win[1],'S']+ \
                                   self.df['S'].shift(-2).loc[win[0:1]].mean()*4/1]) # Sell-out　０～1未満の平均、尚、0=n+1の実用祖（Shiftのため）
                self.df_br.loc[win[1],'I']=self.df_br.loc[win[0],'I']+ self.df_br.loc[win[1],'P'] - self.df_br.loc[win[1],'S']   
                return 0
            
            # Apply 'cus' function to each date item after 'act_date'
            df[(name, self.act_date):].P.rolling(window=3).apply(cus, raw=False)
            #print(name,'\n',df)
        
    def roll_i(self, _date=None,trim_so=False):
        if _date is None: #----- Full row calculation
            #display(self.df)
            tmp = self.df_br.copy()
            #print(tmp.groupby(level=0)['I'].transform(lambda x: x.cumsum()))  
            idx = pd.IndexSlice
            self.df_br.loc[idx[~self.df_br.index.get_level_values(1).isin([self.start_date])],'I'] = 0 # Mask 0 to Index level=1, index order 0 (second) 
            # Calculate accumulative inventory considering up-down from P and S gap
            self.df_br['I'] = self.df_br.groupby(level=0)['I'].transform(lambda x: x.cumsum()) + \
            self.df_br.groupby(level=0)['P'].apply(lambda x: pd.Series(np.cumsum(x[1:]), index=x.index)).fillna(0) - \
            self.df_br.groupby(level=0)['S'].apply(lambda x: pd.Series(np.cumsum(x[1:]), index=x.index)).fillna(0)
        else: #----------------- Specific week calculation (single)
            _date=pd.Timestamp(_date)
            def update_sellable_inv_qty(group):
                
                if _date in group.index.get_level_values(1):                                          # current week date 
                    current_week_row = group.loc[(slice(None), _date), :]                                   # Select current week row
                    last_week_date = _date - pd.Timedelta(weeks=1)                                          # previous week date
                    last_week_row = group.loc[(slice(None), last_week_date), :]
                    # Calculate inventory 
                    last_week_sellable_inv_qty = last_week_row['I'].sum() if not last_week_row.empty else 0
                    current_week_sellin_qty = current_week_row['P'].sum()
                    current_week_sellout_qty = current_week_row['S'].sum()
                    
                    if trim_so :
                        current_week_sellout_qty = max(min(last_week_sellable_inv_qty + current_week_sellin_qty , current_week_sellout_qty),0)
                        group.loc[(slice(None), _date), 'S'] = current_week_sellout_qty
                
                    updated_qty = last_week_sellable_inv_qty + current_week_sellin_qty - current_week_sellout_qty
                    group.loc[(slice(None), _date), 'I'] = updated_qty
                    
                return group.reset_index(level=0, drop=True)
            # Calculate latest week for all dealers
            self.df_br = self.df_br.groupby(level=0).apply(update_sellable_inv_qty)#.reset_index(drop=True)
            a=1

    def predict_s(self):
        # Use ARIMA for forecast prediction based on the ACT S date
        if pd.isnull(self.act_date):
            raise ValueError("Actual data is not available.")
        
        # Loop for each PSI by store level=0
        for name, group in  self.df_br.groupby(level=0):
            print(name)
            s_df = group['demand'].copy()
            if len(s_df) < 2:
                raise ValueError("Insufficient data for prediction.")
            # Find optimal ARIMA order
            act_n = group.index.get_level_values(1).tolist().index(self.act_date)
            #display(s_df[:act_n])
            arima_model = auto_arima(s_df[:act_n], seasonal=True, suppress_warnings=True)
            order = arima_model.order
            # Fit ARIMA model
            model = arima_model.fit(s_df[:act_n])
            # Predict for the next 5 weeks
            count_future = len(s_df[act_n+1:])
            #print(count_future)
            future_y, conf_int = model.predict(n_periods=count_future, return_conf_int=True)
            #print(future_y)
            s_df[act_n+1:] = future_y
            #print(f"Level1={name}, \n{s_df}")
            self.df_br.update(s_df)
            self.CalcRunrate()
        print("S values are predicted for the next 5 weeks.")

    def predict_s_state_space(self):
        if pd.isnull(self.act_date):
            raise ValueError("Actual data is not available.")

        for name, group in self.df_br.groupby(level=0, group_keys=False):
            print(name)
            s_df = group['demand'].reset_index(level=0, drop=True).copy()
            if len(s_df) < 2:
                raise ValueError("Insufficient data for prediction.")

            # act_date以前のデータを学習データとする
            act_n = group.index.get_level_values(1).tolist().index(self.act_date)
            training_data = s_df[:act_n]

            # 状態空間モデルの定義
            mod_season_trend_d = sm.tsa.UnobservedComponents(
                training_data,
                'local level',
                seasonal=52  # ここでは季節周期を12に設定
            )

            # モデルのフィット
            res_season_trend_d = mod_season_trend_d.fit(
                method='bfgs', 
                maxiter=100, 
                start_params=mod_season_trend_d.fit(method='nm', maxiter=100).params
            )

            # 予測を行う
            pred_start_date = self.act_date  # 予測開始日
            pred_end_date = group.index.get_level_values(1)[-1]     # 予測終了日
            #print(res_season_trend_d,pred_start_date, pred_end_date)
            pred = res_season_trend_d.predict(start=pred_start_date, end=pred_end_date)
            #print(pred)
            # predをDataFrameに変換
            #pred_df = pred.to_frame(name='demand')
            pred_df = pd.DataFrame({'demand': pred})
            # マルチインデックスの再構築
            pred_df['node'] = name  # 'node'列にnameを代入
            pred_df.set_index(['node', pred_df.index], inplace=True)  # 'node'と既存のインデックスでマルチインデックスを作成
            
            #print(pred_df)
            
            # 予測値を元のDataFrameに割り当て
            #s_df.loc[self.act_date:pred_end_date, 'demand'] = pred_df  # 予測値の割り当て

            # 元のクラス属性を更新
            self.df_br.update(pred_df)
            self.CalcRunrate()

        print("S values are predicted for the next period using State Space Model.")

    def predict_s_ETS(self):
        if pd.isnull(self.act_date):
            raise ValueError("Actual data is not available.")

        for name, group in self.df_br.groupby(level=0):
            s_df = group['demand'].reset_index(level=0, drop=True).copy()
            if len(s_df) < 2:
                raise ValueError("Insufficient data for prediction.")

            # act_date以前のデータを学習データとする
            act_n = group.index.get_level_values(1).tolist().index(self.act_date)
            training_data = s_df[:act_n]
            # fit in statsmodels

            model = ETSModel(
                training_data,
                error="add",
                trend="add",
                seasonal="add",
                damped_trend=True,
                seasonal_periods=52,
            )
            fit = model.fit(disp=0)



            # 予測を行う
            pred_start_date = self.act_date  # 予測開始日
            pred_end_date = group.index.get_level_values(1)[-1]     # 
            

            pred = fit.get_prediction(start=pred_start_date, end=pred_end_date)
            
            df = pred.summary_frame(alpha=0.05)


            #print(df)
            #df['demand'] = df['mean']
            # predをDataFrameに変換
            #pred_df = pred.to_frame(name='demand')
            
            pred_df = pd.DataFrame({'demand': df['mean']})
            # マルチインデックスの再構築
            pred_df['node'] = name  # 'node'列にnameを代入
            pred_df.set_index(['node', pred_df.index], inplace=True)  # 'node'と既存のインデックスでマルチインデックスを作成
            
            pred_df['demand'] = pred_df['demand'].clip(lower=0)
            #print(pred_df)
            
            # 予測値を元のDataFrameに割り当て
            #s_df.loc[self.act_date:pred_end_date, 'demand'] = pred_df  # 予測値の割り当て

            # 元のクラス属性を更新
            self.df_br.update(pred_df)
            self.CalcRunrate()

        print("S values are predicted for the next period using State Space Model.")



        '''
        def predict_s_prophet(self):
            if pd.isnull(self.act_date):
                raise ValueError("Actual data is not available.")

            for name, group in self.df_br.groupby(level=0):
                s_df = group[['date', 'demand']].rename(columns={'date': 'ds', 'demand': 'y'})
                if len(s_df) < 2:
                    raise ValueError("Insufficient data for prediction.")

                act_n = group.index.get_level_values(1).tolist().index(self.act_date)

                model = Prophet()
                model.fit(s_df[:act_n])
                
                future = model.make_future_dataframe(periods=len(s_df) - act_n)
                forecast = model.predict(future)

                # Update the original DataFrame
                s_df['y'] = forecast['yhat']
                self.df_br.update(s_df[['ds', 'y']].rename(columns={'ds': 'date', 'y': 'demand'}))
                self.CalcRunrate()

            print("S values are predicted for the next period using Prophet.")
        '''
    # El Condor 2.0 functions ========= Auto Allocation (wos base)

    def CalcRunrate(self):
        # Calculate 'RR' as the rolling mean of the next 4 weeks of sales for each store
        self.df_br['RR'] = self.df_br.groupby(level=0, group_keys=False)['demand'].apply(
            lambda x: x.rolling(window=4, min_periods=1).mean().shift(-4)
        )
        self.df_br['RR'] = self.df_br['RR']+0.00000001 # Add delta (small value) in order to avoid #zerodiv at qty from DS calc





    def AchvRate(self,_date): # Dataframe contains so, rr, previnv, stdds, adj
        # Now, extract the information for the selected date
        df2 = self.df_br.xs(_date, level='date')[['demand', 'prev_inv', 'RR','tgt_wos', 'pre_PI','ADJ']].rename(columns={'demand': 'so'})
        # Drop rows where prev_inv is NaN because there is no previous inventory for the first date
        df2 = df2.dropna(subset=['prev_inv'])
        df2['tgt_wos'].replace(0,1e-8,inplace =True)
        predecessor_PI = 0 if self.predecessor is None else df2['pre_PI']
        df2.drop('pre_PI', axis=1, inplace=True)
        #df2['ACHV_RATE'] = ((df2['prev_inv'] + predecessor_PI) - df2['so']).round(10) / (df2['RR'] * df2['tgt_wos'])  # diivdebyzero = 0

        df2['ACHV_RATE'] = ((df2['prev_inv'] + predecessor_PI) - df2['so'] + df2['ADJ']).round(10) / (df2['RR'] * df2['tgt_wos']) #  divide by zero  = 1
        #display(df2.style.format({'ACHV_RATE':'{:.2%}'}))
        return df2
        # Result: 100% = target stock (ratio\)


    def cache_AchvRate(self,_date): # Dataframe contains so, rr, previnv, stdds, adj

        #self._cache_achvrate = self.df_br.loc[(slice(None), _date), ['demand', 'prev_inv', 'RR', 'tgt_wos', 'pre_PI', 'ADJ','io_delta', 'tgt_stock']]#.dropna(subset=['prev_inv'])#★ dropna外す（ACTの最初の行しか該当しないため、FCT計算では出てこない）
        self._cache_achvrate = self._cache_df_curr[['demand', 'prev_inv', 'RR', 'tgt_wos', 'pre_PI', 'ADJ','io_delta', 'tgt_stock']].copy()
        
        
        
        
        #self._cache_achvrate.reset_index(level='date', drop=True, inplace=True)
        #self._cache_achvrate['tgt_wos'].replace(0, 1e-8, inplace=True) #★事前に一括計算をする --> CalcRRで計算済み 
        #predecessor_PI = 0 if self.predecessor is None else self._cache_achvrate['pre_PI']#★　Predecessorリンクする際に計算pre_PIを計算。ここで分岐しない。事前計算するならなくてよい
        #self._cache_achvrate.drop('pre_PI', axis=1, inplace=True)#★PREPIは事前計算するなら、なくてよい
        self._cache_achvrate['prev_inv'] = self._cache_achvrate['prev_inv'].astype(float)
        self._cache_achvrate['ACHV_RATE'] = ((self._cache_achvrate['prev_inv'] + self._cache_achvrate['io_delta']).round(10)) / (self._cache_achvrate['tgt_stock'])#★ RR*WOSは事前計算、PREPI-DMD+ADJも事前計算？

    @staticmethod
    def fx_y_LIST(df_achv):# Dataframe contains so, rr, previnv,  stdds, adj  ## list vERSION = slower than SERIES ver
        # 数値を四捨五入
        lst_achv = df_achv['ACHV_RATE'].round(10)
        # 重複を排除し、ソート
        lst_achv = np.sort(lst_achv.unique())
        return lst_achv.tolist()  # numpy arrayをリストに変換

    @staticmethod
    def fx_y_old(df_achv):# Dataframe contains so, rr, previnv, stdds, adj  ## SERIES ver
        '''
        Issues to be solved:
        - Currently the input is too unique (returnd df from ACHVRATE), needs to change to series in - series out 
            1. Create a function to get a specific date slice (dataframe)
            2. Change acvrate method to a date input -> series output
            3. Change fx_? functions to be series input -> series output (simple)
        '''
        # 数値を四捨五入
        df_achv['ACHV_RATE']=df_achv['ACHV_RATE'].fillna(0)
        #print(df_achv['ACHV_RATE'], type(df_achv['ACHV_RATE']))
        srs_achv = df_achv['ACHV_RATE'].round(10)
        # 重複を排除し、ソート
        srs_achv = pd.Series(srs_achv.unique()).sort_values().reset_index(drop=True)
        return srs_achv
    
    @staticmethod
    def fx_y(df_achv):
        # NumPyを使用してNaNを0で置き換える
        achv_rate_array = np.nan_to_num(df_achv['ACHV_RATE'].values, nan=0.0)
        
        # 小数点以下を丸める
        achv_rate_array = np.round(achv_rate_array, 10)
        
        # unique と sort の組み合わせ、np.unique は自動的にソートも行う
        unique_sorted_achv_rate = np.unique(achv_rate_array)
        
        # numpy 配列から pandas Series へ変換
        srs_achv = pd.Series(unique_sorted_achv_rate)
        
        return srs_achv

    def cache_fx_y(self):
        # NumPyを使用してNaNを0で置き換える
        achv_rate_array = np.nan_to_num(self._cache_achvrate['ACHV_RATE'].values, nan=0.0) #★ demand, prev_inv, RR, tgt_wos, ADJ, pre_PIがゼロじゃないことを計算前に保証すればこの行要らない。
        # 小数点以下を丸める
        achv_rate_array = np.round(achv_rate_array, 10)
        # unique と sort の組み合わせ、np.unique は自動的にソートも行う
        unique_sorted_achv_rate = np.unique(achv_rate_array)
        # numpy 配列から pandas Series へ変換
        self._cache_fx_y = pd.Series(unique_sorted_achv_rate)
        
    @staticmethod
    def fx_r_old(df_achv):# Dataframe contains so, rr, previnv, stdds, adj
        # Seriesをデータフレームに変換し、列名をAとする
        df_ret = PSI.fx_y(df_achv)
        # 元のdf_achv_numericにおけるrrとdsの積を計算し、重複するAの値に対して平均を取る
        df_tgtstock = df_achv.copy()
        df_tgtstock['ACHV_RATE']  = df_tgtstock['ACHV_RATE'].round(10)
        acc_avg = lambda x: df_tgtstock[df_tgtstock['ACHV_RATE'] <= x]['ACHV_RATE'].count()
        srs_map = df_ret.map(acc_avg)
        
        return srs_map    
    
    @staticmethod
    def fx_r(df_achv):
        # Dataframe contains so, rr, previnv, stdds, adj
        # Seriesをデータフレームに変換し、列名をAとする
        df_ret = PSI.fx_y(df_achv)
        # 元のdf_achv_numericにおけるrrとdsの積を計算し、重複するAの値に対して平均を取る
        df_tgtstock = df_achv.copy()
        df_tgtstock['ACHV_RATE']  = df_tgtstock['ACHV_RATE'].round(10)

        # NumPy配列に変換
        achv_rate_array = df_tgtstock['ACHV_RATE'].to_numpy()
        ret_values_array = df_ret.to_numpy()

        # ベクトル化された操作
        counts = np.sum(np.less_equal.outer(achv_rate_array, ret_values_array), axis=0)
        srs_map = pd.Series(counts, index=df_ret.index)

        return srs_map

    def cache_fx_r(self):
        # Dataframe contains so, rr, previnv, stdds, adj
        # Seriesをデータフレームに変換し、列名をAとする
        df_ret = self._cache_fx_y
        # 元のdf_achv_numericにおけるrrとdsの積を計算し、重複するAの値に対して平均を取る
        df_tgtstock = self._cache_achvrate['ACHV_RATE'].round(10)#★はAcvrate計算時にRoundしておけば、fx_y含めて削除可能。

        # NumPy配列に変換
        achv_rate_array = df_tgtstock.to_numpy()#★最初の行を省略
        ret_values_array = df_ret.to_numpy()#★最初を省略

        # ベクトル化された操作
        counts = np.sum(np.less_equal.outer(achv_rate_array, ret_values_array), axis=0)
        self._cache_fx_r = pd.Series(counts, index=df_ret.index)

    @staticmethod
    def fx_x(df_achv):# Dataframe contains so, rr, previnv, stdds, adj
        #FX_Yの差分配列を作る
        df_ydiff = PSI.fx_y(df_achv).diff().dropna()
        df_ydiff.reset_index(drop=True)
        df_ydiff.index -= 1        
        df_rwolast = PSI.fx_r(df_achv)[:-1]
        df_rwolast.reset_index(drop=True)
        # 掛け算
        df_ret = df_ydiff * df_rwolast
        # 累積和の計算
        df_ret = df_ret.cumsum()
        # 累積和のシリーズをゼロからスタートさせる
        #df_ret = pd.Series([0]).append(df_ret, ignore_index=True)
        df_ret = [0] if df_ret.empty else pd.concat([pd.Series([0]), df_ret], ignore_index=True)
        return df_ret

    def cache_fx_x(self):# Dataframe contains so, rr, previnv, stdds, adj
        df_ydiff = self._cache_fx_y.diff().dropna()
        df_ydiff.reset_index(drop=True)
        df_ydiff.index -= 1        
        df_rwolast = self._cache_fx_r[:-1]
        df_rwolast.reset_index(drop=True)
        # 掛け算
        df_ret = df_ydiff * df_rwolast
        # 累積和の計算
        df_ret = df_ret.cumsum()
        # 累積和のシリーズをゼロからスタートさせる
        #df_ret = pd.Series([0]).append(df_ret, ignore_index=True)
        self._cache_fx_x = [0] if df_ret.empty else pd.concat([pd.Series([0]), df_ret], ignore_index=True)
    
    @staticmethod
    def RatePerUnit_old(df_achv):# Dataframe contains so, rr, previnv, stdds, adj
        ### Returns the rate % increment per '1 unit'
        #round > unique > short process
        # Seriesをデータフレームに変換し、列名をAとする
        df_ret = PSI.fx_y(df_achv)
        # 元のdf_achv_numericにおけるrrとdsの積を計算し、重複するAの値に対して平均を取る
        df_tgtstock =df_achv.copy()
        df_tgtstock['ACHV_RATE']  = df_tgtstock['ACHV_RATE'].round(10)
        df_tgtstock['UNITS_PER_RATE'] = df_tgtstock['RR'] / 1 * df_tgtstock['tgt_wos'] /100
        # Rate per Unit を計算
        acc_avg = lambda x: 0.01/df_tgtstock[df_tgtstock['ACHV_RATE'] <= x]['UNITS_PER_RATE'].mean()
        srs_map = df_ret.map(acc_avg)
        return srs_map
    
    @staticmethod
    def RatePerUnit(df_achv):
        df_ret = PSI.fx_y(df_achv)
        df_tgtstock = df_achv.copy()
        df_tgtstock['ACHV_RATE'] = df_tgtstock['ACHV_RATE'].round(10)
        df_tgtstock['UNITS_PER_RATE'] = (df_tgtstock['RR'] * df_tgtstock['tgt_wos']) / 100

        # ACHV_RATE でソート
        df_sorted = df_tgtstock.sort_values('ACHV_RATE')
        
        # 累積平均の計算
        df_sorted['CUM_MEAN'] = df_sorted['UNITS_PER_RATE'].expanding().mean()

        # 累積平均を df_ret とマッピング
        df_ret_rounded = df_ret.round(10)
        srs_map = pd.merge_asof(df_ret_rounded.to_frame('ACHV_RATE'), df_sorted[['ACHV_RATE', 'CUM_MEAN']],
                                on='ACHV_RATE', direction='backward')['CUM_MEAN']

        # 最終的な結果
        result = 0.01 / srs_map

        return result



    def RatePerUnit_new(self):
        df_ret = self._cache_fx_y
        df_tgtstock = self._cache_achvrate
        df_tgtstock.loc[:,'ACHV_RATE'] = df_tgtstock.loc[:,'ACHV_RATE'].round(10)
        df_tgtstock.loc[:,'UNITS_PER_RATE'] = (df_tgtstock.loc[:,'RR'] * df_tgtstock.loc[:,'tgt_wos']) / 100
        # ACHV_RATE でソート
        df_sorted = df_tgtstock.sort_values('ACHV_RATE')
        # 累積平均の計算
        df_sorted['CUM_MEAN'] = df_sorted['UNITS_PER_RATE'].expanding().mean()
        # 累積平均を df_ret とマッピング
        df_ret_rounded = df_ret.round(10)
        srs_map = pd.merge_asof(df_ret_rounded.to_frame('ACHV_RATE'), df_sorted[['ACHV_RATE', 'CUM_MEAN']],
                                on='ACHV_RATE', direction='backward')['CUM_MEAN']
        result = 0.01 / srs_map

        return result

    @staticmethod
    def unit_ancher(rpu, fxx):
        # x分岐点(fxx)の各要素の差分を計算し、それをRatePerUnit配列(rpu)の対応する要素で割る
        try:
            if rpu.dtype == 'object':
                rpu = rpu.astype(float)
            diffs = np.diff(fxx) / rpu[:-1]
            # 累積和を計算
            cumulative_sum = np.cumsum(diffs)
            # 累積和の先頭と末尾に0を追加
            cumulative_sum_with_zeros = np.concatenate(([0], cumulative_sum))
            '''
            print('===============================\n',
                  "\nRPU: ", rpu,
                  "\nFXX: ", fxx, 
                  "\nCSUM: ",cumulative_sum)
            '''

        except Exception as e:
            cumulative_sum_with_zeros = np.array([0])

        return pd.Series(cumulative_sum_with_zeros)

    @staticmethod
    def fx_fill(df_achv, y):
        x_achv_rate = pd.Series(PSI.fx_x(df_achv))
        y_achv_rate = PSI.fx_y(df_achv)
        r_achv_rate = PSI.fx_r(df_achv)
        # Find the position where y would fit in the x_achv_rate
        y_position = x_achv_rate.searchsorted(y, side='right') - 1
        # If y is less than the smallest x or greater than the largest x, extrapolate linearly
        if y_position == -1:
            y_position = 0
        elif y_position >= len(x_achv_rate):
            y_position = len(x_achv_rate) - 1
        # Calculate the linear interpolation
        x_low = x_achv_rate.iloc[y_position]
        x_high = x_achv_rate.iloc[min(y_position+1, len(x_achv_rate)-1)]
        y_low = y_achv_rate.iloc[y_position]
        y_high = y_achv_rate.iloc[min(y_position+1, len(y_achv_rate)-1)]
        r = r_achv_rate.iloc[y_position]
        return (1/r) * (y - x_low) + y_low


    def fx_fill_new(self, y):
        x_achv_rate = pd.Series(self._cache_fx_x)
        y_achv_rate = self._cache_fx_y
        r_achv_rate = self._cache_fx_r
        # Find the position where y would fit in the x_achv_rate
        y_position = x_achv_rate.searchsorted(y, side='right') - 1
        # If y is less than the smallest x or greater than the largest x, extrapolate linearly
        if y_position == -1:
            y_position = 0
        elif y_position >= len(x_achv_rate):
            y_position = len(x_achv_rate) - 1
        # Calculate the linear interpolation
        x_low = x_achv_rate.iloc[y_position]
        #x_high = x_achv_rate.iloc[min(y_position+1, len(x_achv_rate)-1)]
        y_low = y_achv_rate.iloc[y_position]
        #y_high = y_achv_rate.iloc[min(y_position+1, len(y_achv_rate)-1)]
        r = r_achv_rate.iloc[y_position]
        return (1/r) * (y - x_low) + y_low
    


    def GrossDemand(self, base_date):
        '''
        Issues/requirements to be solved:
        - Currently the future weeks to be considered is fixed to 5 weeks, dpending on the average range it should be flex
        - In future similar information will go to root psi (RR?)
        '''
        def _nested_arr(n, stk_arr, src_arr):
            """
            Recursively builds a nested array by stacking columns and performing calculations on them.
            :param n      : Current iteration number.
            :param stk_arr: The array to be horizontally stacked.
            :param src_arr: The array from which columns are chosen and calculations are made.
            :return: The final nested array after recursion stops.
            """
            #--- Termination condition for the recursive call 
            if n > src_arr.shape[1]:
                return stk_arr
            #--- Recursive call 
            # Select the nth column of y, subtract the sum of each row of x from each element, and ensure no negative values.
            chosen_col = src_arr[:, [n-1]]
            modified_col = chosen_col - stk_arr.sum(axis=1, keepdims=True)
            # Stack the modified column with x and call nested_arr recursively.
            modified_col[modified_col < 0] = 0
            return _nested_arr(n + 1, np.hstack((stk_arr, modified_col)), src_arr)
        
        def _si_fct_h(so_n, so_n1, inv, ds):
            """
            The optimized main function to calculate the SI_FCT_H using vectorized operations for array_calc.
            :param so_n: The initial supply order array.
            :param so_n1: The supply order array at time n+1.
            :param inv: The inventory array.
            :param ds: The demand array.
            :return: Transformed array after the NESTEDARR2 calculation and summing up the columns.
            """
            # Create a zero array with the same number of rows as so_n and one column.
            zero_array = np.zeros((so_n.shape[0], 1))
            # Calculate the nested array using a comprehension to replace MAKEARRAY function.
            array_calc = np.cumsum(so_n, axis=1)
            precalculated_y = ds * so_n1 - inv + array_calc
            nested_array = _nested_arr(1, zero_array, precalculated_y)
            # Drop the first column and sum each column.
            col_sums = np.sum(nested_array[:, 1:], axis=0)
            return col_sums

        # Create a date range which includes 'base_date' and the next 5 weeks (5 records in total)
        date_range=pd.date_range(base_date, periods=5, freq='W-MON') # Slice range definition
        #selected_period = self.df_br.loc[(slice(None), date_range), :] # Date range slice
        # Pivot the selected DataFrame including Mondays for a better view
        #so_n = np.array(selected_period.unstack(level='date')['demand'])
        #so_n1 = np.array(selected_period.unstack(level='date')['RR'])
        #ds = np.array(selected_period.unstack(level='date')['tgt_wos'])#

        # まず、selected_periodをインデックスでリセットして、dateを列として扱います
        numpy_arr = self.df_br.loc[(slice(None), date_range), ['demand','RR','tgt_wos']].values
        rows = int(len(numpy_arr[:,0])/5)

        # 必要な列を抽出し、ピボットテーブルを使用して転置します
        so_n = numpy_arr[:, 0].reshape(rows,5)
        so_n1 = numpy_arr[:,1].reshape(rows,5)
        ds = numpy_arr[:, 2].reshape(rows,5)

        ds = np.nan_to_num(ds.astype(float))

        # Get Prev week data
        #inv = np.array(self.df_br.xs(base_date, level='date')['prev_inv']).reshape(-1, 1)
        inv = np.array(self._cache_df_curr['prev_inv']).reshape(-1,1)
        # Calculate the result using the optimized si_fct_h function
        result_optimized = _si_fct_h(so_n, so_n1, inv, ds)    
        return result_optimized


    def GrossDemand_old(self, base_date):
        '''
        Issues/requirements to be solved:
        - Currently the future weeks to be considered is fixed to 5 weeks, dpending on the average range it should be flex
        - In future similar information will go to root psi (RR?)
        '''
        def _nested_arr(n, stk_arr, src_arr):
            """
            Recursively builds a nested array by stacking columns and performing calculations on them.
            :param n      : Current iteration number.
            :param stk_arr: The array to be horizontally stacked.
            :param src_arr: The array from which columns are chosen and calculations are made.
            :return: The final nested array after recursion stops.
            """
            #--- Termination condition for the recursive call 
            if n > src_arr.shape[1]:
                return stk_arr
            #--- Recursive call 
            # Select the nth column of y, subtract the sum of each row of x from each element, and ensure no negative values.
            chosen_col = src_arr[:, [n-1]]
            modified_col = chosen_col - stk_arr.sum(axis=1, keepdims=True)
            # Stack the modified column with x and call nested_arr recursively.
            modified_col[modified_col < 0] = 0
            return _nested_arr(n + 1, np.hstack((stk_arr, modified_col)), src_arr)
        
        def _si_fct_h(so_n, so_n1, inv, ds):
            """
            The optimized main function to calculate the SI_FCT_H using vectorized operations for array_calc.
            :param so_n: The initial supply order array.
            :param so_n1: The supply order array at time n+1.
            :param inv: The inventory array.
            :param ds: The demand array.
            :return: Transformed array after the NESTEDARR2 calculation and summing up the columns.
            """
            # Create a zero array with the same number of rows as so_n and one column.
            zero_array = np.zeros((so_n.shape[0], 1))
            # Calculate the nested array using a comprehension to replace MAKEARRAY function.
            array_calc = np.cumsum(so_n, axis=1)
            precalculated_y = ds * so_n1 - inv + array_calc
            nested_array = _nested_arr(1, zero_array, precalculated_y)
            # Drop the first column and sum each column.
            col_sums = np.sum(nested_array[:, 1:], axis=0)
            return col_sums

        # Create a date range which includes 'base_date' and the next 5 weeks (5 records in total)
        date_range=pd.date_range(base_date, periods=5, freq='W-MON') # Slice range definition
        selected_period = self.df_br.loc[(slice(None), date_range), :] # Date range slice
        # Pivot the selected DataFrame including Mondays for a better view
        so_n = np.array(selected_period.unstack(level='date')['demand'])
        so_n1 = np.array(selected_period.unstack(level='date')['RR'])
        ds = np.array(selected_period.unstack(level='date')['tgt_wos'])
        ds = np.nan_to_num(ds.astype(float))

        # Get Prev week data
        inv = np.array(self.df_br.xs(base_date, level='date')['prev_inv']).reshape(-1, 1)
        # Calculate the result using the optimized si_fct_h function
        result_optimized = _si_fct_h(so_n, so_n1, inv, ds)    
        return result_optimized


    def NetDemand(self, tgt_date): # SUGGESTED SELLIN
        '''
        Issues to be solved:
        - temporaly set root psi to to SGMC, however it should be auto-detected
        '''
        #tmr1 = time.time()
        tgt =  self.df_rt.loc[(self._rootname, tgt_date)]
        #sellable  =tgt['prev_inv'].astype(int) + tgt['CAP'].astype(int)
    #-----------------------------------------------
        #print(f'##1. slice df: {time.time()-#tmr1}')
        #tmr1 = time.time()

        #print(">>>> SELABLE: ", sellable)
        df_achvrate = self.AchvRate(tgt_date)
    #-----------------------------------------------
        #print(f'##2. Achv Rate: {time.time()-#tmr1}')
        #tmr1 = time.time()

        # if SCPSI ADJ (country) has a value, take the adjustment data.
        sellable  =tgt['prev_inv'] + tgt['CAP'] if np.isnan(tgt['ADJ']) or (tgt['ADJ'] < 0) else (tgt['ADJ']-df_achvrate['ADJ'].sum()).clip(min=0)
    #-----------------------------------------------
        #print(f'##3. Sellable Calc: {time.time()-#tmr1}')
        #tmr1 = time.time()

        # Provided function results converted to pandas Series
        fx_x_series = self.fx_x(df_achvrate)
        fx_x_series = pd.Series(fx_x_series)
    #-----------------------------------------------
        #print(f'##4. FX_X: {time.time()-#tmr1}')
        #tmr1 = time.time()

        #print("ACVRATE: ",df_achvrate)
        rate_per_unit_series = self.RatePerUnit_old(df_achvrate)
    #-----------------------------------------------
        #print(f'##5. Rate Per Unit: {time.time()-#tmr1}')
        #tmr1 = time.time()

        unit_ancher_series = self.unit_ancher(rate_per_unit_series,fx_x_series)

    #-----------------------------------------------
        #print(f'##6. UnitAncher: {time.time()-#tmr1}')
        #tmr1 = time.time()

        # Now we will calculate the result based on the provided formulas and sample data
        # Using .clip to replicate the MAX function in Excel
        sum_byrow_result = (1 - df_achvrate['ACHV_RATE']).clip(lower=0).sum()  if (tgt['ADJ'] < 0) or np.isnan(tgt['ADJ'])  else 999999

    #-----------------------------------------------
        #print(f'##7. Ttl Acvrate: {time.time()-#tmr1}')
        #tmr1 = time.time()

        #temp_for_print = df_achvrate.copy()
        #new_order = [
        #    '004.SONY STORE',
        #    '005.ECOMMERCE',
        #    '001.SALES 1',
        #    '002.SALES 2',
        #    '003.SALES 3',
        #    '007.CORP'
        #]
        # MultiIndexのレベル0にこの新しい順番を適用
        #temp_for_print = temp_for_print.reindex(new_order, level=0)
        #temp_for_print['ACHV_RATE'] = temp_for_print['ACHV_RATE'].apply(lambda val: f'{val:.1%}' )



        #print("####### TEMP FOR PRINT \n", temp_for_print)#.style.format({'ACHV_RATE':'{:.1%}'}))
        # Finding the index for sellable in unit_ancher_series
        # Since unit_ancher_series values are not exactly sellable, we will use the closest value to sellable (which is 0 for index 0)
        #print("####### UNIT ANCHER \n",type(unit_ancher_series),"\n", unit_ancher_series.head(20))
        #print("####### SELLABLE \n",sellable)
        '''
        print("######<>>>>>>>>>>>>>>","\n",
              unit_ancher_series,"\n",
              sellable,"\n",
              (unit_ancher_series - sellable).abs(),"\n", 
              type(unit_ancher_series ))
        '''
        
    #-----------------------------------------------
        #tmr2 = time.time()
        unit_ancher_index = max(0, (unit_ancher_series - sellable).abs().argmin() -1)
    #-----------------------------------------------
        #print(f'##8-1 {time.time()-#tmr2}')
        #tmr2 = time.time()

        # We use the .iloc method to get the value at the index equivalent to the Excel MATCH function
        min_result = min(sum_byrow_result, rate_per_unit_series.iloc[unit_ancher_index])
    #-----------------------------------------------
        #print(f'##8-2 {time.time()-#tmr2}')
        #tmr2 = time.time()
        # Calculate the final result with the provided formulas
        result = min(sum_byrow_result,(min_result * (sellable - unit_ancher_series.iloc[unit_ancher_index])) + fx_x_series.iloc[unit_ancher_index]) # TTL ACHVRATION
    #-----------------------------------------------
        #print(f'##8-3 {time.time()-#tmr2}')
        #tmr2 = time.time()
        #print("## ACVRATE TTL:",result)
        fillrate_ttl = self.fx_fill(df_achvrate,result)
    #-----------------------------------------------
        #print(f'##8-4 {time.time()-#tmr2}')
        #tmr2 = time.time()
        #print(">>>>> FILLRATE TOTAL: ", fillrate_ttl)
        #print("## FX_FILL: ",fx_fill(AchvRate(df,selected_date),result))

        fillrate = fillrate_ttl -df_achvrate['ACHV_RATE']
    #-----------------------------------------------
        #print(f'##8-5 {time.time()-#tmr2}')
        #tmr2 = time.time()
        #print(">>>>> FILLRATE: \n", fillrate)
    
    #-----------------------------------------------
        #print(f'##8. Calc Fillrate: {time.time()-#tmr1}')
        #tmr1 = time.time()

        result = ((fillrate * df_achvrate['tgt_wos'])/1*df_achvrate['RR']).clip(lower=0) + df_achvrate['ADJ'] # SUGGESTED Sell-in

    #-----------------------------------------------
        #print(f'##9. Calc Suggested Sellin: {time.time()-#tmr1}')
        #tmr1 = time.time()
        return result





    def NetDemand_new(self, tgt_date): # SUGGESTED SELLIN
        #tmr1 = time.time()
        '''
        Issues to be solved:
        - temporaly set root psi to to SGMC, however it should be auto-detected
        '''
        self.cache_AchvRate(tgt_date)
        self.cache_fx_y()
        self.cache_fx_r()
        self.cache_fx_x()
        
        #tgt =  self.df_rt.loc[(self._rootname, tgt_date)]
        tgt =  self._cache_df_curr_hi.iloc[0]

        #sellable  =tgt['prev_inv'].astype(int) + tgt['CAP'].astype(int)
    #-----------------------------------------------
        #print(f'##1. slice df: {time.time()-tmr1}')
        #tmr1 = time.time()

    #-----------------------------------------------
        #print(f'##2. Achv Rate: {time.time()-tmr1}')
        #tmr1 = time.time()

        # if SCPSI ADJ (country) has a value, take the adjustment data.
        sellable  = tgt['prev_inv'] + tgt['CAP'] if np.isnan(tgt['ADJ']) or (tgt['ADJ'] < 0) else (tgt['ADJ']-self._cache_achvrate['ADJ'].sum()).clip(min=0)
    #-----------------------------------------------
        #print(f'##3. Sellable Calc: {time.time()-tmr1}')
        #tmr1 = time.time()

        # Provided function results converted to pandas Series
        fx_x_series = pd.Series(self._cache_fx_x)
    #-----------------------------------------------
        #print(f'##4. FX_X: {time.time()-tmr1}')
        #tmr1 = time.time()

        rate_per_unit_series = self.RatePerUnit_new()
    #-----------------------------------------------
        #print(f'##5. Rate Per Unit: {time.time()-tmr1}')
        #tmr1 = time.time()

        unit_ancher_series = self.unit_ancher(rate_per_unit_series,fx_x_series)

    #-----------------------------------------------
        #print(f'##6. UnitAncher: {time.time()-tmr1}')
        #tmr1 = time.time()

        # Now we will calculate the result based on the provided formulas and sample data
        # Using .clip to replicate the MAX function in Excel
        sum_byrow_result = (1 - self._cache_achvrate['ACHV_RATE']).clip(lower=0).sum()  if (tgt['ADJ'] < 0) or np.isnan(tgt['ADJ'])  else 999999

    #-----------------------------------------------
        #print(f'##7. Ttl Acvrate: {time.time()-tmr1}')
        #tmr1 = time.time()

        #temp_for_print = df_achvrate.copy()
        #new_order = [
        #    '004.SONY STORE',
        #    '005.ECOMMERCE',
        #    '001.SALES 1',
        #    '002.SALES 2',
        #    '003.SALES 3',
        #    '007.CORP'
        #]
        # MultiIndexのレベル0にこの新しい順番を適用
        #temp_for_print = temp_for_print.reindex(new_order, level=0)
        #temp_for_print['ACHV_RATE'] = temp_for_print['ACHV_RATE'].apply(lambda val: f'{val:.1%}' )



        #print("####### TEMP FOR PRINT \n", temp_for_print)#.style.format({'ACHV_RATE':'{:.1%}'}))
        # Finding the index for sellable in unit_ancher_series
        # Since unit_ancher_series values are not exactly sellable, we will use the closest value to sellable (which is 0 for index 0)
        #print("####### UNIT ANCHER \n",type(unit_ancher_series),"\n", unit_ancher_series.head(20))
        #print("####### SELLABLE \n",sellable)
        '''
        print("######<>>>>>>>>>>>>>>","\n",
              unit_ancher_series,"\n",
              sellable,"\n",
              (unit_ancher_series - sellable).abs(),"\n", 
              type(unit_ancher_series ))
        '''
        
    #-----------------------------------------------
        #tmr2 = time.time()
        #unit_ancher_index = max(0, (unit_ancher_series - sellable).abs().argmin() )　#MATCH(,1)として完全に機能していない
        filtered_unit_ancher = unit_ancher_series[unit_ancher_series <= sellable]
        unit_ancher_index = max(0, filtered_unit_ancher.argmax() if not filtered_unit_ancher.empty else 0 )
    #-----------------------------------------------
        #print(f'##8-1: {time.time()-tmr2}')
        #tmr2 = time.time()

        # We use the .iloc method to get the value at the index equivalent to the Excel MATCH function
        min_result = min(sum_byrow_result, rate_per_unit_series.iloc[unit_ancher_index])
    #-----------------------------------------------
        #print(f'##8-2: {time.time()-tmr2}')
        #tmr2 = time.time()
        # Calculate the final result with the provided formulas
        result = min(sum_byrow_result,(min_result * (sellable - unit_ancher_series.iloc[unit_ancher_index])) + fx_x_series.iloc[unit_ancher_index]) # TTL ACHVRATION
    #-----------------------------------------------
        #print(f'##8-3: {time.time()-tmr2}')
        #tmr2 = time.time()
        #print("## ACVRATE TTL:",result)
        fillrate_ttl = self.fx_fill_new(result)
    #-----------------------------------------------
        #print(f'##8-4: {time.time()-tmr2}')
        #tmr2 = time.time()
        #print(">>>>> FILLRATE TOTAL: ", fillrate_ttl)
        #print("## FX_FILL: ",fx_fill(AchvRate(df,selected_date),result))

        fillrate = fillrate_ttl -self._cache_achvrate['ACHV_RATE']
    #-----------------------------------------------
        #print(f'##8-5: {time.time()-tmr2}')
        #tmr2 = time.time()
    
    #-----------------------------------------------
        #print(f'##8. Calc Fillrate: {time.time()-tmr1}')
        #tmr1 = time.time()

        result = ((fillrate * self._cache_achvrate['tgt_wos'])/1*self._cache_achvrate['RR']).clip(lower=0) + self._cache_achvrate['ADJ'] # SUGGESTED Sell-in

    #-----------------------------------------------
        #print(f'##9. Calc Suggested Sellin: {time.time()-tmr1}')
        #tmr1 = time.time()
        return result


    # 特定の日付の前の日付を返す関数を定義
    def get_previous_date(self, current_date):
        # dateレベルの値を取得
        dates = self.df_rt.index.get_level_values('date').unique()
        # 現在の日付の位置を取得
        current_date_index = dates.get_loc(current_date)
        # 現在の日付の前の日付を取得（存在する場合）
        if current_date_index > 0:
            previous_date = dates[current_date_index - 1]
        else:
            previous_date = None  # 最初の日付の場合はNoneを返す
        return previous_date

    def roll_psi_old(self, x, mute=True):   
        #=== Get date value (日付の抽出)
        curr_date = x.index.get_level_values('date').unique()[0]            # Current week date
        prev_date = self.get_previous_date(current_date=curr_date)          # Previous week date
        if not mute: 
            print("CURR DATE: ", curr_date, ", PREV DATE: ", prev_date)     # For debugging
        #=== Locate "current week" data from PSI dataframe
        df_curr=self.df_br.xs(curr_date, level='date')                      # Slice of current date from "PSI dataframe" 
        df_curr_hi = self.df_rt.xs(curr_date, level='date')

        #=== Locate "previous week" data from PSI dataframe:    XXXXXXXXXXXXXx   -> this not work because xs returns only a copy...
        if prev_date != None :
            df_prev=self.df_br.xs(prev_date, level='date')
            df_curr.loc[:, 'prev_inv'] = df_prev['I']                       # retreive inventory (I) from previous week and set to 'prev_inv' 
            df_prev_hi=self.df_rt.xs(prev_date, level='date')
            df_curr_hi.loc[:, 'prev_inv'] = df_prev_hi['I'].iloc[0]         # retreive inventory (I) from previous week and set to 'prev_inv' 
            prev_cap = df_prev_hi['CAP'].iloc[0]
            prev_inc = df_prev_hi['INCOMING'].iloc[0]
        else:
            df_curr.loc[:, 'prev_inv']=0
            df_curr_hi.loc[:, 'prev_inv']=0

        self.df_br['prev_inv'] = self.df_br.groupby(level=0)['I'].shift(1)  # Roll channel inventory (same as above, remove??
        ################ calculate currnt month CAP
        if df_curr_hi['FROZEN'].iloc[0] in ['Fix','CAP']:
            # Case current date reffers to a frozen period
            df_curr_hi['CAP'] = df_curr_hi['OSI']
        else:
            # Case current period is a free period ()
            if np.isnan(df_curr_hi['EOL_QTY'].iloc[0]) or df_curr_hi['EOL_QTY'].iloc[0]==0:
                if df_curr_hi['EOL'].iloc[0] =='EOL':
                    df_curr_hi['CAP'] = prev_cap-prev_inc                   #previos cap - previous incoming

                else:
                    df_curr_hi['CAP'] =   df_curr_hi['OSI'] * 1.2 + self._free_alloc # seihan rule here
            else:
                df_curr_hi['CAP'] = df_curr_hi['EOL_QTY']
        ################ Calculate Theoritical Incoming (Incoming calc as no dependency to channel PSI calculation, only refer natural demand)
        ltgt =  self.df_rt.loc[(self._rootname, curr_date)]
        gdmd = self.GrossDemand(curr_date)
        si_rr = gdmd[1:].mean()
        si_n = gdmd[0]
        CAP =  df_curr_hi['CAP'].iloc[0]
        #Calculation of theoritical "income"p
        df_curr_hi['INCOMING'] = min(CAP,max(0,si_rr*float(ltgt['std_ds'])/1+si_n-float(ltgt['prev_inv']))) if CAP > 0 else 0
        if  (df_curr_hi['FROZEN'] == 'Fix').any():
            df_curr_hi['INCOMING'] = df_curr_hi['OSI'] # If 'Fix' force set INCOMING


        df_curr_hi['date'] = curr_date # Dateの列を戻す
        df_curr_hui =df_curr_hi.set_index('date',append=True)
        #Writeback to original psi_c
        self.df_rt.loc[pd.IndexSlice[:,curr_date], :] = df_curr_hi
        #=== SELL-IN:  Suggested sell-in calculation
        sr_suggested_si = pd.DataFrame(self.NetDemand(curr_date),columns=['P']) #<< accessing to country psi here
        #print(sr_suggested_si)
        df_curr['P'] = sr_suggested_si 
        #=== SELL-OUT:

        if self.predecessor is None:
            df_curr['canib_loss'] = df_curr['demand']
        else:
            df_curr['canib_loss'] = (df_curr['demand'] - df_curr['pre_S']).fillna(0).clip(lower=0)
        
        df_curr['S'] = np.minimum(df_curr['prev_inv'] + df_curr['P'] ,df_curr['canib_loss'])

        #=== INVENTORY:
        # Calculate new PSI based on the split logic and set to df_curr
        df_curr['I'] = df_curr['prev_inv']+df_curr['P']-df_curr['S']   

        # Recover the original index structure (in order to write back to original DF)
        df_curr['date'] = curr_date # Dateの列を戻す
        df_curr =df_curr.set_index('date',append=True)

        #Writeback to original psi
        df_curr.fillna(value=0)
        self.df_br.loc[pd.IndexSlice[:,curr_date], :] = df_curr
        #print( self.df_br.loc[pd.IndexSlice[:,curr_date], :] )
        # Calculate Rolling OH inventort  #####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>...本当に２回Writebackする必要があるのか？CAP→I
        df_curr_hi['S'] =df_curr['P'].sum()   
        df_curr_hi['I'] = df_curr_hi['prev_inv']+df_curr_hi['INCOMING']-df_curr_hi['S']
        #Writeback to original psi_c
        #df_curr_hi.fillna(value=0)
        self.df_rt.loc[pd.IndexSlice[:,curr_date], :] = df_curr_hi

        return df_curr


    def roll_psi(self, _date, mute=True):   
        #=== Get date value (日付の抽出)
        curr_date = _date   # Current week date
        prev_date = self.get_previous_date(current_date=curr_date)              # Previous week date
        if not mute: 
            print("CURR DATE: ", curr_date, ", PREV DATE: ", prev_date) # For debugging
        #=== Locate "current week" data from PSI dataframe
        #self._cache_df_curr = self.df_br.loc[(slice(None), curr_date), :]
        #self._cache_df_curr_hi = self.df_rt.loc[(slice(None), curr_date), :]
        self._cache_df_curr = self.df_br.loc[(slice(None), slice(prev_date, curr_date)), :]
        self._cache_df_curr_hi = self.df_rt.loc[(slice(None), slice(prev_date, curr_date)),:]
       
        #=== Locate "previous week" data from PSI dataframe:    XXXXXXXXXXXXXx   -> this not work because xs returns only a copy...
        if prev_date != None :
            #self._cache_df_curr.loc[:, 'prev_inv'] = self.df_br.loc[(slice(None), slice(prev_date, curr_date)), ['I']].groupby(level=0).shift(1).loc[(slice(None), curr_date),'I']
            self._cache_df_curr.loc[:, 'prev_inv'] = self._cache_df_curr['I'].groupby(level=0).shift(1)

            #prev_cpsi = self.df_rt.loc[(slice(None), slice(prev_date, curr_date)), ['I', 'INCOMING', 'CAP']].groupby(level=0).shift(1)
            #self._cache_df_curr_hi.loc[:, 'prev_inv'] = prev_cpsi.loc[(slice(None), curr_date),'I'] # retreive inventory (I) from previous week and set to 'prev_inv' 

            self._cache_df_curr_hi.loc[:, 'prev_inv'] = self._cache_df_curr_hi['I'].iloc[0]

            #prev_cap = prev_cpsi.loc[(slice(None), curr_date),'CAP'] 
            #prev_inc = prev_cpsi['INCOMING'].values[0]

            prev_cap = self._cache_df_curr_hi['CAP'].iloc[0]
            prev_inc = self._cache_df_curr_hi['INCOMING'].iloc[0]
            '''
            prev_inv_df_br = self.df_br.loc[(slice(None), prev_date), 'I'].copy()
            self._cache_df_curr.loc[:, 'prev_inv'] = prev_inv_df_br
            prev_cpsi = self.df_rt.loc[(slice(None), prev_date), ['I', 'INCOMING', 'CAP']].copy()
            self._cache_df_curr_hi.loc[:, 'prev_inv'] = prev_cpsi['I']
            prev_cap = prev_cpsi['CAP'].values[0]
            prev_inc = prev_cpsi['INCOMING'].values[0]
            '''
        else:
            self._cache_df_curr.loc[:, 'prev_inv']=0
            self._cache_df_curr_hi.loc[:, 'prev_inv']=0


        self._cache_df_curr = self._cache_df_curr[self._cache_df_curr.index.get_level_values('date') == curr_date]
        self._cache_df_curr_hi = self._cache_df_curr_hi[self._cache_df_curr_hi.index.get_level_values('date') == curr_date]


        #self.df_br['prev_inv'] = self.df_br.groupby(level=0)['I'].shift(1)    # Roll channel inventory (same as above, remove??
        ################ calculate currnt month CAP
        if self._cache_df_curr_hi['FROZEN'].iloc[0] in ['Fix','CAP']:
            # Case current date reffers to a frozen period
            self._cache_df_curr_hi.loc[:,'CAP'] = self._cache_df_curr_hi['OSI']
        else:
            # Case current period is a free period ()
            if np.isnan(self._cache_df_curr_hi['EOL_QTY'].iloc[0]) or self._cache_df_curr_hi['EOL_QTY'].iloc[0]==0:
                if self._cache_df_curr_hi['EOL'].iloc[0] =='EOL':
                    self._cache_df_curr_hi.loc[:,'CAP'] = prev_cap-prev_inc  #previos cap - previous incoming
                else:
                    self._cache_df_curr_hi.loc[:,'CAP'] =   self._cache_df_curr_hi['OSI'] * 1.2 + self._free_alloc # seihan rule here
            else:
                self._cache_df_curr_hi.loc[:,'CAP'] = self._cache_df_curr_hi['EOL_QTY']

        ################ Calculate Theoritical Incoming (Incoming calc as no dependency to channel PSI calculation, only refer natural demand)
        gdmd = self.GrossDemand(curr_date)
        si_rr = gdmd[1:].mean()
        si_n = gdmd[0]
        CAP =  self._cache_df_curr_hi['CAP'].iloc[0]
        #Calculation of theoritical "income"p
        self._cache_df_curr_hi.loc[:,'INCOMING'] = min(CAP,max(0,si_rr*(self._cache_df_curr_hi['std_ds'].iloc[0])+si_n-(self._cache_df_curr_hi['prev_inv'].iloc[0]))) if CAP > 0 else 0
        if  (self._cache_df_curr_hi['FROZEN'] == 'Fix').any():
            self._cache_df_curr_hi.loc[:,'INCOMING'] = self._cache_df_curr_hi['OSI'] # If 'Fix' force set INCOMING
        #Writeback to original psi_c
        self.df_rt.update(self._cache_df_curr_hi, overwrite=True)
        #self.df_rt.loc[pd.IndexSlice[:,curr_date], :] = self._cache_df_curr_hi
        #=== SELL-IN:  Suggested sell-in calculation
        sr_suggested_si = pd.DataFrame(self.NetDemand_new(curr_date),columns=['P']) #<< accessing to country psi here
        #print(sr_suggested_si)
        self._cache_df_curr.loc[:,'P'] = sr_suggested_si 
        #=== SELL-OUT:
        if self.predecessor is None:
            self._cache_df_curr.loc[:,'canib_loss'] = self._cache_df_curr['demand']
        else:
            self._cache_df_curr.loc[:,'canib_loss'] = (self._cache_df_curr['demand'] - self._cache_df_curr['pre_S']).fillna(0).clip(lower=0)
        
        self._cache_df_curr.loc[:,'S'] = np.minimum(self._cache_df_curr['prev_inv'] + self._cache_df_curr['P'] ,self._cache_df_curr['canib_loss'])

        #=== INVENTORY:
        # Calculate new PSI based on the split logic and set to self._cache_df_curr
        self._cache_df_curr.loc[:,'I'] = self._cache_df_curr['prev_inv']+self._cache_df_curr['P']-self._cache_df_curr['S']   

        #Writeback to original psi
        self._cache_df_curr.fillna(value=0)

        #self.df_br.loc[pd.IndexSlice[:,curr_date], :] = self._cache_df_curr
        self.df_br.update(self._cache_df_curr, overwrite=True)
        self._cache_df_curr_hi.loc[:,'S'] =self._cache_df_curr['P'].sum()   
        self._cache_df_curr_hi.loc[:,'I'] = self._cache_df_curr_hi['prev_inv']+self._cache_df_curr_hi['INCOMING']-self._cache_df_curr_hi['S']
        
        #Writeback to original psi_c
        self.df_rt.update(self._cache_df_curr_hi, overwrite=True)
        #self.df_rt.loc[pd.IndexSlice[:,curr_date], :] = self._cache_df_curr_hi


    @staticmethod
    def PrintPSI(d, itm = ['P','S','I']):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        pd.options.display.float_format = '{:.1f}'.format
        # dwクラスのインスタンスを作成
        print("PSI Template generation ==========================================")
        print(d.loc[:,itm].stack().unstack(level=1))
        d.reorder_levels([1, 0])

    def print(self, **kwargs):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.options.display.float_format = '{:.1f}'.format
        # Declare which psi to show, target='rt': df_rt,  target='br': df_(default)
        if  'target' in kwargs:
            kwargs['target'] = 'rt' if kwargs['target'] == 'rt' else 'br'
        else:
            kwargs['target'] = 'br'
        df = self.df_rt.copy() if kwargs['target'] == 'rt'  else self.df_br.copy()

        # Specify item to be shown (default P S I  or INC S I )
        item = ['P','S','I'] if kwargs['target'] == 'br' else ['INCOMING','S', 'I']
        if 'item' in kwargs:
            item = kwargs['item'] if isinstance(kwargs['item'],list) else [kwargs['item']]
        
        
        
        #df = df[df['month'].isin(['DEC','JAN'])] # Restrict month (to be implemented)
        if  'n_weeks' in kwargs:
            act_date = pd.Timestamp(self.act_date) - pd.DateOffset(weeks=1)
            end_date = act_date + pd.DateOffset(weeks=kwargs['n_weeks'])
        else:
            act_date = pd.Timestamp(self.act_date) - pd.DateOffset(weeks=1)
            end_date = act_date + pd.DateOffset(weeks=18)
            #df = df[(df['date'] >= self.act_date ) & 
            #        (df['date'] <= self.act_date + timedelta(weeks=kwargs['n_weeks']))  ]    
        df = df[(df.index.get_level_values(1) >= act_date) & 
                (df.index.get_level_values(1) <= end_date)]
        

        #Replace date index
        if  'time_disp' in kwargs:
            if kwargs['time_disp'] == "date":
                df = df[item].stack().unstack(level=[1])
        else:
            df['week'] = df['week'].apply(lambda x: f"WK{x:02}")
            if df.index.duplicated().any():
                # Show warning and duplications
                print("WARNING: There are duplicated indexes (date index)in current dataframe: ")
                duplicate_indices = df.index[df.index.duplicated(keep=False)]
                unique_duplicates = set(duplicate_indices)
                print(unique_duplicates)
                # this is not processed (need to be checked in df_br/df_rt)

            df = df.set_index(['year', 'week'], append=True)
            df = df.droplevel(1)
            if df.index.duplicated().any():
                # Show warning and duplications
                print("WARNING: There are duplicated indexes (year/week columns)in current dataframe:")
                duplicate_indices = df.index[df.index.duplicated(keep=False)]
                unique_duplicates = set(duplicate_indices)
                print(unique_duplicates)
                # Handle error only for display
                df = df[~df.index.duplicated(keep='first')]


            #重複したYearとWeekのコンビネーションがあるとエラーが出る（要対応）：ValueError: Index contains duplicate entries, cannot reshape
            df = df[item].stack().unstack(level=[1, 2])

        if kwargs['target'] == 'br':
            #account order 
            if 'sort_order' in kwargs:
                index0_list = df.index.get_level_values(0).unique().tolist()
                sort_order  = list(filter(lambda item: item  in index0_list,kwargs['sort_order'])) +  list(filter(lambda item: item not in kwargs['sort_order'], index0_list))
                #df.index = df.index.set_levels(pd.Categorical(df.index.levels[0], categories=sort_order, ordered=True), level=0)
                # 一時的なカラムを追加（ソート順を反映）
                df['sort_order'] = pd.Categorical(df.index.get_level_values(0), categories=sort_order, ordered=True)
                df = df.sort_values('sort_order')# 一時的なカラムでソート
                df = df.drop('sort_order', axis=1)# 一時的なカラムを削除

            # dertermine index display mode: Item-account or Account-item
            if  'reverse' in kwargs:
                kwargs['reverse'] = True if kwargs['reverse'] == True else False
            else:
                kwargs['reverse'] = False

            if kwargs['reverse']:
                df = df.reorder_levels([1, 0]).reindex(item, level=0)

        df = df[(~df.index.get_level_values(0).str.contains('_DUMMY_'))]

        #print(df)
        return df


    @staticmethod
    def PrintPSI2(d, itm = ['P','S','I']):
        pd.set_option('display.max_columns', 52)
        pd.options.display.float_format = '{:.1f}'.format
        # dwクラスのインスタンスを作成
        print("PSI Template generation ==========================================")
        
        print(d.loc[:,itm].stack().unstack(level=1).reorder_levels([1, 0]).reindex(itm, level=0))



    def ExecRollingCalc(self, progress_callback=None):
        pred_start_date = pd.Timestamp(self.act_date + timedelta(weeks=1))
        pred_end_date = pd.Timestamp(self.df_br.index.get_level_values('date')[-5])
        date_range = pd.date_range(pred_start_date, pred_end_date, freq='W-MON')


        self.calc_RR()
        # print first row and dtype

        self.df_br['demand'] = self.df_br['demand'].astype(float)
        self.df_br['io_delta'] = round(self.df_br['pre_PI'] - self.df_br['demand'] + self.df_br['ADJ'],4)
        self.df_br['tgt_stock'] = self.df_br['RR'] * self.df_br['tgt_wos']
        if self.predecessor is not None:
            self.df_br['pre_S'] = self.predecessor.df_br['S'].fillna(0)
            self.df_br['pre_PI'] = self.predecessor.df_br['prev_inv'].fillna(0) + self.predecessor.df_br['P'].fillna(0)

        self.df_br.sort_index(inplace=True)

        #.loc[(slice(None),act_date)
        # "!!!issue when multi country branch!!!!
        self.df_rt.loc[pd.IndexSlice[:, pred_start_date:],'OSI']= self.df_rt.loc[pd.IndexSlice[:, pred_start_date:],'OSI'].shift(1).fillna(0)
        self.df_rt.loc[pd.IndexSlice[:, pred_start_date:],'FROZEN']= self.df_rt.loc[pd.IndexSlice[:, pred_start_date:],'FROZEN'].shift(1).fillna('Fix')
        #self.df_rt['FROZEN']=self.df_rt['FROZEN'].shift(1)
        for date in date_range:
            #row = self.df_br.xs(date, level='date')
            self.roll_psi(date)
            #results.append(result)
            if progress_callback:
                progress_callback(100 * (date - pred_start_date).days / (pred_end_date - pred_start_date).days)
        self.df_rt.loc[pd.IndexSlice[:, pred_start_date:],'OSI']= self.df_rt.loc[pd.IndexSlice[:, pred_start_date:],'OSI'].shift(-1).fillna(0)
        self.df_rt.loc[pd.IndexSlice[:, pred_start_date:],'FROZEN']= self.df_rt.loc[pd.IndexSlice[:, pred_start_date:],'FROZEN'].shift(-1)
        # Optional: Update df_br with results if needed
        #self.df_br.update(pd.concat(results))

        #return pd.concat(results)


    def ExecRollingCalc2(self, progress_callback=None):

        pred_start_date = self.act_date +timedelta(weeks=1) # 予測開始日
        pred_end_date = self.df_br.index.get_level_values(1)[-5]     # 
        #pred_end_date = self.df_br.index.get_level_values(1)[-40]     # 
        date_range = pd.date_range(pred_start_date, pred_end_date, freq='W-MON')
        percentage_range = np.linspace(0, 100, len(date_range))
        # 週単位の日付リストをKeyとした辞書を作成
        date_dict  = dict(zip(date_range, percentage_range))


        self.calc_RR()
        if self.predecessor is not None:
            self.df_br['pre_S'] = self.predecessor.df_br['S'].fillna(0)  
            #self.df_br['canib_loss'] = (self.df_br['demand'] - self.df_br['pre_S']).fillna(0).clip(lower=0)
            self.df_br['pre_PI'] = self.predecessor.df_br['prev_inv'].fillna(0) + self.predecessor.df_br['P'].fillna(0) 

        # マルチインデックスをソート
        self.df_br.sort_index(inplace=True)
        test = self.df_br.loc[pd.IndexSlice[:, pred_start_date:pred_end_date], :].groupby(level='date').apply(lambda row: (
                                                                                                                        self.roll_psi_old(row), 
                                                                                                                        progress_callback(date_dict[row.index.get_level_values(1)[0]])))#argsはタプル指定
        #test = self.df_br.loc[pd.IndexSlice[:, '2023-10-30':'2024-09-09'], :].groupby(level='date').apply(lambda row: self.roll_psi(row))#argsはタプル指定]


        
        #print('Exec Rolling Calc: Completed ')
'''

from datetime import datetime
import csv
#key=date, value=[wk, cy, mth, fy]
def csv_to_dict(csv_file_path):
    result = {}
    
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        t=1
        for row in csv_reader:
            if t >2:#=365*3 and t < 365*5:
                key = datetime.strptime(row[0],'%Y/%m/%d')
                values = row[1:5]
                result[key] = values
            t+=1
    
    return result

'''

import datetime
 
def get_week_and_month(date_obj):
    # Convert the date string to a datetime object
    #date_obj = datetime.datetime.strptime(date_str, '%Y/%m/%d')
   
    # Get the Monday date of the week
    monday_date = date_obj - datetime.timedelta(days=date_obj.weekday())
   
    # Get the week number
    week_number = monday_date.isocalendar()[1]
   
    # Determine the month based on the majority of days in the week
    count = sum(1 for day in range(7) if (monday_date + datetime.timedelta(days=day)).month == monday_date.month)
    if count >= 4:
        month_name = monday_date.strftime("%b").upper()
        month_number = monday_date.month
        CY = monday_date.year
        FY = CY if monday_date.month >= 4 else CY - 1
    else:
        next_week = monday_date + datetime.timedelta(days=7)
        month_name = next_week.strftime("%b").upper()
        month_number = next_week.month
        CY = next_week.year
        FY = CY if next_week.month >= 4 else CY - 1
   
    return FY, CY, week_number, month_name#, month_number




class PSIContainer:
    def __init__(self, df = None, df_c = None, start_date = None, fctlen = 52, _nsmp_list = None, _account_map = None, _product_map= None, scenario_key ='LIVE', fct_type= 'DEMAND', _act_week_si = 200001, _act_week_so = 200001):

        self.psi ={}
        self.fsmc = 'TEMP'
        if (df is None) and (df_c is None):
            self.account_list = []
            self.product_list = []
            
            #--- ACCOUNT LIST COLLECTION
            if _account_map is None:
                # TBD: case user do not provide info -> create a duumy list
                account_map = None
                self.account_list = ['Account A', 'Account B', 'Account C'] # dummy data
                self.country = 'COUNTRY A'

            elif isinstance(_account_map, list):
                # TBD: case user provided a list -> create a dummy mapping table
                self.account_list = _account_map

            elif isinstance(_account_map, pd.DataFrame):
                # case user provide a dataframe (regular pattern)
                self.account_map = _account_map.drop_duplicates().reset_index()
                self.account_list = self.account_map['FCT_ACCOUNT'].unique().tolist()
                self.country = self.account_map['COUNTRY'].unique()[0]
                self.fsmc = self.account_map['SALES_COMPANY'].unique()[0]
            
            #--- PRODUCT LIST COLLECTION
            if _product_map is None:
                # TBD: case user do not provide info -> create a duumy list
                account_map = None
                self.product_map = pd.DataFrame({'CATEGORY_NAME':['CATEGORY 1','CATEGORY 1','CATEGORY 1'],
                                                'SUB_CATEGORY':['OTHERS','OTHERS','OTHERS'],
                                                'SEGMENT_2':['OTHERS','OTHERS','OTHERS'],
                                                'KATABAN':['OTHERS','OTHERS','OTHERS'],
                                                'MODEL_CODE':['10000001','10000002','10000003'],
                                                'MODEL_NAME':['Product 1','Product 2','Product 3'],})
                self.product_list = self.product_map['MODEL_CODE'].unique().tolist()


            elif isinstance(_product_map, list):
                # TBD: case user provided a list -> create a dummy mapping table
                self.product_list = _product_map

            elif isinstance(_product_map, pd.DataFrame):
                # case user provide a dataframe (regular pattern)
                self.product_map = _product_map[['MODEL_CODE','MODEL_NAME','CATEGORY_NAME','SUB_CATEGORY','KATABAN','SEGMENT_2','PREDECESSOR','SUCCESSOR']].drop_duplicates().reset_index()
                self.product_list = _product_map.loc[_product_map['FCT_MODEL']==1]['MODEL_CODE'].unique().tolist()


            # Replace None (blank values) to avoid indexing error
            self.product_map[['CATEGORY_NAME', 'SUB_CATEGORY']] = self.product_map[['CATEGORY_NAME', 'SUB_CATEGORY']].fillna('OTHERS')
            self.product_map['MODEL_NAME'] = np.where(self.product_map['MODEL_NAME'].isnull(), self.product_map['MODEL_CODE'], self.product_map['MODEL_NAME'])
            self.product_map['SEGMENT_2'] = np.where(self.product_map['SEGMENT_2'].isnull(), self.product_map['MODEL_CODE'], self.product_map['SEGMENT_2'])
            self.product_map['KATABAN'] = np.where(self.product_map['KATABAN'].isnull(), self.product_map['MODEL_CODE'], self.product_map['KATABAN'])

            self.scenario = scenario_key

            self.act_week_si = _act_week_si
            self.act_week_so = _act_week_so
            self.act_date = (date_from_cy_wk(_act_week_si//100,_act_week_si%100))#df['ACT_WEEK'].unique()[0]
            #self.monthly_flash = 0
            self.forecast_type = fct_type
            
            self.nsmp_list = None if _nsmp_list is None else _nsmp_list.drop_duplicates()
            self.seihan_rule = True if "FINANCIAL" in self.scenario else False
            
            # if start_date is not available, set current year first monday date
            start_date = pd.to_datetime(start_date) if start_date is not None else pd.to_datetime(datetime.date(datetime.datetime.now().year, 1, ((7 - datetime.date(datetime.datetime.now().year, 1, 1).weekday()) % 7) + 1)) 
            end_date = start_date + pd.Timedelta(weeks=fctlen - 1)


            for product_code in self.product_list: #df['Model Name'].unique():
                #start_date, num_weeks, br_list=['Branch A'], rt_name = 'Root', predecessor = None, product_code=None
                self.psi[product_code] = PSI.from_weeks(start_date, fctlen, self.account_list, self.country)


        else:
            #wkmap = csv_to_dict('WKMAP.csv')
            # df validation (check if all fields exists)
            # To be developed
            # Get product / account list
            # Model list (Model with predecessors are sorted to be at the bottom)
            self.product_list = df[['MODEL_CODE', 'PREDECESSOR', 'SUCCESSOR']].drop_duplicates().sort_values(by=['PREDECESSOR', 'MODEL_CODE'], ascending=[True, True], na_position='first')['MODEL_CODE'].to_list()
            account_list = df['FCT_ACCOUNT'].unique()
            self.scenario = df['SCENARIO'].unique()[0] + '-' + df['FCT_TYPE'].unique()[0]
            #self.act_week_si = df['ACT_WEEK_SI'].unique()[0]
            #self.act_week_so = df['ACT_WEEK_SO'].unique()[0]
            #self.act_date = date_from_cy_wk(df['ACT_WEEK_SI'].unique()[0]//100,df['ACT_WEEK_SI'].unique()[0]%100)#df['ACT_WEEK'].unique()[0]

            self.act_week_si = _act_week_si
            self.act_week_so = _act_week_so
            self.act_date = date_from_cy_wk(_act_week_si//100,_act_week_si%100)

            #self.monthly_flash = df['MONTHLY_FLASH'].unique()[0]
            self.forecast_type = df['FCT_TYPE'].unique()[0]

            self.product_map = df[['MODEL_CODE','MODEL_NAME','CATEGORY_NAME','SUB_CATEGORY','KATABAN','SEGMENT_2','PREDECESSOR','SUCCESSOR']].drop_duplicates().reset_index()
            #self.account_map = df[[ 'ACCOUNT_GROUP', 'FCT_ACCOUNT']].drop_duplicates().reset_index()
            self.account_map = _account_map.drop_duplicates().reset_index()
            self.country = df['COUNTRY'].unique()[0]

            
            self.nsmp_list = None if _nsmp_list is None else _nsmp_list.drop_duplicates()

            # Replace None (blank values) to avoid indexing error
            self.product_map[['CATEGORY_NAME', 'SUB_CATEGORY']] = self.product_map[['CATEGORY_NAME', 'SUB_CATEGORY']].fillna('OTHERS')
            self.product_map['MODEL_NAME'] = np.where(self.product_map['MODEL_NAME'].isnull(), self.product_map['MODEL_CODE'], self.product_map['MODEL_NAME'])
            self.product_map['SEGMENT_2'] = np.where(self.product_map['SEGMENT_2'].isnull(), self.product_map['MODEL_CODE'], self.product_map['SEGMENT_2'])
            self.product_map['KATABAN'] = np.where(self.product_map['KATABAN'].isnull(), self.product_map['MODEL_CODE'], self.product_map['KATABAN'])

            self.seihan_rule = True if "FINANCIAL" in self.scenario else False

            
            # 最大日付（Max Date）を計算
            max_date = df['WEEK_DATE'].max()
            # 最小日付（Min Date）を計算
            min_date = df['WEEK_DATE'].min()

            # preprocess_data関数の定義（dateの計算は除外）
            def preprocess_data(df):
                new_df = pd.DataFrame({
                    'store': df['FCT_ACCOUNT'],
                    'date': df['WEEK_DATE'],
                    'cy': df['SEIHAN_YEAR'].astype(int),
                    'fy': df['SEIHAN_FISCAL_YEAR'].astype(int),
                    'wk': df['SEIHAN_WEEK'].astype(int),
                    'P': df['SELLIN_QTY'],
                    'S': df['SELLOUT_QTY'],
                    'demand': df['NATURAL_DEMAND'],
                    'I': df['CH_STOCK_OH'],
                    'O': df['SELLOUT_QTY'],
                    'std_ds': df['WOS_TARGET'],
                    'min_stock': df['MINSTOCK_TARGET'],
                    'prev_inv': np.nan,
                    'RR': 0,
                    'ADJ': df['SELL_IN_ADJ'],
                    'year': df['SEIHAN_YEAR'].astype(int),
                    'week': df['SEIHAN_WEEK'].astype(int),
                    'month':df['SEIHAN_MONTH_NAME']
                })
                new_df.set_index(['store', 'date'], inplace=True)
                return new_df

            # preprocess_data関数の定義（dateの計算は除外）
            def preprocess_data_c(df):
                new_df = pd.DataFrame({
                    'node': df['COUNTRY'],
                    'date': df['WEEK_DATE'],
                    'EOL': df['EOL'],
                    'EOL_QTY': df['EOL_QTY'],
                    'FROZEN': df['FROZEN'],
                    'CAP': df['CAP'],
                    'OSI': df['CAP'],
                    'INCOMING': df['PO_ETA_QTY'],
                    'I': df['SC_STOCK_OH'],
                    'S': df['SELLIN_QTY'],
                    'std_ds': df['WOS_TARGET'],
                    'prev_inv': np.nan,
                    'ADJ': df['SELL_IN_ADJ'],
                    'year': df['SEIHAN_YEAR'].astype(int),
                    'week': df['SEIHAN_WEEK'].astype(int),
                    'month':df['SEIHAN_MONTH_NAME']
                })
                new_df.set_index(['node', 'date'], inplace=True)
                return new_df

            
            predecessor_dict = pd.Series(self.product_map.PREDECESSOR.values,index=self.product_map.MODEL_CODE).to_dict()

            # ***** ACCOOUNT FOR DEBUGGING PURPOSE *********
            '''
            account_list = ['SONY STORE', 'ECOMMERCE', 'BEST DENKI', 'COURTS', 'GAIN CITY', 'HARVEY NORMAN', 'MUSTAFA', 'WD', 'ALAN PHOTO', 'BALLY PHOTO','CATHAY PHOTO', 
                        'MAX PHOTO', 'MS COLOR', 'V3', 'DISTY SALES2', 'T K FOTO TECHNIC', 'LUCKY STORE', 'PARISILK', 'SPRINT CASS', 'CORPORATE', 'OTHERS',
                        '_DUMMY_1', '_DUMMY_2', '_DUMMY_3', '_DUMMY_4', '_DUMMY_5', '_DUMMY_6', '_DUMMY_7', '_DUMMY_8', '_DUMMY_9']
         '''

            for product_code in self.product_list: #df['Model Name'].unique():
                try:
                    pred_model = None
                    if product_code in predecessor_dict.keys() and predecessor_dict[product_code] in self.psi.keys():
                        #pred_model = predecessor_dict[product_code] if product_code in predecessor_dict.keys() else None 
                        pred_model = self.psi[predecessor_dict[product_code]] # set pointer of predecessor PSI class

                    product_df = df[df['MODEL_CODE'] == product_code]
                    product_df_c = df_c[df_c['MODEL_CODE'] == product_code]
                    processed_df = preprocess_data(product_df)
                    # Create PSI
                    self.psi[product_code] = PSI.from_weeks(min_date, fctlen, account_list,self.country, predecessor = pred_model, product_code=product_code)
                    #self.psi[product_code].df_br['std_ds'] = 4 # Temporary current db is NULL
                    self.psi[product_code].act_date = self.act_date
                    
                    # Fill PSI-D
                    processed_df_c = preprocess_data_c(product_df_c)

                    self.psi[product_code].update(_df_br=processed_df, _df_rt=processed_df_c, _calc_rr = True)

                    # Canibloss  calculation -> moved to class 
                    results = self.psi[product_code].df_br.index.get_level_values('date').map(lambda x: get_week_and_month(x))
                    self.psi[product_code].df_br[['fy', 'year', 'week', 'month']] = pd.DataFrame(results.tolist(), index=self.psi[product_code].df_br.index)

                    results = self.psi[product_code].df_rt.index.get_level_values('date').map(lambda x: get_week_and_month(x))
                    self.psi[product_code].df_rt[['fy', 'year', 'week', 'month']] = pd.DataFrame(results.tolist(), index=self.psi[product_code].df_rt.index)

                    self.psi[product_code]._free_alloc = 0 if self.seihan_rule else 999999
                except Exception as e:
                    # すべての例外をキャッチして処理
                    print(f"Error creating {product_code} PSI: {e}")
                    print(product_df_c)
        a=1



    def save(self, filename):
        """インスタンスをファイルに保存するメソッド"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"データを{filename}に保存しました。")

    @staticmethod
    def load(filename):
        """ファイルからインスタンスを読み込む静的メソッド"""
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def batchCalculatePSI(self, model_code=None, callback=None ):
        # loop calculation (serial calculation)
        #bar = tqdm(total = 100, ncols=100)

        current_product = None
        def pr(x):
            #print(x)
            bar.update(x - bar.n )
            #print(current_product)
            #if callback is not None:
            #    callback(prog=x, prod=current_product)


        # Bulk calculation (all models)
        if model_code is None:
            for product in self.product_list:
                current_product = product           
                bar = tqdm(total = 100, ncols=100) 
                bar.set_description(f'MODEL PROGRESS:{product}' )
                self.psi[product].ExecRollingCalc(pr)# if callback is None else callback)
                bar.close()
        # Case model list is received
        elif isinstance(model_code, list) and set(model_code).issubset(set(self.product_list)):    
            for product in model_code:
                self.psi[product].ExecRollingCalc(pr)
        # case single model (string) is received
        elif isinstance(model_code, str) and model_code in self.product_list:
            self.psi[model_code].ExecRollingCalc(pr if callback is None else callback)
        else:
            print(f'ERROR: {model_code} not exists in the product list')

    def transform_all_dataframes(self):
        # df_prodmapを使用して多次元Keyを作成
        keys = []
        for modelcode in self.psi.keys():
            model_info = self.product_map.set_index('MODEL_CODE').loc[modelcode]
            key = (model_info['CATEGORY_NAME'] , model_info['SUB_CATEGORY'], model_info['KATABAN'], model_info['MODEL_NAME'], modelcode)
            keys.append(key)

        # 辞書内のすべてのDataFrameを結合
        concatenated_df = pd.concat([p.df_br for p in self.psi.values()], keys=keys, names=['CATEGORY_NAME', 'SUB_CATEGORY', 'KATABAN', 'MODEL_NAME', 'MODEL_CODE'])
        
        # インデックスをカラムに変換
        transformed_df = concatenated_df.reset_index()



        splitted = (self.scenario + '-na').split("-") # -na is to avoid errors when scenario or eventname is not available (secure minimun split qty)
        # 変換前のテーブルにないフィールドの生成
        transformed_df['SCENARIO'] = splitted[0]
        transformed_df['FCT_TYPE'] = splitted[1]
        transformed_df['SCENARIO_EVENT'] = self.scenario

        # 直接対応するフィールドのマッピング
        transformed_df.rename(columns={
            'node': 'FCT_ACCOUNT',
            'P': 'SELLIN_QTY',
            'S': 'SELLOUT_QTY',
            'I': 'CH_STOCK_OH',
            'week': 'SEIHAN_WEEK',
            'month': 'SEIHAN_MONTH_NAME',
            'year': 'SEIHAN_YEAR',
            'fy': 'SEIHAN_FISCAL_YEAR'
        }, inplace=True)

        # カラムの順序を整理
        column_order = ['CATEGORY_NAME', 'SUB_CATEGORY', 'KATABAN', 'MODEL_NAME', 'MODEL_CODE', 'FCT_ACCOUNT', 
                        'SEIHAN_YEAR', 'SEIHAN_FISCAL_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK', 
                        'SELLIN_QTY', 'SELLOUT_QTY','CH_STOCK_OH','SCENARIO', 'FCT_TYPE', 'SCENARIO_EVENT']
        transformed_df = transformed_df[column_order]

        # 1. Copy SELLIN_QTY to a new column SELLIN_QTY_ORI
        transformed_df['SELLIN_QTY_ORI'] = transformed_df['SELLIN_QTY']

        # 2. Copy SELLOUT_QTY to SELLIN_QTY for specified FCT_ACCOUNT values
        mask = transformed_df['FCT_ACCOUNT'].isin(['SONY STORE', 'ECOMMERCE'])
        transformed_df.loc[mask, 'SELLIN_QTY'] = transformed_df.loc[mask, 'SELLOUT_QTY']
        transformed_df.loc[mask, 'CH_STOCK_OH'] = 0

        # 3. AMOUNT CALCULATION
        transformed_df = transformed_df.merge(self.nsmp_list,how='left')
        transformed_df['NSP_LC'] = transformed_df['NSP_LC'].fillna(0)
        transformed_df['MP_COST_LC'] = transformed_df['MP_COST_LC'].fillna(0)
        transformed_df['SELLIN_QTY'] = transformed_df['SELLIN_QTY'].astype(float)
        transformed_df['NET_SALES_LC']=transformed_df['SELLIN_QTY'].round(0) * transformed_df['NSP_LC']
        transformed_df['MP_AMOUNT_LC']=transformed_df['SELLIN_QTY'].round(0) * (transformed_df['NSP_LC'] - transformed_df['MP_COST_LC'])
        transformed_df.drop(columns=['NSP_LC','MP_COST_LC' ], inplace=True)

        #4. ADD ACCOUT_GROUP
        transformed_df = transformed_df.merge(self.account_map[[ 'ACCOUNT_GROUP', 'FCT_ACCOUNT']].drop_duplicates(), on='FCT_ACCOUNT', how='left')
        transformed_df['ACCOUNT_GROUP'] = transformed_df['ACCOUNT_GROUP'].fillna('OTHERS')
        transformed_df['FCT_ACCOUNT'] = transformed_df['FCT_ACCOUNT'].fillna('OTHERS')

        return transformed_df


    def get_rt_dataframe(self):
        # df_prodmapを使用して多次元Keyを作成
        keys = []
        for modelcode in self.psi.keys():
            model_info = self.product_map.set_index('MODEL_CODE').loc[modelcode]
            key = (model_info['CATEGORY_NAME'] , model_info['SUB_CATEGORY'], model_info['KATABAN'], model_info['MODEL_NAME'], modelcode)
            keys.append(key)
        # 辞書内のすべてのDataFrameを結合
        concatenated_df_c = pd.concat([p.df_rt for p in self.psi.values()], keys=keys, names=['CATEGORY_NAME', 'SUB_CATEGORY', 'KATABAN', 'MODEL_NAME', 'MODEL_CODE'])
        transformed_df_c = concatenated_df_c.reset_index()
        splitted = (self.scenario + '-na').split("-") # -na is to avoid errors when scenario or eventname is not available (secure minimun split qty)
        transformed_df_c['SCENARIO'] = splitted[0]
        transformed_df_c['FCT_TYPE'] = splitted[1]
        transformed_df_c.rename(columns={
            'date': 'WEEK_DATE',
            'FROZEN':   'FROZEN',
            'CAP':      'PO_ETA_CAP',
            'OSI':      'PO_ETA_OSI', 
            'INCOMING': 'PO_ETA_QTY', 
            'S':        'SELLIN_QTY', 
            'I':        'SC_STOCK_OH',
            'std_ds':   'WOS_TARGET',
            'prev_inv': 'NA',
            'RR':       'NA',
            'ADJ':      'SELL_IN_ADJ',
            'EOL':      'EOL',
            'EOL_QTY':  'EOL_QTY',
            'demand':   'NATURAL_DEMAND',
            'week':     'SEIHAN_WEEK',
            'month':    'SEIHAN_MONTH_NAME',
            'year':     'SEIHAN_YEAR',
            'fy':       'SEIHAN_FISCAL_YEAR'
        }, inplace=True)
        transformed_df_c['COUNTRY'] = self.country
        transformed_df_c['SELLOUT_QTY'] = 0
        transformed_df_c['CH_STOCK_OH'] = 0
        column_order_c =['SCENARIO','FCT_TYPE','CATEGORY_NAME', 'SUB_CATEGORY', 'KATABAN', 'MODEL_NAME', 'MODEL_CODE','COUNTRY','SEIHAN_YEAR','SEIHAN_FISCAL_YEAR','SEIHAN_MONTH_NAME','SEIHAN_WEEK','WEEK_DATE','PO_ETA_OSI','PO_ETA_CAP','PO_ETA_QTY','SC_STOCK_OH','SELLIN_QTY','SELLOUT_QTY','CH_STOCK_OH','WOS_TARGET', 'SELL_IN_ADJ','EOL_QTY', 'EOL', 'FROZEN' ]
        sc_psi_fct =transformed_df_c[column_order_c].copy()
        sc_psi_fct['WEEK_DATE'] = pd.to_datetime(sc_psi_fct['WEEK_DATE']).dt.date
        return sc_psi_fct

        #'WOS_TARGET', 'SELL_IN_ADJ', 'CAP', 'EOL_QTY', 'EOL', 'FROZEN'

    def get_writeback_df(self):
        db_data = get_db_data()
        # df_prodmapを使用して多次元Keyを作成
        keys = list(self.psi.keys())
        #for modelcode in self.psi.keys():
            #model_info = self.product_map.set_index('MODEL_CODE').loc[modelcode]
            #key = (model_info['CATEGORY_NAME'] , model_info['SUB_CATEGORY'], model_info['KATABAN'], model_info['MODEL_NAME'], modelcode)
            #keys.append(key)

        # 辞書内のすべてのDataFrameを結合 
        concatenated_df = pd.concat([p.df_br for p in self.psi.values()], keys=keys, names=['MODEL_CODE'])
        concatenated_df_c = pd.concat([p.df_rt for p in self.psi.values()], keys=keys, names=['MODEL_CODE'])

        # インデックスをカラムに変換
        transformed_df = concatenated_df.reset_index()
        transformed_df_c = concatenated_df_c.reset_index()

        splitted = (self.scenario + '-na').split("-") # -na is to avoid errors when scenario or eventname is not available (secure minimun split qty)
        # 変換前のテーブルにないフィールドの生成
        transformed_df['SCENARIO'] = splitted[0]
        transformed_df['FCT_TYPE'] = splitted[1]
        
        transformed_df_c['SCENARIO'] = splitted[0]
        transformed_df_c['FCT_TYPE'] = splitted[1]

        # 直接対応するフィールドのマッピング
        transformed_df.rename(columns={
            'node': 'FCT_ACCOUNT',
            'date': 'WEEK_DATE',
            'P': 'SELLIN_QTY',
            'S': 'SELLOUT_QTY',
            'I': 'CH_STOCK_OH',
            'std_ds': 'WOS_TARGET',
            'min_stock': 'MINSTOCK_TARGET',
            'ADJ':      'SELL_IN_ADJ',
            'demand': 'NATURAL_DEMAND',
            'week': 'SEIHAN_WEEK',
            'month': 'SEIHAN_MONTH_NAME',
            'year': 'SEIHAN_YEAR',
            'fy': 'SEIHAN_FISCAL_YEAR'
        }, inplace=True)


        # 直接対応するフィールドのマッピング  columns=['INCOMING', 'S', 'I','demand','std_ds','prev_inv','RR','ADJ','EOL','EOL_QTY','FROZEN','CAP','OSI'week','month','year','fy'])
        transformed_df_c.rename(columns={
            'date': 'WEEK_DATE',
            'FROZEN':   'FROZEN',
            'CAP':      'PO_ETA_CAP',
            'OSI':      'PO_ETA_OSI', 
            'INCOMING': 'PO_ETA_QTY', 
            'S':        'SELLIN_QTY', 
            'I':        'SC_STOCK_OH',
            'std_ds':   'WOS_TARGET',
            'prev_inv': 'NA',
            'RR':       'NA',
            'ADJ':      'SELL_IN_ADJ',
            'EOL':      'EOL',
            'EOL_QTY':  'EOL_QTY',
            'demand':   'NATURAL_DEMAND',
            'week':     'SEIHAN_WEEK',
            'month':    'SEIHAN_MONTH_NAME',
            'year':     'SEIHAN_YEAR',
            'fy':       'SEIHAN_FISCAL_YEAR'
        }, inplace=True)


        #transformed_df_c['ACT_WEEK_SI'] = self.act_week_si
        #transformed_df_c['ACT_WEEK_SO'] = self.act_week_so
        #transformed_df_c['MONTHLY_FLASH'] = 1
        #transformed_df_c['FCT_TYPE'] = 'DEMAND'
        #transformed_df_c['FSMC'] = self.fsmc
        #transformed_df_c['SALES_COMPANY'] = self.country
        transformed_df_c['COUNTRY'] = self.country
        # Not available yet in PSI class
        transformed_df_c['SELLOUT_QTY'] = 0
        transformed_df_c['CH_STOCK_OH'] = 0

                         #'PO_FROZEN_PERIOD','LEAD_TIME','ETA_FROZEN_PERIOD','ALLOCATION_YEAR','ALLOCATION_WEEK','EOL_SEIHAN_YEAR','EOL_SEIHAN_WEEK','EOL_QTY','EOL_TYPE']
        #transformed_df_c =transformed_df_c[column_order_c]
        # カラムの順序を整理
        #column_order = ['CATEGORY_NAME', 'SUB_CATEGORY', 'SEGMENT_2', 'MODEL_NAME', 'MODEL_CODE', 'FCT_ACCOUNT', 'WOS_TARGET','MINSTOCK_TARGET', 'NATURAL_DEMAND',
        #                'SEIHAN_YEAR', 'SEIHAN_FISCAL_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK', \
        #                'SELLIN_QTY', 'SELLOUT_QTY','CH_STOCK_OH','SCENARIO', 'FCT_TYPE', 'SCENARIO_EVENT', 'SELL_IN_ADJ']


        # 1. Copy SELLIN_QTY to a new column SELLIN_QTY_ORI
        transformed_df['SELLIN_QTY_ORI'] = transformed_df['SELLIN_QTY']
        #transformed_df['SELLOUT_QTY_ORI'] = transformed_df['SELLOUT_QTY']
        transformed_df['CH_STOCK_OH_ORI'] = transformed_df['CH_STOCK_OH']

        # 2. Copy SELLOUT_QTY to SELLIN_QTY for specified FCT_ACCOUNT values
        mask = transformed_df['FCT_ACCOUNT'].isin(['SONY STORE', 'ECOMMERCE'])
        transformed_df.loc[mask, 'SELLIN_QTY'] = transformed_df.loc[mask, 'SELLOUT_QTY']
        transformed_df.loc[mask, 'CH_STOCK_OH'] = 0


        # 3. AMOUNT CALCULATION
        transformed_df = transformed_df.merge(self.nsmp_list,how='left')
        transformed_df['SELLIN_QTY'] = transformed_df['SELLIN_QTY'].astype(float)
        transformed_df['NET_SALES_LC']=transformed_df['SELLIN_QTY'].round(0) * transformed_df['NSP_LC']
        transformed_df['MP_AMOUNT_LC']=transformed_df['SELLIN_QTY'].round(0) * (transformed_df['NSP_LC'] - transformed_df['MP_COST_LC'])
        transformed_df.drop(columns=['NSP_LC','MP_COST_LC' ], inplace=True)
        transformed_df['GROSS_SALES_LC']=0
        transformed_df['RANGING_TARGET']=0
        transformed_df['DISPLAY_INV_QTY']=0
        #4. ADD ACCOUT_GROUP
        transformed_df = transformed_df.merge(self.account_map, on='FCT_ACCOUNT', how='left')
        transformed_df['FCT_ACCOUNT'].fillna('OTHERS', inplace=True)
        transformed_df['COUNTRY'] = self.country


        #transformed_df['ACT_WEEK_SI'] = self.act_week_si
        #transformed_df['ACT_WEEK_SO'] = self.act_week_so
        #transformed_df['MONTHLY_FLASH'] = self.monthly_flash
        #transformed_df['FCT_TYPE'] = self.forecast_type
        #transformed_df['FSMC'] = self.fsmc
        
        ####### dataframe for LLM ##############
        
                        
        llm_df_ch = transformed_df.copy()
        llm_df_sc = transformed_df_c.copy()
                
        llm_df_ch = pd.merge(llm_df_ch, db_data['product_map'][['CATEGORY_NAME', 'SUB_CATEGORY', 'MODEL_CODE', 'MODEL_NAME', 'PREDECESSOR', 'PRE_NAME', 'KATABAN']],
                             on=['MODEL_CODE'], how='left')
        columns = llm_df_ch.columns.tolist()
        move_cols = ['MODEL_NAME', 'PREDECESSOR', 'PRE_NAME', 'CATEGORY_NAME', 'SUB_CATEGORY']
        remaining_cols = [col for col in columns if col not in move_cols]
        new_columns = remaining_cols[:1] + move_cols + remaining_cols[1:]
        # Reorder the DataFrame
        llm_df_ch = llm_df_ch[new_columns]
        llm_df_ch = pd.merge(llm_df_ch, db_data['pricing'][['MODEL_CODE', 'SCENARIO', 'SEIHAN_YEAR', 'SEIHAN_WEEK', 'SRP', 'DISCOUNT', 'DISCOUNT_PCT', 'PROMO_PRICE']], 
                             on=['MODEL_CODE', 'SCENARIO', 'SEIHAN_YEAR', 'SEIHAN_WEEK'], how='left')
        llm_df_ch = pd.merge(llm_df_ch, db_data['event'][['SCENARIO', 'SEIHAN_YEAR', 'SEIHAN_WEEK', 'PROMO_EVENT']], 
                             on=['SCENARIO', 'SEIHAN_YEAR', 'SEIHAN_WEEK'], how='left')
        llm_df_ch = pd.merge(llm_df_ch, db_data['dmd_fct'][['MODEL_CODE', 'SCENARIO', 'SEIHAN_YEAR', 'SEIHAN_WEEK', 'SEASONALITY', 'RR_UPPER', 'RR_LOWER', 'ADJ2_RR']], 
                             on=['MODEL_CODE', 'SCENARIO', 'SEIHAN_YEAR', 'SEIHAN_WEEK'], how='left')
        # llm_df_ch = pd.merge(llm_df_ch, db_data['sellin_param'][['SCENARIO', 'FCT_TYPE', 'MODEL_CODE', 'FCT_ACCOUNT', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK', 'WOS_TARGET', 'MINSTOCK_TARGET', 'RANGING_TARGET']],
                            #  on=['SCENARIO', 'FCT_TYPE', 'MODEL_CODE', 'FCT_ACCOUNT', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK'], how='left')
        llm_df_ch = pd.merge(llm_df_ch, db_data['seihan_param'][['SCENARIO', 'FCT_TYPE', 'MODEL_CODE', 'PO_FROZEN_PERIOD', 'FACTORY_LEAD_TIME', 'ETA_FROZEN_PERIOD', 'ALLOCATION_YEAR', 'ALLOCATION_WEEK', 'EOL_SEIHAN_YEAR', 'EOL_SEIHAN_WEEK', 'EOL_QTY', 
                                                                 ]],
                             on=['SCENARIO', 'FCT_TYPE', 'MODEL_CODE'], how='left')
        demand_split = db_data['demand_split'].copy()
        demand_split.rename(columns={'ADJ2_RR': 'ADJ2_RR_ACCOUNT', 'SEASONALITY': 'SEASONALITY_ACCOUNT'}, inplace=True)
        llm_df_ch = pd.merge(llm_df_ch, demand_split[['MODEL_CODE', 'FCT_ACCOUNT', 'SCENARIO', 'SEIHAN_YEAR', 'SEIHAN_WEEK', 'SEASONALITY_ACCOUNT', 'ADJ2_RR_ACCOUNT',
                                                      'SEASONALITY_IMPACT', 'PROMOTION_IMPACT', 'EVENT_IMPACT', 'RANGING_IMPACT', 'TREND_ADJ_IMPACT', 'SHORTAGE_IMPACT']], 
                             on=['MODEL_CODE', 'FCT_ACCOUNT', 'SCENARIO', 'SEIHAN_YEAR', 'SEIHAN_WEEK'], how='left')
        
        llm_df_sc = pd.merge(llm_df_sc, db_data['product_map'][['CATEGORY_NAME', 'SUB_CATEGORY', 'MODEL_CODE', 'MODEL_NAME', 'PREDECESSOR', 'PRE_NAME', 'KATABAN']],
                                on=['MODEL_CODE'], how='left')
        columns = llm_df_sc.columns.tolist()
        move_cols = ['MODEL_NAME', 'PREDECESSOR', 'PRE_NAME', 'CATEGORY_NAME', 'SUB_CATEGORY']
        remaining_cols = [col for col in columns if col not in move_cols]
        new_columns = remaining_cols[:1] + move_cols + remaining_cols[1:]
        # Reorder the DataFrame
        llm_df_sc = llm_df_sc[new_columns]
        llm_df_sc = pd.merge(llm_df_sc, db_data['pricing'][['MODEL_CODE', 'SCENARIO', 'SEIHAN_YEAR', 'SEIHAN_WEEK', 'SRP', 'DISCOUNT', 'DISCOUNT_PCT', 'PROMO_PRICE']],
                                on=['MODEL_CODE', 'SCENARIO', 'SEIHAN_YEAR', 'SEIHAN_WEEK'], how='left')
        llm_df_sc = pd.merge(llm_df_sc, db_data['event'][['SCENARIO', 'SEIHAN_YEAR', 'SEIHAN_WEEK', 'PROMO_EVENT']],
                                on=['SCENARIO', 'SEIHAN_YEAR', 'SEIHAN_WEEK'], how='left')
        llm_df_sc = pd.merge(llm_df_sc, db_data['dmd_fct'][['MODEL_CODE', 'SCENARIO', 'SEIHAN_YEAR', 'SEIHAN_WEEK', 'SEASONALITY', 'RR_UPPER', 'RR_LOWER', 'ADJ2_RR',
                                                            'SEASONALITY_IMPACT', 'PROMOTION_IMPACT', 'EVENT_IMPACT', 'RANGING_IMPACT', 'TREND_ADJ_IMPACT', 'SHORTAGE_IMPACT']],
                                on=['MODEL_CODE', 'SCENARIO', 'SEIHAN_YEAR', 'SEIHAN_WEEK'], how='left')
        # llm_df_sc = pd.merge(llm_df_sc, db_data['sellin_param'][['SCENARIO', 'FCT_TYPE', 'MODEL_CODE', 'FCT_ACCOUNT', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK', 'WOS_TARGET', 'MINSTOCK_TARGET', 'RANGING_TARGET']],
        #                     on=['SCENARIO', 'FCT_TYPE', 'MODEL_CODE', 'FCT_ACCOUNT', 'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK'], how='left')
        llm_df_sc = pd.merge(llm_df_sc, db_data['seihan_param'][['SCENARIO', 'FCT_TYPE', 'MODEL_CODE', 'PO_FROZEN_PERIOD', 'FACTORY_LEAD_TIME', 'ETA_FROZEN_PERIOD', 'ALLOCATION_YEAR', 'ALLOCATION_WEEK', 'EOL_SEIHAN_YEAR', 'EOL_SEIHAN_WEEK',
                                                                 ]],
                                on=['SCENARIO', 'FCT_TYPE', 'MODEL_CODE'], how='left')
        
        llm_df_sc.rename(columns={'SELLIN_QTY': 'SELLIN_QTY_SC'}, inplace=True)
        sellin_replace_country = llm_df_ch.groupby(['MODEL_CODE', 'SCENARIO', 'SEIHAN_YEAR', 'SEIHAN_WEEK']).agg({'SELLIN_QTY':'sum'}).reset_index()
        
        # replace SELLIN_QTY in llm_df_sc with sellin_replace_country
        llm_df_sc = llm_df_sc.merge(sellin_replace_country, on=['MODEL_CODE', 'SCENARIO', 'SEIHAN_YEAR', 'SEIHAN_WEEK'], how='left')
        
        # llm_df_ch.to_csv('channel_level.csv')
        # llm_df_sc.to_csv('country_level.csv')
        
        country_name = llm_df_sc['COUNTRY'].unique()[0]
        category_name = llm_df_sc['CATEGORY_NAME'].unique()[0]
        scenario_name = llm_df_sc['SCENARIO'].unique()[0]
        target_directory = "E:\LLM"
        
        # file path
        channel_csv_path = os.path.join(target_directory, f"channel_level_{country_name}_{category_name}_{scenario_name}.csv")
        country_csv_path = os.path.join(target_directory, f"country_level_{country_name}_{category_name}_{scenario_name}.csv")

        llm_df_ch.to_csv(channel_csv_path)
        llm_df_sc.to_csv(country_csv_path)
        
        ######################################


        sellin_param_order = ['SCENARIO','FCT_TYPE','MODEL_CODE','COUNTRY','FCT_ACCOUNT','SEIHAN_YEAR','SEIHAN_MONTH_NAME','SEIHAN_WEEK','WEEK_DATE','WOS_TARGET','MINSTOCK_TARGET','RANGING_TARGET']
        sellin_param = transformed_df[sellin_param_order].copy()


        sellin_param_ch = transformed_df[['SCENARIO','FCT_TYPE','MODEL_CODE','COUNTRY','FCT_ACCOUNT','SEIHAN_YEAR','SEIHAN_MONTH_NAME','SEIHAN_WEEK','WEEK_DATE','WOS_TARGET','MINSTOCK_TARGET','RANGING_TARGET']].copy()
        sellin_param_sc = transformed_df_c[['SCENARIO','FCT_TYPE','MODEL_CODE','COUNTRY','SEIHAN_YEAR','SEIHAN_MONTH_NAME','SEIHAN_WEEK','WEEK_DATE','WOS_TARGET']].copy()
        sellin_param_sc['FCT_ACCOUNT'] = '_TOTAL_COUNTRY' 
        sellin_param_sc['MINSTOCK_TARGET'] = 0
        sellin_param_sc['RANGING_TARGET'] = 0
        sellin_param_ch = sellin_param_ch.dropna(how='all', axis=1)
        sellin_param_sc = sellin_param_sc.dropna(how='all', axis=1)
        sellin_param = pd.concat([sellin_param_sc, sellin_param_ch])
        sellin_param = sellin_param.dropna(subset=['WOS_TARGET','MINSTOCK_TARGET','RANGING_TARGET'])



        # === SELLIN FIX 

        fix_sellin_adj_ch = transformed_df[['SCENARIO','FCT_TYPE','COUNTRY','MODEL_CODE','FCT_ACCOUNT','SEIHAN_YEAR','SEIHAN_MONTH_NAME','SEIHAN_WEEK','WEEK_DATE','SELL_IN_ADJ']].copy()
        fix_sellin_adj_sc = transformed_df_c[['SCENARIO','FCT_TYPE','COUNTRY','MODEL_CODE','SEIHAN_YEAR','SEIHAN_MONTH_NAME','SEIHAN_WEEK','WEEK_DATE','SELL_IN_ADJ']].copy()
        fix_sellin_adj_sc['FCT_ACCOUNT'] = '_TOTAL_COUNTRY' 
        fix_sellin_adj_ch = fix_sellin_adj_ch.dropna(how='all', axis=1)
        fix_sellin_adj_sc = fix_sellin_adj_sc.dropna(how='all', axis=1)
        fix_sellin_adj = pd.concat([fix_sellin_adj_sc, fix_sellin_adj_ch])
        fix_sellin_adj = fix_sellin_adj.dropna(subset=['SELL_IN_ADJ'])
        fix_sellin_adj=fix_sellin_adj[fix_sellin_adj['SELL_IN_ADJ']>0]

        fix_sellin_adj['WEEK_DATE'] = pd.to_datetime(fix_sellin_adj['WEEK_DATE'])
        fix_sellin_adj['CONFIRMED'] = 1
        #print(fix_sellin_adj.reset_index())

        
        # CH_PSI_FCTカラムの順序を整理
        column_order = ['SCENARIO', 'FCT_TYPE', 'MODEL_CODE', 'COUNTRY', 'FCT_ACCOUNT', 
                        'SEIHAN_YEAR', 'SEIHAN_MONTH_NAME', 'SEIHAN_WEEK', 'WEEK_DATE' ,
                        'SELLIN_QTY', 'SELLOUT_QTY','CH_STOCK_OH','DISPLAY_INV_QTY', 'NATURAL_DEMAND', 'SELLIN_QTY_ORI','CH_STOCK_OH_ORI','GROSS_SALES_LC','NET_SALES_LC', 'MP_AMOUNT_LC']
        ch_psi_fct = transformed_df[column_order].copy()


        # === SEIHAN SETTING generation
        tmp_df = transformed_df_c[['SCENARIO','FCT_TYPE','MODEL_CODE','COUNTRY','FROZEN', 'WEEK_DATE','EOL_QTY']].copy()

        # TRim PSI_C columns
        column_order_c =['SCENARIO','FCT_TYPE','MODEL_CODE','COUNTRY','SEIHAN_YEAR','SEIHAN_MONTH_NAME','SEIHAN_WEEK','WEEK_DATE','PO_ETA_OSI','PO_ETA_CAP','PO_ETA_QTY','SC_STOCK_OH','SELLIN_QTY','SELLOUT_QTY','CH_STOCK_OH']

        sc_psi_fct =transformed_df_c[column_order_c].copy()

        # Remove data that includes and predates the act_date
        tmp_df = tmp_df[tmp_df['WEEK_DATE'] >  pd.Timestamp(self.act_date)]

        tmp_df.sort_values(by='WEEK_DATE', inplace=True)
        tmp_df.reset_index(drop=True, inplace=True)

        # Replace 0 with NaN in 'EOL_QTY' before grouping
        tmp_df['EOL_QTY'].replace(0, np.nan, inplace=True)


        # Group and aggregate data
        seihan_setting = tmp_df.groupby(['SCENARIO','FCT_TYPE','MODEL_CODE','COUNTRY']).agg(
            ETA_FROZEN_PERIOD=('FROZEN', lambda x: (x == 'Fix').sum()),  # Count 'Fix' occurrences
            _LAST_CAP=('FROZEN', lambda x: x[x == 'CAP'].index.max()),  # Index of the last 'Cap'
            _LAST_EOL_INDEX=('EOL_QTY', lambda x: x[x != 0].last_valid_index()),  # Index of the last non-zero EOL_QTY
            EOL_QTY=('EOL_QTY', 'last')  # Last non-zero value of EOL_QTY
        )

        # Extract the date of the last 'Cap' and calculate its year and week
        if '_LAST_CAP' in seihan_setting.columns:
            seihan_setting['ALLOCATION_YEAR'] = seihan_setting['_LAST_CAP'].apply(lambda x: tmp_df.loc[x, 'WEEK_DATE'].year if pd.notna(x) else np.nan)
            seihan_setting['ALLOCATION_WEEK'] = seihan_setting['_LAST_CAP'].apply(lambda x: tmp_df.loc[x, 'WEEK_DATE'].isocalendar()[1] if pd.notna(x) else np.nan)

        # Extract the EOL date and calculate its year and week
        if '_LAST_EOL_INDEX' in seihan_setting.columns:
            seihan_setting['_EOL_YEAR_WEEK_MONTH'] = seihan_setting['_LAST_EOL_INDEX'].apply(lambda x: date_to_cy_mth_wk(tmp_df.loc[x, 'WEEK_DATE']) if pd.notna(x) else (np.nan,np.nan,np.nan))
            seihan_setting['EOL_SEIHAN_YEAR'] = seihan_setting['_EOL_YEAR_WEEK_MONTH'].apply(lambda x: x[0])
            seihan_setting['EOL_SEIHAN_WEEK'] = seihan_setting['_EOL_YEAR_WEEK_MONTH'].apply(lambda x: x[1])


        seihan_setting['EOL_QTY'].fillna(0, inplace=True)
        # Remove unnecessary columns
        seihan_setting.drop(columns=['_LAST_CAP', '_LAST_EOL_INDEX', '_EOL_YEAR_WEEK_MONTH'], inplace=True)
        seihan_setting = seihan_setting.reset_index()
        
        seihan_setting['FACTORY_LOCATION'] = None
        seihan_setting['WAREHOUSE'] = None
        seihan_setting['TR_TYPE'] = None
        seihan_setting['PO_FROZEN_PERIOD'] = np.nan
        seihan_setting['FACTORY_LEAD_TIME'] = np.nan
        seihan_setting['EOL_TYPE'] = 'SINGLE_SHIP'
        seihan_setting['LAST_SHIP_SEIHAN_YEAR']=np.nan
        seihan_setting['LAST_SHIP_SEIHAN_WEEK']=np.nan
        seihan_setting['LAST_SHIP_QTY']=np.nan
        seihan_setting['MCQ']=None
        seihan_setting['SELLIN_LEAD_TIME']=1


        seihan_setting = seihan_setting[['SCENARIO','FCT_TYPE','MODEL_CODE','COUNTRY','FACTORY_LOCATION','TR_TYPE','WAREHOUSE','PO_FROZEN_PERIOD','FACTORY_LEAD_TIME','ETA_FROZEN_PERIOD','ALLOCATION_YEAR','ALLOCATION_WEEK','EOL_SEIHAN_YEAR','EOL_SEIHAN_WEEK','EOL_QTY','LAST_SHIP_SEIHAN_YEAR','LAST_SHIP_SEIHAN_WEEK','LAST_SHIP_QTY','MCQ','SELLIN_LEAD_TIME']]


        
        ch_psi_fct['WEEK_DATE'] = pd.to_datetime(ch_psi_fct['WEEK_DATE']).dt.date
        sc_psi_fct['WEEK_DATE'] = pd.to_datetime(sc_psi_fct['WEEK_DATE']).dt.date
        sellin_param['WEEK_DATE'] = pd.to_datetime(sellin_param['WEEK_DATE']).dt.date
        fix_sellin_adj['WEEK_DATE'] = pd.to_datetime(fix_sellin_adj['WEEK_DATE']).dt.date
        return ch_psi_fct,sc_psi_fct,sellin_param,seihan_setting,fix_sellin_adj
    
    def copy(self):
        return copy.deepcopy(self)

    def update_scenario(self, new_scenario = None, new_act_date = None, new_si_act_wk = None, new_so_act_wk = None):
        if new_scenario is not None:
            self.scenario = new_scenario #Need to validate in the future

        if new_act_date is not None:
            self.act_date = new_act_date
            for product_code in self.product_list:
                self.psi[product_code].act_date = self.act_date

        if new_si_act_wk is not None:
            self.act_week_si = new_si_act_wk

        if new_si_act_wk is not None:
            self.act_week_so = new_so_act_wk

    def update_act(self, si_act_yyyyww, so_act_yyyyww, _df = None, _df_c = None, callback=None):

        df=_df.copy()
        df_c = _df_c.copy()
        print(f'#-- PSI Container: Updating ACT data.. ({self.scenario})')
        si_act_date = si_act_yyyyww
        so_act_date = so_act_yyyyww
        inv_act_date = so_act_yyyyww

        act_date = pd.Timestamp(date_from_cy_wk(si_act_date//100, si_act_date%100))
        prev_date = act_date - pd.Timedelta(weeks=1)
        df['YYYYWW'] =df['SEIHAN_YEAR'] *100 + df['SEIHAN_WEEK']
        df= df[df['YYYYWW'] <= max(si_act_date,so_act_date,inv_act_date)]   
        contains_dtc = self.account_map['FCT_ACCOUNT'].isin(['SONY STORE','ECOMMERCE']).sum() > 0

        if contains_dtc: # if account list contains DTC 
            # Merge DTC inventory
            df_sonystore = df_c[['MODEL_CODE', 'SEIHAN_YEAR', 'SEIHAN_WEEK', 'DTC_OFFLINE_INV_QTY']].copy()
            df_ecommerce = df_c[['MODEL_CODE', 'SEIHAN_YEAR', 'SEIHAN_WEEK', 'DTC_OFFLINE_INV_QTY']].copy()
            df_sonystore['FCT_ACCOUNT'] = 'SONY STORE'                                                                  # Set DTC account name 
            df_ecommerce['FCT_ACCOUNT'] = 'ECOMMERCE'                                                                  # Set DTC account name 
            df_sonystore['DTC_OFFLINE_INV_QTY'] = df_c['DTC_OFFLINE_INV_QTY'] * 0.75
            df_ecommerce['DTC_OFFLINE_INV_QTY'] = df_c['DTC_OFFLINE_INV_QTY'] * 0.25
            df = df.merge(pd.concat([df_sonystore, df_ecommerce]),         # Merge DTC_OFFLINE_INV_QTY
                        on=['MODEL_CODE','FCT_ACCOUNT','SEIHAN_YEAR','SEIHAN_WEEK',], 
                        how='left')
            df['DTC_OFFLINE_INV_QTY'].fillna(0,inplace=True)                                                            # set nulls -> 0
            df.loc[df['FCT_ACCOUNT'].isin(['SONY STORE','ECOMMERCE']), 'CH_STOCK_OH'] = 0                                   # clear dtc inventory
            df.loc[df['FCT_ACCOUNT'].isin(['SONY STORE','ECOMMERCE']),'SELLIN_QTY'] = df.loc[df['FCT_ACCOUNT'].isin(['SONY STORE','ECOMMERCE']),'SELLIN_QTY']                                          # clear dtc sell-in
            df.loc[df['FCT_ACCOUNT'].isin(['SONY STORE','ECOMMERCE']),'SELLOUT_QTY'] = df.loc[df['FCT_ACCOUNT'].isin(['SONY STORE','ECOMMERCE']),'SELLIN_QTY']                                          # clear dtc sell-in
            df.loc[df['FCT_ACCOUNT'].isin(['SONY STORE','ECOMMERCE']), 'CH_STOCK_OH'] = df.loc[df['FCT_ACCOUNT'].isin(['SONY STORE','ECOMMERCE']), 'DTC_OFFLINE_INV_QTY']
            '''
            df.loc[df['FCT_ACCOUNT'].isin(['SONY STORE','ECOMMERCE']),'SELLOUT_QTY'] = np.where(df.loc[df['FCT_ACCOUNT'].isin(['SONY STORE','ECOMMERCE']), 'YYYYWW']  <= so_act_date, 
                                                                                                df.loc[df['FCT_ACCOUNT'].isin(['SONY STORE','ECOMMERCE']),'SELLIN_QTY'] , 
                                                                                                np.nan) 
            df.loc[df['FCT_ACCOUNT'].isin(['SONY STORE','ECOMMERCE']),'CH_STOCK_OH'] = np.where(df.loc[df['FCT_ACCOUNT'].isin(['SONY STORE','ECOMMERCE']), 'YYYYWW'] <= inv_act_date, 
                                                                                                    df.loc[df['FCT_ACCOUNT'].isin(['SONY STORE','ECOMMERCE']), 'DTC_OFFLINE_INV_QTY'], 
                                                                                                    np.nan)   
'''
        # Restrict sellin act range 
        df['SELLIN_QTY'] = df['SELLIN_QTY'].mask(  (df['YYYYWW'] > si_act_date), np.nan)




        if contains_dtc: # if account list contains DTC 
            # Update SELLOUT value: DTC-> SELLIN,  DEALERS -> SELLOUT
            df['SELLOUT_QTY'] = df['SELLIN_QTY'].where(df['FCT_ACCOUNT'].isin(['SONY STORE', 'ECOMMERCE']), df['SELLOUT_QTY'])
            # Update SELLOUT range restriction: DTC-> keep until 'si_act_date', DEALERS -> keep until 'so_act_date' , rest set NULL
            df.loc[(df['FCT_ACCOUNT'].isin(['SONY STORE', 'ECOMMERCE'])) & (df['YYYYWW'] > si_act_date), 'SELLOUT_QTY'] = np.nan        # DTC
            df.loc[~(df['FCT_ACCOUNT'].isin(['SONY STORE', 'ECOMMERCE'])) & (df['YYYYWW'] > so_act_date), 'SELLOUT_QTY'] = np.nan       # NON-DTC

            
            # Update CH INV range restriction: DTC-> keep until 'si_act_date', DEALERS -> keep until 'so_act_date' , rest set NULL
            df.loc[(df['FCT_ACCOUNT'].isin(['SONY STORE', 'ECOMMERCE'])) & (df['YYYYWW'] > si_act_date), 'CH_STOCK_OH'] = np.nan   # DTC
            df.loc[~(df['FCT_ACCOUNT'].isin(['SONY STORE', 'ECOMMERCE'])) & (df['YYYYWW'] > so_act_date), 'CH_STOCK_OH'] = np.nan  # NON-DTC

        else:
            df['SELLOUT_QTY'] = df['SELLOUT_QTY'].mask(  (df['YYYYWW'] > so_act_date), np.nan)

    

        ''' # This code should be run after updating PSI
        # calcuate LE inventory (Accounts / DTC)
        def update_sellable_inv_qty(group):
            current_week_row = group[group['YYYYWW'] == si_act_date]                                        # Select current week row
            if not current_week_row.empty:
                act_date = current_week_row['date'].to_list()[0]                                                   # current week date 
                last_week_date = act_date - pd.Timedelta(weeks=1)                                           # previous week date
                last_week_row = group[group['date'] == last_week_date]
                # Calculate inventory 
                last_week_sellable_inv_qty = last_week_row['CH_STOCK_OH'].sum() if not last_week_row.empty else 0
                current_week_sellin_qty = current_week_row['SELLIN_QTY'].sum()
                current_week_sellout_qty = current_week_row['SELLOUT_QTY'].sum()
                updated_qty = last_week_sellable_inv_qty + current_week_sellin_qty - current_week_sellout_qty
                group.loc[group['YYYYWW'] == si_act_date, 'CH_STOCK_OH'] = updated_qty
            return group
        # Calculate latest week for all dealers
        df = df.groupby(['MODEL_CODE', 'FCT_ACCOUNT']).apply(update_sellable_inv_qty).reset_index(drop=True)
        '''
        
        # Create 'date' column (if not available)
        if 'WEEK_DATE' not in df.columns:
            df['WEEK_DATE'] = vec_date_from_cy_wk(df['SEIHAN_YEAR'], df['SEIHAN_WEEK'])

        # Create 'date' column (if not available)
        if 'WEEK_DATE' not in df_c.columns:
            df_c['WEEK_DATE'] = vec_date_from_cy_wk(df_c['SEIHAN_YEAR'], df_c['SEIHAN_WEEK'])




        def incomming_shift(group):

            group['date_delta'] = np.where(group['WEEK_DATE'] > act_date, timedelta(weeks=1), timedelta(weeks=0 ))
            group['date_shifted'] = pd.to_datetime(group['WEEK_DATE'] + group['date_delta'])
            #group['date_shifted'] = group['date_shifted'])
            #group['date_shifted'] = group['date'].mask(group['date'] > act_date, group['date'] + pd.to_timedelta(7, unit='days'))
            #group['date_shifted'] = group['date'].where(group['date'] <= act_date, group['date'] + pd.to_timedelta(7, 'days'))


            merged_group = pd.merge(group,  group[['MODEL_CODE', 'date_shifted', 'PO_ETA_QTY']], 
                                            left_on=['MODEL_CODE', 'WEEK_DATE'], 
                                            right_on=['MODEL_CODE', 'date_shifted'], 
                                            how='left', 
                                            suffixes=('', '_shifted'))
            merged_group.loc[:,'PO_ETA_QTY_shifted'] = merged_group['PO_ETA_QTY_shifted'].fillna(0)
            final_group = merged_group.drop(columns=['date_shifted', 'date_shifted_shifted','date_delta'])
            return final_group

        # 全てのMODEL_CODEグループに対して処理を適用
        df_processed = df_c.groupby('MODEL_CODE').apply(incomming_shift).reset_index(drop=True)
        df_c = df_processed




        # Transform dataframe into psi-dataframe template (munti index)
        def preprocess_data(df):
            new_df = pd.DataFrame({
                'store': df['FCT_ACCOUNT'],
                'date': df['WEEK_DATE'],
                'cy': df['SEIHAN_YEAR'].astype(int),
                'fy': df['SEIHAN_FISCAL_YEAR'].astype(int),
                'wk': df['SEIHAN_WEEK'].astype(int),
                'P': df['SELLIN_QTY'],
                'S': df['SELLOUT_QTY'],
                #'demand': df['SELLOUT_QTY'],
                'I': df['CH_STOCK_OH'],
                #'O': df['SELLOUT_QTY'],
                #'std_ds': df['WOS_TARGET'],
                #'min_stock': df['MINSTOCK_TARGET'],
                #'prev_inv': np.nan,
                #'RR': 0,
                #'ADJ': 0,
                'year': df['SEIHAN_YEAR'].astype(int),
                'week': df['SEIHAN_WEEK'].astype(int),
                'month':df['SEIHAN_MONTH_NAME']
            })
            new_df.set_index(['store', 'date'], inplace=True)
            return new_df

        # preprocess_data関数の定義（dateの計算は除外）
        def preprocess_data_c(_df):
            new_df = pd.DataFrame({
                'node': self.country,
                'date': _df['WEEK_DATE'],
                'CAP': _df['PO_ETA_QTY_shifted'],
                'OSI': _df['PO_ETA_QTY'],
                'INCOMING': _df['PO_ETA_QTY_shifted'],
                'I': _df['SC_STOCK_OH'],
                'year': _df['SEIHAN_YEAR'].astype(int),
                'week': _df['SEIHAN_WEEK'].astype(int),
                'month':_df['SEIHAN_MONTH_NAME']
            })
            new_df.set_index(['node', 'date'], inplace=True)
            return new_df

        model_count = len(self.product_list)

        for index, product_code in enumerate(self.product_list): 
            product_df = df[df['MODEL_CODE'] == product_code]
            product_df_c = df_c[df_c['MODEL_CODE'] == product_code]
            processed_df = preprocess_data(product_df)
            processed_df_c = preprocess_data_c(product_df_c)
            
            try:
                self.psi[product_code].update(_df_br=processed_df, _df_rt=processed_df_c)
            except Exception as e:
                print (f"ERROR: could not update model {product_code} -  {e}")

            if si_act_yyyyww != so_act_yyyyww and (not processed_df.empty):

                '''
                # Calculate LE OH Inv = Prev_OH + Curr_Incoming - sum_of_NonDTCsellin 
                psid = self.psi[product_code].df_br
                psic = self.psi[product_code].df_rt
                psic.loc[(psic['year']==si_act_date//100) & (psic['week']==si_act_date%100),'I'] = (
                    psic[(psic['year']==so_act_date//100) & (psic['week']==so_act_date%100)]['I'].sum() + 
                    psic[(psic['year']==si_act_date//100) & (psic['week']==si_act_date%100)]['INCOMING'].sum() - 
                    psid[(psid['year']==si_act_date//100) & (psid['week']==si_act_date%100) & (~psid.index.get_level_values(0).isin(['SONY STORE', 'ECOMMERCE']))]['P'].sum())
                '''

                # Calculate LE Ch. inventory and 
                self.psi[product_code].roll_i(act_date) #================================================================!!!! 
            

                prod_contains_dtc = product_df['FCT_ACCOUNT'].drop_duplicates().isin(['SONY STORE','ECOMMERCE']).sum() > 0
       
                if prod_contains_dtc: # if account list contains DTC 
                    try:
                        self.psi[product_code].update(_df_br=processed_df.loc[list(set(product_df['FCT_ACCOUNT']).intersection({'SONY STORE', 'ECOMMERCE'}))])
                    except KeyError as e:
                        print(f"Error: {e}")

                self.psi[product_code].df_br['I'] = self.psi[product_code].df_br['I'].clip(lower=0)

                # Stack up Sell-in to df_rt
                root = self.psi[product_code]._rootname
                # df_br summarize all accounts sellin (P column)
                p_sum = self.psi[product_code].df_br.groupby('date')['P'].sum()

                # Create a temp dataframe for updating
                df_bottomup_sellin = p_sum.reset_index()
                df_bottomup_sellin['node'] = root
                df_bottomup_sellin.set_index(['node', 'date'], inplace=True)
                df_bottomup_sellin.rename(columns={'P': 'S'}, inplace=True)

                # Update Country PSI (df_rt)
                self.psi[product_code].df_rt.update(df_bottomup_sellin)

                non_dtc_filter =  self.psi[product_code].df_br.index.get_level_values('node').difference(['SONY STORE', 'ECOMMERCE'])
                self.psi[product_code].df_rt.loc[(slice(None),act_date),'I'] = self.psi[product_code].df_rt.loc[(slice(None),prev_date),'I'].iloc[0] + \
                                                                                self.psi[product_code].df_rt.loc[(slice(None),act_date),'INCOMING'].iloc[0] - \
                                                                                self.psi[product_code].df_br.loc[(non_dtc_filter,act_date),'P'].sum()

            else:
                # Stack up Sell-in to df_rt
                root = self.psi[product_code]._rootname
                # df_br summarize all accounts sellin (P column)
                p_sum = self.psi[product_code].df_br.groupby('date')['P'].sum()

                # Create a temp dataframe for updating
                df_bottomup_sellin = p_sum.reset_index()
                df_bottomup_sellin['node'] = root
                df_bottomup_sellin.set_index(['node', 'date'], inplace=True)
                df_bottomup_sellin.rename(columns={'P': 'S'}, inplace=True)
                
                 # Update Country PSI (df_rt)
                self.psi[product_code].df_rt.update(df_bottomup_sellin)

            if callback:
                callback(1)


                

        print('#-- done..')

    def update_demand(self, _df = None, callback=None):

        df=_df.query(f"SCENARIO=='{self.scenario.split('-')[0]}'").copy()
        print('#-- PSI Container: Updating demand data..')

        # Create 'date' column (if not available)
        if 'WEEK_DATE' not in df.columns:
            df['WEEK_DATE'] = vec_date_from_cy_wk(df['SEIHAN_YEAR'], df['SEIHAN_WEEK'])
        
        act_date_so = date_from_cy_wk(self.act_week_so//100,self.act_week_so%100)
        act_date_si = date_from_cy_wk(self.act_week_si//100,self.act_week_si%100)

        df2=df.copy()
        df2=df2.loc[(df2['WEEK_DATE'].dt.date <= self.act_date) & (df2['WEEK_DATE'].dt.date > act_date_so)]
        df2['SELLOUT_TEMP'] = df2['NATURAL_DEMAND_SPLIT']
        
        df2['NATURAL_DEMAND_SPLIT'] = np.nan
        
        df=df.loc[df['WEEK_DATE'].dt.date > self.act_date]
        df = pd.concat([df, df2])

        # Transform dataframe into psi-dataframe template (munti index)
        def preprocess_data(df):
            new_df = pd.DataFrame({
                'store': df['FCT_ACCOUNT'],
                'date': df['WEEK_DATE'],
                'cy': df['SEIHAN_YEAR'].astype(int),
                'fy': df['SEIHAN_FISCAL_YEAR'].astype(int),
                'wk': df['SEIHAN_WEEK'].astype(int),
                'demand': df['NATURAL_DEMAND_SPLIT'],
                'S': df['SELLOUT_TEMP'],
                'year': df['SEIHAN_YEAR'].astype(int),
                'week': df['SEIHAN_WEEK'].astype(int),
                'month':df['SEIHAN_MONTH_NAME']
            })
            new_df.set_index(['store', 'date'], inplace=True)
            return new_df


        for product_code in self.product_list: 
            product_df = df[df['MODEL_CODE'] == product_code]
            processed_df = preprocess_data(product_df)
            

            self.psi[product_code].update(_df_br=processed_df)
            self.psi[product_code].roll_i(act_date_si,True) #================================================================!!!! 
            
            if callback:
                callback(1)

        print('#-- done..')
    
    def update_params(self, _df_si_prm = None, _df_shn_set = None):

        df_si_prm=_df_si_prm.copy()
        df_shn_set=_df_shn_set.copy()
        print('#-- PSI Container: Importing Parmeters..')

        # Create 'date' column (if not available)
        if 'WEEK_DATE' not in df_si_prm.columns:
            df_si_prm['WEEK_DATE'] = vec_date_from_cy_wk(df_si_prm['SEIHAN_YEAR'], df_si_prm['SEIHAN_WEEK'])

        # Create 'date' column (if not available)
        #if 'date' not in df_shn_set.columns:
        #    df_shn_set['date'] = vec_date_from_cy_wk(df_shn_set['SEIHAN_YEAR'], df_shn_set['SEIHAN_WEEK'])

        # Transform dataframe into psi-dataframe template (munti index)
        def preprocess_data(df):
            new_df = pd.DataFrame({
                'store': df['FCT_ACCOUNT'],
                'date': df['WEEK_DATE'],
                #'cy': df['SEIHAN_YEAR'].astype(int),
                #'fy': df['SEIHAN_FISCAL_YEAR'].astype(int),
                #'wk': df['SEIHAN_WEEK'].astype(int),
                'std_ds': df['WOS_TARGET'],
                'min_stock': df['MINSTOCK_TARGET'],
                #'year': df['SEIHAN_YEAR'].astype(int),
                #'week': df['SEIHAN_WEEK'].astype(int),
                #'month':df['SEIHAN_MONTH_NAME']
            })
            new_df.set_index(['store', 'date'], inplace=True)
            return new_df


        for product_code in self.product_list: 
            product_df = df_si_prm[df_si_prm['MODEL_CODE'] == product_code]
            processed_df = preprocess_data(product_df)






            # Filter psi-c dataframe
            _df_rt  = self.psi[product_code].df_rt.copy().drop(columns=['EOL','EOL_QTY','FROZEN'])
            #_df_c['SCENARIO'] = self.scenario.split('-')[0]
            #_df_c['FCT_TYPE'] = self.scenario.split('-')[1]
            _df_rt.reset_index(inplace=True)

            scn = self.scenario.split('-')[0]
            evt = self.scenario.split('-')[1]

            
            # filter seihan_settings dataframe
            shn_prm = _df_shn_set[(_df_shn_set['MODEL_CODE']== product_code) ]


            # Re-Merge EOL_QTY
            _df_rt = _df_rt.merge(shn_prm[[ 'EOL_SEIHAN_YEAR', 'EOL_SEIHAN_WEEK','EOL_QTY']],
                                        left_on = ['year', 'week'],
                                        right_on = [ 'EOL_SEIHAN_YEAR', 'EOL_SEIHAN_WEEK'], 
                                        how='left').drop(columns=[ 'EOL_SEIHAN_YEAR', 'EOL_SEIHAN_WEEK'])



            #act_year = self.act_week_si // 100                                                  # Get Year from YYYYWW
            #act_week = self.act_week_si % 100                                                   # Get Week from YYYYWW
            #act_date = pd.Timestamp(date_from_cy_wk(act_year,act_week))                         # calculate the date
            act_date = self.act_date
            #=== eol_timing calculation (case EOL is not included in the total period, then set a "dummy far date" (999 wks ahead) : only for comparison calculation)
            eol_timing_filter = _df_rt['EOL_QTY'].notna()
            eol_timing = _df_rt[eol_timing_filter]['date'].min() if eol_timing_filter.any() else pd.Timestamp(act_date + timedelta(weeks=999))

            #=== Frozen weeks calculatoion ( Get Model frozen period from SEIHAN_PARAMETER TABLE --> and determine frozen terimination date (from ACT_DATE))
            current_gp_params = shn_prm
            frozen_weeks =  current_gp_params['ETA_FROZEN_PERIOD']   # find the PO_FROZEN_PERIOD of current'SCENARIO','FCT_TYPE','MODEL_CODE' from seihan_params table
            frozen_weeks = 0 if frozen_weeks.empty or pd.isna(frozen_weeks).all() else frozen_weeks.to_list()[0]# if FROZEN_PERIOD not found in SETTING TABLE, set 0 time delts
            frozen_date = pd.Timestamp(act_date + timedelta(weeks=frozen_weeks))                                # Frozen date = act date + frozen weeks

            if current_gp_params.empty or current_gp_params['ALLOCATION_WEEK'].isna().any() or current_gp_params['ALLOCATION_YEAR'].isna().any():        
                alloc_date = pd.Timestamp(act_date)
            else:
                alloc_week = current_gp_params['ALLOCATION_WEEK'].astype(int).to_list()[0]
                alloc_year = current_gp_params['ALLOCATION_YEAR'].astype(int).to_list()[0]
                alloc_date = pd.Timestamp(date_from_cy_wk(alloc_year, alloc_week))         

            #=== add EOL and FROZEN columns
            _df_rt['EOL'] = np.where(_df_rt['date'] >= eol_timing, 'EOL', np.nan)                                 # set value 'EOL' for EOL onwards: if 'date' is earlier than eol_timing set NaN
            #group['FROZEN'] = np.where(group['date'] <= frozen_date, 'Fix', np.nan)                             # set value 'Fix' during frozen period: if 'date' is earlier than frozen_date set NaN
            _df_rt['FROZEN'] = np.where(_df_rt['date'] <= frozen_date, 'Fix', 
                            np.where(_df_rt['date'] <= alloc_date, 'CAP', np.nan))
            
            _df_rt = _df_rt.set_index(['node', 'date'])[['EOL','EOL_QTY','FROZEN']]
            self.psi[product_code].update(_df_br=processed_df, _df_rt=_df_rt)






        print('#-- done..')
    
    def update_osi(self, _scenario=None, _df = None):
        print('#-- PSI Container: Overwriting incomming OSI data..')
        start_time = time.time()

        _df = _df[_df['SCENARIO']+'-'+_df['FCT_TYPE']==_scenario]                                               # Filter scenario

        # exit if empty
        if _df.empty:
            raise StopIteration(f"'{_scenario}' supply adj information not found in database [supply_adj]. ")

        # Create 'date' column (if not available)
        if 'WEEK_DATE' not in _df.columns:
            _df['WEEK_DATE'] = vec_date_from_cy_wk(_df['SEIHAN_YEAR'], _df['SEIHAN_WEEK'])

        #_df['WEEK_DATE'] += pd.Timedelta(weeks=1)                                                # Delay 2 weeks
        # preprocess_data関数の定義（dateの計算は除外）



        def incomming_shift(group):
            group['date_delta'] = np.where(group['WEEK_DATE'] > pd.Timestamp(self.act_date), timedelta(weeks=1), timedelta(weeks=0 ))
            group['WEEK_DATE'] = pd.to_datetime(group['WEEK_DATE'] + group['date_delta']) 
            group['PO_ETA_QTY_shifted']  = group['PO_ETA_QTY']   
            final_group = group.drop(columns=['date_delta','PO_ETA_QTY'])
            return final_group

        # 全てのMODEL_CODEグループに対して処理を適用
        _df_shift = _df.groupby('MODEL_CODE').apply(incomming_shift).reset_index(drop=True)
        _df['PO_ETA_QTY_shifted'] = np.nan
        _df_shift['PO_ETA_QTY']  = np.nan



        def preprocess_data_c(_df):
            new_df = pd.DataFrame({
                'node': _df['COUNTRY'],
                'date': _df['WEEK_DATE'],
                'CAP': _df['PO_ETA_QTY_shifted'],
                'OSI': _df['PO_ETA_QTY'],
                'INCOMING': _df['PO_ETA_QTY_shifted'],
            })
            new_df.set_index(['node', 'date'], inplace=True)
            return new_df
        

        for product_code in self.product_list: 
            product__df1 = _df[_df['MODEL_CODE'] == product_code]
            if not product__df1.empty:
                processed__df1 = preprocess_data_c(product__df1)
                self.psi[product_code].update(_df_rt=processed__df1)

            product__df2 = _df_shift[_df_shift['MODEL_CODE'] == product_code]
            if not product__df2.empty:
                processed__df2 = preprocess_data_c(product__df2)
                self.psi[product_code].update(_df_rt=processed__df2)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('#-- done... ({elapsed_time:} secs)' )

def update_data_multiprocess_warm(psi_pickle, progress):
    try:
        # ピクル化されたインスタンスをロード
        def prt2(x):
            progress.value = x
        psi = pickle.loads(psi_pickle)
        tm = time.time()
        psi.ExecRollingCalc(prt2)
        print(f'child total time new: {time.time() - tm}')
        return pickle.dumps(psi)
    except Exception as e:
        print(f"Exception in child process: {e}")
        return pickle.dumps(None)

import multiprocessing
import pickle
from multiprocessing import Process, Pool, Manager, Value
def prt(x):
    print(x)

def child_process(psi_pickle):
    # ピクル化されたインスタンスをロード
    psi = pickle.loads(psi_pickle)
    tm = time.time()
    psi.ExecRollingCalc(prt)
    print(f'child total time new: {time.time()-tm}')
    return pickle.dumps(psi)

#if __name__ == "__main__":
def offline():
    # PSIインスタンスを作成
    psi = PSI.from_weeks('2000-01-01', 200, ['A','B','C'],'SGMC')
    psi.act_date = pd.to_datetime('2003-03-24')
    psi.df_br.loc[psi.df_br.index.get_level_values('date') <= psi.act_date, 'demand'] = 100


    # 条件に合致する行のインデックスを取得
    condition_indices = psi.df_br.index.get_level_values('date') <= psi.act_date

    # 乱数の範囲を指定（例: 0から1の間乱数）
    random_values = 1+np.random.rand(condition_indices.sum())*5

    # 'S'列に対して乱数をかける
    psi.df_br.loc[condition_indices, 'demand'] *= random_values

    #psi.predict_s()
    
    # インスタンスをピクル化
    psi_pickle = pickle.dumps(psi)
    '''
    # マルチプロセスを開始
    with multiprocessing.Pool(1) as pool:
        result_pickle = pool.apply(child_process, [psi_pickle])

    # 結果をロード
    psi_updated = pickle.loads(result_pickle)

    # 結果を確認
    print(psi_updated)  # 必要に応じて属性やメソッドの結果を確認
    '''

    #random_values = 1+np.random.rand(condition_indices.sum())*5
    psi.df_br.loc[condition_indices, 'demand'] *= random_values
    print(psi.df_br.tail(60)['demand'])
    psi_pickle2 = pickle.dumps(psi)
    psi_pickles =[psi_pickle,psi_pickle2]
    # マルチプロセスプールを作成（5プロセス）
    with multiprocessing.Pool(2) as pool:
        #results = [pool.apply_async(child_process, [psi_pickle]) for _ in range(2)]        
        results = [pool.apply_async(child_process, [psi_pickle]) for psi_pickle in psi_pickles]

        # 結果を取得
        result_pickles = [result.get() for result in results]

    # 結果をロード
    psi_updated_list = [pickle.loads(result_pickle) for result_pickle in result_pickles]

    # 結果を確認
    for psi_updated in psi_updated_list:
        print(psi_updated.df_br)  # 必要に応じて属性やメソッドの結果を確認


    '''
    - TO DO: Try with ExecRolling function...
    Issue -> cannot callback update
    
    '''

if __name__ == "__main__":


    test  = PSI()

    # PSIインスタンスを作成
    psi = PSI.from_weeks('2000-01-01', 200, ['A','B','C','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W'],'SGMC')
    psi.act_date = pd.to_datetime('2001-01-01')
    psi.df_br.loc[psi.df_br.index.get_level_values('date') <= psi.act_date, ['P','S','I','demand']] = 100


    # 条件に合致する行のインデックスを取得
    condition_indices = psi.df_br.index.get_level_values('date') <= psi.act_date

    # 乱数の範囲を指定（例: 0から1の間乱数）
    random_values = 1+np.random.rand(condition_indices.sum())*5

    # 'S'列に対して乱数をかける
    psi.df_br.loc[condition_indices, 'demand'] *= random_values
    psi.df_br.loc[condition_indices, 'P'] *= random_values
    psi.df_br.loc[condition_indices, 'S'] *= random_values
    psi.df_br.loc[condition_indices, 'I'] *= random_values

    #psi.predict_s()
    
    # インスタンスをピクル化
    #psi_pickle = pickle.dumps(psi)

    #random_values = 1+np.random.rand(condition_indices.sum())*5
    psi.df_br.loc[condition_indices, 'demand'] *= random_values
    print(psi.df_br.tail(60)['demand'])

    # Container save test====================================================================================================
    container = PSIContainer() 
    print(container)
    container.psi['10000001'].df_br.loc[:, 'P'] =1
    container.psi['10000001'].df_br.loc[:, 'S'] = 2
    container.psi['10000001'].df_br.loc[:, 'I'] = 3
    print(container.psi['10000001'].print())
    container.save('test.elc')
    newcontainer = PSIContainer.load('test.elc')
    print(newcontainer.psi['10000001'].print())    
    # #psi.predict_s()

    # psi_pickle = pickle.dumps(psi)
    # psi_pickle2 = pickle.dumps(psi)
    # psi_pickle3 = pickle.dumps(psi)
    # psi_pickle4 = pickle.dumps(psi)
    # psi_pickle5 = pickle.dumps(psi)


    # #random_values = 1+np.random.rand(condition_indices.sum())*5
    # #psi.df_br.loc[condition_indices, 'demand'] *= random_values
    # #print(psi.df_br.tail(60)['demand'])
    # psi2 = copy.deepcopy(psi)


    # tm = time.time()
    # psi.ExecRollingCalc(prt)
    # print(f'total time new: {time.time()-tm}')

    
    

    # print("original:==========================================================================================================")
    # print(psi2.df_br)

    # print("main    :==========================================================================================================")
    # print(psi.df_br)


    # psi_pickles =[psi_pickle,psi_pickle2,psi_pickle3,psi_pickle4,psi_pickle5]
    # # マルチプロセスプールを作成（5プロセス）
    # with multiprocessing.Pool(5) as pool:
    #     #results = [pool.apply_async(child_process, [psi_pickle]) for _ in range(2)]        
    #     results = [pool.apply_async(child_process, [psi_pickle]) for psi_pickle in psi_pickles]

    #     # 結果を取得
    #     result_pickles = [result.get() for result in results]

    # # 結果をロード
    # psi_updated_list = [pickle.loads(result_pickle) for result_pickle in result_pickles]

    # # 結果を確認
    # for psi_updated in psi_updated_list:
    #     print(f"subproc :==========================================================================================================")
    #     print(psi_updated.df_br)  # 必要に応じて属性やメソッドの結果を確認





