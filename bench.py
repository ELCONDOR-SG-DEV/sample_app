# import time
# import ElCondor.ElCondor as ec
# from ElCondor.ElCondor import date_to_cy_mth_wk, child_process,update_data_multiprocess_warm
# import pandas as pd
# import numpy as np
# from datetime import  date
# import threading
# from datetime import datetime, timedelta
# from sqlalchemy import create_engine,inspect as sqlalchemy_inspect
# from sqlalchemy.sql import text
# import multiprocessing
# import pickle
# from multiprocessing import Process, Pool, Manager, Value
# from utils.data_connection import  db_data, psicontainer, db2_data,global_progress,preprocess_data, get_db_data
# from utils.login_handler import require_login



# '''
# load sample.elc
# '''
# temp_container = ec.PSIContainer.load('sample.elc')
# scenario_name = temp_container.scenario
# print(scenario_name)
# psicontainer[scenario_name] = temp_container
# #psicontainer['FY24 9F_TEST-FINANCIAL'] = ec.PSIContainer.load('sample.elc')
# '''
# end of load sample.elc
# '''



# '''
# process_batch
# '''
# # グローバル関数として定義
# def process_batch(batch, scenario):
#     global psicontainer
#     psi_pickles = [pickle.dumps(psicontainer[scenario].psi[item["PRODUCT"]]) for item in batch]
#     manager = Manager()
#     progress_list = [manager.Value('d', 0.0) for _ in psi_pickles]

#     with multiprocessing.Pool(len(batch)) as pool:
#         results = [pool.apply_async(update_data_multiprocess_warm, (psi_pickle, progress_list[i])) for i, psi_pickle in enumerate(psi_pickles)]
        
#         while any(progress.value < 100 for progress in progress_list):
#             time.sleep(1)
#             progress_values = ["{:3.0f}%".format(progress.value) for progress in progress_list]
#             print(f'Progress: {progress_values}')
#             for index, item in enumerate(batch):
#                 item["PROGRESS"] = [progress_list[index].value / 100]
#                 item["pct"] = progress_list[index].value / 100
        
#         try:
#             result_pickles = [result.get(timeout=600) for result in results]
#         except multiprocessing.TimeoutError:
#             print("A child process timed out.")
#             pool.terminate()
#             pool.join()
#             raise

#         pool.close()
#         pool.join()

#     manager.shutdown()

#     psi_updated_list = [pickle.loads(result_pickle) for result_pickle in result_pickles]
#     for psi_updated in psi_updated_list:
#         model_code = psi_updated.code
#         psicontainer[scenario].psi[model_code].copy_from(psi_updated)
# '''
# process_batch
# '''


# '''
# update_data_multiprocess_master_2
# '''
# from concurrent.futures import ThreadPoolExecutor, as_completed
# def update_data_multiprocess_master_2(item_list, scenario):
#     psicontainer[scenario].save('FY24 9F_TEST-FINANCIAL_before.elc')
#     #global disp_df
#     # バッチサイズを50に設定
#     batch_size = 50
#     #batches = [item_list[i:i + batch_size] for i in range(0, len(item_list), batch_size)]

#     batches = [item_list[max(i - batch_size, 0):i] for i in range(len(item_list), 0, -batch_size)]
#     batches.reverse()


#     # マルチスレッドで複数のプールを同時に立ち上げる
#     threads = []
#     for batch in batches:
#         t = threading.Thread(target=process_batch, args=(batch, scenario), daemon=True)
#         t.start()
#         threads.append(t)

#     # すべてのスレッドの完了を待つ
#     for t in threads:
#         t.join()

#     #newSCENARIO = psicontainer[scenario].transform_all_dataframes()                         # Get selected SCENARIO PSI from PSIcontainer, and Transform PSI format -> standard dataframe  
#     #disp_df = disp_df[disp_df['SCENARIO_EVENT'] != scenario]        # Delete SELECTED scenario from display dataframe 
#     #disp_df = pd.concat([disp_df,newSCENARIO], ignore_index=True)           # append calculated SCENARIO dataframe into display dataframe
#     psicontainer[scenario].save('FY24 9F_TEST-FINANCIAL_after.elc')
# '''
# end of update_data_multiprocess_master_2
# '''

# # start time
# start_time = time.time() 


# modal_data = [{"PRODUCT": prd, "pct": 0, "PROGRESS": 0} for prd in psicontainer[scenario_name].product_list]
# update_data_multiprocess_master_2(modal_data, scenario_name)

# # end time
# end_time = time.time()  
# elapsed_time = end_time - start_time
# print(f"update_data_multiprocess_master_2 completed in {elapsed_time:.2f} seconds.")









import time
import ElCondor.ElCondor as ec
from ElCondor.ElCondor import date_to_cy_mth_wk, child_process,update_data_multiprocess_warm
import pandas as pd
import numpy as np
from datetime import  date
import threading
from datetime import datetime, timedelta
from sqlalchemy import create_engine,inspect as sqlalchemy_inspect
from sqlalchemy.sql import text
import multiprocessing
import pickle
from multiprocessing import Process, Pool, Manager, Value, freeze_support
from utils.data_connection import  db_data, psicontainer, db2_data,global_progress,preprocess_data, get_db_data
from utils.login_handler import require_login


# Define your main function where all execution will happen
def main():
    '''
    load sample.elc
    '''
    temp_container = ec.PSIContainer.load('sample.elc')
    scenario_name = temp_container.scenario
    print(scenario_name)
    psicontainer[scenario_name] = temp_container
    #psicontainer['FY24 9F_TEST-FINANCIAL'] = ec.PSIContainer.load('sample.elc')
    '''
    end of load sample.elc
    '''

    '''
    process_batch
    '''
    # グローバル関数として定義
    def process_batch(batch, scenario):
        global psicontainer
        psi_pickles = [pickle.dumps(psicontainer[scenario].psi[item["PRODUCT"]]) for item in batch]
        manager = Manager()
        progress_list = [manager.Value('d', 0.0) for _ in psi_pickles]

        with multiprocessing.Pool(len(batch)) as pool:
            results = [pool.apply_async(update_data_multiprocess_warm, (psi_pickle, progress_list[i])) for i, psi_pickle in enumerate(psi_pickles)]
            
            while any(progress.value < 100 for progress in progress_list):
                time.sleep(1)
                progress_values = ["{:3.0f}%".format(progress.value) for progress in progress_list]
                print(f'Progress: {progress_values}')
                for index, item in enumerate(batch):
                    item["PROGRESS"] = [progress_list[index].value / 100]
                    item["pct"] = progress_list[index].value / 100
            
            try:
                result_pickles = [result.get(timeout=600) for result in results]
            except multiprocessing.TimeoutError:
                print("A child process timed out.")
                pool.terminate()
                pool.join()
                raise

            pool.close()
            pool.join()

        manager.shutdown()

        psi_updated_list = [pickle.loads(result_pickle) for result_pickle in result_pickles]
        for psi_updated in psi_updated_list:
            model_code = psi_updated.code
            psicontainer[scenario].psi[model_code].copy_from(psi_updated)
    '''
    process_batch
    '''

    '''
    update_data_multiprocess_master_2
    '''
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def update_data_multiprocess_master_2(item_list, scenario):
        psicontainer[scenario].save('FY24 9F_TEST-FINANCIAL_before.elc')
        # バッチサイズを50に設定
        batch_size = 50
        #batches = [item_list[i:i + batch_size] for i in range(0, len(item_list), batch_size)]

        batches = [item_list[max(i - batch_size, 0):i] for i in range(len(item_list), 0, -batch_size)]
        batches.reverse()

        # マルチスレッドで複数のプールを同時に立ち上げる
        threads = []
        for batch in batches:
            t = threading.Thread(target=process_batch, args=(batch, scenario), daemon=True)
            t.start()
            threads.append(t)

        # すべてのスレッドの完了を待つ
        for t in threads:
            t.join()

        psicontainer[scenario].save('FY24 9F_TEST-FINANCIAL_after.elc')
    '''
    end of update_data_multiprocess_master_2
    '''

    # start time
    start_time = time.time() 

    modal_data = [{"PRODUCT": prd, "pct": 0, "PROGRESS": 0} for prd in psicontainer[scenario_name].product_list]
    update_data_multiprocess_master_2(modal_data, scenario_name)

    # end time
    end_time = time.time()  
    elapsed_time = end_time - start_time
    print(f"update_data_multiprocess_master_2 completed in {elapsed_time:.2f} seconds.")

# Ensure freeze_support() is called when the main function is actually going to be run
if __name__ == '__main__':
    freeze_support()
    main()
