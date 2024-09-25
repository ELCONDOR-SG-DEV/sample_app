import platform
import psutil
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
import subprocess


def get_cpu_name_windows():
    try:
        # Use WMIC command on Windows
        result = subprocess.run(['wmic', 'cpu', 'get', 'name'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout:
            lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            if len(lines) > 1:
                return lines[1]  # Assuming the CPU name is on the second line
    except Exception as e:
        print(f"Failed to get CPU name (Windows): {e}")
    return "CPU name not available"

def get_cpu_name_linux():
    try:
        # Use /proc/cpuinfo on Linux
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if "model name" in line:
                    return line.split(':')[1].strip()
    except Exception as e:
        print(f"Failed to get CPU name (Linux): {e}")
    return "CPU name not available"

def get_cpu_name():
    if platform.system() == 'Windows':
        return get_cpu_name_windows()
    elif platform.system() == 'Linux':
        return get_cpu_name_linux()
    else:
        return "Unsupported operating system"

def print_system_info():
    cpu_name = get_cpu_name()
    total_memory = psutil.virtual_memory().total / (1024 ** 3)  # Convert from bytes to GB
    print(f"Processor (CPU) Name: {cpu_name}")
    print(f"Total System Memory: {total_memory:.2f} GB")

class ResourceMonitor(threading.Thread):
    def __init__(self, interval=1):
        super().__init__()
        self.interval = interval
        self.peak_cpu = 0
        self.peak_memory = 0
        self.running = True
        # Capture initial system resources usage
        self.initial_cpu = psutil.cpu_percent(interval=None)
        self.initial_memory = psutil.virtual_memory().used / (1024 ** 2)  # Initial memory in MB

    def run(self):
        while self.running:
            current_cpu = psutil.cpu_percent(interval=1)
            current_memory = psutil.virtual_memory().used / (1024 ** 2)  # Current memory usage in MB

            # Update peak values if current usage is higher than previously recorded peak
            if current_cpu > self.peak_cpu:
                self.peak_cpu = current_cpu
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory

            time.sleep(self.interval)

    def stop(self):
        self.running = False

    def get_peak_usage(self):
        return self.peak_cpu, self.peak_memory

    def calculate_additional_load(self):
        peak_cpu, peak_memory = self.get_peak_usage()
        additional_cpu_load = peak_cpu - self.initial_cpu
        additional_memory_load = peak_memory - self.initial_memory
        return additional_cpu_load, additional_memory_load

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

def update_data_multiprocess_master_2(item_list, scenario):
    psicontainer[scenario].save('FY24_9F_TEST-FINANCIAL_mp_MDR_before.elc')
    batch_size = 50
    batches = [item_list[max(i - batch_size, 0):i] for i in range(len(item_list), 0, -batch_size)]
    batches.reverse()

    threads = []
    for batch in batches:
        t = threading.Thread(target=process_batch, args=(batch, scenario), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    psicontainer[scenario].save('FY24_9F_TEST-FINANCIAL_mp_MDR_after.elc')

def main():
    monitor = ResourceMonitor()
    monitor.start()

    temp_container = ec.PSIContainer.load('sample_mdr.elc')
    scenario_name = temp_container.scenario
    print(scenario_name)
    psicontainer[scenario_name] = temp_container

    start_time = time.time()

    modal_data = [{"PRODUCT": prd, "pct": 0, "PROGRESS": 0} for prd in psicontainer[scenario_name].product_list]
    update_data_multiprocess_master_2(modal_data, scenario_name)

    monitor.stop()
    monitor.join()
    additional_cpu_load, additional_memory_load = monitor.calculate_additional_load()
    print(f"Additional CPU Load: {additional_cpu_load:.2f}%")
    print(f"Additional Memory Load: {additional_memory_load:.2f} MB")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"update_data_multiprocess_master_2 completed in {elapsed_time:.2f} seconds.")
    print_system_info()

if __name__ == '__main__':
    freeze_support()
    main()