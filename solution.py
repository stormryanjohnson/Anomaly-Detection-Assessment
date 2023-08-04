# %%

import pandas as pd
import os, sqlite3, csv, json

# helper functions for Task 1
def create_tables(cursor: sqlite3.Cursor) -> None:
    
    try:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS time_series (
                date DATE,
                device_id INTEGER,
                series_type TEXT,
                value REAL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS meta (
                id INTEGER,
                name TEXT,
                PRIMARY KEY (id)
            )
        ''')
    
    except Exception as e:
        print(f'Error creating tables: {e}')
        
        
def list_files_in_folder(folder_path: str) -> list:
    
    try:
        file_list = []
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                file_list.append(os.path.join(root, file_name))
        return file_list
    
    except Exception as e:
        print(f'Error reading files from {folder_path} folder: {e}')


def insert_data_from_csv(cursor: sqlite3.Cursor, csv_file: str, insert_query: str) -> None:

    try:
        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                cursor.execute(insert_query, row)
    
    except Exception as e:
        print(f'Error inserting {csv_file} into table: {e}')
        
        
def insert_data_from_txt(cursor: sqlite3.Cursor, txt_file: str, insert_query: str) -> None:

    try:
        with open(txt_file, 'r') as file:
            json_objects = json.load(file)
            for json_obj in json_objects:
                cursor.execute(insert_query, (json_obj['id'], json_obj['name']))
    
    except Exception as e:
        print(f'Error inserting {txt_file} into table: {e}')
        
        
def find_device_id_from_file_name(csv_file: str) -> int:
    
    try:
        device_id = csv_file[csv_file.rfind('_')+1: csv_file.rfind('.')]
    
    except:
        print(f'Unable to find device id for {csv_file}. Setting device id to -1.')
        device_id = -1
        
    return device_id



    
# %%
if __name__ == '__main__':
    db_name = 'my_sqlite.db'

    try:
        os.remove(db_name)
    except:
        pass
    # %%

    # Task 1

    # start db connection
    db_conn = sqlite3.connect(db_name)
    cursor = db_conn.cursor()
    
    create_tables(cursor)
    
    files = list_files_in_folder('./data')
    for file_name in files:
        if file_name.endswith('.csv'):
            device_id = find_device_id_from_file_name(file_name)
            insert_query = f'INSERT INTO time_series (date, device_id, series_type, value) VALUES (?, {device_id}, "internet traffic", ?)'
            insert_data_from_csv(cursor=cursor, csv_file=file_name, insert_query=insert_query)
        elif file_name.endswith('.txt'):
            insert_query = f'INSERT INTO meta (id, name) VALUES (?, ?)'
            insert_data_from_txt(cursor=cursor, txt_file=file_name, insert_query=insert_query)
    
    # return all records in meta table
    query_meta_records = 'SELECT * FROM meta'  
    
    # total network traffic on the first day
    query_total_traffic_first_day = '''
    SELECT SUM(value) AS total_network_traffic_first_day
    FROM time_series
    WHERE date = (
    SELECT MIN(date)
    FROM time_series
    )
    '''  
    
    # top 5 devices total network traffic by device name
    query_top_5_devices = '''
    SELECT name, SUM(value) AS total_network_traffic
    FROM time_series AS ts
    LEFT JOIN meta AS m
    ON ts.device_id = m.id
    GROUP BY name
    ORDER BY total_network_traffic DESC
    LIMIT 5
    '''  
    
    df_meta = pd.read_sql_query(query_meta_records, db_conn)
    #print(df_meta.head())
    
    df_total_traffic_day_1 = pd.read_sql_query(query_total_traffic_first_day, db_conn)
    #print(df_total_traffic_day_1.head())
    
    df_top_5_devices = pd.read_sql_query(query_top_5_devices, db_conn)
    #print(df_top_5_devices)
    
    # commit and close connection
    db_conn.commit()
    db_conn.close()

    # %%

    # Task 2

    # code here

