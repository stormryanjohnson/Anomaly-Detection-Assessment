# %%

import pandas as pd
import os, sqlite3, csv, json, logging, inspect
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf, pacf
from scipy import stats
import numpy as np


##################################################################################################
#                                       HELPER FUNCTIONS
##################################################################################################


def setup_logger(log_folder: str) -> logging.Logger:
    """
    Set up a custom logger with both file and console handlers.

    This function creates a logging folder (if it doesn't exist) and configures a logger named 'my_logger' with two handlers:
    - A FileHandler to save logs to a file named 'solution.log' in the specified logging folder.
    - A StreamHandler to print logs to the console.

    The logger is configured to log messages with a logging level of DEBUG, while the file handler has a logging level
    of DEBUG, and the console handler has a logging level of INFO.

    Parameters:
    ----------
    log_folder : str
        The path to the logging folder where the log file will be saved.

    Returns:
    -------
    logging.Logger
        A customized Logger object that can be used to log messages to the file and console.

    Example:
    -------
    >>> logger = setup_logger('logging_folder')
    >>> logger.debug('This is a debug message.')
    >>> logger.info('This is an info message.')
    >>> logger.warning('This is a warning message.')
    >>> logger.error('This is an error message.')
    >>> logger.critical('This is a critical message.')
    """
    os.makedirs(log_folder, exist_ok=True)
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    log_file = os.path.join(log_folder, 'solution.log')
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def current_function_name() -> str:
    """
    Get the name of the currently executing function.

    This function utilizes the inspect module to introspect the call stack and retrieve the name of the function that
    called it.

    Returns:
    -------
    str
        The name of the currently executing function.

    Example:
    -------
    def example_function():
        return current_function_name()

    >>> example_function()
    'example_function'
    """
    return inspect.currentframe().f_back.f_code.co_name


# HELPER FUNCTIONS FOR TASK 1

def create_tables(cursor: sqlite3.Cursor) -> None:
    """
    Create necessary tables in the SQLite database if they don't already exist.

    This function takes an SQLite cursor as input and executes SQL queries to create two tables:
    - 'time_series': A table to store time series data with columns 'date' (DATE), 'device_id' (INTEGER),
                      'series_type' (TEXT), and 'value' (REAL).
    - 'meta': A table to store metadata with columns 'id' (INTEGER) and 'name' (TEXT). The 'id' column is set as
              the primary key.

    Parameters:
    ----------
    cursor : sqlite3.Cursor
        The cursor object connected to the SQLite database.

    Returns:
    -------
    None

    Raises:
    ------
    Exception
        If there is an error during the table creation process.

    Example:
    -------
    >>> connection = sqlite3.connect('example.db')
    >>> cursor = connection.cursor()
    >>> create_tables(cursor)
    >>> connection.close()
    """
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
        logger.error(f'{current_function_name()}: {e}')
        raise e
        

def list_files_in_folder(folder_path: str) -> list:
    """
    List all files in a given folder and its subfolders.

    This function takes a folder path as input and recursively traverses through the directory to retrieve a list of
    all file paths within the folder and its subfolders.

    Parameters:
    ----------
    folder_path : str
        The path of the folder for which the files need to be listed.

    Returns:
    -------
    list
        A list containing the file paths of all files found in the specified folder and its subfolders.

    Raises:
    ------
    Exception
        If there is an error while listing the files.

    Example:
    -------
    >>> folder_path = '/path/to/folder'
    >>> files = list_files_in_folder(folder_path)
    >>> for file_path in files:
    ...     print(file_path)
    '/path/to/folder/file1.txt'
    '/path/to/folder/subfolder/file2.txt'
    '/path/to/folder/subfolder/file3.jpg'
    '/path/to/folder/file4.png'
    ...
    """
    try:
        file_list = []
        for root, _, files in os.walk(folder_path):
            file_list.extend(os.path.join(root, file_name) for file_name in files)
            
        return file_list
    
    except Exception as e:
        logger.error(f'{current_function_name()}: {e}')
        raise e
    
    
def insert_files_into_tables(files: list) -> None:
    """
    Insert data from files into the appropriate tables in the database.

    This function takes a list of file paths as input and inserts data from each file into the appropriate tables in the
    database based on the file extension.
    
    For files with the '.csv' extension, the data is assumed to be time series data, and the function uses the
    'find_device_id_from_file_name' function to retrieve the device_id from the file name. The data is then inserted into
    the 'time_series' table with the columns 'date', 'device_id', 'series_type' set to 'internet traffic', and 'value'
    from the CSV file.
    
    For files with the '.txt' extension, the data is assumed to be metadata, and the data is inserted into the 'meta'
    table with columns 'id' and 'name' from the TXT file.
    
    Parameters:
    ----------
    files : list
        A list of file paths containing data to be inserted into the database.

    Returns:
    -------
    None

    Raises:
    ------
    Exception
        If there is an error during the data insertion process.

    Example:
    -------
    >>> files = ['/path/to/data.csv', '/path/to/metadata.txt']
    >>> insert_files_into_tables(files)
    """
    try:
        for file_name in files:
            if file_name.endswith('.csv'):
                device_id = find_device_id_from_file_name(file_name)
                insert_query = f'INSERT INTO time_series (date, device_id, series_type, value) VALUES \
                                (?, {device_id}, "internet traffic", ?)'
                insert_data_from_csv(cursor=cursor, csv_file=file_name, insert_query=insert_query)
            elif file_name.endswith('.txt'):
                insert_query = 'INSERT INTO meta (id, name) VALUES (?, ?)'
                insert_data_from_txt(cursor=cursor, txt_file=file_name, insert_query=insert_query)
                
    except Exception as e:
        logger.error(f'{current_function_name()}: {e}')
        raise e


def insert_data_from_csv(cursor: sqlite3.Cursor, csv_file: str, insert_query: str) -> None:
    """
    Insert data from a CSV file into an SQLite database using a provided insert query.

    This function takes an SQLite cursor, the path of a CSV file, and an insert query as input. It reads the data from
    the CSV file and inserts each row into the database using the provided insert query.

    Parameters:
    ----------
    cursor : sqlite3.Cursor
        The cursor object connected to the SQLite database.
    csv_file : str
        The path of the CSV file containing the data to be inserted into the database.
    insert_query : str
        The SQL insert query to be used for inserting data from the CSV file into the database.

    Returns:
    -------
    None

    Raises:
    ------
    Exception
        If there is an error during the data insertion process.

    Example:
    -------
    >>> connection = sqlite3.connect('example.db')
    >>> cursor = connection.cursor()
    >>> csv_file = '/path/to/data.csv'
    >>> insert_query = 'INSERT INTO time_series (date, device_id, series_type, value) VALUES (?, ?, ?, ?)'
    >>> insert_data_from_csv(cursor, csv_file, insert_query)
    >>> connection.commit()
    >>> connection.close()
    """
    try:
        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                cursor.execute(insert_query, row)
                
    except Exception as e:
        logger.error(f'{current_function_name()}: {e}')
        raise e
        
        
def insert_data_from_txt(cursor: sqlite3.Cursor, txt_file: str, insert_query: str) -> None:
    """
    Insert data from a JSON-formatted text file into an SQLite database using a provided insert query.

    This function takes an SQLite cursor, the path of a text file in JSON format, and an insert query as input. It reads
    the JSON objects from the text file and inserts each object's 'id' and 'name' fields into the database using the
    provided insert query.

    Parameters:
    ----------
    cursor : sqlite3.Cursor
        The cursor object connected to the SQLite database.
    txt_file : str
        The path of the text file in JSON format containing the data to be inserted into the database.
    insert_query : str
        The SQL insert query to be used for inserting data from the JSON objects into the database.

    Returns:
    -------
    None

    Raises:
    ------
    Exception
        If there is an error during the data insertion process.

    Example:
    -------
    >>> connection = sqlite3.connect('example.db')
    >>> cursor = connection.cursor()
    >>> txt_file = '/path/to/data.txt'
    >>> insert_query = 'INSERT INTO meta (id, name) VALUES (?, ?)'
    >>> insert_data_from_txt(cursor, txt_file, insert_query)
    >>> connection.commit()
    >>> connection.close()
    """
    try:
        with open(txt_file, 'r') as file:
            json_objects = json.load(file)
            for json_obj in json_objects:
                cursor.execute(insert_query, (json_obj['id'], json_obj['name']))
    except Exception as e:
        logger.error(f'{current_function_name()}: {e}')
        raise e
        
        
def find_device_id_from_file_name(csv_file: str) -> int:
    """
    Extract the device ID from a CSV file name.

    This function takes a CSV file name as input and extracts the device ID from the file name based on the following
    naming convention:
    - The file name is expected to have the format '..._deviceID.csv', where 'deviceID' represents the numerical ID of
      the device, and '...' can be any other text.

    Parameters:
    ----------
    csv_file : str
        The CSV file name from which the device ID needs to be extracted.

    Returns:
    -------
    int
        The device ID extracted from the CSV file name as an integer.

    Raises:
    ------
    ValueError
        If the device ID cannot be extracted due to an incorrect file name format.

    Example:
    -------
    >>> file_name = 'data_device123.csv'
    >>> find_device_id_from_file_name(file_name)
    123

    >>> file_name = 'some_file.csv'
    >>> find_device_id_from_file_name(file_name)
    Traceback (most recent call last):
        ...
    ValueError: Invalid file name format. Unable to extract device ID.
    """
    try:
        return int(csv_file[csv_file.rfind('_')+1: csv_file.rfind('.')])
        
        
    except Exception as e:
        logger.error(f'{current_function_name()}: {e}')
        raise ValueError('Invalid file name format. Unable to extract device ID.') from e


# HELPER FUNCTIONS FOR TASK 2

def generate_dataframes_from_files(files: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate DataFrames from CSV and TXT files.

    Args:
        files: A list of file paths.

    Returns:
        A tuple of two DataFrames, one for the CSV files and one for the TXT files.
    """
    try:   
        all_dataframes_csv = []
        all_dataframes_txt = []
        for file in files:
            if file.endswith('.csv'):
                device_id = find_device_id_from_file_name(file)
                df = pd.read_csv(file)
                df['device_id'] = device_id
                all_dataframes_csv.append(df)
            elif file.endswith('.txt'):
                with open(file, 'r') as file_name:
                    lines = file_name.readlines()
                json_objects = [json.loads(line.strip()) for line in lines][0]
                df = pd.DataFrame(json_objects)
                all_dataframes_txt.append(df)
        combined_dataframe_csv = pd.concat(all_dataframes_csv, ignore_index=True)
        combined_dataframe_csv['device_id'] = combined_dataframe_csv['device_id'].astype(int)
        combined_dataframe_txt = pd.concat(all_dataframes_txt, ignore_index=True)
        combined_dataframe_txt['id'] = combined_dataframe_txt['id'].astype(int)
        
        return combined_dataframe_txt, combined_dataframe_csv    
    
    except Exception as e:
        logger.error(f'{current_function_name()}: {e}')
        raise e
        
        
def check_odd_dates(time_series_df: pd.DataFrame, oddities: set, device_id_set: set) -> set:
    try:
        test_failure_list = []
        unique_dates = set(time_series_df.date)
        for device_id in device_id_set:
            device_df = time_series_df[time_series_df.device_id == device_id]
            device_dates = set(device_df.date)
            # checking for missing or duplicate dates respectively
            if device_dates != unique_dates or len(device_dates) != len(device_df.date):  
                oddities.add(int(device_id))
                test_failure_list.append(device_id)
    
        return oddities, test_failure_list
    
    except Exception as e:
        logger.error(f'{current_function_name()}: {e}')
        raise e
    
    
def get_set_of_device_id(files: list) -> set:
    try:
        
        return {
            find_device_id_from_file_name(file) for file in files if file.endswith('.csv')
            }
            
    except Exception as e:
        logger.error(f'{current_function_name()}: {e}')
        raise e


def check_negative_traffic(time_series_df: pd.DataFrame, oddities: set) -> set:
    try:
        test_failure_list = []
        negative_network_df = time_series_df[time_series_df.traffic < 0]
        for _, row in negative_network_df.iterrows():
            oddities.add(int(row.device_id))
            test_failure_list.append(int(row.device_id))
            
        return oddities, test_failure_list
    
    except Exception as e:
        logger.error(f'{current_function_name()}: {e}')
        raise e


def check_outliers(time_series_df: pd.DataFrame, oddities: set, device_id_set: set, threshold: float) -> set:
    try:
        test_failure_list = []
        for device_id in device_id_set:
            device_traffic = list(time_series_df[time_series_df.device_id == device_id].traffic)
            if find_outliers_iqr(device_traffic, threshold):
                oddities.add(device_id)
                test_failure_list.append(device_id)
                
        return oddities, test_failure_list
    
    except Exception as e:
        logger.error(f'{current_function_name()}: {e}')
        raise e


def check_means(time_series_df: pd.DataFrame, oddities: set, device_id_set: set, threshold: float) -> set:
    try:
        device_traffic_mean_list = []
        device_id_list = []
        test_failure_list = []
        # find the outlier mean values relative to means of all devices
        for device_id in device_id_set:
            device_traffic_mean = np.mean(time_series_df[time_series_df.device_id == device_id].traffic)
            device_traffic_mean_list.append(device_traffic_mean)
            device_id_list.append(device_id)
        outliers = find_outliers_iqr(device_traffic_mean_list, threshold)
        # find device_id for all outliers and to oddities
        for outlier in outliers:
            positions = find_positions(device_traffic_mean_list, outlier)
            for position in positions:
                oddities.add(device_id_list[position])
                test_failure_list.append(device_id_list[position])
                
        return oddities, test_failure_list
    
    except Exception as e:
        logger.error(f'{current_function_name()}: {e}')
        raise e


def check_variances(time_series_df: pd.DataFrame, oddities: set, device_id_set: set, threshold: float) -> set:
    try:
        device_traffic_var_list = []
        device_id_list = []
        test_failure_list = []
        # find the outlier variance values relative to variancess of all devices
        for device_id in device_id_set:
            device_traffic_var = np.var(time_series_df[time_series_df.device_id == device_id].traffic)
            device_traffic_var_list.append(device_traffic_var)
            device_id_list.append(device_id)
        outliers = find_outliers_iqr(device_traffic_var_list, threshold)
        # find device_id for all outliers and to oddities
        for outlier in outliers:
            positions = find_positions(device_traffic_var_list, outlier)
            for position in positions:
                oddities.add(device_id_list[position])
                test_failure_list.append(device_id_list[position])
                
        return oddities, test_failure_list
    
    except Exception as e:
        logger.error(f'{current_function_name()}: {e}')
        raise e


def check_autocorrelation(time_series_df: pd.DataFrame, oddities: set, device_id_set: set, 
                          lag_max: int, acf_threshold: float) -> set:
    try:
        test_failure_list = []
        for device_id in device_id_set:
            acf_values = acf(time_series_df[time_series_df.device_id == device_id].traffic, nlags=lag_max)
            for acf_value in acf_values[1:]:
                if np.absolute(acf_value) > acf_threshold:
                    oddities.add(device_id)
                    test_failure_list.append(device_id)
        test_failure_list = list(set(test_failure_list))
        
        return oddities, test_failure_list
    
    except Exception as e:
        logger.error(f'{current_function_name()}: {e}')
        raise e


def find_outliers_iqr(data: list, threshold: float) -> list: 
    """
    Find significant outliers in a list using the IQR (Interquartile Range) method.

    Parameters:
        data (list): A list of numerical values.
        threshold (float): IQR threshold for outlier detection (default=1.5).

    Returns:
        outliers (list): A list containing the significant outlier values.
    """
    try:
        data_array = np.array(data)
        q1 = np.percentile(data_array, 25)
        q3 = np.percentile(data_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = data_array[(data_array < lower_bound) | (data_array > upper_bound)]
        
        return outliers.tolist()

    except Exception as e:
        logger.error(f'{current_function_name()}: {e}')
        raise e

def check_normal_distribution(time_series_df: pd.DataFrame, oddities: set, device_id_set: set, alpha: float) -> set:
    try:
        test_failure_list = []
        for device_id in device_id_set:
            device_traffic = time_series_df[time_series_df.device_id == device_id].traffic
            if not is_normal_distribution(device_traffic, alpha):
                oddities.add(device_id)
                test_failure_list.append(device_id)
                
        return oddities, test_failure_list
    
    except Exception as e:
        logger.error(f'{current_function_name()}: {e}')
        raise e


def is_normal_distribution(data: list, alpha: float) -> bool:
    """
    Check if a list of data is normally distributed using the Shapiro-Wilk test.

    Parameters:
        data (list): A list of numerical values.
        alpha (float): The significance level 

    Returns:
        bool: True if the data is normally distributed, False otherwise.
    """
    _, p_value = stats.shapiro(data)
    return p_value > alpha


def find_positions(input_list: list, element) -> list:
    """
    Find all positions of a given element in a list.

    This function takes a list and a target element as input and returns a list of all positions (indices)
    where the target element is found in the input list.

    Parameters:
    ----------
    input_list : list
        The input list where we want to find the positions of the element.
    element : any
        The target element whose positions need to be found in the input list.

    Returns:
    -------
    list
        A list containing all the positions (indices) of the target element in the input list.

    Examples:
    --------
    >>> input_list = [1, 2, 3, 2, 4, 2, 5]
    >>> find_positions(input_list, 2)
    [1, 3, 5]

    >>> input_list = [7, 7, 7, 7, 7]
    >>> find_positions(input_list, 7)
    [0, 1, 2, 3, 4]

    >>> input_list = [1, 2, 3, 4, 5]
    >>> find_positions(input_list, 6)
    []
    """
    try:
        
        return [
            i for i, item in enumerate(input_list) if item == element
            ]

    except Exception as e:
        logger.error(f'{current_function_name()}: {e}')
        raise e


def create_oddities_dictionary(oddities: set, time_series_df: pd.DataFrame, meta_df: pd.DataFrame) -> dict:
    try:
        oddities_dict = {}
        for device_id in oddities:
            traffic = list(time_series_df[time_series_df.device_id == device_id].traffic)
            date = list(time_series_df[time_series_df.device_id == device_id].date)
            name = list(meta_df[meta_df.id == device_id].name)
            name = f'Device ID {device_id}: {name[0]}'
            oddities_dict[name] = [date, traffic]
            
        return oddities_dict
    
    except Exception as e:
        logger.error(f'{current_function_name()}: {e}')
        raise e
    
    
def create_oddities_dataframe(oddities: set, meta_df: pd.DataFrame) -> pd.DataFrame:    
    try:
        oddity_df_list = []
        for device_id in oddities:
            df = meta_df[meta_df.id == device_id].dropna()
            oddity_df_list.append(df)

        return pd.concat(oddity_df_list, ignore_index=True) if oddity_df_list else pd.DataFrame()
    
    except Exception as e:
        logger.error(f'{current_function_name()}: {e}')
        raise e


def plot_multiple_scatter(data_dict: dict) -> None:
    """
    Plot multiple scatter plots on a single figure with each plot on its own set of axes.

    Parameters:
        data_dict (dict): A dictionary where each key represents the title of the scatter plot,
                          and each value is a list containing the x and y values.
                          Example: {"Plot 1": [[x1, y1], "Plot 2": [[x2, y2]], ...}

    Returns:
        None (displays the plot)
    """
    try:
        num_plots = len(data_dict)
        rows = num_plots // 2 + num_plots % 2
        cols = min(2, num_plots)

        # Create a new figure with subplots
        fig, axs = plt.subplots(rows, cols, figsize=(10, 6))

        # Flatten the axs array if only one row or column of plots
        axs = np.ravel(axs)

        # Plot each scatter plot from the data_dict on its own set of axes
        for i, (title, data) in enumerate(data_dict.items()):
            x, y = data
            axs[i].scatter(x, y)
            subset_x_values = x[::90]  # Show every 90th value
            axs[i].set_xticks(subset_x_values)
            axs[i].set_ylabel('Traffic (MB)')
            axs[i].set_title(title)
            if i in [len(data_dict) - 1, len(data_dict)-2]:
                axs[i].set_xlabel('Date')
        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        logger.error(f'{current_function_name()}: {e}')
        raise e


def write_dataframe_to_json(dataframe: pd.DataFrame, file_name: str) -> None:
    try:
        json_data = dataframe.to_json(orient='records', indent=4)
        with open(file_name, 'w') as json_file:
            json_file.write(json_data)
    
    except Exception as e:
        logger.error(f'{current_function_name()}: {e}')
        raise e
    
    
# %%

##################################################################################################
#                                       MAIN
##################################################################################################

if __name__ == '__main__':
    db_name = 'my_sqlite.db'

    try:
        os.remove(db_name)
    except Exception:
        pass
    
    # initiate logger for monitoring and maintainability
    log_folder = 'logging'
    logger = setup_logger(log_folder)
    
    # %%

    # Task 1

    # start db connection
    db_conn = sqlite3.connect(db_name)
    cursor = db_conn.cursor()
    
    # create the tables in the sql database
    create_tables(cursor)
    
    # obtain list of files in the data folder
    files = list_files_in_folder('./data')
    
    # insert the contents of each file into their respective table
    insert_files_into_tables(files)
    
    # return all records in meta table
    query_meta_records = 'SELECT * FROM meta'  
    df_meta = pd.read_sql_query(query_meta_records, db_conn)
    
    # total network traffic on the first day
    query_total_traffic_first_day = '''
        SELECT SUM(value) AS total_network_traffic_first_day
        FROM time_series
        WHERE date = (
        SELECT MIN(date)
        FROM time_series
        )
    '''  
    df_total_traffic_day_1 = pd.read_sql_query(query_total_traffic_first_day, db_conn)
    
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
    df_top_5_devices = pd.read_sql_query(query_top_5_devices, db_conn)

    # commit and close connection
    db_conn.commit()
    db_conn.close()

    # %%

    # Task 2
    
    '''
    Since it's stated in info.md that SQL cannot be used for this task, I will generate the 
    dataframes for time_series and meta by parsing the data files 
    directly into dataframes using pandas
    '''
    meta_df, time_series_df = generate_dataframes_from_files(files)
    
    device_id_set = get_set_of_device_id(files)
    
    '''
    We want to identify the IoT devices that exhibit odd behaviours. According to the info:
        statement 1 - A device is active for the entire year
        statement 2 - Network traffic is always non-negative
        statement 3 - Network traffic is largely predictable throughout the year
    Therefore, we can define a device as "odd" if it contradicts any of the 3 above statements. 
    In addition, we would generally expect the traffic data points to be both independent and normally distributed. 
    Therefore, it will also be meaningful to perform tests for autocorrelation and normality.
    '''
    
    # set to store odd devices
    oddities = set()
    
    '''
    Below are the various constants used as thresholds and the significance level of tests which will be conducted. Note that these values
    have been chosen in such a way to select only the most extreme (and therefore, odd) data from the devices. Naturally these values can be 
    changed as needed to adjust the sensitivity of each test.
    '''
    iqr_threshold = 6
    alpha = 0.000005
    acf_threshold = 0.25
    lag_max = 30
    
    # check statement 1: look for missing/duplicate dates in data files for each device
    oddities, odd_dates = check_odd_dates(time_series_df, oddities, device_id_set)
            
    # check statement 2: look for negative network traffic for any of the data points
    oddities, negative_traffic = check_negative_traffic(time_series_df, oddities)
    
    # check statement 3: look for significant network traffic outlier values for each device using Inter Quartile Range (IQR)
    oddities, outliers = check_outliers(time_series_df, oddities, device_id_set, iqr_threshold)
    
    # check statement 3: look for significant statistical deviations of mean from means of all datasets
    oddities, means = check_means(time_series_df, oddities, device_id_set, iqr_threshold)
    
    # check statement 3: look for significant statistical deviations of variance from variances of all datasets
    oddities, variances = check_variances(time_series_df, oddities, device_id_set, iqr_threshold)
    
    # check for autocorrelation using the Autocorrelation Function (ACF)
    oddities, autocorrelation = check_autocorrelation(time_series_df, oddities, device_id_set, lag_max, acf_threshold)
    
    # check for normality using the Shapiro-Wilk test
    oddities, normal_distribution = check_normal_distribution(time_series_df, oddities, device_id_set, alpha)
    
    # print list of devices that failed each test
    print('Failed check_odd_dates: ', odd_dates)
    print('Failed check_negative_traffic: ', negative_traffic)
    print('Failed check_outliers: ', outliers)
    print('Failed check_means: ', means)
    print('Failed check_variances: ', variances)
    print('Failed check_autocorrelation: ', autocorrelation)
    print('Failed check_normal_distribution: ', normal_distribution)
    print('All odd devices: ', list(oddities))
    
    # dynamically plot the odd devices
    oddities_dict = create_oddities_dictionary(oddities, time_series_df, meta_df)  
    plot_multiple_scatter(oddities_dict)
    
    '''
    From the test results, we see that there are 2 devices which show extremely odd behaviours: hungry_elephant (7540) and sneaking_catfish (4399). 
    There are some interesting features to note in each plot:
    
    - hungry_elephant
        - data is highly autocorrelated,
        - range of data values is non-continuous (i.e. can apparently only take integer values),
        - the mean network traffic is signifcantly smaller than the other devices,
        - there are noticeable dips near the end/beginning of months.
        Assuming that this is not erroneous data, it can be deduced that this device is on some daily network traffic cap/limit which depends on the previous 
        day's traffic utilization. It is also possible that these limits are lowered towards the end of the month.

    - sneaking_catfish
        - the highest variance dataset
        - appears to be a random uniform distribution
        This device is likely something which sporadically can use large amounts of network traffic - perhaps a motion detector or a security camera.
    '''
    
    # store odd devices in `oddities.json`
    oddity_df = create_oddities_dataframe(oddities, meta_df)
    write_dataframe_to_json(dataframe=oddity_df, file_name='oddities.json')
            

    # %% 
