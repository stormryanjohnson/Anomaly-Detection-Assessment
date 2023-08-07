# Coding Test

The purpose of the coding test is for us to determine your level of coding and ability to apply logic when presented with a problem. Attempt all but do not dwell too long on a particular question, instead work towards presenting a complete solution.

# Problem Description

You are given files:

1. info.md (this file)
2. folder "data" which contains data files and a single meta data file
3. solution_starter.py

The data contains a subset of a 1000 IoT devices' daily network traffic for a particular year together with an associated meta data file. The device locations are obfuscated in the device name. For example, IoT device 4 has device name `running_gopher` and on `2020-01-01` the total upload and download usage was 96.4 megabtyes (`MB`).

Some details about a device's network traffic are given below

- A device is active for the entire year
- Network traffic is always non-negative
- Network traffic is largely predictable throughout the year

# Deliverable

Your solution should be in a single `solution.py` script and should produce the solution to Task 1 and 2 in a single run of the script for any subsequent subset of this IoT devices. Answer any analysis questions as comments in the script and comment the code. Your script should be easily readable and maintanable.

Together with the `solution.py`, send the following files:

1. solution.py
2. my_sqlite.db
3. oddities.json

# Task 1 (SQL Test)

Use a SQLite database to store the devices network trafiic series and run some initial analysis

1. Create a SQLite database called `my_sqlite.db`
2. Write SQL to create a table called `time_series` with columns date, device_id, series_type and value which will store the numerical data.
3. Insert the network traffic series as is into the `time_series` table for each IoT device (you can use any viable series_type descriptor)
4. Write SQL to create a table called `meta` with columns id and name
5. Parse the meta data file and insert the data into the `meta` table
6. Write SQL to select all the records from the `meta` table and store it in a Pandas DataFrame called `df_meta`
7. Write SQL to find the total network traffic on the first day of the year across all the devices
8. Write SQL to find the top 5 devices total network traffic by device name

Tips and Notes:

- Use Python Modules `pandas` and `sqlite3`
- Do the insert using Python

# Task 2 (Python Test)

Do this task without using SQL. 

1. Identify the IoT devices that exhibit odd behaviour relative to the subset.
2. Plot the daily network traffic of the odd behaved IoT devices
3. What can you deduce about these oddities?
4. Output the odd behaved devices ID and name to a JSON formatted file called `oddities.json`

Tips and Notes:

- Your method of identifying odd behaved devices may be statistical or heuristic
- Consider the notes about the network traffic
