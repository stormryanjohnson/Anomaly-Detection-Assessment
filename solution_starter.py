# %%

import pandas as pd
import os, sqlite3

# %%

db_name = 'my_sqlite.db'

try:
    os.remove(db_name)
except:
    pass

# %%

# Task 1

# start db connection
db_conn = sqlite3.connect(db_name)

# code here

# finally close connection
db_conn.close()

# %%

# Task 2

# code here

