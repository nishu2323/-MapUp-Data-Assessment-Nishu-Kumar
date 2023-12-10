#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np


# # Question 1: Car Matrix Generation

# # Under the function named generate_car_matrix write a logic that takes the dataset-1.csv as a DataFrame. Return a new DataFrame that follows the following rules:
# 
# values from id_2 as columns
# values from id_1 as index
# dataframe should have values from car column
# diagonal values should be 0.

# In[2]:


import pandas as pd

def generate_car_matrix(file_path="dataset-1.csv"):
    try:
        car_data = pd.read_csv(file_path)
        car_matrix = car_data.pivot(index='id_1', columns='id_2')
        print(car_matrix)
        return car_matrix

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse file '{file_path}'. Check the file format.")
        return None
car_matrix = generate_car_matrix()


# In[3]:


import pandas as pd

def generate_car_matrix(file_path="dataset-1.csv"):
    try:
        car_data = pd.read_csv(file_path)
        car_matrix = car_data.pivot_table(index='id_1', columns='id_2', values='car', fill_value=0)
        for col in car_matrix.columns:
            car_matrix.loc[car_matrix.index == col, col] = 0
        print(car_matrix)
        return car_matrix

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse file '{file_path}'. Check the file format.")
        return None
car_matrix = generate_car_matrix()


# # Question 2: Car Type Count Calculation

# # Create a Python function named get_type_count that takes the dataset-1.csv as a DataFrame. Add a new categorical column car_type based on values of the column car:
# 
# low for values less than or equal to 15,
# medium for values greater than 15 and less than or equal to 25,
# high for values greater than 25.
# Calculate the count of occurrences for each car_type category and return the result as a dictionary. Sort the dictionary alphabetically based on keys.

# In[4]:


import pandas as pd

def get_type_count(file_path="dataset-1.csv"):
    try:
        car_data = pd.read_csv(file_path)
        car_data['car_type'] = pd.cut(car_data['car'],
                                     bins=[float('-inf'), 15, 25, float('inf')],
                                     labels=['low', 'medium', 'high'],
                                     right=False)
        type_counts = car_data['car_type'].value_counts().to_dict()
        sorted_type_counts = dict(sorted(type_counts.items()))
        print(sorted_type_counts)
        return sorted_type_counts
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse file '{file_path}'. Check the file format.")
        return None
type_count_result = get_type_count()


# # Question 3: Bus Count Index Retrieval

# # Create a Python function named get_bus_indexes that takes the dataset-1.csv as a DataFrame. The function should identify and return the indices as a list (sorted in ascending order) where the bus values are greater than twice the mean value of the bus column in the DataFrame.

# In[5]:


import pandas as pd

def get_bus_indexes(file_path="dataset-1.csv"):
    try:
        car_data = pd.read_csv(file_path)
        bus_mean = car_data['bus'].mean()
        bus_indexes = car_data[car_data['bus'] > 2 * bus_mean].index.tolist()
        bus_indexes.sort()
        print(bus_indexes)
        return bus_indexes

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse file '{file_path}'. Check the file format.")
        return None
bus_indexes_result = get_bus_indexes()


# # Question 4: Route Filtering

# # Create a python function filter_routes that takes the dataset-1.csv as a DataFrame. The function should return the sorted list of values of column route for which the average of values of truck column is greater than 7.

# In[6]:


import pandas as pd

def filter_routes(file_path="dataset-1.csv"):
    try:
        car_data = pd.read_csv(file_path)
        avg_truck_by_route = car_data.groupby('route')['truck'].mean()
        selected_routes = avg_truck_by_route[avg_truck_by_route > 7].index.tolist()
        selected_routes.sort()
        print(selected_routes)
        return selected_routes
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse file '{file_path}'. Check the file format.")
        return None
filtered_routes_result = filter_routes()


# # Question 5: Matrix Value Modification

# # Create a Python function named multiply_matrix that takes the resulting DataFrame from Question 1, as input and modifies each value according to the following logic:
# 
# If a value in the DataFrame is greater than 20, multiply those values by 0.75,
# If a value is 20 or less, multiply those values by 1.25.
# The function should return the modified DataFrame which has values rounded to 1 decimal place.

# In[7]:


import pandas as pd

def multiply_matrix(car_matrix):
    try:
        modified_matrix = car_matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)
        modified_matrix = modified_matrix.round(1)
        print(modified_matrix)
        return modified_matrix
    except AttributeError:
        print("Error: Input must be a DataFrame.")
        return None
car_matrix = generate_car_matrix()
modified_matrix_result = multiply_matrix(car_matrix)


# # Question 6: Time Check
# 

# # You are given a dataset, dataset-2.csv, containing columns id, id_2, and timestamp (startDay, startTime, endDay, endTime). The goal is to verify the completeness of the time data by checking whether the timestamps for each unique (id, id_2) pair cover a full 24-hour period (from 12:00:00 AM to 11:59:59 PM) and span all 7 days of the week (from Monday to Sunday).
# 
# Create a function that accepts dataset-2.csv as a DataFrame and returns a boolean series that indicates if each (id, id_2) pair has incorrect timestamps. The boolean series must have multi-index (id, id_2).

# In[8]:


import pandas as pd

def verify_timestamp_completeness(file_path="dataset-2.csv"):
    try:
        df = pd.read_csv(file_path)
        required_columns = ['startDay', 'startTime', 'endDay', 'endTime']
        if not all(column in df.columns for column in required_columns):
            raise KeyError("Required columns are missing.")
        day_map = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2,
            'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6
        }

        df['start_timestamp'] = pd.to_datetime(
            df['startDay'].str.lower().map(day_map).astype(str) + ' ' + df['startTime'],
            format='%H:%M:%S', errors='coerce'
        )
        df['end_timestamp'] = pd.to_datetime(
            df['endDay'].str.lower().map(day_map).astype(str) + ' ' + df['endTime'],
            format='%H:%M:%S', errors='coerce'
        )
        invalid_rows = df[df['start_timestamp'].isna() | df['end_timestamp'].isna()]
        if not invalid_rows.empty:
            print("Rows with invalid timestamps:")
            print(invalid_rows)
        if df['start_timestamp'].isna().any() or df['end_timestamp'].isna().any():
            raise ValueError("Invalid timestamp values in the dataset.")
        mask = (
            (df['start_timestamp'].dt.time != pd.Timestamp('00:00:00').time()) |
            (df['end_timestamp'].dt.time != pd.Timestamp('23:59:59').time()) |
            (df['start_timestamp'].dt.weekday < 0) | (df['start_timestamp'].dt.weekday > 6) |
            (df['end_timestamp'].dt.weekday < 0) | (df['end_timestamp'].dt.weekday > 6)
        )
        result_series = df[mask].groupby(['id', 'id_2']).any().any(axis=1)
        return result_series

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse file '{file_path}'. Check the file format.")
        return None
    except KeyError as e:
        print(f"Error: {e}")
        return None
    except ValueError as e:
        print(f"Error: {e}")
        return None
result_series = verify_timestamp_completeness("dataset-2.csv")
print(result_series)


# # Python Task 2

# # Question 1: Distance Matrix Calculation

# # Create a function named calculate_distance_matrix that takes the dataset-3.csv as input and generates a DataFrame representing distances between IDs.
# 
# The resulting DataFrame should have cumulative distances along known routes, with diagonal values set to 0. If distances between toll locations A to B and B to C are known, then the distance from A to C should be the sum of these distances. Ensure the matrix is symmetric, accounting for bidirectional distances between toll locations (i.e. A to B is equal to B to A)

# In[9]:


df = pd.read_csv("dataset-3.csv")
print(df.columns)


# In[10]:


import pandas as pd

def calculate_distance_matrix(file_path="dataset-3.csv"):
    try:
        df = pd.read_csv(file_path)
        required_columns = ['id_start', 'id_end', 'distance']
        if not all(column in df.columns for column in required_columns):
            raise KeyError("Required columns are missing.")
        distance_matrix = df.pivot(index='id_start', columns='id_end', values='distance').fillna(0)
        distance_matrix = distance_matrix.add(distance_matrix.T, fill_value=0)
        distance_matrix.values[[range(len(distance_matrix))]*2] = 0

        return distance_matrix

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse file '{file_path}'. Check the file format.")
        return None
    except KeyError as e:
        print(f"Error: {e}")
        return None
distance_matrix = calculate_distance_matrix("dataset-3.csv")
print(distance_matrix)


# # Question 2: Unroll Distance Matrix

# # Create a function unroll_distance_matrix that takes the DataFrame created in Question 1. The resulting DataFrame should have three columns: columns id_start, id_end, and distance.
# 
# All the combinations except for same id_start to id_end must be present in the rows with their distance values from the input DataFrame.

# In[17]:


import pandas as pd

def unroll_distance_matrix(distance_matrix):
    try:
        if not isinstance(distance_matrix, pd.DataFrame):
            raise ValueError("Input must be a DataFrame.")
        upper_triangle = distance_matrix.where(np.triu(np.ones(distance_matrix.shape), k=1).astype(bool))
        stacked_distances = upper_triangle.stack()
        unrolled_distances = stacked_distances.reset_index()
        unrolled_distances.columns = ['id_start', 'id_end', 'distance']

        return unrolled_distances

    except ValueError as e:
        print(f"Error: {e}")
        return None
unrolled_distances = unroll_distance_matrix(distance_matrix)
print(unrolled_distances)


# In[ ]:





# In[ ]:




