# This processing pipeline is for gettign the before, during, after rainfall variables WITHOUT the per municipality enter and exit times, instead using LANDFALL only
import os
import pandas as pd
import datetime
import math
import time

os.chdir("/home/jovyan/work/Typhoon_IBF_Rice_Damage_Model/")
cdir = os.getcwd()

# Load expanded metadata
typhoon_metadata_filename = os.path.join(
    cdir, "IBF_typhoon_model/data/rainfall_data/input/expanded_metadata_typhoons.csv"
)
typhoon_metadata = pd.read_csv(typhoon_metadata_filename, delimiter=",")
typhoon_dict = dict(zip(typhoon_metadata['typhoon'], typhoon_metadata['SID']))

# Load entry and exit times for each municipality
entry_exit_filename = os.path.join(
    cdir, "IBF_typhoon_model/data/gis_data/typhoon_enter_exit_per_municipality.csv"
)
entry_exit_data = pd.read_csv(entry_exit_filename, delimiter=",")


entry_exit_data = entry_exit_data[entry_exit_data['SID'] != '2020355N10115']

# Processing the data
pre_hours = 75  # hours before landfall
t3_hours = 3   # hours before landfall
during_hours = 24  # hours after landfall
post_hours = 72  # hours after landfall
window_size_6h = 6  # for 6-hour rolling window (6 hours)
window_size_24h = 24  # for 24-hour rolling window (24 hours)

# Initialize an empty list to collect result DataFrames
results = []

# Group entry and exit data by SID to process each typhoon once
grouped_entry_exit_data = entry_exit_data.groupby('SID')

# Iterate through each unique typhoon (SID)
for sid, group in grouped_entry_exit_data:
    # Start timing
    start_time = time.time()

    # Find the typhoon name corresponding to SID
    typhoon_name = [k for k, v in typhoon_dict.items() if v == sid][0]
    
    # Get landfall date and time
    landfall_info = typhoon_metadata[typhoon_metadata['SID'] == sid]
    landfall_date = landfall_info['landfalldate'].values[0]
    landfall_time = landfall_info['landfall_time'].values[0]
    landfall_datetime = datetime.datetime.strptime(f"{landfall_date} {landfall_time}", '%d/%m/%Y %H:%M:%S')

    # Define the periods based on landfall time
    pre_start = landfall_datetime - datetime.timedelta(hours=pre_hours)
    pre_end = landfall_datetime - datetime.timedelta(hours=t3_hours)
    during_start = pre_end
    during_end = landfall_datetime + datetime.timedelta(hours=during_hours)
    post_start = during_end
    post_end = during_end + datetime.timedelta(hours=post_hours)

    # Load the rainfall data for the typhoon
    file_name = f"IBF_typhoon_model/data/rainfall_data/output_hhr/{typhoon_name}_matrix.csv"
    path = os.path.join(cdir, file_name)
    df_rainfall = pd.read_csv(path)

    # Convert column names to date format
    for col in df_rainfall.columns[1:]:
        date_format = datetime.datetime.strptime(col, "%Y%m%d-S%H%M%S")
        df_rainfall = df_rainfall.rename(columns={col: date_format})

    # Iterate through each municipality in the current typhoon group
    for index, row in group.iterrows():
        mun_code = row['ADM3_PCODE']
        
        df_mun_rainfall = df_rainfall[df_rainfall['ADM3_PCODE'] == mun_code]
        df_mun_rainfall = df_mun_rainfall.set_index('ADM3_PCODE')
        df_mun_rainfall = df_mun_rainfall.T  # Transpose for easier date filtering

        # Pre-Typhoon Period Calculation
        pre_rainfall = df_mun_rainfall[(df_mun_rainfall.index >= pre_start) & (df_mun_rainfall.index < pre_end)]
        lf_pre_max_6h = pre_rainfall.rolling(window=window_size_6h*2).sum().max().values[0]
        lf_pre_max_24h = pre_rainfall.rolling(window=window_size_24h*2).sum().max().values[0]

        # During Typhoon Period Calculation
        during_rainfall = df_mun_rainfall[(df_mun_rainfall.index >= during_start) & (df_mun_rainfall.index <= during_end)]
        lf_during_total_rainfall = during_rainfall.sum().values[0]
        lf_during_max_rainfall = during_rainfall.max().values[0]
        lf_during_mean_rainfall = during_rainfall.mean().values[0]
        lf_during_max_1h_intensity = during_rainfall.rolling(window=2).sum().max().values[0]  # max 1-hour intensity
        lf_during_max_3h_intensity = during_rainfall.rolling(window=6).sum().max().values[0]  # max 3-hour intensity
        lf_during_max_6h_intensity = during_rainfall.rolling(window=12).sum().max().values[0]  # max 6-hour intensity

        # Post-Typhoon Period Calculation
        post_rainfall = df_mun_rainfall[(df_mun_rainfall.index > post_start) & (df_mun_rainfall.index <= post_end)]
        lf_post_max_6h = post_rainfall.rolling(window=window_size_6h*2).sum().max().values[0]
        lf_post_max_24h = post_rainfall.rolling(window=window_size_24h*2).sum().max().values[0]

        # Collect the results in a DataFrame
        result = pd.DataFrame([{
            'typhoon': typhoon_name,
            'mun_code': mun_code,
            'lf_pre_max_6h': lf_pre_max_6h,
            'lf_pre_max_24h': lf_pre_max_24h,
            'lf_during_total_rainfall': lf_during_total_rainfall,
            'lf_during_max_rainfall': lf_during_max_rainfall,
            'lf_during_mean_rainfall': lf_during_mean_rainfall,
            'lf_during_max_1h_intensity': lf_during_max_1h_intensity,
            'lf_during_max_3h_intensity': lf_during_max_3h_intensity,
            'lf_during_max_6h_intensity': lf_during_max_6h_intensity,
            'lf_post_max_6h': lf_post_max_6h,
            'lf_post_max_24h': lf_post_max_24h
        }])
        results.append(result)

    # End timing
    end_time = time.time()
    duration = end_time - start_time
    print(f"Processing time for typhoon {typhoon_name}: {duration:.2f} seconds")

# Concatenate all the results into the final DataFrame
df_rainfall_final = pd.concat(results, ignore_index=True)

# Save the final dataframe to a CSV file
file_path = "IBF_typhoon_model/data/rainfall_data/lf_rainfall_variables.csv"
df_rainfall_final.to_csv(file_path, index=False)
