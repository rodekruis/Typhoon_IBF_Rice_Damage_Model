import geopandas as gpd
from shapely.geometry import LineString
import pandas as pd
from datetime import timedelta
import os
import time
import warnings

# Suppress SettingWithCopyWarning
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

os.chdir("/home/jovyan/work/Typhoon_IBF_Rice_Damage_Model/")
cdir = os.getcwd()

# Load the typhoon tracks
tracks_path = 'IBF_typhoon_model/data/gis_data/typhoon_tracks/tracks_filtered.shx'
typhoon_tracks = gpd.read_file(tracks_path)

# Load the municipal borders
municipalities_path = 'IBF_typhoon_model/data/phl_administrative_boundaries/phl_admbnda_adm3.shx'
municipalities = gpd.read_file(municipalities_path)

# DEBUG
#municipalities = municipalities.sample(n=10, random_state=42)
#typhoon_tracks = typhoon_tracks.query("SID == '2011205N12130'")

# Load the data overview Excel file
data_overview_path = 'IBF_typhoon_model/data/data_overview.xlsx'
data_overview = pd.read_excel(data_overview_path, sheet_name='typhoon_overview')

# Combine all municipality boundaries into a single shape
philippines_shape = municipalities.union_all()

# Convert datetime columns to actual datetime objects if not already done
typhoon_tracks['datetime'] = pd.to_datetime(typhoon_tracks[['year', 'month', 'day', 'hour']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')

# Function to fill NA values with the last available value for USA_ROCI
def fill_roci(track):
    track['USA_ROCI'] = track['USA_ROCI'].fillna(method='ffill').fillna(0)
    return track

# Function to create a circle in kilometers around a point
def create_circle_km(center_point, radius_km):
    lat_radius = radius_km / 111  # Convert km to degrees of latitude
    return center_point.buffer(lat_radius)

# Function to determine if a shape is affected by a circle
def shape_affected(shape, circle):
    return shape.intersects(circle)

# Function to determine if a municipality is affected by a circle
def municipality_affected(municipality, circle):
    return municipality.intersects(circle)

# Initialize a DataFrame with all municipalities and typhoons
results = []
for typhoon_id in typhoon_tracks['SID'].unique():
    for _, municipality in municipalities.iterrows():
        results.append({
            "SID": typhoon_id,
            "ADM3_PCODE": municipality["ADM3_PCODE"],
            "entry_date": pd.NaT,
            "exit_date": pd.NaT,
            "is_intersect": False,
            "is_third_pass": False
        })

results_df = pd.DataFrame(results)

# First pass: Determine which municipalities intersect with the circle
start_time = time.time()
for typhoon_id in typhoon_tracks['SID'].unique():
    track = typhoon_tracks[typhoon_tracks['SID'] == typhoon_id]
    track = fill_roci(track).reset_index(drop=True)

    first_contact = None
    last_contact = None

    for index, row in track.iterrows():
        segment = row.geometry
        if isinstance(segment, LineString):
            centroid = segment.centroid
            roci_nm = row['USA_ROCI']
            if roci_nm > 0:
                # Convert nautical miles to kilometers (1 NM = 1.852 km)
                roci_km = roci_nm * 1.852
                circle = create_circle_km(centroid, roci_km)
                
                if shape_affected(philippines_shape, circle):
                    if first_contact is None:
                        first_contact = index
                    last_contact = index

    if first_contact is not None and last_contact is not None:
        print(f"Typhoon {typhoon_id}: first contact at index {first_contact}, last contact at index {last_contact}")
        sliced_track = track.iloc[first_contact:last_contact+1]
        for index, row in sliced_track.iterrows():
            segment = row.geometry
            if isinstance(segment, LineString):
                centroid = segment.centroid
                roci_nm = row['USA_ROCI']
                if roci_nm > 0:
                    # Convert nautical miles to kilometers (1 NM = 1.852 km)
                    roci_km = roci_nm * 1.852
                    circle = create_circle_km(centroid, roci_km)
                    
                    for _, municipality in municipalities.iterrows():
                        if municipality_affected(municipality.geometry, circle):
                            entry_date = row['datetime']
                            condition = (results_df['SID'] == typhoon_id) & (results_df['ADM3_PCODE'] == municipality["ADM3_PCODE"])
                            
                            # Update entry date if earlier
                            if pd.isna(results_df.loc[condition, 'entry_date'].values[0]):
                                results_df.loc[condition, 'entry_date'] = entry_date
                            
                            # Always update the exit date to the current row's datetime
                            results_df.loc[condition, 'exit_date'] = entry_date
                            
                            results_df.loc[condition, 'is_intersect'] = True
    else:
        print(f"Typhoon {typhoon_id}: no contact found")

end_time = time.time()
print(f"First pass completed in {end_time - start_time:.2f} seconds")

# Adjust exit times that are the same as entry times
same_entry_exit_mask = results_df['entry_date'] == results_df['exit_date']
results_df.loc[same_entry_exit_mask, 'exit_date'] = results_df.loc[same_entry_exit_mask, 'entry_date'] + timedelta(hours=3)

# Second pass: Handle municipalities that never intersected with any circle, only if there are intersecting municipalities
start_time = time.time()
for typhoon_id in typhoon_tracks['SID'].unique():
    if results_df[(results_df['SID'] == typhoon_id) & (results_df['is_intersect'] == True)].empty:
        continue
    
    non_intersecting = results_df[(results_df['SID'] == typhoon_id) & (results_df['is_intersect'] == False)]
    
    if not non_intersecting.empty:
        print(f"Typhoon {typhoon_id}: {len(non_intersecting)} non-intersecting municipalities to process in second pass")
    
    for _, non_affected_row in non_intersecting.iterrows():
        non_affected_geom = municipalities[municipalities["ADM3_PCODE"] == non_affected_row["ADM3_PCODE"]].geometry.iloc[0]
        closest_distance = float('inf')
        closest_entry_date = None
        closest_exit_date = None
        
        for _, affected_row in results_df[(results_df['SID'] == typhoon_id) & (results_df['is_intersect'] == True)].iterrows():
            affected_geom = municipalities[municipalities["ADM3_PCODE"] == affected_row["ADM3_PCODE"]].geometry.iloc[0]
            distance = non_affected_geom.distance(affected_geom)
            if distance < closest_distance:
                closest_distance = distance
                closest_entry_date = affected_row["entry_date"]
                closest_exit_date = affected_row["exit_date"]
        
        if closest_entry_date and closest_exit_date:
            condition = (results_df['SID'] == typhoon_id) & (results_df['ADM3_PCODE'] == non_affected_row["ADM3_PCODE"])
            results_df.loc[condition, ['entry_date', 'exit_date']] = [closest_entry_date, closest_exit_date]

end_time = time.time()
print(f"Second pass completed in {end_time - start_time:.2f} seconds")

# Third pass: Handle typhoons with no intersection events
start_time = time.time()
for typhoon_id in typhoon_tracks['SID'].unique():
    intersecting = results_df[(results_df['SID'] == typhoon_id) & (results_df['is_intersect'] == True)]
    
    if intersecting.empty:
        print(f"Typhoon {typhoon_id}: no intersecting municipalities found, processing third pass")
        closest_distances = []
        track = typhoon_tracks[typhoon_tracks['SID'] == typhoon_id]
        track = fill_roci(track)
        
        for index, row in track.iterrows():
            segment = row.geometry
            if isinstance(segment, LineString):
                centroid = segment.centroid
                roci_nm = row['USA_ROCI']
                if roci_nm > 0:
                    # Convert nautical miles to kilometers (1 NM = 1.852 km)
                    roci_km = roci_nm * 1.852
                    circle = create_circle_km(centroid, roci_km)
                    
                    for _, municipality in municipalities.iterrows():
                        municipality_geom = municipality.geometry
                        distance = circle.exterior.distance(municipality_geom)
                        closest_distances.append((distance, row['datetime']))
        
        if closest_distances:
            closest_distances.sort(key=lambda x: x[0])
            entry_date, exit_date = sorted([closest_distances[0][1], closest_distances[1][1]])
            
            condition = results_df['SID'] == typhoon_id
            results_df.loc[condition, ['entry_date', 'exit_date']] = [entry_date, exit_date]
            results_df.loc[condition, 'is_third_pass'] = True
            
end_time = time.time()
print(f"Third pass completed in {end_time - start_time:.2f} seconds")

# Merge with typhoon overview to add the name_year column
results_df = results_df.merge(data_overview[['storm_id', 'name_year']], left_on='SID', right_on='storm_id', how='left').drop(columns=['storm_id'])

# Ensure the datetime columns are in datetime format
results_df['entry_date'] = pd.to_datetime(results_df['entry_date'])
results_df['exit_date'] = pd.to_datetime(results_df['exit_date'])

# Calculate the difference in hours between exit and entry
results_df['hours_difference'] = (results_df['exit_date'] - results_df['entry_date']).dt.total_seconds() / 3600

# Rearrange the order of the columns
desired_order = ['SID', 'name_year', 'ADM3_PCODE', 'entry_date', 'exit_date','hours_difference', 'is_intersect', 'is_third_pass']
results_df = results_df[desired_order]

# Save results to CSV
results_df.to_csv('IBF_typhoon_model/data/gis_data/typhoon_enter_exit_per_municipality.csv', index=False)

print("Results saved to IBF_typhoon_model/data/gis_data/typhoon_municipality_impact.csv")