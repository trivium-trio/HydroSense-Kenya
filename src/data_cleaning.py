# Loading the actual datasets 

import pandas as pd
import os


weather_df = pd.read_csv('data/raw/weather_data.csv', na_values=['NA',   ''])
soil_df = pd.read_csv('data/raw/soil_sensor_data.csv', na_values=['NA',   ''])
crop_parameters_df = pd.read_csv('data/raw/crop_zone_parameters_data.csv', na_values=['NA',   ''])

# Phase 1: Identify Quality Issues
def phase_1_diagnostics(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Executes Phase 1: Identify Quality Issues.
    Outputs diagnostic metadata to the standard output for initial exploratory analysis.
    """
    print(f"--- DIAGNOSTICS: {dataset_name.upper()} DATASET ---")
    
    print("\n1. Structural Info:")
    df.info()
    
    print("\n2. Missing Values Count:")
    print(df.isnull().sum())
    
    print("\n3. Duplicate Rows Count:")
    print(f"Total duplicates: {df.duplicated().sum()}")
    
    print("\n4. Statistical Summary:")
    print(df.describe(include='all'))
    print("-" * 50 + "\n")

# Phase 2: Data Preparation

def clean_hydrosense_data(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Executes the sequential data cleaning phases for HydroSense datasets.
    Valid dataset_name arguments: 'weather', 'soil', 'crop'
    """
    # Phase 2: Data Preparation (Renaming & Type Casting)
    if dataset_name == 'weather':
        if 'rainfall mm' in df.columns:
            df = df.rename(columns={'rainfall mm': 'rainfall_mm', 'solar index': 'solar_index'})
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
    elif dataset_name == 'soil':
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'soil_moisture_pct' in df.columns:
            df['soil_moisture_pct'] = pd.to_numeric(df['soil_moisture_pct'], errors='coerce')

    # Phase 3: Duplicates & Structural Reduction
    df = df.drop_duplicates()

    # Phase 4: Filter Outliers
    if dataset_name == 'weather':
        # Remove temperature anomaly (e.g., 45.80) by setting realistic upper bound
        if 'temperature_c' in df.columns:
            df = df[df['temperature_c'] < 40.0]
            
    elif dataset_name == 'soil':
        # Drop tank_level_liters physical impossibility (9900)
        if 'tank_level_liters' in df.columns:
            df = df[df['tank_level_liters'] <= 5000]
            
        # Drop rows with explicit sensor faults
        if 'pump_flow_lpm' in df.columns and 'sensor_status' in df.columns:
            df = df[~((df['pump_flow_lpm'] == 0.0) & (df['sensor_status'] == 'CHECK'))]
            
        # Drop anomalous soil moisture drops violating agronomic minimums
        if 'soil_moisture_pct' in df.columns:
            df = df[df['soil_moisture_pct'] > 15.0]

    # Phase 5: Handling Missing Data
    if dataset_name == 'weather':
        if 'humidity_pct' in df.columns:
            df['humidity_pct'] = df['humidity_pct'].ffill()
        if 'rainfall_mm' in df.columns:
            df['rainfall_mm'] = df['rainfall_mm'].fillna(0.0)
            
    elif dataset_name == 'soil':
        if 'soil_moisture_pct' in df.columns:
            # Forward fill sequential time-series sensor data
            df['soil_moisture_pct'] = df['soil_moisture_pct'].ffill()

    return df.reset_index(drop=True)

"""phase_1_diagnostics(weather_df, 'weather')
phase_1_diagnostics(soil_df, 'soil')
phase_1_diagnostics(crop_parameters_df, 'crop_parameters')"""

cleaned_weather_df = clean_hydrosense_data(weather_df, 'weather')
cleaned_soil_df = clean_hydrosense_data(soil_df, 'soil')
cleaned_crop_parameters_df = clean_hydrosense_data(crop_parameters_df, 'crop_parameters')

print("Data cleaning completed. Cleaned datasets are ready for analysis.")

# Dataset Merging

def combine_hydrosense_datasets(df_weather: pd.DataFrame, df_soil: pd.DataFrame, df_crop: pd.DataFrame) -> pd.DataFrame:
    """
    Merges weather, soil, and crop datasets into a single denormalized array.
    Assumes all input DataFrames have already passed the cleaning pipeline.
    """
    
    # Step 1: Normalize temporal keys for merging
    # Ensure weather date is datetime format
    df_weather['date'] = pd.to_datetime(cleaned_weather_df['ts']).dt.tz_localize(None).dt.normalize()
    
    # Extract date from soil timestamp (e.g., converting '2026-03-01 12:00' to '2026-03-01')
    df_soil['date'] = pd.to_datetime(cleaned_soil_df['timestamp']).dt.tz_localize(None).dt.normalize()

    
    # Step 2: Temporal Join (Soil + Weather)
    # Merges daily weather data across all three zone readings for that specific day
    merged_temporal = pd.merge(df_soil, df_weather, on='date', how='left')
    
    # Step 3: Spatial Join (Temporal Data + Crop Parameters)
    # Broadcasts static zone parameters (e.g., target_moisture) across all rows matching the zone_id
    final_dataset = pd.merge(merged_temporal, df_crop, on='zone_id', how='left')
    
    return final_dataset

# Execute combination of cleaned datasets into a single denormalized DataFrame ready for analysis and modeling
combined_df = combine_hydrosense_datasets(cleaned_weather_df, cleaned_soil_df, cleaned_crop_parameters_df)

# Define output path
output_directory = 'data/processed'
output_file = 'cleaned_irrigation_dataset.csv'
output_path = os.path.join(output_directory, output_file)

# Create directory if it does not exist (fulfills project folder structure requirement)
os.makedirs(output_directory, exist_ok=True)

# Export to CSV
combined_df.to_csv(output_path, index=False)