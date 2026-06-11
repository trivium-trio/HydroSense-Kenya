# Loading the actual datasets 

import pandas as pd
import numpy as np
import os


def detect_outliers_iqr(data: pd.Series) -> pd.Series:
    """Detect outliers using the Interquartile Range (IQR) method.

    Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are flagged as outliers.

    Args:
        data: A pandas Series of numeric values.

    Returns:
        A boolean Series where True indicates an outlier.
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (data < lower) | (data > upper)


def detect_outliers_zscore(data: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Detect outliers using the z-score method.

    Values whose absolute z-score exceeds the threshold are flagged.

    Args:
        data: A pandas Series of numeric values.
        threshold: Number of standard deviations to use as the cutoff.

    Returns:
        A boolean Series where True indicates an outlier.
    """
    mean = data.mean()
    std = data.std()
    if std == 0:
        return pd.Series([False] * len(data), index=data.index)
    z_scores = np.abs((data - mean) / std)
    return z_scores > threshold


def load_datasets():
    # 1. Dynamically locate the absolute path of this exact Python file (the 'src' folder)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Navigate up one level to the project root, then down into 'data/raw'
    data_dir = os.path.join(src_dir, '..', 'data', 'raw')
    
    # 3. Load the EXACT physical filenames currently sitting in your VS Code folder
    weather_df = pd.read_csv(os.path.join(data_dir, 'weather_daily.csv'), na_values=['NA', ''])
    soil_df = pd.read_csv(os.path.join(data_dir, 'soil_sensor_data.csv'), na_values=['NA', ''])
    crop_parameters_df = pd.read_csv(os.path.join(data_dir, 'crop_zone_parameters.csv'), na_values=['NA', ''])
    
    return weather_df, soil_df, crop_parameters_df

# Execute Loading
weather_df, soil_df, crop_parameters_df = load_datasets()

# Phase 1: Identify Quality Issues
def phase_1_diagnostics(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Executes Phase 1: Identify Quality Issues.
    Explicitly targets missing values, outliers, units, and sensor anomalies.
    """
    print(f"--- DIAGNOSTICS: {dataset_name.upper()} DATASET ---")
    
    # 1. Missing Values (Filtered for clarity)
    print("\n1. MISSING VALUES:")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        print(missing_cols)
    else:
        print("No missing values detected.")
        
    # 2. Structural Duplicates
    print("\n2. DUPLICATES:")
    print(f"Total identical rows: {df.duplicated().sum()}")
    
    # 3. Domain-Specific Anomalies, Outliers & Inconsistent Units
    print("\n3. SENSOR ANOMALIES & OUTLIERS:")
    anomaly_found = False
    
    if dataset_name == 'weather':
        if 'temperature_c' in df.columns:
            high_temp = df[df['temperature_c'] > 40.0]
            if not high_temp.empty:
                print(f"[!] OUTLIER: {len(high_temp)} temperature readings exceed 40.0°C.")
                anomaly_found = True
                
        if 'rainfall_mm' in df.columns:
            extreme_rain = df[df['rainfall_mm'] > 50.0]
            if not extreme_rain.empty:
                print(f"[!] EXTREME: {len(extreme_rain)} rainfall events exceed 50.0mm (Verify if storm or sensor error).")
                anomaly_found = True
                
    elif dataset_name == 'soil':
        if 'tank_level_liters' in df.columns:
            overflow = df[df['tank_level_liters'] > 5000]
            if not overflow.empty:
                print(f"[!] IMPOSSIBLE PHYSICAL UNIT: {len(overflow)} tank readings exceed absolute 5000L capacity.")
                anomaly_found = True
                
        if 'sensor_status' in df.columns:
            faults = df[df['sensor_status'] == 'CHECK']
            if not faults.empty:
                print(f"[!] HARDWARE FAULT: {len(faults)} sensors explicitly report 'CHECK' status.")
                anomaly_found = True
                
        if 'pump_flow_lpm' in df.columns and 'pump_power_watts' in df.columns:
            logical_error = df[(df['pump_flow_lpm'] == 0.0) & (df['pump_power_watts'] > 0)]
            if not logical_error.empty:
                print(f"[!] INCONSISTENT LOGIC: {len(logical_error)} records show pump drawing power but 0.0 LPM flow.")
                anomaly_found = True
                
        if 'soil_moisture_pct' in df.columns:
            critical_drop = df[df['soil_moisture_pct'] < 10.0]
            if not critical_drop.empty:
                print(f"[!] OUTLIER: {len(critical_drop)} moisture readings drop below 10% (Severe wilt or disconnected probe).")
                anomaly_found = True
                
    if not anomaly_found:
        print("No immediate physical anomalies detected.")
        
    # 4. Standard Statistical Baseline
    print("\n4. STATISTICAL SUMMARY (Min/Max verification):")
    print(df.describe().T[['min', 'max', 'mean', 'std']])
    print("-" * 50 + "\n")

# Phase 2: Data Cleaning

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

cleaned_weather_df = clean_hydrosense_data(weather_df, 'weather')
cleaned_soil_df = clean_hydrosense_data(soil_df, 'soil')
cleaned_crop_df = clean_hydrosense_data(crop_parameters_df, 'crop_parameters')
    

# Dataset Merging
def combine_hydrosense_datasets(df_weather: pd.DataFrame, df_soil: pd.DataFrame, df_crop: pd.DataFrame) -> pd.DataFrame:
    """
    Merges weather, soil, and crop datasets into a single denormalized array.
    Assumes all input DataFrames have already passed the cleaning pipeline.
    """
    
    # Step 1: Normalize temporal keys for merging
    # Ensure weather date is datetime format
    df_weather['date'] = pd.to_datetime(weather_df['date']).dt.tz_localize(None).dt.normalize()
    
    # Extract date from soil timestamp (e.g., converting '2026-03-01 12:00' to '2026-03-01')
    df_soil['date'] = pd.to_datetime(soil_df['timestamp']).dt.tz_localize(None).dt.normalize()

    # Step 2: Temporal Join (Soil + Weather)
    # Merges daily weather data across all three zone readings for that specific day
    merged_temporal = pd.merge(df_soil, df_weather, on='date', how='left')
    
    # Step 3: Spatial Join (Temporal Data + Crop Parameters)
    # Broadcasts static zone parameters (e.g., target_moisture) across all rows matching the zone_id
    final_dataset = pd.merge(merged_temporal, df_crop, on='zone_id', how='left')
    
    return final_dataset


if __name__ == '__main__':

    # Diagnostic check before merge
    print("\n--- PRE-MERGE DIAGNOSTICS ---")
    print(f"Weather dataset shape: {cleaned_weather_df.shape}")
    print(f"Soil Date Range: {cleaned_soil_df['timestamp'].min()} to {cleaned_soil_df['timestamp'].max()}")
    print(f"Weather Date Range: {cleaned_weather_df['date'].min()} to {cleaned_weather_df['date'].max()}")
    print("-----------------------------\n")
    
    # C. Execute Merge
    combined_df = combine_hydrosense_datasets(cleaned_weather_df, cleaned_soil_df, cleaned_crop_df)
    
    # D. Execute Export
    src_dir = os.path.dirname(os.path.abspath(__file__))
    output_directory = os.path.join(src_dir, '..', 'data', 'processed')
    output_file = 'cleaned_irrigation_dataset.csv'
    output_path = os.path.join(output_directory, output_file)
    
    os.makedirs(output_directory, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    
    print(f"Export successful. Absolute path resolved to: {output_path}")