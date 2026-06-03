"""
data_cleaning.py
Functions for loading, cleaning, and preprocessing HydroSense-Kenya datasets.
"""

import pandas as pd
import numpy as np


def load_datasets(data_dir='../data/raw'):
    """Load all three raw datasets with proper NA handling and date parsing."""
    weather = pd.read_csv(f'{data_dir}/weather_daily.csv', na_values=['NA', ''])
    soil = pd.read_csv(f'{data_dir}/soil_sensor_data.csv', na_values=['NA', ''])
    params = pd.read_csv(f'{data_dir}/crop_zone_parameters.csv')

    weather['date'] = pd.to_datetime(weather['date'])
    soil['timestamp'] = pd.to_datetime(soil['timestamp'])
    return weather, soil, params


def detect_outliers_iqr(series, factor=1.5):
    """Detect outliers using the IQR method. Returns boolean mask."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return (series < lower) | (series > upper)


def detect_outliers_zscore(series, threshold=3.0):
    """Detect outliers using z-score method. Returns boolean mask."""
    z = np.abs((series - series.mean()) / series.std())
    return z > threshold


def clean_weather(weather):
    """
    Clean weather dataset. Returns cleaned DataFrame and a log of decisions.
    """
    log = []
    df = weather.copy()

    # 1. Missing rainfall on Mar 8 — interpolate from neighbours
    mask = df['rainfall_mm'].isna()
    n_missing = mask.sum()
    if n_missing > 0:
        df['rainfall_mm'] = df['rainfall_mm'].interpolate(method='linear')
        log.append(f"Rainfall: {n_missing} missing value(s) filled via linear interpolation")

    # 2. Missing humidity on Mar 21 — interpolate
    mask = df['humidity_pct'].isna()
    n_missing = mask.sum()
    if n_missing > 0:
        df['humidity_pct'] = df['humidity_pct'].interpolate(method='linear')
        log.append(f"Humidity: {n_missing} missing value(s) filled via linear interpolation")

    # 3. Temperature outlier: 45.8°C on Mar 14 — replace with median
    temp_outliers = detect_outliers_zscore(df['temperature_c'], threshold=3.0)
    if temp_outliers.any():
        outlier_vals = df.loc[temp_outliers, 'temperature_c'].tolist()
        median_temp = df.loc[~temp_outliers, 'temperature_c'].median()
        df.loc[temp_outliers, 'temperature_c'] = median_temp
        log.append(f"Temperature: {temp_outliers.sum()} outlier(s) {outlier_vals} "
                   f"replaced with median ({median_temp:.1f}°C)")

    # 4. Rainfall outlier: 85mm on Mar 26 — flag but retain (plausible extreme event)
    rain_outliers = detect_outliers_iqr(df['rainfall_mm'], factor=1.5)
    if rain_outliers.any():
        outlier_vals = df.loc[rain_outliers, 'rainfall_mm'].tolist()
        log.append(f"Rainfall: {rain_outliers.sum()} potential outlier(s) {outlier_vals} "
                   f"flagged but RETAINED (plausible extreme events)")

    return df, log


def clean_soil(soil):
    """
    Clean soil sensor dataset. Returns cleaned DataFrame and a log of decisions.
    """
    log = []
    df = soil.copy()

    # 1. Missing soil moisture on Mar 6 Zone_B — interpolate within zone
    mask = df['soil_moisture_pct'].isna()
    n_missing = mask.sum()
    if n_missing > 0:
        df['soil_moisture_pct'] = df.groupby('zone_id')['soil_moisture_pct'].transform(
            lambda x: x.interpolate(method='linear')
        )
        log.append(f"Soil moisture: {n_missing} missing value(s) filled via "
                   f"within-zone linear interpolation")

    # 2. Tank level outlier: 9900 L on Mar 14 Zone_C — replace with zone median
    tank_outliers = detect_outliers_zscore(df['tank_level_liters'], threshold=3.0)
    if tank_outliers.any():
        outlier_vals = df.loc[tank_outliers, 'tank_level_liters'].tolist()
        for zone in df.loc[tank_outliers, 'zone_id'].unique():
            zone_mask = (df['zone_id'] == zone) & ~tank_outliers
            median_val = df.loc[zone_mask, 'tank_level_liters'].median()
            df.loc[(df['zone_id'] == zone) & tank_outliers, 'tank_level_liters'] = median_val
            log.append(f"Tank level: outlier {outlier_vals} in {zone} "
                       f"replaced with zone median ({median_val:.0f} L)")

    # 3. Pump flow 0.0 LPM on Mar 21 Zone_B (sensor_status=CHECK) — replace with zone median
    check_mask = df['sensor_status'] == 'CHECK'
    if check_mask.any():
        for zone in df.loc[check_mask, 'zone_id'].unique():
            zone_ok = (df['zone_id'] == zone) & (df['sensor_status'] == 'OK')
            median_flow = df.loc[zone_ok, 'pump_flow_lpm'].median()
            df.loc[(df['zone_id'] == zone) & check_mask, 'pump_flow_lpm'] = median_flow
            log.append(f"Pump flow: CHECK-status reading in {zone} "
                       f"replaced with zone median ({median_flow:.1f} LPM)")
        df.loc[check_mask, 'sensor_status'] = 'CORRECTED'

    # 4. Anomalous soil moisture: 8.5% on Mar 25 Zone_B — replace with zone interpolation
    for zone in df['zone_id'].unique():
        zone_mask = df['zone_id'] == zone
        zone_data = df.loc[zone_mask, 'soil_moisture_pct']
        outliers = detect_outliers_iqr(zone_data, factor=2.0)
        if outliers.any():
            outlier_vals = zone_data[outliers].tolist()
            idx = zone_data[outliers].index
            df.loc[idx, 'soil_moisture_pct'] = np.nan
            df['soil_moisture_pct'] = df.groupby('zone_id')['soil_moisture_pct'].transform(
                lambda x: x.interpolate(method='linear')
            )
            log.append(f"Soil moisture: anomaly {outlier_vals} in {zone} "
                       f"replaced via interpolation")

    return df, log


def save_cleaned_dataset(weather_clean, soil_clean, params, output_dir='../data/processed'):
    """Merge and save the cleaned irrigation dataset."""
    # Add date column to soil for merging
    soil_clean['date'] = soil_clean['timestamp'].dt.date.astype(str)
    soil_clean['date'] = pd.to_datetime(soil_clean['date'])

    merged = soil_clean.merge(weather_clean, on='date', how='left')
    merged = merged.merge(params, on='zone_id', how='left')

    output_path = f'{output_dir}/cleaned_irrigation_dataset.csv'
    merged.to_csv(output_path, index=False)
    return merged, output_path
