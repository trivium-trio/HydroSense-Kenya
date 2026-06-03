# HydroSense-Kenya

A Scientific Computing System for Smart Irrigation, Water Balance Simulation, and Climate-Aware Decision Support.

## Course

ICS 2207: Scientific Computing

## Project Structure

```
HydroSense-Kenya/
├── data/
│   ├── raw/                  # Original datasets
│   │   ├── weather_daily.csv
│   │   ├── soil_sensor_data.csv
│   │   └── crop_zone_parameters.csv
│   └── processed/            # Cleaned datasets
├── notebooks/                # Jupyter notebooks (Levels 1-6)
├── src/                      # Python source modules
├── tests/                    # Automated tests (pytest)
├── reports/                  # Final report and presentation
├── AI_USE_LOG.md
├── README.md
└── requirements.txt
```

## Setup

```bash
uv sync
```

## Datasets

| File | Description |
|------|-------------|
| `weather_daily.csv` | Daily rainfall, temperature, humidity, wind speed, solar index (30 days) |
| `soil_sensor_data.csv` | Daily noon sensor readings for 3 farm zones (90 rows) |
| `crop_zone_parameters.csv` | Crop and zone-specific moisture thresholds and drainage coefficients |

## Core Model

Water balance equation:

```
S(t+1) = S(t) + R(t) + I(t) - ET(t) - D(t)
```

Simplified evapotranspiration:

```
ET = max(0, 0.12*T + 0.35*W + 2.4*Solar - 0.025*H)
```
