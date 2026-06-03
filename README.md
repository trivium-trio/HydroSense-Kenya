# HydroSense-Kenya

**A Scientific Computing System for Smart Irrigation, Water Balance Simulation, and Climate-Aware Decision Support**

> ICS 2207: Scientific Computing — Capstone Project, February–May 2026 Semester

---

## Project Overview

HydroSense-Kenya is an end-to-end scientific computing project that addresses water-use efficiency for smallholder farming in Kenya. Using daily weather records, soil-moisture sensor readings, and crop-zone parameters, the system:

- **Models** soil-water balance using a discrete difference equation
- **Estimates** daily evapotranspiration from weather variables
- **Detects** data quality issues (missing values, outliers, sensor faults)
- **Implements** numerical methods from scratch (root finding, integration, differentiation, linear systems)
- **Simulates** soil moisture trajectories using Euler and Runge-Kutta ODE solvers
- **Quantifies** rainfall uncertainty through Monte Carlo simulation (1000+ scenarios)
- **Optimizes** irrigation schedules to minimize water use while preventing crop stress
- **Validates** all results with automated tests (pytest)

### Central Scientific Question

> Given weather and soil-sensor data, how can we model water availability, estimate water deficit, simulate future soil moisture, and recommend an efficient irrigation plan that minimizes water use without exposing crops to moisture stress?

### Core Models

**Water Balance Equation:**
```
S(t+1) = S(t) + R(t) + I(t) - ET(t) - D(t)
```

| Term | Meaning |
|------|---------|
| S(t) | Soil moisture at time t (%) |
| R(t) | Rainfall (mm) |
| I(t) | Irrigation applied (mm) |
| ET(t) | Evapotranspiration (mm) |
| D(t) | Drainage loss (mm) |

**Simplified Evapotranspiration:**
```
ET = max(0, 0.12*T + 0.35*W + 2.4*Solar - 0.025*H)
```

**Drainage Model:**
```
D(t) = drainage_coefficient * max(0, S_interim - field_capacity)
```

---

## Project Structure

```
HydroSense-Kenya/
│
├── data/
│   ├── raw/                              # Original unmodified datasets
│   │   ├── weather_daily.csv             # 30 days, 6 weather variables
│   │   ├── soil_sensor_data.csv          # 90 rows, 3 zones × 30 days
│   │   └── crop_zone_parameters.csv      # Zone-specific thresholds
│   └── processed/
│       └── cleaned_irrigation_dataset.csv # Merged + cleaned output
│
├── notebooks/                            # Jupyter notebooks (Levels 1-6)
│   ├── Level_1_Problem_Framing.ipynb
│   ├── Level_2_Vectorization_and_Error.ipynb
│   ├── Level_3_Numerical_Methods.ipynb
│   ├── Level_4_Data_Analysis_and_Visualization.ipynb
│   ├── Level_5_Simulation_and_Optimization.ipynb
│   └── Level_6_Final_Integration.ipynb
│
├── src/                                  # Reusable Python modules
│   ├── __init__.py
│   ├── data_cleaning.py                  # Loading, outlier detection, cleaning
│   ├── numerical_methods.py              # Root finding, integration, linear systems
│   ├── simulation.py                     # Euler, RK4, Monte Carlo
│   ├── optimization.py                   # Irrigation schedule optimization
│   └── visualization.py                  # Plotting utilities
│
├── tests/                                # Automated tests (pytest)
│   ├── test_root_finding.py              # 11 tests
│   ├── test_integration.py               # 9 tests
│   ├── test_linear_systems.py            # 7 tests
│   └── test_simulation.py               # 12 tests
│
├── reports/                              # Final deliverables
│   ├── README.md
│   ├── final_scientific_report.pdf       # (pending)
│   └── presentation_slides.pdf           # (pending)
│
├── Capstone_Project                      # Original project brief (PDF)
├── AI_USE_LOG.md                         # AI usage documentation
├── README.md                             # This file
├── requirements.txt                      # Python dependencies
├── pyproject.toml                        # uv project config
└── uv.lock                              # Locked dependencies
```

---

## Datasets

### weather_daily.csv (30 rows × 6 columns)

Daily weather observations for March 2026. Contains **intentional data quality issues** for cleaning practice:
- Missing rainfall on Mar 8 (NA)
- Missing humidity on Mar 21 (NA)
- Temperature outlier: 45.8°C on Mar 14 (sensor error)
- Extreme rainfall: 85 mm on Mar 26 (flagged but retained as plausible)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| date | datetime | YYYY-MM-DD | Observation date |
| rainfall_mm | float | mm | Daily cumulative rainfall |
| temperature_c | float | °C | Mean daily air temperature |
| humidity_pct | float | % | Mean relative humidity |
| wind_speed_mps | float | m/s | Mean wind speed |
| solar_index | float | 0–1 | Solar radiation index |

### soil_sensor_data.csv (90 rows × 7 columns)

Daily noon sensor readings across three farm zones. Contains **deliberate anomalies**:
- Missing soil moisture on Mar 6 Zone_B
- Tank level 9900 L on Mar 14 Zone_C (data entry error; normal range 3400–4800)
- Pump flow 0.00 LPM on Mar 21 Zone_B (sensor fault, status = CHECK)
- Soil moisture 8.5% on Mar 25 Zone_B (anomalously low)

### crop_zone_parameters.csv (3 rows × 7 columns)

| Zone | Crop | Area (m²) | Min Moisture | Target | Field Capacity | Drainage Coeff |
|------|------|-----------|-------------|--------|---------------|---------------|
| Zone_A | Tomato | 120 | 22% | 33% | 41% | 0.18 |
| Zone_B | Kale | 90 | 24% | 35% | 43% | 0.15 |
| Zone_C | Maize | 180 | 20% | 31% | 40% | 0.22 |

---

## Setup and Installation

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/HydroSense-Kenya.git
cd HydroSense-Kenya

# Install dependencies with uv
uv sync

# Or with pip
pip install -r requirements.txt
```

### Run Notebooks

```bash
# Start Jupyter
uv run jupyter notebook

# Navigate to notebooks/ and run Level_1 through Level_6 in order
```

Notebooks should be run **from the `notebooks/` directory** — they use relative paths (`../data/raw/...`) to access datasets.

### Run Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_root_finding.py -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=term-missing
```

---

## Six-Level Project Structure

| Level | Topic | Marks | Key Deliverable |
|-------|-------|-------|-----------------|
| 1 | Problem Framing & Python Foundation | 10 | Problem statement, data dictionary, basic plot |
| 2 | NumPy, Vectorization & Floating-Point Errors | 15 | Timing comparison, error propagation analysis |
| 3 | Core Numerical Methods | 20 | Root finding, integration, linear systems |
| 4 | Data Cleaning & Visualization | 15 | 6 scientific visualizations with interpretation |
| 5 | Simulation, Monte Carlo & Optimization | 25 | Euler/RK4 simulation, 1000 MC scenarios, optimized schedule |
| 6 | AI Use, Testing & Reproducibility | 15 | 39 automated tests, AI log, final report |

---

## Numerical Methods Implemented (all from scratch)

| Category | Methods |
|----------|---------|
| Root Finding | Bisection, Newton-Raphson, Secant |
| Differentiation | Forward, Backward, Central finite differences |
| Integration | Composite Trapezoidal, Simpson's 1/3 Rule |
| Linear Systems | Gaussian Elimination (partial pivoting), LU Decomposition |
| ODEs | Euler Method, 4th-order Runge-Kutta |
| Stochastic | Monte Carlo rainfall simulation (gamma distribution) |
| Optimization | Greedy threshold-based, Gradient descent |

> SciPy/NumPy's `linalg.solve` is used **only for verification**, never as the primary solver.

---

## Key Results

- **Vectorization speedup:** NumPy is 10-50x faster than Python loops for ET computation
- **Root finding:** Newton-Raphson converges in ~4 iterations vs ~20 for bisection
- **Simulation:** Without irrigation, all zones breach minimum moisture by day 20-25
- **Monte Carlo:** >80% probability of water shortage without intervention
- **Optimized schedule:** Greedy strategy eliminates all stress days while minimizing total water use
- **Testing:** 39 automated tests, all passing

---

## Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Numerical computation and vectorization |
| pandas | Data loading, cleaning, and manipulation |
| matplotlib | Scientific visualization |
| pytest | Automated testing |
| scipy | Verification only (not used for primary computations) |
| jupyter | Running notebooks |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
