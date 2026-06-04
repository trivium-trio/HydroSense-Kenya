# AI Use Log — HydroSense-Kenya

This log documents all AI-assisted programming used in this project. Every AI-supported output was inspected, tested, and validated before inclusion.

**AI Tool Used:** GitHub Copilot (VS Code extension) and ChatGPT (GPT-4)

---

## Entry 1: Problem Statement Drafting

| Field | Details |
|-------|---------|
| **Task** | Draft the Level 1 problem statement (500-700 words) |
| **Prompt** | "Write a 600-word scientific problem statement for a smart irrigation project in Kenya. The project uses soil moisture sensors, weather data, and a water-balance model to optimize irrigation for smallholder farmers. Mention water scarcity, climate change, and the limitations of schedule-based irrigation." |
| **AI Output** | A well-structured draft covering Kenya's agricultural dependence, water scarcity statistics, and the computational framing of the irrigation problem. |
| **Accepted?** | Partly |
| **Modifications** | Rewrote the second paragraph to include correct per-capita water figures for Kenya. Added the specific water-balance equation and ET formula from the project brief. Removed generic claims about "AI and IoT" that were not relevant to our scope. Shortened from ~750 words to meet the 700-word limit. |
| **Validation** | Cross-checked water scarcity statistics against UNEP and Kenya Water Resources Authority reports. Verified the mathematical notation matches the project brief exactly. |

---

## Entry 2: Generating pytest Test Cases

| Field | Details |
|-------|---------|
| **Task** | Generate pytest test cases for the bisection, Newton-Raphson, and secant root-finding methods |
| **Prompt** | "Generate pytest test cases for three root-finding functions: bisection(f, a, b, tol), newton_raphson(f, f_prime, x0, tol), and secant(f, x0, x1, tol). Each returns a dict with keys 'root', 'iterations', 'error', 'converged', 'history'. Test with f(x) = x^2 - 4 (root at 2) and f(x) = x^3 - x - 2. Include edge cases like invalid brackets." |
| **AI Output** | Generated 12 test functions covering basic roots, edge cases, convergence verification, and history tracking. |
| **Accepted?** | Partly |
| **Modifications** | Changed tolerance from 1e-8 to 1e-6 to match our implementation default. Added the `test_fewer_iterations_than_bisection` test to verify Newton's quadratic convergence advantage. Fixed import paths to match our project structure. Added the irrigation-specific test case for Gaussian elimination. |
| **Validation** | Ran all tests with `pytest -v`. Verified that expected roots match known analytical solutions (x=2 for x²−4, verified x³−x−2 root numerically). |

---

## Entry 3: README Documentation

| Field | Details |
|-------|---------|
| **Task** | Create a comprehensive README.md for the project repository |
| **Prompt** | "Write a detailed README.md for a Python scientific computing project called HydroSense-Kenya. It has 6 levels covering problem framing, NumPy vectorization, numerical methods, data analysis, simulation/optimization, and testing. Uses uv for package management. Include sections for project overview, setup, folder structure, datasets, how to run notebooks, and how to run tests." |
| **AI Output** | A structured README with all requested sections, markdown tables, and code blocks. |
| **Accepted?** | Yes, with edits |
| **Modifications** | Added the actual water-balance and ET equations. Corrected the folder structure to match our real directory layout. Added team member names. Replaced generic setup instructions with uv-specific commands. |
| **Validation** | Followed the setup instructions on a clean environment to confirm they work. Verified all file paths referenced in the README actually exist. |

---

## Entry 4: Code Documentation and Docstrings

| Field | Details |
|-------|---------|
| **Task** | Add docstrings to numerical methods in `src/numerical_methods.py` |
| **Prompt** | "Add NumPy-style docstrings to these Python functions: bisection, newton_raphson, secant, trapezoidal_rule, simpsons_rule, gaussian_elimination. Each docstring should include Parameters, Returns, and a brief description of the algorithm." |
| **AI Output** | Complete docstrings for all 6 functions with parameter types, return descriptions, and algorithm summaries. |
| **Accepted?** | Yes, with edits |
| **Modifications** | Corrected the convergence order descriptions (bisection is O(1/2^n), not O(n)). Added the specific mathematical formulas to each docstring. Fixed parameter names to match our actual function signatures. |
| **Validation** | Ran `python -c "from src.numerical_methods import bisection; help(bisection)"` to verify docstrings render correctly. Checked mathematical claims against lecture notes. |

---

## Entry 5: Data Cleaning Strategy

| Field | Details |
|-------|---------|
| **Task** | Get suggestions for handling specific data anomalies in sensor data |
| **Prompt** | "I have soil sensor data with these anomalies: (1) tank_level = 9900 L when normal range is 3400-4800, (2) pump_flow = 0.0 LPM with sensor_status = CHECK, (3) soil_moisture = 8.5% when zone average is ~25%. What are appropriate cleaning strategies for each in a scientific computing context?" |
| **AI Output** | Suggested three strategies: median replacement for tank outlier, zone-median for pump fault, and interpolation for moisture anomaly. Recommended documenting each decision. |
| **Accepted?** | Yes |
| **Modifications** | Implemented as suggested but added the IQR and z-score detection functions rather than hard-coding thresholds. Added the cleaning log mechanism to automatically document every decision made. |
| **Validation** | Compared cleaned values against zone-level statistics. Verified that before/after plots (Visualization 6 in Level 4) show targeted corrections without distorting overall trends. |

---

## Summary

| Metric | Value |
|--------|-------|
| Total AI-assisted tasks | 5 |
| Fully accepted without changes | 0 |
| Accepted with modifications | 5 |
| Rejected | 0 |
| Lines of AI-generated code used verbatim | ~30 (test boilerplate) |
| Lines of AI-generated code modified | ~80 |
| Total project lines of code | ~1500+ |

**Conclusion:** AI tools were used as a productivity aid for documentation, test scaffolding, and brainstorming cleaning strategies. All numerical methods, simulation logic, optimization algorithms, and scientific analysis were implemented and validated by the team. No AI-generated code was used without inspection and testing.
