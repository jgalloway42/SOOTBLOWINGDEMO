# Soot Blowing Optimization Demo

## Abstract

Industrial boiler soot blowing optimization system using physics-based simulation and machine learning. Focuses on optimizing cleaning schedules to maximize efficiency while minimizing operational costs.

**Current Status**: ⚠️ **Mixed Results** - Architecture working, physics correlations failing validation.

## Current System Status (Updated 2025-09-04)

### ✅ **Working Components**
- **Centralized Architecture**: SootBlowingSimulator unified, fire-side only logic correct
- **Boolean Detection**: Fixed notebook analysis - now detects 1,002 cleaning events
- **Event Generation**: Section-specific cleaning schedules operating correctly (7 sections)
- **Data Output**: Proper file structure with comprehensive operational parameters
- **Analysis Framework**: EDA functions working, correctly identifying physics issues

### ❌ **Critical Physics Issues Identified**

#### Physics Correlations Failing Validation
- **Time-Fouling**: r=-0.001 (target >+0.4) - Near zero correlation, no fouling buildup pattern
- **Fouling-Efficiency**: r=+0.018 (target <-0.25) - Wrong direction, too weak for ML training
- **Stack Temp-Fouling**: r=+0.016 (target >+0.2) - Insufficient temperature response
- **Impact**: ❌ **Physics relationships too weak for meaningful ML optimization**

#### Root Causes Identified  
- **Fouling Rates**: 0.00004-0.00030/hr too conservative for realistic accumulation
- **Impact Scaling**: Efficiency (0.25x) and temperature (120°F) factors too weak
- **Static Temperatures**: Steam temp hardcoded 700°F, furnace gas outlet unresponsive
- **Fouling Range**: Actual 1.000-1.005 vs expected 1.0-1.25 industrial range
- **Status**: ❌ **Physics strengthening required before ML training**

### ⚠️ **Current Capabilities & Limitations**
**What Works**:
- Event detection and cleaning schedule execution
- Centralized soot blowing architecture with proper fire-side only logic
- Comprehensive data generation (220 parameters, 8,760 hourly records)
- Analysis tools correctly identifying the physics limitations

**What Needs Fixing**:
- Fouling accumulation rates (increase 3-5x for realistic buildup)
- Physics impact scaling (strengthen efficiency/temperature relationships)
- Dynamic temperature calculations (eliminate static 700°F fallbacks)
- Fouling range expansion (enable 1.0-1.25 degradation patterns)

### 🎯 **Current Project Status**
Build a working soot blowing optimization system that can:
- ✅ Generate cleaning event data from centralized simulation
- ❌ **NEEDS WORK**: Create realistic fouling-performance correlations  
- ❌ **BLOCKED**: Train ML models (requires stronger physics signals)
- ❌ **BLOCKED**: Provide optimization recommendations (needs correlation patterns)
- ⚠️ **PARTIAL**: Demonstrate architecture (events work, physics insufficient)

## Dataset

## Data Representation and Processing
### Combustion Simulation
* Inputs
    - Coal Ultimate Analysis (by mass including ash and moisture)
    - Coal feed (pph)
    - Fuel-bound NOx Conversion Effciency
    - Ambient Temperature, Relative Humidity, & Atmospheric Pressure
    - Air Flow (scfh)
* Interim Calculations (in order)
    1. Coal HHV adjusted for ash & moisture
    2. Air humidity ratio
    3. Calculate fuel-bound NOx generation
    4. Calculate excess air and assuming no CO or thermal NOx (as a first iteration)
    5. Calculate CO based on excess air.
    6. Update flue gas (excess air, & CO).
    7. Estimate flame temperature from simple model.
    8. Calculate Thermal NOx from flame temperature estimation.
    9. Update flue gas.
    10. Calculate Combustion efficiency and actual heat-release.
    11. Calculate Flame Temperature.
* Outputs
    - Flue gas constiuents (raw calculations and then dry ppm/% as would be normally output by a flue gas analyzer)
    - Heat Release
    - Flame Temperature
    - Flue gas flow (lb/hr)

#### Class Structure
##### Input Properties (with setters that trigger recalculation):

- **ultimate_analysis** - Coal ultimate analysis (% by mass)
- **coal_lb_per_hr** - Coal mass flow rate
- **air_scfh** - Air volumetric flow rate
- **NOx_eff** - Fuel NOx conversion efficiency
- **air_temp_F, air_RH_pct, atm_press_inHg** - Air conditions

##### Interim Calculation Properties:

- **coal_HHV_btu_per_lb** - Higher heating value
- **coal_dry_ash_free_lb_per_hr** - Dry ash-free coal rate
- **humidity_ratio** - Air humidity ratio
- **CO2_fraction** - CO₂ formation fraction (1.0 is complete combution, anything <1 gives the rest as CO)
- **combustion_efficiency** - Overall efficiency
- **actual_heat_release_btu_per_hr** - Actual heat release
- **flame_temp_K** - Flame temperature in Kelvin

##### Results Properties:

- **total_flue_gas_lb_per_hr** - Total flue gas flow
- **CO2_lb_per_hr**, **CO_lb_per_hr**, **SO2_lb_per_hr** - Emissions
- **NO_fuel_lb_per_hr**, **NO_thermal_lb_per_hr**, **NO_total_lb_per_hr** - NOx emissions
- **H2O_lb_per_hr**, **N2_lb_per_hr**, **O2_lb_per_hr** - Other components
- **dry_O2_pct** - Excess oxygen percentage
- **heat_released_btu_per_hr** - Heat release
- **flame_temp_F** - Flame temperature in Fahrenheit

##### Key Features

- **Lazy Evaluation:** Calculations are only performed when properties are accessed or `calculate()` is called.
- **Auto-recalculation:** Changing any input property automatically invalidates cached results.
- **Property-based Interface:** Clean access to all values through properties.
- **Debugging Support:** `calculate(debug=True)` shows detailed output and plots.
- **Input Validation:** Ensures ultimate analysis sums to 100%.
- **Error Handling:** Graceful fallbacks for temperature calculations.

Usage Example
```python
# Create model
model = CoalCombustionModel(ultimate_analysis, 10000, 2000000, 0.35)

# Calculate and view results
model.calculate(debug=True)

# Access properties
print(f"CO2 Emissions: {model.CO2_lb_per_hr:,.0f} lb/hr")
print(f"Flame Temp: {model.flame_temp_F:,.0f} °F")

# Change parameters and automatically recalculate
model.NOx_eff = 0.50  # Triggers recalculation when next accessed
print(f"New NOx: {model.NO_total_lb_per_hr:.2f} lb/hr")
```

### Soot Generation Approximation
#### Method
* **Higher CO in flue gas** &rarr; Higher likelihood of incomplete combustion &rarr; Higher likelihood of soot formation. (CO &uarr;, Soot &uarr;)

* **Fuel-Bound NOx** Creation of fuel-rich zones that reduce fuel-bound NOx &rarr; Increased propensity for soot formation if secondary air mixing/burnout is insufficient. (FB NOx efficiency &darr;, Soot &uarr;)

* **Thermal NOx**  Low peak temperatures can indirectly lead to conditions that are also less favorable for complete soot oxidation, or may involve operating closer to fuel-rich conditions to reduce temperature. (Thermal NOx &uarr;, Soot &uarr;)

* **Lower Excess Air** &rarr; Increased risk of fuel-rich zones and incomplete combustion &rarr; Higher soot. (Excess Air &uarr;, Soot &uarr;)

#### Implentation Method

### Tube Fouling Approximation
#### TDB

### Tube Temperature Simulation
#### TDB

## Model Derivation

## Results

## Conclusion



###### Repo Structure Duplicated from here: https://github.com/drivendata/cookiecutter-data-science
