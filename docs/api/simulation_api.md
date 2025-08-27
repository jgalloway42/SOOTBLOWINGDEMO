# SIMULATION API REFERENCE

**File Location:** `docs/api/SIMULATION_API.md`  
**Created:** August 2025  
**Status:** CRITICAL - Required for annual dataset generation

---

## AnnualBoilerSimulator

**File:** `src/models/complete_boiler_simulation/simulation/annual_boiler_simulator.py`

### Constructor

```python
AnnualBoilerSimulator(start_date: str = "2024-01-01")
```

**Parameters:**
- `start_date` (str): Start date in YYYY-MM-DD format (default: "2024-01-01")

**Properties Set:**
- `start_date` (pd.Timestamp): Parsed start date
- `end_date` (pd.Timestamp): Start date + 1 year
- `boiler` (EnhancedCompleteBoilerSystem): Internal boiler system instance

### Key Methods

```python
generate_annual_data(
    hours_per_day: int = 24, 
    save_interval_hours: int = 1
) -> pd.DataFrame
```

**Parameters:**
- `hours_per_day` (int): Operating hours per day (default: 24)
- `save_interval_hours` (int): Data recording interval in hours (default: 1)

**IMPORTANT:** Do NOT pass `duration_days` parameter - it will cause "unexpected keyword argument" error.

**Returns:** pandas DataFrame with columns including:
- `timestamp` (datetime): Record timestamp
- `load_factor` (float): Operating load factor (0.0-1.0+)
- `system_efficiency` (float): System efficiency (0.0-1.0)
- `final_steam_temp_F` (float): Steam temperature in °F
- `stack_temp_F` (float): Stack temperature in °F
- `coal_quality` (str): Coal quality type
- `soot_blowing_active` (bool): Soot blowing status
- [148 total columns with operational, performance, and fouling data]

```python
save_annual_data(
    df: pd.DataFrame, 
    filename_prefix: str = "massachusetts_boiler_annual"
) -> Tuple[str, str]
```

**Returns:** Tuple of (data_filepath, metadata_filepath)

### Internal Methods (for reference)

```python
_generate_hourly_conditions(current_datetime: datetime) -> Dict
```
Generates realistic hourly operating conditions.

```python
_simulate_boiler_operation(
    current_datetime: datetime,
    operating_conditions: Dict,
    soot_blowing_actions: Dict
) -> Dict
```
Simulates single hour of boiler operation.

---

## Test Runner Functions

**File:** `src/models/complete_boiler_simulation/simulation/run_annual_simulation.py`

### Main Functions

```python
run_quick_test() -> Optional[pd.DataFrame]
```
**Purpose:** Run 48-hour test simulation
**Returns:** DataFrame with test results or None if failed
**Note:** Uses shortened date range, not duration_days parameter

```python
run_full_simulation() -> Optional[Dict]
```
**Purpose:** Run complete 8760-hour annual simulation  
**Returns:** Dictionary with 'dataset' and 'filename' keys or None if failed

```python
check_dependencies() -> bool
```
**Purpose:** Verify IAPWS and other dependencies are available
**Returns:** True if all dependencies available

### Usage Pattern
```python
def main():
    # Menu-driven interface
    choice = input("Select option (1-4): ")
    
    if choice == "1":
        run_quick_test()
    elif choice == "2": 
        run_full_simulation()
    elif choice == "3":
        # Comprehensive test sequence
        quick_results = run_quick_test()
        if quick_results is not None:
            run_full_simulation()
```

---

## Debug and Validation Functions

**File:** `src/models/complete_boiler_simulation/simulation/debug_script.py`

### Main Validation Functions

```python
test_phase3_realistic_load_variation() -> Tuple[bool, List[Dict]]
```
**Purpose:** Test system across realistic load range (60-105%)
**Returns:** (success_status, load_test_results)

```python
test_phase3_component_integration() -> bool  
```
**Purpose:** Test heat transfer component integration
**Returns:** True if all components integrate successfully

```python
test_phase3_combustion_enhancement() -> bool
```
**Purpose:** Test combustion efficiency variation
**Returns:** True if combustion variation meets targets

```python
generate_phase3_realistic_validation_report() -> Tuple[bool, str]
```
**Purpose:** Generate comprehensive validation report
**Returns:** (overall_success, report_filename)

### Usage Pattern
```python
def main():
    """Main validation execution"""
    overall_success, report_file = generate_phase3_realistic_validation_report()
    
    if overall_success:
        print("[SUCCESS] PHASE 3 VALIDATION SUCCESSFUL!")
    
    return overall_success
```

---

# CRITICAL API FIXES NEEDED

## Issue #1: AnnualBoilerSimulator.generate_annual_data()
**Problem:** run_annual_simulation.py passes unsupported `duration_days` parameter
**Error:** `AnnualBoilerSimulator.generate_annual_data() got an unexpected keyword argument 'duration_days'`

**Current Broken Code:**
```python
test_data = simulator.generate_annual_data(
    hours_per_day=24,
    save_interval_hours=2,
    duration_days=2  # THIS PARAMETER DOESN'T EXIST
)
```

**Correct Fix:**
```python
# For short test, modify simulator date range instead
simulator = AnnualBoilerSimulator(start_date="2024-01-01")
test_data = simulator.generate_annual_data(
    hours_per_day=24,
    save_interval_hours=2
)
# Will generate full year - filter afterward if needed
```

**Alternative Fix:** Modify simulator's end_date before generation:
```python
simulator = AnnualBoilerSimulator(start_date="2024-01-01") 
simulator.end_date = simulator.start_date + pd.DateOffset(days=2)  # Override end date
test_data = simulator.generate_annual_data(hours_per_day=24, save_interval_hours=2)
```

## Issue #2: EnhancedCompleteBoilerSystem Constructor in debug_script.py
**Problem:** debug_script.py passes unsupported parameters to boiler constructor
**Error:** `EnhancedCompleteBoilerSystem.__init__() got an unexpected keyword argument 'steam_pressure'`

**Current Broken Code:**
```python
boiler = EnhancedCompleteBoilerSystem(
    fuel_input=fuel_input,
    flue_gas_mass_flow=gas_flow,
    furnace_exit_temp=3000,
    steam_pressure=150,  # THIS PARAMETER DOESN'T EXIST
    target_steam_temp=700,
    feedwater_temp=220,
    base_fouling_multiplier=1.0
)
```

**Correct Fix:**
```python
boiler = EnhancedCompleteBoilerSystem(
    fuel_input=fuel_input,
    flue_gas_mass_flow=gas_flow,
    furnace_exit_temp=3000,
    # Remove steam_pressure - it's hardcoded internally to 150 psia
    target_steam_temp=700,
    feedwater_temp=220,
    base_fouling_multiplier=1.0
)
```

---

# INTEGRATION PATTERNS

## Basic Annual Simulation Usage
```python
from simulation.annual_boiler_simulator import AnnualBoilerSimulator

# Create simulator
simulator = AnnualBoilerSimulator(start_date="2024-01-01")

# Generate full year dataset  
annual_data = simulator.generate_annual_data(
    hours_per_day=24,      # Continuous operation
    save_interval_hours=1  # Record every hour
)

# Save dataset
data_file, metadata_file = simulator.save_annual_data(annual_data)
print(f"Dataset saved: {data_file}")
```

## Quick Test Pattern
```python
from simulation.annual_boiler_simulator import AnnualBoilerSimulator
import pandas as pd

# Create simulator with short duration
simulator = AnnualBoilerSimulator(start_date="2024-01-01")
simulator.end_date = simulator.start_date + pd.DateOffset(days=2)  # 2-day test

# Generate test dataset
test_data = simulator.generate_annual_data(
    hours_per_day=24,
    save_interval_hours=2  # Every 2 hours for testing
)

print(f"Test records: {len(test_data)}")
```

## Validation Testing Pattern
```python
from simulation.debug_script import main as run_validation

# Run comprehensive validation
success = run_validation()

if success:
    print("System ready for annual simulation")
else:
    print("Fix issues before running annual simulation")
```

---

# DATA STRUCTURE REFERENCE

## Generated Dataset Columns (148 total)

**Timestamp & Operational (7 columns):**
- `timestamp`, `year`, `month`, `day`, `hour`, `day_of_year`, `season`

**Operating Conditions (8 columns):**
- `load_factor`, `ambient_temp_F`, `ambient_humidity_pct`, `coal_quality`
- `coal_rate_lb_hr`, `air_flow_scfh`, `fuel_input_btu_hr`, `flue_gas_flow_lb_hr`

**System Performance (13 columns):**
- `system_efficiency`, `final_steam_temp_F`, `stack_temp_F`
- `energy_balance_error_pct`, `solution_converged`
- Various energy flows and temperatures

**Coal & Combustion (17 columns):**
- Coal composition, heating values, combustion parameters

**Emissions (11 columns):**
- NOx, CO, CO2, SO2 concentrations and mass flows

**Soot Blowing (16 columns):**
- Section-specific soot blowing status and effectiveness

**Fouling Factors (42 columns):**
- Gas and water fouling factors for all boiler sections

**Section Heat Transfer (42 columns):**
- Temperature and heat transfer data for all sections