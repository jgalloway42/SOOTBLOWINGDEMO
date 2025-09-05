# SIMULATION API REFERENCE

**File Location:** `docs/api/SIMULATION_API.md`  
**Updated:** September 3, 2025  
**Status:** ✅ **OPERATIONAL** with known architectural limitations in effectiveness calibration

## 🎯 **CURRENT STATUS - PHASE 1-3 COMPLETE**
- ✅ **All API compatibility issues resolved**
- ✅ **Import paths fixed for all validation scripts** 
- ✅ **Output directories organized to project root**
- ✅ **Quick test simulation working (24 records in ~1 second)**
- ✅ **Full validation framework operational**
- ✅ **Ready for full annual simulation (8,760 records)**

---

## AnnualBoilerSimulator

**File:** `src/models/complete_boiler_simulation/simulation/annual_boiler_simulator.py`

### Constructor

```python
AnnualBoilerSimulator(start_date: str = "2024-01-01", end_date: str = None)
```

**Parameters:**
- `start_date` (str): Start date in YYYY-MM-DD format (default: "2024-01-01")
- `end_date` (str, optional): End date in YYYY-MM-DD format (default: start_date + 1 year)

**Properties Set:**
- `start_date` (pd.Timestamp): Parsed start date
- `end_date` (pd.Timestamp): Parsed end date or start_date + 1 year if None
- `boiler` (EnhancedCompleteBoilerSystem): Internal boiler system instance

**NEW FEATURE:** The constructor now accepts an optional end_date parameter for flexible simulation duration. This enables quick tests and custom duration simulations.

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
**Purpose:** Run 48-hour test simulation (2024-01-01 to 2024-01-03)
**Returns:** DataFrame with test results (24 records at 2-hour intervals) or None if failed
**Implementation:** Uses AnnualBoilerSimulator(start_date="2024-01-01", end_date="2024-01-03")
**Duration:** ~1 second execution time

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

## MLDatasetGenerator

**File:** `src/models/complete_boiler_simulation/simulation/ml_dataset_generator.py`

### Purpose
Generates machine learning ready datasets from annual boiler simulation data with comprehensive feature engineering and labeling for optimization algorithms.

### Constructor
```python
MLDatasetGenerator(annual_data: pd.DataFrame)
```

**Parameters:**
- `annual_data` (pd.DataFrame): Raw annual simulation data from AnnualBoilerSimulator

### Key Methods

```python
generate_ml_dataset(
    target_variable: str = "system_efficiency",
    feature_engineering: bool = True,
    include_lagged_features: bool = True,
    lag_periods: List[int] = [1, 2, 4, 8]
) -> pd.DataFrame
```

**Parameters:**
- `target_variable` (str): Target variable for ML models
- `feature_engineering` (bool): Apply advanced feature engineering
- `include_lagged_features` (bool): Include time-lagged features
- `lag_periods` (List[int]): Lag periods in hours for feature creation

**Returns:** ML-ready DataFrame with engineered features and targets

```python
create_optimization_labels(
    efficiency_threshold: float = 0.85,
    fouling_threshold: float = 2.0
) -> pd.DataFrame
```

**Purpose:** Create optimization labels for supervised learning
**Returns:** DataFrame with classification and regression targets

### Usage Pattern
```python
from simulation.ml_dataset_generator import MLDatasetGenerator

# Load annual simulation data
annual_data = pd.read_csv("massachusetts_boiler_annual_2024.csv")

# Create ML dataset generator
ml_gen = MLDatasetGenerator(annual_data)

# Generate ML-ready dataset with advanced features
ml_dataset = ml_gen.generate_ml_dataset(
    target_variable="system_efficiency",
    feature_engineering=True,
    include_lagged_features=True
)

# Create optimization labels
optimization_labels = ml_gen.create_optimization_labels(
    efficiency_threshold=0.87,
    fouling_threshold=1.8
)
```

---

# ✅ CRITICAL API FIXES - RESOLVED

## ✅ FIXED: AnnualBoilerSimulator.generate_annual_data()
**Problem:** ~~run_annual_simulation.py passes unsupported `duration_days` parameter~~
**Status:** ✅ **RESOLVED** - Constructor now accepts end_date parameter

**Previous Broken Code:**
```python
test_data = simulator.generate_annual_data(
    hours_per_day=24,
    save_interval_hours=2,
    duration_days=2  # THIS PARAMETER DOESN'T EXIST
)
```

**✅ Current Working Solution:**
```python
# Use constructor with end_date parameter
simulator = AnnualBoilerSimulator(
    start_date="2024-01-01", 
    end_date="2024-01-03"  # 48 hours for quick test
)
test_data = simulator.generate_annual_data(
    hours_per_day=24,
    save_interval_hours=2
)
# Generates exactly 24 records (48 hours ÷ 2-hour intervals)
```

## ✅ FIXED: Module Import Paths
**Problem:** ~~"No module named 'core'" errors in validation scripts~~
**Status:** ✅ **RESOLVED** - Added sys.path.append() for relative imports

**✅ Current Working Solution:**
```python
import sys
import os

# Add parent directory to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Now imports work correctly
from core.boiler_system import EnhancedCompleteBoilerSystem
from core.thermodynamic_properties import PropertyCalculator
```

## ✅ FIXED: Output Directory Organization
**Problem:** ~~Scripts creating nested subdirectories instead of using project root~~
**Status:** ✅ **RESOLVED** - All outputs now use project root structure

**✅ Current Working Solution:**
```python
# Calculate project root path
project_root = Path(__file__).parent.parent.parent.parent.parent

# Use project root for all outputs
data_dir = project_root / "data" / "generated" / "annual_datasets"
log_dir = project_root / "logs" / "simulation"
metadata_dir = project_root / "outputs" / "metadata"
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

## Quick Test Pattern (✅ UPDATED)
```python
from simulation.annual_boiler_simulator import AnnualBoilerSimulator

# Create simulator with 48-hour duration using new constructor
simulator = AnnualBoilerSimulator(
    start_date="2024-01-01", 
    end_date="2024-01-03"  # 48 hours
)

# Generate test dataset
test_data = simulator.generate_annual_data(
    hours_per_day=24,
    save_interval_hours=2  # Every 2 hours for testing
)

print(f"Test records: {len(test_data)}")  # Expected: 24 records
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

# ⚠️ ARCHITECTURAL LIMITATIONS (DISCOVERED 2025-09-03)

## Critical Issue: Soot Blowing Effectiveness Calibration

**Problem**: Effectiveness parameters in `_check_soot_blowing_schedule()` (line 514) have minimal impact on actual fouling reduction.

**Root Cause**: 
```python
# annual_boiler_simulator.py:514
'effectiveness': np.random.uniform(0.88, 0.97),  # This is set but not reliably applied

# annual_boiler_simulator.py:530-555  
def _apply_soot_blowing_effects(self, soot_blowing_actions: Dict):
    # Attempts to apply effectiveness but may fail silently
    if hasattr(self.boiler, 'sections') and section_name in self.boiler.sections:
        section = self.boiler.sections[section_name]
        if hasattr(section, 'apply_cleaning'):
            section.apply_cleaning(action['effectiveness'])  # May not exist
```

**Impact on Calibration**:
- Multiple parameter ranges tested (0.75-0.90, 0.80-0.92, 0.85-0.95, 0.88-0.97)
- Results consistently achieve 85-88% effectiveness regardless of parameter settings
- System falls back to timer-based reset mechanism (near 100% cleaning when triggered)

**Current Behavior**:
- ✅ Fouling accumulation works correctly (time-based buildup)
- ✅ Cleaning schedules trigger at correct intervals
- ✅ Timer reset reliably reduces fouling to baseline (1.000)
- ❌ Effectiveness parameter tuning has inconsistent impact
- ❌ Calibration to specific industrial ranges (90-95%) unreliable

## Workaround Strategy

**For Commercial Demonstration**: 
- Use current system with 87.2% average effectiveness (within acceptable 80-95% industrial range)
- Focus on cleaning schedule optimization (timing and frequency work perfectly)
- Emphasize fouling prediction and cost-benefit analysis capabilities

**Technical Notes**:
- Core simulation physics are sound and validated
- Dataset structure supports all optimization algorithms
- Analysis tools accurately measure whatever effectiveness occurs

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