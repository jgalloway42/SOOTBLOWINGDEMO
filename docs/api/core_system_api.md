# CORE SYSTEM API REFERENCE

**File Location:** `docs/api/CORE_SYSTEM_API.md`  
**Created:** August 2025  
**Status:** CRITICAL - Required for fixing API compatibility issues

---

## EnhancedCompleteBoilerSystem

**File:** `src/models/complete_boiler_simulation/core/boiler_system.py`

### Constructor

```python
EnhancedCompleteBoilerSystem(
    fuel_input: float = 100e6,
    flue_gas_mass_flow: float = 84000,
    furnace_exit_temp: float = 3000,
    target_steam_temp: float = 700,
    feedwater_temp: float = 220,
    base_fouling_multiplier: float = 1.0
)
```

**Parameters:**
- `fuel_input` (float): Fuel input in Btu/hr (default: 100e6)
- `flue_gas_mass_flow` (float): Flue gas mass flow in lb/hr (default: 84000)
- `furnace_exit_temp` (float): Furnace exit temperature in °F (default: 3000)
- `target_steam_temp` (float): Target steam temperature in °F (default: 700)
- `feedwater_temp` (float): Feedwater temperature in °F (default: 220)
- `base_fouling_multiplier` (float): Base fouling multiplier (default: 1.0)

**IMPORTANT:** Do NOT pass `steam_pressure` parameter - it will cause "unexpected keyword argument" error.

### Key Methods

```python
solve_enhanced_system(max_iterations: int = 50, tolerance: float = 5.0) -> Dict
```

**Parameters:**
- `max_iterations` (int): Maximum solver iterations (default: 50)
- `tolerance` (float): Energy balance tolerance in % (default: 5.0)

**Returns Dictionary with keys:**
- `converged` (bool): Whether solver converged
- `iterations` (int): Number of iterations used
- `final_efficiency` (float): Final system efficiency (0.0-1.0)
- `final_steam_temperature` (float): Final steam temperature in °F
- `final_stack_temperature` (float): Final stack temperature in °F
- `energy_balance_error` (float): Energy balance error (0.0-1.0)
- `system_performance` (Dict): Detailed performance metrics

```python
update_operating_conditions(
    new_fuel_input: float,
    new_flue_gas_flow: float, 
    new_furnace_temp: float
)
```
Updates operating conditions for dynamic simulation.

### Properties
- `design_capacity` (float): Design capacity in Btu/hr
- `feedwater_flow` (float): Feedwater flow in lb/hr
- `system_performance` (Dict): Last calculated performance data

---

## PropertyCalculator  

**File:** `src/models/complete_boiler_simulation/core/thermodynamic_properties.py`

### Constructor
```python
PropertyCalculator()
```
No parameters required. Automatically detects IAPWS and Thermo library availability.

### Key Methods

```python
get_steam_properties(pressure: float, temperature: float) -> SteamProperties
```
**Parameters:**
- `pressure` (float): Pressure in psia
- `temperature` (float): Temperature in °F

**Returns:** SteamProperties object with attributes:
- `temperature` (float): Temperature in °F
- `pressure` (float): Pressure in psia  
- `enthalpy` (float): Enthalpy in Btu/lb
- `density` (float): Density in lb/ft³
- `cp` (float): Specific heat in Btu/lb-°F

```python
get_water_properties(pressure: float, temperature: float) -> SteamProperties
```
Same interface as `get_steam_properties()` - handles both liquid water and steam.

```python
get_flue_gas_properties(
    temperature: float, 
    pressure: float = 14.7,
    composition: Optional[Dict[str, float]] = None
) -> GasProperties
```

---

## CoalCombustionModel

**File:** `src/models/complete_boiler_simulation/core/coal_combustion_models.py`

### Constructor
```python
CoalCombustionModel(
    ultimate_analysis: Dict[str, float],
    coal_lb_per_hr: float,
    air_scfh: float,
    NOx_eff: float = 0.35,
    air_temp_F: float = 80.0,
    air_RH_pct: float = 60.0,
    atm_press_inHg: float = 29.92,
    design_coal_rate: float = 8333.0
)
```

**Parameters:**
- `ultimate_analysis` (Dict): Coal composition with keys: 'C', 'H', 'O', 'N', 'S', 'Ash', 'Moisture'
- `coal_lb_per_hr` (float): Coal feed rate in lb/hr
- `air_scfh` (float): Air flow rate in SCFH
- `NOx_eff` (float): NOx control efficiency (0.0-1.0)
- `design_coal_rate` (float): Design coal rate for load calculations

### Key Methods
```python
calculate(debug: bool = False)
```
Performs combustion calculations. Must be called before accessing properties.

### Properties (available after calculate())
- `combustion_efficiency` (float): Combustion efficiency (0.0-1.0)
- `flame_temp_F` (float): Flame temperature in °F  
- `dry_O2_pct` (float): Dry O2 percentage
- `total_flue_gas_lb_per_hr` (float): Total flue gas flow in lb/hr

---

## HeatTransferCalculator

**File:** `src/models/complete_boiler_simulation/core/heat_transfer_calculations.py`

### Constructor
```python
HeatTransferCalculator()
```
No parameters required. Initializes with PropertyCalculator integration.

### Key Methods
```python
calculate_gas_side_htc(
    gas_flow: float,
    gas_props: GasProperties,
    tube_od: float,
    tube_spacing: float
) -> float
```

```python
calculate_water_side_htc(
    water_flow: float,
    water_props: SteamProperties,
    tube_id: float,
    tube_length: float
) -> float
```

---

## BoilerSection

**File:** `src/models/complete_boiler_simulation/core/fouling_and_soot_blowing.py`

### Constructor  
```python
BoilerSection(
    name: str,
    num_segments: int,
    tube_count: int,
    tube_length: float,
    tube_od: float
)
```

**Parameters:**
- `name` (str): Section name identifier
- `num_segments` (int): Number of segments in section
- `tube_count` (int): Number of tubes in section
- `tube_length` (float): Tube length in feet
- `tube_od` (float): Tube outer diameter in inches

### Key Methods
```python
get_current_fouling_arrays() -> Dict[str, List[float]]
```
Returns dictionary with 'gas' and 'water' fouling factor arrays.

```python
apply_soot_blowing(segment_list: List[int], effectiveness: float = 0.8)
```
Applies soot blowing to specified segments.

---

# CRITICAL API FIXES NEEDED

## Issue #1: EnhancedCompleteBoilerSystem Constructor
**Problem:** debug_script.py passes unsupported `steam_pressure` parameter
**Fix:** Remove `steam_pressure` from constructor calls
**Correct Usage:**
```python
boiler = EnhancedCompleteBoilerSystem(
    fuel_input=100e6,
    flue_gas_mass_flow=84000,
    furnace_exit_temp=3000
    # Do NOT pass steam_pressure
)
```

## Issue #2: Missing steam_pressure Property  
**Problem:** Code tries to access `boiler.steam_pressure` 
**Fix:** Use hardcoded value of 150 psia or add as property
**Note:** Steam pressure is internally set to 150 psia in current implementation

---

# INTEGRATION PATTERNS

## Basic Boiler System Usage
```python
from core.boiler_system import EnhancedCompleteBoilerSystem

# Create boiler system
boiler = EnhancedCompleteBoilerSystem(fuel_input=100e6)

# Solve system
results = boiler.solve_enhanced_system(max_iterations=10, tolerance=5.0)

# Check results
if results['converged']:
    efficiency = results['final_efficiency']
    steam_temp = results['final_steam_temperature'] 
    print(f"Efficiency: {efficiency:.1%}, Steam: {steam_temp:.0f}°F")
```

## Property Calculator Integration
```python  
from core.thermodynamic_properties import PropertyCalculator

prop_calc = PropertyCalculator()
steam_props = prop_calc.get_steam_properties(150, 700)  # 150 psia, 700°F
print(f"Steam enthalpy: {steam_props.enthalpy:.1f} Btu/lb")
```

## Combustion Model Usage
```python
from core.coal_combustion_models import CoalCombustionModel

ultimate_analysis = {
    'C': 75.0, 'H': 5.0, 'O': 8.0, 'N': 1.5, 
    'S': 2.5, 'Ash': 8.0, 'Moisture': 2.0
}

combustion = CoalCombustionModel(
    ultimate_analysis=ultimate_analysis,
    coal_lb_per_hr=8000,
    air_scfh=80000
)

combustion.calculate()
print(f"Combustion efficiency: {combustion.combustion_efficiency:.1%}")
```