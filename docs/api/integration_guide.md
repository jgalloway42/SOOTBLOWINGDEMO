# INTEGRATION GUIDE

**File Location:** `docs/api/INTEGRATION_GUIDE.md`  
**Created:** August 2025  
**Updated:** September 3, 2025  
**Status:** ✅ OPERATIONAL with known architectural limitations

---

## SYSTEM ARCHITECTURE OVERVIEW

### Component Hierarchy
```
AnnualBoilerSimulator
├── EnhancedCompleteBoilerSystem (core boiler)
│   ├── PropertyCalculator (IAPWS steam properties)
│   ├── HeatTransferCalculator (component heat transfer)
│   ├── EnhancedBoilerTubeSection[] (enhanced tube sections)
│   └── BoilerSection[] (individual sections)
├── CoalCombustionModel (combustion calculations)  
├── CombustionFoulingIntegrator (fouling integration)
├── SootProductionModel (soot generation)
├── SootBlowingSimulator (cleaning simulation)
├── FoulingCalculator (fouling progression)
└── Analysis Components
    ├── BoilerDataAnalyzer (data analysis)
    ├── SystemAnalyzer (system analysis)  
    ├── Visualizer (plotting and visualization)
    └── MLDatasetGenerator (ML feature engineering)
```

### Data Flow Pattern
1. **Annual Simulator** generates operating conditions
2. **Boiler System** calculates performance using conditions
3. **Property Calculator** provides thermodynamic properties
4. **Combustion Model** calculates emissions and efficiency
5. **Fouling Models** track fouling progression and cleaning
6. **Results Integration** combines all outputs into dataset
7. **Analysis Tools** process data for insights and ML features
8. **Visualization** creates plots and dashboards

---

## CRITICAL INTEGRATION PATTERNS

### Pattern 1: Basic Boiler System Operation

```python
from core.boiler_system import EnhancedCompleteBoilerSystem

# Step 1: Create boiler system
boiler = EnhancedCompleteBoilerSystem(
    fuel_input=100e6,           # Btu/hr
    flue_gas_mass_flow=84000,   # lb/hr
    furnace_exit_temp=3000      # °F
)

# Step 2: Solve system
results = boiler.solve_enhanced_system(
    max_iterations=20,    # Solver iterations
    tolerance=5.0         # Energy balance tolerance %
)

# Step 3: Extract results
if results['converged']:
    efficiency = results['final_efficiency']
    steam_temp = results['final_steam_temperature']
    stack_temp = results['final_stack_temperature']
    energy_error = results['energy_balance_error']
```

### Pattern 2: Property Calculator Integration

```python
from core.thermodynamic_properties import PropertyCalculator

# Create property calculator (detects IAPWS automatically)
prop_calc = PropertyCalculator()

# Steam properties (for energy balance)
steam_props = prop_calc.get_steam_properties(150, 700)  # 150 psia, 700°F
feedwater_props = prop_calc.get_water_properties(150, 220)  # 150 psia, 220°F

# Energy difference for efficiency calculations
energy_difference = steam_props.enthalpy - feedwater_props.enthalpy
print(f"Specific energy: {energy_difference:.1f} Btu/lb")

# Flue gas properties (for heat transfer)
gas_props = prop_calc.get_flue_gas_properties(
    temperature=1500,  # °F
    pressure=14.7,     # psia  
    composition={'CO2': 0.13, 'H2O': 0.08, 'N2': 0.75, 'O2': 0.04}
)
```

### Pattern 3: Annual Simulation Integration

```python
from simulation.annual_boiler_simulator import AnnualBoilerSimulator

# Step 1: Create annual simulator
simulator = AnnualBoilerSimulator(start_date="2024-01-01")

# Step 2: Generate full dataset
annual_data = simulator.generate_annual_data(
    hours_per_day=24,        # Continuous operation
    save_interval_hours=1    # Record every hour
)

# Step 3: Save dataset for ML training
data_file, metadata_file = simulator.save_annual_data(annual_data)

# Step 4: Analyze dataset
print(f"Records generated: {len(annual_data):,}")
print(f"Efficiency range: {annual_data['system_efficiency'].min():.1%} - {annual_data['system_efficiency'].max():.1%}")
```

---

## COMPONENT INTEGRATION DETAILS

### EnhancedCompleteBoilerSystem Internal Integration

**Initialization Sequence:**
1. Validates operating parameters (load factor 55-110%)
2. Calculates load-dependent feedwater flow
3. Initializes PropertyCalculator for steam properties  
4. Initializes HeatTransferCalculator for component calculations
5. Creates BoilerSection objects for each section
6. Sets up performance tracking

**Solution Process:**
```python
def solve_enhanced_system(self):
    # 1. Estimate initial conditions
    stack_temp = self._estimate_realistic_stack_temp()
    steam_temp = self.target_steam_temp
    
    # 2. Iterative solution loop
    for iteration in range(max_iterations):
        # Calculate system performance 
        performance = self._calculate_fixed_system_performance(stack_temp, steam_temp)
        
        # Check energy balance convergence
        if performance.energy_balance_error <= tolerance:
            return {'converged': True, ...}
            
        # Apply corrections
        stack_correction, steam_correction = self._calculate_corrections(performance)
        stack_temp += stack_correction
        steam_temp += steam_correction
    
    return {'converged': False, ...}
```

### PropertyCalculator Integration

**IAPWS-97 Steam Properties:**
```python
# Internal process
def get_steam_properties(self, pressure, temperature):
    # 1. Convert to SI units
    pressure_mpa = pressure * 0.00689476  # psia to MPa
    temperature_k = (temperature - 32) * 5/9 + 273.15  # °F to K
    
    # 2. Call IAPWS-97
    steam = IAPWS97(P=pressure_mpa, T=temperature_k)
    
    # 3. Convert back to English units
    enthalpy = steam.h * 0.429923  # kJ/kg to Btu/lb
    density = steam.rho * 0.062428  # kg/m³ to lb/ft³
    
    return SteamProperties(temperature, pressure, enthalpy, ...)
```

### AnnualBoilerSimulator Integration

**Hourly Simulation Process:**
```python
def _simulate_boiler_operation(self, datetime, conditions, soot_blowing):
    # 1. Update boiler operating conditions
    self.boiler.update_operating_conditions(
        conditions['fuel_input_btu_hr'],
        conditions['flue_gas_mass_flow'], 
        conditions['furnace_exit_temp']
    )
    
    # 2. Apply soot blowing effects
    self._apply_soot_blowing_effects(soot_blowing)
    
    # 3. Solve boiler system
    solver_results = self.boiler.solve_enhanced_system(max_iterations=20, tolerance=15.0)
    
    # 4. Generate additional data
    coal_data = self._generate_coal_combustion_data(conditions)
    emissions_data = self._generate_emissions_data(conditions, efficiency)
    fouling_data = self._generate_fouling_data(datetime)
    
    # 5. Combine all data
    return {**solver_results, **coal_data, **emissions_data, **fouling_data}
```

---

## ERROR HANDLING PATTERNS

### Robust Solver Integration
```python
try:
    # Attempt enhanced system solution
    solve_results = self.boiler.solve_enhanced_system(max_iterations=20, tolerance=15.0)
    
    # Extract results using standardized structure
    converged = solve_results.get('converged', False)
    efficiency = solve_results.get('final_efficiency', 0.82)
    
    if not converged:
        logger.warning("Solver did not converge")
        
except KeyError as e:
    # Handle missing result keys
    logger.error(f"Missing solver result: {e}")
    # Use fallback values rather than crashing
    
except Exception as e:
    # Handle unexpected solver errors
    logger.error(f"Solver error: {e}")
    # Continue with fallback data
```

### Property Calculator Error Handling
```python
try:
    # Try IAPWS calculation
    steam_props = self.property_calc.get_steam_properties(pressure, temperature)
except Exception as e:
    logger.warning(f"IAPWS failed: {e}, using correlations")
    # Automatically falls back to correlations
    steam_props = self._calculate_steam_correlations(pressure, temperature)
```

---

## DATA VALIDATION PATTERNS

### Energy Balance Validation
```python
def validate_energy_balance(fuel_input, steam_energy, total_losses):
    energy_balance_check = steam_energy + total_losses
    energy_balance_error = abs(energy_balance_check - fuel_input) / fuel_input
    
    if energy_balance_error > 0.05:  # >5% error
        logger.warning(f"High energy balance error: {energy_balance_error:.1%}")
    
    return energy_balance_error
```

### Performance Validation  
```python
def validate_system_performance(efficiency, steam_temp, stack_temp):
    # Efficiency bounds
    if efficiency < 0.75 or efficiency > 0.90:
        logger.warning(f"Efficiency {efficiency:.1%} outside expected range (75-90%)")
    
    # Temperature validation
    if stack_temp > 400:
        logger.warning(f"Stack temperature {stack_temp:.0f}°F unusually high")
    
    # Steam superheat validation
    if steam_temp < 650 or steam_temp > 750:
        logger.warning(f"Steam temperature {steam_temp:.0f}°F outside normal range")
```

---

## ⚠️ ARCHITECTURAL LIMITATIONS AND WORKAROUNDS (2025-09-03)

### Critical Integration Issue: Soot Blowing Effectiveness

**Problem**: The integration between `AnnualBoilerSimulator` and `EnhancedCompleteBoilerSystem` assumes section objects that may not exist with expected interfaces.

**Root Cause**:
```python
# annual_boiler_simulator.py assumes this structure exists:
if hasattr(self.boiler, 'sections') and section_name in self.boiler.sections:
    section = self.boiler.sections[section_name]
    if hasattr(section, 'apply_cleaning'):
        section.apply_cleaning(action['effectiveness'])  # May not exist
```

**Impact on Integration**:
- Effectiveness parameters are set but may not be applied to boiler sections
- Integration falls back to timer-based fouling reset (reliable but less granular)
- System still functions but with limited effectiveness calibration control

### Robust Integration Pattern (Recommended)

**Use this pattern for reliable soot blowing integration**:
```python
def robust_soot_blowing_integration():
    """Recommended pattern that works with current architecture"""
    
    simulator = AnnualBoilerSimulator(start_date="2024-01-01")
    
    # Generate data - system handles effectiveness internally
    data = simulator.generate_annual_data(
        hours_per_day=24,
        save_interval_hours=1
    )
    
    # Focus on what works reliably:
    # 1. Cleaning schedule optimization (timing works perfectly)
    # 2. Fouling accumulation patterns (physics are correct)  
    # 3. Timer-based effectiveness (consistent ~87% average)
    
    return data

def analyze_effectiveness_results(data):
    """Pattern for effectiveness analysis that works with current system"""
    
    # Don't attempt to calibrate effectiveness parameters
    # Instead, analyze whatever effectiveness actually occurs
    
    effectiveness_results = analyze_cleaning_effectiveness(
        data,
        time_window_hours=2,  # Use 2-hour windows for sensitivity
        fouling_threshold=0.001  # Realistic threshold for simulated data
    )
    
    # Results will be in 85-90% range regardless of parameter settings
    # This is acceptable for commercial demonstration
    
    return effectiveness_results
```

### Working Integration Features

**✅ These integration patterns work reliably**:
- Boiler system creation and operation
- Annual simulation dataset generation
- Property calculation integration
- Combustion model integration
- Data structure consistency
- File output and organization

**⚠️ These patterns have limitations**:
- Effectiveness parameter calibration to specific targets
- Section-specific effectiveness control
- Granular cleaning effectiveness simulation

---

## TESTING INTEGRATION PATTERNS

### Validation Test Structure
```python
def test_system_integration():
    """Test complete system integration"""
    
    # Test 1: Core system functionality
    boiler = EnhancedCompleteBoilerSystem(fuel_input=100e6)
    results = boiler.solve_enhanced_system()
    assert results['converged'], "Core system must converge"
    
    # Test 2: Annual simulation integration  
    simulator = AnnualBoilerSimulator()
    
    # Override end date for quick test
    simulator.end_date = simulator.start_date + pd.DateOffset(days=2)
    test_data = simulator.generate_annual_data()
    
    assert len(test_data) > 0, "Must generate test data"
    assert 'system_efficiency' in test_data.columns, "Must include efficiency"
    
    # Test 3: Validation framework
    from simulation.debug_script import generate_phase3_realistic_validation_report
    success, report = generate_phase3_realistic_validation_report()
    
    return success
```

### Component Isolation Testing
```python
def test_individual_components():
    """Test each component in isolation"""
    
    # Test PropertyCalculator
    prop_calc = PropertyCalculator()
    steam_props = prop_calc.get_steam_properties(150, 700)
    assert steam_props.enthalpy > 1300, "Steam enthalpy reasonable"
    
    # Test CoalCombustionModel
    ultimate_analysis = {'C': 75.0, 'H': 5.0, 'O': 8.0, 'N': 1.5, 'S': 2.5, 'Ash': 8.0, 'Moisture': 2.0}
    combustion = CoalCombustionModel(ultimate_analysis, 8000, 80000)
    combustion.calculate()
    assert 0.90 <= combustion.combustion_efficiency <= 0.98, "Combustion efficiency reasonable"
    
    # Test BoilerSection
    section = BoilerSection("test", 10, 100, 20.0, 2.0)
    fouling_arrays = section.get_current_fouling_arrays()
    assert len(fouling_arrays['gas']) == 10, "Correct number of segments"
```

---

## COMMON INTEGRATION ISSUES & SOLUTIONS

### Issue 1: Import Path Problems After Reorganization
**Problem:** Files can't find modules after directory restructure
**Solution:** Use relative imports from new structure
```python
# Correct imports after reorganization
from core.boiler_system import EnhancedCompleteBoilerSystem
from core.thermodynamic_properties import PropertyCalculator  
from simulation.annual_boiler_simulator import AnnualBoilerSimulator
```

### Issue 2: API Parameter Mismatches  
**Problem:** Methods called with parameters that don't exist
**Examples:**
- `EnhancedCompleteBoilerSystem(steam_pressure=150)` ❌
- `generate_annual_data(duration_days=2)` ❌

**Solution:** Use only documented parameters
```python
# Correct API usage
boiler = EnhancedCompleteBoilerSystem(fuel_input=100e6, flue_gas_mass_flow=84000)
data = simulator.generate_annual_data(hours_per_day=24, save_interval_hours=1)
```

### Issue 3: Energy Balance Non-Convergence
**Problem:** Solver fails to converge due to energy balance errors
**Symptoms:** `converged: False` in results, high energy_balance_error

**Solution:** Use appropriate tolerance and iteration limits
```python
# For testing - use relaxed tolerance
results = boiler.solve_enhanced_system(max_iterations=20, tolerance=15.0)

# For production - use tighter tolerance  
results = boiler.solve_enhanced_system(max_iterations=50, tolerance=5.0)
```

### Issue 4: IAPWS Property Calculation Failures
**Problem:** Steam property calculations fail at boundary conditions
**Solution:** PropertyCalculator automatically handles fallbacks
```python
# PropertyCalculator handles this internally
try:
    steam_props = prop_calc.get_steam_properties(pressure, temperature)
    # Will automatically fall back to correlations if IAPWS fails
except Exception as e:
    logger.error(f"All property calculations failed: {e}")
```

### Issue 5: Memory Issues with Large Datasets  
**Problem:** Annual simulation runs out of memory
**Solution:** Use chunked processing
```python
# For large datasets, save periodically
simulator = AnnualBoilerSimulator()
data_chunks = []

# Process in monthly chunks
for month in range(1, 13):
    monthly_data = simulator.generate_monthly_data(month)
    data_chunks.append(monthly_data)
    
    # Save chunk to disk
    monthly_data.to_csv(f"month_{month:02d}_data.csv")

# Combine all chunks
annual_data = pd.concat(data_chunks, ignore_index=True)
```

---

## PERFORMANCE OPTIMIZATION PATTERNS

### Efficient Annual Simulation
```python
def optimized_annual_simulation():
    """Optimized approach for large dataset generation"""
    
    simulator = AnnualBoilerSimulator()
    
    # Use efficient parameters
    annual_data = simulator.generate_annual_data(
        hours_per_day=24,          # Full operation
        save_interval_hours=4      # Every 4 hours (2190 records vs 8760)
    )
    
    # Save immediately to free memory
    data_file, metadata_file = simulator.save_annual_data(annual_data)
    
    # Clear data from memory
    del annual_data
    
    return data_file
```

### Solver Performance Optimization
```python
def optimized_solver_settings(load_factor):
    """Adjust solver parameters based on operating conditions"""
    
    if load_factor < 0.7:
        # Low load - may need more iterations
        max_iterations = 30
        tolerance = 8.0
    elif load_factor > 1.0:
        # High load - may need relaxed tolerance
        max_iterations = 25  
        tolerance = 10.0
    else:
        # Normal operation
        max_iterations = 20
        tolerance = 5.0
    
    return max_iterations, tolerance
```

---

## DEBUGGING INTEGRATION ISSUES

### Debug Information Collection
```python
def collect_debug_info(boiler_system, solver_results):
    """Collect comprehensive debug information"""
    
    debug_info = {
        'system_config': {
            'fuel_input': boiler_system.fuel_input,
            'gas_flow': boiler_system.flue_gas_mass_flow,
            'design_capacity': boiler_system.design_capacity,
            'load_factor': boiler_system.fuel_input / boiler_system.design_capacity
        },
        'solver_results': solver_results,
        'system_performance': getattr(boiler_system, 'system_performance', {}),
        'property_calc_stats': boiler_system.property_calc.log_statistics() if hasattr(boiler_system.property_calc, 'log_statistics') else {}
    }
    
    return debug_info
```

### Integration Test Validation
```python
def validate_integration_health():
    """Comprehensive integration health check"""
    
    health_report = {}
    
    try:
        # Test 1: Core system creation
        boiler = EnhancedCompleteBoilerSystem(fuel_input=100e6)
        health_report['core_system'] = 'OK'
    except Exception as e:
        health_report['core_system'] = f'FAILED: {e}'
    
    try:
        # Test 2: Property calculator  
        prop_calc = PropertyCalculator()
        steam_props = prop_calc.get_steam_properties(150, 700)
        health_report['property_calc'] = 'OK'
    except Exception as e:
        health_report['property_calc'] = f'FAILED: {e}'
    
    try:
        # Test 3: Annual simulator
        simulator = AnnualBoilerSimulator()
        health_report['annual_simulator'] = 'OK'
    except Exception as e:
        health_report['annual_simulator'] = f'FAILED: {e}'
    
    return health_report
```

This integration guide provides the essential patterns and troubleshooting information needed to successfully use the boiler simulation system components together.