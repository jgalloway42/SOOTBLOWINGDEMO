# QUICK START GUIDE

**File Location:** `docs/QUICK_START_GUIDE.md`  
**Created:** August 2025  
**Status:** ESSENTIAL - How to run the system

---

## PREREQUISITES

### Required Dependencies
```bash
pip install iapws          # Steam properties (critical)
pip install thermo         # Gas mixture properties (optional)
pip install pandas numpy   # Data processing  
pip install matplotlib     # Visualization
pip install scipy          # Statistical analysis
```

### Verify Installation
```bash
python -c "import iapws; print('IAPWS OK')"
python -c "import pandas as pd; print('Pandas OK')"
```

---

## FIRST-TIME SETUP

### 1. Navigate to Working Directory
```bash
cd src/models/complete_boiler_simulation
```

### 2. Verify File Structure
```
complete_boiler_simulation/
├── core/                   # Core system files
├── simulation/             # Simulation runners  
├── analysis/               # Analysis tools
└── tests/                  # Test scripts
```

### 3. Test Basic Imports
```bash
python -c "from core.boiler_system import EnhancedCompleteBoilerSystem; print('Core imports working')"
python -c "from simulation.annual_boiler_simulator import AnnualBoilerSimulator; print('Simulation imports working')"
```

If imports fail, check `docs/api/KNOWN_ISSUES_AND_FIXES.md`

---

## BASIC USAGE

### Option 1: Quick System Test (2 minutes)

**Create a simple test file:**
```python
# File: quick_test.py
from core.boiler_system import EnhancedCompleteBoilerSystem

# Create boiler system
boiler = EnhancedCompleteBoilerSystem(fuel_input=100e6)

# Solve system
results = boiler.solve_enhanced_system(max_iterations=10, tolerance=10.0)

# Display results
print(f"Converged: {results['converged']}")
if results['converged']:
    print(f"Efficiency: {results['final_efficiency']:.1%}")
    print(f"Steam Temp: {results['final_steam_temperature']:.0f}°F")
    print(f"Stack Temp: {results['final_stack_temperature']:.0f}°F")
    print(f"Energy Balance Error: {results['energy_balance_error']:.1%}")
```

**Run test:**
```bash
python quick_test.py
```

**Expected Output:**
```
Converged: True
Efficiency: 85.0%
Steam Temp: 700°F
Stack Temp: 280°F
Energy Balance Error: 3.2%
```

### Option 2: Generate Sample Dataset (5 minutes)

```python
# File: sample_dataset.py
import pandas as pd
from simulation.annual_boiler_simulator import AnnualBoilerSimulator

# Create simulator for 2-day test
simulator = AnnualBoilerSimulator(start_date="2024-01-01")
simulator.end_date = simulator.start_date + pd.DateOffset(days=2)

# Generate sample data
sample_data = simulator.generate_annual_data(
    hours_per_day=24,
    save_interval_hours=2  # Every 2 hours = 24 records
)

# Display sample
print(f"Generated {len(sample_data)} records")
print("\nSample columns:")
for col in sample_data.columns[:10]:
    print(f"  {col}")
    
print(f"\nEfficiency range: {sample_data['system_efficiency'].min():.1%} - {sample_data['system_efficiency'].max():.1%}")
```

---

## STANDARD WORKFLOWS

### Workflow 1: Run System Validation

**Purpose:** Verify all components work correctly  
**Time:** 15-30 minutes

```bash
# Run comprehensive validation
python simulation/debug_script.py
```

**Expected Output:**
```
Tests Passed: 4/4
Success Rate: 100.0%
OVERALL PHASE 3 SUCCESS: YES
```

**If validation fails:** Check `logs/debug/` for detailed error logs

### Workflow 2: Generate Annual Dataset

**Purpose:** Create full 8760-hour dataset for ML training  
**Time:** 1-3 hours depending on system performance

```bash
# Run annual simulation with menu interface
python simulation/run_annual_simulation.py
```

**Menu Options:**
- **Option 1:** Quick Test (48 hours) - for testing
- **Option 2:** Full Annual Simulation (8760 hours) - for production
- **Option 3:** Comprehensive (quick test + full simulation)

**Expected Output Files:**
- `data/generated/annual_datasets/massachusetts_boiler_annual_YYYYMMDD_HHMMSS.csv`
- `outputs/metadata/massachusetts_boiler_annual_metadata_YYYYMMDD_HHMMSS.txt`

### Workflow 3: Analyze Generated Dataset

```python
# File: analyze_dataset.py
import pandas as pd
from analysis.data_analysis_tools import BoilerDataAnalyzer

# Load generated dataset
df = pd.read_csv('data/generated/annual_datasets/massachusetts_boiler_annual_LATEST.csv')

# Create analyzer
analyzer = BoilerDataAnalyzer(df)

# Generate comprehensive analysis
report = analyzer.generate_comprehensive_report(save_plots=True)

# Display key findings
print(f"Dataset Summary:")
print(f"  Records: {len(df):,}")
print(f"  Efficiency Range: {df['system_efficiency'].min():.1%} - {df['system_efficiency'].max():.1%}")
print(f"  Load Factor Range: {df['load_factor'].min():.1%} - {df['load_factor'].max():.1%}")
```

---

## TROUBLESHOOTING COMMON ISSUES

### Issue: Import Errors
```
ModuleNotFoundError: No module named 'core.boiler_system'
```

**Solution:** Make sure you're in the correct directory
```bash
cd src/models/complete_boiler_simulation
python -c "import core.boiler_system"  # Should work from this directory
```

### Issue: API Parameter Errors  
```
TypeError: got an unexpected keyword argument 'steam_pressure'
```

**Solution:** Check `docs/api/KNOWN_ISSUES_AND_FIXES.md` for correct parameter usage

### Issue: Solver Convergence Failures
```
Converged: False
Energy Balance Error: 25.3%
```

**Solution:** Use relaxed solver settings for testing
```python
results = boiler.solve_enhanced_system(
    max_iterations=20,    # More iterations
    tolerance=15.0        # Relaxed tolerance for testing
)
```

### Issue: Missing IAPWS Library
```
ImportError: No module named 'iapws'
```

**Solution:** Install required dependency
```bash
pip install iapws
```

---

## PERFORMANCE EXPECTATIONS

### System Performance Targets

**Efficiency Variation:** 15-20% across load range (excellent)  
**Energy Balance Error:** <5% (good), <15% (acceptable for testing)  
**Solver Convergence:** >90% of scenarios  
**Dataset Generation:** 1000-2000 records per hour  

### Expected Results Ranges

**System Efficiency:** 75% - 88%  
**Steam Temperature:** 680°F - 720°F  
**Stack Temperature:** 250°F - 350°F  
**Load Factor Range:** 60% - 105%  

### File Sizes

**48-hour dataset:** ~0.5 MB (24 records)  
**Full annual dataset:** ~15-25 MB (8760 records)  
**Metadata files:** ~5-10 KB  

---

## NEXT STEPS

### For System Development:
1. **Fix Known Issues:** Address API compatibility problems
2. **Energy Balance:** Improve energy balance accuracy
3. **Component Integration:** Fix heat transfer calculations
4. **Validation:** Achieve 100% test pass rate

### For ML Model Development:
1. **Generate Dataset:** Create full annual dataset
2. **Data Quality Check:** Validate dataset completeness  
3. **Feature Engineering:** Extract relevant features for soot blowing optimization
4. **Model Training:** Develop prediction algorithms

### For Production Deployment:
1. **Performance Optimization:** Improve solver stability
2. **User Interface:** Create production interface
3. **Integration Testing:** Test with real plant data
4. **Documentation:** Complete user manuals

---

## SUPPORT AND DOCUMENTATION

**API Reference:** `docs/api/CORE_SYSTEM_API.md`  
**Integration Guide:** `docs/api/INTEGRATION_GUIDE.md`  
**Known Issues:** `docs/api/KNOWN_ISSUES_AND_FIXES.md`  
**Project Status:** `docs/CURRENT_PROJECT_STATUS.md`  

**Log Files:** `logs/debug/`, `logs/simulation/`  
**Generated Data:** `data/generated/annual_datasets/`  
**Metadata:** `outputs/metadata/`

---

**Success Indicator:** When you can run `python simulation/run_annual_simulation.py` and generate a complete 8760-hour dataset without errors, the system is ready for ML model development.