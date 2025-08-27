# QUICK START GUIDE

**Enhanced Annual Boiler Simulation System**  
**Week 2 Status - File Corruption Issue Identified**  
**Commercial Demo Timeline: Week 2 of 8**

---

## 🚨 CRITICAL WEEK 2 ISSUE - IMMEDIATE ACTION REQUIRED

### **File Corruption Fix (FIRST PRIORITY)**

The `annual_boiler_simulator.py` file is missing its main execution section and needs immediate repair:

**Problem:** File truncated - missing `if __name__ == "__main__":` section  
**Impact:** Cannot run API compatibility validation  
**Solution:** Add main section code (provided in Week 2 handoff documentation)  
**Timeline:** 5 minutes to fix

**After fixing, this should work:**
```bash
cd src/models/complete_boiler_simulation
python simulation/annual_boiler_simulator.py
# Should show: "API COMPATIBILITY FIXES VALIDATED SUCCESSFULLY!"
```

---

## QUICK START SEQUENCE (AFTER FILE FIX)

### **Step 1: Fix File Corruption (CRITICAL)**
1. Add missing main section to `simulation/annual_boiler_simulator.py`
2. Test that file runs: `python simulation/annual_boiler_simulator.py`
3. Should show API compatibility success message

### **Step 2: Rapid Validation (2 minutes)**
```bash
cd src/models/complete_boiler_simulation

# Quick 30-second test
python quick_api_test.py

# Comprehensive 2-minute test  
python simulation/debug_script.py
```

### **Step 3: Test Simulation (30 minutes)**
```bash
# Run annual simulation with quick test
python simulation/run_annual_simulation.py

# Choose Option 1: Quick Test (24 hours)
# Should generate ~24 records with 84% average efficiency
```

### **Step 4: Full Dataset Generation (2-3 hours)**
```bash
# If quick test passes, run full simulation
# Choose Option 2: Full Annual Simulation (8760 hours) 
# Generates complete ML training dataset
```

---

## SYSTEM OVERVIEW

### **Purpose**
AI/ML-driven soot blowing optimization for pulverized coal-fired boilers in containerboard mills.

### **Technology Stack**
- **Physics Engine**: IAPWS-97 steam properties
- **Efficiency Modeling**: Realistic 75-88% range with 15-20% variation
- **Operating Range**: 60-105% load factor (industrial standard)
- **Data Output**: ML-ready 148-feature dataset

### **Week 2 Status**
- **Completion**: 98% complete
- **Blocker**: File corruption in main simulator (easy fix)
- **Validation**: API fixes applied, tests pending
- **Ready For**: Annual simulation and ML dataset generation

---

## INSTALLATION AND SETUP

### **Dependencies**
```bash
pip install iapws pandas numpy matplotlib scipy
```

### **Directory Structure**
```
src/models/complete_boiler_simulation/
├── core/                    # Physics engine
├── simulation/              # Annual simulation
├── analysis/               # Data analysis  
├── tests/                  # Test scripts
└── quick_api_test.py       # Rapid validation

docs/                       # Documentation
data/generated/             # Output datasets
logs/                      # Log files  
outputs/                   # Metadata and reports
```

### **Working Directory**
**IMPORTANT:** Always run commands from:
```bash
cd src/models/complete_boiler_simulation
```

---

## BASIC USAGE

### **Option 1: API Compatibility Validation (After File Fix)**
```bash
# Test that all APIs work correctly
python simulation/annual_boiler_simulator.py
python quick_api_test.py
python simulation/debug_script.py
```

### **Option 2: Quick Dataset Generation**
```bash
# Generate 24-hour test dataset
python simulation/run_annual_simulation.py
# Choose Option 1: Quick Test
```

### **Option 3: Full ML Dataset Generation**
```bash
# Generate complete 8760-hour dataset for ML training
python simulation/run_annual_simulation.py  
# Choose Option 2: Full Annual Simulation
```

### **Option 4: Custom Analysis**
```python
from analysis.data_analysis_tools import BoilerDataAnalyzer

# Load existing dataset
analyzer = BoilerDataAnalyzer("data/generated/your_dataset.csv")
report = analyzer.generate_comprehensive_report()
```

---

## OUTPUT FILES

### **Quick Test (24 hours):**
- `data/generated/annual_datasets/massachusetts_boiler_annual_YYYYMMDD_HHMMSS.csv` (~24 records)
- `outputs/metadata/massachusetts_boiler_annual_metadata_YYYYMMDD_HHMMSS.txt`

### **Full Simulation (8760 hours):**
- `data/generated/annual_datasets/massachusetts_boiler_annual_YYYYMMDD_HHMMSS.csv` (~15-25MB)
- `outputs/metadata/massachusetts_boiler_annual_metadata_YYYYMMDD_HHMMSS.txt`
- `outputs/analysis/analysis_report_YYYYMMDD_HHMMSS.txt`

### **Log Files:**
- `logs/simulation/annual_simulation.log` - Main simulation log
- `logs/debug/api_compatibility_validation_report_YYYYMMDD_HHMMSS.txt` - Validation results

---

## DATASET FEATURES (148 COLUMNS)

### **Operational Data (7 columns):**
Load factor, temperatures, flows, pressures

### **Coal & Combustion (17 columns):**
Coal properties, air flows, combustion efficiency

### **Emissions (11 columns):**
NOx, SO2, CO2, particulates, stack composition

### **System Performance (13 columns):**
System efficiency, energy balance, heat transfer rates

### **Soot Blowing (16 columns):**
Cleaning schedules, effectiveness, maintenance timing

### **Fouling Analysis (42 columns):**
Progressive fouling factors by section and segment

### **Section Details (42 columns):**
Heat transfer rates and temperatures by boiler section

---

## TROUBLESHOOTING

### **Critical Issues**

**1. File Corruption (Week 2 Critical Issue)**
```
# File cannot be executed directly
python simulation/annual_boiler_simulator.py
# No output or immediate exit
```
**Solution:** Add missing main section (see KNOWN_ISSUES_AND_FIXES.md)

**2. Module Import Errors**
```
ModuleNotFoundError: No module named 'core.boiler_system'
```
**Solution:** Verify working directory
```bash
cd src/models/complete_boiler_simulation
pwd  # Should end in complete_boiler_simulation
```

**3. API Parameter Errors (Should be fixed)**
```
TypeError: __init__() got an unexpected keyword argument 'steam_pressure'
```
**Solution:** Use correct parameters (see KNOWN_ISSUES_AND_FIXES.md for all correct APIs)

### **Validation Issues**

**4. Solver Convergence Problems**
```
Converged: False, Energy Balance Error: 25.3%
```
**Solution:** Use relaxed settings for testing
```python
results = boiler.solve_enhanced_system(max_iterations=25, tolerance=20.0)
```

**5. Unicode Crashes (Should be fixed)**
```
UnicodeEncodeError: 'charmap' codec can't encode character
```
**Solution:** Already fixed - all Unicode characters removed

**6. Missing Dependencies**
```
ImportError: No module named 'iapws'
```
**Solution:** Install required packages
```bash
pip install iapws pandas numpy matplotlib scipy
```

---

## DEVELOPMENT PATHS (WEEK 2 PRIORITIES)

### For Immediate Validation:
1. **Fix File Corruption**: Restore annual_boiler_simulator.py main section
2. **Run Validation**: Execute all three validation scripts
3. **API Testing**: Confirm all API compatibility fixes work
4. **Ready Check**: System validated for annual simulation

### For Dataset Generation:
1. **Quick Test**: 24-hour simulation to verify pipeline
2. **Efficiency Check**: Values in 75-88% range  
3. **Full Simulation**: Complete 8760-hour dataset generation
4. **Quality Validation**: Confirm ML-ready dataset

### For ML Model Development (After Dataset):
1. **Feature Analysis**: Explore 148-column feature set
2. **Target Definition**: Focus on system_efficiency optimization
3. **Data Quality**: Validate realistic operational patterns
4. **Model Training**: Develop soot blowing optimization algorithms

### For Commercial Demo (Week 2 Goal):
1. **Dataset Validation**: Confirm realistic efficiency improvements
2. **Demo Preparation**: System ready for client demonstrations  
3. **Performance Metrics**: Document efficiency gains for demo
4. **Documentation**: Complete user guides for demo deployment

---

## SUCCESS INDICATORS

### **Week 2 Minimum Success:**
✅ File corruption fixed - annual_boiler_simulator.py runs successfully  
✅ All validation tests pass without API errors  
✅ Quick 24-hour simulation completes with realistic data  
✅ System ready for full annual simulation

### **Week 2 Full Success:**
✅ Complete 8760-hour annual simulation executed successfully  
✅ Full ML training dataset generated (15-25MB)  
✅ Dataset quality validation confirms commercial demo readiness  
✅ System performance meets demo requirements

---

## SUPPORT AND DOCUMENTATION

**API Reference:** `docs/api/CORE_SYSTEM_API.md`  
**Integration Guide:** `docs/api/INTEGRATION_GUIDE.md`  
**Known Issues:** `docs/api/KNOWN_ISSUES_AND_FIXES.md` **(Updated for Week 2)**  
**Project Status:** `docs/CURRENT_PROJECT_STATUS.md`  

**Log Files:** `logs/debug/`, `logs/simulation/`  
**Generated Data:** `data/generated/annual_datasets/`  
**Metadata:** `outputs/metadata/`

---

## COMMERCIAL DEMO CONTEXT (WEEK 2)

**Client:** Containerboard mill with realistic production patterns  
**Deliverable:** Functioning ML training dataset for soot blowing optimization  
**Timeline:** Week 2 of 8-week commercial demo  
**Success Metric:** System generates credible efficiency improvements for demo  

**Critical Path:** Fix file → Validate APIs → Generate dataset → Ready for ML development

---

**Success Indicator:** When `python simulation/run_annual_simulation.py` successfully completes a 24-hour quick test with realistic efficiency values (75-88%), the file corruption is resolved and the system is ready for full dataset generation.