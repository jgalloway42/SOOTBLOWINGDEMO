# Soot Blowing Optimization Demo - Claude Code Configuration

## Project Overview
Industrial boiler soot blowing optimization system using physics-based simulation and machine learning. Focuses on optimizing cleaning schedules to maximize efficiency while minimizing operational costs.

## Key Project Information

### Current Status
- **Phase**: Physics Correlation Analysis - Critical Issues Identified
- **Focus**: Soot blowing optimization for industrial boilers  
- **Architecture**: ✅ **WORKING** - Centralized soot blowing logic, boolean detection fixed
- **Physics**: ❌ **CORRELATION FAILURES** - Time-fouling, efficiency-fouling, temperature-fouling all failing
- **Dataset**: ✅ **EVENT DETECTION WORKING** - 1,002 cleaning events detected, but physics impact insufficient
- **Status**: Mixed results - Architecture solid, physics relationships need major strengthening
- **Next Steps**: Fix fouling rates, impact scaling, and temperature dynamics for proper correlations

### Main Directories
```
├── src/
│   ├── models/complete_boiler_simulation/    # Core simulation engine
│   │   ├── simulation/                       # Boiler physics simulation
│   │   └── analysis/                        # EDA and optimization analysis
│   └── data_generation/                     # Dataset generation scripts
├── notebooks/                               # Jupyter analysis notebooks
│   ├── 2.4-*-eda-validated-simulation.ipynb # Comprehensive EDA (FIXED)
│   ├── 2.5-*-physics-corrected-sim-eda.ipynb # Physics validation
│   └── 2.6-*-fouling-corrected-sim-eda.ipynb # Calibrated effectiveness analysis
├── data/generated/annual_datasets/          # Simulation datasets
└── outputs/                                # Analysis reports and metadata
```

### Key Files
- `annual_boiler_simulator.py` - Main simulation engine with physics corrections
- `boiler_eda_analysis.py` - Comprehensive analysis functions (Windows-compatible)
- `massachusetts_boiler_annual_20250903_115813.csv` - Latest calibrated dataset (87.2% effectiveness)

### ⚠️ **CURRENT STATUS: Mixed Architecture/Physics Results (2025-09-04)**
- **Boolean Detection**: ✅ **FIXED** - Notebook analysis now properly detects cleaning events (1,002 vs 0 previously)
- **Centralized Architecture**: ✅ **WORKING** - SootBlowingSimulator unified, fire-side only logic correct
- **Physics Correlations**: ❌ **FAILING** - All three validation metrics below target ranges
- **Temperature Issues**: ❌ **STATIC** - Steam temp constant 700°F, furnace gas outlet unresponsive
- **Fouling Range**: ❌ **TOO NARROW** - Actual 1.000-1.005 vs expected 1.0-1.25 for ML training
- **Status**: Architecture success, physics impact insufficient for realistic behavior

## Technical Context

### Recent Major Work  
1. **PHYSICS CORRELATION ANALYSIS** (LATEST - 2025-09-04):
   - **Discovered**: All three physics validation metrics failing target ranges
   - **Time-Fouling**: r=-0.001 (target >+0.4) - near zero correlation
   - **Fouling-Efficiency**: r=+0.018 (target <-0.25) - wrong direction, too weak
   - **Stack Temp-Fouling**: r=+0.016 (target >+0.2) - insufficient temperature response
   - **Root Causes**: Conservative fouling rates, weak impact scaling, static temperatures
   - **Impact**: ❌ **Physics relationships too weak for meaningful ML training**

2. **BOOLEAN DETECTION SUCCESS** (2025-09-04):
   - **Fixed**: Notebook analysis using `== True` instead of `== 1` 
   - **Result**: Now detects 1,002 cleaning events vs 0 previously
   - **Impact**: ✅ **Analysis framework working, reveals true physics issues**

3. **CENTRALIZED ARCHITECTURE COMPLETED** (2025-09-03):
   - **Unified**: All soot blowing methods into SootBlowingSimulator
   - **Physics**: Fire-side only cleaning logic correctly implemented
   - **Structure**: 90-95% effectiveness framework in place
   - **Impact**: ✅ **Clean architecture ready for physics improvements**

4. **CONSTANT TEMPERATURE ISSUES IDENTIFIED**:
   - **Steam Temperature**: Hardcoded 700°F fallbacks eliminating correlations
   - **Furnace Gas Outlet**: Static calculation unresponsive to fouling levels
   - **Impact**: ❌ **Temperature-fouling relationships impossible with current code**

5. **Previous Foundation Work**:
   - Fixed output file paths to use project root directories  
   - Windows compatibility improvements (Unicode → ASCII)
   - Comprehensive analysis infrastructure and visualization dashboards

### Dataset Validation Results (CURRENT - massachusetts_boiler_annual_20250904_140843.csv)

**❌ PHYSICS CORRELATIONS - FAILING VALIDATION**:
- **Time-fouling correlation**: r=-0.001 (target >+0.4) - MAJOR FAILURE
- **Fouling-efficiency correlation**: r=+0.018 (target <-0.25) - WRONG DIRECTION & WEAK
- **Stack temp-fouling correlation**: r=+0.016 (target >+0.2) - INSUFFICIENT IMPACT

**✅ OPERATIONAL PARAMETERS - MIXED RESULTS**:
- **Load factor compliance**: 100% within 60.0%-103.0% range - EXCELLENT
- **Boolean detection**: 1,002 cleaning events detected - WORKING  
- **Section coverage**: 7 major boiler sections with cleaning schedules - COMPREHENSIVE
- **Constant columns**: 3 static parameters (year, final_steam_temp_F, furnace_gas_temp_out_F)

**⚠️ FOULING BEHAVIOR - INSUFFICIENT RANGE**:
- **Fouling accumulation**: 1.000-1.005 range (expected 1.0-1.25) - TOO NARROW
- **Physics impact**: Minimal efficiency/temperature effects - REQUIRES STRENGTHENING
- **Cleaning effectiveness**: Framework working but shows 0.8% due to physics issues

**Commercial Readiness**: ⚠️ **ARCHITECTURE READY, PHYSICS INSUFFICIENT FOR ML TRAINING**

### Commands to Run
```bash
# Test analysis functions
python -c "from src.models.complete_boiler_simulation.analysis.boiler_eda_analysis import run_comprehensive_analysis"

# Run latest dataset analysis (current system with architectural constraints)
jupyter notebook notebooks/2.6-jdg-boiler-fouling-dataset-fouling-corrected-sim-eda.ipynb

# Run physics validation analysis
jupyter notebook notebooks/2.5-jdg-boiler-fouling-dataset-physics-corrected-sim-eda.ipynb

# Generate new simulation dataset
python src/data_generation/run_annual_simulation.py
```

## Project Backup and Compression

### Creating Efficient Backups
The project can be compressed efficiently by excluding the virtual environment and cache files:

```bash
# Navigate to workspace directory
cd C:\Users\sesa703683\Documents\workspace

# Create compressed backup (recommended - excludes virtual environment)
tar --exclude='SootblowingDemoJune2025/SOOTBLOWER' --exclude='*.pyc' --exclude='__pycache__' -czf SootblowingDemo_backup_$(date +%Y%m%d_%H%M%S).tar.gz SootblowingDemoJune2025/

# Alternative: Include everything (much larger file)
tar -czf SootblowingDemo_full_backup_$(date +%Y%m%d_%H%M%S).tar.gz SootblowingDemoJune2025/
```

### Backup Specifications
- **Compressed Size**: ~158MB (excluding virtual environment)
- **Full Size**: ~777MB (including virtual environment)
- **Compression Ratio**: ~95% reduction when excluding SOOTBLOWER/
- **Location**: Backups created in parent workspace directory

### What's Included in Recommended Backup
- All source code (`src/` directory)
- All notebooks (cleaned and functional)
- Analysis modules and datasets
- Documentation and configuration files
- Generated data and outputs

### What's Excluded (for efficiency)
- Virtual environment (`SOOTBLOWER/` directory)
- Python cache files (`__pycache__/`, `*.pyc`)
- Git objects (can be restored from remote)

### Restoration
To restore from backup:
```bash
cd C:\Users\sesa703683\Documents\workspace
tar -xzf SootblowingDemo_backup_YYYYMMDD_HHMMSS.tar.gz
cd SootblowingDemoJune2025
# Recreate virtual environment and reinstall dependencies as needed
```

## Current Priorities

### Immediate Tasks (Updated after physics fix)
1. **Generate new realistic dataset** - With corrected fouling physics ✅ **Ready to proceed**
2. **Validate physics correction** - Verify furnace fouls faster than air heater
3. **Fix effectiveness calculation architecture** - Make parameters actually work
4. **Build optimization model** - For real-time soot blowing recommendations
5. **Develop demo interface** - To show optimization recommendations

### Demo Requirements Status  
- Physics-realistic simulation: ✅ **FIXED** - Fouling physics now correct
- Comprehensive soot blowing analysis: ⚠️ **Ready when new dataset generated**
- Industrial-grade visualization: ✅ **Exists and ready for real data**
- Commercial viability assessment: ⚠️ **Can assess once new dataset validated**

## Important Notes

### For Claude Code Sessions

#### CRITICAL DEVELOPMENT CONSTRAINTS:
- **NO NEW FEATURES**: Do not add any new features, components, or capabilities unless explicitly requested and approved
- **FOCUS ON CORE ISSUES**: Only work on fixing the broken simulation architecture
- **ASCII ONLY**: Use only ASCII characters in ALL code, documentation, and output - no Unicode, emojis, or special characters
- **PRAGMATIC STATUS**: Use realistic, honest language when describing project status - no marketing speak or overly optimistic assessments

#### Technical Guidelines:
- **Always run analysis functions through the module**: Import from `boiler_eda_analysis.py` 
- **Current dataset**: `massachusetts_boiler_annual_20250903_115813.csv` (NOT VALID - from broken simulation)
- **Dataset metadata**: Available in `outputs/metadata/massachusetts_boiler_annual_metadata_20250903_115813.txt`
- **Dataset status**: PREVIOUS DATASETS INVALID - generated with backwards physics
- **PHYSICS FIXED**: Core simulation now realistic - ready to generate valid dataset
- **Soot blowing columns**: 16 section-specific columns available (`[section]_cleaning` pattern)
- **Character encoding**: ASCII only - use [SUCCESS], [ERROR], [WARNING], [DATA] bracket notation
- **Model preference**: Use Claude Sonnet 4 as default (current model)

### Issues Status
- AttributeError in outlier detection: FIXED - Simplified for simulated data
- Unicode encoding on Windows: FIXED - ASCII-safe bracket notation only
- Positive efficiency-fouling correlation: FIXED - Negative correlation (-0.664)  
- API compatibility: FIXED - Parameter extraction working
- Effectiveness calculation: BROKEN - Core simulation architecture flawed

### Business Context
- **Industry**: Power generation / Industrial boilers
- **Goal**: Optimize soot blowing schedules for maximum efficiency
- **Savings Potential**: 2-5% efficiency improvement through optimized cleaning (theoretical)
- **Timeline**: Major development work required - not ready for deployment

## Lint/Build Commands
```bash
# No specific lint commands configured yet
# To add: python -m flake8 src/ or similar
```

---
**Last Updated**: 2025-09-03  
**Status**: ✅ **Core physics fixed** - Major fouling physics breakthrough, ready for realistic dataset generation