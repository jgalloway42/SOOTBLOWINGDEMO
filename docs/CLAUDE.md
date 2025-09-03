# Soot Blowing Optimization Demo - Claude Code Configuration

## Project Overview
Industrial boiler soot blowing optimization system using physics-based simulation and machine learning. Focuses on optimizing cleaning schedules to maximize efficiency while minimizing operational costs.

## Key Project Information

### Current Status
- **Phase**: Early Development - Simulation Issues Identified  
- **Focus**: Soot blowing optimization for industrial boilers
- **Dataset**: ❌ **No working dataset** - simulation has architectural flaws
- **Effectiveness**: 87.2% from broken simulation (not realistic)
- **Status**: ⚠️ **Major development work required** - core simulation needs fixing
- **Next Steps**: Fix simulation physics, generate realistic dataset, build optimizer

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

### ⚠️ **CRITICAL: Simulation Architecture Broken (2025-09-03)**
- **Effectiveness calculation**: Completely broken - parameters don't affect fouling reduction
- **Dataset validity**: Current datasets are NOT realistic due to simulation flaws
- **Optimization**: Cannot proceed without working simulation and realistic dataset
- **Demo readiness**: ❌ **NOT READY** - major architectural fixes required first

## Technical Context

### Recent Major Work
1. **Critical Simulation Flaws Identified** (LATEST - 2025-09-03):
   - **Discovered**: Effectiveness parameters are completely non-functional
   - **Impact**: Generated datasets are not realistic for optimization work
   - **Root Cause**: Broken architecture assumes objects that don't exist
   - **Status**: ❌ **Simulation must be fixed before proceeding**

2. **Previous Calibration Work**:
   - Fixed output file paths to use project root directories  
   - Created notebook 2.6 for effectiveness analysis
   - Generated multiple calibrated datasets with varying parameter ranges

2. **Physics Corrections Applied** (VALIDATED):
   - Fixed efficiency-fouling correlation (now negative as expected)
   - Implemented time-based fouling accumulation (time since last cleaning)
   - Added CEMS stack temperature correlations with fouling
   - Resolved API compatibility issues

2. **Analysis Infrastructure**:
   - Comprehensive soot blowing effectiveness analysis
   - Cleaning schedule optimization algorithms
   - Industrial-grade visualization dashboards
   - Refactored code into reusable modules

3. **Windows Compatibility Fixes**:
   - Replaced Unicode emojis with ASCII-safe alternatives
   - Fixed AttributeError in outlier detection (simplified for simulated data)
   - All analysis functions now run without encoding errors

### Dataset Validation Results
- **Time-fouling correlation**: +0.974 (excellent)
- **Efficiency-fouling correlation**: -0.664 (excellent) 
- **Load factor compliance**: 100% within 60-105%
- **Parameter variation**: Only 3 constant columns
- **Effectiveness Performance**: 87.2% average (within 80-95% industrial range)
- **Commercial readiness**: ✅ **READY FOR DEPLOYMENT** with known architectural constraints

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

### Immediate Critical Tasks
1. **Fix simulation architecture** - Make effectiveness parameters actually work
2. **Generate realistic dataset** - With working fouling/cleaning physics  
3. **Build optimization model** - For real-time soot blowing recommendations
4. **Develop demo interface** - To show optimization recommendations

### Demo Requirements Status
- Comprehensive soot blowing analysis: ❌ **Needs realistic dataset**
- Physics-realistic simulation: ❌ **Broken - must fix**
- Industrial-grade visualization: ⚠️ **Exists but needs real data** 
- Commercial viability assessment: ❌ **Cannot assess without working system**

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
- **Dataset status**: BROKEN - simulation architecture flawed, datasets unrealistic
- **CRITICAL**: Simulation must be fixed before any meaningful work can proceed
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
**Status**: ❌ **Early development phase** - Critical simulation architecture issues identified, major work required