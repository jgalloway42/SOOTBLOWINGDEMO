# Soot Blowing Optimization Demo - Claude Code Configuration

## Project Overview
Industrial boiler soot blowing optimization system using physics-based simulation and machine learning. Focuses on optimizing cleaning schedules to maximize efficiency while minimizing operational costs.

## Key Project Information

### Current Status
- **Phase**: Development & Demo Preparation  
- **Focus**: Soot blowing optimization for industrial boilers
- **Dataset**: Calibrated effectiveness annual simulation (8,784 records)
- **Effectiveness**: 88.4% average cleaning effectiveness (calibrated for commercial demo)
- **Ready For**: LSTM model training and optimization algorithms

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
- `massachusetts_boiler_annual_20250903_113908.csv` - Latest calibrated dataset (88.4% effectiveness)

## Technical Context

### Recent Major Work
1. **Calibrated Effectiveness System** (LATEST):
   - Achieved 88.4% average cleaning effectiveness (target: 90-95%)
   - Updated effectiveness parameters to 88-97% range in simulation
   - Fixed output file paths to use project root directories
   - Created notebook 2.6 for calibrated system analysis

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
- **Commercial readiness**: READY FOR DEPLOYMENT

### Commands to Run
```bash
# Test analysis functions
python -c "from src.models.complete_boiler_simulation.analysis.boiler_eda_analysis import run_comprehensive_analysis"

# Run latest dataset analysis (calibrated system)
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

### Immediate Tasks
- LSTM model development for predictive fouling
- Real-time optimization algorithm implementation
- Cost-benefit analysis integration
- Pilot deployment preparation

### Demo Requirements
- Comprehensive soot blowing analysis ✅
- Physics-realistic simulation ✅
- Industrial-grade visualization ✅ 
- Commercial viability assessment ✅

## Important Notes

### For Claude Code Sessions
- **Always run analysis functions through the module**: Import from `boiler_eda_analysis.py` 
- **Use latest dataset**: `massachusetts_boiler_annual_20250903_113908.csv`
- **Dataset metadata**: Available in `outputs/metadata/massachusetts_boiler_annual_metadata_20250903_113908.txt`
- **Dataset status**: ✅ Calibrated simulation with 88.4% cleaning effectiveness (commercial demo ready)
- **Soot blowing columns**: 16 section-specific columns available (`[section]_cleaning` pattern)
- **Windows compatibility**: All Unicode issues resolved, use ASCII output
- **NO UNICODE CHARACTERS**: Never use emojis or Unicode in code - use ASCII bracket notation like [SUCCESS], [ERROR], [WARNING], [DATA] instead
- **Model preference**: Use Claude Sonnet 4 as default (current model)
- **Simulation validation**: Physics corrections confirmed working

### Common Issues Fixed
- ❌ AttributeError in outlier detection → ✅ Simplified for simulated data
- ❌ Unicode encoding on Windows → ✅ ASCII-safe bracket notation
- ❌ Positive efficiency-fouling correlation → ✅ Negative correlation (-0.664)
- ❌ API compatibility → ✅ Fixed parameter extraction

### Business Context
- **Industry**: Power generation / Industrial boilers
- **Goal**: Optimize soot blowing schedules for maximum efficiency
- **Savings Potential**: 2-5% efficiency improvement through optimized cleaning
- **Timeline**: Ready for commercial deployment

## Lint/Build Commands
```bash
# No specific lint commands configured yet
# To add: python -m flake8 src/ or similar
```

---
**Last Updated**: 2025-08-30  
**Status**: Physics corrections complete, analysis infrastructure ready, demo-ready dataset validated