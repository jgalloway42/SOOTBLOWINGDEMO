# ANALYSIS & VISUALIZATION API REFERENCE

**File Location:** `docs/api/analysis_api.md`  
**Created:** September 4, 2025  
**Status:** ✅ OPERATIONAL

---

## BoilerDataAnalyzer

**File:** `src/models/complete_boiler_simulation/analysis/data_analysis_tools.py`

### Constructor

```python
BoilerDataAnalyzer(data: pd.DataFrame)
```

**Parameters:**
- `data` (pd.DataFrame): Boiler operation dataset with required columns

**Key Features:**
- Comprehensive efficiency analysis
- Fouling progression tracking  
- Soot blowing effectiveness evaluation
- Economic optimization analysis
- ML-ready feature extraction
- ASCII-safe logging and output

### Key Methods

```python
analyze_efficiency_trends() -> Dict
```
**Purpose:** Analyze system efficiency trends over time
**Returns:** Dictionary with efficiency statistics and trends

```python
analyze_fouling_progression() -> Dict
```
**Purpose:** Track fouling factor changes across all boiler sections
**Returns:** Dictionary with fouling analysis results

```python
evaluate_soot_blowing_effectiveness() -> Dict
```
**Purpose:** Evaluate effectiveness of soot blowing operations
**Returns:** Dictionary with effectiveness metrics and recommendations

```python
generate_economic_analysis() -> Dict
```
**Purpose:** Perform economic optimization analysis
**Returns:** Dictionary with cost-benefit analysis results

```python
extract_ml_features() -> pd.DataFrame
```
**Purpose:** Extract machine learning ready features from raw data
**Returns:** DataFrame with engineered features for ML models

```python
generate_comprehensive_report(output_filename: str = None) -> str
```
**Purpose:** Generate comprehensive analysis report
**Parameters:**
- `output_filename` (str, optional): Custom output filename
**Returns:** Path to generated report file

### Properties
- `data` (pd.DataFrame): Original dataset
- `logger` (logging.Logger): Analysis logger instance

---

## SystemAnalyzer

**File:** `src/models/complete_boiler_simulation/analysis/analysis_and_visualization.py`

### Constructor

```python
SystemAnalyzer(boiler_system: EnhancedCompleteBoilerSystem)
```

**Parameters:**
- `boiler_system` (EnhancedCompleteBoilerSystem): Boiler system instance to analyze

### Key Methods

```python
print_comprehensive_summary()
```
**Purpose:** Print comprehensive system performance summary to console
**Output:** Detailed system configuration and performance metrics

```python
analyze_system_performance() -> Dict
```
**Purpose:** Analyze current system performance metrics
**Returns:** Dictionary with performance analysis results

```python
generate_efficiency_report() -> Dict
```
**Purpose:** Generate detailed efficiency analysis report
**Returns:** Dictionary with efficiency metrics and recommendations

### Properties
- `system` (EnhancedCompleteBoilerSystem): Referenced boiler system

---

## Visualizer

**File:** `src/models/complete_boiler_simulation/analysis/analysis_and_visualization.py`

### Constructor

```python
Visualizer(boiler_system: EnhancedCompleteBoilerSystem)
```

**Parameters:**
- `boiler_system` (EnhancedCompleteBoilerSystem): Boiler system instance for visualization

### Key Methods

```python
plot_efficiency_trends(data: pd.DataFrame, save_path: str = None)
```
**Purpose:** Create efficiency trend plots
**Parameters:**
- `data` (pd.DataFrame): Time series data
- `save_path` (str, optional): Path to save plot

```python
plot_fouling_progression(data: pd.DataFrame, save_path: str = None)
```
**Purpose:** Visualize fouling factor progression over time
**Parameters:**
- `data` (pd.DataFrame): Data with fouling information
- `save_path` (str, optional): Path to save plot

```python
plot_soot_blowing_impact(data: pd.DataFrame, save_path: str = None)
```
**Purpose:** Visualize soot blowing effectiveness
**Parameters:**
- `data` (pd.DataFrame): Data with soot blowing events
- `save_path` (str, optional): Path to save plot

```python
create_dashboard(data: pd.DataFrame, save_path: str = None)
```
**Purpose:** Create comprehensive visualization dashboard
**Parameters:**
- `data` (pd.DataFrame): Complete dataset
- `save_path` (str, optional): Path to save dashboard

### Properties
- `system` (EnhancedCompleteBoilerSystem): Referenced boiler system
- `figures_dir` (Path): Directory for saving figures

---

# INTEGRATION PATTERNS

## Basic Analysis Usage

```python
from analysis.data_analysis_tools import BoilerDataAnalyzer
import pandas as pd

# Load annual simulation data
data = pd.read_csv("data/generated/annual_datasets/massachusetts_boiler_annual_2024.csv")

# Create analyzer
analyzer = BoilerDataAnalyzer(data)

# Generate comprehensive analysis
efficiency_analysis = analyzer.analyze_efficiency_trends()
fouling_analysis = analyzer.analyze_fouling_progression()
economic_analysis = analyzer.generate_economic_analysis()

# Generate full report
report_path = analyzer.generate_comprehensive_report()
print(f"Analysis report saved: {report_path}")
```

## System Performance Analysis

```python
from analysis.analysis_and_visualization import SystemAnalyzer
from core.boiler_system import EnhancedCompleteBoilerSystem

# Create boiler system
boiler = EnhancedCompleteBoilerSystem(fuel_input=100e6)
results = boiler.solve_enhanced_system()

# Analyze system performance
analyzer = SystemAnalyzer(boiler)
analyzer.print_comprehensive_summary()

performance_metrics = analyzer.analyze_system_performance()
efficiency_report = analyzer.generate_efficiency_report()
```

## Visualization Usage

```python
from analysis.analysis_and_visualization import Visualizer
import pandas as pd

# Create visualizer
visualizer = Visualizer(boiler_system)

# Load data and create visualizations
data = pd.read_csv("annual_data.csv")

# Create individual plots
visualizer.plot_efficiency_trends(data, "outputs/figures/efficiency_trends.png")
visualizer.plot_fouling_progression(data, "outputs/figures/fouling_progression.png")
visualizer.plot_soot_blowing_impact(data, "outputs/figures/soot_blowing_impact.png")

# Create comprehensive dashboard
visualizer.create_dashboard(data, "outputs/figures/boiler_dashboard.png")
```

## ML Feature Engineering

```python
from analysis.data_analysis_tools import BoilerDataAnalyzer

# Load raw simulation data
raw_data = pd.read_csv("massachusetts_boiler_annual_2024.csv")

# Create analyzer and extract ML features
analyzer = BoilerDataAnalyzer(raw_data)
ml_features = analyzer.extract_ml_features()

# Features include:
# - Engineered efficiency metrics
# - Fouling progression indicators
# - Operating condition derivatives
# - Economic performance indicators
# - Predictive maintenance features

# Save ML-ready dataset
ml_features.to_csv("data/processed/ml_features_2024.csv", index=False)
```

---

# OUTPUT DIRECTORY STRUCTURE

The analysis modules automatically create and use the following directory structure:

```
project_root/
├── outputs/
│   ├── analysis/          # Analysis reports and results
│   │   ├── efficiency_reports/
│   │   ├── fouling_analysis/
│   │   └── economic_analysis/
│   └── figures/           # Generated visualizations
│       ├── efficiency_plots/
│       ├── fouling_plots/
│       └── dashboards/
├── data/
│   ├── generated/
│   │   └── annual_datasets/   # Raw simulation data
│   └── processed/         # ML-ready processed data
└── logs/
    └── analysis/          # Analysis execution logs
```

---

# FEATURES AND CAPABILITIES

## Analysis Features
- **Efficiency Analysis**: Trend analysis, performance benchmarking, optimization recommendations
- **Fouling Analysis**: Progression tracking, section-specific analysis, cleaning effectiveness
- **Economic Analysis**: Cost-benefit optimization, maintenance scheduling, fuel efficiency
- **ML Feature Engineering**: Automated feature extraction for machine learning models

## Visualization Features
- **Time Series Plots**: Efficiency trends, temperature profiles, load variations
- **Fouling Visualization**: Heat maps, progression curves, cleaning impact
- **Economic Plots**: Cost analysis, ROI calculations, optimization targets
- **Interactive Dashboards**: Comprehensive system overview with multiple metrics

## Data Processing Features
- **ASCII-Safe Output**: Windows compatibility with safe character encoding
- **Robust Error Handling**: Graceful handling of missing or corrupted data
- **Flexible Input**: Support for various data formats and time ranges
- **Scalable Processing**: Efficient handling of large annual datasets (8,760+ records)