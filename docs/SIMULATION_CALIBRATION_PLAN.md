# Simulation Calibration Plan

**Created**: 2025-09-03  
**Status**: Planning Phase  
**Priority**: Medium (Optional Enhancement)  

## Current System Status

### ✅ **What's Working Perfectly**
- **Simulation Physics**: Fouling builds realistically, cleaning events reset properly
- **Analysis Tools**: Correctly measure effectiveness and generate insights  
- **Dataset Quality**: 8,784 records with 282 validated cleaning events
- **Section-Specific Control**: All 7 boiler sections have independent cleaning schedules

### ⚠️ **What Needs Fine-Tuning**

#### 1. **Cleaning Effectiveness Calibration**
- **Current**: 98.7% average effectiveness  
- **Target**: 85-90% average effectiveness
- **Reason**: Current is slightly above typical industrial performance range

#### 2. **Output Folder Path Issues**
- **Current**: Simulation outputs to `src/models/data/` and `src/models/outputs/`
- **Target**: Output to project root `data/` and `outputs/` directories
- **Impact**: Minor - doesn't affect functionality, just organization

## Calibration Strategy

### Phase 1: Effectiveness Tuning
**File**: `src/models/complete_boiler_simulation/simulation/annual_boiler_simulator.py`

**Current Effectiveness Range**: Lines 514 in `_check_soot_blowing_schedule()`
```python
'effectiveness': np.random.uniform(0.75, 0.95),  # Current: 75-95%
```

**Proposed Adjustment**:
```python
'effectiveness': np.random.uniform(0.70, 0.88),  # Target: 70-88% 
```

**Expected Result**: Average effectiveness ~85% (middle of industrial range)

### Phase 2: Output Path Correction  
**Files**: 
- `annual_boiler_simulator.py` - Lines with `save_annual_data()`
- `run_annual_simulation.py` - Output path calculations

**Current Issue**: Uses relative paths that resolve to `src/models/` subdirectory

**Solution**: Update path calculations to use project root consistently

### Phase 3: Validation Testing
1. **Generate new test dataset** with calibrated effectiveness
2. **Validate effectiveness range** falls within 80-90%
3. **Confirm output paths** use project root structure
4. **Run analysis tools** to verify continued compatibility

## Implementation Timeline

### Immediate (Optional)
- **Effectiveness tuning**: 15 minutes to adjust and test
- **Path fixes**: 30 minutes to update and validate
- **New dataset generation**: 2-3 minutes for full annual simulation

### Validation
- **Quick test**: Run 48-hour simulation to verify changes
- **Full validation**: Generate complete annual dataset
- **Analysis verification**: Run effectiveness analysis on new data

## Risk Assessment

**Risk Level**: **LOW**  
- Changes are minor parameter adjustments
- Core physics and logic remain unchanged  
- Can easily revert if issues arise
- Current system is fully functional as-is

## Success Criteria

### Effectiveness Calibration Success
- Average effectiveness: 82-88% range
- No cleaning events below 70% effectiveness  
- No cleaning events above 95% effectiveness
- Maintains realistic fouling buildup patterns

### Path Correction Success  
- All output files appear in project root `data/` and `outputs/`
- No duplicate files in `src/models/` subdirectories
- Metadata references correct file locations

## Notes

- **Current system is production-ready** - these are optimization enhancements
- **98.7% effectiveness is not wrong** - just slightly optimistic for industrial demo
- **All analysis tools will work unchanged** - no breaking changes planned
- **Historical datasets preserved** - for comparison and validation

---

**Decision**: These calibrations are **optional enhancements** for a more realistic demo experience. The current system delivers excellent cleaning effectiveness and is ready for commercial demonstration.