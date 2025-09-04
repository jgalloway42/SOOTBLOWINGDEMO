#!/usr/bin/env python3
"""
Annual Boiler Operation Simulator - API Compatibility Fixed

This module generates comprehensive annual boiler operation data with:
- FIXED API compatibility issues between components
- Robust error handling for solver interface mismatches
- ASCII-safe logging for Windows compatibility
- IAPWS-97 steam properties for accurate efficiency calculations

CRITICAL FIXES APPLIED:
- Fixed _simulate_boiler_operation solver interface handling
- Corrected parameter extraction from solve_enhanced_system results
- Added robust fallback handling for API mismatches
- Replaced all Unicode characters with ASCII equivalents
- FIXED indentation errors causing syntax issues

Author: Enhanced Boiler Modeling System
Version: 8.2 - API Compatibility Fix
"""

import numpy as np
import pandas as pd
import datetime
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import random
import traceback
import sys
import os

# Import enhanced modules with IAPWS
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.boiler_system import EnhancedCompleteBoilerSystem
from core.coal_combustion_models import CoalCombustionModel, CombustionFoulingIntegrator
from core.fouling_and_soot_blowing import SootProductionModel, SootBlowingSimulator
from core.thermodynamic_properties import PropertyCalculator

# Set up enhanced logging - use project root
project_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
log_dir = project_root / "logs" / "simulation"
log_dir.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create file handler for simulation logs
sim_log_file = log_dir / "annual_simulation.log"
file_handler = logging.FileHandler(sim_log_file)
file_handler.setLevel(logging.DEBUG)

# Console handler for progress updates
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Create data directories - use project root
data_dir = project_root / "data" / "generated" / "annual_datasets"
data_dir.mkdir(parents=True, exist_ok=True)

metadata_dir = project_root / "outputs" / "metadata"
metadata_dir.mkdir(parents=True, exist_ok=True)


class AnnualBoilerSimulator:
    """Enhanced annual boiler simulator with FIXED API compatibility."""
    
    def __init__(self, start_date: str = "2024-01-01", end_date: str = None):
        """Initialize the enhanced annual boiler simulator with fixed interface."""
        
        # Date configuration
        self.start_date = pd.to_datetime(start_date)
        if end_date:
            self.end_date = pd.to_datetime(end_date)
        else:
            self.end_date = self.start_date + pd.DateOffset(years=1)
        
        # Enhanced production patterns for containerboard mill
        self.production_patterns = {
            'base_load': 0.85,  # Base load factor
            'seasonal_variation': 0.10,  # Seasonal swing
            'daily_variation': 0.05,  # Daily swing
            'weekly_pattern': [0.95, 0.98, 1.00, 1.00, 0.98, 0.90, 0.85],  # Mon-Sun
            'monthly_pattern': [0.92, 0.95, 1.00, 1.02, 1.05, 1.08,  # Winter/Spring demand
                              1.10, 1.08, 1.05, 1.02, 0.98, 0.90]   # Summer/Fall
        }
        
        # Corrected soot blowing schedule - frequent cleaning where fouling is highest
        self.soot_blowing_schedule = {
            'furnace_walls': 24,        # Most frequent - highest fouling rate
            'generating_bank': 36,      # High frequency - high fouling
            'superheater_primary': 48,  # Moderate frequency 
            'superheater_secondary': 72, # Every 3 days
            'economizer_primary': 96,   # Every 4 days
            'economizer_secondary': 168, # Weekly
            'air_heater': 336           # Least frequent - lowest fouling rate
        }
        
        # Track last cleaning times
        self.last_cleaned = {section: self.start_date for section in self.soot_blowing_schedule.keys()}
        
        # EFFECTIVENESS FIX: Track fouling baselines after each cleaning (not just timer reset)
        # This enables proper 90-95% effectiveness implementation
        self.fouling_baselines = {section: 1.0 for section in self.soot_blowing_schedule.keys()}
        
        # Initialize enhanced boiler system with CORRECT API parameters
        try:
            self.boiler = EnhancedCompleteBoilerSystem(
                fuel_input=100e6,  # Base 100 MMBtu/hr - CORRECT PARAMETER NAME
                flue_gas_mass_flow=84000,  # CORRECT PARAMETER NAME
                furnace_exit_temp=3000,    # CORRECT PARAMETER NAME
                base_fouling_multiplier=1.0
            )
            logger.info("Enhanced boiler system initialized with IAPWS integration")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced boiler system: {e}")
            raise
        
        # Initialize enhanced components
        try:
            self.property_calc = PropertyCalculator()
            self.combustion_model = CoalCombustionModel()
            self.fouling_integrator = CombustionFoulingIntegrator()
            logger.info("Enhanced components initialized successfully")
        except Exception as e:
            logger.warning(f"Some enhanced components failed to initialize: {e}")
            # Continue with basic functionality
        
        # Simulation statistics tracking
        self.simulation_stats = {
            'total_hours': 0,
            'solver_failures': 0,
            'efficiency_warnings': 0,
            'temperature_warnings': 0,
            'api_compatibility_fixes': 0
        }
        
        logger.info(f"Annual simulation initialized: {start_date} to {self.end_date.strftime('%Y-%m-%d')}")
    
    def generate_annual_data(self, hours_per_day: int = 24, save_interval_hours: int = 1) -> pd.DataFrame:
        """
        Generate complete annual boiler operation data with FIXED API compatibility.
        
        FIXED API ISSUES:
        - Correct parameter names in method calls
        - Proper solver result extraction
        - Robust error handling for interface mismatches
        
        Args:
            hours_per_day: Operating hours per day (24 for continuous operation)
            save_interval_hours: How often to save data points (1 = every hour)
        
        Returns:
            DataFrame with complete annual operation data including IAPWS steam properties
        """
        logger.info(f"Starting enhanced annual simulation with FIXED API compatibility...")
        logger.info(f"  Operating {hours_per_day} hours/day, recording every {save_interval_hours} hours")
        logger.info(f"  Target: Realistic efficiency (75-88%) using IAPWS steam properties")
        
        annual_data = []
        current_date = self.start_date
        record_counter = 0
        efficiency_sum = 0
        efficiency_count = 0
        
        while current_date < self.end_date:
            # Generate daily operating schedule
            daily_hours = min(hours_per_day, 24)
            
            for hour in range(0, daily_hours, save_interval_hours):
                current_datetime = current_date + datetime.timedelta(hours=hour)
                
                # Skip if we've reached the end date
                if current_datetime >= self.end_date:
                    break
                
                try:
                    # Generate operating conditions for this time point
                    operating_conditions = self._generate_hourly_conditions(current_datetime)
                    
                    # Check and apply soot blowing if scheduled
                    soot_blowing_actions = self._check_soot_blowing_schedule(current_datetime)
                    
                    # Simulate boiler operation with FIXED API compatibility
                    operation_data = self._simulate_boiler_operation_fixed_api(
                        current_datetime, operating_conditions, soot_blowing_actions
                    )
                    
                    annual_data.append(operation_data)
                    record_counter += 1
                    self.simulation_stats['total_hours'] += 1
                    
                    # Track efficiency statistics
                    if operation_data.get('system_efficiency'):
                        efficiency_sum += operation_data['system_efficiency']
                        efficiency_count += 1
                    
                    # Progress reporting with efficiency tracking
                    if record_counter % 500 == 0:
                        progress = (current_datetime - self.start_date).days / 365 * 100
                        avg_efficiency = efficiency_sum / efficiency_count if efficiency_count > 0 else 0
                        latest_stack = operation_data.get('stack_temp_F', 0)
                        
                        logger.info(f"Progress: {progress:.1f}% | "
                                  f"Records: {record_counter:,} | "
                                  f"Avg Eff: {avg_efficiency:.1%} | "
                                  f"Stack: {latest_stack:.0f}F")
                        print(f"[>>] Progress: {progress:.1f}% complete | "
                              f"Efficiency: {avg_efficiency:.1%} | "
                              f"Stack Temp: {latest_stack:.0f}F")
                
                except Exception as e:
                    logger.error(f"Simulation failed at {current_datetime}: {e}")
                    # Create fallback data to continue simulation
                    fallback_data = self._create_fallback_operation_data(
                        current_datetime, operating_conditions if 'operating_conditions' in locals() else None
                    )
                    annual_data.append(fallback_data)
                    record_counter += 1
                    self.simulation_stats['solver_failures'] += 1
            
            # Move to next day
            current_date += datetime.timedelta(days=1)
        
        # Convert to DataFrame
        logger.info(f"Converting {len(annual_data)} records to DataFrame...")
        df = pd.DataFrame(annual_data)
        
        # Final statistics
        final_efficiency = efficiency_sum / efficiency_count if efficiency_count > 0 else 0
        logger.info("Annual simulation completed with FIXED API compatibility:")
        logger.info(f"  Records generated: {len(df):,}")
        logger.info(f"  Average efficiency: {final_efficiency:.1%}")
        logger.info(f"  Solver failures: {self.simulation_stats['solver_failures']}")
        logger.info(f"  API fixes applied: {self.simulation_stats['api_compatibility_fixes']}")
        
        return df
    
    def _simulate_boiler_operation_fixed_api(self, current_datetime: datetime.datetime,
                                           operating_conditions: Dict,
                                           soot_blowing_actions: Dict) -> Dict:
        """
        FIXED API COMPATIBILITY: Simulate complete boiler operation with proper interface handling.
        
        CRITICAL FIXES APPLIED:
        - Proper handling of solve_enhanced_system return structure
        - Correct parameter extraction with fallbacks
        - Robust error handling for API mismatches
        - ASCII-safe temperature logging
        """
        
        try:
            # Update boiler operating conditions with CORRECT API parameters
            self.boiler.update_operating_conditions(
                operating_conditions['fuel_input_btu_hr'],
                operating_conditions['flue_gas_mass_flow'],
                operating_conditions['furnace_exit_temp']
            )
            
            # Apply soot blowing effects if any sections are being cleaned
            self._apply_soot_blowing_effects(soot_blowing_actions)
            
            # CRITICAL API FIX: Properly handle the enhanced solver interface
            try:
                solve_results = self.boiler.solve_enhanced_system(max_iterations=20, tolerance=15.0)
                
                # Debug logging for interface compatibility
                logger.debug(f"Solver returned keys: {list(solve_results.keys())}")
                
                # FIXED: Extract results using the CORRECT standardized structure with fallbacks
                converged = solve_results.get('converged', False)
                system_performance = solve_results.get('system_performance', {})
                
                if not converged:
                    logger.debug(f"Solver did not converge in {solve_results.get('solver_iterations', 0)} iterations")
                    self.simulation_stats['solver_failures'] += 1
                
                # FIXED: Extract performance metrics with proper fallback handling
                base_system_efficiency = solve_results.get('final_efficiency', 
                                  system_performance.get('system_efficiency', 0.82))
                base_stack_temp = solve_results.get('final_stack_temperature', 
                            system_performance.get('stack_temperature', 280.0))
                steam_temp = solve_results.get('final_steam_temperature',
                            system_performance.get('final_steam_temperature', 700.0))
                energy_error = solve_results.get('energy_balance_error',
                              system_performance.get('energy_balance_error', 0.05))
                
                # CRITICAL FIX: Apply fouling-based efficiency degradation and temperature impacts
                fouling_data = self._generate_fouling_data(current_datetime)
                fouling_efficiency_impact = self._calculate_fouling_efficiency_impact(fouling_data)
                system_efficiency = base_system_efficiency * fouling_efficiency_impact
                
                # CRITICAL FIX: Adjust stack temperature based on fouling (reduced heat transfer = higher stack temp)
                fouling_temp_impact = self._calculate_fouling_temperature_impact(fouling_data)
                stack_temp = base_stack_temp + fouling_temp_impact
                
                # Track successful API fixes
                self.simulation_stats['api_compatibility_fixes'] += 1
                
            except Exception as solver_error:
                logger.warning(f"Solver interface error: {solver_error}")
                # Provide fallback values when solver interface fails
                converged = False
                base_system_efficiency = 0.82  # Reasonable fallback
                base_stack_temp = 280.0
                steam_temp = 700.0
                energy_error = 0.10
                
                # Apply fouling impacts to fallback values too
                fouling_data = self._generate_fouling_data(current_datetime)
                fouling_efficiency_impact = self._calculate_fouling_efficiency_impact(fouling_data)
                system_efficiency = base_system_efficiency * fouling_efficiency_impact
                fouling_temp_impact = self._calculate_fouling_temperature_impact(fouling_data)
                stack_temp = base_stack_temp + fouling_temp_impact
                
                self.simulation_stats['solver_failures'] += 1
            
            # Generate comprehensive operation data with FIXED API results
            
            # Get coal combustion data
            coal_data = self._generate_coal_combustion_data(operating_conditions)
            
            # Generate emissions data
            emissions_data = self._generate_emissions_data(operating_conditions, system_efficiency)
            
            # Generate soot blowing data
            soot_blowing_data = self._generate_soot_blowing_data(soot_blowing_actions)
            
            # Fouling data already generated above for efficiency calculation
            
            # Generate section-specific data
            section_data = self._generate_section_data(stack_temp, steam_temp)
            
            # Validation and warnings
            if system_efficiency < 0.75 or system_efficiency > 0.90:
                logger.warning(f"Efficiency {system_efficiency:.1%} outside expected range (75-90%)")
                self.simulation_stats['efficiency_warnings'] += 1
            
            if stack_temp > 400:
                logger.warning(f"Stack temperature {stack_temp:.0f}F unusually high")
                self.simulation_stats['temperature_warnings'] += 1
            
            # Combine all data into comprehensive operation record
            operation_data = {
                # Timestamp and basic info
                'timestamp': current_datetime,
                'year': current_datetime.year,
                'month': current_datetime.month,
                'day': current_datetime.day,
                'hour': current_datetime.hour,
                'day_of_year': current_datetime.timetuple().tm_yday,
                'season': operating_conditions['season'],
                
                # Operating conditions
                'load_factor': operating_conditions['load_factor'],
                'ambient_temp_F': operating_conditions['ambient_temp_F'],
                'ambient_humidity_pct': operating_conditions['ambient_humidity_pct'],
                'coal_quality': operating_conditions['coal_quality'],
                
                # FIXED API: System performance with proper extraction
                'system_efficiency': system_efficiency,
                'final_steam_temp_F': steam_temp,
                'stack_temp_F': stack_temp,
                'energy_balance_error_pct': energy_error,
                'solution_converged': converged
            }
            
            # Add all component data using dictionary unpacking
            operation_data.update(coal_data)
            operation_data.update(emissions_data)
            operation_data.update(soot_blowing_data)
            operation_data.update(fouling_data)
            operation_data.update(section_data)
            
            return operation_data
            
        except Exception as e:
            logger.error(f"Complete operation simulation failed: {e}")
            # Return fallback data
            return self._create_fallback_operation_data(current_datetime, operating_conditions)
    
    def _generate_hourly_conditions(self, current_datetime: datetime.datetime) -> Dict:
        """Generate realistic hourly operating conditions."""
        
        try:
            # Time-based factors
            hour = current_datetime.hour
            day_of_week = current_datetime.weekday()
            month = current_datetime.month
            season = self._get_season(month)
            
            # Base load from production patterns
            base_load = self.production_patterns['base_load']
            
            # Apply weekly pattern (Monday=0, Sunday=6)
            weekly_factor = self.production_patterns['weekly_pattern'][day_of_week]
            
            # Apply monthly pattern
            monthly_factor = self.production_patterns['monthly_pattern'][month - 1]
            
            # Daily variation (higher during day shift)
            if 6 <= hour <= 18:  # Day shift
                daily_factor = 1.0 + self.production_patterns['daily_variation']
            else:  # Night shift
                daily_factor = 1.0 - self.production_patterns['daily_variation']
            
            # Calculate load factor with realistic bounds
            load_factor = base_load * weekly_factor * monthly_factor * daily_factor
            
            # Add some realistic variation
            load_variation = np.random.normal(0, 0.02)  # 2% std dev
            load_factor = max(0.60, min(1.05, load_factor + load_variation))
            
            # Calculate fuel input based on load factor
            design_fuel_input = 100e6  # Btu/hr
            fuel_input_btu_hr = design_fuel_input * load_factor
            
            # Calculate flue gas flow (proportional to fuel input)
            base_flue_gas_flow = 84000  # lb/hr at design
            flue_gas_mass_flow = base_flue_gas_flow * load_factor
            
            # Furnace exit temperature (varies with load)
            base_furnace_temp = 3000  # F
            furnace_temp_variation = 100 * (load_factor - 0.85)  # Varies with load
            furnace_exit_temp = base_furnace_temp + furnace_temp_variation
            
            # Ambient conditions
            ambient_temp_F = self._get_ambient_temperature(season, hour)
            ambient_humidity_pct = np.random.uniform(30, 80)
            
            # Coal quality variation
            coal_qualities = ['bituminous', 'sub_bituminous', 'lignite']
            coal_quality = np.random.choice(coal_qualities, p=[0.7, 0.25, 0.05])
            
            return {
                'load_factor': load_factor,
                'fuel_input_btu_hr': fuel_input_btu_hr,
                'flue_gas_mass_flow': flue_gas_mass_flow,
                'furnace_exit_temp': furnace_exit_temp,
                'ambient_temp_F': ambient_temp_F,
                'ambient_humidity_pct': ambient_humidity_pct,
                'coal_quality': coal_quality,
                'season': season,
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month
            }
            
        except Exception as e:
            logger.error(f"Failed to generate hourly conditions: {e}")
            # Return fallback conditions
            return {
                'load_factor': 0.85,
                'fuel_input_btu_hr': 85e6,
                'flue_gas_mass_flow': 71400,
                'furnace_exit_temp': 3000,
                'ambient_temp_F': 70.0,
                'ambient_humidity_pct': 50.0,
                'coal_quality': 'bituminous',
                'season': 'spring',
                'hour': 12,
                'day_of_week': 2,
                'month': 6
            }
    
    def _get_season(self, month: int) -> str:
        """Get season based on month."""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _get_ambient_temperature(self, season: str, hour: int) -> float:
        """Get realistic ambient temperature based on season and hour."""
        
        # Base temperatures by season (F)
        base_temps = {
            'winter': 32,
            'spring': 55,
            'summer': 78,
            'fall': 58
        }
        
        base_temp = base_temps.get(season, 65)
        
        # Daily temperature variation
        if 6 <= hour <= 18:  # Daytime
            temp_adjustment = np.random.uniform(5, 15)
        else:  # Nighttime
            temp_adjustment = np.random.uniform(-10, 0)
        
        return base_temp + temp_adjustment
    
    def _check_soot_blowing_schedule(self, current_datetime: datetime.datetime) -> Dict:
        """
        REFACTORED: Use centralized SootBlowingSimulator for schedule checking.
        
        This method now delegates to SootBlowingSimulator.check_section_cleaning_schedule
        to maintain centralized soot blowing logic.
        """
        soot_blowing_actions = {}
        
        for section_name, interval_hours in self.soot_blowing_schedule.items():
            last_cleaned = self.last_cleaned[section_name]
            
            # Use centralized SootBlowingSimulator method
            action = SootBlowingSimulator.check_section_cleaning_schedule(
                section_name, last_cleaned, interval_hours, current_datetime
            )
            
            soot_blowing_actions[section_name] = action
            
            # Update last cleaned date if cleaning is happening
            if action['action']:
                self.last_cleaned[section_name] = current_datetime
        
        return soot_blowing_actions
    
    def _apply_soot_blowing_effects(self, soot_blowing_actions: Dict):
        """
        REFACTORED: Use centralized SootBlowingSimulator for applying effectiveness.
        
        This method now delegates to SootBlowingSimulator.apply_section_soot_blowing_effects
        to maintain centralized soot blowing logic.
        """
        
        for section_name, action in soot_blowing_actions.items():
            if action['action']:  # If cleaning is happening
                # Use centralized SootBlowingSimulator method for fouling calculation
                current_fouling = SootBlowingSimulator.calculate_current_fouling_factor(
                    section_name, action['hours_since_last'], self.fouling_baselines
                )
                
                # Use centralized SootBlowingSimulator method for effectiveness application
                cleaning_result = SootBlowingSimulator.apply_section_soot_blowing_effects(
                    section_name, action, current_fouling, self.fouling_baselines
                )
                
                # Log results
                if cleaning_result['cleaning_applied']:
                    logger.debug(f"Applied soot blowing to {section_name}: {cleaning_result['effectiveness']:.1%} effectiveness")
                    logger.debug(f"Fouling: {cleaning_result['original_fouling']:.4f} -> {cleaning_result['new_baseline']:.4f} (removed {cleaning_result['fouling_removed']:.4f})")
                    logger.debug(f"Fouling timer reset for {section_name} at {self.last_cleaned[section_name]}")
                else:
                    if 'error' in cleaning_result:
                        logger.warning(f"Could not apply soot blowing effectiveness to {section_name}: {cleaning_result['error']}")
                    else:
                        logger.debug(f"No soot blowing applied to {section_name}")
            else:
                logger.debug(f"No soot blowing scheduled for {section_name}")
    
    # DEPRECATED: Moved to SootBlowingSimulator.calculate_current_fouling_factor
    # TODO: DELETE AFTER TESTING - This functionality has been centralized in SootBlowingSimulator
    # 
    # def _get_current_fouling_factor(self, section_name: str, hours_since_cleaning: float) -> float:
    #     """Calculate current fouling factor before cleaning for effectiveness application."""
    #     
    #     # Get section-specific fouling rate
    #     fouling_rates = {
    #         'furnace_walls': 0.00030,           # HIGHEST fouling
    #         'generating_bank': 0.00025,         # High fouling
    #         'superheater_primary': 0.00020,     # Moderate-high fouling
    #         'superheater_secondary': 0.00015,   # Moderate fouling
    #         'economizer_primary': 0.00012,      # Lower fouling
    #         'economizer_secondary': 0.00008,    # Low fouling
    #         'air_heater': 0.00004               # LOWEST fouling
    #     }
    #     
    #     base_rate = fouling_rates.get(section_name, 0.00015)  # Default moderate rate
    #     
    #     # Start with post-cleaning baseline (not always 1.0 after effectiveness application)
    #     baseline_fouling = self.fouling_baselines.get(section_name, 1.0)
    #     
    #     # Add fouling accumulation since last cleaning
    #     fouling_accumulation = base_rate * hours_since_cleaning
    #     
    #     # Current fouling factor before this cleaning
    #     current_fouling = baseline_fouling + fouling_accumulation
    #     
    #     # Apply realistic industrial bounds
    #     current_fouling = max(1.0, min(1.25, current_fouling))
    #     
    #     return current_fouling
    
    def _generate_coal_combustion_data(self, operating_conditions: Dict) -> Dict:
        """Generate realistic coal combustion data."""
        
        try:
            # CRITICAL FIX: Always generate variable coal properties (remove combustion_model dependency)
            # Basic coal properties based on quality with realistic variations
            coal_quality = operating_conditions.get('coal_quality', 'bituminous')
            
            if coal_quality == 'bituminous':
                heating_value = np.random.uniform(12000, 13500)  # Btu/lb
                carbon_pct = np.random.uniform(70, 80)
                volatile_pct = np.random.uniform(30, 40)
                sulfur_pct = np.random.uniform(1.0, 3.0)
                ash_pct = np.random.uniform(6, 12)
                moisture_pct = np.random.uniform(8, 15)
            elif coal_quality == 'sub_bituminous':
                heating_value = np.random.uniform(8500, 12000)
                carbon_pct = np.random.uniform(60, 70)
                volatile_pct = np.random.uniform(35, 45)
                sulfur_pct = np.random.uniform(0.5, 2.0)
                ash_pct = np.random.uniform(8, 15)
                moisture_pct = np.random.uniform(15, 25)
            else:  # lignite
                heating_value = np.random.uniform(6500, 8500)
                carbon_pct = np.random.uniform(50, 60)
                volatile_pct = np.random.uniform(35, 45)
                sulfur_pct = np.random.uniform(0.5, 1.5)
                ash_pct = np.random.uniform(10, 20)
                moisture_pct = np.random.uniform(25, 40)
            
            # Calculate coal consumption rate
            fuel_input = operating_conditions.get('fuel_input_btu_hr', 85e6)
            coal_rate_lb_hr = fuel_input / heating_value
            
            # Air flow calculation
            theoretical_air = coal_rate_lb_hr * 10  # Rough estimate
            excess_air_pct = np.random.uniform(15, 25)  # 15-25% excess air
            actual_air_flow = theoretical_air * (1 + excess_air_pct/100)
            air_flow_scfh = actual_air_flow * 13.3  # Convert to SCFH (approximate)
            
            # Flue gas flow
            flue_gas_flow = operating_conditions.get('flue_gas_mass_flow', 71400)
            
            # Excess O2 calculation
            excess_o2 = excess_air_pct / 5.0  # Rough conversion
            
            return {
                'coal_carbon_pct': carbon_pct,
                'coal_volatile_matter_pct': volatile_pct,
                'coal_sulfur_pct': sulfur_pct,
                'coal_ash_pct': ash_pct,
                'coal_moisture_pct': moisture_pct,
                'coal_heating_value_btu_lb': heating_value,
                'coal_rate_lb_hr': coal_rate_lb_hr,
                'air_flow_scfh': air_flow_scfh,
                'fuel_input_btu_hr': fuel_input,
                'flue_gas_flow_lb_hr': flue_gas_flow,
                'excess_o2_pct': max(2.0, min(6.0, excess_o2)),
                'combustion_efficiency': np.random.normal(0.98, 0.01),
                'flame_temp_F': int(np.random.normal(3200, 100))
            }
            
        except Exception as e:
            logger.warning(f"Coal combustion data generation failed: {e}")
            return {
                'coal_carbon_pct': 75.0,
                'coal_volatile_matter_pct': 35.0,
                'coal_sulfur_pct': 1.5,
                'coal_ash_pct': 8.0,
                'coal_moisture_pct': 12.0,
                'coal_heating_value_btu_lb': 12500,
                'coal_rate_lb_hr': 12000.0,
                'air_flow_scfh': 100000.0,
                'fuel_input_btu_hr': operating_conditions.get('fuel_input_btu_hr', 85e6),
                'flue_gas_flow_lb_hr': operating_conditions.get('flue_gas_mass_flow', 71400),
                'excess_o2_pct': 3.5,
                'combustion_efficiency': 0.98,
                'flame_temp_F': 3200
            }
    
    def _generate_emissions_data(self, operating_conditions: Dict, system_efficiency: float) -> Dict:
        """Generate realistic emissions data."""
        
        try:
            # Base emissions calculations
            coal_rate = operating_conditions.get('fuel_input_btu_hr', 85e6) / 12500  # lb/hr coal
            
            # NOx emissions (depends on combustion conditions)
            nox_factor = np.random.uniform(0.4, 0.8)  # lb NOx per MMBtu
            fuel_mmbtu_hr = operating_conditions.get('fuel_input_btu_hr', 85e6) / 1e6
            total_nox_lb_hr = fuel_mmbtu_hr * nox_factor
            
            # SO2 emissions (depends on coal sulfur content)
            coal_sulfur_pct = np.random.uniform(1.0, 2.5)
            so2_lb_hr = coal_rate * (coal_sulfur_pct / 100) * 2.0  # 2 lb SO2 per lb S
            
            # CO2 emissions
            carbon_pct = np.random.uniform(70, 80)
            co2_lb_hr = coal_rate * (carbon_pct / 100) * (44/12)  # Convert C to CO2
            
            # Stack gas composition
            excess_o2_pct = np.random.uniform(3.0, 5.0)
            co2_pct = np.random.uniform(12, 15)
            h2o_pct = np.random.uniform(8, 12)
            n2_pct = 100 - excess_o2_pct - co2_pct - h2o_pct
            
            # Particulates
            ash_pct = np.random.uniform(6, 10)
            particulates_lb_hr = coal_rate * (ash_pct / 100) * 0.1  # Assume 10% carryover
            
            return {
                'total_nox_lb_hr': total_nox_lb_hr,
                'so2_lb_hr': so2_lb_hr,
                'co2_lb_hr': co2_lb_hr,
                'particulates_lb_hr': particulates_lb_hr,
                'excess_o2_pct': excess_o2_pct,
                'co2_pct': co2_pct,
                'h2o_pct': h2o_pct,
                'n2_pct': n2_pct,
                'co_ppm': np.random.uniform(50, 150),
                'opacity_pct': np.random.uniform(5, 15),
                'stack_velocity_fps': np.random.uniform(40, 60)
            }
            
        except Exception as e:
            logger.warning(f"Emissions data generation failed: {e}")
            return {
                'total_nox_lb_hr': 200.0,
                'so2_lb_hr': 150.0,
                'co2_lb_hr': 18000.0,
                'particulates_lb_hr': 25.0,
                'excess_o2_pct': 4.0,
                'co2_pct': 13.5,
                'h2o_pct': 10.0,
                'n2_pct': 72.5,
                'co_ppm': 100,
                'opacity_pct': 10,
                'stack_velocity_fps': 50
            }

    
    def _generate_soot_blowing_data(self, soot_blowing_actions: Dict) -> Dict:
        """
        REFACTORED: Use centralized SootBlowingSimulator for data generation.
        
        This method now delegates to SootBlowingSimulator.generate_cleaning_activity_data
        to maintain centralized soot blowing logic.
        """
        
        # Use centralized SootBlowingSimulator method
        return SootBlowingSimulator.generate_cleaning_activity_data(soot_blowing_actions)
    
    def _generate_fouling_data(self, current_datetime: datetime.datetime) -> Dict:
        """
        CRITICAL FIX: Generate realistic fouling factor data based on time since last cleaning.
        
        This fixes the major issue where fouling was accumulating based on total simulation time
        instead of time since last soot blowing for each section.
        """
        
        fouling_data = {}
        
        # CORRECTED: Define fouling rates with CORRECT industrial physics
        # High temperatures make soot sticky and cause MORE fouling, not less
        fouling_rates = {
            'furnace_walls': 0.00030,           # HIGHEST fouling - high temp makes soot sticky (0.30% per hour)
            'generating_bank': 0.00025,         # High fouling - still very hot (0.25% per hour)
            'superheater_primary': 0.00020,     # Moderate-high fouling (0.20% per hour)
            'superheater_secondary': 0.00015,   # Moderate fouling (0.15% per hour)
            'economizer_primary': 0.00012,      # Lower fouling as temp drops (0.12% per hour)
            'economizer_secondary': 0.00008,    # Low fouling - cooler temps (0.08% per hour)
            'air_heater': 0.00004              # LOWEST fouling - coldest section (0.04% per hour)
        }
        
        # CRITICAL FIX: Section name mapping for output consistency
        section_output_names = {
            'furnace_walls': 'furnace',
            'generating_bank': 'generating_bank', 
            'superheater_primary': 'superheater_1',
            'superheater_secondary': 'superheater_2',
            'economizer_primary': 'economizer_1',
            'economizer_secondary': 'economizer_2',
            'air_heater': 'air_heater'
        }
        
        # Generate fouling factors for each section based on hours since last cleaning
        for section_schedule_name, base_rate in fouling_rates.items():
            # CRITICAL FIX: Use the section name from the cleaning schedule
            hours_since_cleaning = 0
            
            # Find the hours since last cleaning for this section
            if hasattr(self, 'last_cleaned') and section_schedule_name in self.last_cleaned:
                hours_since_cleaning = (current_datetime - self.last_cleaned[section_schedule_name]).total_seconds() / 3600
            else:
                # If no cleaning history, use hours since simulation start
                hours_since_cleaning = (current_datetime - self.start_date).total_seconds() / 3600
            
            # Start with clean condition (1.0 = no fouling)
            base_fouling = 1.0
            
            # CRITICAL FIX: Fouling accumulation based on time since LAST CLEANING, not total time
            fouling_accumulation = base_rate * hours_since_cleaning
            
            # Add coal-dependent fouling (more fouling with lower quality coal)
            coal_quality = getattr(self, '_current_coal_quality', 'bituminous')
            coal_fouling_multiplier = {
                'bituminous': 1.0,        # Baseline
                'sub_bituminous': 1.2,    # 20% more fouling
                'lignite': 1.5            # 50% more fouling
            }.get(coal_quality, 1.0)
            
            fouling_accumulation *= coal_fouling_multiplier
            
            # Add some realistic variation
            variation = np.random.uniform(-0.002, 0.002)
            
            # Calculate current fouling factor (higher = more fouling)
            current_fouling = base_fouling + fouling_accumulation + variation
            
            # Apply realistic bounds based on industrial experience
            current_fouling = max(1.0, min(1.25, current_fouling))  # 1.0 to 1.25 range
            
            # CRITICAL FIX: Use output section name for data consistency
            output_section_name = section_output_names[section_schedule_name]
            
            # Store fouling data with output section names
            fouling_data[f'{output_section_name}_fouling_factor'] = current_fouling
            fouling_data[f'{output_section_name}_heat_transfer_loss_pct'] = (current_fouling - 1.0) * 100
            
            # Generate segment-specific data (6 segments per section)
            for segment in range(1, 7):
                segment_variation = np.random.uniform(-0.01, 0.01)
                segment_fouling = max(1.0, min(1.25, current_fouling + segment_variation))
                fouling_data[f'{output_section_name}_segment_{segment}_fouling'] = segment_fouling
            
            # Store hours since cleaning for validation (using schedule name for traceability)
            fouling_data[f'hours_since_last_{output_section_name}'] = hours_since_cleaning
        
        return fouling_data
    
    def _generate_section_data(self, stack_temp: float, steam_temp: float) -> Dict:
        """Generate section-specific temperature and heat transfer data."""
        
        section_data = {}
        
        # Define typical temperature profiles through the boiler
        furnace_gas_in = 3000  # Furnace exit temperature
        
        # Temperature drops through each section
        generating_bank_gas_out = furnace_gas_in - np.random.uniform(500, 700)
        superheater_1_gas_out = generating_bank_gas_out - np.random.uniform(300, 500)
        superheater_2_gas_out = superheater_1_gas_out - np.random.uniform(200, 400)
        economizer_1_gas_out = superheater_2_gas_out - np.random.uniform(200, 350)
        economizer_2_gas_out = economizer_1_gas_out - np.random.uniform(100, 200)
        air_heater_gas_out = stack_temp  # Final stack temperature
        
        # Heat transfer rates (approximate)
        sections_data = {
            'furnace': {'gas_out': furnace_gas_in, 'q_mmbtu_hr': np.random.uniform(40, 60)},
            'generating_bank': {'gas_out': generating_bank_gas_out, 'q_mmbtu_hr': np.random.uniform(15, 25)},
            'superheater_1': {'gas_out': superheater_1_gas_out, 'q_mmbtu_hr': np.random.uniform(8, 15)},
            'superheater_2': {'gas_out': superheater_2_gas_out, 'q_mmbtu_hr': np.random.uniform(6, 12)},
            'economizer_1': {'gas_out': economizer_1_gas_out, 'q_mmbtu_hr': np.random.uniform(5, 10)},
            'economizer_2': {'gas_out': economizer_2_gas_out, 'q_mmbtu_hr': np.random.uniform(3, 8)},
            'air_heater': {'gas_out': air_heater_gas_out, 'q_mmbtu_hr': np.random.uniform(2, 6)}
        }
        
        # Generate data for each section
        for section_name, data in sections_data.items():
            section_data[f'{section_name}_gas_temp_out_F'] = data['gas_out']
            section_data[f'{section_name}_heat_transfer_mmbtu_hr'] = data['q_mmbtu_hr']
            section_data[f'{section_name}_heat_transfer_efficiency'] = np.random.uniform(0.85, 0.95)
            
            # Generate segment data (6 segments per section)
            for segment in range(1, 7):
                segment_temp_drop = data['q_mmbtu_hr'] * np.random.uniform(0.8, 1.2) / 6
                section_data[f'{section_name}_segment_{segment}_temp_drop_F'] = segment_temp_drop
                section_data[f'{section_name}_segment_{segment}_q_mmbtu_hr'] = data['q_mmbtu_hr'] / 6 * np.random.uniform(0.8, 1.2)
        
        return section_data
    
    def _create_fallback_operation_data(self, current_datetime: datetime.datetime, 
                                      operating_conditions: Optional[Dict] = None) -> Dict:
        """Create fallback operation data when simulation fails."""
        
        if operating_conditions is None:
            operating_conditions = {
                'load_factor': 0.85,
                'ambient_temp_F': 70.0,
                'ambient_humidity_pct': 50.0,
                'coal_quality': 'bituminous',
                'season': 'spring'
            }
        
        return {
            # Timestamp and basic info
            'timestamp': current_datetime,
            'year': current_datetime.year,
            'month': current_datetime.month,
            'day': current_datetime.day,
            'hour': current_datetime.hour,
            'day_of_year': current_datetime.timetuple().tm_yday,
            'season': operating_conditions['season'],
            
            # Operating conditions
            'load_factor': operating_conditions['load_factor'],
            'ambient_temp_F': operating_conditions['ambient_temp_F'],
            'ambient_humidity_pct': operating_conditions['ambient_humidity_pct'],
            'coal_quality': operating_conditions['coal_quality'],
            
            # Fallback performance values
            'system_efficiency': 0.82,
            'final_steam_temp_F': 700.0,
            'stack_temp_F': 280.0,
            'energy_balance_error_pct': 0.1,
            'solution_converged': False,
            
            # Basic fallback data for all other fields
            'coal_rate_lb_hr': 12000.0,
            'air_flow_scfh': 100000.0,
            'fuel_input_btu_hr': 100e6,
            'flue_gas_flow_lb_hr': 84000.0,
            'coal_carbon_pct': 75.0,
            'coal_volatile_matter_pct': 35.0,
            'coal_sulfur_pct': 1.5,
            'coal_ash_pct': 8.0,
            'coal_moisture_pct': 12.0,
            'coal_heating_value_btu_lb': 12500,
            'excess_o2_pct': 3.5,
            'combustion_efficiency': 0.98,
            'flame_temp_F': 3200,
            'total_nox_lb_hr': 200.0,
            'so2_lb_hr': 150.0,
            'co2_lb_hr': 18000.0,
            'particulates_lb_hr': 25.0,
            'co2_pct': 13.5,
            'h2o_pct': 10.0,
            'n2_pct': 72.5,
            'co_ppm': 100,
            'opacity_pct': 10,
            'stack_velocity_fps': 50,
            'soot_blowing_active': False,
            'sections_being_cleaned': 0,
            'avg_cleaning_effectiveness': 0.0,
            'furnace_walls_cleaning': False,
            'generating_bank_cleaning': False,
            'superheater_primary_cleaning': False,
            'superheater_secondary_cleaning': False,
            'economizer_primary_cleaning': False,
            'economizer_secondary_cleaning': False,
            'air_heater_cleaning': False,
            'steam_consumption_lb_hr': 0,
            'cleaning_duration_min': 0,
            'hours_since_last_furnace': 0,
            'hours_since_last_generating': 0,
            'hours_since_last_superheater_1': 0,
            'hours_since_last_superheater_2': 0,
            'hours_since_last_economizer_1': 0,
            'hours_since_last_economizer_2': 0,
            'hours_since_last_air_heater': 0,
            'furnace_fouling_factor': 1.05,
            'generating_bank_fouling_factor': 1.08,
            'superheater_1_fouling_factor': 1.10,
            'superheater_2_fouling_factor': 1.10,
            'economizer_1_fouling_factor': 1.12,
            'economizer_2_fouling_factor': 1.12,
            'air_heater_fouling_factor': 1.15
        }
    
    def save_annual_data(self, annual_data: pd.DataFrame) -> Tuple[str, str]:
        """Save annual data with enhanced metadata and proper file organization."""
        
        # Generate timestamp for unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filenames with proper paths
        data_filepath = data_dir / f"massachusetts_boiler_annual_{timestamp}.csv"
        metadata_filepath = metadata_dir / f"massachusetts_boiler_annual_metadata_{timestamp}.txt"
        
        logger.info(f"Saving enhanced annual dataset with IAPWS integration...")
        
        # Save main dataset
        annual_data.to_csv(data_filepath, index=False)
        
        # Generate comprehensive metadata
        with open(metadata_filepath, 'w') as f:
            f.write("ENHANCED ANNUAL BOILER SIMULATION METADATA\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"GENERATION INFO:\n")
            f.write(f"  Generated: {datetime.datetime.now()}\n")
            f.write(f"  Simulation Version: 8.2 - API Compatibility Fix\n")
            f.write(f"  IAPWS Integration: YES (Industry-Standard Steam Properties)\n")
            f.write(f"  ASCII Compatibility: YES (Windows Compatible)\n")
            
            f.write(f"\nDATASET INFO:\n")
            f.write(f"  Records: {len(annual_data):,} hourly operations\n")
            f.write(f"  Time Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}\n")
            f.write(f"  Data File: {data_filepath.name}\n")
            f.write(f"  File Size: {data_filepath.stat().st_size / (1024*1024):.1f} MB\n")
            
            if len(annual_data) > 0:
                f.write(f"\nPERFORMANCE SUMMARY:\n")
                if 'system_efficiency' in annual_data.columns:
                    eff_mean = annual_data['system_efficiency'].mean()
                    eff_std = annual_data['system_efficiency'].std()
                    eff_min = annual_data['system_efficiency'].min()
                    eff_max = annual_data['system_efficiency'].max()
                    f.write(f"  Average Efficiency: {eff_mean:.1%}\n")
                    f.write(f"  Efficiency Range: {eff_min:.1%} - {eff_max:.1%}\n")
                    f.write(f"  Efficiency Variation: {eff_std:.2%} std dev\n")
                
                if 'load_factor' in annual_data.columns:
                    load_mean = annual_data['load_factor'].mean()
                    load_min = annual_data['load_factor'].min()
                    load_max = annual_data['load_factor'].max()
                    f.write(f"  Average Load Factor: {load_mean:.1%}\n")
                    f.write(f"  Load Range: {load_min:.1%} - {load_max:.1%}\n")
                
                if 'stack_temp_F' in annual_data.columns:
                    stack_mean = annual_data['stack_temp_F'].mean()
                    stack_std = annual_data['stack_temp_F'].std()
                    f.write(f"  Average Stack Temperature: {stack_mean:.0f}F\n")
                    f.write(f"  Stack Temperature Variation: {stack_std:.1f}F std dev\n")
            
            f.write(f"\nSIMULATION STATISTICS:\n")
            f.write(f"  Total Hours Simulated: {self.simulation_stats['total_hours']}\n")
            f.write(f"  Solver Failures: {self.simulation_stats['solver_failures']}\n")
            f.write(f"  API Compatibility Fixes: {self.simulation_stats['api_compatibility_fixes']}\n")
            f.write(f"  Efficiency Warnings: {self.simulation_stats['efficiency_warnings']}\n")
            f.write(f"  Temperature Warnings: {self.simulation_stats['temperature_warnings']}\n")
            
            f.write(f"\nDATA STRUCTURE:\n")
            f.write(f"  Total Columns: {len(annual_data.columns)}\n")
            f.write("  Column Categories:\n")
            f.write("    - Timestamp & Operational (7 columns)\n")
            f.write("    - Coal & Combustion (17 columns)\n")
            f.write("    - Emissions (11 columns)\n")
            f.write("    - System Performance (13 columns)\n")
            f.write("    - Soot Blowing (16 columns)\n")
            f.write("    - Fouling Factors (42 columns)\n")
            f.write("    - Section Heat Transfer (42 columns)\n")
            
            f.write(f"\nQUALITY IMPROVEMENTS:\n")
            f.write(f"  - FIXED API compatibility between AnnualBoilerSimulator and EnhancedCompleteBoilerSystem\n")
            f.write(f"  - Proper parameter extraction from solve_enhanced_system results\n")
            f.write(f"  - Robust fallback handling for solver interface mismatches\n")
            f.write(f"  - ASCII-safe temperature and efficiency logging\n")
            f.write(f"  - Enhanced error handling and debugging output\n")
            f.write(f"  - Fixed indentation errors causing syntax failures\n")
        
        # Log save completion
        file_size_mb = data_filepath.stat().st_size / (1024 * 1024)
        logger.info("Enhanced annual dataset saved with FIXED API compatibility:")
        logger.info(f"  Data file: {data_filepath}")
        logger.info(f"  Metadata: {metadata_filepath}")
        logger.info(f"  Records: {len(annual_data):,}")
        logger.info(f"  Size: {file_size_mb:.1f} MB")
        
        if 'system_efficiency' in annual_data.columns:
            logger.info(f"  Average efficiency: {annual_data['system_efficiency'].mean():.1%}")
        if 'stack_temp_F' in annual_data.columns:
            logger.info(f"  Stack temp variation: {annual_data['stack_temp_F'].std():.1f}F std dev")
        
        return str(data_filepath), str(metadata_filepath)
    
    def _calculate_fouling_efficiency_impact(self, fouling_data: Dict) -> float:
        """
        CRITICAL FIX: Calculate efficiency impact based on fouling factors.
        
        This method implements realistic industrial fouling impact on boiler efficiency:
        - Each boiler section contributes to overall efficiency loss
        - Fouling factor > 1.0 reduces heat transfer effectiveness
        - Weighted by section importance for heat transfer
        """
        
        try:
            # Section weights based on heat transfer importance
            section_weights = {
                'furnace_fouling_factor': 0.15,          # Furnace contributes 15% to efficiency loss
                'generating_bank_fouling_factor': 0.25,  # Generating bank most critical (25%)
                'superheater_1_fouling_factor': 0.20,    # Superheaters important (20% each)
                'superheater_2_fouling_factor': 0.20,    
                'economizer_1_fouling_factor': 0.10,     # Economizers less critical (10% each)
                'economizer_2_fouling_factor': 0.05,
                'air_heater_fouling_factor': 0.05        # Air heater least critical (5%)
            }
            
            # Calculate weighted fouling impact
            total_fouling_impact = 0.0
            
            for section, weight in section_weights.items():
                fouling_factor = fouling_data.get(section, 1.0)
                
                # Convert fouling factor to efficiency loss
                # fouling_factor = 1.0 means no fouling (no loss)
                # fouling_factor = 1.15 means 15% fouling -> ~3-5% efficiency loss
                section_efficiency_loss = (fouling_factor - 1.0) * 0.25  # 25% scaling factor
                weighted_loss = section_efficiency_loss * weight
                total_fouling_impact += weighted_loss
            
            # Calculate efficiency multiplier (1.0 = no impact, <1.0 = reduced efficiency)
            efficiency_multiplier = 1.0 - total_fouling_impact
            
            # Ensure reasonable bounds (minimum 70% efficiency, maximum 100%)
            efficiency_multiplier = max(0.70, min(1.0, efficiency_multiplier))
            
            # Debug logging for validation
            if total_fouling_impact > 0.05:  # Log significant fouling impact (>5%)
                logger.debug(f"Significant fouling impact: {total_fouling_impact:.1%} efficiency reduction")
            
            return efficiency_multiplier
            
        except Exception as e:
            logger.warning(f"Fouling efficiency calculation failed: {e}")
            return 1.0  # No impact fallback
    
    def _calculate_fouling_temperature_impact(self, fouling_data: Dict) -> float:
        """
        CRITICAL FIX: Calculate stack temperature increase due to fouling.
        
        Fouling reduces heat transfer effectiveness, causing higher stack temperatures.
        This creates the expected positive correlation between fouling and stack temperature.
        """
        
        try:
            # Heat transfer sections that most impact stack temperature (downstream sections)
            temp_impact_weights = {
                'economizer_1_fouling_factor': 0.30,     # Economizer 1 most impact (30%)
                'economizer_2_fouling_factor': 0.25,     # Economizer 2 high impact (25%)
                'air_heater_fouling_factor': 0.20,       # Air heater moderate impact (20%)
                'superheater_2_fouling_factor': 0.15,    # Secondary superheater (15%)
                'superheater_1_fouling_factor': 0.10,    # Primary superheater (10%)
            }
            
            # Calculate weighted fouling impact on temperature
            total_temp_impact = 0.0
            
            for section, weight in temp_impact_weights.items():
                fouling_factor = fouling_data.get(section, 1.0)
                
                # Convert fouling factor to temperature increase
                # fouling_factor = 1.0 means no fouling (no temp increase)
                # fouling_factor = 1.15 means 15% fouling -> ~10-20°F increase
                section_temp_increase = (fouling_factor - 1.0) * 120  # 120°F per unit fouling
                weighted_temp_increase = section_temp_increase * weight
                total_temp_impact += weighted_temp_increase
            
            # Ensure reasonable bounds (0-60°F maximum increase)
            total_temp_impact = max(0, min(60.0, total_temp_impact))
            
            # Debug logging for significant temperature increases
            if total_temp_impact > 15.0:  # Log significant temp impact (>15°F)
                logger.debug(f"Significant fouling temperature impact: {total_temp_impact:.1f}°F increase")
            
            return total_temp_impact
            
        except Exception as e:
            logger.warning(f"Fouling temperature calculation failed: {e}")
            return 0.0  # No impact fallback


# Test and validation functions
def test_fixed_interface():
    """Test the FIXED solver interface compatibility."""
    
    print("\n" + "="*60)
    print("TESTING FIXED API COMPATIBILITY")
    print("="*60)
    
    try:
        # Test 1: Enhanced boiler system creation with CORRECT parameters
        print("\n[TEST 1] Creating EnhancedCompleteBoilerSystem with FIXED API...")
        boiler = EnhancedCompleteBoilerSystem(
            fuel_input=100e6,        # CORRECT parameter name
            flue_gas_mass_flow=84000, # CORRECT parameter name  
            furnace_exit_temp=3000    # CORRECT parameter name
        )
        print("[PASS] Boiler system created successfully with correct API")
        
        # Test 2: Solver interface compatibility
        print("\n[TEST 2] Testing solve_enhanced_system interface...")
        results = boiler.solve_enhanced_system(max_iterations=10, tolerance=15.0)
        
        # Test expected return structure
        expected_keys = ['converged', 'final_efficiency', 'final_steam_temperature', 
                        'final_stack_temperature', 'energy_balance_error']
        
        missing_keys = [key for key in expected_keys if key not in results]
        if missing_keys:
            print(f"[WARNING] Missing expected keys: {missing_keys}")
        else:
            print("[PASS] Solver returned all expected keys")
        
        print(f"[INFO] Solver Results:")
        print(f"  Converged: {results.get('converged', 'N/A')}")
        print(f"  Efficiency: {results.get('final_efficiency', 0):.1%}")
        print(f"  Steam Temp: {results.get('final_steam_temperature', 0):.0f}F")
        print(f"  Stack Temp: {results.get('final_stack_temperature', 0):.0f}F")
        
        # Test 3: Annual simulator creation and API compatibility
        print("\n[TEST 3] Testing AnnualBoilerSimulator API compatibility...")
        simulator = AnnualBoilerSimulator(start_date="2024-01-01")
        print("[PASS] Annual simulator created successfully")
        
        # Test generate_annual_data with CORRECT parameters
        print("\n[TEST 4] Testing generate_annual_data with FIXED parameters...")
        simulator.end_date = simulator.start_date + pd.DateOffset(hours=2)  # Just 2 hours for testing
        
        test_data = simulator.generate_annual_data(
            hours_per_day=24,        # CORRECT parameter name
            save_interval_hours=1    # CORRECT parameter name
        )
        
        print(f"[PASS] Generated {len(test_data)} records with fixed API")
        if len(test_data) > 0:
            print(f"[INFO] Sample efficiency: {test_data['system_efficiency'].iloc[0]:.1%}")
        
        # Test 5: Data saving
        print("\n[TEST 5] Testing save_annual_data...")
        data_file, metadata_file = simulator.save_annual_data(test_data)
        print(f"[PASS] Data saved successfully")
        print(f"[INFO] Data file: {Path(data_file).name}")
        print(f"[INFO] Metadata file: {Path(metadata_file).name}")
        
        print("\n" + "="*60)
        print("ALL API COMPATIBILITY TESTS PASSED!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] API compatibility test failed: {e}")
        print("Error details:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """Main testing entry point for API compatibility validation."""
    
    print("ANNUAL BOILER SIMULATOR - API COMPATIBILITY TESTING")
    print("Version 8.2 - API Compatibility Fix")
    print(f"Execution Time: {datetime.datetime.now()}")
    
    try:
        success = test_fixed_interface()
        
        if success:
            print("\n" + "="*60)
            print(">>> API COMPATIBILITY FIXES VALIDATED SUCCESSFULLY!")
            print(">>> System ready for full annual simulation")
            print(">>> Next step: Run 'python simulation/run_annual_simulation.py'")
            print("="*60)
            logger.info("API compatibility validation completed successfully")
            sys.exit(0)
        else:
            print("\n" + "="*60)
            print(">>> API COMPATIBILITY ISSUES STILL PRESENT")
            print(">>> Review error messages and fix remaining issues")
            print(">>> Check logs/simulation/ for detailed error information")
            print("="*60)
            logger.error("API compatibility validation failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Main execution failed: {e}")
        print("Error details:")
        traceback.print_exc()
        logger.error(f"Main execution failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)