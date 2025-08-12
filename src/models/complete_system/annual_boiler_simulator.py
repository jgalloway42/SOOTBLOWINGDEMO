#!/usr/bin/env python3
"""
Annual Boiler Operation Simulator for Massachusetts

This module generates a comprehensive year's worth of boiler operation data including:
- Variable load operation (45-100% of max)
- Seasonal ambient conditions for Massachusetts
- Coal quality variations
- Scheduled soot blowing cycles
- Complete fouling tracking
- Stack gas analysis
- All temperatures, flows, and performance metrics

Author: Enhanced Boiler Modeling System
Version: 6.0 - Annual Operation Simulation
"""

import numpy as np
import pandas as pd
import datetime
from typing import Dict, List, Tuple, Optional
import random

# Import existing modules
from boiler_system import EnhancedCompleteBoilerSystem
from coal_combustion_models import CoalCombustionModel, CombustionFoulingIntegrator
from thermodynamic_properties import PropertyCalculator
from fouling_and_soot_blowing import SootBlowingSimulator
from analysis_and_visualization import SystemAnalyzer


class AnnualBoilerSimulator:
    """Simulate a full year of boiler operation with realistic conditions."""
    
    def __init__(self, start_date: str = "2024-01-01"):
        """Initialize the annual boiler simulator."""
        self.start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = self.start_date + datetime.timedelta(days=365)
        
        # Initialize boiler system
        self.boiler = EnhancedCompleteBoilerSystem(
            fuel_input=100e6,
            flue_gas_mass_flow=84000,
            furnace_exit_temp=3000,
            base_fouling_multiplier=1.0
        )
        
        self.fouling_integrator = CombustionFoulingIntegrator()
        self.property_calc = PropertyCalculator()
        
        # Soot blowing schedule configuration (realistic frequencies in hours)
        self.soot_blowing_schedule = {
            'furnace_walls': 8,          # Every 8 hours (3x per day)
            'generating_bank': 12,       # Every 12 hours (2x per day)
            'superheater_primary': 16,   # Every 16 hours (1.5x per day)
            'superheater_secondary': 24, # Every 24 hours (1x per day)
            'economizer_primary': 48,    # Every 48 hours (every 2 days)
            'economizer_secondary': 72,  # Every 72 hours (every 3 days)
            'air_heater': 168           # Every 168 hours (every 7 days)
        }
        
        # Track last cleaning dates
        self.last_cleaned = {section: self.start_date for section in self.soot_blowing_schedule.keys()}
        
        # Massachusetts weather data patterns
        self.ma_weather_patterns = self._initialize_ma_weather()
        
        # Coal quality variations
        self.coal_quality_profiles = self._initialize_coal_profiles()
        
        print("✅ Annual Boiler Simulator initialized for Massachusetts operation")
        print(f"   Simulation period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print(f"   Soot blowing schedule (hours between cycles):")
        for section, hours in self.soot_blowing_schedule.items():
            times_per_day = 24 / hours
            print(f"     • {section}: {hours}h cycles ({times_per_day:.1f}x/day)")
    
    def _initialize_ma_weather(self) -> Dict:
        """Initialize Massachusetts weather patterns by month."""
        return {
            1: {'temp_avg': 30, 'temp_range': 25, 'humidity_avg': 65, 'humidity_range': 20},  # January
            2: {'temp_avg': 35, 'temp_range': 25, 'humidity_avg': 62, 'humidity_range': 20},  # February
            3: {'temp_avg': 45, 'temp_range': 25, 'humidity_avg': 60, 'humidity_range': 20},  # March
            4: {'temp_avg': 55, 'temp_range': 25, 'humidity_avg': 58, 'humidity_range': 20},  # April
            5: {'temp_avg': 65, 'temp_range': 25, 'humidity_avg': 62, 'humidity_range': 20},  # May
            6: {'temp_avg': 75, 'temp_range': 20, 'humidity_avg': 68, 'humidity_range': 15},  # June
            7: {'temp_avg': 80, 'temp_range': 20, 'humidity_avg': 70, 'humidity_range': 15},  # July
            8: {'temp_avg': 78, 'temp_range': 20, 'humidity_avg': 72, 'humidity_range': 15},  # August
            9: {'temp_avg': 70, 'temp_range': 20, 'humidity_avg': 68, 'humidity_range': 15},  # September
            10: {'temp_avg': 60, 'temp_range': 25, 'humidity_avg': 65, 'humidity_range': 20}, # October
            11: {'temp_avg': 48, 'temp_range': 25, 'humidity_avg': 65, 'humidity_range': 20}, # November
            12: {'temp_avg': 35, 'temp_range': 25, 'humidity_avg': 68, 'humidity_range': 20}  # December
        }
    
    def _initialize_coal_profiles(self) -> Dict:
        """Initialize different coal quality profiles."""
        return {
            'high_quality': {
                'carbon': 75.0, 'volatile_matter': 32.0, 'fixed_carbon': 58.0,
                'sulfur': 0.8, 'ash': 7.0, 'moisture': 2.0,
                'heating_value': 13000,  # BTU/lb
                'description': 'Premium bituminous coal'
            },
            'medium_quality': {
                'carbon': 72.0, 'volatile_matter': 28.0, 'fixed_carbon': 55.0,
                'sulfur': 1.2, 'ash': 8.5, 'moisture': 3.0,
                'heating_value': 12000,
                'description': 'Standard bituminous coal'
            },
            'low_quality': {
                'carbon': 68.0, 'volatile_matter': 25.0, 'fixed_carbon': 50.0,
                'sulfur': 2.0, 'ash': 12.0, 'moisture': 5.0,
                'heating_value': 11000,
                'description': 'Sub-bituminous coal'
            },
            'waste_coal': {
                'carbon': 65.0, 'volatile_matter': 22.0, 'fixed_carbon': 45.0,
                'sulfur': 2.5, 'ash': 15.0, 'moisture': 6.0,
                'heating_value': 10000,
                'description': 'Waste/refuse coal blend'
            }
        }
    
    def generate_annual_data(self, hours_per_day: int = 24, save_interval_hours: int = 1) -> pd.DataFrame:
        """
        Generate comprehensive annual operation data.
        
        Args:
            hours_per_day: Operating hours per day (24 for continuous operation)
            save_interval_hours: How often to save data points (4 = every 4 hours)
        
        Returns:
            DataFrame with complete annual operation data
        """
        print(f"\n🔄 Starting annual simulation...")
        print(f"   Operating {hours_per_day} hours/day, recording every {save_interval_hours} hours")
        
        annual_data = []
        current_date = self.start_date
        record_counter = 0
        
        while current_date < self.end_date:
            # Generate daily operating schedule
            daily_hours = min(hours_per_day, 24)
            
            for hour in range(0, daily_hours, save_interval_hours):
                current_datetime = current_date + datetime.timedelta(hours=hour)
                
                # Skip if we've reached the end date
                if current_datetime >= self.end_date:
                    break
                
                # Generate operating conditions for this time point
                operating_conditions = self._generate_hourly_conditions(current_datetime)
                
                # Check and apply soot blowing if scheduled
                soot_blowing_actions = self._check_soot_blowing_schedule(current_datetime)
                
                # Simulate boiler operation
                operation_data = self._simulate_boiler_operation(
                    current_datetime, operating_conditions, soot_blowing_actions
                )
                
                annual_data.append(operation_data)
                record_counter += 1
                
                # Progress reporting
                if record_counter % 500 == 0:
                    progress = (current_datetime - self.start_date).days / 365 * 100
                    print(f"   Progress: {progress:.1f}% - {current_datetime.strftime('%Y-%m-%d %H:%M')} - {len(annual_data)} records")
            
            # Move to next day
            current_date += datetime.timedelta(days=1)
        
        # Convert to DataFrame
        df = pd.DataFrame(annual_data)
        
        print(f"\n✅ Annual simulation complete!")
        print(f"   Total records generated: {len(df)}")
        print(f"   Data columns: {len(df.columns)}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def _generate_hourly_conditions(self, current_datetime: datetime.datetime) -> Dict:
        """Generate realistic operating conditions for a specific hour."""
        month = current_datetime.month
        hour = current_datetime.hour
        day_of_year = current_datetime.timetuple().tm_yday
        
        # Load variation patterns
        load_factor = self._calculate_load_factor(current_datetime, hour, day_of_year)
        
        # Weather conditions for Massachusetts
        weather = self._generate_weather_conditions(month, day_of_year)
        
        # Coal quality (changes periodically)
        coal_quality = self._select_coal_quality(day_of_year)
        
        # Operating parameters based on load
        base_coal_rate = 8500  # lb/hr for 100% load
        coal_rate = base_coal_rate * load_factor
        
        # Air flow adjusted for load and ambient conditions
        stoichiometric_air = coal_rate * 10  # Simplified stoichiometric ratio
        excess_air_factor = np.random.uniform(1.15, 1.35)  # 15-35% excess air
        air_flow = stoichiometric_air * excess_air_factor
        
        # Adjust air flow for temperature (density effect)
        air_density_correction = (460 + 60) / (460 + weather['temperature'])
        air_flow_corrected = air_flow * air_density_correction
        
        return {
            'load_factor': load_factor,
            'coal_rate_lb_hr': coal_rate,
            'air_flow_scfh': air_flow_corrected,
            'ambient_temp_F': weather['temperature'],
            'ambient_humidity_pct': weather['humidity'],
            'coal_quality': coal_quality,
            'nox_efficiency': np.random.uniform(0.25, 0.45),
            'season': self._get_season(month),
            'day_of_year': day_of_year,
            'hour_of_day': hour
        }
    
    def _calculate_load_factor(self, current_datetime: datetime.datetime, 
                              hour: int, day_of_year: int) -> float:
        """Calculate realistic load factor based on time patterns."""
        # Base load varies seasonally (higher in winter for heating)
        month = current_datetime.month
        if month in [12, 1, 2]:  # Winter
            base_load = 0.85
        elif month in [6, 7, 8]:  # Summer
            base_load = 0.65
        else:  # Spring/Fall
            base_load = 0.75
        
        # Daily load profile (industrial pattern)
        if 6 <= hour <= 18:  # Daytime
            daily_multiplier = 1.1
        elif 22 <= hour or hour <= 2:  # Night
            daily_multiplier = 0.9
        else:  # Transition
            daily_multiplier = 1.0
        
        # Weekly pattern (lower on weekends)
        if current_datetime.weekday() in [5, 6]:  # Weekend
            weekly_multiplier = 0.85
        else:
            weekly_multiplier = 1.0
        
        # Random variation
        random_variation = np.random.uniform(0.95, 1.05)
        
        # Calculate final load factor
        load_factor = base_load * daily_multiplier * weekly_multiplier * random_variation
        
        # Clamp to specified range (45-100%)
        return max(0.45, min(1.0, load_factor))
    
    def _generate_weather_conditions(self, month: int, day_of_year: int) -> Dict:
        """Generate realistic weather conditions for Massachusetts."""
        weather_pattern = self.ma_weather_patterns[month]
        
        # Add daily and random variations
        daily_temp_variation = 10 * np.sin((day_of_year % 365) * 2 * np.pi / 365)
        random_temp_variation = np.random.uniform(-weather_pattern['temp_range']/2, 
                                                weather_pattern['temp_range']/2)
        
        temperature = (weather_pattern['temp_avg'] + 
                      daily_temp_variation + 
                      random_temp_variation)
        
        # Humidity variation
        random_humidity_variation = np.random.uniform(-weather_pattern['humidity_range']/2,
                                                    weather_pattern['humidity_range']/2)
        humidity = max(20, min(95, weather_pattern['humidity_avg'] + random_humidity_variation))
        
        return {
            'temperature': temperature,
            'humidity': humidity
        }
    
    def _select_coal_quality(self, day_of_year: int) -> str:
        """Select coal quality based on delivery schedules and market conditions."""
        # Coal delivery typically every 2-4 weeks
        delivery_cycle = (day_of_year // 21) % 4
        
        # Quality distribution (realistic for industrial boiler)
        quality_probabilities = {
            'high_quality': 0.20,    # 20% premium coal
            'medium_quality': 0.60,  # 60% standard coal
            'low_quality': 0.15,     # 15% lower grade
            'waste_coal': 0.05       # 5% waste coal blend
        }
        
        # Seasonal adjustments (better coal in winter for reliability)
        month = ((day_of_year - 1) // 30) + 1
        if month in [12, 1, 2]:  # Winter - prefer higher quality
            quality_probabilities['high_quality'] = 0.35
            quality_probabilities['medium_quality'] = 0.50
            quality_probabilities['low_quality'] = 0.10
            quality_probabilities['waste_coal'] = 0.05
        
        # Select based on probabilities
        rand_val = random.random()
        cumulative = 0
        for quality, prob in quality_probabilities.items():
            cumulative += prob
            if rand_val <= cumulative:
                return quality
        
        return 'medium_quality'  # Default fallback
    
    def _get_season(self, month: int) -> str:
        """Get season name from month."""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _check_soot_blowing_schedule(self, current_datetime: datetime.datetime) -> Dict:
        """Check if any sections need soot blowing based on realistic hourly schedule."""
        soot_blowing_actions = {}
        
        for section_name, interval_hours in self.soot_blowing_schedule.items():
            last_cleaned = self.last_cleaned[section_name]
            hours_since_cleaned = (current_datetime - last_cleaned).total_seconds() / 3600
            
            if hours_since_cleaned >= interval_hours:
                # Time for soot blowing
                soot_blowing_actions[section_name] = {
                    'action': True,
                    'hours_since_last': hours_since_cleaned,
                    'effectiveness': np.random.uniform(0.75, 0.95),  # Cleaning effectiveness
                    'segments_cleaned': 'all'  # Clean all segments
                }
                
                # Update last cleaned date
                self.last_cleaned[section_name] = current_datetime
            else:
                soot_blowing_actions[section_name] = {
                    'action': False,
                    'hours_since_last': hours_since_cleaned,
                    'effectiveness': 0.0,
                    'segments_cleaned': None
                }
        
        return soot_blowing_actions
    
    def _simulate_boiler_operation(self, current_datetime: datetime.datetime,
                                  operating_conditions: Dict,
                                  soot_blowing_actions: Dict) -> Dict:
        """Simulate complete boiler operation for one time point."""
        
        # Update boiler operating conditions
        fuel_input = operating_conditions['coal_rate_lb_hr'] * \
                    self.coal_quality_profiles[operating_conditions['coal_quality']]['heating_value']
        
        # Calculate flue gas flow based on coal and air
        flue_gas_flow = operating_conditions['coal_rate_lb_hr'] * 12 + \
                       operating_conditions['air_flow_scfh'] * 0.075
        
        # Update boiler system
        self.boiler.update_operating_conditions(
            fuel_input=fuel_input,
            flue_gas_mass_flow=flue_gas_flow,
            furnace_exit_temp=3000 - (1.0 - operating_conditions['load_factor']) * 200
        )
        
        # Apply soot blowing if scheduled
        for section_name, action in soot_blowing_actions.items():
            if action['action']:
                section = self.boiler.sections[section_name]
                # Clean all segments with specified effectiveness
                all_segments = list(range(section.num_segments))
                section.apply_soot_blowing(all_segments, action['effectiveness'])
        
        # Set up combustion model
        coal_props = self.coal_quality_profiles[operating_conditions['coal_quality']]
        ultimate_analysis = {
            'C': coal_props['carbon'],
            'H': 5.0,
            'O': 10.0,
            'N': 1.5,
            'S': coal_props['sulfur'],
            'Ash': coal_props['ash'],
            'Moisture': coal_props['moisture']
        }
        
        combustion_model = CoalCombustionModel(
            ultimate_analysis=ultimate_analysis,
            coal_lb_per_hr=operating_conditions['coal_rate_lb_hr'],
            air_scfh=operating_conditions['air_flow_scfh'],
            NOx_eff=operating_conditions['nox_efficiency'],
            air_temp_F=operating_conditions['ambient_temp_F'],
            air_RH_pct=operating_conditions['ambient_humidity_pct']
        )
        
        combustion_model.calculate()
        
        # Solve boiler system
        try:
            self.boiler.solve_enhanced_system(max_iterations=8, tolerance=10.0)
            system_performance = self.boiler.system_performance
            solution_converged = True
        except Exception as e:
            print(f"Warning: Boiler solution failed at {current_datetime}: {e}")
            # Use default values if solution fails
            system_performance = {
                'system_efficiency': 0.80,
                'final_steam_temperature': 700,
                'stack_temperature': 350,
                'total_heat_absorbed': fuel_input * 0.80,
                'steam_production': 68000,
                'attemperator_flow': 0
            }
            solution_converged = False
        
        # Collect comprehensive data
        operation_data = {
            # Timestamp and conditions
            'timestamp': current_datetime,
            'year': current_datetime.year,
            'month': current_datetime.month,
            'day': current_datetime.day,
            'hour': current_datetime.hour,
            'day_of_year': operating_conditions['day_of_year'],
            'season': operating_conditions['season'],
            
            # Operating conditions
            'load_factor': operating_conditions['load_factor'],
            'ambient_temp_F': operating_conditions['ambient_temp_F'],
            'ambient_humidity_pct': operating_conditions['ambient_humidity_pct'],
            'coal_quality': operating_conditions['coal_quality'],
            
            # Fuel and air flows
            'coal_rate_lb_hr': operating_conditions['coal_rate_lb_hr'],
            'air_flow_scfh': operating_conditions['air_flow_scfh'],
            'fuel_input_btu_hr': fuel_input,
            'flue_gas_flow_lb_hr': flue_gas_flow,
            
            # Coal properties
            'coal_carbon_pct': coal_props['carbon'],
            'coal_volatile_matter_pct': coal_props['volatile_matter'],
            'coal_sulfur_pct': coal_props['sulfur'],
            'coal_ash_pct': coal_props['ash'],
            'coal_moisture_pct': coal_props['moisture'],
            'coal_heating_value_btu_lb': coal_props['heating_value'],
            
            # Combustion results
            'thermal_nox_lb_hr': combustion_model.NO_thermal_lb_per_hr,
            'fuel_nox_lb_hr': combustion_model.NO_fuel_lb_per_hr,
            'total_nox_lb_hr': combustion_model.NO_total_lb_per_hr,
            'excess_o2_pct': combustion_model.dry_O2_pct,
            'combustion_efficiency': combustion_model.combustion_efficiency,
            'flame_temp_F': combustion_model.flame_temp_F,
            
            # System performance
            'system_efficiency': system_performance['system_efficiency'],
            'final_steam_temp_F': system_performance['final_steam_temperature'],
            'stack_temp_F': system_performance['stack_temperature'],
            'total_heat_absorbed_btu_hr': system_performance['total_heat_absorbed'],
            'steam_production_lb_hr': system_performance['steam_production'],
            'attemperator_flow_lb_hr': system_performance['attemperator_flow'],
            
            # Solution status
            'solution_converged': solution_converged,
            
            # Soot blowing status
            'soot_blowing_active': any(action['action'] for action in soot_blowing_actions.values()),
            'sections_cleaned_count': sum(1 for action in soot_blowing_actions.values() if action['action'])
        }
        
        # Add fouling factors for each section
        for section_name, section in self.boiler.sections.items():
            fouling_arrays = section.get_current_fouling_arrays()
            
            # Statistical summary of fouling
            gas_fouling = fouling_arrays['gas']
            water_fouling = fouling_arrays['water']
            
            operation_data.update({
                f'{section_name}_fouling_gas_avg': np.mean(gas_fouling),
                f'{section_name}_fouling_gas_max': np.max(gas_fouling),
                f'{section_name}_fouling_gas_min': np.min(gas_fouling),
                f'{section_name}_fouling_water_avg': np.mean(water_fouling),
                f'{section_name}_fouling_water_max': np.max(water_fouling),
                f'{section_name}_hours_since_cleaning': soot_blowing_actions[section_name]['hours_since_last']
            })
        
        # Add section temperatures and heat transfer
        for section_name, data in self.boiler.section_results.items():
            summary = data['summary']
            operation_data.update({
                f'{section_name}_gas_temp_in_F': summary['gas_temp_in'],
                f'{section_name}_gas_temp_out_F': summary['gas_temp_out'],
                f'{section_name}_water_temp_in_F': summary['water_temp_in'],
                f'{section_name}_water_temp_out_F': summary['water_temp_out'],
                f'{section_name}_heat_transfer_btu_hr': summary['total_heat_transfer'],
                f'{section_name}_overall_U_avg': summary['average_overall_U']
            })
        
        return operation_data
    
    def save_annual_data(self, df: pd.DataFrame, 
                        filename_prefix: str = "massachusetts_boiler_annual") -> str:
        """Save annual data to CSV with comprehensive metadata."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.csv"
        
        # Save main data
        df.to_csv(filename, index=False)
        
        # Create metadata file
        metadata_filename = f"{filename_prefix}_metadata_{timestamp}.txt"
        with open(metadata_filename, 'w') as f:
            f.write("MASSACHUSETTS BOILER ANNUAL OPERATION DATASET\n")
            f.write("="*50 + "\n\n")
            f.write(f"Generation Date: {datetime.datetime.now()}\n")
            f.write(f"Simulation Period: {self.start_date} to {self.end_date}\n")
            f.write(f"Total Records: {len(df)}\n")
            f.write(f"Total Columns: {len(df.columns)}\n\n")
            
            f.write("OPERATIONAL PARAMETERS:\n")
            f.write(f"- Load Range: 45-100% of maximum capacity\n")
            f.write(f"- Recording Interval: Every 4 hours\n")
            f.write(f"- Maximum Capacity: 100 MMBtu/hr\n")
            f.write(f"- Location: Massachusetts, USA\n\n")
            
            f.write("SOOT BLOWING SCHEDULE:\n")
            for section, interval in self.soot_blowing_schedule.items():
                times_per_day = 24 / interval
                f.write(f"- {section}: Every {interval} hours ({times_per_day:.1f}x per day)\n")
            
            f.write(f"\nCOAL QUALITY PROFILES:\n")
            for quality, props in self.coal_quality_profiles.items():
                f.write(f"- {quality}: {props['description']}\n")
                f.write(f"  Carbon: {props['carbon']}%, Sulfur: {props['sulfur']}%, Ash: {props['ash']}%\n")
            
            f.write(f"\nDATASET COLUMNS ({len(df.columns)} total):\n")
            for i, col in enumerate(df.columns, 1):
                f.write(f"{i:3d}. {col}\n")
        
        print(f"\n💾 Annual dataset saved:")
        print(f"   Data file: {filename}")
        print(f"   Metadata: {metadata_filename}")
        print(f"   Records: {len(df):,}")
        print(f"   Size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        return filename


def main():
    """Demonstrate the annual boiler simulation."""
    print("🏭" * 25)
    print("MASSACHUSETTS BOILER ANNUAL OPERATION SIMULATION")
    print("🏭" * 25)
    
    # Initialize simulator
    simulator = AnnualBoilerSimulator(start_date="2024-01-01")
    
    # Generate annual data (recording every 4 hours = 2190 records per year)
    print("\n🔄 Generating annual operation dataset...")
    annual_df = simulator.generate_annual_data(
        hours_per_day=24,  # Continuous operation
        save_interval_hours=1  # Record every 4 hours
    )
    
    # Display dataset summary
    print(f"\n📊 DATASET SUMMARY:")
    print(f"   Total records: {len(annual_df):,}")
    print(f"   Date range: {annual_df['timestamp'].min()} to {annual_df['timestamp'].max()}")
    print(f"   Columns: {len(annual_df.columns)}")
    
    print(f"\n📈 OPERATIONAL STATISTICS:")
    print(f"   Load factor: {annual_df['load_factor'].mean():.1%} ± {annual_df['load_factor'].std():.1%}")
    print(f"   System efficiency: {annual_df['system_efficiency'].mean():.1%} ± {annual_df['system_efficiency'].std():.1%}")
    print(f"   Steam temperature: {annual_df['final_steam_temp_F'].mean():.0f}°F ± {annual_df['final_steam_temp_F'].std():.0f}°F")
    print(f"   Stack temperature: {annual_df['stack_temp_F'].mean():.0f}°F ± {annual_df['stack_temp_F'].std():.0f}°F")
    
    print(f"\n🌡️ AMBIENT CONDITIONS:")
    print(f"   Temperature: {annual_df['ambient_temp_F'].min():.0f}°F to {annual_df['ambient_temp_F'].max():.0f}°F")
    print(f"   Humidity: {annual_df['ambient_humidity_pct'].min():.0f}% to {annual_df['ambient_humidity_pct'].max():.0f}%")
    
    print(f"\n⚡ COAL USAGE:")
    coal_quality_counts = annual_df['coal_quality'].value_counts()
    for quality, count in coal_quality_counts.items():
        percentage = count / len(annual_df) * 100
        print(f"   {quality}: {count} records ({percentage:.1f}%)")
    
    print(f"\n🧹 SOOT BLOWING ACTIVITY:")
    soot_blowing_events = annual_df['soot_blowing_active'].sum()
    print(f"   Total soot blowing events: {soot_blowing_events}")
    print(f"   Percentage of time with active cleaning: {soot_blowing_events/len(annual_df)*100:.1f}%")
    
    # Save the dataset
    filename = simulator.save_annual_data(annual_df, "massachusetts_boiler_annual")
    
    print(f"\n✅ SIMULATION COMPLETE!")
    print(f"   Annual operation dataset ready for analysis")
    print(f"   File: {filename}")
    print(f"   This dataset contains all requested data:")
    print(f"   ✓ Variable load operation (45-100% capacity)")
    print(f"   ✓ Massachusetts seasonal weather patterns")
    print(f"   ✓ Coal quality variations (4 different grades)")
    print(f"   ✓ Scheduled soot blowing cycles for all sections")
    print(f"   ✓ Complete fouling factors for all tube sections")
    print(f"   ✓ All temperatures (gas/water in/out for each section)")
    print(f"   ✓ All flow rates (coal, air, steam, flue gas)")
    print(f"   ✓ Complete stack gas analysis (NOx, O2, efficiency)")
    print(f"   ✓ System performance metrics")
    
    # Show sample of key columns
    print(f"\n📋 KEY DATA COLUMNS SAMPLE:")
    key_columns = [
        'timestamp', 'load_factor', 'ambient_temp_F', 'coal_quality',
        'system_efficiency', 'stack_temp_F', 'total_nox_lb_hr', 'excess_o2_pct',
        'soot_blowing_active', 'furnace_walls_fouling_gas_avg'
    ]
    
    sample_data = annual_df[key_columns].head(10)
    print(sample_data.to_string(index=False))
    
    return annual_df


if __name__ == "__main__":
    # Run the annual simulation
    annual_data = main()
    
    print(f"\n🎯 DATASET READY FOR ANALYSIS!")
    print(f"   Use this data for:")
    print(f"   • Predictive maintenance scheduling")
    print(f"   • Soot blowing optimization")
    print(f"   • Efficiency trend analysis")
    print(f"   • Emissions modeling")
    print(f"   • Coal quality impact studies")
    print(f"   • Seasonal performance analysis")