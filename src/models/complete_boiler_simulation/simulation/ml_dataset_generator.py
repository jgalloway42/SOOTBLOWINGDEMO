#!/usr/bin/env python3
"""
ML Dataset Generation for Soot Blowing Optimization

This module generates comprehensive datasets for machine learning-based soot blowing
optimization, linking combustion conditions to fouling patterns and cleaning effectiveness.

Classes:
    MLDatasetGenerator: Main dataset generation class

Dependencies:
    - numpy: Numerical calculations
    - pandas: Data manipulation and analysis
    - coal_combustion_models: Combustion and soot production models
    - boiler_system: Complete boiler system model
    - typing: Type hints

Author: Enhanced Boiler Modeling System
Version: 5.0 - ML Dataset Generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from core.coal_combustion_models import CoalCombustionModel, CombustionFoulingIntegrator
from core.boiler_system import EnhancedCompleteBoilerSystem


class MLDatasetGenerator:
    """Generate comprehensive datasets for machine learning-based soot blowing optimization."""
    
    def __init__(self, boiler_system: EnhancedCompleteBoilerSystem):
        """Initialize ML dataset generator."""
        self.boiler_system = boiler_system
        self.fouling_integrator = CombustionFoulingIntegrator()
        self.dataset = []
        
    def generate_comprehensive_dataset(self, num_scenarios: int = 10000) -> pd.DataFrame:
        """Generate comprehensive dataset for ML training."""
        print(f"🤖 Generating comprehensive dataset with {num_scenarios} scenarios...")
        
        scenarios_data = []
        
        for i in range(num_scenarios):
            if i % 1000 == 0:
                print(f"  Progress: {i}/{num_scenarios} scenarios generated...")
            
            # Generate random operating scenario
            scenario = self._generate_random_scenario()
            
            # Calculate combustion and soot production
            combustion_data = self._calculate_combustion_scenario(scenario)
            
            # Calculate fouling buildup over time
            fouling_timeline = self._simulate_fouling_timeline(
                scenario, combustion_data, timeline_hours=720  # 30 days
            )
            
            # Generate multiple cleaning scenarios for this operating condition
            cleaning_scenarios = self._generate_cleaning_scenarios(fouling_timeline)
            
            # Evaluate each cleaning scenario
            for cleaning in cleaning_scenarios:
                scenario_data = self._evaluate_cleaning_scenario(
                    scenario, combustion_data, fouling_timeline, cleaning
                )
                scenarios_data.append(scenario_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(scenarios_data)
        
        print(f"✅ Dataset generation complete: {len(df)} total data points")
        return df
    
    def _generate_random_scenario(self) -> Dict:
        """Generate random operating scenario."""
        # Coal properties variation
        coal_properties = {
            'carbon': np.random.normal(72, 5),      # % carbon
            'volatile_matter': np.random.normal(30, 5),  # % VM
            'fixed_carbon': np.random.normal(50, 5),     # % FC
            'sulfur': np.random.uniform(0.5, 3.0),       # % sulfur
            'ash': np.random.uniform(5, 15)              # % ash
        }
        
        # Ultimate analysis (normalized to 100%)
        moisture = np.random.uniform(2, 8)
        ultimate = {
            'C': coal_properties['carbon'],
            'H': np.random.uniform(4, 6),
            'O': np.random.uniform(8, 12),
            'N': np.random.uniform(1, 2),
            'S': coal_properties['sulfur'],
            'Ash': coal_properties['ash'],
            'Moisture': moisture
        }
        
        # Normalize to 100%
        total = sum(ultimate.values())
        ultimate = {k: v * 100 / total for k, v in ultimate.items()}
        
        # Operating conditions
        base_coal_rate = 8500  # lb/hr for 100 MMBtu/hr
        load_factor = np.random.uniform(0.6, 1.2)  # 60-120% load
        
        scenario = {
            'coal_lb_per_hr': base_coal_rate * load_factor,
            'air_scfh': np.random.uniform(800000, 1200000),  # Varying excess air
            'NOx_eff': np.random.uniform(0.2, 0.6),          # NOx conversion efficiency
            'air_temp_F': np.random.uniform(60, 100),        # Ambient air temp
            'air_RH_pct': np.random.uniform(30, 80),         # Humidity
            'ultimate_analysis': ultimate,
            'coal_properties': coal_properties,
            'operating_hours_since_clean': np.random.uniform(0, 1440),  # 0-60 days
            'season': np.random.choice(['winter', 'spring', 'summer', 'fall']),
            'fuel_quality': np.random.choice(['high', 'medium', 'low'])
        }
        
        return scenario
    
    def _calculate_combustion_scenario(self, scenario: Dict) -> Dict:
        """Calculate combustion results for scenario."""
        # Create combustion model
        combustion_model = CoalCombustionModel(
            ultimate_analysis=scenario['ultimate_analysis'],
            coal_lb_per_hr=scenario['coal_lb_per_hr'],
            air_scfh=scenario['air_scfh'],
            NOx_eff=scenario['NOx_eff'],
            air_temp_F=scenario['air_temp_F'],
            air_RH_pct=scenario['air_RH_pct']
        )
        
        # Calculate combustion
        combustion_model.calculate()
        
        # Calculate fouling rates
        fouling_rates = self.fouling_integrator.calculate_section_fouling_rates(
            combustion_model, scenario['coal_properties'], self.boiler_system
        )
        
        return {
            'combustion_model': combustion_model,
            'fouling_rates': fouling_rates,
            'thermal_nox': combustion_model.NO_thermal_lb_per_hr,
            'fuel_nox': combustion_model.NO_fuel_lb_per_hr,
            'total_nox': combustion_model.NO_total_lb_per_hr,
            'excess_o2': combustion_model.dry_O2_pct,
            'combustion_efficiency': combustion_model.combustion_efficiency,
            'flame_temp': combustion_model.flame_temp_F,
            'heat_release': combustion_model.heat_released_btu_per_hr
        }
    
    def _simulate_fouling_timeline(self, scenario: Dict, combustion_data: Dict,
                                  timeline_hours: int = 720) -> Dict:
        """Simulate fouling buildup over time."""
        fouling_rates = combustion_data['fouling_rates']
        
        # Initialize current fouling levels
        current_fouling = {}
        for section_name in fouling_rates.keys():
            section = self.boiler_system.sections[section_name]
            current_fouling[section_name] = {
                'gas': [section.base_fouling_gas] * section.num_segments,
                'water': [section.base_fouling_water] * section.num_segments
            }
        
        # Simulate hourly buildup
        timeline = []
        for hour in range(timeline_hours):
            # Apply fouling buildup
            for section_name in fouling_rates.keys():
                rates = fouling_rates[section_name]
                for i in range(len(rates['gas'])):
                    current_fouling[section_name]['gas'][i] += rates['gas'][i]
                    current_fouling[section_name]['water'][i] += rates['water'][i]
            
            # Record state every 24 hours
            if hour % 24 == 0:
                timeline.append({
                    'hour': hour,
                    'fouling_state': {k: {'gas': v['gas'].copy(), 'water': v['water'].copy()} 
                                    for k, v in current_fouling.items()},
                    'performance_impact': self._calculate_performance_impact(current_fouling)
                })
        
        return {
            'timeline': timeline,
            'final_fouling': current_fouling,
            'total_hours': timeline_hours
        }
    
    def _calculate_performance_impact(self, fouling_state: Dict) -> Dict:
        """Calculate performance impact of current fouling state."""
        # Simplified performance calculation
        total_thermal_resistance = 0
        total_segments = 0
        
        for section_name, fouling in fouling_state.items():
            for gas_foul in fouling['gas']:
                total_thermal_resistance += gas_foul
                total_segments += 1
        
        avg_thermal_resistance = total_thermal_resistance / total_segments if total_segments > 0 else 0
        
        # Estimate performance degradation
        efficiency_loss = min(0.05, avg_thermal_resistance * 1000)  # Max 5% loss
        heat_transfer_loss = min(0.10, avg_thermal_resistance * 2000)  # Max 10% loss
        
        return {
            'efficiency_loss': efficiency_loss,
            'heat_transfer_loss': heat_transfer_loss,
            'avg_thermal_resistance': avg_thermal_resistance
        }
    
    def _generate_cleaning_scenarios(self, fouling_timeline: Dict) -> List[Dict]:
        """Generate various cleaning scenarios to evaluate."""
        cleaning_scenarios = []
        
        # Get final fouling state
        final_fouling = fouling_timeline['final_fouling']
        
        # Strategy 1: Clean everything
        cleaning_scenarios.append({
            'strategy': 'clean_all',
            'sections_to_clean': list(final_fouling.keys()),
            'segments_per_section': {name: list(range(len(fouling['gas']))) 
                                   for name, fouling in final_fouling.items()},
            'cleaning_effectiveness': 0.9
        })
        
        # Strategy 2: Clean worst sections only
        worst_sections = self._identify_worst_sections(final_fouling, top_n=3)
        cleaning_scenarios.append({
            'strategy': 'clean_worst',
            'sections_to_clean': worst_sections,
            'segments_per_section': {name: list(range(len(final_fouling[name]['gas']))) 
                                   for name in worst_sections},
            'cleaning_effectiveness': 0.85
        })
        
        # Strategy 3: Selective segment cleaning
        cleaning_scenarios.append({
            'strategy': 'selective_segments',
            'sections_to_clean': list(final_fouling.keys()),
            'segments_per_section': self._identify_worst_segments(final_fouling),
            'cleaning_effectiveness': 0.8
        })
        
        # Strategy 4: Progressive cleaning (economizers first)
        econ_sections = [name for name in final_fouling.keys() if 'economizer' in name]
        cleaning_scenarios.append({
            'strategy': 'economizers_first',
            'sections_to_clean': econ_sections,
            'segments_per_section': {name: list(range(len(final_fouling[name]['gas']))) 
                                   for name in econ_sections},
            'cleaning_effectiveness': 0.9
        })
        
        # Strategy 5: No cleaning (baseline)
        cleaning_scenarios.append({
            'strategy': 'no_cleaning',
            'sections_to_clean': [],
            'segments_per_section': {},
            'cleaning_effectiveness': 0.0
        })
        
        # Strategy 6: Random partial cleaning
        random_sections = np.random.choice(list(final_fouling.keys()), 
                                         size=np.random.randint(1, 4), replace=False)
        cleaning_scenarios.append({
            'strategy': 'random_partial',
            'sections_to_clean': random_sections.tolist(),
            'segments_per_section': {name: np.random.choice(range(len(final_fouling[name]['gas'])), 
                                                          size=np.random.randint(2, len(final_fouling[name]['gas'])), 
                                                          replace=False).tolist()
                                   for name in random_sections},
            'cleaning_effectiveness': np.random.uniform(0.7, 0.95)
        })
        
        return cleaning_scenarios
    
    def _identify_worst_sections(self, fouling_state: Dict, top_n: int = 3) -> List[str]:
        """Identify sections with worst fouling."""
        section_scores = {}
        
        for section_name, fouling in fouling_state.items():
            # Calculate average fouling level
            avg_gas_fouling = np.mean(fouling['gas'])
            max_gas_fouling = np.max(fouling['gas'])
            section_scores[section_name] = avg_gas_fouling + max_gas_fouling * 0.5
        
        # Sort by score and return top N
        sorted_sections = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)
        return [name for name, score in sorted_sections[:top_n]]
    
    def _identify_worst_segments(self, fouling_state: Dict) -> Dict[str, List[int]]:
        """Identify worst segments in each section."""
        worst_segments = {}
        
        for section_name, fouling in fouling_state.items():
            gas_fouling = fouling['gas']
            avg_fouling = np.mean(gas_fouling)
            threshold = avg_fouling * 1.3  # 30% above average
            
            worst_segs = [i for i, f in enumerate(gas_fouling) if f > threshold]
            if not worst_segs:  # If none above threshold, take worst half
                sorted_indices = sorted(range(len(gas_fouling)), 
                                      key=lambda i: gas_fouling[i], reverse=True)
                worst_segs = sorted_indices[:len(gas_fouling)//2 + 1]
            
            worst_segments[section_name] = worst_segs
        
        return worst_segments
    
    def _evaluate_cleaning_scenario(self, scenario: Dict, combustion_data: Dict,
                                  fouling_timeline: Dict, cleaning: Dict) -> Dict:
        """Evaluate a specific cleaning scenario."""
        # Calculate performance before cleaning
        before_fouling = fouling_timeline['final_fouling']
        before_performance = self._calculate_performance_impact(before_fouling)
        
        # Apply cleaning
        after_fouling = self._apply_cleaning(before_fouling, cleaning)
        after_performance = self._calculate_performance_impact(after_fouling)
        
        # Calculate cleaning costs and benefits
        cleaning_cost = self._calculate_cleaning_cost(cleaning)
        cleaning_time = self._calculate_cleaning_time(cleaning)
        
        # Performance improvements
        efficiency_gain = before_performance['efficiency_loss'] - after_performance['efficiency_loss']
        heat_transfer_gain = before_performance['heat_transfer_loss'] - after_performance['heat_transfer_loss']
        
        # Economic calculations
        fuel_savings_per_hour = efficiency_gain * scenario['coal_lb_per_hr'] * 12000 * 5.0 / 1e6  # $/hr
        payback_hours = cleaning_cost / fuel_savings_per_hour if fuel_savings_per_hour > 0 else 9999
        
        # Compile feature set
        features = self._extract_features(scenario, combustion_data, before_fouling, cleaning)
        
        # Compile targets
        targets = {
            'efficiency_gain': efficiency_gain,
            'heat_transfer_gain': heat_transfer_gain,
            'fuel_savings_per_hour': fuel_savings_per_hour,
            'cleaning_cost': cleaning_cost,
            'cleaning_time': cleaning_time,
            'payback_hours': min(payback_hours, 9999),  # Cap at reasonable value
            'roi_24hr': fuel_savings_per_hour * 24 / cleaning_cost if cleaning_cost > 0 else 0,
            'performance_score': self._calculate_performance_score(efficiency_gain, heat_transfer_gain, cleaning_cost)
        }
        
        # Combine features and targets
        return {**features, **targets}
    
    def _apply_cleaning(self, fouling_state: Dict, cleaning: Dict) -> Dict:
        """Apply cleaning scenario to fouling state."""
        after_fouling = {}
        
        for section_name, fouling in fouling_state.items():
            after_fouling[section_name] = {
                'gas': fouling['gas'].copy(),
                'water': fouling['water'].copy()
            }
            
            # Apply cleaning if this section is included
            if section_name in cleaning['sections_to_clean']:
                segments_to_clean = cleaning['segments_per_section'].get(section_name, [])
                effectiveness = cleaning['cleaning_effectiveness']
                
                for seg_id in segments_to_clean:
                    if seg_id < len(after_fouling[section_name]['gas']):
                        after_fouling[section_name]['gas'][seg_id] *= (1 - effectiveness)
                        after_fouling[section_name]['water'][seg_id] *= (1 - effectiveness)
        
        return after_fouling
    
    def _calculate_cleaning_cost(self, cleaning: Dict) -> float:
        """Calculate cost of cleaning scenario."""
        base_cost_per_segment = 150  # $ per segment
        fixed_cost_per_section = 500  # $ per section (setup, crew, etc.)
        
        total_segments = sum(len(segments) for segments in cleaning['segments_per_section'].values())
        total_sections = len(cleaning['sections_to_clean'])
        
        return total_sections * fixed_cost_per_section + total_segments * base_cost_per_segment
    
    def _calculate_cleaning_time(self, cleaning: Dict) -> float:
        """Calculate time required for cleaning scenario."""
        time_per_segment = 0.25  # hours per segment
        setup_time_per_section = 1.0  # hours per section
        
        total_segments = sum(len(segments) for segments in cleaning['segments_per_section'].values())
        total_sections = len(cleaning['sections_to_clean'])
        
        return total_sections * setup_time_per_section + total_segments * time_per_segment
    
    def _calculate_performance_score(self, efficiency_gain: float, heat_transfer_gain: float, 
                                   cleaning_cost: float) -> float:
        """Calculate overall performance score for cleaning scenario."""
        benefit = efficiency_gain * 1000 + heat_transfer_gain * 500  # Weighted benefits
        cost_factor = cleaning_cost / 1000  # Normalize cost
        return max(0, benefit - cost_factor)
    
    def _extract_features(self, scenario: Dict, combustion_data: Dict, 
                         fouling_state: Dict, cleaning: Dict) -> Dict:
        """Extract comprehensive feature set for ML model."""
        features = {}
        
        # Operating condition features
        features.update({
            'coal_rate_lb_hr': scenario['coal_lb_per_hr'],
            'air_scfh': scenario['air_scfh'],
            'excess_air_ratio': scenario['air_scfh'] / (scenario['coal_lb_per_hr'] * 10),  # Simplified
            'nox_efficiency': scenario['NOx_eff'],
            'air_temp_F': scenario['air_temp_F'],
            'air_humidity_pct': scenario['air_RH_pct'],
            'operating_hours': scenario['operating_hours_since_clean']
        })
        
        # Coal property features
        features.update({
            'coal_carbon_pct': scenario['coal_properties']['carbon'],
            'coal_volatile_matter_pct': scenario['coal_properties']['volatile_matter'],
            'coal_fixed_carbon_pct': scenario['coal_properties']['fixed_carbon'],
            'coal_sulfur_pct': scenario['coal_properties']['sulfur'],
            'coal_ash_pct': scenario['coal_properties']['ash']
        })
        
        # Combustion result features
        features.update({
            'thermal_nox_lb_hr': combustion_data['thermal_nox'],
            'fuel_nox_lb_hr': combustion_data['fuel_nox'],
            'total_nox_lb_hr': combustion_data['total_nox'],
            'excess_o2_pct': combustion_data['excess_o2'],
            'combustion_efficiency': combustion_data['combustion_efficiency'],
            'flame_temp_F': combustion_data['flame_temp'],
            'heat_release_btu_hr': combustion_data['heat_release']
        })
        
        # Fouling state features (statistical summaries per section)
        for section_name, fouling in fouling_state.items():
            prefix = f"fouling_{section_name}_"
            gas_fouling = fouling['gas']
            
            features.update({
                f"{prefix}avg_gas": np.mean(gas_fouling),
                f"{prefix}max_gas": np.max(gas_fouling),
                f"{prefix}std_gas": np.std(gas_fouling),
                f"{prefix}range_gas": np.max(gas_fouling) - np.min(gas_fouling),
                f"{prefix}segments": len(gas_fouling),
                f"{prefix}above_threshold": sum(1 for f in gas_fouling if f > np.mean(gas_fouling) * 1.2)
            })
        
        # Cleaning strategy features
        features.update({
            'clean_all_sections': 1 if cleaning['strategy'] == 'clean_all' else 0,
            'clean_worst_only': 1 if cleaning['strategy'] == 'clean_worst' else 0,
            'selective_cleaning': 1 if cleaning['strategy'] == 'selective_segments' else 0,
            'economizer_priority': 1 if cleaning['strategy'] == 'economizers_first' else 0,
            'no_cleaning': 1 if cleaning['strategy'] == 'no_cleaning' else 0,
            'sections_to_clean_count': len(cleaning['sections_to_clean']),
            'total_segments_to_clean': sum(len(segments) for segments in cleaning['segments_per_section'].values()),
            'cleaning_effectiveness': cleaning['cleaning_effectiveness']
        })
        
        # Seasonal and operational context
        season_encoding = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
        fuel_quality_encoding = {'low': 0, 'medium': 1, 'high': 2}
        
        features.update({
            'season_code': season_encoding[scenario['season']],
            'fuel_quality_code': fuel_quality_encoding[scenario['fuel_quality']],
            'is_winter': 1 if scenario['season'] == 'winter' else 0,
            'is_high_load': 1 if scenario['coal_lb_per_hr'] > 9000 else 0,
            'is_low_nox': 1 if combustion_data['total_nox'] < 50 else 0
        })
        
        return features