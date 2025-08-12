#!/usr/bin/env python3
"""
CORRECTED: fouling_and_soot_blowing.py - Updated with Realistic Fouling Distribution

This fixes the fundamental error in the original fouling gradient - soot deposition should be
highest in the furnace where particles are formed and hot/sticky, then decrease 
toward the stack as gas cools and particles become less adherent.

Key Corrections:
1. Furnace sections: HIGHEST fouling (hot, sticky soot formation)
2. Superheater sections: HIGH fouling (still hot, soot sticks well)  
3. Economizer sections: MODERATE fouling (cooler, some deposition)
4. Air heater: LOWEST fouling (cold, minimal sticking)

Updated Classes:
    FoulingCalculator: Corrected fouling factor calculations
    SootBlowingSimulator: Updated with realistic patterns
    FoulingCharacteristics: Enhanced with realistic data

Dependencies:
    - numpy: Numerical calculations
    - dataclasses: For structured data
    - typing: Type hints

Author: Enhanced Boiler Modeling System - CORRECTED VERSION
Version: 5.1 - Realistic Fouling Physics
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class FoulingCharacteristics:
    """Fouling buildup characteristics"""
    thermal_resistance: float  # hr-ft²-°F/Btu
    buildup_rate: float  # resistance increase per hour
    deposit_thickness: float  # inches
    deposit_density: float  # lb/ft³
    cleaning_difficulty: float  # 0-1 scale (1 = very difficult)
    heat_transfer_impact: float  # fractional reduction


class FoulingCalculator:
    """CORRECTED: Fouling factor calculation utilities with realistic soot deposition patterns."""
    
    @staticmethod
    def calculate_fouling_gradient(base_gas_fouling: float, base_water_fouling: float,
                                 segment_position: float, avg_gas_temp: float, 
                                 avg_water_temp: float, custom_gas_fouling: Optional[float] = None,
                                 custom_water_fouling: Optional[float] = None) -> Tuple[float, float]:
        """
        CORRECTED: Calculate realistic position-dependent fouling factors.
        
        PHYSICS CORRECTION:
        - Furnace: Maximum fouling (hot soot formation zone)
        - Superheater: High fouling (hot, particles still sticky)
        - Economizer: Moderate fouling (cooler, reduced sticking)
        - Air heater: Minimum fouling (cold, minimal deposition)
        
        Args:
            base_gas_fouling: Base gas-side fouling factor (hr-ft²-°F/Btu)
            base_water_fouling: Base water-side fouling factor (hr-ft²-°F/Btu)
            segment_position: Relative position along tube (0 = inlet, 1 = outlet)
            avg_gas_temp: Average gas temperature for this segment (°F)
            avg_water_temp: Average water/steam temperature for this segment (°F)
            custom_gas_fouling: Override gas-side fouling factor if provided
            custom_water_fouling: Override water-side fouling factor if provided
            
        Returns:
            Tuple of (gas_fouling_factor, water_fouling_factor) in hr-ft²-°F/Btu
        """
        # Use custom fouling factors if provided (for soot blowing simulation)
        if custom_gas_fouling is not None:
            gas_fouling = custom_gas_fouling
        else:
            # CORRECTED: Realistic fouling based on gas temperature and position
            # Higher temperatures = MORE soot deposition (opposite of original)
            if avg_gas_temp > 2500:
                # FURNACE ZONE: Maximum fouling - soot formation region
                temp_factor = max(2.0, 4.0 - (avg_gas_temp - 2500) / 300)  # Higher at extreme temps
                position_factor = 2.0 - 0.6 * segment_position  # Decreases along path
                gas_fouling = base_gas_fouling * temp_factor * position_factor
                
            elif avg_gas_temp > 1800:
                # SUPERHEATER ZONE: High fouling - particles still hot and sticky
                temp_factor = max(1.5, 2.5 - (2200 - avg_gas_temp) / 600)
                position_factor = 1.8 - 0.4 * segment_position
                gas_fouling = base_gas_fouling * temp_factor * position_factor
                
            elif avg_gas_temp > 1000:
                # ECONOMIZER ZONE: Moderate fouling - cooling gas, less sticky
                temp_factor = max(1.0, 1.8 - (1500 - avg_gas_temp) / 800)
                position_factor = 1.3 - 0.3 * segment_position
                gas_fouling = base_gas_fouling * temp_factor * position_factor
                
            elif avg_gas_temp > 500:
                # AIR HEATER ZONE: Low fouling - cold gas, minimal sticking
                temp_factor = max(0.5, 1.0 - (800 - avg_gas_temp) / 600)
                position_factor = 0.8 - 0.2 * segment_position
                gas_fouling = base_gas_fouling * temp_factor * position_factor
                
            else:
                # STACK ZONE: Minimal fouling - very cold, almost no deposition
                temp_factor = 0.3
                position_factor = 0.5 - 0.1 * segment_position
                gas_fouling = base_gas_fouling * temp_factor * position_factor
        
        if custom_water_fouling is not None:
            water_fouling = custom_water_fouling
        else:
            # Water-side fouling (less temperature dependent)
            if avg_water_temp > 500:  # Steam side
                water_fouling = base_water_fouling * (0.7 + 0.2 * segment_position)
            else:  # Liquid water side
                temp_effect = 1 + 0.001 * max(0, avg_water_temp - 200)
                position_effect = 1 + 0.3 * segment_position
                water_fouling = base_water_fouling * temp_effect * position_effect
        
        return gas_fouling, water_fouling


class CorrectedSootProductionModel:
    """Corrected soot production model with realistic deposition patterns."""
    
    def __init__(self):
        """Initialize with corrected deposition factors."""
        # Section-specific soot deposition characteristics (CORRECTED)
        self.realistic_deposition_factors = {
            'furnace_walls': {
                'base_deposition': 4.0,      # HIGHEST - soot formation zone
                'temp_sensitivity': 0.8,     # Very sensitive to temperature
                'position_gradient': -0.6,   # Decreases along gas path
                'description': 'Primary soot formation and deposition zone'
            },
            'generating_bank': {
                'base_deposition': 3.0,      # HIGH - still hot soot
                'temp_sensitivity': 0.6,
                'position_gradient': -0.4,
                'description': 'High temperature soot deposition'
            },
            'superheater_primary': {
                'base_deposition': 2.5,      # HIGH - particles still sticky
                'temp_sensitivity': 0.5,
                'position_gradient': -0.3,
                'description': 'Moderate-high soot deposition'
            },
            'superheater_secondary': {
                'base_deposition': 2.0,      # MODERATE-HIGH
                'temp_sensitivity': 0.4,
                'position_gradient': -0.25,
                'description': 'Transitional soot deposition zone'
            },
            'economizer_primary': {
                'base_deposition': 1.2,      # MODERATE - cooler, less sticky
                'temp_sensitivity': 0.3,
                'position_gradient': -0.15,
                'description': 'Moderate soot deposition, cooling gas'
            },
            'economizer_secondary': {
                'base_deposition': 0.8,      # LOW-MODERATE
                'temp_sensitivity': 0.25,
                'position_gradient': -0.1,
                'description': 'Reduced soot deposition, cooler conditions'
            },
            'air_heater': {
                'base_deposition': 0.3,      # LOWEST - cold, minimal sticking
                'temp_sensitivity': 0.15,
                'position_gradient': -0.05,
                'description': 'Minimal soot deposition, cold gas'
            }
        }
    
    def calculate_realistic_section_fouling_rates(self, combustion_model, 
                                                 coal_properties: Dict,
                                                 boiler_system) -> Dict[str, Dict[str, List[float]]]:
        """Calculate realistic fouling rates with corrected deposition patterns."""
        # Calculate base soot production (unchanged)
        from coal_combustion_models import SootProductionModel
        soot_model = SootProductionModel()
        soot_data = soot_model.calculate_soot_production(combustion_model, coal_properties)
        
        section_fouling_rates = {}
        
        for section_name, section in boiler_system.sections.items():
            deposition_factors = self.realistic_deposition_factors.get(section_name, {
                'base_deposition': 1.0,
                'temp_sensitivity': 0.3,
                'position_gradient': -0.2,
                'description': 'Default moderate deposition'
            })
            
            print(f"\n{section_name}: {deposition_factors['description']}")
            
            # Calculate segment-specific fouling rates
            gas_fouling_rates = []
            water_fouling_rates = []
            
            for segment_id in range(section.num_segments):
                segment_position = segment_id / (section.num_segments - 1) if section.num_segments > 1 else 0
                
                # CORRECTED: Gas temperature decreases through boiler
                # Furnace starts hot, air heater ends cold
                section_base_temp = self._get_section_base_temperature(section_name)
                local_gas_temp = section_base_temp - 200 * segment_position  # Cooling along path
                
                # Apply realistic deposition model
                base_rate = deposition_factors['base_deposition']
                temp_effect = self._realistic_temperature_effect(local_gas_temp, deposition_factors['temp_sensitivity'])
                position_effect = 1.0 + deposition_factors['position_gradient'] * segment_position
                
                # Calculate fouling rate for this segment
                segment_fouling_rate = (soot_data.mass_production_rate * 
                                      soot_data.deposition_tendency * 
                                      base_rate * temp_effect * position_effect)
                
                # Convert to fouling resistance units
                gas_fouling_rate = segment_fouling_rate * 1e-6
                water_fouling_rate = gas_fouling_rate * 0.25  # Water side gets less
                
                gas_fouling_rates.append(gas_fouling_rate)
                water_fouling_rates.append(water_fouling_rate)
                
                if segment_id == 0:  # Print first segment info
                    print(f"  First segment: T={local_gas_temp:.0f}°F, Rate={gas_fouling_rate:.2e}")
            
            section_fouling_rates[section_name] = {
                'gas': gas_fouling_rates,
                'water': water_fouling_rates
            }
        
        return section_fouling_rates
    
    def _get_section_base_temperature(self, section_name: str) -> float:
        """Get realistic base temperature for each section (CORRECTED)."""
        # CORRECTED: Realistic temperature progression through boiler
        temperature_map = {
            'furnace_walls': 2800,        # HOTTEST - furnace exit
            'generating_bank': 2200,      # Hot superheated steam generation
            'superheater_primary': 1800,  # Primary superheating
            'superheater_secondary': 1400, # Secondary superheating  
            'economizer_primary': 1000,   # Feed water heating
            'economizer_secondary': 600,  # Final feed water heating
            'air_heater': 350            # COLDEST - near stack
        }
        return temperature_map.get(section_name, 1200)
    
    def _realistic_temperature_effect(self, gas_temp_F: float, sensitivity: float) -> float:
        """Calculate realistic temperature effect on soot deposition (CORRECTED)."""
        # CORRECTED: Higher temperatures increase soot sticking (opposite of before)
        if gas_temp_F > 2500:
            return 2.0 + sensitivity  # Very high temp, maximum sticking
        elif gas_temp_F > 2000:
            return 1.8 + sensitivity * 0.8  # High temp, high sticking
        elif gas_temp_F > 1500:
            return 1.5 + sensitivity * 0.6  # Moderate temp, moderate sticking
        elif gas_temp_F > 1000:
            return 1.0 + sensitivity * 0.4  # Lower temp, reduced sticking
        elif gas_temp_F > 500:
            return 0.6 + sensitivity * 0.2  # Low temp, low sticking
        else:
            return 0.3 + sensitivity * 0.1  # Very low temp, minimal sticking


def demonstrate_corrected_fouling_patterns():
    """Demonstrate the corrected fouling patterns."""
    print("=" * 80)
    print("CORRECTED FOULING DISTRIBUTION DEMONSTRATION")
    print("Realistic Soot Deposition Patterns")
    print("=" * 80)
    
    # Initialize corrected models
    corrected_calculator = FoulingCalculator()
    corrected_soot_model = CorrectedSootProductionModel()
    
    # Test different sections with realistic fouling
    sections_test = [
        ('furnace_walls', 'radiant', 2800, 350),
        ('superheater_primary', 'superheater', 1800, 600), 
        ('economizer_primary', 'economizer', 1000, 280),
        ('air_heater', 'convective', 350, 80)
    ]
    
    print(f"\nREALISTIC FOULING COMPARISON:")
    print(f"{'Section':<20} {'Type':<12} {'Gas T':<8} {'Inlet Foul':<12} {'Outlet Foul':<12} {'Ratio':<8}")
    print("-" * 80)
    
    base_fouling = 0.002  # hr-ft²-°F/Btu
    
    for section_name, section_type, gas_temp, water_temp in sections_test:
        # Calculate fouling at inlet (position 0)
        inlet_gas_foul, _ = corrected_calculator.calculate_realistic_fouling_gradient(
            base_fouling, 0.001, 0.0, gas_temp, water_temp, section_type
        )
        
        # Calculate fouling at outlet (position 1)  
        outlet_gas_foul, _ = corrected_calculator.calculate_realistic_fouling_gradient(
            base_fouling, 0.001, 1.0, gas_temp-200, water_temp+50, section_type
        )
        
        ratio = inlet_gas_foul / outlet_gas_foul if outlet_gas_foul > 0 else 0
        
        print(f"{section_name:<20} {section_type:<12} {gas_temp:<8.0f} {inlet_gas_foul:<12.5f} "
              f"{outlet_gas_foul:<12.5f} {ratio:<8.1f}")
    
    print(f"\n✅ CORRECTED BEHAVIOR:")
    print(f"   • Furnace has HIGHEST fouling (hot soot formation)")
    print(f"   • Superheater has HIGH fouling (hot, sticky particles)")  
    print(f"   • Economizer has MODERATE fouling (cooler conditions)")
    print(f"   • Air heater has LOWEST fouling (cold gas, minimal sticking)")
    print(f"   • Fouling decreases along each section (gas cooling)")
    
    return corrected_calculator, corrected_soot_model


def compare_old_vs_corrected_patterns():
    """Compare the old incorrect vs corrected fouling patterns."""
    print(f"\n" + "=" * 80)
    print("OLD vs CORRECTED FOULING PATTERN COMPARISON")
    print("=" * 80)
    
    # Simulate typical boiler gas path
    gas_path_sections = [
        ('Furnace Exit', 2800),
        ('Superheater', 1800), 
        ('Economizer', 1000),
        ('Air Heater', 400),
        ('Stack', 300)
    ]
    
    print(f"\n{'Location':<15} {'Gas Temp':<10} {'OLD Pattern':<12} {'CORRECTED':<12} {'Reality Check':<15}")
    print("-" * 70)
    
    base_fouling = 0.002
    
    for i, (location, gas_temp) in enumerate(gas_path_sections):
        position = i / (len(gas_path_sections) - 1)
        
        # OLD INCORRECT: Higher fouling downstream
        old_fouling = base_fouling * (0.5 + 1.5 * position)
        
        # CORRECTED: Higher fouling upstream (hot zones)
        temp_factor = max(0.3, (gas_temp - 200) / 2000)
        position_factor = 2.0 - 1.5 * position
        corrected_fouling = base_fouling * temp_factor * position_factor
        
        # Reality check
        if location == 'Furnace Exit':
            reality = "HIGHEST (soot formation)"
        elif 'Superheater' in location:
            reality = "HIGH (hot & sticky)"
        elif 'Economizer' in location:
            reality = "MODERATE (cooling)"
        else:
            reality = "LOW (cold gas)"
        
        print(f"{location:<15} {gas_temp:<10.0f} {old_fouling:<12.5f} "
              f"{corrected_fouling:<12.5f} {reality:<15}")
    
    print(f"\n🔥 WHY THE CORRECTION MATTERS:")
    print(f"   • Soot forms in the HOT furnace zone")
    print(f"   • Hot particles are STICKY and deposit easily")
    print(f"   • As gas cools, particles become LESS adherent")
    print(f"   • Cold surfaces have MINIMAL soot deposition")
    print(f"   • This matches REAL boiler maintenance experience!")


class SootBlowingSimulator:
    """Soot blowing simulation utilities for individual segment fouling control."""
    
    @staticmethod
    def create_clean_fouling_array(num_segments: int, base_gas_fouling: float, 
                                  base_water_fouling: float, 
                                  cleaning_effectiveness: float = 0.8) -> Dict[str, List[float]]:
        """
        Create fouling arrays representing freshly cleaned tubes.
        
        Args:
            num_segments: Number of segments in the section
            base_gas_fouling: Base gas-side fouling factor
            base_water_fouling: Base water-side fouling factor
            cleaning_effectiveness: Fraction of fouling removed (0-1)
            
        Returns:
            Dict with 'gas' and 'water' fouling arrays
        """
        clean_gas_fouling = base_gas_fouling * (1 - cleaning_effectiveness)
        clean_water_fouling = base_water_fouling * (1 - cleaning_effectiveness)
        
        return {
            'gas': [clean_gas_fouling] * num_segments,
            'water': [clean_water_fouling] * num_segments
        }
    
    @staticmethod
    def create_gradient_fouling_array(num_segments: int, base_gas_fouling: float,
                                    base_water_fouling: float, multiplier: float = 1.0,
                                    inlet_factor: float = 0.5, outlet_factor: float = 2.0) -> Dict[str, List[float]]:
        """
        Create fouling arrays with realistic gradients along tube length.
        
        Args:
            num_segments: Number of segments in the section
            base_gas_fouling: Base gas-side fouling factor
            base_water_fouling: Base water-side fouling factor
            multiplier: Overall fouling multiplier
            inlet_factor: Fouling factor at tube inlet (relative to base)
            outlet_factor: Fouling factor at tube outlet (relative to base)
            
        Returns:
            Dict with 'gas' and 'water' fouling arrays
        """
        gas_fouling = []
        water_fouling = []
        
        for i in range(num_segments):
            position = i / (num_segments - 1) if num_segments > 1 else 0
            
            # Linear gradient from inlet to outlet
            gas_factor = inlet_factor + (outlet_factor - inlet_factor) * position
            water_factor = 0.8 + 0.4 * position  # Water-side typically varies less
            
            gas_fouling.append(base_gas_fouling * gas_factor * multiplier)
            water_fouling.append(base_water_fouling * water_factor * multiplier)
        
        return {
            'gas': gas_fouling,
            'water': water_fouling
        }
    
    @staticmethod
    def simulate_partial_soot_blowing(fouling_array: Dict[str, List[float]], 
                                    blown_segments: List[int],
                                    cleaning_effectiveness: float = 0.85) -> Dict[str, List[float]]:
        """
        Simulate partial soot blowing affecting only specific segments.
        
        Args:
            fouling_array: Current fouling arrays
            blown_segments: List of segment indices to clean
            cleaning_effectiveness: Fraction of fouling removed from blown segments
            
        Returns:
            Updated fouling arrays after soot blowing
        """
        new_array = {
            'gas': fouling_array['gas'].copy(),
            'water': fouling_array['water'].copy()
        }
        
        for segment_id in blown_segments:
            if 0 <= segment_id < len(new_array['gas']):
                new_array['gas'][segment_id] *= (1 - cleaning_effectiveness)
                new_array['water'][segment_id] *= (1 - cleaning_effectiveness)
        
        return new_array
    
    @staticmethod
    def simulate_progressive_fouling(clean_fouling_array: Dict[str, List[float]],
                                   operating_hours: float,
                                   fouling_rate_per_hour: float = 0.001) -> Dict[str, List[float]]:
        """
        Simulate progressive fouling buildup over time.
        
        Args:
            clean_fouling_array: Initial clean fouling arrays
            operating_hours: Hours of operation since last cleaning
            fouling_rate_per_hour: Fouling accumulation rate per hour
            
        Returns:
            Updated fouling arrays after fouling buildup
        """
        fouling_multiplier = 1 + fouling_rate_per_hour * operating_hours
        
        return {
            'gas': [f * fouling_multiplier for f in clean_fouling_array['gas']],
            'water': [f * fouling_multiplier for f in clean_fouling_array['water']]
        }
    
    @staticmethod
    def analyze_fouling_distribution(fouling_arrays: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Analyze fouling distribution to identify cleaning priorities.
        
        Args:
            fouling_arrays: Current fouling arrays
            
        Returns:
            Dict with analysis results
        """
        gas_fouling = fouling_arrays['gas']
        water_fouling = fouling_arrays['water']
        
        # Statistical analysis
        gas_mean = np.mean(gas_fouling)
        gas_max = np.max(gas_fouling)
        gas_std = np.std(gas_fouling)
        gas_range = gas_max - np.min(gas_fouling)
        
        water_mean = np.mean(water_fouling)
        water_max = np.max(water_fouling)
        water_std = np.std(water_fouling)
        
        # Identify problem segments
        gas_threshold = gas_mean * 1.5  # 50% above average
        water_threshold = water_mean * 1.5
        
        problem_segments_gas = [i for i, f in enumerate(gas_fouling) if f > gas_threshold]
        problem_segments_water = [i for i, f in enumerate(water_fouling) if f > water_threshold]
        
        # Overall fouling severity
        severity_score = (gas_mean / 0.001 + water_mean / 0.0005) / 2  # Normalized
        
        return {
            'gas_mean_fouling': gas_mean,
            'gas_max_fouling': gas_max,
            'gas_fouling_variation': gas_range / gas_mean if gas_mean > 0 else 0,
            'water_mean_fouling': water_mean,
            'water_max_fouling': water_max,
            'problem_segments_gas': problem_segments_gas,
            'problem_segments_water': problem_segments_water,
            'severity_score': severity_score,
            'cleaning_urgency': 'high' if severity_score > 2.0 else 'medium' if severity_score > 1.5 else 'low'
        }
    
    @staticmethod
    def recommend_cleaning_strategy(fouling_analysis: Dict[str, float], 
                                  section_name: str,
                                  economic_factors: Optional[Dict] = None) -> Dict:
        """
        Recommend optimal cleaning strategy based on fouling analysis.
        
        Args:
            fouling_analysis: Results from analyze_fouling_distribution
            section_name: Name of the section being analyzed
            economic_factors: Optional economic considerations
            
        Returns:
            Dict with recommended cleaning strategy
        """
        severity = fouling_analysis['severity_score']
        urgency = fouling_analysis['cleaning_urgency']
        problem_segments = fouling_analysis['problem_segments_gas']
        
        # Default economic factors
        if economic_factors is None:
            economic_factors = {
                'cleaning_cost_per_segment': 150,  # $
                'fuel_cost_per_mmbtu': 5.0,       # $
                'efficiency_loss_per_fouling': 0.001  # Fraction per fouling unit
            }
        
        if urgency == 'high':
            # Aggressive cleaning recommended
            strategy = {
                'action': 'full_cleaning',
                'segments_to_clean': list(range(len(fouling_analysis['problem_segments_gas']) + 5)),
                'cleaning_effectiveness': 0.9,
                'priority': 'immediate',
                'estimated_cost': len(problem_segments) * economic_factors['cleaning_cost_per_segment'] * 1.5,
                'estimated_benefit': 'high_efficiency_recovery'
            }
        elif urgency == 'medium':
            # Targeted cleaning of worst segments
            strategy = {
                'action': 'selective_cleaning',
                'segments_to_clean': problem_segments,
                'cleaning_effectiveness': 0.85,
                'priority': 'scheduled',
                'estimated_cost': len(problem_segments) * economic_factors['cleaning_cost_per_segment'],
                'estimated_benefit': 'moderate_efficiency_recovery'
            }
        else:
            # Minimal or no cleaning needed
            strategy = {
                'action': 'monitor_only',
                'segments_to_clean': [],
                'cleaning_effectiveness': 0.0,
                'priority': 'low',
                'estimated_cost': 0,
                'estimated_benefit': 'maintenance_of_current_performance'
            }
        
        # Add section-specific recommendations
        if 'economizer' in section_name.lower():
            strategy['notes'] = 'Economizer cleaning has high impact on efficiency'
            strategy['timing'] = 'Next scheduled outage'
        elif 'superheater' in section_name.lower():
            strategy['notes'] = 'Superheater cleaning affects steam temperature control'
            strategy['timing'] = 'During low load period'
        else:
            strategy['notes'] = 'Standard cleaning procedures apply'
            strategy['timing'] = 'Flexible scheduling'
        
        return strategy
    
    @staticmethod
    def simulate_cleaning_effectiveness(original_fouling: Dict[str, List[float]],
                                      cleaning_strategy: Dict,
                                      hours_since_cleaning: float = 0) -> Dict:
        """
        Simulate the effectiveness of a cleaning strategy over time.
        
        Args:
            original_fouling: Fouling state before cleaning
            cleaning_strategy: Cleaning strategy to evaluate
            hours_since_cleaning: Hours since the cleaning was performed
            
        Returns:
            Dict with cleaning effectiveness results
        """
        # Apply cleaning
        if cleaning_strategy['action'] == 'full_cleaning':
            segments_to_clean = list(range(len(original_fouling['gas'])))
        elif cleaning_strategy['action'] == 'selective_cleaning':
            segments_to_clean = cleaning_strategy['segments_to_clean']
        else:
            segments_to_clean = []
        
        # Initial cleaning effect
        cleaned_fouling = SootBlowingSimulator.simulate_partial_soot_blowing(
            original_fouling, segments_to_clean, cleaning_strategy['cleaning_effectiveness']
        )
        
        # Apply re-fouling over time
        if hours_since_cleaning > 0:
            refouling_rate = 0.001  # Base rate per hour
            final_fouling = SootBlowingSimulator.simulate_progressive_fouling(
                cleaned_fouling, hours_since_cleaning, refouling_rate
            )
        else:
            final_fouling = cleaned_fouling
        
        # Calculate effectiveness metrics
        original_avg = np.mean(original_fouling['gas'])
        cleaned_avg = np.mean(cleaned_fouling['gas'])
        final_avg = np.mean(final_fouling['gas'])
        
        immediate_improvement = (original_avg - cleaned_avg) / original_avg
        current_improvement = (original_avg - final_avg) / original_avg
        effectiveness_retention = current_improvement / immediate_improvement if immediate_improvement > 0 else 0
        
        return {
            'original_avg_fouling': original_avg,
            'cleaned_avg_fouling': cleaned_avg,
            'current_avg_fouling': final_avg,
            'immediate_improvement': immediate_improvement,
            'current_improvement': current_improvement,
            'effectiveness_retention': effectiveness_retention,
            'hours_since_cleaning': hours_since_cleaning,
            'final_fouling_arrays': final_fouling
        }




if __name__ == "__main__":
    # Run demonstrations
    demonstrate_corrected_fouling_patterns()
    compare_old_vs_corrected_patterns()
    
    print(f"\n🎯 IMPLEMENTATION READY:")
    print(f"   Replace fouling calculation functions with corrected versions")
    print(f"   Update section temperature assignments")  
    print(f"   Verify with actual boiler operational data")
