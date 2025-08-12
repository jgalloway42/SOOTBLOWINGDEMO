#!/usr/bin/env python3
"""
Fouling and Soot Blowing Models

This module contains fouling calculation utilities and soot blowing simulation
for individual segment control in boiler heat transfer sections.

Classes:
    FoulingCalculator: Fouling factor calculations with position dependencies
    SootBlowingSimulator: Soot blowing simulation utilities
    FoulingCharacteristics: Dataclass for fouling buildup characteristics

Dependencies:
    - numpy: Numerical calculations
    - dataclasses: For structured data
    - typing: Type hints

Author: Enhanced Boiler Modeling System
Version: 5.0 - Individual Segment Fouling Control
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
    """Fouling factor calculation utilities with position and temperature dependencies."""
    
    @staticmethod
    def calculate_fouling_gradient(base_gas_fouling: float, base_water_fouling: float,
                                 segment_position: float, avg_gas_temp: float, 
                                 avg_water_temp: float, custom_gas_fouling: Optional[float] = None,
                                 custom_water_fouling: Optional[float] = None) -> Tuple[float, float]:
        """
        Calculate position-dependent fouling factors with optional custom overrides.
        
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
            # Standard gradient calculation
            temp_factor = max(0.5, (3500 - avg_gas_temp) / 2500)
            position_factor = 1 + 0.8 * segment_position
            gas_fouling = base_gas_fouling * temp_factor * position_factor
        
        if custom_water_fouling is not None:
            water_fouling = custom_water_fouling
        else:
            # Standard gradient calculation
            if avg_water_temp > 500:  # Steam side
                water_fouling = base_water_fouling * (0.5 + 0.3 * segment_position)
            else:  # Liquid water side
                temp_effect = 1 + 0.001 * max(0, avg_water_temp - 200)
                position_effect = 1 + 0.5 * segment_position
                water_fouling = base_water_fouling * temp_effect * position_effect
        
        return gas_fouling, water_fouling


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
