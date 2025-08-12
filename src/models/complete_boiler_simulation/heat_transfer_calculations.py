#!/usr/bin/env python3
"""
CORRECTED: Heat Transfer Calculations and Tube Section Models

This module contains heat transfer coefficient calculations and the enhanced
boiler tube section model with CORRECTED fouling distribution patterns.

MAJOR CORRECTION: Fouling now follows realistic physics:
- Highest fouling in hot furnace zones (soot formation)
- Decreasing fouling toward stack (cooling gas, less adhesion)

Classes:
    SegmentResult: Dataclass for individual segment results
    HeatTransferCalculator: Heat transfer coefficient calculations
    EnhancedBoilerTubeSection: Tube section with corrected fouling control

Dependencies:
    - numpy: Numerical calculations
    - math: Mathematical functions
    - dataclasses: For structured data
    - typing: Type hints

Author: Enhanced Boiler Modeling System
Version: 5.1 - CORRECTED Fouling Physics
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Union

from thermodynamic_properties import PropertyCalculator, SteamProperties, GasProperties
from fouling_and_soot_blowing import FoulingCalculator, SootBlowingSimulator

# Module constants
DEFAULT_TUBE_THERMAL_CONDUCTIVITY = 26.0  # Btu/hr-ft-°F (carbon steel)
MIN_SEGMENTS = 5
MAX_SEGMENTS = 10
SEGMENT_AREA_THRESHOLD = 1000  # ft² per segment


@dataclass
class SegmentResult:
    """Results dataclass for individual tube segment heat transfer analysis."""
    segment_id: int
    position: float  # 0 to 1 along tube length
    gas_temp_in: float  # °F
    gas_temp_out: float  # °F
    water_temp_in: float  # °F
    water_temp_out: float  # °F
    heat_transfer_rate: float  # Btu/hr
    overall_U: float  # Btu/hr-ft²-°F
    gas_htc: float  # Btu/hr-ft²-°F
    water_htc: float  # Btu/hr-ft²-°F
    fouling_gas: float  # hr-ft²-°F/Btu
    fouling_water: float  # hr-ft²-°F/Btu
    LMTD: float  # °F
    area: float  # ft²


class HeatTransferCalculator:
    """Heat transfer coefficient calculation utilities with correlations for different geometries."""
    
    @staticmethod
    def calculate_reynolds_number(flow_rate: float, density: float, 
                                hydraulic_diameter: float, viscosity: float,
                                flow_area: float) -> float:
        """Calculate Reynolds number for flow conditions."""
        if viscosity <= 0 or flow_area <= 0:
            return 0
        
        velocity = flow_rate / (density * flow_area)
        return density * velocity * hydraulic_diameter / viscosity
    
    @staticmethod
    def calculate_nusselt_number(reynolds: float, prandtl: float, geometry: str,
                               section_type: str, phase: str = 'liquid') -> float:
        """Calculate Nusselt number based on flow conditions and geometry."""
        if geometry == 'tube_side':
            if reynolds > 2300:  # Turbulent flow
                if phase == 'superheated_steam':
                    # Modified Dittus-Boelter for superheated steam
                    return 0.021 * (reynolds ** 0.8) * (prandtl ** 0.4)
                else:
                    # Standard Dittus-Boelter correlation
                    return 0.023 * (reynolds ** 0.8) * (prandtl ** 0.4)
            else:  # Laminar flow
                return 4.36  # Fully developed laminar flow in circular tube
                
        else:  # Shell side cross-flow
            if section_type == 'radiant':
                # Enhanced heat transfer in radiant sections
                return 0.4 * (reynolds ** 0.6) * (prandtl ** 0.36)
            else:
                # Standard cross-flow correlations for tube bundles
                if reynolds > 10000:
                    return 0.27 * (reynolds ** 0.63) * (prandtl ** 0.36)
                elif reynolds > 1000:
                    return 0.35 * (reynolds ** 0.60) * (prandtl ** 0.36)
                else:
                    return 2.0  # Minimum reasonable Nusselt number
    
    @staticmethod
    def calculate_heat_transfer_coefficient(flow_rate: float, hydraulic_diameter: float,
                                          properties: Union[SteamProperties, GasProperties], 
                                          geometry: str, section_type: str,
                                          tube_count: int, tube_id: float) -> float:
        """Calculate convective heat transfer coefficient."""
        if flow_rate <= 0:
            return 1.0  # Minimum value
        
        # Calculate flow area
        if geometry == 'tube_side':
            flow_area = tube_count * math.pi * (tube_id ** 2) / 4.0
        else:  # Shell side
            flow_area = tube_count * 0.05  # Approximate shell-side flow area
        
        # Calculate Reynolds number
        reynolds = HeatTransferCalculator.calculate_reynolds_number(
            flow_rate, properties.density, hydraulic_diameter, 
            properties.viscosity, flow_area
        )
        
        # Get phase information
        phase = getattr(properties, 'phase', 'gas')
        
        # Calculate Nusselt number
        nusselt = HeatTransferCalculator.calculate_nusselt_number(
            reynolds, properties.prandtl, geometry, section_type, phase
        )
        
        # Calculate heat transfer coefficient
        h = nusselt * properties.thermal_conductivity / hydraulic_diameter
        return max(h, 1.0)  # Ensure minimum reasonable value


class EnhancedBoilerTubeSection:
    """Enhanced tube section model with CORRECTED fouling distribution patterns."""
    
    def __init__(self, name: str, tube_od: float, tube_id: float, tube_length: float, 
                 tube_count: int, base_fouling_gas: float, base_fouling_water: float,
                 section_type: str = 'convective'):
        """Initialize enhanced boiler tube section with corrected fouling."""
        if tube_od <= tube_id:
            raise ValueError(f"Tube OD ({tube_od}) must be greater than ID ({tube_id})")
        if tube_length <= 0 or tube_count <= 0:
            raise ValueError("Tube length and count must be positive")
        
        self.name = name
        self.tube_od = tube_od
        self.tube_id = tube_id
        self.tube_length = tube_length
        self.tube_count = tube_count
        self.base_fouling_gas = base_fouling_gas
        self.base_fouling_water = base_fouling_water
        self.section_type = section_type
        
        # Calculate geometry parameters
        self.area = math.pi * tube_od * tube_length * tube_count
        self.num_segments = max(MIN_SEGMENTS, min(MAX_SEGMENTS, int(self.area / SEGMENT_AREA_THRESHOLD)))
        self.segment_length = tube_length / self.num_segments
        self.tube_wall_thickness = (tube_od - tube_id) / 2.0
        
        # Initialize property calculator
        self.property_calc = PropertyCalculator()
        
        # Custom fouling arrays for soot blowing simulation
        self.custom_fouling_arrays: Optional[Dict[str, List[float]]] = None
        
        # Results storage
        self.results: List[SegmentResult] = []
    
    def get_current_fouling_arrays(self) -> Dict[str, List[float]]:
        """Get current fouling arrays using CORRECTED gradient calculation."""
        if self.custom_fouling_arrays is not None:
            return self.custom_fouling_arrays.copy()
        
        # Calculate CORRECTED gradients
        gas_fouling = []
        water_fouling = []
        
        for i in range(self.num_segments):
            segment_position = i / (self.num_segments - 1) if self.num_segments > 1 else 0
            
            # CORRECTED: Estimate realistic temperatures based on section type
            avg_gas_temp, avg_water_temp = self._estimate_realistic_temperatures(segment_position)
            
            # Use CORRECTED fouling calculation
            gas_foul, water_foul = FoulingCalculator.calculate_fouling_gradient(
                self.base_fouling_gas, self.base_fouling_water,
                segment_position, avg_gas_temp, avg_water_temp
            )
            
            gas_fouling.append(gas_foul)
            water_fouling.append(water_foul)
        
        return {'gas': gas_fouling, 'water': water_fouling}
    
    def _estimate_realistic_temperatures(self, segment_position: float) -> tuple:
        """CORRECTED: Estimate realistic temperatures based on section type and position."""
        # CORRECTED: Realistic temperature mapping by section
        section_temp_map = {
            'radiant': {'base_gas': 2800, 'gas_drop': 400, 'base_water': 350, 'water_rise': 50},
            'furnace': {'base_gas': 2800, 'gas_drop': 400, 'base_water': 350, 'water_rise': 50},
            'generating': {'base_gas': 2200, 'gas_drop': 300, 'base_water': 400, 'water_rise': 100},
            'superheater': {'base_gas': 1800, 'gas_drop': 200, 'base_water': 550, 'water_rise': 80},
            'economizer': {'base_gas': 1000, 'gas_drop': 200, 'base_water': 220, 'water_rise': 60},
            'convective': {'base_gas': 800, 'gas_drop': 150, 'base_water': 200, 'water_rise': 40}
        }
        
        # Get temperature profile for this section type
        temp_profile = section_temp_map.get(self.section_type, section_temp_map['convective'])
        
        # Calculate temperatures at this position
        avg_gas_temp = temp_profile['base_gas'] - temp_profile['gas_drop'] * segment_position
        avg_water_temp = temp_profile['base_water'] + temp_profile['water_rise'] * segment_position
        
        return avg_gas_temp, avg_water_temp
    
    def solve_segment(self, segment_id: int, gas_temp_in: float, water_temp_in: float,
                     gas_flow: float, water_flow: float, steam_pressure: float) -> SegmentResult:
        """Solve heat transfer analysis for individual tube segment with CORRECTED fouling."""
        if segment_id < 0 or segment_id >= self.num_segments:
            raise ValueError(f"Invalid segment_id {segment_id}")
        
        segment_position = segment_id / (self.num_segments - 1) if self.num_segments > 1 else 0
        
        # Estimate temperature changes based on section type
        temp_drops = self._calculate_temperature_drops(segment_position)
        gas_temp_out = gas_temp_in - temp_drops['gas'] / self.num_segments
        water_temp_out = water_temp_in + temp_drops['water'] / self.num_segments
        
        # Average temperatures for property evaluation
        avg_gas_temp = (gas_temp_in + gas_temp_out) / 2
        avg_water_temp = (water_temp_in + water_temp_out) / 2
        
        # Get fluid properties using thermo library
        gas_props = self.property_calc.get_flue_gas_properties_safe(avg_gas_temp)
        water_props = self.property_calc.get_steam_properties_safe(avg_water_temp, steam_pressure)
        
        # Calculate CORRECTED fouling factors
        if self.custom_fouling_arrays is not None:
            # Use custom fouling arrays
            gas_fouling = self.custom_fouling_arrays['gas'][segment_id]
            water_fouling = self.custom_fouling_arrays['water'][segment_id]
        else:
            # Use CORRECTED gradient calculation
            gas_fouling, water_fouling = FoulingCalculator.calculate_fouling_gradient(
                self.base_fouling_gas, self.base_fouling_water, 
                segment_position, avg_gas_temp, avg_water_temp
            )
        
        # Calculate heat transfer coefficients
        h_gas = HeatTransferCalculator.calculate_heat_transfer_coefficient(
            gas_flow, self.tube_od, gas_props, 'shell_side', 
            self.section_type, self.tube_count, self.tube_id
        )
        h_water = HeatTransferCalculator.calculate_heat_transfer_coefficient(
            water_flow, self.tube_id, water_props, 'tube_side', 
            self.section_type, self.tube_count, self.tube_id
        )
        
        # Calculate overall heat transfer coefficient
        segment_area_out = math.pi * self.tube_od * self.segment_length * self.tube_count
        segment_area_in = math.pi * self.tube_id * self.segment_length * self.tube_count
        
        # Overall thermal resistance
        overall_U = self._calculate_overall_U(
            h_gas, h_water, gas_fouling, water_fouling, 
            segment_area_out, segment_area_in
        )
        
        # Calculate LMTD and heat transfer rate
        LMTD = self._calculate_LMTD(gas_temp_in, gas_temp_out, water_temp_in, water_temp_out)
        Q = overall_U * segment_area_out * LMTD
        
        # Refine outlet temperatures using energy balance
        gas_temp_out_refined = self._refine_gas_temperature(gas_temp_in, Q, gas_flow, gas_props.cp)
        water_temp_out_refined = self._refine_water_temperature(water_temp_in, Q, water_flow, water_props.cp)
        
        return SegmentResult(
            segment_id=segment_id,
            position=segment_position,
            gas_temp_in=gas_temp_in,
            gas_temp_out=gas_temp_out_refined,
            water_temp_in=water_temp_in,
            water_temp_out=water_temp_out_refined,
            heat_transfer_rate=Q,
            overall_U=overall_U,
            gas_htc=h_gas,
            water_htc=h_water,
            fouling_gas=gas_fouling,
            fouling_water=water_fouling,
            LMTD=LMTD,
            area=segment_area_out
        )
    
    def _calculate_temperature_drops(self, segment_position: float) -> Dict[str, float]:
        """Calculate realistic temperature drops for different section types."""
        if self.section_type == 'radiant' or 'furnace' in self.section_type:
            return {
                'gas': 300 + 200 * (1 - segment_position),  # Higher drops in hot sections
                'water': 40 + 30 * segment_position
            }
        elif 'superheater' in self.section_type:
            return {
                'gas': 200 - 50 * segment_position,
                'water': 60 + 40 * segment_position
            }
        elif 'economizer' in self.section_type:
            return {
                'gas': 150 + 50 * (1 - segment_position),
                'water': 45 + 25 * segment_position
            }
        else:  # convective/air heater
            return {
                'gas': 100 + 30 * segment_position,
                'water': 30 + 20 * segment_position
            }
    
    def _calculate_overall_U(self, h_gas: float, h_water: float, 
                           gas_fouling: float, water_fouling: float,
                           area_out: float, area_in: float) -> float:
        """Calculate overall heat transfer coefficient."""
        # Log mean area for conduction
        if area_out > area_in:
            A_lm = (area_out - area_in) / math.log(area_out / area_in)
        else:
            A_lm = area_out
        
        # Total thermal resistance
        R_total = (1.0 / h_gas + 
                  gas_fouling + 
                  self.tube_wall_thickness / (DEFAULT_TUBE_THERMAL_CONDUCTIVITY * A_lm / area_out) +
                  water_fouling * (area_out / area_in) + 
                  (1.0 / h_water) * (area_out / area_in))
        
        return 1.0 / R_total
    
    def _calculate_LMTD(self, gas_in: float, gas_out: float, 
                       water_in: float, water_out: float) -> float:
        """Calculate log mean temperature difference."""
        delta_T1 = gas_in - water_out
        delta_T2 = gas_out - water_in
        
        if abs(delta_T1 - delta_T2) < 1.0:
            return (delta_T1 + delta_T2) / 2.0
        else:
            try:
                return (delta_T1 - delta_T2) / math.log(delta_T1 / delta_T2)
            except (ValueError, ZeroDivisionError):
                return (delta_T1 + delta_T2) / 2.0
    
    def _refine_gas_temperature(self, gas_temp_in: float, Q: float, 
                               gas_flow: float, gas_cp: float) -> float:
        """Refine gas outlet temperature using energy balance."""
        if gas_flow > 0 and gas_cp > 0:
            return gas_temp_in - Q / (gas_flow * gas_cp)
        return gas_temp_in
    
    def _refine_water_temperature(self, water_temp_in: float, Q: float,
                                 water_flow: float, water_cp: float) -> float:
        """Refine water outlet temperature using energy balance."""
        if water_flow > 0 and water_cp > 0:
            return water_temp_in + Q / (water_flow * water_cp)
        return water_temp_in
    
    def solve_section(self, gas_temp_in: float, water_temp_in: float, 
                     gas_flow: float, water_flow: float, 
                     steam_pressure: float) -> List[SegmentResult]:
        """Solve heat transfer for entire section with all segments."""
        if gas_flow < 0 or water_flow < 0:
            raise ValueError("Flow rates cannot be negative")
        if steam_pressure < 1:
            raise ValueError("Steam pressure must be positive")
        
        self.results = []
        current_gas_temp = gas_temp_in
        current_water_temp = water_temp_in
        
        for i in range(self.num_segments):
            segment_result = self.solve_segment(
                i, current_gas_temp, current_water_temp, 
                gas_flow, water_flow, steam_pressure
            )
            
            self.results.append(segment_result)
            current_gas_temp = segment_result.gas_temp_out
            current_water_temp = segment_result.water_temp_out
        
        return self.results
    
    def get_section_summary(self) -> Dict[str, Union[str, float, int]]:
        """Get overall section performance summary."""
        if not self.results:
            return {}
        
        total_Q = sum(r.heat_transfer_rate for r in self.results)
        total_area = sum(r.area for r in self.results)
        avg_U = sum(r.overall_U * r.area for r in self.results) / total_area if total_area > 0 else 0
        
        return {
            'section_name': self.name,
            'total_heat_transfer': total_Q,
            'average_overall_U': avg_U,
            'total_area': total_area,
            'gas_temp_in': self.results[0].gas_temp_in,
            'gas_temp_out': self.results[-1].gas_temp_out,
            'water_temp_in': self.results[0].water_temp_in,
            'water_temp_out': self.results[-1].water_temp_out,
            'num_segments': len(self.results),
            'max_gas_fouling': max(r.fouling_gas for r in self.results),
            'max_water_fouling': max(r.fouling_water for r in self.results)
        }
    
    # Inherit other methods from original implementation for soot blowing
    def set_custom_fouling_arrays(self, gas_fouling_array: List[float], 
                                 water_fouling_array: List[float]):
        """Set custom fouling factor arrays for individual segment control."""
        if len(gas_fouling_array) != self.num_segments:
            raise ValueError(f"Gas fouling array length ({len(gas_fouling_array)}) must match segments ({self.num_segments})")
        if len(water_fouling_array) != self.num_segments:
            raise ValueError(f"Water fouling array length ({len(water_fouling_array)}) must match segments ({self.num_segments})")
        
        self.custom_fouling_arrays = {
            'gas': gas_fouling_array.copy(),
            'water': water_fouling_array.copy()
        }
        
        print(f"✓ Custom fouling arrays set for {self.name}: {self.num_segments} segments")
    
    def clear_custom_fouling_arrays(self):
        """Clear custom fouling arrays and return to CORRECTED gradient calculation."""
        self.custom_fouling_arrays = None
        print(f"✓ Custom fouling arrays cleared for {self.name}, using CORRECTED gradients")
    
    def apply_soot_blowing(self, blown_segments: List[int], 
                          cleaning_effectiveness: float = 0.85):
        """Apply soot blowing to specific segments."""
        if self.custom_fouling_arrays is None:
            # Initialize with current CORRECTED gradients
            self.custom_fouling_arrays = self.get_current_fouling_arrays()
        
        # Apply cleaning to specified segments
        self.custom_fouling_arrays = SootBlowingSimulator.simulate_partial_soot_blowing(
            self.custom_fouling_arrays, blown_segments, cleaning_effectiveness
        )
        
        print(f"✓ Soot blowing applied to {self.name}, segments: {blown_segments}")
        print(f"  Cleaning effectiveness: {cleaning_effectiveness:.1%}")
    
    def simulate_fouling_buildup(self, operating_hours: float, 
                               fouling_rate_per_hour: float = 0.001):
        """Simulate progressive fouling buildup over time."""
        if self.custom_fouling_arrays is None:
            # Start with clean baseline using CORRECTED gradients
            clean_arrays = SootBlowingSimulator.create_clean_fouling_array(
                self.num_segments, self.base_fouling_gas, self.base_fouling_water, 0.0
            )
            self.custom_fouling_arrays = clean_arrays
        
        # Apply fouling buildup
        self.custom_fouling_arrays = SootBlowingSimulator.simulate_progressive_fouling(
            self.custom_fouling_arrays, operating_hours, fouling_rate_per_hour
        )
        
        print(f"✓ Fouling buildup simulated for {self.name}: {operating_hours} hours")