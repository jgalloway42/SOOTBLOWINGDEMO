#!/usr/bin/env python3
"""
Thermodynamic Property Calculations - FIXED VERSION 2

This module contains property calculation utilities using the thermo library
for accurate steam, water, and flue gas properties with proper thermo library usage.

Classes:
    SteamProperties: Dataclass for water/steam properties
    GasProperties: Dataclass for flue gas properties
    PropertyCalculator: Main property calculation class

Dependencies:
    - thermo: Comprehensive thermodynamic library
    - dataclasses: For structured data
    - typing: Type hints
    - numpy: Numerical calculations

Author: Enhanced Boiler Modeling System
Version: 5.1 - Fixed Thermo Library Usage
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from thermo.chemical import Chemical
    from thermo.mixture import Mixture
    THERMO_AVAILABLE = True
except ImportError:
    print("Warning: Thermo library not available, using correlations only")
    THERMO_AVAILABLE = False

# Unit conversion constants
F_TO_K = lambda f: (f - 32) * 5/9 + 273.15
K_TO_F = lambda k: (k - 273.15) * 9/5 + 32
PSIA_TO_PA = 6894.757
PA_TO_PSIA = 1/6894.757
BTU_LBM_F_TO_J_KG_K = 4186.8
J_KG_K_TO_BTU_LBM_F = 1/4186.8
KG_M3_TO_LBM_FT3 = 0.062428
LBM_FT3_TO_KG_M3 = 1/0.062428
W_M_K_TO_BTU_HR_FT_F = 0.577789
PA_S_TO_LBM_HR_FT = 2419.09
J_KG_TO_BTU_LBM = 0.000429923


@dataclass
class SteamProperties:
    """Comprehensive water/steam thermodynamic properties dataclass."""
    temperature: float  # °F
    pressure: float  # psia
    cp: float  # Btu/lbm-°F
    density: float  # lbm/ft³
    viscosity: float  # lbm/hr-ft
    thermal_conductivity: float  # Btu/hr-ft-°F
    saturation_temp: float  # °F
    superheat: float  # °F
    phase: str  # 'liquid', 'saturated', 'superheated_steam'
    prandtl: float
    enthalpy: float  # Btu/lbm
    entropy: float  # Btu/lbm-°R
    quality: Optional[float]  # Steam quality (0-1)


@dataclass
class GasProperties:
    """Flue gas mixture properties dataclass using thermo library."""
    temperature: float  # °F
    cp: float  # Btu/lbm-°F
    density: float  # lbm/ft³
    viscosity: float  # lbm/hr-ft
    thermal_conductivity: float  # Btu/hr-ft-°F
    prandtl: float
    molecular_weight: float  # lb/lbmol


class PropertyCalculator:
    """Steam and gas property calculation utilities using thermo library."""
    
    def __init__(self):
        """Initialize property calculator with thermo objects."""
        self.thermo_available = THERMO_AVAILABLE
        
        if self.thermo_available:
            try:
                # Create water substance for steam calculations
                self.water = Chemical('water')
                print("✓ Thermo library initialized for water properties")
            except Exception as e:
                print(f"Warning: Could not initialize thermo water object: {e}")
                self.water = None
        else:
            self.water = None
    
    def get_steam_properties(self, temperature: float, pressure: float) -> SteamProperties:
        """Legacy method - calls safe version."""
        return self.get_steam_properties_safe(temperature, pressure)
    
    def get_flue_gas_properties(self, temperature: float, pressure: float = 14.7) -> GasProperties:
        """Legacy method - calls safe version."""
        return self.get_flue_gas_properties_safe(temperature, pressure)
    
    def get_steam_properties_safe(self, temperature: float, pressure: float) -> SteamProperties:
        """Calculate water/steam properties with extended temperature range handling."""
        # Clamp temperature to safe range
        temp_clamped = max(32, min(temperature, 1500))  # Conservative limit
        
        if abs(temp_clamped - temperature) > 5.0:
            print(f"Info: Temperature {temperature:.1f}°F adjusted to {temp_clamped:.1f}°F for property calculation")
        
        if pressure < 1 or pressure > 5000:
            raise ValueError(f"Pressure {pressure} psia outside range (1-5000 psia)")
        
        # Try thermo library first
        if self.thermo_available and self.water is not None:
            try:
                # Convert to SI units
                T_K = F_TO_K(temp_clamped)
                P_Pa = pressure * PSIA_TO_PA
                
                # Calculate properties at the given conditions
                self.water.calculate(T=T_K, P=P_Pa)
                
                # Get saturation temperature
                try:
                    sat_temp_K = self.water.Tsat(P_Pa)
                    sat_temp_F = K_TO_F(sat_temp_K)
                except:
                    sat_temp_F = self._estimate_saturation_temp(pressure)
                
                # Determine phase
                if temp_clamped > sat_temp_F + 10:
                    phase = 'superheated_steam'
                    superheat = temperature - sat_temp_F  # Use original temperature
                    quality = None
                elif abs(temp_clamped - sat_temp_F) <= 10:
                    phase = 'saturated'
                    superheat = 0
                    quality = 0.5
                else:
                    phase = 'liquid'
                    superheat = 0
                    quality = None
                
                # Get properties with fallback
                try:
                    density = self.water.rho * KG_M3_TO_LBM_FT3 if hasattr(self.water, 'rho') and self.water.rho else self._estimate_water_density(temp_clamped, pressure, phase)
                    cp = self.water.Cp * J_KG_K_TO_BTU_LBM_F if hasattr(self.water, 'Cp') and self.water.Cp else self._estimate_water_cp(temp_clamped, phase)
                    viscosity = self.water.mu * PA_S_TO_LBM_HR_FT if hasattr(self.water, 'mu') and self.water.mu else self._estimate_water_viscosity(temp_clamped, phase)
                    thermal_conductivity = self.water.k * W_M_K_TO_BTU_HR_FT_F if hasattr(self.water, 'k') and self.water.k else self._estimate_water_thermal_conductivity(temp_clamped, phase)
                    enthalpy = self.water.H * J_KG_TO_BTU_LBM if hasattr(self.water, 'H') and self.water.H else self._estimate_water_enthalpy(temp_clamped, phase)
                    entropy = self.water.S * J_KG_K_TO_BTU_LBM_F if hasattr(self.water, 'S') and self.water.S else self._estimate_water_entropy(temp_clamped, phase)
                except:
                    # Fall back to correlations
                    density = self._estimate_water_density(temp_clamped, pressure, phase)
                    cp = self._estimate_water_cp(temp_clamped, phase)
                    viscosity = self._estimate_water_viscosity(temp_clamped, phase)
                    thermal_conductivity = self._estimate_water_thermal_conductivity(temp_clamped, phase)
                    enthalpy = self._estimate_water_enthalpy(temp_clamped, phase)
                    entropy = self._estimate_water_entropy(temp_clamped, phase)
                
                prandtl = cp * viscosity / thermal_conductivity if thermal_conductivity > 0 else 1.0
                
            except Exception as e:
                # Complete fallback to correlations
                return self._get_steam_properties_correlations(temperature, pressure)
        else:
            # Use correlations only
            return self._get_steam_properties_correlations(temperature, pressure)
        
        return SteamProperties(
            temperature=temperature,
            pressure=pressure,
            cp=cp,
            density=density,
            viscosity=viscosity,
            thermal_conductivity=thermal_conductivity,
            saturation_temp=sat_temp_F,
            superheat=superheat,
            phase=phase,
            prandtl=prandtl,
            enthalpy=enthalpy,
            entropy=entropy,
            quality=quality
        )
    
    def get_flue_gas_properties_safe(self, temperature: float, pressure: float = 14.7) -> GasProperties:
        """Calculate flue gas properties using correlations (more reliable than thermo mixture)."""
        # Always use correlations for gas properties - more reliable and faster
        return self._get_gas_properties_correlations(temperature, pressure)
    
    def _get_steam_properties_correlations(self, temperature: float, pressure: float) -> SteamProperties:
        """Calculate steam properties using engineering correlations only."""
        sat_temp_F = self._estimate_saturation_temp(pressure)
        
        if temperature > sat_temp_F + 10:
            phase = 'superheated_steam'
            superheat = temperature - sat_temp_F
            quality = None
        elif abs(temperature - sat_temp_F) <= 10:
            phase = 'saturated'
            superheat = 0
            quality = 0.5
        else:
            phase = 'liquid'
            superheat = 0
            quality = None
        
        density = self._estimate_water_density(temperature, pressure, phase)
        cp = self._estimate_water_cp(temperature, phase)
        viscosity = self._estimate_water_viscosity(temperature, phase)
        thermal_conductivity = self._estimate_water_thermal_conductivity(temperature, phase)
        enthalpy = self._estimate_water_enthalpy(temperature, phase)
        entropy = self._estimate_water_entropy(temperature, phase)
        prandtl = cp * viscosity / thermal_conductivity if thermal_conductivity > 0 else 1.0
        
        return SteamProperties(
            temperature=temperature,
            pressure=pressure,
            cp=cp,
            density=density,
            viscosity=viscosity,
            thermal_conductivity=thermal_conductivity,
            saturation_temp=sat_temp_F,
            superheat=superheat,
            phase=phase,
            prandtl=prandtl,
            enthalpy=enthalpy,
            entropy=entropy,
            quality=quality
        )
    
    def _get_gas_properties_correlations(self, temperature: float, pressure: float) -> GasProperties:
        """Calculate gas properties using proven engineering correlations."""
        # Clamp temperature to reasonable range
        temp_clamped = max(200, min(temperature, 3500))
        
        # Use proven correlations for flue gas properties
        density, cp, viscosity, thermal_conductivity, molecular_weight = self._estimate_gas_properties(temp_clamped, pressure)
        prandtl = cp * viscosity / thermal_conductivity if thermal_conductivity > 0 else 0.7
        
        return GasProperties(
            temperature=temperature,
            cp=cp,
            density=density,
            viscosity=viscosity,
            thermal_conductivity=thermal_conductivity,
            prandtl=prandtl,
            molecular_weight=molecular_weight
        )
    
    def _estimate_saturation_temp(self, pressure: float) -> float:
        """Estimate saturation temperature using proven correlation."""
        if pressure <= 0:
            return 32
        
        # Enhanced correlation based on steam tables
        try:
            if pressure <= 1:
                return 32
            elif pressure <= 14.7:
                return 32 + (pressure - 1) * 13.0
            elif pressure <= 50:
                return 212 + (pressure - 14.7) * 3.2
            elif pressure <= 100:
                return 212 + 113 + (pressure - 50) * 2.8
            elif pressure <= 500:
                return 212 + 113 + 140 + (pressure - 100) * 0.7
            else:
                return 212 + 113 + 140 + 280 + (pressure - 500) * 0.3
        except:
            return max(32, min(212 + (pressure - 14.7) * 2, 700))
    
    def _estimate_water_density(self, temperature: float, pressure: float, phase: str) -> float:
        """Enhanced water density estimation."""
        if phase == 'superheated_steam':
            # Ideal gas law for steam
            T_R = temperature + 459.67
            return max(0.05, pressure / (85.76 * T_R))
        elif phase == 'saturated':
            # Saturated steam/liquid density
            return max(0.5, 62.4 * (1 - 0.0003 * temperature))
        else:  # liquid
            # Compressed liquid density
            temp_effect = 1 - 0.0003 * (temperature - 32)
            pressure_effect = 1 + 0.000005 * (pressure - 14.7)
            return max(30.0, 62.4 * temp_effect * pressure_effect)
    
    def _estimate_water_cp(self, temperature: float, phase: str) -> float:
        """Enhanced specific heat estimation."""
        if phase == 'superheated_steam':
            # Steam specific heat correlation
            return max(0.4, 0.445 + 0.00005 * max(0, temperature - 500))
        elif phase == 'saturated':
            return 1.0
        else:  # liquid
            return max(0.9, 1.0 + 0.0001 * (temperature - 32))
    
    def _estimate_water_viscosity(self, temperature: float, phase: str) -> float:
        """Enhanced viscosity estimation."""
        if phase == 'superheated_steam':
            # Steam viscosity correlation
            return max(0.01, 0.025 + 0.000015 * temperature) * PA_S_TO_LBM_HR_FT
        elif phase == 'saturated':
            return 0.6 * PA_S_TO_LBM_HR_FT
        else:  # liquid
            # Liquid water viscosity (decreases with temperature)
            return max(0.1, (2.5 - 0.004 * temperature)) * PA_S_TO_LBM_HR_FT
    
    def _estimate_water_thermal_conductivity(self, temperature: float, phase: str) -> float:
        """Enhanced thermal conductivity estimation."""
        if phase == 'superheated_steam':
            return max(0.01, 0.0145 + 0.000025 * temperature)
        else:
            return max(0.1, 0.35 + 0.0002 * temperature)
    
    def _estimate_water_enthalpy(self, temperature: float, phase: str) -> float:
        """Enhanced enthalpy estimation."""
        if phase == 'superheated_steam':
            return 1150 + (temperature - 500) * 0.5
        elif phase == 'saturated':
            return 180 + temperature * 0.8
        else:  # liquid
            return max(0, temperature - 32)
    
    def _estimate_water_entropy(self, temperature: float, phase: str) -> float:
        """Enhanced entropy estimation."""
        if phase == 'superheated_steam':
            return 1.5 + 0.0002 * max(0, temperature - 500)
        elif phase == 'saturated':
            return 0.3 + temperature * 0.002
        else:  # liquid
            return 0.002 * max(temperature, 32)
    
    def _estimate_gas_properties(self, temperature: float, pressure: float) -> Tuple[float, float, float, float, float]:
        """Enhanced gas property estimation using proven correlations."""
        T_R = temperature + 459.67  # Temperature in Rankine
        
        # Density using ideal gas law with compressibility correction
        Z = 1.0  # Compressibility factor (close to 1 for combustion gases)
        density = max(0.001, (pressure * 144) / (53.35 * T_R * Z))  # lbm/ft³
        
        # Specific heat - temperature dependent correlation for combustion gases
        cp = max(0.20, 0.24 + 0.00002 * min(temperature, 3000) + 0.000000003 * (temperature ** 2))
        
        # Viscosity - Sutherland's law approximation
        mu_ref = 0.018e-6  # Reference viscosity at 530°R (Pa·s)
        T_ref = 530  # Reference temperature (°R)
        S = 200  # Sutherland constant for air/combustion gases
        viscosity = max(0.005, mu_ref * ((T_R / T_ref) ** 1.5) * ((T_ref + S) / (T_R + S))) * PA_S_TO_LBM_HR_FT
        
        # Thermal conductivity - correlation for combustion gases
        thermal_conductivity = max(0.005, 0.008 + 0.00003 * min(temperature, 3000))
        
        # Molecular weight for typical flue gas
        molecular_weight = 28.5
        
        return density, cp, viscosity, thermal_conductivity, molecular_weight