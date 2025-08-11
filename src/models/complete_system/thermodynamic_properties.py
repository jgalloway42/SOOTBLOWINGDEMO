#!/usr/bin/env python3
"""
Thermodynamic Property Calculations

This module contains property calculation utilities using the thermo library
for accurate steam, water, and flue gas properties.

Classes:
    SteamProperties: Dataclass for water/steam properties
    GasProperties: Dataclass for flue gas properties
    PropertyCalculator: Main property calculation class

Dependencies:
    - thermo: Comprehensive thermodynamic library
    - dataclasses: For structured data
    - typing: Type hints

Author: Enhanced Boiler Modeling System
Version: 5.0 - Thermo Library Integration
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from thermo.chemical import Chemical
from thermo.mixture import Mixture

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
        # Create water substance for steam calculations
        self.water = Chemical('water')
        
        # Create typical flue gas mixture
        self.flue_gas_components = ['nitrogen', 'carbon dioxide', 'water', 'oxygen']
        self.flue_gas_fractions = [0.75, 0.15, 0.08, 0.02]  # Mole fractions
        
        # Initialize flue gas mixture
        self.flue_gas_mixture = Mixture(
            IDs=self.flue_gas_components,
            zs=self.flue_gas_fractions,
            T=800,  # Initial temperature (K)
            P=101325  # Initial pressure (Pa)
        )
    
    def get_steam_properties(self, temperature: float, pressure: float) -> SteamProperties:
        """Calculate comprehensive water/steam properties using thermo library."""
        if temperature < 32 or temperature > 2000:
            raise ValueError(f"Temperature {temperature}°F outside range (32-2000°F)")
        if pressure < 1 or pressure > 5000:
            raise ValueError(f"Pressure {pressure} psia outside range (1-5000 psia)")
        
        # Convert to SI units
        T_K = F_TO_K(temperature)
        P_Pa = pressure * PSIA_TO_PA
        
        # Update water object with conditions
        self.water.calculate(T=T_K, P=P_Pa)
        
        # Get saturation temperature
        sat_temp_K = self.water.Tsat(P_Pa)
        sat_temp_F = K_TO_F(sat_temp_K)
        
        # Determine phase and calculate quality if applicable
        if T_K > sat_temp_K + 1:  # Superheated steam
            phase = 'superheated_steam'
            quality = None
            superheat = temperature - sat_temp_F
        elif abs(T_K - sat_temp_K) <= 1:  # Near saturation
            phase = 'saturated'
            quality = 0.5  # Assume 50% quality for saturated conditions
            superheat = 0
        else:  # Compressed liquid
            phase = 'liquid'
            quality = None
            superheat = 0
        
        # Get properties and convert to English units
        try:
            # Density
            if hasattr(self.water, 'rho') and self.water.rho is not None:
                density = self.water.rho * KG_M3_TO_LBM_FT3
            else:
                density = self._estimate_water_density(temperature, pressure, phase)
            
            # Specific heat
            if hasattr(self.water, 'Cp') and self.water.Cp is not None:
                cp = self.water.Cp * J_KG_K_TO_BTU_LBM_F
            else:
                cp = self._estimate_water_cp(temperature, phase)
            
            # Viscosity
            if hasattr(self.water, 'mu') and self.water.mu is not None:
                viscosity = self.water.mu * PA_S_TO_LBM_HR_FT
            else:
                viscosity = self._estimate_water_viscosity(temperature, phase)
            
            # Thermal conductivity
            if hasattr(self.water, 'k') and self.water.k is not None:
                thermal_conductivity = self.water.k * W_M_K_TO_BTU_HR_FT_F
            else:
                thermal_conductivity = self._estimate_water_thermal_conductivity(temperature, phase)
            
            # Enthalpy
            if hasattr(self.water, 'H') and self.water.H is not None:
                enthalpy = self.water.H * J_KG_TO_BTU_LBM
            else:
                enthalpy = self._estimate_water_enthalpy(temperature, phase)
            
            # Entropy
            if hasattr(self.water, 'S') and self.water.S is not None:
                entropy = self.water.S * J_KG_K_TO_BTU_LBM_F
            else:
                entropy = self._estimate_water_entropy(temperature, phase)
            
            # Prandtl number
            prandtl = cp * viscosity / thermal_conductivity
            
        except Exception as e:
            print(f"Warning: Thermo calculation failed, using estimates: {e}")
            density = self._estimate_water_density(temperature, pressure, phase)
            cp = self._estimate_water_cp(temperature, phase)
            viscosity = self._estimate_water_viscosity(temperature, phase)
            thermal_conductivity = self._estimate_water_thermal_conductivity(temperature, phase)
            enthalpy = self._estimate_water_enthalpy(temperature, phase)
            entropy = self._estimate_water_entropy(temperature, phase)
            prandtl = cp * viscosity / thermal_conductivity
        
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
    
    def get_flue_gas_properties(self, temperature: float, pressure: float = 14.7) -> GasProperties:
        """Calculate flue gas mixture properties using thermo library."""
        # Convert to SI units
        T_K = F_TO_K(temperature)
        P_Pa = pressure * PSIA_TO_PA
        
        try:
            # Update mixture conditions
            self.flue_gas_mixture.calculate(T=T_K, P=P_Pa)
            
            # Get properties and convert to English units
            density = self.flue_gas_mixture.rho * KG_M3_TO_LBM_FT3
            cp = self.flue_gas_mixture.Cp * J_KG_K_TO_BTU_LBM_F
            viscosity = self.flue_gas_mixture.mu * PA_S_TO_LBM_HR_FT
            thermal_conductivity = self.flue_gas_mixture.k * W_M_K_TO_BTU_HR_FT_F
            molecular_weight = self.flue_gas_mixture.MW
            
            prandtl = cp * viscosity / thermal_conductivity
            
        except Exception as e:
            print(f"Warning: Thermo gas calculation failed, using correlations: {e}")
            # Fallback to correlations
            density, cp, viscosity, thermal_conductivity, molecular_weight = self._estimate_gas_properties(temperature, pressure)
            prandtl = cp * viscosity / thermal_conductivity
        
        return GasProperties(
            temperature=temperature,
            cp=cp,
            density=density,
            viscosity=viscosity,
            thermal_conductivity=thermal_conductivity,
            prandtl=prandtl,
            molecular_weight=molecular_weight
        )
    
    def _estimate_water_density(self, temperature: float, pressure: float, phase: str) -> float:
        """Estimate water density using correlations."""
        if phase == 'superheated_steam':
            return pressure / (85.76 * (temperature + 459.67))
        elif phase == 'saturated':
            return 50.0 - 0.015 * temperature
        else:  # liquid
            return 62.4 - 0.008 * temperature
    
    def _estimate_water_cp(self, temperature: float, phase: str) -> float:
        """Estimate water specific heat using correlations."""
        if phase == 'superheated_steam':
            return 0.445 + 0.000025 * max(0, temperature - 500)
        elif phase == 'saturated':
            return 1.0
        else:  # liquid
            return 1.0 + 0.0002 * temperature
    
    def _estimate_water_viscosity(self, temperature: float, phase: str) -> float:
        """Estimate water viscosity using correlations."""
        if phase == 'superheated_steam':
            return (0.025 + 0.000012 * temperature) * PA_S_TO_LBM_HR_FT
        elif phase == 'saturated':
            return 0.6 * PA_S_TO_LBM_HR_FT
        else:  # liquid
            return max(0.1, (2.4 - 0.003 * temperature)) * PA_S_TO_LBM_HR_FT
    
    def _estimate_water_thermal_conductivity(self, temperature: float, phase: str) -> float:
        """Estimate water thermal conductivity using correlations."""
        if phase == 'superheated_steam':
            return 0.0145 + 0.000020 * temperature
        else:
            return 0.35 + 0.0001 * temperature
    
    def _estimate_water_enthalpy(self, temperature: float, phase: str) -> float:
        """Estimate water enthalpy using correlations."""
        if phase == 'superheated_steam':
            return 1150 + (temperature - 500) * 0.5
        elif phase == 'saturated':
            return 180 + temperature * 0.8
        else:  # liquid
            return temperature - 32
    
    def _estimate_water_entropy(self, temperature: float, phase: str) -> float:
        """Estimate water entropy using correlations."""
        if phase == 'superheated_steam':
            return 1.5 + 0.0001 * (temperature - 500)
        elif phase == 'saturated':
            return 0.3 + temperature * 0.001
        else:  # liquid
            return 0.001 * temperature
    
    def _estimate_gas_properties(self, temperature: float, pressure: float) -> Tuple[float, float, float, float, float]:
        """Estimate gas properties using engineering correlations."""
        T_R = temperature + 459.67
        
        density = 0.0458 * (530 / T_R) * (pressure / 14.7)
        cp = 0.24 + 0.000012 * temperature
        viscosity = 0.018 * (T_R / 530) ** 0.7 * PA_S_TO_LBM_HR_FT
        thermal_conductivity = 0.008 + 0.000022 * temperature
        molecular_weight = 28.5  # Approximate for flue gas
        
        return density, cp, viscosity, thermal_conductivity, molecular_weight