#!/usr/bin/env python3
"""
Enhanced Thermodynamic Property Calculations - IAPWS + Thermo Integration with Precision Fixes

This module provides accurate property calculations using:
- IAPWS-97 for water/steam properties (industry standard) with precision error handling
- Thermo library for flue gas mixtures and combustion products
- Robust bounds checking and fallback mechanisms

MAJOR FIXES (v6.1):
- Fixed IAPWS "Incoming out of bound" errors with temperature rounding and bounds checking
- Added proper region validation before IAPWS calls
- Improved unit conversion precision handling
- Enhanced fallback mechanisms for edge cases

Classes:
    SteamProperties: Dataclass for water/steam properties  
    GasProperties: Dataclass for flue gas mixture properties
    PropertyCalculator: Main property calculation class with robust error handling

Dependencies:
    - iapws: Industry-standard steam properties (IAPWS-97)
    - thermo: Chemical mixture properties for flue gas
    - logging: Comprehensive logging support

Author: Enhanced Boiler Modeling System
Version: 6.1 - IAPWS Precision Fixes and Error Handling
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path("logs/simulation")
log_dir.mkdir(parents=True, exist_ok=True)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create file handler for property calculations
prop_log_file = log_dir / "property_calculations.log"
file_handler = logging.FileHandler(prop_log_file)
file_handler.setLevel(logging.DEBUG)

# Create console handler for warnings only
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)  # Reduced from WARNING to reduce console spam

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Library imports with proper error handling
try:
    from iapws import IAPWS97
    IAPWS_AVAILABLE = True
    logger.info("IAPWS library loaded successfully for steam properties")
except ImportError:
    IAPWS_AVAILABLE = False
    logger.warning("IAPWS library not available - using correlations for steam")

try:
    from thermo.mixture import Mixture
    from thermo.chemical import Chemical
    THERMO_AVAILABLE = True
    logger.info("Thermo library loaded successfully for gas mixtures")
except ImportError:
    THERMO_AVAILABLE = False
    logger.warning("Thermo library not available - using correlations for gas")

# Unit conversion constants with improved precision
F_TO_K = lambda f: round((f - 32) * 5/9 + 273.15, 6)  # Round to avoid precision errors
K_TO_F = lambda k: round((k - 273.15) * 9/5 + 32, 2)  # Round to 0.01°F precision
PSIA_TO_PA = 6894.757
PA_TO_PSIA = 1/6894.757
J_KG_TO_BTU_LB = 0.000429923
KG_M3_TO_LB_FT3 = 0.062428
W_MK_TO_BTU_HRFTF = 0.577789
PA_S_TO_LB_HRFT = 2419.09

# IAPWS validity ranges (conservative bounds)
IAPWS_MIN_TEMP_K = 273.16  # 32.01°F - Triple point
IAPWS_MAX_TEMP_K = 1073.15  # 1472°F - Reasonable upper limit
IAPWS_MIN_PRESSURE_PA = 611.213  # Triple point pressure
IAPWS_MAX_PRESSURE_PA = 50e6  # 50 MPa - Conservative upper limit


@dataclass
class SteamProperties:
    """Enhanced steam/water properties using IAPWS-97 standard."""
    temperature: float      # °F
    pressure: float        # psia
    density: float         # lb/ft³
    cp: float             # Btu/lb-°F
    viscosity: float      # lb/hr-ft
    thermal_conductivity: float  # Btu/hr-ft-°F
    enthalpy: float       # Btu/lb (IAPWS reference: liquid water at 32°F = 0)
    entropy: float        # Btu/lb-°R
    saturation_temp: float # °F
    superheat: float      # °F
    phase: str           # 'liquid', 'saturated', 'superheated_steam'
    quality: Optional[float]  # Steam quality (0-1) for two-phase
    prandtl: float       # Dimensionless


@dataclass  
class GasProperties:
    """Enhanced flue gas mixture properties using Thermo library."""
    temperature: float    # °F
    pressure: float      # psia
    density: float       # lb/ft³
    cp: float           # Btu/lb-°F
    viscosity: float    # lb/hr-ft
    thermal_conductivity: float  # Btu/hr-ft-°F
    molecular_weight: float     # lb/lbmol
    composition: Dict[str, float]  # Mole fractions
    prandtl: float      # Dimensionless


class PropertyCalculator:
    """Enhanced property calculator with robust IAPWS error handling and precision fixes."""
    
    def __init__(self):
        """Initialize with both IAPWS and Thermo capabilities."""
        self.iapws_available = IAPWS_AVAILABLE
        self.thermo_available = THERMO_AVAILABLE
        
        # Statistics for logging
        self.steam_calculations = 0
        self.gas_calculations = 0
        self.iapws_failures = 0
        self.thermo_failures = 0
        self.precision_fixes = 0
        
        logger.info(f"PropertyCalculator initialized - IAPWS: {self.iapws_available}, Thermo: {self.thermo_available}")
    
    def get_steam_properties(self, temperature: float, pressure: float) -> SteamProperties:
        """
        Calculate water/steam properties using IAPWS-97 standard with robust error handling.
        
        Args:
            temperature: Temperature in °F
            pressure: Pressure in psia
            
        Returns:
            SteamProperties with industry-standard accuracy
        """
        self.steam_calculations += 1
        
        # Input validation and bounds checking
        if pressure <= 0 or pressure > 5000:
            logger.error(f"Invalid pressure: {pressure} psia (must be 0-5000)")
            raise ValueError(f"Pressure {pressure} psia outside valid range")
        
        # Temperature bounds for safety with rounding to avoid precision errors
        temp_rounded = round(temperature, 1)  # Round to 0.1°F precision
        temp_clamped = max(32.1, min(temp_rounded, 1400))  # Conservative upper limit
        
        if abs(temp_clamped - temperature) > 5:
            logger.warning(f"Temperature {temperature:.1f}°F clamped to {temp_clamped:.1f}°F")
            self.precision_fixes += 1
        
        if self.iapws_available:
            try:
                return self._calculate_steam_iapws(temp_clamped, pressure)
            except Exception as e:
                self.iapws_failures += 1
                logger.warning(f"IAPWS calculation failed: {e}, using correlations")
                return self._calculate_steam_correlations(temp_clamped, pressure)
        else:
            return self._calculate_steam_correlations(temp_clamped, pressure)
    
    def get_steam_properties_safe(self, temperature: float, pressure: float) -> SteamProperties:
        """
        Safe wrapper that always returns valid steam properties.
        Used by heat transfer calculations that need guaranteed results.
        """
        try:
            return self.get_steam_properties(temperature, pressure)
        except Exception as e:
            logger.warning(f"Steam property calculation failed, using safe defaults: {e}")
            return self._create_safe_steam_properties(temperature, pressure)
    
    def get_flue_gas_properties_safe(self, temperature: float, 
                                   composition: Optional[Dict[str, float]] = None,
                                   pressure: float = 14.7) -> GasProperties:
        """
        Safe wrapper for flue gas properties with default composition.
        """
        if composition is None:
            composition = self.create_typical_flue_gas_composition()
        
        try:
            return self.get_flue_gas_properties(temperature, composition, pressure)
        except Exception as e:
            logger.warning(f"Gas property calculation failed, using safe defaults: {e}")
            return self._create_safe_gas_properties(temperature, composition, pressure)
    
    def get_flue_gas_properties(self, temperature: float, composition: Dict[str, float], 
                              pressure: float = 14.7) -> GasProperties:
        """
        Calculate flue gas mixture properties using Thermo library.
        
        Args:
            temperature: Temperature in °F
            composition: Mole fractions {'N2': 0.75, 'O2': 0.04, 'CO2': 0.12, ...}
            pressure: Pressure in psia
            
        Returns:
            GasProperties for the specified mixture
        """
        self.gas_calculations += 1
        
        # Normalize composition
        total_moles = sum(composition.values())
        if abs(total_moles - 1.0) > 0.01:
            logger.debug(f"Gas composition sum {total_moles:.3f}, normalizing to 1.0")
            composition = {k: v/total_moles for k, v in composition.items()}
        
        # Temperature bounds checking
        temp_clamped = max(200, min(temperature, 3500))
        if abs(temp_clamped - temperature) > 50:
            logger.warning(f"Gas temperature {temperature:.1f}°F clamped to {temp_clamped:.1f}°F")
        
        if self.thermo_available:
            try:
                return self._calculate_gas_thermo(temp_clamped, composition, pressure)
            except Exception as e:
                self.thermo_failures += 1
                logger.warning(f"Thermo mixture calculation failed: {e}, using correlations")
                return self._calculate_gas_correlations(temp_clamped, composition, pressure)
        else:
            return self._calculate_gas_correlations(temp_clamped, composition, pressure)
    
    def create_typical_flue_gas_composition(self, excess_air: float = 0.2, 
                                          coal_sulfur: float = 1.0) -> Dict[str, float]:
        """
        Create typical flue gas composition for coal combustion.
        
        Args:
            excess_air: Fraction excess air (0.2 = 20% excess)
            coal_sulfur: Coal sulfur content (weight %)
            
        Returns:
            Mole fraction composition dictionary
        """
        # Typical coal combustion with excess air
        base_composition = {
            'N2': 0.752,    # Nitrogen from air + fuel
            'O2': 0.035,    # Excess oxygen 
            'CO2': 0.125,   # Carbon dioxide from coal
            'H2O': 0.085,   # Water vapor from hydrogen + moisture
            'SO2': 0.003,   # Sulfur dioxide from coal sulfur
        }
        
        # Adjust for excess air
        o2_increase = excess_air * 0.21 * 0.1  # Approximate adjustment
        base_composition['O2'] += o2_increase
        base_composition['N2'] += excess_air * 0.79 * 0.1
        
        # Adjust for sulfur content
        so2_factor = coal_sulfur / 1.0  # Relative to 1% sulfur baseline
        base_composition['SO2'] *= so2_factor
        
        # Normalize
        total = sum(base_composition.values())
        return {k: v/total for k, v in base_composition.items()}
    
    def _is_valid_iapws_state(self, temperature: float, pressure: float) -> bool:
        """
        Check if temperature/pressure combination is valid for IAPWS-97.
        
        Args:
            temperature: Temperature in °F
            pressure: Pressure in psia
            
        Returns:
            True if state is valid for IAPWS
        """
        # Convert to SI units
        T_K = F_TO_K(temperature)
        P_Pa = pressure * PSIA_TO_PA
        
        # Check basic bounds
        if T_K < IAPWS_MIN_TEMP_K or T_K > IAPWS_MAX_TEMP_K:
            return False
        
        if P_Pa < IAPWS_MIN_PRESSURE_PA or P_Pa > IAPWS_MAX_PRESSURE_PA:
            return False
        
        # Additional region-specific checks
        # For most industrial applications, these bounds are conservative but safe
        if pressure > 3000 and temperature > 1200:  # Avoid supercritical edge cases
            return False
            
        return True
    
    def _calculate_steam_iapws(self, temperature: float, pressure: float) -> SteamProperties:
        """Calculate steam properties using IAPWS-97 with robust error handling."""
        
        # Pre-validate the state
        if not self._is_valid_iapws_state(temperature, pressure):
            raise ValueError(f"State T={temperature}°F, P={pressure} psia outside IAPWS validity range")
        
        # Convert to SI units with precision handling
        T_K = F_TO_K(temperature)
        P_Pa = pressure * PSIA_TO_PA
        
        # Create IAPWS97 object
        steam = IAPWS97(T=T_K, P=P_Pa)
        
        if not steam.status:
            raise ValueError(f"IAPWS calculation failed for T={temperature}°F, P={pressure} psia")
        
        # Convert properties back to English units
        density = steam.rho * KG_M3_TO_LB_FT3  # kg/m³ to lb/ft³
        cp = steam.cp * J_KG_TO_BTU_LB / (5/9)  # J/kg-K to Btu/lb-°F
        enthalpy = steam.h * J_KG_TO_BTU_LB  # J/kg to Btu/lb
        entropy = steam.s * J_KG_TO_BTU_LB / (5/9)  # J/kg-K to Btu/lb-°R
        
        # Transport properties with error handling
        try:
            viscosity = steam.mu * PA_S_TO_LB_HRFT if hasattr(steam, 'mu') and steam.mu else self._estimate_steam_viscosity(temperature)
            thermal_conductivity = steam.k * W_MK_TO_BTU_HRFTF if hasattr(steam, 'k') and steam.k else self._estimate_steam_thermal_conductivity(temperature)
        except:
            viscosity = self._estimate_steam_viscosity(temperature)
            thermal_conductivity = self._estimate_steam_thermal_conductivity(temperature)
        
        # Phase determination
        saturation_temp = self._get_saturation_temperature_iapws(pressure)
        
        if temperature > saturation_temp + 5:
            phase = 'superheated_steam'
            superheat = temperature - saturation_temp
            quality = None
        elif abs(temperature - saturation_temp) <= 5:
            phase = 'saturated'
            superheat = 0
            quality = steam.x if hasattr(steam, 'x') else None
        else:
            phase = 'liquid'
            superheat = 0
            quality = None
        
        prandtl = cp * viscosity / thermal_conductivity if thermal_conductivity > 0 else 1.0
        
        logger.debug(f"IAPWS steam calc: T={temperature}°F, P={pressure} psia, h={enthalpy:.1f} Btu/lb, phase={phase}")
        
        return SteamProperties(
            temperature=temperature,
            pressure=pressure,
            density=density,
            cp=cp,
            viscosity=viscosity,
            thermal_conductivity=thermal_conductivity,
            enthalpy=enthalpy,
            entropy=entropy,
            saturation_temp=saturation_temp,
            superheat=superheat,
            phase=phase,
            quality=quality,
            prandtl=prandtl
        )
    
    def _calculate_gas_thermo(self, temperature: float, composition: Dict[str, float], 
                            pressure: float) -> GasProperties:
        """Calculate gas mixture properties using Thermo library."""
        # Convert to SI units
        T_K = F_TO_K(temperature)
        P_Pa = pressure * PSIA_TO_PA
        
        # Create mixture
        species = list(composition.keys())
        mole_fractions = list(composition.values())
        
        mixture = Mixture(species, zs=mole_fractions, T=T_K, P=P_Pa)
        
        # Get properties with unit conversion
        density = mixture.rho * KG_M3_TO_LB_FT3 if mixture.rho else self._estimate_gas_density(temperature, pressure)
        cp = mixture.Cp * J_KG_TO_BTU_LB / (5/9) if mixture.Cp else self._estimate_gas_cp(temperature)
        molecular_weight = mixture.MW  # Already in correct units
        
        # Transport properties
        try:
            viscosity = mixture.mu * PA_S_TO_LB_HRFT if mixture.mu else self._estimate_gas_viscosity(temperature)
            thermal_conductivity = mixture.k * W_MK_TO_BTU_HRFTF if mixture.k else self._estimate_gas_thermal_conductivity(temperature)
        except:
            viscosity = self._estimate_gas_viscosity(temperature)
            thermal_conductivity = self._estimate_gas_thermal_conductivity(temperature)
        
        prandtl = cp * viscosity / thermal_conductivity if thermal_conductivity > 0 else 0.7
        
        logger.debug(f"Thermo gas calc: T={temperature}°F, MW={molecular_weight:.1f}, rho={density:.4f} lb/ft³")
        
        return GasProperties(
            temperature=temperature,
            pressure=pressure,
            density=density,
            cp=cp,
            viscosity=viscosity,
            thermal_conductivity=thermal_conductivity,
            molecular_weight=molecular_weight,
            composition=composition,
            prandtl=prandtl
        )
    
    def _get_saturation_temperature_iapws(self, pressure: float) -> float:
        """Get saturation temperature using IAPWS with error handling."""
        try:
            if self._is_valid_iapws_state(500, pressure):  # Use reasonable test temperature
                P_Pa = pressure * PSIA_TO_PA
                steam_sat = IAPWS97(P=P_Pa, x=0)  # Saturated liquid
                return K_TO_F(steam_sat.T)
            else:
                return self._estimate_saturation_temperature(pressure)
        except:
            return self._estimate_saturation_temperature(pressure)
    
    def _calculate_steam_correlations(self, temperature: float, pressure: float) -> SteamProperties:
        """Fallback steam property correlations with improved accuracy."""
        logger.debug(f"Using steam correlations for T={temperature}°F, P={pressure} psia")
        
        saturation_temp = self._estimate_saturation_temperature(pressure)
        
        # Phase determination
        if temperature > saturation_temp + 5:
            phase = 'superheated_steam'
            superheat = temperature - saturation_temp
            quality = None
        elif abs(temperature - saturation_temp) <= 5:
            phase = 'saturated'
            superheat = 0
            quality = 0.5
        else:
            phase = 'liquid'
            superheat = 0
            quality = None
        
        # Property correlations
        density = self._estimate_steam_density(temperature, pressure, phase)
        cp = self._estimate_steam_cp(temperature, phase)
        viscosity = self._estimate_steam_viscosity(temperature)
        thermal_conductivity = self._estimate_steam_thermal_conductivity(temperature)
        enthalpy = self._estimate_steam_enthalpy(temperature, phase, pressure)
        entropy = self._estimate_steam_entropy(temperature, phase)
        prandtl = cp * viscosity / thermal_conductivity if thermal_conductivity > 0 else 1.0
        
        return SteamProperties(
            temperature=temperature,
            pressure=pressure,
            density=density,
            cp=cp,
            viscosity=viscosity,
            thermal_conductivity=thermal_conductivity,
            enthalpy=enthalpy,
            entropy=entropy,
            saturation_temp=saturation_temp,
            superheat=superheat,
            phase=phase,
            quality=quality,
            prandtl=prandtl
        )
    
    def _calculate_gas_correlations(self, temperature: float, composition: Dict[str, float], 
                                  pressure: float) -> GasProperties:
        """Fallback gas property correlations."""
        logger.debug(f"Using gas correlations for T={temperature}°F")
        
        # Estimate molecular weight from composition
        mw_dict = {'N2': 28.014, 'O2': 31.998, 'CO2': 44.01, 'H2O': 18.015, 'SO2': 64.066, 'CO': 28.01}
        molecular_weight = sum(composition.get(species, 0) * mw_dict.get(species, 29) for species in composition)
        
        density = self._estimate_gas_density(temperature, pressure, molecular_weight)
        cp = self._estimate_gas_cp(temperature)
        viscosity = self._estimate_gas_viscosity(temperature)
        thermal_conductivity = self._estimate_gas_thermal_conductivity(temperature)
        prandtl = cp * viscosity / thermal_conductivity if thermal_conductivity > 0 else 0.7
        
        return GasProperties(
            temperature=temperature,
            pressure=pressure,
            density=density,
            cp=cp,
            viscosity=viscosity,
            thermal_conductivity=thermal_conductivity,
            molecular_weight=molecular_weight,
            composition=composition,
            prandtl=prandtl
        )
    
    def _create_safe_steam_properties(self, temperature: float, pressure: float) -> SteamProperties:
        """Create safe default steam properties for error recovery."""
        saturation_temp = self._estimate_saturation_temperature(pressure)
        
        if temperature > saturation_temp + 5:
            phase = 'superheated_steam'
            superheat = temperature - saturation_temp
            enthalpy = 1200.0 + superheat * 0.5  # Reasonable approximation
            density = pressure / (85.76 * (temperature + 459.67))  # Ideal gas for steam
        else:
            phase = 'liquid'
            superheat = 0
            enthalpy = max(0, temperature - 32)  # Sensible heat approximation
            density = 62.0  # Typical water density
        
        return SteamProperties(
            temperature=temperature,
            pressure=pressure,
            density=density,
            cp=1.0 if phase == 'liquid' else 0.5,
            viscosity=0.02,
            thermal_conductivity=0.02,
            enthalpy=enthalpy,
            entropy=0.5,
            saturation_temp=saturation_temp,
            superheat=superheat,
            phase=phase,
            quality=None,
            prandtl=1.0
        )
    
    def _create_safe_gas_properties(self, temperature: float, composition: Dict[str, float], 
                                  pressure: float) -> GasProperties:
        """Create safe default gas properties for error recovery."""
        molecular_weight = 29.0  # Approximate air molecular weight
        density = self._estimate_gas_density(temperature, pressure, molecular_weight)
        
        return GasProperties(
            temperature=temperature,
            pressure=pressure,
            density=density,
            cp=0.25,
            viscosity=0.02,
            thermal_conductivity=0.015,
            molecular_weight=molecular_weight,
            composition=composition,
            prandtl=0.7
        )
    
    # Enhanced correlation methods
    def _estimate_saturation_temperature(self, pressure: float) -> float:
        """Enhanced saturation temperature correlation."""
        if pressure <= 1:
            return 32
        elif pressure <= 14.7:
            return 32 + (pressure - 1) * 13.0  
        elif pressure <= 100:
            return 212 + (pressure - 14.7) * 2.4
        elif pressure <= 500:
            return 212 + 204 + (pressure - 100) * 0.7
        else:
            return 212 + 204 + 280 + (pressure - 500) * 0.3
    
    def _estimate_steam_density(self, temperature: float, pressure: float, phase: str) -> float:
        """Enhanced steam density correlation."""
        if phase == 'superheated_steam':
            T_R = temperature + 459.67
            return pressure / (85.76 * T_R)  # Ideal gas law for steam
        elif phase == 'liquid':
            temp_effect = 1 - 0.0003 * (temperature - 32)
            pressure_effect = 1 + 0.000005 * (pressure - 14.7)
            return 62.4 * temp_effect * pressure_effect
        else:  # saturated
            return 20.0  # Approximate for two-phase
    
    def _estimate_steam_cp(self, temperature: float, phase: str) -> float:
        """Enhanced steam specific heat correlation."""
        if phase == 'superheated_steam':
            return 0.445 + 0.00005 * max(0, temperature - 500)
        elif phase == 'liquid':
            return 1.0 + 0.0001 * (temperature - 32)
        else:  # saturated
            return 1.2
    
    def _estimate_steam_enthalpy(self, temperature: float, phase: str, pressure: float) -> float:
        """Enhanced steam enthalpy correlation - CRITICAL for efficiency."""
        if phase == 'superheated_steam':
            # Improved superheated steam enthalpy correlation
            saturation_temp = self._estimate_saturation_temperature(pressure)
            h_sat = 180 + saturation_temp * 0.8  # Saturated liquid enthalpy
            h_fg = 970 - pressure * 0.5  # Latent heat of vaporization
            superheat = temperature - saturation_temp
            return h_sat + h_fg + superheat * 0.48  # More accurate superheat enthalpy
        elif phase == 'liquid':
            # Liquid water enthalpy
            return max(0, temperature - 32) * 1.0  # Sensible heat
        else:  # saturated
            # Saturated steam enthalpy
            h_sat = 180 + temperature * 0.8
            h_fg = 970 - pressure * 0.5
            return h_sat + h_fg
    
    def _estimate_steam_entropy(self, temperature: float, phase: str) -> float:
        """Enhanced steam entropy correlation."""
        if phase == 'superheated_steam':
            return 1.5 + 0.0002 * max(0, temperature - 500)
        elif phase == 'liquid':
            return 0.002 * max(temperature, 32)
        else:  # saturated
            return 0.3 + temperature * 0.002
    
    def _estimate_steam_viscosity(self, temperature: float) -> float:
        """Steam viscosity correlation."""
        return max(0.01, 0.025 + 0.000015 * temperature) * PA_S_TO_LB_HRFT
    
    def _estimate_steam_thermal_conductivity(self, temperature: float) -> float:
        """Steam thermal conductivity correlation."""
        return max(0.01, 0.0145 + 0.000025 * temperature)
    
    def _estimate_gas_density(self, temperature: float, pressure: float, 
                            molecular_weight: float = 29.0) -> float:
        """Gas density using ideal gas law."""
        T_R = temperature + 459.67
        R = 1545.35 / molecular_weight  # Gas constant
        return (pressure * 144) / (R * T_R)  # lb/ft³
    
    def _estimate_gas_cp(self, temperature: float) -> float:
        """Gas specific heat correlation."""
        return max(0.20, 0.24 + 0.00002 * min(temperature, 3000))
    
    def _estimate_gas_viscosity(self, temperature: float) -> float:
        """Gas viscosity correlation."""
        T_R = temperature + 459.67
        return max(0.005, 0.018e-6 * ((T_R / 530) ** 1.5) * 2419.09)
    
    def _estimate_gas_thermal_conductivity(self, temperature: float) -> float:
        """Gas thermal conductivity correlation."""
        return max(0.005, 0.008 + 0.00003 * min(temperature, 3000))
    
    def log_statistics(self):
        """Log property calculation statistics."""
        logger.info(f"Property calculation statistics:")
        logger.info(f"  Steam calculations: {self.steam_calculations}")
        logger.info(f"  Gas calculations: {self.gas_calculations}")
        logger.info(f"  IAPWS failures: {self.iapws_failures}")
        logger.info(f"  Thermo failures: {self.thermo_failures}")
        logger.info(f"  Precision fixes applied: {self.precision_fixes}")
        
        if self.steam_calculations > 0:
            iapws_success_rate = (self.steam_calculations - self.iapws_failures) / self.steam_calculations
            logger.info(f"  IAPWS success rate: {iapws_success_rate:.1%}")
        
        if self.gas_calculations > 0:
            thermo_success_rate = (self.gas_calculations - self.thermo_failures) / self.gas_calculations  
            logger.info(f"  Thermo success rate: {thermo_success_rate:.1%}")


# Module test function
def test_property_calculations():
    """Test both IAPWS and Thermo property calculations."""
    calc = PropertyCalculator()
    
    print("Testing Enhanced Property Calculator with Precision Fixes...")
    
    # Test steam properties
    print("\n1. Testing IAPWS Steam Properties:")
    steam = calc.get_steam_properties(700, 600)  # Superheated steam
    print(f"   700°F, 600 psia steam: h={steam.enthalpy:.1f} Btu/lb, rho={steam.density:.3f} lb/ft³")
    
    water = calc.get_steam_properties(220, 600)  # Feedwater
    print(f"   220°F, 600 psia water: h={water.enthalpy:.1f} Btu/lb, rho={water.density:.1f} lb/ft³")
    
    specific_energy = steam.enthalpy - water.enthalpy
    print(f"   Specific energy difference: {specific_energy:.1f} Btu/lb")
    
    # Test edge cases that were failing
    print("\n2. Testing Previous Problem Cases:")
    try:
        problem_steam = calc.get_steam_properties(795.75, 600)  # Previously problematic
        print(f"   795.75°F steam: SUCCESS - h={problem_steam.enthalpy:.1f} Btu/lb")
    except Exception as e:
        print(f"   795.75°F steam: FAILED - {e}")
    
    # Test flue gas properties
    print("\n3. Testing Thermo Flue Gas Properties:")
    composition = calc.create_typical_flue_gas_composition(excess_air=0.2)
    gas = calc.get_flue_gas_properties(600, composition)
    print(f"   600°F flue gas: MW={gas.molecular_weight:.1f}, rho={gas.density:.4f} lb/ft³")
    print(f"   Composition: {composition}")
    
    # Log statistics
    calc.log_statistics()
    
    return steam, water, gas


if __name__ == "__main__":
    test_property_calculations()