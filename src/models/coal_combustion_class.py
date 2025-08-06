import numpy as np
import thermo
import scipy.optimize as opt
import matplotlib.pyplot as plt


class CoalCombustionModel:
    """
    A comprehensive coal combustion model that calculates emissions, heat release,
    and flame temperature based on coal properties and combustion conditions.
    """
    
    def __init__(self, ultimate_analysis, coal_lb_per_hr, air_scfh, NOx_eff=0.35,
                 air_temp_F=75.0, air_RH_pct=55.0, atm_press_inHg=30.25):
        """
        Initialize the coal combustion model.
        
        Parameters:
        -----------
        ultimate_analysis : dict
            Ultimate analysis of coal with keys: 'C', 'H', 'O', 'N', 'S', 'Ash', 'Moisture'
            Values are weight percentages.
        coal_lb_per_hr : float
            Mass flow rate of coal in pounds per hour.
        air_scfh : float
            Volumetric flow rate of combustion air in standard cubic feet per hour.
        NOx_eff : float, optional
            Efficiency of conversion from fuel-bound nitrogen to NOx (default: 0.35).
        air_temp_F : float, optional
            Temperature of incoming air in degrees Fahrenheit (default: 75.0).
        air_RH_pct : float, optional
            Relative humidity of incoming air as percentage (default: 55.0).
        atm_press_inHg : float, optional
            Atmospheric pressure in inches of mercury (default: 30.25).
        """
        
        # Store input parameters
        self._ultimate_analysis = ultimate_analysis.copy()
        self._coal_lb_per_hr = coal_lb_per_hr
        self._air_scfh = air_scfh
        self._NOx_eff = NOx_eff
        self._air_temp_F = air_temp_F
        self._air_RH_pct = air_RH_pct
        self._atm_press_inHg = atm_press_inHg
        
        # Molecular weights
        self.MW = {
            'C': 12.01, 'H': 1.008, 'O': 16.00, 'N': 14.01, 'S': 32.07,
            'CO2': 44.01, 'CO': 28.01, 'H2O': 18.02, 'SO2': 64.07,
            'O2': 32.00, 'N2': 28.02, 'NO': 30.01
        }
        
        # Initialize calculated values to None
        self._interim_calcs = {}
        self._results = {}
        self._calculated = False
        
    # Input Properties
    @property
    def ultimate_analysis(self):
        """Ultimate analysis of coal (% by mass)."""
        return self._ultimate_analysis.copy()
    
    @ultimate_analysis.setter
    def ultimate_analysis(self, value):
        self._ultimate_analysis = value.copy()
        self._calculated = False
    
    @property
    def coal_lb_per_hr(self):
        """Coal mass flow rate (lb/hr)."""
        return self._coal_lb_per_hr
    
    @coal_lb_per_hr.setter
    def coal_lb_per_hr(self, value):
        self._coal_lb_per_hr = value
        self._calculated = False
    
    @property
    def air_scfh(self):
        """Air volumetric flow rate (SCFH)."""
        return self._air_scfh
    
    @air_scfh.setter
    def air_scfh(self, value):
        self._air_scfh = value
        self._calculated = False
    
    @property
    def NOx_eff(self):
        """NOx conversion efficiency (fraction)."""
        return self._NOx_eff
    
    @NOx_eff.setter
    def NOx_eff(self, value):
        self._NOx_eff = value
        self._calculated = False
    
    @property
    def air_temp_F(self):
        """Air temperature (°F)."""
        return self._air_temp_F
    
    @air_temp_F.setter
    def air_temp_F(self, value):
        self._air_temp_F = value
        self._calculated = False
    
    @property
    def air_RH_pct(self):
        """Air relative humidity (%)."""
        return self._air_RH_pct
    
    @air_RH_pct.setter
    def air_RH_pct(self, value):
        self._air_RH_pct = value
        self._calculated = False
    
    @property
    def atm_press_inHg(self):
        """Atmospheric pressure (inHg)."""
        return self._atm_press_inHg
    
    @atm_press_inHg.setter
    def atm_press_inHg(self, value):
        self._atm_press_inHg = value
        self._calculated = False
    
    # Interim Calculation Properties
    @property
    def coal_HHV_btu_per_lb(self):
        """Coal Higher Heating Value (BTU/lb)."""
        self._ensure_calculated()
        return self._interim_calcs['coal_HHV_btu_per_lb']
    
    @property
    def coal_dry_ash_free_lb_per_hr(self):
        """Coal dry ash-free mass flow rate (lb/hr)."""
        self._ensure_calculated()
        return self._interim_calcs['coal_dry_ash_free_lb_per_hr']
    
    @property
    def humidity_ratio(self):
        """Humidity ratio (kg_water/kg_dry_air)."""
        self._ensure_calculated()
        return self._interim_calcs['humidity_ratio']
    
    @property
    def CO2_fraction(self):
        """CO2 fraction (CO2 formation vs CO2 + CO)."""
        self._ensure_calculated()
        return self._interim_calcs['CO2_fraction']
    
    @property
    def combustion_efficiency(self):
        """Combustion efficiency (fraction)."""
        self._ensure_calculated()
        return self._interim_calcs['combustion_efficiency']
    
    @property
    def actual_heat_release_btu_per_hr(self):
        """Actual heat release (BTU/hr)."""
        self._ensure_calculated()
        return self._interim_calcs['actual_heat_release_btu_per_hr']
    
    @property
    def flame_temp_K(self):
        """Estimated flame temperature (K)."""
        self._ensure_calculated()
        return self._interim_calcs['flame_temp_K']
    
    # Results Properties
    @property
    def total_flue_gas_lb_per_hr(self):
        """Total flue gas flow rate (lb/hr)."""
        self._ensure_calculated()
        return self._results['total_flue_gas_lb_per_hr']
    
    @property
    def CO2_lb_per_hr(self):
        """CO2 emissions (lb/hr)."""
        self._ensure_calculated()
        return self._results['CO2_lb_per_hr']
    
    @property
    def CO_lb_per_hr(self):
        """CO emissions (lb/hr)."""
        self._ensure_calculated()
        return self._results['CO_lb_per_hr']
    
    @property
    def H2O_lb_per_hr(self):
        """H2O in flue gas (lb/hr)."""
        self._ensure_calculated()
        return self._results['H2O_lb_per_hr']
    
    @property
    def SO2_lb_per_hr(self):
        """SO2 emissions (lb/hr)."""
        self._ensure_calculated()
        return self._results['SO2_lb_per_hr']
    
    @property
    def NO_fuel_lb_per_hr(self):
        """NO from fuel-bound nitrogen (lb/hr)."""
        self._ensure_calculated()
        return self._results['NO_fuel_lb_per_hr']
    
    @property
    def NO_thermal_lb_per_hr(self):
        """Thermal NO emissions (lb/hr)."""
        self._ensure_calculated()
        return self._results['NO_thermal_lb_per_hr']
    
    @property
    def NO_total_lb_per_hr(self):
        """Total NO emissions (lb/hr)."""
        self._ensure_calculated()
        return self._results['NO_total_lb_per_hr']
    
    @property
    def N2_lb_per_hr(self):
        """N2 in flue gas (lb/hr)."""
        self._ensure_calculated()
        return self._results['N2_lb_per_hr']
    
    @property
    def O2_lb_per_hr(self):
        """Excess O2 in flue gas (lb/hr)."""
        self._ensure_calculated()
        return self._results['O2_lb_per_hr']
    
    @property
    def dry_O2_pct(self):
        """Dry O2 concentration (% by volume)."""
        self._ensure_calculated()
        return self._results['dry_O2_pct']
    
    @property
    def heat_released_btu_per_hr(self):
        """Heat released (BTU/hr)."""
        self._ensure_calculated()
        return self._results['heat_released_btu_per_hr']
    
    @property
    def flame_temp_F(self):
        """Estimated flame temperature (°F)."""
        self._ensure_calculated()
        return self._results['flame_temp_F']
    
    def _ensure_calculated(self):
        """Ensure calculations have been performed."""
        if not self._calculated:
            self.calculate()
    
    def calculate(self, debug=False):
        """
        Perform all combustion calculations.
        
        Parameters:
        -----------
        debug : bool, optional
            Enable debugging output and plots (default: False).
        """
        
        # Validate inputs
        self._validate_inputs()
        
        # 1. Calculate corrected coal HHV
        HHV_btu_per_lb = self._calculate_corrected_hhv()
        
        # 2. Calculate air humidity ratio
        air_humidity_ratio = self._calculate_humidity_ratio()
        
        # 3. Calculate air composition
        air_moles = self._calculate_moles_from_scf_humid_air(air_humidity_ratio)
        
        # 4. Calculate coal composition
        frac = {k: v / 100.0 for k, v in self._ultimate_analysis.items()}
        coal_dry_ash_free_frac = 1.0 - frac.get('Moisture', 0) - frac.get('Ash', 0)
        coal_dry_ash_free_lb_per_hr = self._coal_lb_per_hr * coal_dry_ash_free_frac
        coal_moles = self._calculate_ash_free_moles()
        
        # 5. Calculate fuel-bound NOx generation
        prod_mol_NO_fuelbound = self._estimate_fuel_NO(coal_moles['N'])
        
        # 6. Initial combustion products calculation
        CO2_frac = 1.0
        products_mol = self._calculate_flue_gas_composition(coal_moles, air_moles, CO2_frac, prod_mol_NO_fuelbound)
        
        # 7. Calculate CO2 fraction based on excess O2
        excess_o2_pct = products_mol['O2'] / sum(products_mol.values()) * 100.0
        CO2_frac = self._co2_mole_fraction(excess_o2_pct)
        
        # 8. Update flue gas composition
        products_mol = self._calculate_flue_gas_composition(coal_moles, air_moles, CO2_frac, prod_mol_NO_fuelbound)
        
        # 9. Calculate flame temperature
        T_flame_K = self._simple_flame_temperature_estimate(HHV_btu_per_lb)
        
        # 10. Estimate thermal NOx
        total_moles = sum(products_mol.values())
        O2_mole_frac = products_mol['O2'] / total_moles
        N2_mole_frac = products_mol['N2'] / total_moles
        thermal_NO_rate = self._estimate_thermal_NO(O2_mole_frac, N2_mole_frac, T_flame_K)
        prod_mol_NO_thermal = thermal_NO_rate * products_mol['N2']
        
        # 11. Final flue gas composition with total NOx
        products_mol = self._calculate_flue_gas_composition(coal_moles, air_moles, CO2_frac,
                                                           prod_mol_NO_fuelbound + prod_mol_NO_thermal)
        
        # 12. Calculate combustion efficiency and heat release
        combustion_efficiency, actual_heat_release = self._calculate_combustion_efficiency_and_heat_release(
            coal_dry_ash_free_lb_per_hr, HHV_btu_per_lb, CO2_frac)
        
        # 13. Calculate accurate flame temperature
        T_flame_K = self._estimate_flame_temp_Cp_method(products_mol, HHV_btu_per_lb,
                                                       coal_dry_ash_free_lb_per_hr, CO2_frac, debug=debug)
        
        # Convert moles to pounds
        products_lb = {k: self._moles_to_lbs(v, self.MW[k]) for k, v in products_mol.items()}
        
        # Store interim calculations
        self._interim_calcs = {
            'coal_HHV_btu_per_lb': HHV_btu_per_lb,
            'coal_dry_ash_free_lb_per_hr': coal_dry_ash_free_lb_per_hr,
            'humidity_ratio': air_humidity_ratio,
            'CO2_fraction': CO2_frac,
            'combustion_efficiency': combustion_efficiency,
            'actual_heat_release_btu_per_hr': actual_heat_release,
            'flame_temp_K': T_flame_K
        }
        
        # Store results
        total_flue_gas_pph = sum(products_lb.values())
        total_dry_flue_gas_pph = total_flue_gas_pph - products_lb['H2O']
        excess_o2_pct = products_lb['O2'] / total_dry_flue_gas_pph * 100.0
        
        self._results = {
            'total_flue_gas_lb_per_hr': total_flue_gas_pph,
            'CO2_lb_per_hr': products_lb['CO2'],
            'CO_lb_per_hr': products_lb['CO'],
            'H2O_lb_per_hr': products_lb['H2O'],
            'SO2_lb_per_hr': products_lb['SO2'],
            'NO_fuel_lb_per_hr': self._moles_to_lbs(prod_mol_NO_fuelbound, self.MW['NO']),
            'NO_thermal_lb_per_hr': self._moles_to_lbs(prod_mol_NO_thermal, self.MW['NO']),
            'NO_total_lb_per_hr': products_lb['NO'],
            'N2_lb_per_hr': products_lb['N2'],
            'O2_lb_per_hr': products_lb['O2'],
            'dry_O2_pct': excess_o2_pct,
            'heat_released_btu_per_hr': actual_heat_release,
            'flame_temp_F': (T_flame_K - 273.15) * 9/5 + 32
        }
        
        self._calculated = True
        
        if debug:
            self.print_summary()
    
    def print_summary(self):
        """Print a summary of inputs, interim calculations, and results."""
        self._ensure_calculated()
        
        inputs = {
            'Air (scfh)': self._air_scfh,
            'Coal (pph) As-received': self._coal_lb_per_hr,
            'NOx Conversion Efficiency': self._NOx_eff,
            'Air temperature (F)': self._air_temp_F,
            'Relative Humidity (%)': self._air_RH_pct,
            'Atmospheric Pressure (inHg)': self._atm_press_inHg,
        }
        
        interim_calcs = {
            'Coal HHV (Btu/lb)': self.coal_HHV_btu_per_lb,
            'Coal Dry Ash-Free (pph)': self.coal_dry_ash_free_lb_per_hr,
            'Humidity Ratio (h2o/dry air by mass)': self.humidity_ratio,
            'CO2 Fraction (CO2 formation vs CO2 + CO)': self.CO2_fraction,
            'Combustion Efficiency': self.combustion_efficiency,
            'Actual Heat Release (BTU/hr)': self.actual_heat_release_btu_per_hr,
        }
        
        results = {
            'Total Flue Gas (lb/hr)': self.total_flue_gas_lb_per_hr,
            'CO2 (lb/hr)': self.CO2_lb_per_hr,
            'CO (lb/hr)': self.CO_lb_per_hr,
            'H2O (lb/hr)': self.H2O_lb_per_hr,
            'SO2 (lb/hr)': self.SO2_lb_per_hr,
            'NO (lb/hr from fuel-N)': self.NO_fuel_lb_per_hr,
            'NO (lb/hr thermal)': self.NO_thermal_lb_per_hr,
            'NO (lb/hr total)': self.NO_total_lb_per_hr,
            'N2 (lb/hr)': self.N2_lb_per_hr,
            'O2 (lb/hr)': self.O2_lb_per_hr,
            'Dry O2 (% by vol)': self.dry_O2_pct,
            'Heat Released (BTU/hr)': self.heat_released_btu_per_hr,
            'Estimated Flame Temp (F)': self.flame_temp_F
        }
        
        print("\n=== Inputs ===\n")
        for k, v in inputs.items():
            print(f"{k}: {v:,.2f}")
        
        print("\n=== Interim Calculations ===\n")
        for k, v in interim_calcs.items():
            print(f"{k}: {v:,.4f}")
        
        print("\n=== Combustion Results (per hour) ===\n")
        for k, v in results.items():
            print(f"{k}: {v:,.2f}")
    
    def _validate_inputs(self):
        """Validate input parameters."""
        total = sum(self._ultimate_analysis.values())
        if abs(total - 100.0) > 0.01:
            raise ValueError(f"Total mass fraction must equal 100%, current value is {total}")
    
    def _moles_to_lbs(self, mol, mw):
        """Convert moles to pounds."""
        return mol * mw / 453.592
    
    # All the calculation methods from the original functions would go here
    # I'll include a few key ones to demonstrate the pattern
    
    def _calculate_corrected_hhv(self):
        """Calculate corrected HHV using Dulong's formula."""
        C = self._ultimate_analysis.get('C', 0.0) / 100.0
        H = self._ultimate_analysis.get('H', 0.0) / 100.0
        O = self._ultimate_analysis.get('O', 0.0) / 100.0
        S = self._ultimate_analysis.get('S', 0.0) / 100.0
        Ash = self._ultimate_analysis.get('Ash', 0.0) / 100.0
        Moisture = self._ultimate_analysis.get('Moisture', 0.0) / 100.0

        HHV_dry_MJ_per_kg = 33.8 * C + 144.2 * (H - O / 8) + 9.4 * S
        dry_fraction = 1 - Moisture - Ash
        HHV_corrected_MJ_per_kg = HHV_dry_MJ_per_kg * dry_fraction
        HHV_corrected_BTU_per_lb = HHV_corrected_MJ_per_kg * 429.923

        return HHV_corrected_BTU_per_lb
    
    def _calculate_humidity_ratio(self):
        """Calculate humidity ratio of air."""
        INHGC_TO_PA_FACTOR = 3386.39 
        EPSILON = 0.62198 

        temperature_k = (self._air_temp_F - 32) * 5/9 + 273.15
        pressure_pa = self._atm_press_inHg * INHGC_TO_PA_FACTOR
        temperature_c = (self._air_temp_F - 32) * 5/9  
        
        saturation_vapor_pressure_pa = 611.21 * np.exp((18.678 - (temperature_c / 234.5)) * (temperature_c / (257.14 + temperature_c)))
        relative_humidity_ratio = self._air_RH_pct / 100.0  
        actual_vapor_pressure_pa = relative_humidity_ratio * saturation_vapor_pressure_pa
        humidity_ratio = (EPSILON * actual_vapor_pressure_pa) / (pressure_pa - actual_vapor_pressure_pa)

        return humidity_ratio
    
    def _calculate_moles_from_scf_humid_air(self, humidity_ratio, standard_temp_f=60.0, standard_pressure_psia=14.7):
        """Calculate moles of O2, N2, and H2O from humid air SCF."""
        standard_temp_r = standard_temp_f + 459.67
        molar_volume_scf = (10.731 * standard_temp_r) / standard_pressure_psia
        total_moles_humid_air = self._air_scfh / molar_volume_scf
        moles_dry_air = total_moles_humid_air * (1 - humidity_ratio)
        moles_h2o = total_moles_humid_air * humidity_ratio
        moles_o2 = moles_dry_air * 0.21
        moles_n2 = moles_dry_air * 0.79

        return {'O2': moles_o2, 'N2': moles_n2, 'H2O': moles_h2o}
    
    def _calculate_ash_free_moles(self):
        """Calculate ash-free moles of coal elements."""
        ultimate_analysis_frac = {key: value / 100 for key, value in self._ultimate_analysis.items()}
        daf_factor = 1 - ultimate_analysis_frac['Ash'] - ultimate_analysis_frac['Moisture']
        
        if daf_factor <= 0:
            raise ValueError("Invalid input: (Ash + Moisture) >= 100%. Cannot calculate DAF basis.")

        ash_free_basis = {}
        for element in ['C', 'H', 'O', 'N', 'S']:
            ash_free_basis[element] = ultimate_analysis_frac[element] / daf_factor

        coal_ash_free_pounds = self._coal_lb_per_hr * (1 - ultimate_analysis_frac['Ash'])

        moles_ash_free = {}
        for element, percentage in ash_free_basis.items():
            moles_ash_free[element] = percentage * coal_ash_free_pounds / self.MW[element]

        # Add moisture
        moles_ash_free['Moisture'] = (ultimate_analysis_frac['Moisture'] * self._coal_lb_per_hr) / self.MW['H2O']

        return moles_ash_free
    
    def _estimate_fuel_NO(self, fuel_N_moles, T_flame_K=2000.0):
        """Estimate NO from fuel-bound nitrogen."""
        T_ref = 2000
        temp_factor = (T_flame_K / T_ref) ** 0.5
        adjusted_NOx_eff = self._NOx_eff * temp_factor
        adjusted_NOx_eff = min(adjusted_NOx_eff, 0.8)
        return fuel_N_moles * adjusted_NOx_eff
    
    def _estimate_thermal_NO(self, O2_mole_frac, N2_mole_frac, T_flame_K, residence_time_s=2.0):
        """Estimate thermal NO formation using Zeldovich mechanism."""
        A1 = 1.8e14
        E1 = 76000
        R = 1.987
        
        k1f = A1 * np.exp(-E1 / (R * T_flame_K))
        O_equilibrium = 1e-6 * np.sqrt(O2_mole_frac) * np.exp(-59000 / (R * T_flame_K))
        thermal_NO_rate = k1f * N2_mole_frac * O_equilibrium * residence_time_s
        
        return min(thermal_NO_rate, 1e-4)
    
    def _co2_mole_fraction(self, excess_air_pct):
        """Estimate CO2 mole fraction based on excess air."""
        def co_co2_ratio(excess_air_pct):
            if excess_air_pct <= 0:
                return 0.5
            elif excess_air_pct < 10:
                return 0.1 / excess_air_pct
            elif excess_air_pct < 25:
                return 0.01 / (excess_air_pct - 5)
            else:
                return 0.0005

        r = co_co2_ratio(excess_air_pct)
        return 1.0 / (1.0 + r)
    
    def _simple_flame_temperature_estimate(self, HHV_BTU_per_lb):
        """Simple flame temperature estimate."""
        T_flame_estimate = 1200 + (HHV_BTU_per_lb - 8000) * 0.15
        return max(T_flame_estimate, 1200)
    
    def _calculate_combustion_efficiency_and_heat_release(self, coal_dry_ash_free_pph, HHV_BTU_per_lb, CO2_frac):
        """Calculate combustion efficiency and heat release."""
        CO_frac = 1.0 - CO2_frac
        combustion_efficiency = CO2_frac + 0.28 * CO_frac
        actual_heat_released_btu_per_hr = HHV_BTU_per_lb * coal_dry_ash_free_pph * combustion_efficiency
        return combustion_efficiency, actual_heat_released_btu_per_hr
    
    def _calculate_flue_gas_composition(self, ash_free_wet_coal_composition, wet_air_composition, co2_factor, product_no_moles):
        """Calculate combustion products in flue gas."""
        C_coal = ash_free_wet_coal_composition.get('C', 0)
        H_coal = ash_free_wet_coal_composition.get('H', 0)
        O_coal = ash_free_wet_coal_composition.get('O', 0)
        N_coal = ash_free_wet_coal_composition.get('N', 0)
        S_coal = ash_free_wet_coal_composition.get('S', 0)
        H2O_coal = ash_free_wet_coal_composition.get('Moisture', 0)
        
        N2_air = wet_air_composition.get('N2', 0)
        O2_air = wet_air_composition.get('O2', 0)
        H2O_air = wet_air_composition.get('H2O', 0)
        
        CO2_moles = co2_factor * C_coal
        CO_moles = (1 - co2_factor) * C_coal
        
        O2_for_C = CO2_moles + 0.5 * CO_moles
        O2_for_H = 0.25 * H_coal
        O2_for_S = S_coal
        O2_available = O2_air + 0.5 * O_coal
        
        products = {}
        products['CO2'] = CO2_moles
        products['CO'] = CO_moles
        products['H2O'] = 0.5 * H_coal + H2O_coal + H2O_air
        products['SO2'] = S_coal
        products['NO'] = product_no_moles
        
        N2_consumed_for_NO = 0.5 * product_no_moles
        products['N2'] = N2_air + 0.5 * N_coal - N2_consumed_for_NO
        
        O2_consumed = O2_for_C + O2_for_H + O2_for_S + 0.5 * product_no_moles
        if O2_consumed > O2_available:
            raise ValueError(f'Not Enough O2 Available, Required Moles {O2_consumed:0.1f}, Available {O2_available:0.1f}')
        products['O2'] = O2_available - O2_consumed
        
        for key in products:
            products[key] = max(0, products[key])
        
        return products
    
    def _temperature_dependent_Cp(self, species, T):
        """Temperature-dependent Cp using thermo library."""
        Cp_data = {
            'CO2': {'CAS': '124-38-9', 'MW': 44.01},
            'H2O': {'CAS': '7732-18-5', 'MW': 18.02},
            'SO2': {'CAS': '7446-09-5', 'MW': 64.07},
            'CO':  {'CAS': '630-08-0', 'MW': 28.01},
            'O2':  {'CAS': '7782-44-7', 'MW': 32.00},
            'N2':  {'CAS': '7727-37-9', 'MW': 28.02},
            'NO':  {'CAS': '10102-43-9', 'MW': 30.01}
        }
        
        if species in Cp_data:
            try:
                CpGas = thermo.heat_capacity.HeatCapacityGas(CASRN=Cp_data[species]['CAS'],
                                                           MW=Cp_data[species]['MW'])
                return CpGas(T)  # J/mol/K
            except:
                # Fallback to polynomial method if thermo library fails
                return self._polynomial_Cp(species, T)
        else:
            return 30.0  # Default fallback value
    
    def _polynomial_Cp(self, species, T):
        """Polynomial approximation for Cp when thermo library fails."""
        Cp_data = {
            'CO2': lambda T: 22.26 + 5.981e-2 * T[0] - 3.501e-5 * T[1] + 7.469e-9 * T[2],
            'H2O': lambda T: 30.092 + 6.832e-1 * T[0] - 6.793e-4 * T[1] + 2.534e-7 * T[2],
            'SO2': lambda T: 24.997 + 5.914e-2 * T[0] - 3.281e-5 * T[1] + 6.089e-9 * T[2],
            'CO':  lambda T: 25.567 + 6.096e-2 * T[0] - 4.055e-5 * T[1] + 9.104e-9 * T[2],
            'O2':  lambda T: 31.322 + -2.755e-3 * T[0] + 4.551e-6 * T[1] - 3.211e-9 * T[2],
            'N2':  lambda T: 28.986 + 1.853e-2 * T[0] - 9.647e-6 * T[1] + 1.312e-9 * T[2],
            'NO':  lambda T: 30.752 + 9.630e-3 * T[0] - 1.292e-6 * T[1] + 4.800e-10 * T[2]
        }
        
        if species in Cp_data:
            Tp = [T, T*T, T*T*T]
            return Cp_data[species](Tp)
        else:
            return 30.0
    
    def _estimate_flame_temp_Cp_method(self, products_mol, HHV_BTU_per_lb, coal_mass_lb_per_hr, CO2_frac, T_ref_K=298.15, debug=False):
        """
        Estimate adiabatic flame temperature using temperature-dependent Cp values.
        """
        CO_frac = 1.0 - CO2_frac
        combustion_efficiency = CO2_frac + 0.28 * CO_frac
        
        # Convert HHV from BTU/lb to J per hour for the given coal flow rate
        actual_heat_released_J_per_hr = HHV_BTU_per_lb * coal_mass_lb_per_hr * 1055.06 * combustion_efficiency
        
        def energy_balance(T):
            """Energy balance equation: Heat input - Heat absorbed by products = 0"""
            total_energy_absorbed = 0.0
            for species, n_moles_per_hr in products_mol.items():
                if n_moles_per_hr > 0:
                    T_mid = (T + T_ref_K) / 2
                    Cp_avg = self._temperature_dependent_Cp(species, T_mid)
                    energy_absorbed = n_moles_per_hr * Cp_avg * (T - T_ref_K)
                    total_energy_absorbed += energy_absorbed
            
            return actual_heat_released_J_per_hr - total_energy_absorbed
        
        if debug:
            # Debugging: plot energy balance vs temperature
            T_range = np.linspace(500, 4000, 100)
            energy_values = [energy_balance(T) for T in T_range]
            
            plt.figure(figsize=(10, 6))
            plt.plot(T_range, energy_values, 'b-', linewidth=2)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            plt.xlabel('Temperature (K)')
            plt.ylabel('Energy Balance (J/hr)')
            plt.title('Energy Balance vs Temperature\n(Zero crossing = Flame Temperature)')
            plt.grid(True, alpha=0.3)
            plt.show()
       
        try:
            # Find the temperature where energy balance equals zero
            T_flame = opt.brentq(energy_balance, 800, 4000)
            return T_flame
        except ValueError as e:
            print(f"Warning: Could not solve for flame temperature. Error: {e}")
            print("Using simplified estimate based on fuel heating value")
            T_flame_estimate = self._simple_flame_temperature_estimate(HHV_BTU_per_lb * combustion_efficiency)
            return max(T_flame_estimate, 1200)


# Example usage and demonstration
if __name__ == "__main__":
    # Sample ultimate analysis (mass percent)
    ultimate = {
        'C': 72.0,
        'H': 5.0,
        'O': 10.0,
        'N': 1.5,
        'S': 1.0,
        'Ash': 8.0,
        'Moisture': 2.5
    }

    # Create model instance
    model = CoalCombustionModel(
        ultimate_analysis=ultimate,
        coal_lb_per_hr=10000,     # lb/hr of coal
        air_scfh=2000000,         # SCFH of air
        NOx_eff=0.35              # NOx efficiency
    )

    # Perform calculations
    print("=== COAL COMBUSTION MODEL - CLASS IMPLEMENTATION ===")
    model.calculate(debug=True)

    # Access individual properties
    print(f"\n=== ACCESSING INDIVIDUAL PROPERTIES ===")
    print(f"Coal HHV: {model.coal_HHV_btu_per_lb:,.0f} BTU/lb")
    print(f"Flame Temperature: {model.flame_temp_F:,.0f} °F")
    print(f"CO2 Emissions: {model.CO2_lb_per_hr:,.0f} lb/hr")
    print(f"NOx Emissions: {model.NO_total_lb_per_hr:,.2f} lb/hr")
    print(f"Combustion Efficiency: {model.combustion_efficiency:.3f}")

    # Demonstrate property changes trigger recalculation
    print(f"\n=== TESTING PROPERTY CHANGES ===")
    print(f"Original NOx efficiency: {model.NOx_eff}")
    print(f"Original NOx emissions: {model.NO_total_lb_per_hr:.2f} lb/hr")
    
    # Change NOx efficiency
    model.NOx_eff = 0.50
    print(f"New NOx efficiency: {model.NOx_eff}")
    print(f"New NOx emissions: {model.NO_total_lb_per_hr:.2f} lb/hr")
    
    # Change coal flow rate
    original_coal_rate = model.coal_lb_per_hr
    model.coal_lb_per_hr = 15000
    print(f"\nCoal rate changed from {original_coal_rate:,.0f} to {model.coal_lb_per_hr:,.0f} lb/hr")
    print(f"New CO2 emissions: {model.CO2_lb_per_hr:,.0f} lb/hr")
    print(f"New heat release: {model.heat_released_btu_per_hr:,.0f} BTU/hr")