import numpy as np
import thermo
import scipy.optimize as opt
import matplotlib.pyplot as plt # for debugging purposes

def calculate_corrected_hhv(ultimate):
    """
    Calculate the corrected Higher Heating Value (HHV) of coal using Dulong's formula,
    accounting for moisture and ash content.

    Parameters:
    ultimate (dict): Dictionary containing ultimate analysis with keys 'C', 'H', 'O', 'S', 'Ash', 'Moisture'

    Returns:
    float: Corrected HHV in BTU/lb
    """
    # Extract relevant elements from the dictionary and convert to mass fractions
    C = ultimate.get('C', 0.0) / 100.0
    H = ultimate.get('H', 0.0) / 100.0
    O = ultimate.get('O', 0.0) / 100.0
    S = ultimate.get('S', 0.0) / 100.0
    Ash = ultimate.get('Ash', 0.0) / 100.0
    Moisture = ultimate.get('Moisture', 0.0) / 100.0

     # Dulong's formula in MJ/kg (dry basis)
    HHV_dry_MJ_per_kg = 33.8 * C + 144.2 * (H - O / 8) + 9.4 * S

    # Correct for moisture and ash content
    dry_fraction = 1 - Moisture - Ash
    HHV_corrected_MJ_per_kg = HHV_dry_MJ_per_kg * dry_fraction

    # Convert MJ/kg to BTU/lb (1 MJ/kg = 429.923 BTU/lb)
    HHV_corrected_BTU_per_lb = HHV_corrected_MJ_per_kg * 429.923

    return HHV_corrected_BTU_per_lb

import numpy as np

def calculate_humidity_ratio(temperature_f, relative_humidity_percent, pressure_inhg):
    """
    Calculates the humidity ratio (mixing ratio) of air.

    Args:
        temperature_f (float): Air temperature in degrees Fahrenheit.
        relative_humidity_percent (float): Relative humidity in percent (0-100).
        pressure_inhg (float): Atmospheric pressure in inches of mercury (inHg).

    Returns:
        float: Humidity ratio in kg_water/kg_dry_air.
    """

    # Constants
    # Conversion factor from inHg to Pascals
    INHGC_TO_PA_FACTOR = 3386.39 
    # Specific gas constant for dry air in J/(kg*K)
    R_DRY_AIR = 287.058 
    # Ratio of molar masses of water vapor to dry air
    EPSILON = 0.62198 

    # 1. Convert temperature from Fahrenheit to Kelvin
    # Formula: K = (F - 32) * 5/9 + 273.15
    temperature_k = (temperature_f - 32) * 5/9 + 273.15

    # 2. Convert pressure from inHg to Pascals
    pressure_pa = pressure_inhg * INHGC_TO_PA_FACTOR

    # 3. Calculate Saturation Vapor Pressure (ps) using the Arden Buck Equation (an approximation)
    # The Arden Buck equation is a common formula for estimating saturation vapor pressure. 
    # Since the previous response used a simplified formula, let's use a more accurate one
    # Note: T in this formula should be in Celsius
    temperature_c = (temperature_f - 32) * 5/9  
    
    # Arden Buck Equation (often used in psychrometrics)
    # This specific version is suitable for temperature in Celsius
    saturation_vapor_pressure_pa = 611.21 * np.exp((18.678 - (temperature_c / 234.5)) * (temperature_c / (257.14 + temperature_c)))

    # 4. Calculate Actual Vapor Pressure (pa)
    # Convert relative humidity from percentage to a decimal
    relative_humidity_ratio = relative_humidity_percent / 100.0  
    actual_vapor_pressure_pa = relative_humidity_ratio * saturation_vapor_pressure_pa

    # 5. Calculate Humidity Ratio (w)
    # Formula: w = (epsilon * actual_vapor_pressure) / (total_pressure - actual_vapor_pressure)
    # where epsilon is the ratio of molar masses of water vapor to dry air (approximately 0.622)
    humidity_ratio = (EPSILON * actual_vapor_pressure_pa) / (pressure_pa - actual_vapor_pressure_pa)

    return humidity_ratio



def estimate_thermal_NO(O2_mole_frac, N2_mole_frac, T_flame_K, residence_time_s=2.0):
    """
    Estimate thermal NO formation using Zeldovich mechanism.
    
    The Zeldovich mechanism for thermal NOx:
    N2 + O ⇌ NO + N
    N + O2 ⇌ NO + O
    N + OH ⇌ NO + H
    
    Parameters:
        O2_mole_frac (float): Oxygen mole fraction in the flame region.
        N2_mole_frac (float): Nitrogen mole fraction in the flame region.
        T_flame_K (float): Flame temperature in Kelvin.
        residence_time_s (float): Residence time in high temperature zone (seconds).

    Returns:
        float: Thermal NO formation rate in mol NO per mol N2 per second.
    """
    # Zeldovich rate constants (simplified Arrhenius forms)
    # k1f = A1 * exp(-E1/RT) for N2 + O -> NO + N
    A1 = 1.8e14  # Pre-exponential factor
    E1 = 76000   # Activation energy (cal/mol)
    R = 1.987    # Gas constant (cal/mol/K)
    
    # Rate constant for the rate-limiting step
    k1f = A1 * np.exp(-E1 / (R * T_flame_K))
    
    # Equilibrium oxygen atom concentration (very simplified)
    # [O] ∝ sqrt([O2]) * exp(-ΔH_dissociation/RT)
    O_equilibrium = 1e-6 * np.sqrt(O2_mole_frac) * np.exp(-59000 / (R * T_flame_K))
    
    # Thermal NOx formation rate (mol NO/mol N2/s)
    # Rate ≈ k1f * [N2] * [O] * residence_time
    thermal_NO_rate = k1f * N2_mole_frac * O_equilibrium * residence_time_s
    
    # Limit to reasonable values (thermal NOx is typically small)
    return min(thermal_NO_rate, 1e-4)  # Max 0.01% conversion per second

def estimate_fuel_NO(fuel_N_moles, NOx_eff, T_flame_K):
    """
    Estimate fuel NOx formation with temperature dependence.
    
    Parameters:
        fuel_N_moles (float): Moles of nitrogen in fuel per hour.
        NOx_eff (float): Base NOx conversion efficiency at reference conditions.
        T_flame_K (float): Flame temperature in Kelvin.
    
    Returns:
        float: Fuel NOx formation in mol NO per hour.
    """
    # Temperature correction factor for fuel NOx
    # Fuel NOx increases with temperature but less dramatically than thermal NOx
    T_ref = 2000  # Reference temperature (K)
    temp_factor = (T_flame_K / T_ref) ** 0.5  # Square root dependence
    
    # Adjusted NOx efficiency
    adjusted_NOx_eff = NOx_eff * temp_factor
    
    # Limit to reasonable range
    adjusted_NOx_eff = min(adjusted_NOx_eff, 0.8)  # Max 80% conversion
    
    return fuel_N_moles * adjusted_NOx_eff

def temperature_dependent_Cp(species, T):
    """
    Return an approximate temperature-dependent Cp value (J/mol·K) for a given species at temperature T (K).
    Uses simplified polynomial fits or linear models from NASA polynomials (approximated).
    """
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
        return 30.0  # Default fallback value
    
def thermo_temperature_dependent_Cp(species, T, P=1.4e5):
    """
    Return an approximate temperature-dependent Cp value (J/mol/K) for a given species at temperature T (K).
    Using thermo library's Pure Gas Heat Capacity class.
    """
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
            return CpGas(T) # J/mol/K
        except:
            # Fallback to polynomial method if thermo library fails
            return temperature_dependent_Cp(species, T)
    else:
        return 30.0  # Default fallback value

def estimate_flame_temp_Cp_method(products_mol, HHV_BTU_per_lb, coal_mass_lb_per_hr=1.0, CO2_frac=0.9, T_ref_K=298.15, debug=False):
    """
    Estimate adiabatic flame temperature using temperature-dependent Cp values.
    Solves the energy balance: Actual_Heat_Released = ∑ ni * ∫Cp(T)dT from T_ref to T_flame.

    Parameters:
        products_mol (dict): Molar amounts of combustion products per hour.
        HHV_BTU_per_lb (float): Higher heating value in BTU/lb (complete combustion basis).
        coal_mass_lb_per_hr (float): Coal mass flow rate in lb/hr (for proper energy scaling).
        CO2_frac (float): Fraction of carbon converted to CO2 (affects actual heat release).
        T_ref_K (float): Reference temperature in Kelvin (default: 298.15 K).
        debug (bool): Enable debugging plots and output.

    Returns:
        float: Estimated flame temperature in Kelvin.
    """
    # Calculate actual heat released based on combustion efficiency
    # CO formation releases much less energy than CO2 formation
    # Heat of formation: C + O2 -> CO2 releases ~393.5 kJ/mol
    # Heat of formation: C + 0.5*O2 -> CO releases ~110.5 kJ/mol
    # So CO formation releases about 28% of the energy of CO2 formation
    
    CO_frac = 1.0 - CO2_frac
    combustion_efficiency = CO2_frac + 0.28 * CO_frac  # Weight CO formation at ~28% of CO2 energy
    
    # Convert HHV from BTU/lb to J per hour for the given coal flow rate, adjusted for actual combustion
    actual_heat_released_J_per_hr = HHV_BTU_per_lb * coal_mass_lb_per_hr * 1055.06 * combustion_efficiency
    
    def energy_balance(T):
        """
        Energy balance equation: Heat input - Heat absorbed by products = 0
        """
        total_energy_absorbed = 0.0
        for species, n_moles_per_hr in products_mol.items():
            if n_moles_per_hr > 0:  # Only consider species that are actually present
                # Use average Cp between reference and flame temperature
                T_mid = (T + T_ref_K) / 2
                Cp_avg = thermo_temperature_dependent_Cp(species, T_mid)
                # Energy absorbed by this species
                energy_absorbed = n_moles_per_hr * Cp_avg * (T - T_ref_K)
                total_energy_absorbed += energy_absorbed
        
        # Return the imbalance (should be zero at equilibrium)
        return actual_heat_released_J_per_hr - total_energy_absorbed
    
    if debug:
        # Debugging: plot energy balance vs temperature
        T_range = np.linspace(500, 3500, 100)
        energy_values = [energy_balance(T) for T in T_range]
        
        plt.figure(figsize=(10, 6))
        plt.plot(T_range, energy_values, 'b-', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.xlabel('Temperature (K)')
        plt.ylabel('Energy Balance (J/hr)')
        plt.title('Energy Balance vs Temperature\n(Zero crossing = Flame Temperature)')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"Heat input (adjusted for combustion efficiency): {actual_heat_released_J_per_hr/1055.06 :.2e} Btu/hr")
        print(f"Combustion efficiency factor: {combustion_efficiency:.3f} (CO2 frac: {CO2_frac:.2f})")
        print(f"Products composition (mol/hr): {products_mol}")
    
    try:
        # Find the temperature where energy balance equals zero
        # Use a reasonable range for combustion temperatures
        T_flame = opt.brentq(energy_balance, 800, 3500)  # 800-3500 K range
        return T_flame
    except ValueError as e:
        print(f"Warning: Could not solve for flame temperature. Error: {e}")
        print("Using simplified estimate based on fuel heating value")
        # Fallback: simplified estimation based on typical combustion temperatures
        # Higher heating value correlates roughly with flame temperature
        T_flame_estimate = 1200 + (HHV_BTU_per_lb - 8000) * 0.15  # Empirical relationship
        return max(T_flame_estimate, 1200)  # Minimum reasonable flame temp

def coal_combustion_from_mass_flow(ultimate, coal_lb_per_hr, air_scfh, CO2_frac=0.9, NOx_eff=0.35,
                                   air_temp_F = 75.0, air_RH_pct = 55.0, atm_press_inHg = 30.25, debug = False):

    """
    Estimate combustion products and emissions based on coal feed rate and combustion air input.

    Parameters:
        ultimate (dict): Ultimate analysis of coal as a dictionary with keys:
            - 'C': Carbon (% by mass)
            - 'H': Hydrogen (% by mass)
            - 'O': Oxygen (% by mass)
            - 'N': Nitrogen (% by mass)
            - 'S': Sulfur (% by mass)
            - 'Moisture': Moisture content (% by mass)
        coal_lb_per_hr (float): Mass flow rate of coal in pounds per hour (lb/hr).
        air_scfh (float): Volumetric flow rate of combustion air in standard cubic feet per hour (SCFH).
        CO2_frac (float, optional): Fraction of carbon converted to CO₂ during combustion. Default is 0.9.
        NOx_eff (float, optional): Efficiency of conversion from fuel-bound nitrogen to NOx. Default is 0.35.
        air_temp_F (float, optional): Temperature of incoming air in degrees Fahrenheit. Default is 75.0°F.
        air_RH_pct (float, optional): Relative humidity of incoming air as a percentage. Default is 55.0%.
        atm_press_inHg (float, optional): Atmospheric pressure in inches of mercury (inHg). Default is 30.25 inHg.

    Returns:
        dict: Dictionary containing estimated hourly emissions and combustion products, including:
            - CO₂ emissions (lb/hr)
            - NOx emissions (lb/hr)
            - SO₂ emissions (lb/hr)
            - H₂O from combustion and moisture (lb/hr)
            - Excess O₂ and N₂ (lb/hr)
            - Other relevant combustion byproducts
    """

    MW = {
        'C': 12.01,
        'H': 1.008,
        'O': 16.00,
        'N': 14.01,
        'S': 32.07,
        'CO2': 44.01,
        'CO': 28.01,
        'H2O': 18.02,
        'SO2': 64.07,
        'O2': 32.00,
        'N2': 28.02,
        'NO': 30.01
    }

    # calculate corrected HHV
    HHV_btu_per_lb = calculate_corrected_hhv(ultimate)

    # Dry Air composition (mole fraction)
    air_O2_frac = 0.21
    air_N2_frac = 0.79
    air_mol_per_scf = 1.0 / 379.0  # lbmols/scf at STP

    # Humidity ratio
    air_humidity_ratio = calculate_humidity_ratio(air_temp_F, air_RH_pct, atm_press_inHg)

    # Convert ultimate analysis to mass fractions
    total = sum(ultimate.values())
    # check if total mass fraction is 100%
    if total != 100.0:
        raise ValueError(f"Total mass fraction must equal 100% value is {total}")

    frac = {k: v / total for k, v in ultimate.items()}
    coal_dry_frac = 1.0 - frac.get('Moisture', 0)

    coal_dry_lb_per_hr = coal_lb_per_hr * coal_dry_frac

    # Moles of elements per lb of dry coal
    mol_C = frac['C'] / MW['C']
    mol_H = frac['H'] / MW['H']
    mol_O = frac['O'] / MW['O']
    mol_S = frac['S'] / MW['S']
    mol_N = frac['N'] / MW['N']

    total_mol_C = mol_C * coal_dry_lb_per_hr
    total_mol_H = mol_H * coal_dry_lb_per_hr
    total_mol_O = mol_O * coal_dry_lb_per_hr
    total_mol_S = mol_S * coal_dry_lb_per_hr
    total_mol_N = mol_N * coal_dry_lb_per_hr

    CO_frac = 1 - CO2_frac
    mol_CO2 = total_mol_C * CO2_frac
    mol_CO = total_mol_C * CO_frac

    mol_O2_required = mol_CO2 + mol_CO * 0.5 + total_mol_H / 4 + total_mol_S - total_mol_O / 2

    mol_wet_air = air_scfh * air_mol_per_scf # approximate based on the very small amount of humidity in normal air
    mol_dry_air = (1.0 - air_humidity_ratio) * mol_wet_air
    mol_O2_actual = mol_dry_air * air_O2_frac
    mol_N2_air = mol_dry_air * air_N2_frac
    mol_H2O_air = mol_wet_air - mol_dry_air
    equivalence_ratio = mol_O2_actual / mol_O2_required # could be used for flame temp

    mol_O2_excess = mol_O2_actual - mol_O2_required
    mol_H2O = total_mol_H / 2 + mol_H2O_air

    # Initial products for flame temperature calculation (before NOx correction)
    products_mol_initial = {
        'CO2': mol_CO2,
        'CO': mol_CO,
        'H2O': mol_H2O,
        'SO2': total_mol_S,
        'N2': mol_N2_air + total_mol_N,  # Assume all N goes to N2 initially
        'O2': mol_O2_excess,
        'NO': 0  # No NOx assumed for flame temperature calculation
    }

    # Calculate flame temperature first (NOx formation doesn't significantly affect energy balance)
    T_flame_K = estimate_flame_temp_Cp_method(products_mol_initial, HHV_btu_per_lb, coal_dry_lb_per_hr, CO2_frac)
    # Now calculate NOx formation based on the calculated flame temperature
    total_moles = sum(products_mol_initial.values())
    O2_mole_frac = mol_O2_excess / total_moles if total_moles > 0 else 0.05
    N2_mole_frac = (mol_N2_air + total_mol_N) / total_moles if total_moles > 0 else 0.75
    
    # Fuel NOx (temperature-dependent)
    mol_NO_fuel = estimate_fuel_NO(total_mol_N, NOx_eff, T_flame_K)
    
    # Thermal NOx (strongly temperature-dependent)
    thermal_NO_rate = estimate_thermal_NO(O2_mole_frac, N2_mole_frac, T_flame_K)
    mol_NO_thermal = thermal_NO_rate * (mol_N2_air + total_mol_N)  # mol NO per hour
    
    # Total NOx
    mol_NO_total = mol_NO_fuel + mol_NO_thermal
    
    # Update N2 balance (subtract NOx formation)
    mol_N2_total_corrected = mol_N2_air + (total_mol_N - mol_NO_total)
    
    # Final product composition
    products_mol = {
        'CO2': mol_CO2,
        'CO': mol_CO,
        'H2O': mol_H2O,
        'SO2': total_mol_S,
        'N2': mol_N2_total_corrected,
        'O2': mol_O2_excess,
        'NO': mol_NO_total
    }

    def moles_to_lbs(mol, mw): return mol * mw / 453.592

    inputs = {
        'Air (scfh)': air_scfh,
        'Coal (pph)': coal_lb_per_hr,
        'CO2 Fraction': CO2_frac,
        'NOx Conversion Efficiency': NOx_eff,
        'Air temperature (F)': air_temp_F,
        'Relative Humidity (%)': air_RH_pct,
        'Atmospheric Pressure (inHg)': atm_press_inHg,
    }
    interim_calcs = {
        'Coal HHV (Btu/lb)': HHV_btu_per_lb,
        'Humidity Ratio (h2o/dry air by mass)': air_humidity_ratio,
        'Equivalence Ratio': equivalence_ratio
    }

    if debug:
        print("\n=== Inputs ===\n")
        for k, v in inputs.items():
            print(f"{k}: {v:,.2f}")
        print("\n=== Interim Calculations ===\n")
        for k, v in interim_calcs.items():
            print(f"{k}: {v:,.4f}")

    results = {
        'CO2 (LB)': moles_to_lbs(mol_CO2, MW['CO2']),
        'CO (LB)': moles_to_lbs(mol_CO, MW['CO']),
        'H2O (LB)': moles_to_lbs(mol_H2O, MW['H2O']),
        'SO2 (LB)': moles_to_lbs(total_mol_S, MW['SO2']),
        'NO (LB from fuel-N)': moles_to_lbs(mol_NO_fuel, MW['NO']),
        'NO (LB thermal)': moles_to_lbs(mol_NO_thermal, MW['NO']),
        'NO (LB total)': moles_to_lbs(mol_NO_total, MW['NO']),
        'N2 (LB)': moles_to_lbs(mol_N2_total_corrected, MW['N2']),
        'O2 (excess %)': moles_to_lbs(mol_O2_excess, MW['O2']),
        'Heat Released (BTU/hr)': HHV_btu_per_lb * coal_dry_frac * coal_lb_per_hr * (CO2_frac + 0.3 * CO_frac),
        'Estimated Flame Temp (F)': (T_flame_K - 273.15) * 9/5 + 32 # convert to Fahrenheit
    }

    results['Total Flue Gas (lb/hr)'] = sum(v for k, v in results.items() if k not in ['Heat Released (BTU/hr)', 'Estimated Flame Temp (F)', 'Total Flue Gas (lb/hr)'])
    return results

# Example usage:
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

    # Input conditions
    coal_rate_lb_hr = 10000    # lb/hr of coal
    air_flow_scfh = 1800000    # standard cubic feet per hour of air

    print("=== TESTING COMBUSTION EFFICIENCY EFFECT ON FLAME TEMPERATURE ===\n")
    
    # Test different CO2 fractions to show effect on flame temperature
    CO2_fractions = [0.70, 0.80, 0.90, 0.95, 0.99]
    
    for co2_frac in CO2_fractions:
        results = coal_combustion_from_mass_flow(
            ultimate,
            coal_lb_per_hr=coal_rate_lb_hr,
            air_scfh=air_flow_scfh,
            CO2_frac=co2_frac,
            NOx_eff=0.35
        )
        print(f"CO2 Fraction: {co2_frac:.2f} -> Flame Temp: {results['Estimated Flame Temp (F)']:,.0f}°F, "
              f"Fuel NOx: {results['NO (LB from fuel-N)']:,.1f} lb/hr, "
              f"Thermal NOx: {results['NO (LB thermal)']:,.1f} lb/hr")
    
    print(2*'\n')
    print(f"\n=== DETAILED RESULTS FOR GIVEN SIMULATION PARAMETERS ===")
    
    # Run combustion model with standard conditions
    results = coal_combustion_from_mass_flow(
        ultimate,
        coal_lb_per_hr=coal_rate_lb_hr,
        air_scfh=air_flow_scfh,
        CO2_frac=0.9,
        NOx_eff=0.35,
        debug=True,
    )

    # Output results
    print("\n=== Combustion Results (per hour) ===\n")
    for k, v in results.items():
        print(f"{k}: {v:,.2f}")
