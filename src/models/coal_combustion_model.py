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



def estimate_thermal_NO(lb_HHV, O2_mole_frac=0.05, T_flame_K=2000):
    """
    Estimate thermal NO formation using an empirical Zeldovich-based model.

    Parameters:
        lb_HHV (float): Heat of combustion in BTU per lb of fuel.
        O2_mole_frac (float): Oxygen mole fraction in the flame region.
        T_flame_K (float): Flame temperature in Kelvin.

    Returns:
        float: Estimated thermal NO in lb per lb of coal.
    """
    B = 1.6e5
    C = 38000
    m = 0.5
    lb_NO_per_MMBTU = B * (O2_mole_frac ** m) * np.exp(-C / T_flame_K)
    lb_NO_per_lb_coal = lb_NO_per_MMBTU * (lb_HHV / 1e6)
    return lb_NO_per_lb_coal

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
    Using thermo library's Pure Gas Heact Capacity class.
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
        # print(f"Calculating Cp for {species} at {T} K")
        CpGas = thermo.heat_capacity.HeatCapacityGas(CASRN=Cp_data[species]['CAS'],
                                                     MW=Cp_data[species]['MW'],
                                                     method='POLING_POLY')
        return CpGas(T) # J/mol/K

def estimate_flame_temp_Cp_method(products_mol, HHV_BTU_per_lb, T_ref_K=298.15):
    """
    Estimate adiabatic flame temperature using temperature-dependent Cp values.
    Solves the energy balance: HHV = ∑ ni * ∫Cp(T)dT from T_ref to T_flame.

    Parameters:
        products_mol (dict): Molar amounts of combustion products.
        HHV_BTU_per_lb (float): Higher heating value in BTU/lb.
        T_ref_K (float): Reference temperature in Kelvin (default: 298.15 K).

    Returns:
        float: Estimated flame temperature in Kelvin.
    """
    HHV_J = HHV_BTU_per_lb * 1055.06 # convert BTU to Joules

    def energy_balance(T):
        total_energy = 0.0
        for sp, n in products_mol.items():
            T_mid = (T + T_ref_K) / 2
            Cp_avg = thermo_temperature_dependent_Cp(sp, T_mid)
            total_energy += n * Cp_avg * (T - T_ref_K)
        return total_energy - HHV_J
    
    
    tplot = np.linspace(100, 4000, 100)  # Debugging range for temperature
    energy_values = np.array([energy_balance(T) for T in tplot]) - HHV_J  # Calculate energy balance for debugging
    plt.plot(energy_values,tplot)  # Debugging plot
    plt.xlabel('Energy Balance (J)')
    plt.ylabel('Temperature (K)')
    plt.title('Energy Balance vs Temperature')
    plt.grid()
    plt.show()  # Show the plot for debugging

    T_flame = opt.brentq(energy_balance, 100, 4000)  # Solve between 1000–4000 K
    return T_flame

def coal_combustion_from_mass_flow(ultimate, coal_lb_per_hr, air_scfh, CO2_frac=0.9, NOx_eff=0.35,
                                   air_temp_F = 75.0, air_RH_pct = 55.0, atm_press_inHg = 30.25):

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
    HHV_btu_per_lb = calculate_corrected_hhv(ultimate) # was hard coded before at 12900

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

    mol_wet_air = air_scfh * air_mol_per_scf # approximate based on the very small amout of humidity in normal air
    mol_dry_air = (1.0 - air_humidity_ratio)* mol_wet_air
    mol_O2_actual = mol_dry_air * air_O2_frac
    mol_N2_air = mol_dry_air * air_N2_frac
    mol_H2O_air = mol_wet_air - mol_dry_air
    equivalence_ratio = mol_O2_actual / mol_O2_required # could be used for flame temp

    mol_NO = total_mol_N * NOx_eff
    mol_N2_total = mol_N2_air + (total_mol_N - mol_NO)
    mol_O2_excess = mol_O2_actual - mol_O2_required
    mol_H2O = total_mol_H / 2 + mol_H2O_air

    products_mol = {
        'CO2': mol_CO2,
        'CO': mol_CO,
        'H2O': mol_H2O,
        'SO2': total_mol_S,
        'N2': mol_N2_total,
        'O2': mol_O2_excess,
        'NO': mol_NO
    }

    T_flame_K = estimate_flame_temp_Cp_method(products_mol, HHV_btu_per_lb * coal_dry_frac)

    O2_mole_frac = mol_O2_excess / sum(products_mol.values())
    lb_NO_thermal = estimate_thermal_NO(HHV_btu_per_lb * coal_dry_frac, O2_mole_frac, T_flame_K) * coal_lb_per_hr

    def moles_to_lbs(mol, mw): return mol * mw / 453.592

    inputs = {
        'Air (scfh)': air_flow_scfh,
        'Coal (pph)': coal_lb_per_hr,
        'CO2 Fraction': CO2_frac,
        'NOx Conversion Efficiency':NOx_eff,
        'Air temperature (F)': air_temp_F,
        'Relative Humidity (%)':air_RH_pct,
        'Atmospheric Pressure (inHg)':atm_press_inHg,
    }
    interim_calcs = {
        'Coal HHV (Btu/lb)':HHV_btu_per_lb,
        'Humidity Ratio (h20/dry air by mass)':air_humidity_ratio,
        'Equivalence Ratio':equivalence_ratio
    }

    debug = True
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
        'NO (LB from fuel-N)': moles_to_lbs(mol_NO, MW['NO']),
        'NO (LB thermal)': lb_NO_thermal,
        'NO (LB total)': moles_to_lbs(mol_NO, MW['NO']) + lb_NO_thermal,
        'N2 (LB)': moles_to_lbs(mol_N2_total, MW['N2']),
        'O2 (excess %)': moles_to_lbs(mol_O2_excess, MW['O2']),
        'Heat Released (BTU/hr)': HHV_btu_per_lb * coal_dry_frac * coal_lb_per_hr * (CO2_frac + 0.3 * CO_frac),
        'Estimated Flame Temp (F)': (T_flame_K - 273.15) * 9/5 + 32 # convert to Farrenheit
    }

    results['Total Flue Gas (lb/hr)'] = sum(v for k, v in results.items() if k not in ['Heat Released (BTU/hr)', 'Estimated Flame Temp (K)', 'Total Flue Gas (lb/hr)'])
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

    # Run combustion model
    results = coal_combustion_from_mass_flow(
        ultimate,
        coal_lb_per_hr=coal_rate_lb_hr,
        air_scfh=air_flow_scfh,
        CO2_frac=0.9,
        NOx_eff=0.35
    )

    # Output results
    print("\n=== Combustion Results (per hour) ===\n")
    for k, v in results.items():
        print(f"{k}: {v:,.2f}")
