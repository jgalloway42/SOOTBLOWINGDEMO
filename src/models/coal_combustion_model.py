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

import numpy as np

def calculate_ash_free_moles(ultimate_analysis_ar, coal_pounds):
    """
    Converts ultimate analysis from as-received to ash-free basis and calculates
    the number of moles of each element.

    Args:
        ultimate_analysis_ar (dict): Dictionary containing ultimate analysis (as-received basis).
                                     Keys: 'C', 'H', 'O', 'N', 'S', 'Ash', 'Moisture'
                                     Values: Weight percentage (%).
        coal_pounds (float): Weight of coal in pounds.

    Returns:
        dict: A dictionary containing the number of moles of each element on an ash-free basis.
    """

    # 1. Convert percentages to fractions
    ultimate_analysis_frac = {key: value / 100 for key, value in ultimate_analysis_ar.items()}

    # 2. Calculate the "dry, ash-free" (DAF) basis
    daf_factor = 1 - ultimate_analysis_frac['Ash'] - ultimate_analysis_frac['Moisture']
    if daf_factor <= 0:
        raise ValueError("Invalid input: (Ash + Moisture) >= 100%. Cannot calculate DAF basis.")

    ash_free_basis = {}
    for element in ['C', 'H', 'O', 'N', 'S']:
        ash_free_basis[element] = ultimate_analysis_frac[element] / daf_factor

    # 3. Calculate the weight of ash-free coal in pounds
    coal_ash_free_pounds = coal_pounds * (1 - ultimate_analysis_frac['Ash'])

    # 4. Convert pounds to grams (1 lb = 453.592 grams)
    coal_ash_free_grams = coal_ash_free_pounds * 453.592

    # 5. Define molecular weights (in g/mol or lb/lbmol)
    molecular_weights = {
        'C': 12.011,  #
        'H': 1.008, #
        'O': 15.999, #
        'N': 14.007, #
        'S': 32.065 #
    }

    # 6. Calculate the moles of each element (ash-free basis)
    moles_ash_free = {}
    for element, percentage in ash_free_basis.items():
        element_grams = coal_ash_free_grams * percentage
        moles_ash_free[element] = element_grams / molecular_weights[element]

    return moles_ash_free

def calculate_flue_gas_composition(coal_composition, wet_air_composition,
                                   co2_factor, product_no_moles):
    """
    Calculate combustion products in flue gas from coal combustion.
    
    Parameters:
    -----------
    coal_composition : dict
        Molar composition of coal including moisture
        Keys: 'C', 'H', 'O', 'N', 'S', 'Moisture' (moles)
    
    wet_air_composition : dict
        Total moles of wet air components per kg coal
        Keys: 'N2', 'O2', 'H2O' (moles)
    
    co2_factor : float
        Carbon in fuel which is converted to CO2 and the remainder is CO (0.0-1.0)
    
    product_no_moles : float
        Moles of NO in combustion products
    
    Returns:
    --------
    dict : Combustion products composition (moles)
        Keys: 'CO2', 'CO', 'H2O', 'N2', 'SO2', 'NO', 'O2'
    """
    
    # Extract coal composition
    C_coal = coal_composition.get('C', 0)
    H_coal = coal_composition.get('H', 0)
    O_coal = coal_composition.get('O', 0)
    N_coal = coal_composition.get('N', 0)
    S_coal = coal_composition.get('S', 0)
    H2O_coal = coal_composition.get('Moisture', 0)
    
    # Extract air composition (total moles per kg coal)
    N2_air = wet_air_composition.get('N2', 0)
    O2_air = wet_air_composition.get('O2', 0)
    H2O_air = wet_air_composition.get('H2O', 0)
    
    # Calculate stoichiometric oxygen requirement
    # Combustion reactions:
    # C + O2 -> CO2 or C + 0.5*O2 -> CO
    # H2 + 0.5*O2 -> H2O (assuming H exists as H2)
    # S + O2 -> SO2
    
    # Carbon splits between CO2 and CO based on given ratio
    CO2_moles = co2_factor * C_coal
    CO_moles = (1 - co2_factor) * C_coal
    
    # Oxygen requirement for carbon combustion
    O2_for_C = CO2_moles + 0.5 * CO_moles
    
    # Oxygen requirement for hydrogen combustion (H -> H2O)
    O2_for_H = 0.25 * H_coal  # Assuming H exists as H2, so H2 + 0.5*O2 -> H2O
    
    # Oxygen requirement for sulfur combustion
    O2_for_S = S_coal
    
    # Total oxygen available from air and coal
    O2_available = O2_air + 0.5 * O_coal
    
    # Calculate combustion products
    products = {}
    
    # CO2 from carbon combustion
    products['CO2'] = CO2_moles
    
    # CO from incomplete combustion
    products['CO'] = CO_moles
    
    # H2O from hydrogen combustion + moisture from coal + moisture from air
    products['H2O'] = 0.5 * H_coal + H2O_coal + H2O_air
    
    # SO2 from sulfur combustion
    products['SO2'] = S_coal
    
    # NO (given as input)
    products['NO'] = product_no_moles
    
    # N2 from air + nitrogen from coal - nitrogen consumed for NO formation
    # Assuming NO formation: N2 + 0.5*O2 -> 2*NO
    N2_consumed_for_NO = 0.5 * product_no_moles
    products['N2'] = N2_air + 0.5 * N_coal - N2_consumed_for_NO
    
    # Excess O2 (unconsumed oxygen)
    O2_consumed = O2_for_C + O2_for_H + O2_for_S + 0.5 * product_no_moles  # Include O2 for NO formation
    products['O2'] = O2_available - O2_consumed
    
    # Remove any negative values (shouldn't occur with proper inputs)
    for key in products:
        products[key] = max(0, products[key])
    
    return products

def calculate_mol_O2_required(
    mol_CO2: float,
    mol_CO: float,
    coal_mol_H: float,
    coal_mol_S: float,
    coal_mol_O: float,
    mol_NO: float = 0.0 # Default to 0.0 if not specified
) -> float:
    """
    Calculates the moles of O2 theoretically required for combustion,
    including oxygen consumed for the formation of CO and NO.

    Args:
        mol_CO2 (float): Moles of CO2 produced.
        mol_CO (float): Moles of CO produced (for incomplete combustion).
        coal_mol_H (float): Moles of hydrogen (H atoms) in the coal fuel.
        coal_mol_S (float): Moles of sulfur (S atoms) in the coal fuel.
        coal_mol_O (float): Moles of oxygen (O atoms) in the coal fuel.
        mol_NO (float, optional): Moles of nitric oxide (NO) produced.
                                  Defaults to 0.0 if not specified.

    Returns:
        float: The total moles of O2 theoretically required.
    """

    mol_O2_required = (
        mol_CO2 +           # O2 for CO2: C + O2 -> CO2
        mol_CO * 0.5 +      # O2 for CO: C + 0.5 O2 -> CO
        coal_mol_H / 4 +    # O2 for H2O: H + 0.25 O2 -> 0.5 H2O (per H atom)
        coal_mol_S +        # O2 for SO2: S + O2 -> SO2
        mol_NO * 0.5 -      # O2 for NO: 0.5 O2 + N -> NO (per NO molecule)
        coal_mol_O / 2      # Subtract O2 already present in the coal (per O atom in fuel)
    )

    return mol_O2_required

def estimate_fuel_NO(fuel_N_moles, NOx_eff, T_flame_K = 2000.0):
    """
    Estimate the amount of NO formed from fuel-bound nitrogen, accounting for flame temperature effects.

    Parameters:
        fuel_N_moles (float): Moles of nitrogen in the fuel per hour.
        NOx_eff (float): Reference conversion efficiency of fuel nitrogen to NO (fraction, e.g., 0.35).
        T_flame_K (float, optional): Flame temperature in Kelvin (default: 2000 K).

    Returns:
        float: Estimated moles of NO formed from fuel nitrogen per hour.
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

def co2_mole_fraction(excess_air_pct):
    """
    Estimate the mole fraction of product carbon as CO2,
    i.e., moles CO2 / (moles CO2 + moles CO)
    """
    # Internal helper for CO/CO2 mole ratio
    # rough engineering estimation of CO vs CO2 generation
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
    mole_fraction_co2 = 1.0 / (1.0 + r)    # CO2 mole fraction
    
    return mole_fraction_co2

def simple_flame_temperature_estimate(HHV_BTU_per_lb):
    """
    Estimate flame temperature based on Higher Heating Value (HHV) and coal mass flow rate.
    
    Parameters:
        HHV_BTU_per_lb (float): Higher heating value in BTU/lb (complete combustion basis).
    
    Returns:
        float: Estimated flame temperature in Kelvin.
    """
    # Simplified empirical relationship for flame temperature
    # Higher heating value correlates roughly with flame temperature
    T_flame_estimate = 1200 + (HHV_BTU_per_lb - 8000) * 0.15  # Empirical relationship
    return max(T_flame_estimate, 1200)  # Minimum reasonable flame temp

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
        
        # # Return the imbalance (should be zero at equilibrium)
        # print(f'Actual Heat Released: {actual_heat_released_J_per_hr:.2e} J/hr, ')
        # print(f'Total Energy Absorbed: {total_energy_absorbed:.2e} J/hr, T: {T:.2f} K')
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
        
        print(f'Heat Release Unadjusted for Combustion Efficiency: {HHV_BTU_per_lb * coal_mass_lb_per_hr :.2e} Btu/hr')
        print(f"Heat input (adjusted for combustion efficiency): {actual_heat_released_J_per_hr/1055.06 :.2e} Btu/hr")
        print(f"Combustion efficiency factor: {combustion_efficiency:.3f} (CO2 frac: {CO2_frac:.2f})")
    
    try:
        # Find the temperature where energy balance equals zero
        # Use a reasonable range for combustion temperatures
        T_flame = opt.brentq(energy_balance, 800, 4000)  # 800-3500 K range
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

    # 1. calculate corrected coal HHV
    HHV_btu_per_lb = calculate_corrected_hhv(ultimate)

    # 2. Calculate Air Humidity ratio
    air_humidity_ratio = calculate_humidity_ratio(air_temp_F, air_RH_pct, atm_press_inHg)

    # Dry Air composition (mole fraction)
    air_O2_frac = 0.21
    air_N2_frac = 0.79
    air_mol_per_scf = 1.0 / 379.0  # lbmols/scf at STP

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

    coal_mol_C = mol_C * coal_dry_lb_per_hr
    coal_mol_H = mol_H * coal_dry_lb_per_hr
    coal_mol_O = mol_O * coal_dry_lb_per_hr
    coal_mol_S = mol_S * coal_dry_lb_per_hr
    coal_mol_N = mol_N * coal_dry_lb_per_hr

    # 3. Calculate fuel-bound NOx generation
    prod_mol_NO_fuelbound = estimate_fuel_NO(coal_mol_N, NOx_eff, T_flame_K)

    # 4. Calculate combustion products assuming no thermal NOx or CO initially
    CO_frac = 0.0
    CO2_frac = 1.0
    prod_mol_CO2 = coal_mol_C
    prod_mol_CO = 0

    mol_O2_required = calculate_mol_O2_required(mol_CO2=prod_mol_CO2, mol_CO=prod_mol_CO,
                                            coal_mol_H=coal_mol_H,
                                            coal_mol_S=coal_mol_S,
                                            coal_mol_O=coal_mol_O,
                                            mol_NO=prod_mol_NO_fuelbound) 
    
    mol_wet_air = air_scfh * air_mol_per_scf # approximate based on the very small amount of humidity in normal air
    mol_dry_air = (1.0 - air_humidity_ratio) * mol_wet_air
    mol_O2_actual = mol_dry_air * air_O2_frac
    mol_N2_air = mol_dry_air * air_N2_frac
    mol_H2O_air = mol_wet_air - mol_dry_air
    equivalence_ratio = mol_O2_actual / mol_O2_required # could be used for flame temp

    prod_mol_O2 = mol_O2_actual - mol_O2_required
    if prod_mol_O2 < 0:
        raise ValueError("Insufficient oxygen in air to support combustion. Increase air flow rate or reduce coal feed rate.")
    prod_mol_H2O = coal_mol_H / 2 + mol_H2O_air

    # Initial products
    products_mol = {
        'CO2': prod_mol_CO2,
        'CO': prod_mol_CO,
        'H2O': prod_mol_H2O,
        'SO2': coal_mol_S,
        'N2': mol_N2_air + (coal_mol_N - prod_mol_NO_fuelbound)*0.5,
        'O2': prod_mol_O2,
        'NO': prod_mol_NO_fuelbound  # No NOx assumed for flame temperature calculation
    }

    # 5. Calculate CO2 Fraction based on excess O2
    excess_o2_pct = products_mol['O2'] / sum(products_mol.values()) * 100.0
    CO2_frac = co2_mole_fraction(excess_o2_pct)

    # 6. Update flue gas composition based on CO2 fraction
    products_mol['CO'] = (1 - CO2_frac) * products_mol['CO2']  # Adjust CO based on CO2 fraction
    products_mol['CO2'] = CO2_frac * products_mol['CO2']  # Adjust CO2 based on CO2 fraction
    

    # Calculate flame temperature first (NOx formation doesn't significantly affect energy balance)
    T_flame_K = estimate_flame_temp_Cp_method(products_mol, HHV_btu_per_lb, coal_dry_lb_per_hr, CO2_frac, debug=debug)
    # Now calculate NOx formation based on the calculated flame temperature
    total_moles = sum(products_mol.values())
    O2_mole_frac = prod_mol_O2 / total_moles if total_moles > 0 else 0.05
    N2_mole_frac = (mol_N2_air + coal_mol_N) / total_moles if total_moles > 0 else 0.75
    

    
    # Thermal NOx (strongly temperature-dependent)
    thermal_NO_rate = estimate_thermal_NO(O2_mole_frac, N2_mole_frac, T_flame_K)
    mol_NO_thermal = thermal_NO_rate * (mol_N2_air + coal_mol_N)  # mol NO per hour
    
    # Total NOx
    mol_NO_total = prod_mol_NO_fuelbound + mol_NO_thermal
    
    # Update N2 balance (subtract NOx formation)
    mol_N2_total_corrected = mol_N2_air + (coal_mol_N - mol_NO_total)
    
    # Final product composition
    products_mol = {
        'CO2': prod_mol_CO2,
        'CO': prod_mol_CO,
        'H2O': prod_mol_H2O,
        'SO2': coal_mol_S,
        'N2': mol_N2_total_corrected,
        'O2': prod_mol_O2,
        'NO': mol_NO_total
    }

    def moles_to_lbs(mol, mw): return mol * mw / 453.592

    # convert moles to pounds for final results
    products_lb = {k:moles_to_lbs(v,MW[k]) for k, v in products_mol.items()}

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


    total_flue_gas_pph = sum(products_lb.values())#sum(v for k, v in results.items() if k not in ['Heat Released (BTU/hr)', 'Estimated Flame Temp (F)', 'Total Flue Gas (lb/hr)'])
    total_dry_flue_gas_pph = total_flue_gas_pph - products_lb['H2O']  # Dry flue gas excludes water vapor
    excess_o2_pct = products_lb['O2']/total_dry_flue_gas_pph * 100.0

    results = {
        'Total Flue Gas (lb/hr)': total_flue_gas_pph,
        'CO2 (lb/hr)': products_lb['CO2'],
        'CO (lb/hr)': products_lb['CO'],
        'H2O (lb/hr)': products_lb['H2O'],
        'SO2 (lb/hr)': products_lb['SO2'],
        'NO (lb/hr from fuel-N)': moles_to_lbs(prod_mol_NO_fuelbound, MW['NO']),
        'NO (lb/hr thermal)': moles_to_lbs(mol_NO_thermal, MW['NO']),
        'NO (lb/hr total)': products_lb['NO'],
        'N2 (lb/hr)': products_lb['N2'],
        'O2 (lb/hr)': products_lb['O2'],
        'Dry O2 (% by vol)': excess_o2_pct,
        'Heat Released (BTU/hr)': HHV_btu_per_lb * coal_dry_frac * coal_lb_per_hr * (CO2_frac + 0.3 * CO_frac),
        'Estimated Flame Temp (F)': (T_flame_K - 273.15) * 9/5 + 32 # convert to Fahrenheit
    }

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
    print(sum(ultimate.values()))

    # Input conditions
    coal_rate_lb_hr = 10000    # lb/hr of coal
    air_flow_scfh = 1800000    # standard cubic feet per hour of air
    CO2_fraction = 0.9         # percent carbon converted to CO2
    NOx_efficiency = 0.35      # percent fuel-bound nitrogen
    
    print(2*'\n')
    print(f"\n=== DETAILED RESULTS FOR GIVEN SIMULATION PARAMETERS ===")
    
    # Run combustion model with standard conditions
    results = coal_combustion_from_mass_flow(
        ultimate,
        coal_lb_per_hr=coal_rate_lb_hr,
        air_scfh=air_flow_scfh,
        CO2_frac=CO2_fraction,
        NOx_eff=NOx_efficiency,
        debug=True,
    )

    # Output results
    print("\n=== Combustion Results (per hour) ===\n")
    for k, v in results.items():
        print(f"{k}: {v:,.2f}")
