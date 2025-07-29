import numpy as np

def calculate_steam_properties(pressure_psig, feedwater_temp_F=220):
    """
    Calculate steam properties at given conditions.
    
    Parameters:
    pressure_psig (float): Steam pressure in psig
    feedwater_temp_F (float): Feedwater temperature in °F (default 220°F, typical for boilers)
    
    Returns:
    dict: Steam properties including enthalpy values
    """
    # Convert psig to psia
    pressure_psia = pressure_psig + 14.7
    
    # Steam properties at 900 psig (914.7 psia) - from steam tables
    # These are approximate values - in practice you'd use steam tables or libraries like CoolProp
    if pressure_psig == 900:
        steam_temp_F = 531.98  # Saturation temperature at 914.7 psia
        h_steam_sat_liquid = 542.4  # Btu/lb - saturated liquid enthalpy
        h_steam_sat_vapor = 1204.4  # Btu/lb - saturated vapor enthalpy
        h_fg = h_steam_sat_vapor - h_steam_sat_liquid  # Latent heat of vaporization
    else:
        # Rough approximation for other pressures
        steam_temp_F = 212 + (pressure_psig * 0.35)  # Very rough estimate
        h_steam_sat_liquid = 180 + (pressure_psig * 0.4)
        h_steam_sat_vapor = 1150 + (pressure_psig * 0.06)
        h_fg = h_steam_sat_vapor - h_steam_sat_liquid
    
    # Feedwater enthalpy (approximate)
    h_feedwater = feedwater_temp_F - 32  # Rough approximation: h ≈ T-32 for liquid water
    
    return {
        'pressure_psia': pressure_psia,
        'steam_temp_F': steam_temp_F,
        'h_feedwater': h_feedwater,
        'h_steam_sat_liquid': h_steam_sat_liquid,
        'h_steam_sat_vapor': h_steam_sat_vapor,
        'h_fg': h_fg,
        'energy_required_per_lb': h_steam_sat_vapor - h_feedwater
    }

def calculate_coal_requirement(steam_flow_lb_hr, steam_pressure_psig, coal_hhv_btu_lb=12900, 
                             boiler_efficiency=0.85, feedwater_temp_F=220):
    """
    Calculate coal requirement for steam generation.
    
    Parameters:
    steam_flow_lb_hr (float): Required steam flow in lb/hr
    steam_pressure_psig (float): Steam pressure in psig
    coal_hhv_btu_lb (float): Coal higher heating value in Btu/lb (default 12900)
    boiler_efficiency (float): Boiler efficiency (default 0.85 or 85%)
    feedwater_temp_F (float): Feedwater temperature in °F
    
    Returns:
    dict: Coal requirement and energy calculations
    """
    
    # Get steam properties
    steam_props = calculate_steam_properties(steam_pressure_psig, feedwater_temp_F)
    
    # Calculate total energy required
    energy_per_lb_steam = steam_props['energy_required_per_lb']  # Btu/lb steam
    total_energy_required = steam_flow_lb_hr * energy_per_lb_steam  # Btu/hr
    
    # Account for boiler efficiency
    fuel_energy_required = total_energy_required / boiler_efficiency  # Btu/hr
    
    # Calculate coal requirement
    coal_required_lb_hr = fuel_energy_required / coal_hhv_btu_lb
    
    # Additional calculations
    coal_required_tons_hr = coal_required_lb_hr / 2000
    coal_required_tons_day = coal_required_tons_hr * 24
    
    return {
        'steam_flow_lb_hr': steam_flow_lb_hr,
        'steam_pressure_psig': steam_pressure_psig,
        'steam_temp_F': steam_props['steam_temp_F'],
        'feedwater_temp_F': feedwater_temp_F,
        'energy_per_lb_steam_btu': energy_per_lb_steam,
        'total_energy_required_btu_hr': total_energy_required,
        'total_energy_required_mmbtu_hr': total_energy_required / 1e6,
        'boiler_efficiency': boiler_efficiency,
        'fuel_energy_required_btu_hr': fuel_energy_required,
        'fuel_energy_required_mmbtu_hr': fuel_energy_required / 1e6,
        'coal_hhv_btu_lb': coal_hhv_btu_lb,
        'coal_required_lb_hr': coal_required_lb_hr,
        'coal_required_tons_hr': coal_required_tons_hr,
        'coal_required_tons_day': coal_required_tons_day,
        'steam_to_coal_ratio': steam_flow_lb_hr / coal_required_lb_hr
    }

# Calculate for the specific case: 500,000 lb/hr steam at 900 psig
if __name__ == "__main__":
    # Main calculation
    results = calculate_coal_requirement(
        steam_flow_lb_hr=500000,  # 500 kpph
        steam_pressure_psig=900,
        coal_hhv_btu_lb=12900,    # Typical bituminous coal
        boiler_efficiency=0.85,   # Typical modern boiler efficiency
        feedwater_temp_F=220      # Typical feedwater temperature
    )
    
    print("=== STEAM GENERATION COAL REQUIREMENT CALCULATION ===\n")
    
    print("STEAM CONDITIONS:")
    print(f"  Steam Flow Rate: {results['steam_flow_lb_hr']:,} lb/hr ({results['steam_flow_lb_hr']/1000:,.0f} kpph)")
    print(f"  Steam Pressure: {results['steam_pressure_psig']} psig")
    print(f"  Steam Temperature: {results['steam_temp_F']:.1f} °F (saturated)")
    print(f"  Feedwater Temperature: {results['feedwater_temp_F']} °F")
    
    print(f"\nENERGY REQUIREMENTS:")
    print(f"  Energy per lb of steam: {results['energy_per_lb_steam_btu']:.1f} Btu/lb")
    print(f"  Total energy to steam: {results['total_energy_required_mmbtu_hr']:.1f} MMBtu/hr")
    print(f"  Boiler efficiency: {results['boiler_efficiency']*100:.1f}%")
    print(f"  Fuel energy required: {results['fuel_energy_required_mmbtu_hr']:.1f} MMBtu/hr")
    
    print(f"\nCOAL REQUIREMENTS:")
    print(f"  Coal HHV: {results['coal_hhv_btu_lb']:,} Btu/lb")
    print(f"  Coal required: {results['coal_required_lb_hr']:,.0f} lb/hr")
    print(f"  Coal required: {results['coal_required_tons_hr']:.1f} tons/hr")
    print(f"  Coal required: {results['coal_required_tons_day']:.1f} tons/day")
    print(f"  Steam-to-coal ratio: {results['steam_to_coal_ratio']:.1f} lb steam/lb coal")
    
    print(f"\n=== SENSITIVITY ANALYSIS ===")
    
    # Test different coal qualities
    print(f"\nCoal Quality Sensitivity (at 85% boiler efficiency):")
    coal_types = [
        ("Lignite", 7000),
        ("Sub-bituminous", 9500),
        ("Bituminous", 12900),
        ("Anthracite", 14500)
    ]
    
    for coal_name, hhv in coal_types:
        result = calculate_coal_requirement(500000, 900, hhv, 0.85)
        print(f"  {coal_name:15} ({hhv:5,} Btu/lb): {result['coal_required_lb_hr']:6,.0f} lb/hr ({result['coal_required_tons_hr']:4.1f} tons/hr)")
    
    # Test different boiler efficiencies
    print(f"\nBoiler Efficiency Sensitivity (with {results['coal_hhv_btu_lb']:,} Btu/lb coal):")
    efficiencies = [0.75, 0.80, 0.85, 0.90, 0.95]
    
    for eff in efficiencies:
        result = calculate_coal_requirement(500000, 900, 12900, eff)
        print(f"  {eff*100:4.1f}% efficiency: {result['coal_required_lb_hr']:6,.0f} lb/hr ({result['coal_required_tons_hr']:4.1f} tons/hr)")
