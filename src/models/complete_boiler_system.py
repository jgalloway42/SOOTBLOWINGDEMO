import math
import numpy as np
import matplotlib.pyplot as plt

class BoilerTubeSection:
    """
    Individual tube section heat transfer model
    """
    
    def __init__(self, name, tube_od=2.0/12, tube_id=1.75/12, tube_length=20.0, 
                 tube_count=100, fouling_gas=0.002, fouling_water=0.001):
        self.name = name
        self.tube_od = tube_od  # ft
        self.tube_id = tube_id  # ft
        self.tube_length = tube_length  # ft
        self.tube_count = tube_count
        self.fouling_gas_side = fouling_gas  # hr-ft²-°F/Btu
        self.fouling_water_side = fouling_water  # hr-ft²-°F/Btu
        
        # Tube material properties (carbon steel)
        self.tube_thermal_conductivity = 26.0  # Btu/hr-ft-°F
        self.tube_wall_thickness = (self.tube_od - self.tube_id) / 2.0
        
        # Operating conditions (will be set by system)
        self.flue_gas_temp_in = 0
        self.flue_gas_temp_out = 0
        self.water_temp_in = 0
        self.water_temp_out = 0
        self.flue_gas_mass_flow = 0
        self.water_mass_flow = 0
        
        # Properties (will be updated based on temperature)
        self.flue_gas_properties = {}
        self.water_properties = {}
    
    def update_flue_gas_properties(self, temperature):
        """Update flue gas properties based on temperature"""
        T_R = temperature + 459.67  # Convert to Rankine
        
        self.flue_gas_properties = {
            'cp': 0.25 + 0.00002 * temperature,  # Btu/lbm-°F
            'density': 0.045 * (530 / T_R),     # lbm/ft³
            'viscosity': 0.02 * (T_R / 530) ** 0.7,  # lbm/hr-ft
            'thermal_conductivity': 0.01 + 0.00003 * temperature,  # Btu/hr-ft-°F
        }
        self.flue_gas_properties['prandtl'] = (self.flue_gas_properties['cp'] * 
                                             self.flue_gas_properties['viscosity'] / 
                                             self.flue_gas_properties['thermal_conductivity'])
    
    def update_water_properties(self, temperature, pressure=1000):
        """Update water properties based on temperature and pressure"""
        if temperature < 500:  # Subcritical liquid
            self.water_properties = {
                'cp': 1.0 + 0.0002 * temperature,  # Btu/lbm-°F
                'density': 62.4 - 0.01 * temperature,  # lbm/ft³
                'viscosity': max(0.1, 2.4 - 0.004 * temperature),  # lbm/hr-ft
                'thermal_conductivity': 0.35 + 0.0001 * temperature,  # Btu/hr-ft-°F
            }
        else:  # Higher temperature water/steam
            self.water_properties = {
                'cp': 1.2,
                'density': max(10.0, 50.0 - 0.02 * (temperature - 500)),
                'viscosity': max(0.05, 0.8 - 0.001 * (temperature - 500)),
                'thermal_conductivity': 0.38,
            }
        
        self.water_properties['prandtl'] = (self.water_properties['cp'] * 
                                          self.water_properties['viscosity'] / 
                                          self.water_properties['thermal_conductivity'])
    
    def calculate_gas_side_htc(self):
        """Calculate gas-side heat transfer coefficient"""
        T_avg = (self.flue_gas_temp_in + self.flue_gas_temp_out) / 2.0
        self.update_flue_gas_properties(T_avg)
        
        # Estimate flow area and velocity
        flow_area = self.tube_count * 0.1  # ft² (approximate)
        velocity = self.flue_gas_mass_flow / (self.flue_gas_properties['density'] * flow_area)
        
        # Reynolds number
        Re = (self.flue_gas_properties['density'] * velocity * self.tube_od / 
              self.flue_gas_properties['viscosity'])
        
        # Nusselt number for cross-flow over tube bundles
        if Re > 10000:
            Nu = 0.27 * (Re ** 0.63) * (self.flue_gas_properties['prandtl'] ** 0.36)
        elif Re > 1000:
            Nu = 0.35 * (Re ** 0.60) * (self.flue_gas_properties['prandtl'] ** 0.36)
        else:
            Nu = 2.0  # Minimum value
        
        h_gas = Nu * self.flue_gas_properties['thermal_conductivity'] / self.tube_od
        return h_gas
    
    def calculate_water_side_htc(self):
        """Calculate water-side heat transfer coefficient"""
        T_avg = (self.water_temp_in + self.water_temp_out) / 2.0
        self.update_water_properties(T_avg)
        
        # Flow area inside tubes
        flow_area = self.tube_count * math.pi * (self.tube_id ** 2) / 4.0
        velocity = self.water_mass_flow / (self.water_properties['density'] * flow_area)
        
        # Reynolds number
        Re = (self.water_properties['density'] * velocity * self.tube_id / 
              self.water_properties['viscosity'])
        
        # Nusselt number
        if Re > 2300:  # Turbulent flow
            Nu = 0.023 * (Re ** 0.8) * (self.water_properties['prandtl'] ** 0.4)
        else:  # Laminar flow
            Nu = 4.36
        
        h_water = Nu * self.water_properties['thermal_conductivity'] / self.tube_id
        return h_water
    
    def calculate_overall_U(self):
        """Calculate overall heat transfer coefficient"""
        h_gas = self.calculate_gas_side_htc()
        h_water = self.calculate_water_side_htc()
        
        # Areas
        A_out = math.pi * self.tube_od * self.tube_length * self.tube_count
        A_in = math.pi * self.tube_id * self.tube_length * self.tube_count
        A_lm = (A_out - A_in) / math.log(A_out / A_in)
        
        # Overall resistance
        R_total = (1.0 / h_gas + 
                  self.fouling_gas_side + 
                  self.tube_wall_thickness / (self.tube_thermal_conductivity * A_lm / A_out) +
                  self.fouling_water_side * (A_out / A_in) + 
                  (1.0 / h_water) * (A_out / A_in))
        
        U = 1.0 / R_total
        return U, h_gas, h_water, A_out
    
    def calculate_heat_transfer(self):
        """Calculate heat transfer rate and temperatures"""
        U, h_gas, h_water, A = self.calculate_overall_U()
        
        # Log mean temperature difference
        delta_T1 = self.flue_gas_temp_in - self.water_temp_out
        delta_T2 = self.flue_gas_temp_out - self.water_temp_in
        
        if abs(delta_T1 - delta_T2) < 1.0:
            LMTD = (delta_T1 + delta_T2) / 2.0
        else:
            LMTD = (delta_T1 - delta_T2) / math.log(delta_T1 / delta_T2)
        
        Q = U * A * LMTD
        
        return Q, U, LMTD, h_gas, h_water, A


class CompleteBoilerSystem:
    """
    Complete boiler system with multiple heat transfer sections
    Sized for 500 MMBtu/hr subcritical boiler
    """
    
    def __init__(self):
        self.design_capacity = 500e6  # Btu/hr
        
        # Initialize all tube sections
        self.sections = self._initialize_sections()
        
        # System operating conditions
        self.fuel_input = 500e6  # Btu/hr
        self.combustion_efficiency = 0.85
        self.flue_gas_mass_flow = 450000  # lbm/hr total
        self.feedwater_flow = 350000  # lbm/hr
        
        # Inlet conditions
        self.furnace_gas_temp = 3200  # °F (after combustion)
        self.feedwater_temp = 220  # °F
        self.stack_temp_target = 275  # °F
        
        # Steam conditions
        self.steam_pressure = 900  # psia
        self.steam_temp = 900  # °F
        
        # Results storage
        self.results = {}
    
    def _initialize_sections(self):
        """Initialize all boiler tube sections with appropriate sizing"""
        sections = {}
        
        # 1. Furnace Wall Tubes (Radiant Heat Transfer)
        sections['furnace_walls'] = BoilerTubeSection(
            name="Furnace Wall Tubes",
            tube_od=3.0/12, tube_id=2.5/12, tube_length=40, tube_count=200,
            fouling_gas=0.001, fouling_water=0.0005  # Less fouling in furnace
        )
        
        # 2. Roof Tubes
        sections['roof_tubes'] = BoilerTubeSection(
            name="Roof Tubes", 
            tube_od=2.5/12, tube_id=2.0/12, tube_length=30, tube_count=150,
            fouling_gas=0.002, fouling_water=0.001
        )
        
        # 3. Superheater Primary
        sections['superheater_primary'] = BoilerTubeSection(
            name="Superheater Primary",
            tube_od=2.0/12, tube_id=1.75/12, tube_length=25, tube_count=120,
            fouling_gas=0.002, fouling_water=0.0005  # Steam side
        )
        
        # 4. Superheater Secondary
        sections['superheater_secondary'] = BoilerTubeSection(
            name="Superheater Secondary",
            tube_od=2.0/12, tube_id=1.75/12, tube_length=20, tube_count=100,
            fouling_gas=0.0025, fouling_water=0.0005
        )
        
        # 5. Generating Bank (Boiler Bank)
        sections['generating_bank'] = BoilerTubeSection(
            name="Generating Bank",
            tube_od=3.0/12, tube_id=2.5/12, tube_length=20, tube_count=300,
            fouling_gas=0.003, fouling_water=0.002
        )
        
        # 6. Economizer Primary
        sections['economizer_primary'] = BoilerTubeSection(
            name="Economizer Primary",
            tube_od=2.0/12, tube_id=1.75/12, tube_length=20, tube_count=200,
            fouling_gas=0.003, fouling_water=0.002
        )
        
        # 7. Economizer Secondary
        sections['economizer_secondary'] = BoilerTubeSection(
            name="Economizer Secondary", 
            tube_od=2.0/12, tube_id=1.75/12, tube_length=20, tube_count=150,
            fouling_gas=0.004, fouling_water=0.0025  # More fouling at lower temps
        )
        
        # 8. Air Heater
        sections['air_heater'] = BoilerTubeSection(
            name="Air Heater",
            tube_od=2.0/12, tube_id=1.75/12, tube_length=15, tube_count=400,
            fouling_gas=0.005, fouling_water=0.001  # Air side has less fouling
        )
        
        return sections
    
    def calculate_water_flow_distribution(self):
        """Calculate water/steam flow through each section"""
        flows = {}
        
        # Main steam flow
        main_steam_flow = self.feedwater_flow * 0.95  # Account for blowdown
        
        # Flow distribution
        flows['furnace_walls'] = self.feedwater_flow * 0.4
        flows['roof_tubes'] = self.feedwater_flow * 0.2
        flows['superheater_primary'] = main_steam_flow
        flows['superheater_secondary'] = main_steam_flow
        flows['generating_bank'] = self.feedwater_flow
        flows['economizer_primary'] = self.feedwater_flow
        flows['economizer_secondary'] = self.feedwater_flow
        flows['air_heater'] = 280000  # Combustion air flow (lbm/hr)
        
        return flows
    
    def solve_system_temperatures(self, max_iterations=20, tolerance=5.0):
        """
        Solve the complete system using iterative method
        """
        # Initial temperature estimates
        gas_temps = [3200, 2800, 2400, 2000, 1600, 1200, 800, 500, 275]
        water_temps = [220, 280, 340, 400, 550, 650, 750, 850, 900]
        
        water_flows = self.calculate_water_flow_distribution()
        
        for iteration in range(max_iterations):
            gas_temps_new = [self.furnace_gas_temp]
            water_temps_new = [self.feedwater_temp]
            total_heat_absorbed = 0
            
            section_names = ['furnace_walls', 'roof_tubes', 'generating_bank', 
                           'superheater_primary', 'superheater_secondary',
                           'economizer_primary', 'economizer_secondary', 'air_heater']
            
            for i, section_name in enumerate(section_names):
                section = self.sections[section_name]
                
                # Set inlet conditions
                section.flue_gas_temp_in = gas_temps[i]
                section.flue_gas_mass_flow = self.flue_gas_mass_flow
                section.water_mass_flow = water_flows[section_name]
                
                if section_name in ['superheater_primary', 'superheater_secondary']:
                    section.water_temp_in = water_temps[i] if i < len(water_temps) else 700
                elif section_name == 'air_heater':
                    section.water_temp_in = 80  # Combustion air temperature
                else:
                    section.water_temp_in = water_temps[i] if i < len(water_temps) else 400
                
                # Estimate outlet temperatures
                if section_name == 'furnace_walls':
                    # High heat flux in furnace
                    delta_T_gas = 400
                    delta_T_water = 50
                elif section_name in ['superheater_primary', 'superheater_secondary']:
                    # Superheating steam
                    delta_T_gas = 300
                    delta_T_water = 75
                elif section_name == 'air_heater':
                    # Air heating
                    delta_T_gas = 200
                    delta_T_water = 400  # Air temperature rise
                else:
                    # Water heating/boiling
                    delta_T_gas = 250
                    delta_T_water = 40
                
                section.flue_gas_temp_out = section.flue_gas_temp_in - delta_T_gas
                section.water_temp_out = section.water_temp_in + delta_T_water
                
                # Calculate heat transfer
                Q, U, LMTD, h_gas, h_water, A = section.calculate_heat_transfer()
                
                # Energy balance to refine temperatures
                if section_name != 'air_heater':
                    gas_cp = 0.25 + 0.00002 * ((section.flue_gas_temp_in + section.flue_gas_temp_out) / 2)
                    water_cp = section.water_properties.get('cp', 1.0)
                    
                    # Refine gas outlet temperature
                    actual_gas_temp_out = section.flue_gas_temp_in - Q / (section.flue_gas_mass_flow * gas_cp)
                    
                    # Refine water outlet temperature
                    if section.water_mass_flow > 0:
                        actual_water_temp_out = section.water_temp_in + Q / (section.water_mass_flow * water_cp)
                    else:
                        actual_water_temp_out = section.water_temp_out
                    
                    gas_temps_new.append(actual_gas_temp_out)
                    water_temps_new.append(actual_water_temp_out)
                else:
                    # Air heater special case
                    gas_temps_new.append(section.flue_gas_temp_out)
                    water_temps_new.append(section.water_temp_out)
                
                total_heat_absorbed += Q
                
                # Store results
                self.results[section_name] = {
                    'heat_transfer_rate': Q,
                    'overall_U': U,
                    'gas_htc': h_gas,
                    'water_htc': h_water,
                    'LMTD': LMTD,
                    'area': A,
                    'gas_temp_in': section.flue_gas_temp_in,
                    'gas_temp_out': actual_gas_temp_out if section_name != 'air_heater' else section.flue_gas_temp_out,
                    'water_temp_in': section.water_temp_in,
                    'water_temp_out': actual_water_temp_out if section_name != 'air_heater' else section.water_temp_out,
                    'water_flow': section.water_mass_flow
                }
            
            # Check convergence
            max_temp_change = max([abs(gas_temps_new[i] - gas_temps[i]) 
                                 for i in range(min(len(gas_temps), len(gas_temps_new)))])
            
            if max_temp_change < tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            # Update temperatures for next iteration
            gas_temps = gas_temps_new
            water_temps = water_temps_new
        
        # Calculate overall efficiency
        self.overall_efficiency = total_heat_absorbed / self.fuel_input
        self.total_heat_absorbed = total_heat_absorbed
        
        return self.results
    
    def print_system_summary(self):
        """Print comprehensive system performance summary"""
        print("=" * 80)
        print(f"BOILER SYSTEM PERFORMANCE SUMMARY")
        print(f"Design Capacity: {self.design_capacity/1e6:.0f} MMBtu/hr")
        print("=" * 80)
        
        total_Q = 0
        total_area = 0
        
        print(f"{'Section Name':<20} {'Q (MMBtu/hr)':<12} {'U':<8} {'Gas In':<8} {'Gas Out':<9} {'Water In':<9} {'Water Out':<10}")
        print("-" * 80)
        
        for section_name, data in self.results.items():
            total_Q += data['heat_transfer_rate']
            total_area += data['area']
            
            print(f"{section_name:<20} {data['heat_transfer_rate']/1e6:<12.1f} {data['overall_U']:<8.2f} "
                  f"{data['gas_temp_in']:<8.0f} {data['gas_temp_out']:<9.0f} "
                  f"{data['water_temp_in']:<9.0f} {data['water_temp_out']:<10.0f}")
        
        print("-" * 80)
        print(f"{'TOTAL':<20} {total_Q/1e6:<12.1f}")
        print()
        print(f"Overall Boiler Efficiency: {self.overall_efficiency:.1%}")
        print(f"Total Heat Transfer Area: {total_area:.0f} ft²")
        print(f"Final Stack Temperature: {self.results['air_heater']['gas_temp_out']:.0f}°F")
        print(f"Steam Production: {self.feedwater_flow * 0.95:.0f} lbm/hr")
        
    def plot_temperature_profile(self):
        """Plot temperature profiles through the boiler"""
        try:
            sections = list(self.results.keys())
            gas_temps_in = [self.results[s]['gas_temp_in'] for s in sections]
            gas_temps_out = [self.results[s]['gas_temp_out'] for s in sections]
            water_temps_in = [self.results[s]['water_temp_in'] for s in sections]
            water_temps_out = [self.results[s]['water_temp_out'] for s in sections]
            
            positions = range(len(sections))
            
            plt.figure(figsize=(14, 8))
            
            # Gas temperature profile
            plt.subplot(2, 1, 1)
            plt.plot(positions, gas_temps_in, 'r-o', label='Gas Inlet', linewidth=2)
            plt.plot(positions, gas_temps_out, 'r--s', label='Gas Outlet', linewidth=2)
            plt.xlabel('Boiler Section')
            plt.ylabel('Temperature (°F)')
            plt.title('Flue Gas Temperature Profile')
            plt.xticks(positions, [s.replace('_', ' ').title() for s in sections], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Water/steam temperature profile
            plt.subplot(2, 1, 2)
            plt.plot(positions, water_temps_in, 'b-o', label='Water/Steam Inlet', linewidth=2)
            plt.plot(positions, water_temps_out, 'b--s', label='Water/Steam Outlet', linewidth=2)
            plt.xlabel('Boiler Section')
            plt.ylabel('Temperature (°F)')
            plt.title('Water/Steam Temperature Profile')
            plt.xticks(positions, [s.replace('_', ' ').title() for s in sections], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available - skipping temperature profile plot")
    
    def analyze_fouling_effects(self):
        """Analyze the effect of fouling on system performance"""
        print("\n" + "=" * 60)
        print("FOULING ANALYSIS")
        print("=" * 60)
        
        # Store original results
        clean_results = self.results.copy()
        clean_efficiency = self.overall_efficiency
        
        # Increase fouling factors by 3x
        for section in self.sections.values():
            section.fouling_gas_side *= 3
            section.fouling_water_side *= 3
        
        # Recalculate with increased fouling
        fouled_results = self.solve_system_temperatures()
        
        print(f"{'Section':<20} {'Clean Q':<12} {'Fouled Q':<12} {'Reduction':<10}")
        print("-" * 55)
        
        for section_name in clean_results.keys():
            clean_Q = clean_results[section_name]['heat_transfer_rate']
            fouled_Q = fouled_results[section_name]['heat_transfer_rate']
            reduction = (clean_Q - fouled_Q) / clean_Q * 100
            
            print(f"{section_name:<20} {clean_Q/1e6:<12.1f} {fouled_Q/1e6:<12.1f} {reduction:<10.1f}%")
        
        print(f"\nOverall Efficiency:")
        print(f"Clean: {clean_efficiency:.1%}")
        print(f"Fouled: {self.overall_efficiency:.1%}")
        print(f"Efficiency Loss: {(clean_efficiency - self.overall_efficiency)/clean_efficiency * 100:.1f}%")
        
        # Restore original fouling factors
        for section in self.sections.values():
            section.fouling_gas_side /= 3
            section.fouling_water_side /= 3


# Example usage and analysis
if __name__ == "__main__":
    # Create complete boiler system
    boiler_system = CompleteBoilerSystem()
    
    print("Analyzing 500 MMBtu/hr Subcritical Boiler System...")
    print("Solving heat transfer through all sections...")
    
    # Solve the system
    results = boiler_system.solve_system_temperatures()
    
    # Print comprehensive summary
    boiler_system.print_system_summary()
    
    # Plot temperature profiles
    boiler_system.plot_temperature_profile()
    
    # Analyze fouling effects
    boiler_system.analyze_fouling_effects()
    
    print("\n" + "=" * 60)
    print("DETAILED SECTION ANALYSIS")
    print("=" * 60)
    
    # Detailed analysis of each section
    for section_name, data in results.items():
        print(f"\n{section_name.replace('_', ' ').upper()}:")
        print(f"  Heat Transfer Rate: {data['heat_transfer_rate']/1e6:.2f} MMBtu/hr")
        print(f"  Overall U: {data['overall_U']:.2f} Btu/hr-ft²-°F")
        print(f"  Gas Side HTC: {data['gas_htc']:.1f} Btu/hr-ft²-°F")
        print(f"  Water Side HTC: {data['water_htc']:.1f} Btu/hr-ft²-°F")
        print(f"  LMTD: {data['LMTD']:.1f} °F")
        print(f"  Heat Transfer Area: {data['area']:.0f} ft²")
        print(f"  Gas Temperature Drop: {data['gas_temp_in'] - data['gas_temp_out']:.0f} °F")
        print(f"  Water Temperature Rise: {data['water_temp_out'] - data['water_temp_in']:.0f} °F")
