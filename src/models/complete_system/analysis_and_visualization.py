#!/usr/bin/env python3
"""
System Analysis and Visualization Tools

This module contains analysis and visualization utilities for the enhanced boiler system
with comprehensive reporting and plotting capabilities.

Classes:
    SystemAnalyzer: Analysis and reporting utilities
    Visualizer: Plotting and visualization tools

Dependencies:
    - matplotlib: Plotting capabilities
    - numpy: Numerical calculations
    - datetime: Timestamp generation
    - boiler_system: Complete boiler system model

Author: Enhanced Boiler Modeling System
Version: 5.0 - Analysis and Visualization
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

from boiler_system import EnhancedCompleteBoilerSystem


class SystemAnalyzer:
    """Analysis and reporting utilities for boiler system performance."""
    
    def __init__(self, boiler_system: EnhancedCompleteBoilerSystem):
        """Initialize analyzer with boiler system reference."""
        self.system = boiler_system
    
    def print_comprehensive_summary(self):
        """Print comprehensive system performance summary."""
        print("\n" + "=" * 100)
        print("ENHANCED 100 MMBtu/hr SUBCRITICAL BOILER SYSTEM ANALYSIS")
        print("With Coal Combustion Integration and ML Dataset Generation")
        print("=" * 100)
        
        # System configuration
        print(f"\nSYSTEM CONFIGURATION:")
        print(f"  Design Capacity:         {self.system.design_capacity/1e6:.0f} MMBtu/hr")
        print(f"  Fuel Input:              {self.system.fuel_input/1e6:.1f} MMBtu/hr")
        print(f"  Flue Gas Mass Flow:      {self.system.flue_gas_mass_flow:,.0f} lbm/hr")
        print(f"  Furnace Exit Temperature: {self.system.furnace_exit_temp:.0f}°F")
        print(f"  Base Fouling Multiplier: {self.system.base_fouling_multiplier:.2f}")
        
        # System performance
        print(f"\nSYSTEM PERFORMANCE:")
        perf = self.system.system_performance
        print(f"  Total Heat Absorbed:     {perf['total_heat_absorbed']/1e6:.1f} MMBtu/hr")
        print(f"  System Efficiency:       {perf['system_efficiency']:.1%}")
        print(f"  Final Steam Temperature: {perf['final_steam_temperature']:.1f}°F")
        print(f"  Steam Superheat:         {perf['steam_superheat']:.1f}°F")
        print(f"  Stack Temperature:       {perf['stack_temperature']:.1f}°F")
        print(f"  Steam Production:        {perf['steam_production']:.0f} lbm/hr")
        print(f"  Attemperator Flow:       {self.system.attemperator_flow:.0f} lbm/hr")
        
        # Section breakdown
        print(f"\nSECTION HEAT TRANSFER BREAKDOWN:")
        print(f"{'Section':<25} {'Q (MMBtu/hr)':<12} {'Segments':<9} {'Gas ΔT':<9} {'Water ΔT':<10} {'Max Fouling':<12}")
        print("-" * 85)
        
        for section_name, data in self.system.section_results.items():
            summary = data['summary']
            gas_dt = summary['gas_temp_in'] - summary['gas_temp_out']
            water_dt = summary['water_temp_out'] - summary['water_temp_in']
            max_fouling = max(summary['max_gas_fouling'], summary['max_water_fouling'])
            
            print(f"{summary['section_name']:<25} {summary['total_heat_transfer']/1e6:<12.1f} "
                  f"{summary['num_segments']:<9} {gas_dt:<9.0f} {water_dt:<10.0f} {max_fouling:<12.4f}")
        
        print(f"\n✓ Using thermo library for accurate thermodynamic properties")
        print(f"✓ Coal combustion modeling with soot production prediction")
        print(f"✓ ML dataset generation for soot blowing optimization")
    
    def analyze_soot_blowing_effectiveness(self):
        """Analyze the effectiveness of soot blowing on different sections."""
        print(f"\n" + "=" * 100)
        print("SOOT BLOWING EFFECTIVENESS ANALYSIS")
        print("=" * 100)
        
        for section_name, data in self.system.section_results.items():
            segments = data['segments']
            if not segments:
                continue
            
            # Get current fouling arrays
            section = self.system.sections[section_name]
            fouling_arrays = section.get_current_fouling_arrays()
            
            # Analyze fouling distribution
            gas_fouling = fouling_arrays['gas']
            water_fouling = fouling_arrays['water']
            
            avg_gas_fouling = sum(gas_fouling) / len(gas_fouling)
            max_gas_fouling = max(gas_fouling)
            min_gas_fouling = min(gas_fouling)
            
            fouling_variation = (max_gas_fouling - min_gas_fouling) / avg_gas_fouling * 100
            
            print(f"\n{data['summary']['section_name'].upper()}:")
            print(f"  Number of segments: {len(segments)}")
            print(f"  Average gas fouling: {avg_gas_fouling:.5f} hr-ft²-°F/Btu")
            print(f"  Fouling range: {min_gas_fouling:.5f} to {max_gas_fouling:.5f}")
            print(f"  Fouling variation: {fouling_variation:.1f}%")
            
            # Identify segments needing cleaning
            threshold = avg_gas_fouling * 1.5  # 50% above average
            dirty_segments = [i for i, f in enumerate(gas_fouling) if f > threshold]
            
            if dirty_segments:
                print(f"  Segments needing cleaning: {dirty_segments}")
                print(f"  Recommended soot blowing: {len(dirty_segments)} of {len(segments)} segments")
            else:
                print(f"  All segments within acceptable fouling limits")
    
    def export_detailed_results(self, filename: str = "enhanced_boiler_analysis.txt"):
        """Export comprehensive analysis to file."""
        with open(filename, 'w') as f:
            f.write("ENHANCED BOILER SYSTEM ANALYSIS WITH COAL COMBUSTION INTEGRATION\n")
            f.write("=" * 70 + "\n")
            f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Property Library: Thermo (Comprehensive Thermodynamics)\n\n")
            
            # System configuration
            f.write("SYSTEM CONFIGURATION:\n")
            f.write(f"  Design Capacity: {self.system.design_capacity/1e6:.0f} MMBtu/hr\n")
            f.write(f"  Steam Pressure:  {self.system.steam_pressure} psia\n")
            f.write(f"  Gas Flow:        {self.system.flue_gas_mass_flow:,} lbm/hr\n")
            f.write(f"  Water Flow:      {self.system.feedwater_flow:,} lbm/hr\n\n")
            
            # Performance results
            f.write("PERFORMANCE RESULTS:\n")
            for key, value in self.system.system_performance.items():
                f.write(f"  {key.replace('_', ' ').title()}: ")
                if 'temp' in key.lower():
                    f.write(f"{value:.1f}°F\n")
                elif 'efficiency' in key.lower():
                    f.write(f"{value:.1%}\n")
                elif 'flow' in key.lower():
                    f.write(f"{value:,.0f} lbm/hr\n")
                elif 'heat' in key.lower():
                    f.write(f"{value/1e6:.1f} MMBtu/hr\n")
                else:
                    f.write(f"{value:.2f}\n")
            
            # Detailed section results
            f.write(f"\nDETAILED SECTION ANALYSIS:\n")
            f.write("=" * 70 + "\n")
            
            for section_name, data in self.system.section_results.items():
                summary = data['summary']
                segments = data['segments']
                
                f.write(f"\n{summary['section_name'].upper()}:\n")
                f.write(f"  Heat Transfer: {summary['total_heat_transfer']/1e6:.2f} MMBtu/hr\n")
                f.write(f"  Segments: {len(segments)}\n")
                f.write(f"  Gas: {summary['gas_temp_in']:.0f}°F → {summary['gas_temp_out']:.0f}°F\n")
                f.write(f"  Water: {summary['water_temp_in']:.0f}°F → {summary['water_temp_out']:.0f}°F\n")
                
                f.write(f"\n  Segment Details:\n")
                f.write(f"    Seg  Pos   Gas In  Gas Out  Water In  Water Out  Q(kBtu)  U      Gas Foul  Water Foul\n")
                f.write(f"    " + "-" * 85 + "\n")
                
                for seg in segments[:5]:  # Show first 5 segments
                    f.write(f"    {seg.segment_id:2d}  {seg.position:4.2f}  "
                           f"{seg.gas_temp_in:6.0f}  {seg.gas_temp_out:7.0f}  "
                           f"{seg.water_temp_in:8.0f}  {seg.water_temp_out:9.0f}  "
                           f"{seg.heat_transfer_rate/1000:7.0f}  {seg.overall_U:5.2f}  "
                           f"{seg.fouling_gas:8.5f}  {seg.fouling_water:10.5f}\n")
                
                if len(segments) > 5:
                    f.write(f"    ... and {len(segments)-5} more segments\n")
        
        print(f"\n✓ Results exported to: {filename}")
    
    def calculate_economic_metrics(self, fuel_cost_per_mmbtu: float = 5.0,
                                 electricity_cost_per_kwh: float = 0.08) -> Dict:
        """Calculate economic performance metrics."""
        perf = self.system.system_performance
        
        # Annual operating hours (typical for industrial boiler)
        annual_hours = 8000
        
        # Fuel costs
        annual_fuel_cost = (self.system.fuel_input / 1e6) * fuel_cost_per_mmbtu * annual_hours
        
        # Steam production value (simplified)
        steam_value_per_lb = 0.01  # $/lb (approximate)
        annual_steam_value = perf['steam_production'] * steam_value_per_lb * annual_hours
        
        # Efficiency impact on costs
        efficiency_factor = perf['system_efficiency'] / 0.85  # Normalized to 85% baseline
        adjusted_fuel_cost = annual_fuel_cost / efficiency_factor
        
        return {
            'annual_fuel_cost': annual_fuel_cost,
            'annual_steam_value': annual_steam_value,
            'efficiency_savings': annual_fuel_cost - adjusted_fuel_cost,
            'cost_per_mmbtu_steam': annual_fuel_cost / (perf['steam_production'] * annual_hours / 1000),
            'system_efficiency': perf['system_efficiency']
        }


class Visualizer:
    """Visualization utilities for boiler system analysis."""
    
    def __init__(self, boiler_system: EnhancedCompleteBoilerSystem):
        """Initialize visualizer with boiler system reference."""
        self.system = boiler_system
    
    def plot_comprehensive_profiles(self):
        """Generate comprehensive temperature and fouling profile plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Enhanced 100 MMBtu/hr Boiler System with Coal Combustion Integration', 
                    fontsize=16, fontweight='bold')
        
        # Configure axes
        self._configure_axes(ax1, 'Flue Gas Temperature Profile', 'Temperature (°F)')
        self._configure_axes(ax2, 'Water/Steam Temperature Profile', 'Temperature (°F)')
        self._configure_axes(ax3, 'Gas-Side Fouling Gradients', 'Fouling Factor (hr-ft²-°F/Btu)', log_scale=True)
        self._configure_axes(ax4, 'Water-Side Fouling Gradients', 'Fouling Factor (hr-ft²-°F/Btu)', log_scale=True)
        
        colors = ['#FF4444', '#FF8800', '#00AA00', '#0088FF', '#8800FF', '#AA6600', '#FF0088']
        position_offset = 0
        
        for i, (section_name, data) in enumerate(self.system.section_results.items()):
            segments = data['segments']
            if not segments:
                continue
            
            positions = np.array([position_offset + j for j in range(len(segments))])
            
            # Extract data
            gas_in = [s.gas_temp_in for s in segments]
            gas_out = [s.gas_temp_out for s in segments]
            water_in = [s.water_temp_in for s in segments]
            water_out = [s.water_temp_out for s in segments]
            gas_fouling = [s.fouling_gas for s in segments]
            water_fouling = [s.fouling_water for s in segments]
            
            color = colors[i % len(colors)]
            label = data['summary']['section_name'].replace('_', ' ').title()
            
            # Plot profiles
            ax1.plot(positions, gas_in, f'{color}-o', label=f'{label} In', linewidth=2, markersize=4)
            ax1.plot(positions, gas_out, f'{color}--s', label=f'{label} Out', linewidth=2, markersize=3)
            
            ax2.plot(positions, water_in, f'{color}-o', label=f'{label} In', linewidth=2, markersize=4)
            ax2.plot(positions, water_out, f'{color}--s', label=f'{label} Out', linewidth=2, markersize=3)
            
            ax3.plot(positions, gas_fouling, f'{color}-', label=label, linewidth=3, marker='o', markersize=4)
            ax4.plot(positions, water_fouling, f'{color}-', label=label, linewidth=3, marker='s', markersize=4)
            
            position_offset += len(segments) + 2
        
        # Add legends and performance text
        for ax in [ax1, ax2, ax3, ax4]:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Performance summary
        perf = self.system.system_performance
        perf_text = (f"System Efficiency: {perf['system_efficiency']:.1%}\n"
                    f"Steam Temperature: {perf['final_steam_temperature']:.0f}°F\n"
                    f"Stack Temperature: {perf['stack_temperature']:.0f}°F\n"
                    f"Steam Superheat: {perf['steam_superheat']:.0f}°F")
        
        fig.text(0.02, 0.02, perf_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        # Integration indicator
        fig.text(0.98, 0.02, "✓ Coal Combustion + Soot Production + ML Dataset", fontsize=10, ha='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1, right=0.85)
        plt.show()
    
    def plot_fouling_analysis(self):
        """Plot detailed fouling analysis for all sections."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Fouling Distribution Analysis', fontsize=16, fontweight='bold')
        
        section_names = []
        avg_gas_fouling = []
        max_gas_fouling = []
        fouling_variation = []
        
        for section_name, data in self.system.section_results.items():
            segments = data['segments']
            if not segments:
                continue
            
            section = self.system.sections[section_name]
            fouling_arrays = section.get_current_fouling_arrays()
            gas_fouling = fouling_arrays['gas']
            
            section_names.append(data['summary']['section_name'].replace('_', '\n'))
            avg_gas_fouling.append(np.mean(gas_fouling))
            max_gas_fouling.append(np.max(gas_fouling))
            fouling_variation.append(np.std(gas_fouling) / np.mean(gas_fouling) * 100)
        
        # Bar chart of average and max fouling
        x = np.arange(len(section_names))
        width = 0.35
        
        ax1.bar(x - width/2, avg_gas_fouling, width, label='Average Fouling', alpha=0.8)
        ax1.bar(x + width/2, max_gas_fouling, width, label='Maximum Fouling', alpha=0.8)
        ax1.set_xlabel('Boiler Section')
        ax1.set_ylabel('Fouling Factor (hr-ft²-°F/Btu)')
        ax1.set_title('Gas-Side Fouling Levels by Section')
        ax1.set_xticks(x)
        ax1.set_xticklabels(section_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Fouling variation chart
        ax2.bar(section_names, fouling_variation, alpha=0.8, color='orange')
        ax2.set_xlabel('Boiler Section')
        ax2.set_ylabel('Fouling Variation (%)')
        ax2.set_title('Fouling Distribution Uniformity')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_performance_trends(self, performance_history: list):
        """Plot performance trends over time (for multiple runs)."""
        if not performance_history:
            print("No performance history available for trending")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Boiler Performance Trends', fontsize=16, fontweight='bold')
        
        # Extract data
        timestamps = [i for i in range(len(performance_history))]
        efficiencies = [p['system_efficiency'] for p in performance_history]
        steam_temps = [p['final_steam_temperature'] for p in performance_history]
        stack_temps = [p['stack_temperature'] for p in performance_history]
        heat_absorbed = [p['total_heat_absorbed']/1e6 for p in performance_history]
        
        # Plot trends
        ax1.plot(timestamps, efficiencies, 'b-o', linewidth=2, markersize=4)
        ax1.set_ylabel('System Efficiency')
        ax1.set_title('Efficiency Trend')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        ax2.plot(timestamps, steam_temps, 'r-s', linewidth=2, markersize=4)
        ax2.set_ylabel('Steam Temperature (°F)')
        ax2.set_title('Steam Temperature Trend')
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(timestamps, stack_temps, 'g-^', linewidth=2, markersize=4)
        ax3.set_xlabel('Run Number')
        ax3.set_ylabel('Stack Temperature (°F)')
        ax3.set_title('Stack Temperature Trend')
        ax3.grid(True, alpha=0.3)
        
        ax4.plot(timestamps, heat_absorbed, 'm-d', linewidth=2, markersize=4)
        ax4.set_xlabel('Run Number')
        ax4.set_ylabel('Heat Absorbed (MMBtu/hr)')
        ax4.set_title('Heat Absorption Trend')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _configure_axes(self, ax, title: str, ylabel: str, log_scale: bool = False):
        """Configure individual axis properties."""
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Position Through Boiler System')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if log_scale:
            ax.set_yscale('log')
    
    def plot_section_comparison(self, section_names: list):
        """Plot detailed comparison of specific sections."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Section Comparison', fontsize=16, fontweight='bold')
        
        ax1, ax2, ax3, ax4 = axes.flatten()
        
        for section_name in section_names:
            if section_name not in self.system.section_results:
                continue
            
            segments = self.system.section_results[section_name]['segments']
            if not segments:
                continue
            
            # Extract segment data
            positions = [s.position for s in segments]
            gas_temps_in = [s.gas_temp_in for s in segments]
            water_temps_in = [s.water_temp_in for s in segments]
            heat_transfer_rates = [s.heat_transfer_rate/1000 for s in segments]  # kBtu/hr
            overall_U = [s.overall_U for s in segments]
            
            label = section_name.replace('_', ' ').title()
            
            # Plot comparisons
            ax1.plot(positions, gas_temps_in, '-o', label=f'{label} Gas', linewidth=2)
            ax1.plot(positions, water_temps_in, '--s', label=f'{label} Water', linewidth=2)
            
            ax2.plot(positions, heat_transfer_rates, '-^', label=label, linewidth=2)
            
            ax3.plot(positions, overall_U, '-d', label=label, linewidth=2)
            
            # Fouling data
            section = self.system.sections[section_name]
            fouling_arrays = section.get_current_fouling_arrays()
            ax4.plot(positions, fouling_arrays['gas'], '-o', label=label, linewidth=2)
        
        # Configure subplots
        ax1.set_title('Temperature Profiles')
        ax1.set_ylabel('Temperature (°F)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Heat Transfer Rate')
        ax2.set_ylabel('Heat Transfer (kBtu/hr)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.set_title('Overall Heat Transfer Coefficient')
        ax3.set_xlabel('Relative Position')
        ax3.set_ylabel('Overall U (Btu/hr-ft²-°F)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.set_title('Gas-Side Fouling')
        ax4.set_xlabel('Relative Position')
        ax4.set_ylabel('Fouling Factor (hr-ft²-°F/Btu)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.show()