        # Economic calculations
        fuel_savings_per_hour = efficiency_gain * scenario['coal_lb_per_hr'] * 12000 * 5.0 / 1e6  # $/hr
        payback_hours = cleaning_cost / fuel_savings_per_hour if fuel_savings_per_hour > 0 else 9999
        
        # Compile feature set
        features = self._extract_features(scenario, combustion_data, before_fouling, cleaning)
        
        # Compile targets
        targets = {
            'efficiency_gain': efficiency_gain,
            'heat_transfer_gain': heat_transfer_gain,
            'fuel_savings_per_hour': fuel_savings_per_hour,
            'cleaning_cost': cleaning_cost,
            'cleaning_time': cleaning_time,
            'payback_hours': min(payback_hours, 9999),  # Cap at reasonable value
            'roi_24hr': fuel_savings_per_hour * 24 / cleaning_cost if cleaning_cost > 0 else 0,
            'performance_score': self._calculate_performance_score(efficiency_gain, heat_transfer_gain, cleaning_cost)
        }
        
        # Combine features and targets
        return {**features, **targets}
    
    def _apply_cleaning(self, fouling_state: Dict, cleaning: Dict) -> Dict:
        """Apply cleaning scenario to fouling state."""
        after_fouling = {}
        
        for section_name, fouling in fouling_state.items():
            after_fouling[section_name] = {
                'gas': fouling['gas'].copy(),
                'water': fouling['water'].copy()
            }
            
            # Apply cleaning if this section is included
            if section_name in cleaning['sections_to_clean']:
                segments_to_clean = cleaning['segments_per_section'].get(section_name, [])
                effectiveness = cleaning['cleaning_effectiveness']
                
                for seg_id in segments_to_clean:
                    if seg_id < len(after_fouling[section_name]['gas']):
                        after_fouling[section_name]['gas'][seg_id] *= (1 - effectiveness)
                        after_fouling[section_name]['water'][seg_id] *= (1 - effectiveness)
        
        return after_fouling
    
    def _calculate_cleaning_cost(self, cleaning: Dict) -> float:
        """Calculate cost of cleaning scenario."""
        base_cost_per_segment = 150  # $ per segment
        fixed_cost_per_section = 500  # $ per section (setup, crew, etc.)
        
        total_segments = sum(len(segments) for segments in cleaning['segments_per_section'].values())
        total_sections = len(cleaning['sections_to_clean'])
        
        return total_sections * fixed_cost_per_section + total_segments * base_cost_per_segment
    
    def _calculate_cleaning_time(self, cleaning: Dict) -> float:
        """Calculate time required for cleaning scenario."""
        time_per_segment = 0.25  # hours per segment
        setup_time_per_section = 1.0  # hours per section
        
        total_segments = sum(len(segments) for segments in cleaning['segments_per_section'].values())
        total_sections = len(cleaning['sections_to_clean'])
        
        return total_sections * setup_time_per_section + total_segments * time_per_segment
    
    def _calculate_performance_score(self, efficiency_gain: float, heat_transfer_gain: float, 
                                   cleaning_cost: float) -> float:
        """Calculate overall performance score for cleaning scenario."""
        benefit = efficiency_gain * 1000 + heat_transfer_gain * 500  # Weighted benefits
        cost_factor = cleaning_cost / 1000  # Normalize cost
        return max(0, benefit - cost_factor)
    
    def _extract_features(self, scenario: Dict, combustion_data: Dict, 
                         fouling_state: Dict, cleaning: Dict) -> Dict:
        """Extract comprehensive feature set for ML model."""
        features = {}
        
        # Operating condition features
        features.update({
            'coal_rate_lb_hr': scenario['coal_lb_per_hr'],
            'air_scfh': scenario['air_scfh'],
            'excess_air_ratio': scenario['air_scfh'] / (scenario['coal_lb_per_hr'] * 10),  # Simplified
            'nox_efficiency': scenario['NOx_eff'],
            'air_temp_F': scenario['air_temp_F'],
            'air_humidity_pct': scenario['air_RH_pct'],
            'operating_hours': scenario['operating_hours_since_clean']
        })
        
        # Coal property features
        features.update({
            'coal_carbon_pct': scenario['coal_properties']['carbon'],
            'coal_volatile_matter_pct': scenario['coal_properties']['volatile_matter'],
            'coal_fixed_carbon_pct': scenario['coal_properties']['fixed_carbon'],
            'coal_sulfur_pct': scenario['coal_properties']['sulfur'],
            'coal_ash_pct': scenario['coal_properties']['ash']
        })
        
        # Combustion result features
        features.update({
            'thermal_nox_lb_hr': combustion_data['thermal_nox'],
            'fuel_nox_lb_hr': combustion_data['fuel_nox'],
            'total_nox_lb_hr': combustion_data['total_nox'],
            'excess_o2_pct': combustion_data['excess_o2'],
            'combustion_efficiency': combustion_data['combustion_efficiency'],
            'flame_temp_F': combustion_data['flame_temp'],
            'heat_release_btu_hr': combustion_data['heat_release']
        })
        
        # Fouling state features (statistical summaries per section)
        for section_name, fouling in fouling_state.items():
            prefix = f"fouling_{section_name}_"
            gas_fouling = fouling['gas']
            
            features.update({
                f"{prefix}avg_gas": np.mean(gas_fouling),
                f"{prefix}max_gas": np.max(gas_fouling),
                f"{prefix}std_gas": np.std(gas_fouling),
                f"{prefix}range_gas": np.max(gas_fouling) - np.min(gas_fouling),
                f"{prefix}segments": len(gas_fouling),
                f"{prefix}above_threshold": sum(1 for f in gas_fouling if f > np.mean(gas_fouling) * 1.2)
            })
        
        # Cleaning strategy features
        features.update({
            'clean_all_sections': 1 if cleaning['strategy'] == 'clean_all' else 0,
            'clean_worst_only': 1 if cleaning['strategy'] == 'clean_worst' else 0,
            'selective_cleaning': 1 if cleaning['strategy'] == 'selective_segments' else 0,
            'economizer_priority': 1 if cleaning['strategy'] == 'economizers_first' else 0,
            'no_cleaning': 1 if cleaning['strategy'] == 'no_cleaning' else 0,
            'sections_to_clean_count': len(cleaning['sections_to_clean']),
            'total_segments_to_clean': sum(len(segments) for segments in cleaning['segments_per_section'].values()),
            'cleaning_effectiveness': cleaning['cleaning_effectiveness']
        })
        
        # Seasonal and operational context
        season_encoding = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
        fuel_quality_encoding = {'low': 0, 'medium': 1, 'high': 2}
        
        features.update({
            'season_code': season_encoding[scenario['season']],
            'fuel_quality_code': fuel_quality_encoding[scenario['fuel_quality']],
            'is_winter': 1 if scenario['season'] == 'winter' else 0,
            'is_high_load': 1 if scenario['coal_lb_per_hr'] > 9000 else 0,
            'is_low_nox': 1 if combustion_data['total_nox'] < 50 else 0
        })
        
        return features


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
    
    def _configure_axes(self, ax, title: str, ylabel: str, log_scale: bool = False):
        """Configure individual axis properties."""
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Position Through Boiler System')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if log_scale:
            ax.set_yscale('log')


def demonstrate_combustion_fouling_integration():
    """Demonstrate the integrated combustion-fouling-ML system."""
    print("=" * 100)
    print("COMBUSTION-FOULING INTEGRATION FOR ML DATASET GENERATION")
    print("Advanced Soot Blowing Optimization with Coal Combustion Modeling")
    print("=" * 100)
    
    # Initialize boiler system
    print("\n🔧 Initializing Enhanced Boiler System...")
    boiler = EnhancedCompleteBoilerSystem(
        fuel_input=100e6,
        flue_gas_mass_flow=84000,
        furnace_exit_temp=3000,
        base_fouling_multiplier=1.0
    )
    
    # Sample coal properties
    coal_properties = {
        'carbon': 72.0,
        'volatile_matter': 28.0,
        'fixed_carbon': 55.0,
        'sulfur': 1.2,
        'ash': 8.5
    }
    
    ultimate_analysis = {
        'C': 72.0, 'H': 5.0, 'O': 10.0, 'N': 1.5, 'S': 1.2, 'Ash': 8.5, 'Moisture': 1.8
    }
    
    # Test different combustion scenarios
    scenarios = [
        {
            'name': 'Optimal Combustion',
            'coal_rate': 8500, 'air_scfh': 900000, 'nox_eff': 0.35,
            'description': 'Clean combustion with optimal air/fuel ratio'
        },
        {
            'name': 'Rich Combustion (High Soot)',
            'coal_rate': 8500, 'air_scfh': 750000, 'nox_eff': 0.45,
            'description': 'Fuel-rich conditions leading to high soot production'
        },
        {
            'name': 'Lean Combustion (Low Soot)',
            'coal_rate': 8500, 'air_scfh': 1100000, 'nox_eff': 0.25,
            'description': 'Excess air conditions with reduced soot formation'
        },
        {
            'name': 'High Load Operation',
            'coal_rate': 10200, 'air_scfh': 1080000, 'nox_eff': 0.40,
            'description': 'High load with increased thermal stress'
        },
        {
            'name': 'Low Quality Coal',
            'coal_rate': 9000, 'air_scfh': 950000, 'nox_eff': 0.50,
            'description': 'Poor coal quality increasing soot production'
        }
    ]
    
    print(f"\n📊 Testing {len(scenarios)} Combustion Scenarios...")
    
    # Initialize fouling integrator
    fouling_integrator = CombustionFoulingIntegrator()
    
    results_comparison = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\n{'-'*60}")
        print(f"SCENARIO {i+1}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"{'-'*60}")
        
        # Create combustion model
        combustion_model = CoalCombustionModel(
            ultimate_analysis=ultimate_analysis,
            coal_lb_per_hr=scenario['coal_rate'],
            air_scfh=scenario['air_scfh'],
            NOx_eff=scenario['nox_eff']
        )
        
        # Calculate combustion
        combustion_model.calculate()
        
        # Calculate soot production
        soot_model = SootProductionModel()
        soot_data = soot_model.calculate_soot_production(combustion_model, coal_properties)
        
        # Calculate section fouling rates
        fouling_rates = fouling_integrator.calculate_section_fouling_rates(
            combustion_model, coal_properties, boiler
        )
        
        # Display results
        print(f"Combustion Results:")
        print(f"  Thermal NOx: {combustion_model.NO_thermal_lb_per_hr:.2f} lb/hr")
        print(f"  Fuel NOx: {combustion_model.NO_fuel_lb_per_hr:.2f} lb/hr")
        print(f"  Total NOx: {combustion_model.NO_total_lb_per_hr:.2f} lb/hr")
        print(f"  Excess O2: {combustion_model.dry_O2_pct:.1f}%")
        print(f"  Combustion Efficiency: {combustion_model.combustion_efficiency:.3f}")
        print(f"  Flame Temperature: {combustion_model.flame_temp_F:.0f}°F")
        
        print(f"\nSoot Production:")
        print(f"  Mass Rate: {soot_data.mass_production_rate:.4f} lb/hr")
        print(f"  Particle Size: {soot_data.particle_size_microns:.2f} μm")
        print(f"  Carbon Content: {soot_data.carbon_content:.1%}")
        print(f"  Deposition Tendency: {soot_data.deposition_tendency:.3f}")
        
        print(f"\nWorst Fouling Rates (Economizer Secondary):")
        econ_rates = fouling_rates.get('economizer_secondary', {'gas': [0], 'water': [0]})
        print(f"  Max Gas-side Rate: {max(econ_rates['gas']):.2e} hr-ft²-°F/Btu per hour")
        print(f"  Avg Gas-side Rate: {np.mean(econ_rates['gas']):.2e} hr-ft²-°F/Btu per hour")
        
        # Store for comparison
        results_comparison.append({
            'name': scenario['name'],
            'thermal_nox': combustion_model.NO_thermal_lb_per_hr,
            'fuel_nox': combustion_model.NO_fuel_lb_per_hr,
            'total_nox': combustion_model.NO_total_lb_per_hr,
            'excess_o2': combustion_model.dry_O2_pct,
            'combustion_eff': combustion_model.combustion_efficiency,
            'soot_rate': soot_data.mass_production_rate,
            'deposition': soot_data.deposition_tendency,
            'max_fouling_rate': max(econ_rates['gas']) if econ_rates['gas'] else 0
        })
    
    # Print comparison table
    print(f"\n" + "=" * 120)
    print("COMBUSTION SCENARIO COMPARISON")
    print("=" * 120)
    
    print(f"{'Scenario':<25} {'Thermal NOx':<12} {'Fuel NOx':<10} {'Excess O2':<10} {'Comb Eff':<10} {'Soot Rate':<12} {'Fouling Rate':<15}")
    print(f"{'Name':<25} {'(lb/hr)':<12} {'(lb/hr)':<10} {'(%)':<10} {'(--)':<10} {'(lb/hr)':<12} {'(hr-ft²-°F/Btu/hr)':<15}")
    print("-" * 120)
    
    for result in results_comparison:
        print(f"{result['name']:<25} {result['thermal_nox']:<12.2f} {result['fuel_nox']:<10.2f} "
              f"{result['excess_o2']:<10.1f} {result['combustion_eff']:<10.3f} "
              f"{result['soot_rate']:<12.4f} {result['max_fouling_rate']:<15.2e}")
    
    # Demonstrate ML dataset generation
    print(f"\n" + "=" * 100)
    print("ML DATASET GENERATION DEMONSTRATION")
    print("=" * 100)
    
    print(f"\n🤖 Generating Sample ML Dataset...")
    
    # Generate smaller dataset for demonstration
    ml_generator = MLDatasetGenerator(boiler)
    sample_dataset = ml_generator.generate_comprehensive_dataset(num_scenarios=100)
    
    print(f"\n📊 Dataset Summary:")
    print(f"  Total Records: {len(sample_dataset)}")
    print(f"  Features: {len([col for col in sample_dataset.columns if col not in ['efficiency_gain', 'heat_transfer_gain', 'fuel_savings_per_hour', 'cleaning_cost', 'cleaning_time', 'payback_hours', 'roi_24hr', 'performance_score']])}")
    print(f"  Target Variables: 8")
    
    # Show feature importance analysis
    print(f"\n📈 Sample Feature Analysis:")
    
    # Key combustion features
    combustion_features = ['thermal_nox_lb_hr', 'fuel_nox_lb_hr', 'excess_o2_pct', 'combustion_efficiency']
    print(f"\nCombustion Features Statistics:")
    for feature in combustion_features:
        if feature in sample_dataset.columns:
            values = sample_dataset[feature]
            print(f"  {feature}: {values.mean():.3f} ± {values.std():.3f} (range: {values.min():.3f} to {values.max():.3f})")
    
    # Target variable analysis
    target_features = ['efficiency_gain', 'payback_hours', 'roi_24hr', 'performance_score']
    print(f"\nTarget Variables Statistics:")
    for target in target_features:
        if target in sample_dataset.columns:
            values = sample_dataset[target]
            print(f"  {target}: {values.mean():.3f} ± {values.std():.3f} (range: {values.min():.3f} to {values.max():.3f})")
    
    # Show correlation with soot production factors
    print(f"\n🔗 Soot Production Correlations:")
    soot_indicators = ['thermal_nox_lb_hr', 'excess_o2_pct', 'combustion_efficiency']
    fouling_indicators = [col for col in sample_dataset.columns if 'fouling_' in col and '_avg_gas' in col]
    
    if fouling_indicators:
        fouling_avg = sample_dataset[fouling_indicators].mean(axis=1)
        for indicator in soot_indicators:
            if indicator in sample_dataset.columns:
                correlation = sample_dataset[indicator].corr(fouling_avg)
                print(f"  {indicator} vs Avg Fouling: {correlation:.3f}")
    
    # Export sample dataset
    output_filename = f"soot_blowing_optimization_dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    sample_dataset.to_csv(output_filename, index=False)
    print(f"\n💾 Sample dataset exported to: {output_filename}")
    
    return sample_dataset


def demonstrate_ml_model_preparation():
    """Demonstrate ML model preparation and training approach."""
    print(f"\n" + "=" * 100)
    print("ML MODEL PREPARATION AND TRAINING APPROACH")
    print("=" * 100)
    
    print(f"""
🎯 RECOMMENDED ML MODELING APPROACH:

1. MULTI-TARGET REGRESSION MODELS:
   - Primary Target: ROI Score (efficiency_gain / cleaning_cost)
   - Secondary Targets: Payback hours, Performance score
   - Features: Combustion conditions, fouling state, cleaning strategy

2. TIME SERIES PREDICTION:
   - LSTM Model for fouling progression over time
   - Features: Operating history, combustion patterns, seasonal effects
   - Target: Fouling buildup rate per segment

3. REINFORCEMENT LEARNING:
   - Agent: Soot blowing controller
   - State: Current fouling levels, combustion conditions, costs
   - Actions: Which segments to clean, when to clean
   - Reward: Long-term performance improvement vs cleaning costs

4. ENSEMBLE APPROACH:
   - Random Forest: Feature importance and non-linear relationships
   - XGBoost: High-performance prediction with feature interactions
   - Neural Network: Complex pattern recognition in high-dimensional data

5. OPTIMIZATION FRAMEWORK:
   - Multi-objective optimization (efficiency vs cost vs time)
   - Constraint satisfaction (max cleaning time, budget limits)
   - Real-time decision making based on current conditions

📊 DATASET SCALING RECOMMENDATIONS:
   - Target Size: 50,000+ scenarios for robust training
   - Validation: 80/10/10 split (train/validation/test)
   - Cross-validation: Time-based splits to avoid data leakage
   - Augmentation: Synthetic scenarios with controlled variations

🔄 REAL-TIME IMPLEMENTATION:
   - Online learning: Update models with new operational data
   - A/B Testing: Compare ML recommendations vs current practice
   - Safety constraints: Never recommend unsafe operating conditions
   - Performance monitoring: Track actual vs predicted improvements
    """)


def demonstrate_soot_blowing_simulation():
    """Comprehensive demonstration of soot blowing simulation capabilities."""
    print("=" * 100)
    print("SOOT BLOWING SIMULATION DEMONSTRATION")
    print("100 MMBtu/hr Boiler with Individual Segment Fouling Control")
    print("=" * 100)
    
    # Initialize system with moderate fouling
    print("\n🔧 Initializing System with Moderate Fouling...")
    boiler = EnhancedCompleteBoilerSystem(
        fuel_input=100e6,
        flue_gas_mass_flow=84000,
        furnace_exit_temp=3000,
        base_fouling_multiplier=1.5  # Moderate fouling
    )
    
    analyzer = SystemAnalyzer(boiler)
    
    # Solve baseline case
    print("\n🔄 Solving Baseline Case (Before Soot Blowing)...")
    baseline_results = boiler.solve_enhanced_system(max_iterations=10, tolerance=5.0)
    baseline_perf = boiler.system_performance.copy()
    
    print(f"Baseline Performance:")
    print(f"  Efficiency: {baseline_perf['system_efficiency']:.1%}")
    print(f"  Steam Temperature: {baseline_perf['final_steam_temperature']:.1f}°F")
    print(f"  Stack Temperature: {baseline_perf['stack_temperature']:.1f}°F")
    
    # Simulate progressive fouling buildup on economizer
    print(f"\n🕒 Simulating 720 Hours of Operation (30 days)...")
    economizer_section = boiler.sections['economizer_primary']
    economizer_section.simulate_fouling_buildup(720, fouling_rate_per_hour=0.002)
    
    # Solve with increased fouling
    fouled_results = boiler.solve_enhanced_system(max_iterations=10, tolerance=5.0)
    fouled_perf = boiler.system_performance.copy()
    
    print(f"After Fouling Buildup:")
    print(f"  Efficiency: {fouled_perf['system_efficiency']:.1%} (Δ{(fouled_perf['system_efficiency'] - baseline_perf['system_efficiency'])/baseline_perf['system_efficiency']*100:+.1f}%)")
    print(f"  Steam Temperature: {fouled_perf['final_steam_temperature']:.1f}°F (Δ{fouled_perf['final_steam_temperature'] - baseline_perf['final_steam_temperature']:+.1f}°F)")
    print(f"  Stack Temperature: {fouled_perf['stack_temperature']:.1f}°F (Δ{fouled_perf['stack_temperature'] - baseline_perf['stack_temperature']:+.1f}°F)")
    
    # Demonstrate targeted soot blowing
    print(f"\n💨 Applying Targeted Soot Blowing...")
    
    # Get current fouling state
    current_fouling = economizer_section.get_current_fouling_arrays()
    print(f"Current fouling levels in economizer:")
    for i, (gas_foul, water_foul) in enumerate(zip(current_fouling['gas'], current_fouling['water'])):
        print(f"  Segment {i}: Gas={gas_foul:.5f}, Water={water_foul:.5f}")
    
    # Apply soot blowing to worst segments (assuming segments 3-6 are dirtiest)
    dirty_segments = [3, 4, 5, 6]
    economizer_section.apply_soot_blowing(dirty_segments, cleaning_effectiveness=0.8)
    
    # Show fouling after cleaning
    cleaned_fouling = economizer_section.get_current_fouling_arrays()
    print(f"\nFouling after soot blowing (segments {dirty_segments}):")
    for i, (gas_foul, water_foul) in enumerate(zip(cleaned_fouling['gas'], cleaned_fouling['water'])):
        change_marker = " ✓" if i in dirty_segments else ""
        print(f"  Segment {i}: Gas={gas_foul:.5f}, Water={water_foul:.5f}{change_marker}")
    
    # Solve after soot blowing
    print(f"\n🔄 Solving After Soot Blowing...")
    cleaned_results = boiler.solve_enhanced_system(max_iterations=10, tolerance=5.0)
    cleaned_perf = boiler.system_performance.copy()
    
    print(f"After Soot Blowing:")
    print(f"  Efficiency: {cleaned_perf['system_efficiency']:.1%} (Δ{(cleaned_perf['system_efficiency'] - fouled_perf['system_efficiency'])/fouled_perf['system_efficiency']*100:+.1f}%)")
    print(f"  Steam Temperature: {cleaned_perf['final_steam_temperature']:.1f}°F (Δ{cleaned_perf['final_steam_temperature'] - fouled_perf['final_steam_temperature']:+.1f}°F)")
    print(f"  Stack Temperature: {cleaned_perf['stack_temperature']:.1f}°F (Δ{cleaned_perf['stack_temperature'] - fouled_perf['stack_temperature']:+.1f}°F)")
    
    # Performance comparison table
    print(f"\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    print(f"{'Condition':<20} {'Efficiency':<10} {'Steam T (°F)':<12} {'Stack T (°F)':<12} {'Heat Absorbed':<15}")
    print("-" * 75)
    print(f"{'Baseline':<20} {baseline_perf['system_efficiency']:<10.1%} {baseline_perf['final_steam_temperature']:<12.1f} "
          f"{baseline_perf['stack_temperature']:<12.1f} {baseline_perf['total_heat_absorbed']/1e6:<15.1f}")
    print(f"{'After Fouling':<20} {fouled_perf['system_efficiency']:<10.1%} {fouled_perf['final_steam_temperature']:<12.1f} "
          f"{fouled_perf['stack_temperature']:<12.1f} {fouled_perf['total_heat_absorbed']/1e6:<15.1f}")
    print(f"{'After Cleaning':<20} {cleaned_perf['system_efficiency']:<10.1%} {cleaned_perf['final_steam_temperature']:<12.1f} "
          f"{cleaned_perf['stack_temperature']:<12.1f} {cleaned_perf['total_heat_absorbed']/1e6:<15.1f}")
    
    # Calculate cleaning effectiveness
    fouling_loss = (baseline_perf['system_efficiency'] - fouled_perf['system_efficiency']) / baseline_perf['system_efficiency']
    cleaning_recovery = (cleaned_perf['system_efficiency'] - fouled_perf['system_efficiency']) / baseline_perf['system_efficiency']
    cleaning_effectiveness = cleaning_recovery / fouling_loss if fouling_loss > 0 else 0
    
    print(f"\nCLEANING EFFECTIVENESS ANALYSIS:")
    print(f"  Fouling Impact: {fouling_loss * 100:.2f}% efficiency loss")
    print(f"  Cleaning Recovery: {cleaning_recovery * 100:.2f}% efficiency recovered")
    print(f"  Cleaning Effectiveness: {cleaning_effectiveness * 100:.1f}% of fouling impact recovered")


def demonstrate_advanced_fouling_scenarios():
    """Demonstrate advanced fouling scenarios and cleaning strategies."""
    print("\n" + "=" * 100)
    print("ADVANCED FOULING SCENARIOS")
    print("=" * 100)
    
    # Initialize clean system
    boiler = EnhancedCompleteBoilerSystem(
        fuel_input=100e6,
        flue_gas_mass_flow=84000,
        furnace_exit_temp=3000,
        base_fouling_multiplier=0.5  # Start clean
    )
    
    scenarios = [
        {
            'name': 'Scenario 1: Uniform Heavy Fouling',
            'description': 'All sections heavily fouled',
            'action': lambda: apply_uniform_fouling(boiler, 2.5)
        },
        {
            'name': 'Scenario 2: Gradient Fouling Pattern',
            'description': 'Progressive fouling from inlet to outlet',
            'action': lambda: apply_gradient_fouling(boiler)
        },
        {
            'name': 'Scenario 3: Selective Cleaning',
            'description': 'Clean only critical sections',
            'action': lambda: apply_selective_cleaning(boiler)
        },
        {
            'name': 'Scenario 4: Progressive Cleaning',
            'description': 'Sequential cleaning of sections',
            'action': lambda: apply_progressive_cleaning(boiler)
        }
    ]
    
    results_comparison = []
    
    for scenario in scenarios:
        print(f"\n{'-'*60}")
        print(f"{scenario['name']}: {scenario['description']}")
        print(f"{'-'*60}")
        
        try:
            # Apply scenario
            scenario['action']()
            
            # Solve system
            results = boiler.solve_enhanced_system(max_iterations=8, tolerance=8.0)
            perf = boiler.system_performance
            
            results_comparison.append({
                'name': scenario['name'],
                'efficiency': perf['system_efficiency'],
                'steam_temp': perf['final_steam_temperature'],
                'stack_temp': perf['stack_temperature']
            })
            
            print(f"Results:")
            print(f"  Efficiency: {perf['system_efficiency']:.1%}")
            print(f"  Steam Temperature: {perf['final_steam_temperature']:.1f}°F")
            print(f"  Stack Temperature: {perf['stack_temperature']:.1f}°F")
            
        except Exception as e:
            print(f"Error in {scenario['name']}: {e}")
            results_comparison.append({
                'name': scenario['name'],
                'efficiency': 0,
                'steam_temp': 0,
                'stack_temp': 0
            })
    
    # Print comparison
    print(f"\n" + "=" * 80)
    print("FOULING SCENARIO COMPARISON")
    print("=" * 80)
    
    print(f"{'Scenario':<35} {'Efficiency':<10} {'Steam T':<9} {'Stack T':<9}")
    print("-" * 70)
    
    for result in results_comparison:
        if result['efficiency'] > 0:
            print(f"{result['name']:<35} {result['efficiency']:<10.1%} {result['steam_temp']:<9.0f} {result['stack_temp']:<9.0f}")
        else:
            print(f"{result['name']:<35} {'FAILED':<10} {'---':<9} {'---':<9}")


def apply_uniform_fouling(boiler, fouling_multiplier):
    """Apply uniform fouling to all sections."""
    for section in boiler.sections.values():
        # Create heavily fouled arrays
        fouling_arrays = SootBlowingSimulator.create_gradient_fouling_array(
            section.num_segments, 
            section.base_fouling_gas * fouling_multiplier,
            section.base_fouling_water * fouling_multiplier,
            multiplier=1.0
        )
        section.set_custom_fouling_arrays(fouling_arrays['gas'], fouling_arrays['water'])


def apply_gradient_fouling(boiler):
    """Apply gradient fouling pattern from clean inlet to dirty outlet."""
    for section_name, section in boiler.sections.items():
        # Create gradient with increasing fouling toward outlet
        fouling_arrays = SootBlowingSimulator.create_gradient_fouling_array(
            section.num_segments,
            section.base_fouling_gas,
            section.base_fouling_water,
            multiplier=1.5,
            inlet_factor=0.3,  # Clean inlet
            outlet_factor=3.0  # Dirty outlet
        )
        section.set_custom_fouling_arrays(fouling_arrays['gas'], fouling_arrays['water'])


def apply_selective_cleaning(boiler):
    """Apply fouling then clean only critical sections."""
    # First apply heavy fouling everywhere
    apply_uniform_fouling(boiler, 2.0)
    
    # Then clean critical sections (superheaters and economizers)
    critical_sections = ['superheater_primary', 'superheater_secondary', 'economizer_primary']
    
    for section_name in critical_sections:
        if section_name in boiler.sections:
            section = boiler.sections[section_name]
            # Clean all segments in critical sections
            all_segments = list(range(section.num_segments))
            section.apply_soot_blowing(all_segments, cleaning_effectiveness=0.9)


def apply_progressive_cleaning(boiler):
    """Apply fouling then progressively clean sections."""
    # Start with heavy fouling
    apply_uniform_fouling(boiler, 2.5)
    
    # Progressive cleaning - clean different amounts of each section
    cleaning_strategies = {
        'economizer_secondary': {'segments': [0, 1, 2], 'effectiveness': 0.95},  # Clean inlet
        'economizer_primary': {'segments': [2, 3, 4, 5], 'effectiveness': 0.85},  # Clean middle
        'superheater_primary': {'segments': list(range(4)), 'effectiveness': 0.90},  # Clean most
        'generating_bank': {'segments': [1, 3, 5], 'effectiveness': 0.75}  # Partial clean
    }
    
    for section_name, strategy in cleaning_strategies.items():
        if section_name in boiler.sections:
            section = boiler.sections[section_name]
            section.apply_soot_blowing(strategy['segments'], strategy['effectiveness'])


def validate_property_calculations():
    """Validate property calculations against known values."""
    print("\n🔬 Validating Property Calculations...")
    
    prop_calc = PropertyCalculator()
    
    # Test steam properties at known conditions
    try:
        # Test saturated steam at 600 psia (should be around 486°F)
        steam_props = prop_calc.get_steam_properties(700, 600)
        expected_superheat = 700 - 486  # Approximate
        
        if abs(steam_props.superheat - expected_superheat) < 50:  # Allow 50°F tolerance
            print("✅ Steam superheat calculation: PASSED")
        else:
            print(f"⚠️  Steam superheat calculation: Expected ~{expected_superheat}°F, got {steam_props.superheat:.1f}°F")
        
        # Test liquid water density (should be reasonable)
        water_props = prop_calc.get_steam_properties(300, 600)
        if 40 < water_props.density < 70:  # Reasonable range for liquid water
            print("✅ Liquid water density: PASSED")
        else:
            print(f"⚠️  Liquid water density: {water_props.density:.2f} lbm/ft³ outside expected range")
        
        # Test gas properties
        gas_props = prop_calc.get_flue_gas_properties(1000)
        if 0.01 < gas_props.density < 0.1:  # Reasonable range for gas
            print("✅ Flue gas density: PASSED")
        else:
            print(f"⚠️  Flue gas density: {gas_props.density:.4f} lbm/ft³ outside expected range")
        
        print("✅ Property validation completed")
        
    except Exception as e:
        print(f"❌ Property validation failed: {e}")


def run_performance_benchmark():
    """Run performance benchmark for the enhanced system."""
    import time
    
    print("\n⏱️  Running Performance Benchmark...")
    
    start_time = time.time()
    
    # Initialize system
    boiler = EnhancedCompleteBoilerSystem()
    init_time = time.time() - start_time
    
    # Solve system
    solve_start = time.time()
    results = boiler.solve_enhanced_system(max_iterations=10, tolerance=5.0)
    solve_time = time.time() - solve_start
    
    total_time = time.time() - start_time
    
    # Calculate performance metrics
    total_segments = sum(len(data['segments']) for data in results.values())
    segments_per_second = total_segments / solve_time if solve_time > 0 else 0
    
    print(f"📊 Performance Benchmark Results:")
    print(f"  System Initialization: {init_time:.2f} seconds")
    print(f"  System Solution: {solve_time:.2f} seconds")
    print(f"  Total Runtime: {total_time:.2f} seconds")
    print(f"  Total Segments Calculated: {total_segments}")
    print(f"  Calculation Rate: {segments_per_second:.1f} segments/second")
    print(f"  Convergence Iterations: {boiler.system_performance.get('iterations_to_converge', 'N/A')}")


if __name__ == "__main__":
    """Main execution block demonstrating the complete combustion-fouling-ML integration."""
    
    # Run the comprehensive demonstration
    dataset = demonstrate_combustion_fouling_integration()
    
    # Show ML preparation approach
    demonstrate_ml_model_preparation()
    
    # Demonstrate soot blowing capabilities
    demonstrate_soot_blowing_simulation()
    
    # Show advanced fouling scenarios
    demonstrate_advanced_fouling_scenarios()
    
    # Run validation tests
    validate_property_calculations()
    run_performance_benchmark()
    
    print(f"\n" + "🎉" + " " * 96 + "🎉")
    print(f"🎉 COMBUSTION-FOULING-ML INTEGRATION COMPLETE!            🎉")
    print(f"🎉 • Coal combustion modeling with soot production        🎉")
    print(f"🎉 • Dynamic fouling linked to combustion conditions     🎉") 
    print(f"🎉 • ML dataset generation for optimization               🎉")
    print(f"🎉 • Individual segment fouling control                   🎉")
    print(f"🎉 • Comprehensive performance analysis                   🎉")
    print(f"🎉" + " " * 96 + "🎉")        # Convert to DataFrame
        df = pd.DataFrame(scenarios_data)
        
        print(f"✅ Dataset generation complete: {len(df)} total data points")
        return df
    
    def _generate_random_scenario(self) -> Dict:
        """Generate random operating scenario."""
        # Coal properties variation
        coal_properties = {
            'carbon': np.random.normal(72, 5),      # % carbon
            'volatile_matter': np.random.normal(30, 5),  # % VM
            'fixed_carbon': np.random.normal(50, 5),     # % FC
            'sulfur': np.random.uniform(0.5, 3.0),       # % sulfur
            'ash': np.random.uniform(5, 15)              # % ash
        }
        
        # Ultimate analysis (normalized to 100%)
        moisture = np.random.uniform(2, 8)
        ultimate = {
            'C': coal_properties['carbon'],
            'H': np.random.uniform(4, 6),
            'O': np.random.uniform(8, 12),
            'N': np.random.uniform(1, 2),
            'S': coal_properties['sulfur'],
            'Ash': coal_properties['ash'],
            'Moisture': moisture
        }
        
        # Normalize to 100%
        total = sum(ultimate.values())
        ultimate = {k: v * 100 / total for k, v in ultimate.items()}
        
        # Operating conditions
        base_coal_rate = 8500  # lb/hr for 100 MMBtu/hr
        load_factor = np.random.uniform(0.6, 1.2)  # 60-120% load
        
        scenario = {
            'coal_lb_per_hr': base_coal_rate * load_factor,
            'air_scfh': np.random.uniform(800000, 1200000),  # Varying excess air
            'NOx_eff': np.random.uniform(0.2, 0.6),          # NOx conversion efficiency
            'air_temp_F': np.random.uniform(60, 100),        # Ambient air temp
            'air_RH_pct': np.random.uniform(30, 80),         # Humidity
            'ultimate_analysis': ultimate,
            'coal_properties': coal_properties,
            'operating_hours_since_clean': np.random.uniform(0, 1440),  # 0-60 days
            'season': np.random.choice(['winter', 'spring', 'summer', 'fall']),
            'fuel_quality': np.random.choice(['high', 'medium', 'low'])
        }
        
        return scenario
    
    def _calculate_combustion_scenario(self, scenario: Dict) -> Dict:
        """Calculate combustion results for scenario."""
        # Create combustion model
        combustion_model = CoalCombustionModel(
            ultimate_analysis=scenario['ultimate_analysis'],
            coal_lb_per_hr=scenario['coal_lb_per_hr'],
            air_scfh=scenario['air_scfh'],
            NOx_eff=scenario['NOx_eff'],
            air_temp_F=scenario['air_temp_F'],
            air_RH_pct=scenario['air_RH_pct']
        )
        
        # Calculate combustion
        combustion_model.calculate()
        
        # Calculate fouling rates
        fouling_rates = self.fouling_integrator.calculate_section_fouling_rates(
            combustion_model, scenario['coal_properties'], self.boiler_system
        )
        
        return {
            'combustion_model': combustion_model,
            'fouling_rates': fouling_rates,
            'thermal_nox': combustion_model.NO_thermal_lb_per_hr,
            'fuel_nox': combustion_model.NO_fuel_lb_per_hr,
            'total_nox': combustion_model.NO_total_lb_per_hr,
            'excess_o2': combustion_model.dry_O2_pct,
            'combustion_efficiency': combustion_model.combustion_efficiency,
            'flame_temp': combustion_model.flame_temp_F,
            'heat_release': combustion_model.heat_released_btu_per_hr
        }
    
    def _simulate_fouling_timeline(self, scenario: Dict, combustion_data: Dict,
                                  timeline_hours: int = 720) -> Dict:
        """Simulate fouling buildup over time."""
        fouling_rates = combustion_data['fouling_rates']
        
        # Initialize current fouling levels
        current_fouling = {}
        for section_name in fouling_rates.keys():
            section = self.boiler_system.sections[section_name]
            current_fouling[section_name] = {
                'gas': [section.base_fouling_gas] * section.num_segments,
                'water': [section.base_fouling_water] * section.num_segments
            }
        
        # Simulate hourly buildup
        timeline = []
        for hour in range(timeline_hours):
            # Apply fouling buildup
            for section_name in fouling_rates.keys():
                rates = fouling_rates[section_name]
                for i in range(len(rates['gas'])):
                    current_fouling[section_name]['gas'][i] += rates['gas'][i]
                    current_fouling[section_name]['water'][i] += rates['water'][i]
            
            # Record state every 24 hours
            if hour % 24 == 0:
                timeline.append({
                    'hour': hour,
                    'fouling_state': {k: {'gas': v['gas'].copy(), 'water': v['water'].copy()} 
                                    for k, v in current_fouling.items()},
                    'performance_impact': self._calculate_performance_impact(current_fouling)
                })
        
        return {
            'timeline': timeline,
            'final_fouling': current_fouling,
            'total_hours': timeline_hours
        }
    
    def _calculate_performance_impact(self, fouling_state: Dict) -> Dict:
        """Calculate performance impact of current fouling state."""
        # Simplified performance calculation
        total_thermal_resistance = 0
        total_segments = 0
        
        for section_name, fouling in fouling_state.items():
            for gas_foul in fouling['gas']:
                total_thermal_resistance += gas_foul
                total_segments += 1
        
        avg_thermal_resistance = total_thermal_resistance / total_segments if total_segments > 0 else 0
        
        # Estimate performance degradation
        efficiency_loss = min(0.05, avg_thermal_resistance * 1000)  # Max 5% loss
        heat_transfer_loss = min(0.10, avg_thermal_resistance * 2000)  # Max 10% loss
        
        return {
            'efficiency_loss': efficiency_loss,
            'heat_transfer_loss': heat_transfer_loss,
            'avg_thermal_resistance': avg_thermal_resistance
        }
    
    def _generate_cleaning_scenarios(self, fouling_timeline: Dict) -> List[Dict]:
        """Generate various cleaning scenarios to evaluate."""
        cleaning_scenarios = []
        
        # Get final fouling state
        final_fouling = fouling_timeline['final_fouling']
        
        # Strategy 1: Clean everything
        cleaning_scenarios.append({
            'strategy': 'clean_all',
            'sections_to_clean': list(final_fouling.keys()),
            'segments_per_section': {name: list(range(len(fouling['gas']))) 
                                   for name, fouling in final_fouling.items()},
            'cleaning_effectiveness': 0.9
        })
        
        # Strategy 2: Clean worst sections only
        worst_sections = self._identify_worst_sections(final_fouling, top_n=3)
        cleaning_scenarios.append({
            'strategy': 'clean_worst',
            'sections_to_clean': worst_sections,
            'segments_per_section': {name: list(range(len(final_fouling[name]['gas']))) 
                                   for name in worst_sections},
            'cleaning_effectiveness': 0.85
        })
        
        # Strategy 3: Selective segment cleaning
        cleaning_scenarios.append({
            'strategy': 'selective_segments',
            'sections_to_clean': list(final_fouling.keys()),
            'segments_per_section': self._identify_worst_segments(final_fouling),
            'cleaning_effectiveness': 0.8
        })
        
        # Strategy 4: Progressive cleaning (economizers first)
        econ_sections = [name for name in final_fouling.keys() if 'economizer' in name]
        cleaning_scenarios.append({
            'strategy': 'economizers_first',
            'sections_to_clean': econ_sections,
            'segments_per_section': {name: list(range(len(final_fouling[name]['gas']))) 
                                   for name in econ_sections},
            'cleaning_effectiveness': 0.9
        })
        
        # Strategy 5: No cleaning (baseline)
        cleaning_scenarios.append({
            'strategy': 'no_cleaning',
            'sections_to_clean': [],
            'segments_per_section': {},
            'cleaning_effectiveness': 0.0
        })
        
        # Strategy 6: Random partial cleaning
        random_sections = np.random.choice(list(final_fouling.keys()), 
                                         size=np.random.randint(1, 4), replace=False)
        cleaning_scenarios.append({
            'strategy': 'random_partial',
            'sections_to_clean': random_sections.tolist(),
            'segments_per_section': {name: np.random.choice(range(len(final_fouling[name]['gas'])), 
                                                          size=np.random.randint(2, len(final_fouling[name]['gas'])), 
                                                          replace=False).tolist()
                                   for name in random_sections},
            'cleaning_effectiveness': np.random.uniform(0.7, 0.95)
        })
        
        return cleaning_scenarios
    
    def _identify_worst_sections(self, fouling_state: Dict, top_n: int = 3) -> List[str]:
        """Identify sections with worst fouling."""
        section_scores = {}
        
        for section_name, fouling in fouling_state.items():
            # Calculate average fouling level
            avg_gas_fouling = np.mean(fouling['gas'])
            max_gas_fouling = np.max(fouling['gas'])
            section_scores[section_name] = avg_gas_fouling + max_gas_fouling * 0.5
        
        # Sort by score and return top N
        sorted_sections = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)
        return [name for name, score in sorted_sections[:top_n]]
    
    def _identify_worst_segments(self, fouling_state: Dict) -> Dict[str, List[int]]:
        """Identify worst segments in each section."""
        worst_segments = {}
        
        for section_name, fouling in fouling_state.items():
            gas_fouling = fouling['gas']
            avg_fouling = np.mean(gas_fouling)
            threshold = avg_fouling * 1.3  # 30% above average
            
            worst_segs = [i for i, f in enumerate(gas_fouling) if f > threshold]
            if not worst_segs:  # If none above threshold, take worst half
                sorted_indices = sorted(range(len(gas_fouling)), 
                                      key=lambda i: gas_fouling[i], reverse=True)
                worst_segs = sorted_indices[:len(gas_fouling)//2 + 1]
            
            worst_segments[section_name] = worst_segs
        
        return worst_segments
    
    def _evaluate_cleaning_scenario(self, scenario: Dict, combustion_data: Dict,
                                  fouling_timeline: Dict, cleaning: Dict) -> Dict:
        """Evaluate a specific cleaning scenario."""
        # Calculate performance before cleaning
        before_fouling = fouling_timeline['final_fouling']
        before_performance = self._calculate_performance_impact(before_fouling)
        
        # Apply cleaning
        after_fouling = self._apply_cleaning(before_fouling, cleaning)
        after_performance = self._calculate_performance_impact(after_fouling)
        
        # Calculate cleaning costs and benefits
        cleaning_cost = self._calculate_cleaning_cost(cleaning)
        cleaning_time = self._calculate_cleaning_time(cleaning)
        
        # Performance improvements
        efficiency_gain = before_performance['efficiency_loss'] - after_performance['efficiency_loss']
        heat_transfer_gain = before_performance['heat_transfer_loss'] - after_performance['heat_transfer_loss']
        
        # Economic calculations
        fuel_savings_per_hour = efficiency_gain * scenario['coal_lb_per_hr'] * 12000 * 5.0 / 1e6  # $/hr#!/usr/bin/env python3
"""
Enhanced Boiler System Heat Transfer Model with Coal Combustion Integration

This module provides a comprehensive heat transfer analysis for a 100 MMBtu/hr subcritical 
boiler system with coal combustion modeling, soot production prediction, and ML dataset generation
for optimizing soot blowing schedules.

Key Features:
- Coal combustion model integration for realistic flue gas properties
- Soot production modeling based on combustion conditions
- Dynamic fouling buildup simulation linked to soot formation
- Comprehensive ML dataset generation for soot blowing optimization
- Individual segment fouling control for cleaning simulation

Classes:
    CoalCombustionModel: Coal combustion analysis (imported from your class)
    SootProductionModel: Soot formation and deposition modeling
    CombustionFoulingIntegrator: Links combustion to fouling buildup
    MLDatasetGenerator: Comprehensive dataset generation for ML training
    [... existing classes ...]

Dependencies:
    - numpy: Numerical calculations
    - matplotlib: Plotting capabilities
    - thermo: Comprehensive thermodynamic property library
    - scipy: Optimization and scientific computing
    - pandas: Data manipulation and analysis
    - dataclasses: For structured data
    - typing: Type hints
    - math: Mathematical functions
    - datetime: Timestamp generation

Author: Enhanced Boiler Modeling System with Combustion Integration
Version: 5.0 - Coal Combustion and ML Dataset Generation
"""

import math
import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from thermo import ChemicalConstantsPackage, PRMIX, FlashVL
from thermo.chemical import Chemical
from thermo.mixture import Mixture
import scipy.optimize as opt

# Module constants
DEFAULT_TUBE_THERMAL_CONDUCTIVITY = 26.0  # Btu/hr-ft-°F (carbon steel)
MIN_SEGMENTS = 5
MAX_SEGMENTS = 10
SEGMENT_AREA_THRESHOLD = 1000  # ft² per segment

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

# Since we can't import, I'll include the essential parts inline
class CoalCombustionModel:
    """Simplified version of your coal combustion model for integration"""
    
    def __init__(self, ultimate_analysis, coal_lb_per_hr, air_scfh, NOx_eff=0.35,
                 air_temp_F=75.0, air_RH_pct=55.0, atm_press_inHg=30.25):
        self._ultimate_analysis = ultimate_analysis.copy()
        self._coal_lb_per_hr = coal_lb_per_hr
        self._air_scfh = air_scfh
        self._NOx_eff = NOx_eff
        self._air_temp_F = air_temp_F
        self._air_RH_pct = air_RH_pct
        self._atm_press_inHg = atm_press_inHg
        self._results = {}
        self._calculated = False
    
    def calculate(self, debug=False):
        """Simplified calculation for integration"""
        # Simplified combustion calculations
        fuel_input = self._coal_lb_per_hr * 12000  # Approximate BTU/lb
        excess_air = max(0, (self._air_scfh / 10 / self._coal_lb_per_hr) - 8)  # Simplified
        
        self._results = {
            'total_flue_gas_lb_per_hr': self._coal_lb_per_hr * 12 + self._air_scfh * 0.075,
            'CO2_lb_per_hr': self._coal_lb_per_hr * 2.5,
            'NO_total_lb_per_hr': self._coal_lb_per_hr * 0.01 * self._NOx_eff,
            'NO_thermal_lb_per_hr': self._coal_lb_per_hr * 0.003,
            'NO_fuel_lb_per_hr': self._coal_lb_per_hr * 0.007 * self._NOx_eff,
            'dry_O2_pct': excess_air * 0.8,
            'combustion_efficiency': 0.98 - excess_air * 0.002,
            'heat_released_btu_per_hr': fuel_input * (0.98 - excess_air * 0.002),
            'flame_temp_F': 3200 - excess_air * 20
        }
        self._calculated = True
    
    @property
    def total_flue_gas_lb_per_hr(self):
        if not self._calculated: self.calculate()
        return self._results['total_flue_gas_lb_per_hr']
    
    @property
    def NO_total_lb_per_hr(self):
        if not self._calculated: self.calculate()
        return self._results['NO_total_lb_per_hr']
    
    @property
    def NO_thermal_lb_per_hr(self):
        if not self._calculated: self.calculate()
        return self._results['NO_thermal_lb_per_hr']
    
    @property
    def NO_fuel_lb_per_hr(self):
        if not self._calculated: self.calculate()
        return self._results['NO_fuel_lb_per_hr']
    
    @property
    def dry_O2_pct(self):
        if not self._calculated: self.calculate()
        return self._results['dry_O2_pct']
    
    @property
    def combustion_efficiency(self):
        if not self._calculated: self.calculate()
        return self._results['combustion_efficiency']
    
    @property
    def heat_released_btu_per_hr(self):
        if not self._calculated: self.calculate()
        return self._results['heat_released_btu_per_hr']
    
    @property
    def flame_temp_F(self):
        if not self._calculated: self.calculate()
        return self._results['flame_temp_F']


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


@dataclass
class SegmentResult:
    """Results dataclass for individual tube segment heat transfer analysis."""
    segment_id: int
    position: float  # 0 to 1 along tube length
    gas_temp_in: float  # °F
    gas_temp_out: float  # °F
    water_temp_in: float  # °F
    water_temp_out: float  # °F
    heat_transfer_rate: float  # Btu/hr
    overall_U: float  # Btu/hr-ft²-°F
    gas_htc: float  # Btu/hr-ft²-°F
    water_htc: float  # Btu/hr-ft²-°F
    fouling_gas: float  # hr-ft²-°F/Btu
    fouling_water: float  # hr-ft²-°F/Btu
    LMTD: float  # °F
    area: float  # ft²


@dataclass
class SootProductionData:
    """Soot production and characteristics data"""
    mass_production_rate: float  # lb/hr
    particle_size_microns: float  # μm
    carbon_content: float  # fraction
    ash_content: float  # fraction
    deposition_tendency: float  # 0-1 scale
    erosion_factor: float  # relative to baseline


@dataclass
class FoulingCharacteristics:
    """Fouling buildup characteristics"""
    thermal_resistance: float  # hr-ft²-°F/Btu
    buildup_rate: float  # resistance increase per hour
    deposit_thickness: float  # inches
    deposit_density: float  # lb/ft³
    cleaning_difficulty: float  # 0-1 scale (1 = very difficult)
    heat_transfer_impact: float  # fractional reduction


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


class SootProductionModel:
    """Model soot production based on combustion conditions."""
    
    def __init__(self):
        """Initialize soot production model with empirical correlations."""
        # Empirical constants based on literature and field data
        self.base_soot_rate = 0.001  # lb soot per lb coal (baseline)
        self.nox_soot_correlation = 0.8  # Higher NOx often means more soot
        self.efficiency_impact = 2.0  # Lower efficiency increases soot
        self.temperature_impact = 1.5  # Temperature effects on soot formation
        
    def calculate_soot_production(self, combustion_model: CoalCombustionModel,
                                coal_properties: Dict) -> SootProductionData:
        """Calculate soot production based on combustion conditions."""
        # Get combustion parameters
        thermal_nox = combustion_model.NO_thermal_lb_per_hr
        fuel_nox = combustion_model.NO_fuel_lb_per_hr
        total_nox = combustion_model.NO_total_lb_per_hr
        excess_o2 = combustion_model.dry_O2_pct
        combustion_eff = combustion_model.combustion_efficiency
        flame_temp = combustion_model.flame_temp_F
        coal_rate = combustion_model._coal_lb_per_hr
        
        # Calculate soot production factors
        nox_factor = self._calculate_nox_soot_factor(thermal_nox, fuel_nox, coal_rate)
        excess_air_factor = self._calculate_excess_air_factor(excess_o2)
        efficiency_factor = self._calculate_efficiency_factor(combustion_eff)
        temperature_factor = self._calculate_temperature_factor(flame_temp)
        coal_factor = self._calculate_coal_factor(coal_properties)
        
        # Base soot production rate
        base_production = self.base_soot_rate * coal_rate
        
        # Apply all factors
        actual_soot_rate = (base_production * nox_factor * excess_air_factor * 
                           efficiency_factor * temperature_factor * coal_factor)
        
        # Calculate soot characteristics
        particle_size = self._calculate_particle_size(flame_temp, excess_o2)
        carbon_content = self._calculate_carbon_content(combustion_eff, coal_properties)
        ash_content = 1.0 - carbon_content
        deposition_tendency = self._calculate_deposition_tendency(
            particle_size, flame_temp, excess_o2
        )
        erosion_factor = self._calculate_erosion_factor(particle_size, coal_properties)
        
        return SootProductionData(
            mass_production_rate=actual_soot_rate,
            particle_size_microns=particle_size,
            carbon_content=carbon_content,
            ash_content=ash_content,
            deposition_tendency=deposition_tendency,
            erosion_factor=erosion_factor
        )
    
    def _calculate_nox_soot_factor(self, thermal_nox: float, fuel_nox: float, 
                                  coal_rate: float) -> float:
        """Calculate soot factor based on NOx formation."""
        # Higher NOx formation often correlates with incomplete combustion zones
        nox_ppm = (thermal_nox + fuel_nox) / coal_rate * 1000  # Approximate ppm
        
        if nox_ppm < 100:
            return 0.5  # Very clean combustion, low soot
        elif nox_ppm < 300:
            return 1.0  # Normal combustion
        elif nox_ppm < 500:
            return 1.5  # Higher soot formation
        else:
            return 2.0  # Poor combustion, high soot
    
    def _calculate_excess_air_factor(self, excess_o2_pct: float) -> float:
        """Calculate soot factor based on excess air."""
        if excess_o2_pct < 1:
            return 3.0  # Very fuel-rich, high soot
        elif excess_o2_pct < 3:
            return 1.8  # Slightly rich, increased soot
        elif excess_o2_pct < 6:
            return 1.0  # Optimal air/fuel ratio
        elif excess_o2_pct < 10:
            return 0.8  # Slightly lean, reduced soot
        else:
            return 0.6  # Very lean, low soot but poor efficiency
    
    def _calculate_efficiency_factor(self, combustion_eff: float) -> float:
        """Calculate soot factor based on combustion efficiency."""
        # Lower efficiency correlates with incomplete combustion and more soot
        return 2.0 - combustion_eff  # Linear relationship
    
    def _calculate_temperature_factor(self, flame_temp_F: float) -> float:
        """Calculate soot factor based on flame temperature."""
        # Soot formation peaks at intermediate temperatures
        optimal_temp = 3000  # °F for minimal soot
        temp_deviation = abs(flame_temp_F - optimal_temp) / 500
        return 1.0 + temp_deviation * 0.5
    
    def _calculate_coal_factor(self, coal_properties: Dict) -> float:
        """Calculate soot factor based on coal properties."""
        # Higher volatile matter and lower fixed carbon increase soot tendency
        volatile_matter = coal_properties.get('volatile_matter', 30)  # %
        fixed_carbon = coal_properties.get('fixed_carbon', 50)  # %
        sulfur = coal_properties.get('sulfur', 1.0)  # %
        
        vm_factor = 1.0 + (volatile_matter - 30) * 0.02  # Higher VM increases soot
        fc_factor = 1.0 + (50 - fixed_carbon) * 0.01  # Lower FC increases soot
        s_factor = 1.0 + sulfur * 0.1  # Sulfur can affect soot formation
        
        return vm_factor * fc_factor * s_factor
    
    def _calculate_particle_size(self, flame_temp_F: float, excess_o2_pct: float) -> float:
        """Calculate average soot particle size."""
        base_size = 2.0  # μm
        temp_effect = (flame_temp_F - 2500) / 1000 * 0.5  # Temperature effect
        o2_effect = (excess_o2_pct - 3) * 0.1  # Oxygen effect
        return max(0.5, base_size + temp_effect + o2_effect)
    
    def _calculate_carbon_content(self, combustion_eff: float, 
                                 coal_properties: Dict) -> float:
        """Calculate carbon content of soot."""
        base_carbon = 0.85  # 85% carbon baseline
        eff_effect = (1.0 - combustion_eff) * 0.5  # Lower efficiency = more carbon
        coal_carbon = coal_properties.get('carbon', 70) / 100
        return min(0.95, base_carbon + eff_effect + coal_carbon * 0.1)
    
    def _calculate_deposition_tendency(self, particle_size: float, 
                                     flame_temp_F: float, excess_o2_pct: float) -> float:
        """Calculate tendency for soot to deposit on surfaces."""
        size_factor = 1.0 / (1.0 + particle_size)  # Smaller particles stick better
        temp_factor = (4000 - flame_temp_F) / 2000  # Lower temps increase deposition
        o2_factor = 1.0 / (1.0 + excess_o2_pct * 0.1)  # Lower O2 increases sticking
        
        return max(0.1, min(1.0, size_factor * temp_factor * o2_factor))
    
    def _calculate_erosion_factor(self, particle_size: float, 
                                 coal_properties: Dict) -> float:
        """Calculate erosion factor due to soot particles."""
        size_effect = particle_size / 5.0  # Larger particles cause more erosion
        ash_content = coal_properties.get('ash', 10) / 100
        ash_effect = ash_content * 2.0  # Ash increases erosion
        
        return max(0.5, 1.0 + size_effect + ash_effect)


class FoulingCalculator:
    """Fouling factor calculation utilities with position and temperature dependencies."""
    
    @staticmethod
    def calculate_fouling_gradient(base_gas_fouling: float, base_water_fouling: float,
                                 segment_position: float, avg_gas_temp: float, 
                                 avg_water_temp: float, custom_gas_fouling: Optional[float] = None,
                                 custom_water_fouling: Optional[float] = None) -> Tuple[float, float]:
        """Calculate position-dependent fouling factors with optional custom overrides."""
        # Use custom fouling factors if provided (for soot blowing simulation)
        if custom_gas_fouling is not None:
            gas_fouling = custom_gas_fouling
        else:
            # Standard gradient calculation
            temp_factor = max(0.5, (3500 - avg_gas_temp) / 2500)
            position_factor = 1 + 0.8 * segment_position
            gas_fouling = base_gas_fouling * temp_factor * position_factor
        
        if custom_water_fouling is not None:
            water_fouling = custom_water_fouling
        else:
            # Standard gradient calculation
            if avg_water_temp > 500:  # Steam side
                water_fouling = base_water_fouling * (0.5 + 0.3 * segment_position)
            else:  # Liquid water side
                temp_effect = 1 + 0.001 * max(0, avg_water_temp - 200)
                position_effect = 1 + 0.5 * segment_position
                water_fouling = base_water_fouling * temp_effect * position_effect
        
        return gas_fouling, water_fouling


class SootBlowingSimulator:
    """Soot blowing simulation utilities for individual segment fouling control."""
    
    @staticmethod
    def create_clean_fouling_array(num_segments: int, base_gas_fouling: float, 
                                  base_water_fouling: float, 
                                  cleaning_effectiveness: float = 0.8) -> Dict[str, List[float]]:
        """Create fouling arrays representing freshly cleaned tubes."""
        clean_gas_fouling = base_gas_fouling * (1 - cleaning_effectiveness)
        clean_water_fouling = base_water_fouling * (1 - cleaning_effectiveness)
        
        return {
            'gas': [clean_gas_fouling] * num_segments,
            'water': [clean_water_fouling] * num_segments
        }
    
    @staticmethod
    def create_gradient_fouling_array(num_segments: int, base_gas_fouling: float,
                                    base_water_fouling: float, multiplier: float = 1.0,
                                    inlet_factor: float = 0.5, outlet_factor: float = 2.0) -> Dict[str, List[float]]:
        """Create fouling arrays with realistic gradients along tube length."""
        gas_fouling = []
        water_fouling = []
        
        for i in range(num_segments):
            position = i / (num_segments - 1) if num_segments > 1 else 0
            
            # Linear gradient from inlet to outlet
            gas_factor = inlet_factor + (outlet_factor - inlet_factor) * position
            water_factor = 0.8 + 0.4 * position  # Water-side typically varies less
            
            gas_fouling.append(base_gas_fouling * gas_factor * multiplier)
            water_fouling.append(base_water_fouling * water_factor * multiplier)
        
        return {
            'gas': gas_fouling,
            'water': water_fouling
        }
    
    @staticmethod
    def simulate_partial_soot_blowing(fouling_array: Dict[str, List[float]], 
                                    blown_segments: List[int],
                                    cleaning_effectiveness: float = 0.85) -> Dict[str, List[float]]:
        """Simulate partial soot blowing affecting only specific segments."""
        new_array = {
            'gas': fouling_array['gas'].copy(),
            'water': fouling_array['water'].copy()
        }
        
        for segment_id in blown_segments:
            if 0 <= segment_id < len(new_array['gas']):
                new_array['gas'][segment_id] *= (1 - cleaning_effectiveness)
                new_array['water'][segment_id] *= (1 - cleaning_effectiveness)
        
        return new_array
    
    @staticmethod
    def simulate_progressive_fouling(clean_fouling_array: Dict[str, List[float]],
                                   operating_hours: float,
                                   fouling_rate_per_hour: float = 0.001) -> Dict[str, List[float]]:
        """Simulate progressive fouling buildup over time."""
        fouling_multiplier = 1 + fouling_rate_per_hour * operating_hours
        
        return {
            'gas': [f * fouling_multiplier for f in clean_fouling_array['gas']],
            'water': [f * fouling_multiplier for f in clean_fouling_array['water']]
        }


class HeatTransferCalculator:
    """Heat transfer coefficient calculation utilities with correlations for different geometries."""
    
    @staticmethod
    def calculate_reynolds_number(flow_rate: float, density: float, 
                                hydraulic_diameter: float, viscosity: float,
                                flow_area: float) -> float:
        """Calculate Reynolds number for flow conditions."""
        if viscosity <= 0 or flow_area <= 0:
            return 0
        
        velocity = flow_rate / (density * flow_area)
        return density * velocity * hydraulic_diameter / viscosity
    
    @staticmethod
    def calculate_nusselt_number(reynolds: float, prandtl: float, geometry: str,
                               section_type: str, phase: str = 'liquid') -> float:
        """Calculate Nusselt number based on flow conditions and geometry."""
        if geometry == 'tube_side':
            if reynolds > 2300:  # Turbulent flow
                if phase == 'superheated_steam':
                    # Modified Dittus-Boelter for superheated steam
                    return 0.021 * (reynolds ** 0.8) * (prandtl ** 0.4)
                else:
                    # Standard Dittus-Boelter correlation
                    return 0.023 * (reynolds ** 0.8) * (prandtl ** 0.4)
            else:  # Laminar flow
                return 4.36  # Fully developed laminar flow in circular tube
                
        else:  # Shell side cross-flow
            if section_type == 'radiant':
                # Enhanced heat transfer in radiant sections
                return 0.4 * (reynolds ** 0.6) * (prandtl ** 0.36)
            else:
                # Standard cross-flow correlations for tube bundles
                if reynolds > 10000:
                    return 0.27 * (reynolds ** 0.63) * (prandtl ** 0.36)
                elif reynolds > 1000:
                    return 0.35 * (reynolds ** 0.60) * (prandtl ** 0.36)
                else:
                    return 2.0  # Minimum reasonable Nusselt number
    
    @staticmethod
    def calculate_heat_transfer_coefficient(flow_rate: float, hydraulic_diameter: float,
                                          properties: Union[SteamProperties, GasProperties], 
                                          geometry: str, section_type: str,
                                          tube_count: int, tube_id: float) -> float:
        """Calculate convective heat transfer coefficient."""
        if flow_rate <= 0:
            return 1.0  # Minimum value
        
        # Calculate flow area
        if geometry == 'tube_side':
            flow_area = tube_count * math.pi * (tube_id ** 2) / 4.0
        else:  # Shell side
            flow_area = tube_count * 0.05  # Approximate shell-side flow area
        
        # Calculate Reynolds number
        reynolds = HeatTransferCalculator.calculate_reynolds_number(
            flow_rate, properties.density, hydraulic_diameter, 
            properties.viscosity, flow_area
        )
        
        # Get phase information
        phase = getattr(properties, 'phase', 'gas')
        
        # Calculate Nusselt number
        nusselt = HeatTransferCalculator.calculate_nusselt_number(
            reynolds, properties.prandtl, geometry, section_type, phase
        )
        
        # Calculate heat transfer coefficient
        h = nusselt * properties.thermal_conductivity / hydraulic_diameter
        return max(h, 1.0)  # Ensure minimum reasonable value


class CombustionFoulingIntegrator:
    """Integrate combustion conditions with fouling buildup in boiler sections."""
    
    def __init__(self):
        """Initialize the combustion-fouling integrator."""
        self.soot_model = SootProductionModel()
        
        # Section-specific fouling characteristics
        self.section_fouling_factors = {
            'furnace_walls': {
                'temperature_factor': 0.3,  # High temp reduces sticking
                'velocity_factor': 0.8,     # High velocity reduces deposition
                'surface_factor': 1.2       # Rough surfaces increase fouling
            },
            'generating_bank': {
                'temperature_factor': 0.8,
                'velocity_factor': 1.0,
                'surface_factor': 1.0
            },
            'superheater_primary': {
                'temperature_factor': 0.6,
                'velocity_factor': 1.2,
                'surface_factor': 0.9
            },
            'superheater_secondary': {
                'temperature_factor': 0.7,
                'velocity_factor': 1.1,
                'surface_factor': 0.9
            },
            'economizer_primary': {
                'temperature_factor': 1.2,  # Lower temp increases sticking
                'velocity_factor': 0.9,
                'surface_factor': 1.1
            },
            'economizer_secondary': {
                'temperature_factor': 1.5,  # Coldest section, most fouling
                'velocity_factor': 0.8,
                'surface_factor': 1.2
            },
            'air_heater': {
                'temperature_factor': 1.8,  # Very cold, high fouling
                'velocity_factor': 0.7,
                'surface_factor': 1.3
            }
        }
    
    def calculate_section_fouling_rates(self, combustion_model: CoalCombustionModel,
                                      coal_properties: Dict,
                                      boiler_system) -> Dict[str, Dict[str, List[float]]]:
        """Calculate fouling rates for each section and segment."""
        # Calculate base soot production
        soot_data = self.soot_model.calculate_soot_production(
            combustion_model, coal_properties
        )
        
        section_fouling_rates = {}
        
        for section_name, section in boiler_system.sections.items():
            section_factors = self.section_fouling_factors.get(section_name, {
                'temperature_factor': 1.0,
                'velocity_factor': 1.0,
                'surface_factor': 1.0
            })
            
            # Calculate segment-specific fouling rates
            gas_fouling_rates = []
            water_fouling_rates = []
            
            for segment_id in range(section.num_segments):
                segment_position = segment_id / (section.num_segments - 1) if section.num_segments > 1 else 0
                
                # Calculate local conditions (simplified)
                local_gas_temp = 2500 - 1500 * segment_position  # Decreasing temp
                local_velocity = 50 + 20 * segment_position       # Increasing velocity
                
                # Apply section and local factors
                temp_effect = self._temperature_fouling_effect(local_gas_temp)
                velocity_effect = self._velocity_fouling_effect(local_velocity)
                position_effect = 1.0 + segment_position * 0.5  # More fouling downstream
                
                # Calculate fouling rate for this segment
                base_gas_fouling_rate = (soot_data.mass_production_rate * 
                                       soot_data.deposition_tendency * 
                                       section_factors['temperature_factor'] *
                                       section_factors['velocity_factor'] *
                                       section_factors['surface_factor'] *
                                       temp_effect * velocity_effect * position_effect)
                
                base_water_fouling_rate = base_gas_fouling_rate * 0.3  # Water side typically less
                
                # Convert to fouling resistance units (hr-ft²-°F/Btu per hour)
                gas_fouling_rate = base_gas_fouling_rate * 1e-6  # Conversion factor
                water_fouling_rate = base_water_fouling_rate * 1e-6
                
                gas_fouling_rates.append(gas_fouling_rate)
                water_fouling_rates.append(water_fouling_rate)
            
            section_fouling_rates[section_name] = {
                'gas': gas_fouling_rates,
                'water': water_fouling_rates
            }
        
        return section_fouling_rates
    
    def _temperature_fouling_effect(self, gas_temp_F: float) -> float:
        """Calculate temperature effect on fouling deposition."""
        if gas_temp_F > 2000:
            return 0.2  # Very high temp, minimal sticking
        elif gas_temp_F > 1500:
            return 0.5
        elif gas_temp_F > 1000:
            return 1.0  # Moderate temp, baseline fouling
        elif gas_temp_F > 500:
            return 1.5  # Lower temp, increased sticking
        else:
            return 2.0  # Very low temp, high fouling
    
    def _velocity_fouling_effect(self, velocity_fps: float) -> float:
        """Calculate velocity effect on fouling deposition."""
        # Higher velocity reduces deposition due to scouring
        if velocity_fps < 20:
            return 1.8  # Low velocity, high deposition
        elif velocity_fps < 40:
            return 1.2
        elif velocity_fps < 60:
            return 1.0  # Baseline
        elif velocity_fps < 80:
            return 0.8
        else:
            return 0.6  # High velocity, low deposition


class EnhancedBoilerTubeSection:
    """Enhanced tube section model with segmented analysis and individual fouling control."""
    
    def __init__(self, name: str, tube_od: float, tube_id: float, tube_length: float, 
                 tube_count: int, base_fouling_gas: float, base_fouling_water: float,
                 section_type: str = 'convective'):
        """Initialize enhanced boiler tube section."""
        if tube_od <= tube_id:
            raise ValueError(f"Tube OD ({tube_od}) must be greater than ID ({tube_id})")
        if tube_length <= 0 or tube_count <= 0:
            raise ValueError("Tube length and count must be positive")
        
        self.name = name
        self.tube_od = tube_od
        self.tube_id = tube_id
        self.tube_length = tube_length
        self.tube_count = tube_count
        self.base_fouling_gas = base_fouling_gas
        self.base_fouling_water = base_fouling_water
        self.section_type = section_type
        
        # Calculate geometry parameters
        self.area = math.pi * tube_od * tube_length * tube_count
        self.num_segments = max(MIN_SEGMENTS, min(MAX_SEGMENTS, int(self.area / SEGMENT_AREA_THRESHOLD)))
        self.segment_length = tube_length / self.num_segments
        self.tube_wall_thickness = (tube_od - tube_id) / 2.0
        
        # Initialize property calculator
        self.property_calc = PropertyCalculator()
        
        # Custom fouling arrays for soot blowing simulation
        self.custom_fouling_arrays: Optional[Dict[str, List[float]]] = None
        
        # Results storage
        self.results: List[SegmentResult] = []
    
    def set_custom_fouling_arrays(self, gas_fouling_array: List[float], 
                                 water_fouling_array: List[float]):
        """Set custom fouling factor arrays for individual segment control."""
        if len(gas_fouling_array) != self.num_segments:
            raise ValueError(f"Gas fouling array length ({len(gas_fouling_array)}) must match segments ({self.num_segments})")
        if len(water_fouling_array) != self.num_segments:
            raise ValueError(f"Water fouling array length ({len(water_fouling_array)}) must match segments ({self.num_segments})")
        
        self.custom_fouling_arrays = {
            'gas': gas_fouling_array.copy(),
            'water': water_fouling_array.copy()
        }
        
        print(f"✓ Custom fouling arrays set for {self.name}: {self.num_segments} segments")
    
    def clear_custom_fouling_arrays(self):
        """Clear custom fouling arrays and return to standard gradient calculation."""
        self.custom_fouling_arrays = None
        print(f"✓ Custom fouling arrays cleared for {self.name}, using standard gradients")
    
    def get_current_fouling_arrays(self) -> Dict[str, List[float]]:
        """Get current fouling arrays (custom if set, otherwise calculated gradients)."""
        if self.custom_fouling_arrays is not None:
            return self.custom_fouling_arrays.copy()
        
        # Calculate standard gradients
        gas_fouling = []
        water_fouling = []
        
        for i in range(self.num_segments):
            segment_position = i / (self.num_segments - 1) if self.num_segments > 1 else 0
            
            # Estimate average temperatures for gradient calculation
            avg_gas_temp = 2000 - 1000 * segment_position  # Rough estimate
            avg_water_temp = 300 + 200 * segment_position   # Rough estimate
            
            gas_foul, water_foul = FoulingCalculator.calculate_fouling_gradient(
                self.base_fouling_gas, self.base_fouling_water,
                segment_position, avg_gas_temp, avg_water_temp
            )
            
            gas_fouling.append(gas_foul)
            water_fouling.append(water_foul)
        
        return {'gas': gas_fouling, 'water': water_fouling}
    
    def apply_soot_blowing(self, blown_segments: List[int], 
                          cleaning_effectiveness: float = 0.85):
        """Apply soot blowing to specific segments."""
        if self.custom_fouling_arrays is None:
            # Initialize with current gradients
            self.custom_fouling_arrays = self.get_current_fouling_arrays()
        
        # Apply cleaning to specified segments
        self.custom_fouling_arrays = SootBlowingSimulator.simulate_partial_soot_blowing(
            self.custom_fouling_arrays, blown_segments, cleaning_effectiveness
        )
        
        print(f"✓ Soot blowing applied to {self.name}, segments: {blown_segments}")
        print(f"  Cleaning effectiveness: {cleaning_effectiveness:.1%}")
    
    def simulate_fouling_buildup(self, operating_hours: float, 
                               fouling_rate_per_hour: float = 0.001):
        """Simulate progressive fouling buildup over time."""
        if self.custom_fouling_arrays is None:
            # Start with clean baseline
            clean_arrays = SootBlowingSimulator.create_clean_fouling_array(
                self.num_segments, self.base_fouling_gas, self.base_fouling_water, 0.0
            )
            self.custom_fouling_arrays = clean_arrays
        
        # Apply fouling buildup
        self.custom_fouling_arrays = SootBlowingSimulator.simulate_progressive_fouling(
            self.custom_fouling_arrays, operating_hours, fouling_rate_per_hour
        )
        
        print(f"✓ Fouling buildup simulated for {self.name}: {operating_hours} hours")
    
    def solve_segment(self, segment_id: int, gas_temp_in: float, water_temp_in: float,
                     gas_flow: float, water_flow: float, steam_pressure: float) -> SegmentResult:
        """Solve heat transfer analysis for individual tube segment with custom fouling support."""
        if segment_id < 0 or segment_id >= self.num_segments:
            raise ValueError(f"Invalid segment_id {segment_id}")
        
        segment_position = segment_id / (self.num_segments - 1) if self.num_segments > 1 else 0
        
        # Estimate temperature changes based on section type
        temp_drops = self._calculate_temperature_drops(segment_position)
        gas_temp_out = gas_temp_in - temp_drops['gas'] / self.num_segments
        water_temp_out = water_temp_in + temp_drops['water'] / self.num_segments
        
        # Average temperatures for property evaluation
        avg_gas_temp = (gas_temp_in + gas_temp_out) / 2
        avg_water_temp = (water_temp_in + water_temp_out) / 2
        
        # Get fluid properties using thermo library
        gas_props = self.property_calc.get_flue_gas_properties(avg_gas_temp)
        water_props = self.property_calc.get_steam_properties(avg_water_temp, steam_pressure)
        
        # Calculate fouling factors (custom or gradient-based)
        if self.custom_fouling_arrays is not None:
            # Use custom fouling arrays
            gas_fouling = self.custom_fouling_arrays['gas'][segment_id]
            water_fouling = self.custom_fouling_arrays['water'][segment_id]
        else:
            # Use standard gradient calculation
            gas_fouling, water_fouling = FoulingCalculator.calculate_fouling_gradient(
                self.base_fouling_gas, self.base_fouling_water, 
                segment_position, avg_gas_temp, avg_water_temp
            )
        
        # Calculate heat transfer coefficients
        h_gas = HeatTransferCalculator.calculate_heat_transfer_coefficient(
            gas_flow, self.tube_od, gas_props, 'shell_side', 
            self.section_type, self.tube_count, self.tube_id
        )
        h_water = HeatTransferCalculator.calculate_heat_transfer_coefficient(
            water_flow, self.tube_id, water_props, 'tube_side', 
            self.section_type, self.tube_count, self.tube_id
        )
        
        # Calculate overall heat transfer coefficient
        segment_area_out = math.pi * self.tube_od * self.segment_length * self.tube_count
        segment_area_in = math.pi * self.tube_id * self.segment_length * self.tube_count
        
        # Overall thermal resistance
        overall_U = self._calculate_overall_U(
            h_gas, h_water, gas_fouling, water_fouling, 
            segment_area_out, segment_area_in
        )
        
        # Calculate LMTD and heat transfer rate
        LMTD = self._calculate_LMTD(gas_temp_in, gas_temp_out, water_temp_in, water_temp_out)
        Q = overall_U * segment_area_out * LMTD
        
        # Refine outlet temperatures using energy balance
        gas_temp_out_refined = self._refine_gas_temperature(gas_temp_in, Q, gas_flow, gas_props.cp)
        water_temp_out_refined = self._refine_water_temperature(water_temp_in, Q, water_flow, water_props.cp)
        
        return SegmentResult(
            segment_id=segment_id,
            position=segment_position,
            gas_temp_in=gas_temp_in,
            gas_temp_out=gas_temp_out_refined,
            water_temp_in=water_temp_in,
            water_temp_out=water_temp_out_refined,
            heat_transfer_rate=Q,
            overall_U=overall_U,
            gas_htc=h_gas,
            water_htc=h_water,
            fouling_gas=gas_fouling,
            fouling_water=water_fouling,
            LMTD=LMTD,
            area=segment_area_out
        )
    
    def _calculate_temperature_drops(self, segment_position: float) -> Dict[str, float]:
        """Calculate realistic temperature drops for different section types."""
        if self.section_type == 'radiant':
            return {
                'gas': 200 + 100 * (1 - segment_position),
                'water': 30 + 20 * segment_position
            }
        elif self.section_type == 'superheater':
            return {
                'gas': 150 - 30 * segment_position,
                'water': 40 + 20 * segment_position
            }
        elif self.section_type == 'economizer':
            return {
                'gas': 120 + 40 * (1 - segment_position),
                'water': 35 + 15 * segment_position
            }
        else:  # convective
            return {
                'gas': 100 + 50 * segment_position,
                'water': 25 + 15 * segment_position
            }
    
    def _calculate_overall_U(self, h_gas: float, h_water: float, 
                           gas_fouling: float, water_fouling: float,
                           area_out: float, area_in: float) -> float:
        """Calculate overall heat transfer coefficient."""
        # Log mean area for conduction
        if area_out > area_in:
            A_lm = (area_out - area_in) / math.log(area_out / area_in)
        else:
            A_lm = area_out
        
        # Total thermal resistance
        R_total = (1.0 / h_gas + 
                  gas_fouling + 
                  self.tube_wall_thickness / (DEFAULT_TUBE_THERMAL_CONDUCTIVITY * A_lm / area_out) +
                  water_fouling * (area_out / area_in) + 
                  (1.0 / h_water) * (area_out / area_in))
        
        return 1.0 / R_total
    
    def _calculate_LMTD(self, gas_in: float, gas_out: float, 
                       water_in: float, water_out: float) -> float:
        """Calculate log mean temperature difference."""
        delta_T1 = gas_in - water_out
        delta_T2 = gas_out - water_in
        
        if abs(delta_T1 - delta_T2) < 1.0:
            return (delta_T1 + delta_T2) / 2.0
        else:
            try:
                return (delta_T1 - delta_T2) / math.log(delta_T1 / delta_T2)
            except (ValueError, ZeroDivisionError):
                return (delta_T1 + delta_T2) / 2.0
    
    def _refine_gas_temperature(self, gas_temp_in: float, Q: float, 
                               gas_flow: float, gas_cp: float) -> float:
        """Refine gas outlet temperature using energy balance."""
        if gas_flow > 0 and gas_cp > 0:
            return gas_temp_in - Q / (gas_flow * gas_cp)
        return gas_temp_in
    
    def _refine_water_temperature(self, water_temp_in: float, Q: float,
                                 water_flow: float, water_cp: float) -> float:
        """Refine water outlet temperature using energy balance."""
        if water_flow > 0 and water_cp > 0:
            return water_temp_in + Q / (water_flow * water_cp)
        return water_temp_in
    
    def solve_section(self, gas_temp_in: float, water_temp_in: float, 
                     gas_flow: float, water_flow: float, 
                     steam_pressure: float) -> List[SegmentResult]:
        """Solve heat transfer for entire section with all segments."""
        if gas_flow < 0 or water_flow < 0:
            raise ValueError("Flow rates cannot be negative")
        if steam_pressure < 1:
            raise ValueError("Steam pressure must be positive")
        
        self.results = []
        current_gas_temp = gas_temp_in
        current_water_temp = water_temp_in
        
        for i in range(self.num_segments):
            segment_result = self.solve_segment(
                i, current_gas_temp, current_water_temp, 
                gas_flow, water_flow, steam_pressure
            )
            
            self.results.append(segment_result)
            current_gas_temp = segment_result.gas_temp_out
            current_water_temp = segment_result.water_temp_out
        
        return self.results
    
    def get_section_summary(self) -> Dict[str, Union[str, float, int]]:
        """Get overall section performance summary."""
        if not self.results:
            return {}
        
        total_Q = sum(r.heat_transfer_rate for r in self.results)
        total_area = sum(r.area for r in self.results)
        avg_U = sum(r.overall_U * r.area for r in self.results) / total_area if total_area > 0 else 0
        
        return {
            'section_name': self.name,
            'total_heat_transfer': total_Q,
            'average_overall_U': avg_U,
            'total_area': total_area,
            'gas_temp_in': self.results[0].gas_temp_in,
            'gas_temp_out': self.results[-1].gas_temp_out,
            'water_temp_in': self.results[0].water_temp_in,
            'water_temp_out': self.results[-1].water_temp_out,
            'num_segments': len(self.results),
            'max_gas_fouling': max(r.fouling_gas for r in self.results),
            'max_water_fouling': max(r.fouling_water for r in self.results)
        }


class EnhancedCompleteBoilerSystem:
    """Complete enhanced boiler system model with thermo library integration."""
    
    def __init__(self, fuel_input: float = 100e6, flue_gas_mass_flow: float = 84000,
                 furnace_exit_temp: float = 3000, base_fouling_multiplier: float = 1.0):
        """Initialize enhanced boiler system with configurable parameters."""
        self.design_capacity = 100e6  # Btu/hr
        
        # User-configurable parameters
        self.fuel_input = fuel_input  # Btu/hr
        self.flue_gas_mass_flow = flue_gas_mass_flow  # lbm/hr
        self.furnace_exit_temp = furnace_exit_temp  # °F
        self.base_fouling_multiplier = base_fouling_multiplier
        
        # Initialize sections with configurable fouling
        self.sections = self._initialize_enhanced_sections()
        
        # Other system operating parameters
        self.combustion_efficiency = 0.85
        self.feedwater_flow = 68000  # lbm/hr (scaled for 100 MMBtu/hr)
        self.steam_pressure = 600  # psia (typical for smaller boiler)
        self.attemperator_flow = 0  # lbm/hr
        
        # Temperature targets
        self.feedwater_temp = 220  # °F
        self.final_steam_temp_target = 700  # °F (realistic superheat ~215°F at 600 psia)
        self.stack_temp_target = 300  # °F (slightly higher for smaller system)
        
        # Initialize property calculator
        self.property_calc = PropertyCalculator()
        
        # Results storage
        self.section_results: Dict[str, Dict] = {}
        self.system_performance: Dict[str, float] = {}
    
    def update_operating_conditions(self, fuel_input: Optional[float] = None,
                                  flue_gas_mass_flow: Optional[float] = None,
                                  furnace_exit_temp: Optional[float] = None,
                                  base_fouling_multiplier: Optional[float] = None):
        """Update operating conditions and reinitialize sections if fouling changed."""
        fouling_changed = False
        
        if fuel_input is not None:
            self.fuel_input = fuel_input
        if flue_gas_mass_flow is not None:
            self.flue_gas_mass_flow = flue_gas_mass_flow
        if furnace_exit_temp is not None:
            self.furnace_exit_temp = furnace_exit_temp
        if base_fouling_multiplier is not None:
            if abs(base_fouling_multiplier - self.base_fouling_multiplier) > 0.001:
                fouling_changed = True
            self.base_fouling_multiplier = base_fouling_multiplier
        
        # Reinitialize sections if fouling factors changed
        if fouling_changed:
            self.sections = self._initialize_enhanced_sections()
            print(f"✓ Sections reinitialized with fouling multiplier: {self.base_fouling_multiplier}")
    
    def _initialize_enhanced_sections(self) -> Dict[str, EnhancedBoilerTubeSection]:
        """Initialize all boiler heat transfer sections sized for 100 MMBtu/hr with configurable fouling."""
        sections = {}
        
        # Base fouling factors (before multiplier)
        base_fouling_factors = {
            'furnace_walls': {'gas': 0.0005, 'water': 0.0003},
            'generating_bank': {'gas': 0.002, 'water': 0.001},
            'superheater_primary': {'gas': 0.0015, 'water': 0.0005},
            'superheater_secondary': {'gas': 0.002, 'water': 0.0005},
            'economizer_primary': {'gas': 0.003, 'water': 0.002},
            'economizer_secondary': {'gas': 0.004, 'water': 0.0025},
            'air_heater': {'gas': 0.006, 'water': 0.001}
        }
        
        # Furnace Wall Tubes (scaled down by ~5x from 500 MMBtu design)
        fouling = base_fouling_factors['furnace_walls']
        sections['furnace_walls'] = EnhancedBoilerTubeSection(
            name="Furnace Wall Tubes",
            tube_od=2.5/12, tube_id=2.0/12, tube_length=35, tube_count=80,
            base_fouling_gas=fouling['gas'] * self.base_fouling_multiplier,
            base_fouling_water=fouling['water'] * self.base_fouling_multiplier,
            section_type='radiant'
        )
        
        # Generating Bank (scaled down)
        fouling = base_fouling_factors['generating_bank']
        sections['generating_bank'] = EnhancedBoilerTubeSection(
            name="Generating Bank", 
            tube_od=2.0/12, tube_id=1.75/12, tube_length=20, tube_count=160,
            base_fouling_gas=fouling['gas'] * self.base_fouling_multiplier,
            base_fouling_water=fouling['water'] * self.base_fouling_multiplier,
            section_type='convective'
        )
        
        # Primary Superheater (scaled down)
        fouling = base_fouling_factors['superheater_primary']
        sections['superheater_primary'] = EnhancedBoilerTubeSection(
            name="Primary Superheater",
            tube_od=2.0/12, tube_id=1.75/12, tube_length=15, tube_count=60,
            base_fouling_gas=fouling['gas'] * self.base_fouling_multiplier,
            base_fouling_water=fouling['water'] * self.base_fouling_multiplier,
            section_type='superheater'
        )
        
        # Secondary Superheater (scaled down)
        fouling = base_fouling_factors['superheater_secondary']
        sections['superheater_secondary'] = EnhancedBoilerTubeSection(
            name="Secondary Superheater",
            tube_od=2.0/12, tube_id=1.75/12, tube_length=15, tube_count=50,
            base_fouling_gas=fouling['gas'] * self.base_fouling_multiplier,
            base_fouling_water=fouling['water'] * self.base_fouling_multiplier,
            section_type='superheater'
        )
        
        # Primary Economizer (scaled down)
        fouling = base_fouling_factors['economizer_primary']
        sections['economizer_primary'] = EnhancedBoilerTubeSection(
            name="Primary Economizer",
            tube_od=2.0/12, tube_id=1.75/12, tube_length=18, tube_count=100,
            base_fouling_gas=fouling['gas'] * self.base_fouling_multiplier,
            base_fouling_water=fouling['water'] * self.base_fouling_multiplier,
            section_type='economizer'
        )
        
        # Secondary Economizer (scaled down)
        fouling = base_fouling_factors['economizer_secondary']
        sections['economizer_secondary'] = EnhancedBoilerTubeSection(
            name="Secondary Economizer",
            tube_od=2.0/12, tube_id=1.75/12, tube_length=15, tube_count=80,
            base_fouling_gas=fouling['gas'] * self.base_fouling_multiplier,
            base_fouling_water=fouling['water'] * self.base_fouling_multiplier,
            section_type='economizer'
        )
        
        # Air Heater (scaled down)
        fouling = base_fouling_factors['air_heater']
        sections['air_heater'] = EnhancedBoilerTubeSection(
            name="Air Heater",
            tube_od=1.75/12, tube_id=1.5/12, tube_length=12, tube_count=200,
            base_fouling_gas=fouling['gas'] * self.base_fouling_multiplier,
            base_fouling_water=fouling['water'] * self.base_fouling_multiplier,
            section_type='convective'
        )
        
        return sections
    
    def calculate_attemperator_flow(self, steam_temp_before: float, 
                                  target_temp: float, steam_flow: float) -> float:
        """Calculate attemperator spray water flow for temperature control."""
        if steam_temp_before <= target_temp:
            return 0
        
        # Get properties for energy balance
        steam_props = self.property_calc.get_steam_properties(steam_temp_before, self.steam_pressure)
        water_props = self.property_calc.get_steam_properties(self.feedwater_temp, self.steam_pressure)
        
        # Energy balance calculation
        numerator = steam_flow * steam_props.cp * (steam_temp_before - target_temp)
        denominator = water_props.cp * (target_temp - self.feedwater_temp) + steam_props.cp * (target_temp - steam_temp_before)
        
        if denominator <= 0:
            return 0
        
        spray_flow = numerator / denominator
        return max(0, min(spray_flow, steam_flow * 0.1))  # Limit to 10%
    
    def solve_enhanced_system(self, max_iterations: int = 15, tolerance: float = 3.0) -> Dict:
        """Solve the complete enhanced boiler system with iterative convergence."""
        print(f"Solving enhanced boiler system with thermo library...")
        print(f"Target: Stack {self.stack_temp_target}°F, Steam {self.final_steam_temp_target}°F")
        
        # Flow distribution (scaled for 100 MMBtu/hr)
        main_steam_flow = self.feedwater_flow * 0.97
        flows = {
            'furnace_walls': self.feedwater_flow * 0.35,
            'generating_bank': self.feedwater_flow,
            'superheater_primary': main_steam_flow,
            'superheater_secondary': main_steam_flow,
            'economizer_primary': self.feedwater_flow,
            'economizer_secondary': self.feedwater_flow,
            'air_heater': 56000,  # Combustion air flow (scaled from 280k)
        }
        
        # Section order
        section_order = ['furnace_walls', 'generating_bank', 'superheater_primary', 
                        'superheater_secondary', 'economizer_primary', 
                        'economizer_secondary', 'air_heater']
        
        # Initial water temperatures
        water_temps = {
            'furnace_walls': 350,
            'generating_bank': 400,
            'superheater_primary': 532,
            'superheater_secondary': 650,
            'economizer_primary': 280,
            'economizer_secondary': self.feedwater_temp,
            'air_heater': 80,
        }
        
        # Iterative solution
        for iteration in range(max_iterations):
            current_gas_temp = self.furnace_exit_temp
            total_heat_absorbed = 0
            section_summaries = {}
            
            print(f"\nIteration {iteration + 1}:")
            
            for section_name in section_order:
                section = self.sections[section_name]
                water_flow = flows[section_name]
                water_temp_in = water_temps[section_name]
                
                print(f"  Solving {section_name}: Gas {current_gas_temp:.0f}°F, Water {water_temp_in:.0f}°F")
                
                try:
                    segment_results = section.solve_section(
                        current_gas_temp, water_temp_in, self.flue_gas_mass_flow, 
                        water_flow, self.steam_pressure
                    )
                    
                    summary = section.get_section_summary()
                    section_summaries[section_name] = summary
                    current_gas_temp = summary['gas_temp_out']
                    total_heat_absorbed += summary['total_heat_transfer']
                    
                    self.section_results[section_name] = {
                        'summary': summary,
                        'segments': segment_results
                    }
                    
                    print(f"    Q: {summary['total_heat_transfer']/1e6:.1f} MMBtu/hr, Gas out: {current_gas_temp:.0f}°F")
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    raise RuntimeError(f"Failed to solve {section_name}: {e}")
            
            # Attemperator control
            final_steam_temp = section_summaries['superheater_secondary']['water_temp_out']
            temp_error = abs(final_steam_temp - self.final_steam_temp_target)
            
            if temp_error > tolerance:
                self.attemperator_flow = self.calculate_attemperator_flow(
                    final_steam_temp, self.final_steam_temp_target, main_steam_flow
                )
                
                if self.attemperator_flow > 0:
                    # Apply attemperator correction
                    steam_props = self.property_calc.get_steam_properties(final_steam_temp, self.steam_pressure)
                    water_props = self.property_calc.get_steam_properties(self.feedwater_temp, self.steam_pressure)
                    
                    adjusted_temp = ((final_steam_temp * main_steam_flow * steam_props.cp + 
                                    self.feedwater_temp * self.attemperator_flow * water_props.cp) / 
                                   (main_steam_flow * steam_props.cp + self.attemperator_flow * water_props.cp))
                    
                    section_summaries['superheater_secondary']['water_temp_out'] = adjusted_temp
                    final_steam_temp = adjusted_temp
                    
                    print(f"    Attemperator: {self.attemperator_flow:.0f} lbm/hr, Final temp: {final_steam_temp:.1f}°F")
            
            # Check convergence
            stack_temp = current_gas_temp
            efficiency = total_heat_absorbed / self.fuel_input
            stack_error = abs(stack_temp - self.stack_temp_target)
            
            print(f"  Stack: {stack_temp:.1f}°F, Steam: {final_steam_temp:.1f}°F, Efficiency: {efficiency:.1%}")
            
            if stack_error < tolerance and temp_error < tolerance:
                print(f"\n✓ Converged after {iteration + 1} iterations")
                break
            
            # Update temperatures with damping
            damping = 0.7
            for section_name in section_order:
                if section_name in section_summaries:
                    old_temp = water_temps.get(section_name, 200)
                    new_temp = section_summaries[section_name]['water_temp_out']
                    water_temps[section_name] = old_temp * (1 - damping) + new_temp * damping
        
        else:
            print(f"\n⚠ Did not converge within {max_iterations} iterations")
        
        # Store system performance
        self.system_performance = {
            'total_heat_absorbed': total_heat_absorbed,
            'system_efficiency': efficiency,
            'final_steam_temperature': final_steam_temp,
            'steam_superheat': final_steam_temp - 532,
            'stack_temperature': stack_temp,
            'attemperator_flow': self.attemperator_flow,
            'steam_production': main_steam_flow + self.attemperator_flow,
            'iterations_to_converge': iteration + 1
        }
        
        return self.section_results


class MLDatasetGenerator:
    """Generate comprehensive datasets for machine learning-based soot blowing optimization."""
    
    def __init__(self, boiler_system):
        """Initialize ML dataset generator."""
        self.boiler_system = boiler_system
        self.fouling_integrator = CombustionFoulingIntegrator()
        self.dataset = []
        
    def generate_comprehensive_dataset(self, num_scenarios: int = 10000) -> pd.DataFrame:
        """Generate comprehensive dataset for ML training."""
        print(f"🤖 Generating comprehensive dataset with {num_scenarios} scenarios...")
        
        scenarios_data = []
        
        for i in range(num_scenarios):
            if i % 1000 == 0:
                print(f"  Progress: {i}/{num_scenarios} scenarios generated...")
            
            # Generate random operating scenario
            scenario = self._generate_random_scenario()
            
            # Calculate combustion and soot production
            combustion_data = self._calculate_combustion_scenario(scenario)
            
            # Calculate fouling buildup over time
            fouling_timeline = self._simulate_fouling_timeline(
                scenario, combustion_data, timeline_hours=720  # 30 days
            )
            
            # Generate multiple cleaning scenarios for this operating condition
            cleaning_scenarios = self._generate_cleaning_scenarios(fouling_timeline)
            
            # Evaluate each cleaning scenario
            for cleaning in cleaning_scenarios:
                scenario_data = self._evaluate_cleaning_scenario(
                    scenario, combustion_data, fouling_timeline, cleaning
                )
                scenarios_data.append(scenario_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(scenarios_data)
        
        print(f"✅ Dataset generation complete: {len(df)} total data points")
        return df