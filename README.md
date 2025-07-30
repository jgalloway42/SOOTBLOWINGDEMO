# Project Title

## Abstract


## Dataset

## Data Representation and Processing
### Combustion Simulation
* Inputs
    - Coal Ultimate Analysis (by mass including ash and moisture)
    - Coal feed (pph)
    - Fuel-bound NOx Conversion Effciency
    - Ambient Temperature, Relative Humidity, & Atmospheric Pressure
    - Air Flow (scfh)
* Interim Calculations (in order)
    1. Coal HHV adjusted for ash & moisture
    2. Air humidity ratio
    3. Calculate fuel-bound NOx generation
    4. Calculate excess air and assuming no CO or thermal NOx (as a first iteration)
    5. Calculate CO based on excess air.
    6. Update flue gas (excess air, & CO).
    7. Estimate flame temperature from simple model.
    8. Calculate Thermal NOx from flame temperature estimation.
    9. Update flue gas.
    10. Calculate Combustion efficiency and actual heat-release.
    11. Calculate Flame Temperature.
* Outputs
    - Flue gas constiuents by mass fraction
    - Excess O2 in percent
    CO2 (LB): 51.04
CO (LB): 3.61
H2O (LB): 11.50
SO2 (LB): 0.43
NO (LB from fuel-N): 0.22
NO (LB thermal): 0.00
NO (LB total): 0.22
N2 (LB): 229.88
O2 (excess %): 23.88
Heat Released (BTU/hr): 104,103,111.74
Estimated Flame Temp (F): 2,640.17
Total Flue Gas (lb/hr): 320.80


### Generation Approximation

* **Higher CO in flue gas** &rarr; Higher likelihood of incomplete combustion &rarr; Higher likelihood of soot formation. (CO &uarr;, Soot &uarr;)

* **Fuel-Bound NOx** Creation of fuel-rich zones that reduce fuel-bound NOx &rarr; Increased propensity for soot formation if secondary air mixing/burnout is insufficient. (FB NOx efficiency &darr;, Soot &uarr;)

* **Thermal NOx**  Low peak temperatures can indirectly lead to conditions that are also less favorable for complete soot oxidation, or may involve operating closer to fuel-rich conditions to reduce temperature. (Thermal NOx &uarr;, Soot &uarr;)

* **Lower Excess Air** &rarr; Increased risk of fuel-rich zones and incomplete combustion &rarr; Higher soot. (Excess Air &uarr;, Soot &uarr;)



## Model Derivation

## Results

## Conclusion



###### Repo Structure Duplicated from here: https://github.com/drivendata/cookiecutter-data-science
