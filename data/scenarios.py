# project_root/data/scenarios.py

"""
96-hour profiles (combination of 4 seasons: spring, summer, autumn, winter)
0 to 23: spring
24 to 47: summer
48 to 71: autumn
72 to 95: winter
"""

# Spring
L_SP = [0.28, 0.22, 0.21, 0.19, 0.19, 0.25, 0.26, 0.25, 0.23, 0.30, 0.35, 0.36, 0.37, 0.35, 0.30, 0.32, 0.43, 0.60, 0.70, 0.75, 0.69, 0.60, 0.50, 0.40]
PV_SP = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.30, 0.50, 0.60, 0.76, 0.81, 0.85, 0.80, 0.75, 0.60, 0.49, 0.30, 0.10, 0.01, 0.01, 0.01, 0.01, 0.01]

# Summer (High Load, High PV)
L_SU = [0.76, 0.68, 0.61, 0.58, 0.57, 0.61, 0.51, 0.46, 0.48, 0.56, 0.66, 0.71, 0.75, 0.74, 0.74, 0.73, 0.75, 0.81, 0.82, 0.88, 0.98, 1.01, 0.99, 0.89]
PV_SU = [0.00, 0.00, 0.00, 0.00, 0.00, 0.36, 0.41, 0.60, 0.81, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.81, 0.61, 0.41, 0.41, 0.00, 0.00, 0.00, 0.00]

# Autumn
L_AU = [0.35, 0.30, 0.27, 0.25, 0.26, 0.32, 0.29, 0.28, 0.28, 0.30, 0.34, 0.35, 0.40, 0.39, 0.37, 0.36, 0.42, 0.42, 0.49, 0.58, 0.61, 0.63, 0.59, 0.52]
PV_AU = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.30, 0.50, 0.60, 0.74, 0.80, 0.84, 0.80, 0.75, 0.60, 0.50, 0.30, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00]

# Winter (High Evening Peak, Low PV)
L_WI = [0.30, 0.27, 0.25, 0.24, 0.25, 0.27, 0.31, 0.30, 0.30, 0.29, 0.38, 0.42, 0.42, 0.34, 0.30, 0.33, 0.44, 0.70, 0.78, 0.82, 0.80, 0.65, 0.50, 0.40]
PV_WI = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.15, 0.30, 0.45, 0.55, 0.70, 0.55, 0.45, 0.30, 0.16, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]

# Concatenating into 96-hour arrays
LOAD_PROFILE = L_SP + L_SU + L_AU + L_WI
PV_PROFILE = PV_SP + PV_SU + PV_AU + PV_WI

# Prices ($/MWh)
# Each number represents the price of purchasing electricity from the 
# upstream substation at a specific hour of the day (00:00 to 12:00).
base_price = [25.0, 22.0, 20.0, 20.0, 22.0, 28.0, 35.0, 45.0, 50.0, 48.0, 40.0, 35.0, 30.0, 30.0, 32.0, 40.0, 60.0, 85.0, 120.0, 110.0, 90.0, 65.0, 45.0, 30.0]
# Summer is 50% more expensive and 
# winter is 20% more expensive due to peak consumption.
summer_price = [p * 1.5 for p in base_price]
winter_price = [p * 1.2 for p in base_price]

RTP_PRICE = base_price + summer_price + base_price + winter_price