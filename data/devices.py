# project_root/data/devices.py

# --- Financial Parameters ---
C_LOSS = 200        # $/MWh
C_RES = 500         # $/MW 
C_AUL = 5000        # $/MW 

# Investment Costs (CapEx)
C_ESS_INV = 200000  # $/MW
C_GAS_INV = 800000   # $/MW
C_SVC_INV = 500000  # $/MVar
C_CB_INV = 40000    # $/MVar
C_PV_INV = 100000   # $/MW 
C_DEG = 20.0

# --- Technical Parameters (Per-Unit based on 1MVA base) ---
P_ESS_MAX = 0.1     
E_ESS_MAX = 0.5     
E_ESS_MIN = 0.1     
EFF_CH = 0.9        
# With EFF_CH=0.9 and EFF_DIS=0.9 (symmetric one-way efficiency),
# the total round-trip efficiency becomes 0.9*0.9=81%, which is 
# consistent with typical lithium batteries in the literature.
# because in constraints.py it is used with the formula
# "P_dis / EFF_DIS" (division, not multiplication).
EFF_DIS = 0.9       

PV_CAPACITY = 0.5   

P_GAS_MAX = 0.5     
Q_SVC_MAX = 1.0     
Q_CB_MAX = 0.5      

# Grid Limits
V_MIN_SQ = 0.95**2
V_MAX_SQ = 1.05**2
# Thermal limit changed to 25.0 (equivalent to 5 MVA).
# Now the root of the network does not choke and the AI ​​has 
# to move equipment to the end of the feeder to fix the voltage drop.
I_MAX_SQ = 25.0
