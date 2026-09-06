"""Technical, economic, and numerical constants.
 
  All power quantities are represented in per-unit inside the optimization  model. Monetary operating-cost coefficients are in USD/MWh and investment
  coefficients are in USD/MW or USD/MVAr as stated below.
  """
 
  # ---------------------------------------------------------------------------
  # Planning technologies
  # ---------------------------------------------------------------------------
 
DEVICE_TYPES = ("ess", "gas", "svc", "cb")
  
  # Existing PV locations. These are zero-based indices corresponding to IEEE
  # buses 9, 11, 14, 17, 19, 21, 23, and 29.
FIXED_PV_NODES = (8, 10, 13, 16, 18, 20, 22, 28)
INCLUDE_FIXED_PV_CAPEX = False
  
# Installed sizes per planning action
P_ESS_MAX = 0.10       # MW on the 1 MVA base
E_ESS_MAX = 0.50       # MWh
E_ESS_MIN = 0.10       # MWh
P_GAS_MAX = 0.50       # MW
Q_SVC_MAX = 0.50       # MVAr (matches the reference study)
Q_CB_MAX = 0.50        # MVAr
PV_CAPACITY = 0.50     # MW per existing PV site
 
# Converter and storage parameters
EFF_CH = 0.90
EFF_DIS = 0.90
PV_INVERTER_OVERSIZE = 1.10
ESS_INVERTER_OVERSIZE = 1.10
GAS_POWER_FACTOR_MARGIN = 1.25
 
# The four 24-hour representative days are independent seasonal scenarios.
SCENARIO_START_HOURS = (0, 24, 48, 72)
SCENARIO_END_HOURS = (23, 47, 71, 95)
 
  # ---------------------------------------------------------------------------
  # Network limits
  # ---------------------------------------------------------------------------
  
V_MIN_SQ = 0.95**2
V_MAX_SQ = 1.05**2
S_SUB_MAX = 10.0       # MVA / per-unit on a 1 MVA base

# The standard IEEE 33-bus data set does not contain conductor ampacities.
# A transparent 5 MVA study rating is therefore used for every branch. The
# vector is deliberately explicit so utility-specific ratings can replace it
# without changing constraints or compliance code.
LINE_RATING_MVA = (5.0,) * 32


# ---------------------------------------------------------------------------
# Investment and operating economics
INTEREST_RATE = 0.05
ASSET_LIFETIME_YEARS = 20
C_ESS_INV = 200_000.0  # USD/MW
C_GAS_INV = 30_000.0   # USD/MW; Zhang et al. (2022), Table 1
C_SVC_INV = 500_000.0  # USD/MVAr
C_CB_INV = 40_000.0    # USD/MVAr
C_PV_INV = 50_000.0    # USD/MW; normally excluded as sunk cost

C_EXPORT_PRICE = 19.99 # USD/MWh
C_GAS_FUEL = 50.0      # USD/MWh
C_EMISSION = 20.0      # USD/MWh
C_DEG = 20.0           # USD/MWh of ESS throughput
C_RES = 500.0          # USD/MWh of renewable curtailment
C_AUL = 5_000.0        # USD/MWh value of lost load
# Soft-model penalties. Final acceptance never relies on these coefficients;
# it uses the independent hard-compliance audit.
PENALTY_VOLT = 1.0e7
PENALTY_OVERLOAD = 1.0e7
PENALTY_THERMAL = 1.0e7
PENALTY_SOC = 1.0e7
 
# A small increasing cost in squared current encourages a tight branch-flow
# relaxation. Exactness is still checked independently after every solve.
EXACTNESS_REGULARIZATION = 1.0e-1
# ---------------------------------------------------------------------------
# Compliance thresholds
# ---------------------------------------------------------------------------
  
CONSTRAINT_TOL = 1.0e-5
SLACK_TOL = 1.0e-6
PHYSICAL_TOL = 1.0e-5
# Engineering numerical-tightness threshold:
# 0.005 pu apparent power = 5 kVA on the 1 MVA system base.
# This is a study assumption and must be reported with sensitivity analysis.
CONE_POWER_TOL_PU = 0.005
CONE_ABS_TOL = CONE_POWER_TOL_PU**2   # 2.5e-5 pu^2
CONE_REL_TOL = 1.0e-4
COMPLEMENTARITY_TOL = 1.0e-7
 
# These targets apply to the unweighted four representative days, matching
# g_expected=[0.1, 0.1] in the reference ADN-planning study. Annualized values
# are also reported but are not compared directly with this threshold.
MAX_LOAD_SHEDDING_MWH_REP = 0.10
MAX_RES_CURTAILMENT_MWH_REP = 0.10
def capital_recovery_factor():
    """Return the annual capital-recovery factor."""
 

    rate = INTEREST_RATE
    years = ASSET_LIFETIME_YEARS
    return rate * (1.0 + rate) ** years / ((1.0 + rate) ** years - 1.0)