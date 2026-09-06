"""IEEE 33-bus radial distribution-system data.

The arrays are stored with zero-based indices. IEEE bus number ``n`` maps to
internal index ``n - 1``. Bus 1 / internal index 0 is the substation and is not
a candidate installation bus.
"""

from data.devices import LINE_RATING_MVA

S_BASE = 1.0     # MVA
V_BASE = 12.66   # kV line-to-line
Z_BASE = V_BASE**2 / S_BASE
I_BASE_KA = S_BASE / (3.0**0.5 * V_BASE)

NUM_BUSES = 33
NUM_BRANCHES = 32
T_HORIZON = 96

RAW_BRANCH_DATA = (
    (0, 1, 0.0922, 0.0470), (1, 2, 0.4930, 0.2511),
    (2, 3, 0.3660, 0.1864), (3, 4, 0.3811, 0.1941),
    (4, 5, 0.8190, 0.7070), (5, 6, 0.1872, 0.6188),
    (6, 7, 0.7114, 0.2351), (7, 8, 1.0300, 0.7400),
    (8, 9, 1.0440, 0.7400), (9, 10, 0.1966, 0.0650),
    (10, 11, 0.3744, 0.1238), (11, 12, 1.4680, 1.1550),
    (12, 13, 0.5416, 0.7129), (13, 14, 0.5910, 0.5260),
    (14, 15, 0.7463, 0.5450), (15, 16, 1.2890, 1.7210),
    (16, 17, 0.7320, 0.5740), (1, 18, 0.1640, 0.1565),
    (18, 19, 1.5042, 1.3554), (19, 20, 0.4095, 0.4784),
    (20, 21, 0.7089, 0.9373), (2, 22, 0.4512, 0.3083),
    (22, 23, 0.8980, 0.7091), (23, 24, 0.8960, 0.7011),
    (5, 25, 0.2030, 0.1034), (25, 26, 0.2842, 0.1447),
    (26, 27, 1.0590, 0.9337), (27, 28, 0.8042, 0.7006),
    (28, 29, 0.5075, 0.2585), (29, 30, 0.9744, 0.9630),
    (30, 31, 0.3105, 0.3619), (31, 32, 0.3410, 0.5302),
)

BRANCHES = {
    k: {
        "from": int(row[0]),
        "to": int(row[1]),
        "r": float(row[2]) / Z_BASE,
        "x": float(row[3]) / Z_BASE,
        "rating_mva": float(LINE_RATING_MVA[k]),
        "i_max_pu": float(LINE_RATING_MVA[k]) / S_BASE,
        "i_max_sq": (float(LINE_RATING_MVA[k]) / S_BASE) ** 2,
    }
    for k, row in enumerate(RAW_BRANCH_DATA)
}

RAW_LOAD_KW_KVAR = (
    (0, 0), (100, 60), (90, 40), (120, 80), (60, 30), (60, 20),

    (200, 100), (200, 100), (60, 20), (60, 20), (45, 30),
    (60, 35), (60, 35), (120, 80), (60, 10), (60, 20),
    (60, 20), (90, 40), (90, 40), (90, 40), (90, 40), (90, 40),
    (90, 50), (420, 200), (420, 200), (60, 25), (60, 25),
    (60, 20), (120, 70), (200, 600), (150, 70), (210, 100),
    (60, 40),
)

LOADS = {
    i: {
        "P": float(row[0]) / 1000.0 / S_BASE,
        "Q": float(row[1]) / 1000.0 / S_BASE,
    }
    for i, row in enumerate(RAW_LOAD_KW_KVAR)
}


def to_ieee_bus(internal_index):
    """Convert an internal zero-based bus index to the IEEE bus number."""

    return int(internal_index) + 1


def to_internal_bus(ieee_bus):
    """Convert an IEEE bus number to the internal zero-based index."""

    bus = int(ieee_bus)
    if not 1 <= bus <= NUM_BUSES:
        raise ValueError(f"IEEE bus must be in [1, {NUM_BUSES}], received {bus}.")