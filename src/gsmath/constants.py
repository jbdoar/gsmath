"""Physical and colorimetric constants used throughout gsmath."""

# Maximum spectral luminous efficacy for photopic vision in lumens per watt.
# Unit conversion belongs in the color analysis rather than in this value.
K_M = 683.002

# Exact SI values used to convert radiant power to photon rate.
PLANCK_CONSTANT = 6.62607015e-34  # J s
SPEED_OF_LIGHT = 299792458.0  # m / s

# Backward-compatible spelling retained for callers of the original API.
K_m = K_M
