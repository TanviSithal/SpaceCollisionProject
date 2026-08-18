from satellite_data import (
    get_available_satellites,
    get_current_state
)

from collision_analysis import (
    analyze_pair
)


# ============================================================
# LOAD SATELLITES
# ============================================================

satellites = get_available_satellites()

print(
    f"Found {len(satellites)} satellites."
)


# ============================================================
# SELECT TWO SATELLITES
# ============================================================

satellite_a_id = satellites[0]["norad_id"]

satellite_b_id = satellites[1]["norad_id"]


print()
print("Testing two satellites:")
print("-----------------------")

print(
    "Satellite A:",
    satellite_a_id
)

print(
    "Satellite B:",
    satellite_b_id
)


# ============================================================
# GET LIVE STATES
# ============================================================

state_a = get_current_state(
    satellite_a_id
)

state_b = get_current_state(
    satellite_b_id
)


# ============================================================
# ANALYZE
# ============================================================

result = analyze_pair(
    state_a,
    state_b
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("==============================")
print("COLLISION ANALYSIS")
print("==============================")

print(
    "Satellite A:",
    result["satellite_a"]
)

print(
    "NORAD A:",
    result["norad_a"]
)

print()

print(
    "Satellite B:",
    result["satellite_b"]
)

print(
    "NORAD B:",
    result["norad_b"]
)

print()

print(
    "Distance:",
    f'{result["distance_km"]:.3f} km'
)

print(
    "Relative velocity:",
    f'{result["relative_velocity_km_s"]:.3f} km/s'
)

print(
    "Altitude difference:",
    f'{result["altitude_difference_km"]:.3f} km'
)

print()

print(
    "Risk:",
    result["risk"]
)