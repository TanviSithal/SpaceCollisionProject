from satellite_data import get_available_satellites
from satellite_data import get_current_state


print("Downloading CelesTrak data...")

satellites = get_available_satellites()

print(
    f"Found {len(satellites)} satellites."
)

print()
print("First 10 satellites:")
print("----------------------")

for sat in satellites[:10]:

    print(
        sat["norad_id"],
        "-",
        sat["name"]
    )


# ============================================================
# TEST FIRST SATELLITE
# ============================================================

satellite_id = satellites[0]["norad_id"]

print()
print(
    "Testing satellite:",
    satellite_id
)

state = get_current_state(
    satellite_id
)


print()
print("==============================")
print("LIVE SATELLITE STATE")
print("==============================")

print(
    "Name:",
    state["name"]
)

print(
    "NORAD:",
    state["norad_id"]
)

print(
    "Latitude:",
    round(
        state["latitude"],
        4
    ),
    "degrees"
)

print(
    "Longitude:",
    round(
        state["longitude"],
        4
    ),
    "degrees"
)

print(
    "Altitude:",
    round(
        state["altitude"],
        2
    ),
    "km"
)

print()
print("Position XYZ (km):")

print(
    state["position"]
)

print()
print("Velocity (km/s):")

print(
    state["velocity"]
)

print()
print(
    "Timestamp:",
    state["timestamp"]
)