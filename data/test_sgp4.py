import json
from sgp4.api import Satrec
from datetime import datetime, timezone

# Path to downloaded CelesTrak data
DATA_FILE = "data/satellites.json"


# ------------------------------------------------------------
# Load satellite data
# ------------------------------------------------------------

with open(DATA_FILE, "r", encoding="utf-8") as file:
    satellites = json.load(file)


print("=" * 60)
print("SGP4 SATELLITE POSITION TEST")
print("=" * 60)

print("Satellite records loaded:", len(satellites))


# ------------------------------------------------------------
# Select first satellite
# ------------------------------------------------------------

satellite = satellites[0]

print()
print("Satellite:")
print("Name:", satellite.get("OBJECT_NAME"))
print("NORAD ID:", satellite.get("NORAD_CAT_ID"))


# ------------------------------------------------------------
# Create SGP4 satellite object
# ------------------------------------------------------------

sat = Satrec()

sat.sgp4init(
    84,                         # WGS84
    "i",                        # improved mode
    satellite["NORAD_CAT_ID"],
    satellite["BSTAR"],
    satellite["MEAN_MOTION_DOT"],
    satellite["MEAN_MOTION_DDOT"],
    satellite["ECCENTRICITY"],
    satellite["ARG_OF_PERICENTER"],
    satellite["INCLINATION"],
    satellite["MEAN_ANOMALY"],
    satellite["MEAN_MOTION"],
    satellite["RA_OF_ASC_NODE"],
    satellite["EPOCH"]
)

print()
print("SGP4 satellite object created successfully!")


# ------------------------------------------------------------
# Calculate position at current time
# ------------------------------------------------------------

now = datetime.now(timezone.utc)

jd = now.timestamp() / 86400.0 + 2440587.5

jd_int = int(jd)
jd_fraction = jd - jd_int


error, position, velocity = sat.sgp4(
    jd_int,
    jd_fraction
)


# ------------------------------------------------------------
# Display result
# ------------------------------------------------------------

print()
print("Current UTC time:", now)

print()
print("SGP4 error code:", error)

if error == 0:

    print()
    print("Satellite position (km):")

    print("X =", position[0])
    print("Y =", position[1])
    print("Z =", position[2])

    print()
    print("Satellite velocity (km/s):")

    print("Vx =", velocity[0])
    print("Vy =", velocity[1])
    print("Vz =", velocity[2])

else:

    print()
    print("SGP4 propagation failed.")

print("=" * 60)