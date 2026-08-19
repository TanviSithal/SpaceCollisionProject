import json
import os
from datetime import datetime, timezone

from sgp4 import omm
from sgp4.api import Satrec
from sgp4.conveniences import jday_datetime


# ============================================================
# FILE LOCATIONS
# ============================================================

DATA_FILE = "data/satellites.json"
OUTPUT_FILE = "data/propagated_satellites.json"


# ============================================================
# LOAD CELESTRAK DATA
# ============================================================

with open(DATA_FILE, "r", encoding="utf-8") as file:
    records = json.load(file)


print("=" * 60)
print("SGP4 + CELESTRAK SATELLITE PROPAGATION")
print("=" * 60)

print("Total satellites:", len(records))


# ============================================================
# CURRENT UTC TIME
# ============================================================

now = datetime.now(timezone.utc)

print()
print("Propagation time:", now)


# ============================================================
# CONVERT TIME TO JULIAN DATE
# ============================================================

jd, fr = jday_datetime(now)


# ============================================================
# STORE PROPAGATED SATELLITES
# ============================================================

propagated_satellites = []

successful = 0
failed = 0


# ============================================================
# PROPAGATE ALL SATELLITES
# ============================================================

for index, record in enumerate(records, start=1):

    try:

        # ----------------------------------------------------
        # PREPARE OMM FIELDS
        # ----------------------------------------------------

        fields = {
            "OBJECT_NAME": record["OBJECT_NAME"],
            "OBJECT_ID": record["OBJECT_ID"],
            "EPOCH": record["EPOCH"],
            "MEAN_MOTION": record["MEAN_MOTION"],
            "ECCENTRICITY": record["ECCENTRICITY"],
            "INCLINATION": record["INCLINATION"],
            "RA_OF_ASC_NODE": record["RA_OF_ASC_NODE"],
            "ARG_OF_PERICENTER": record["ARG_OF_PERICENTER"],
            "MEAN_ANOMALY": record["MEAN_ANOMALY"],
            "EPHEMERIS_TYPE": record["EPHEMERIS_TYPE"],
            "CLASSIFICATION_TYPE": record["CLASSIFICATION_TYPE"],
            "NORAD_CAT_ID": record["NORAD_CAT_ID"],
            "ELEMENT_SET_NO": record["ELEMENT_SET_NO"],
            "REV_AT_EPOCH": record["REV_AT_EPOCH"],
            "BSTAR": record["BSTAR"],
            "MEAN_MOTION_DOT": record["MEAN_MOTION_DOT"],
            "MEAN_MOTION_DDOT": record["MEAN_MOTION_DDOT"],
        }


        # ----------------------------------------------------
        # CREATE SGP4 OBJECT
        # ----------------------------------------------------

        sat = Satrec()

        omm.initialize(sat, fields)


        # ----------------------------------------------------
        # PROPAGATE
        # ----------------------------------------------------

        error, position, velocity = sat.sgp4(jd, fr)


        # ----------------------------------------------------
        # CHECK RESULT
        # ----------------------------------------------------

        if error == 0:

            satellite_data = {
                "object_name": record["OBJECT_NAME"],
                "norad_id": record["NORAD_CAT_ID"],
                "epoch": record["EPOCH"],

                "position": {
                    "x": position[0],
                    "y": position[1],
                    "z": position[2]
                },

                "velocity": {
                    "vx": velocity[0],
                    "vy": velocity[1],
                    "vz": velocity[2]
                }
            }

            propagated_satellites.append(satellite_data)

            successful += 1


            # Print first few satellites
            if index <= 5:

                print()
                print(f"[{index}/{len(records)}] {record['OBJECT_NAME']}")
                print(f"NORAD ID: {record['NORAD_CAT_ID']}")

                print(
                    f"Position: "
                    f"X={position[0]:.3f}, "
                    f"Y={position[1]:.3f}, "
                    f"Z={position[2]:.3f} km"
                )

                print(
                    f"Velocity: "
                    f"Vx={velocity[0]:.6f}, "
                    f"Vy={velocity[1]:.6f}, "
                    f"Vz={velocity[2]:.6f} km/s"
                )


        else:

            failed += 1

            print()
            print(
                f"Propagation failed for "
                f"{record['OBJECT_NAME']} "
                f"(Error {error})"
            )


    except Exception as e:

        failed += 1

        print()
        print(
            f"Error processing "
            f"{record.get('OBJECT_NAME', 'Unknown')}: {e}"
        )


# ============================================================
# SAVE PROPAGATED DATA
# ============================================================

output_data = {
    "propagation_time": now.isoformat(),
    "total_satellites": len(records),
    "successful": successful,
    "failed": failed,
    "satellites": propagated_satellites
}


# Make sure data folder exists
os.makedirs("data", exist_ok=True)


with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    json.dump(
        output_data,
        file,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("PROPAGATION COMPLETE")
print("=" * 60)

print("Total satellites :", len(records))
print("Successful       :", successful)
print("Failed           :", failed)

print()
print("Output file:")
print(OUTPUT_FILE)

print("=" * 60)