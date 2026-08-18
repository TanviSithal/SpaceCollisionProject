import numpy as np
from datetime import datetime, timedelta, timezone

from skyfield.api import load

from satellite_data import (
    get_available_satellites,
    create_satellite,
)


# ============================================================
# SETTINGS
# ============================================================

PREDICTION_HOURS = 24
STEP_MINUTES = 10

# We use this only to decide which pairs to analyze
# after checking their future trajectories.
SCREENING_DISTANCE_KM = 500.0

# Risk thresholds
HIGH_RISK_KM = 5.0
MEDIUM_RISK_KM = 25.0
LOW_RISK_KM = 100.0


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(distance_km):

    if distance_km <= HIGH_RISK_KM:
        return "HIGH"

    elif distance_km <= MEDIUM_RISK_KM:
        return "MEDIUM"

    elif distance_km <= LOW_RISK_KM:
        return "LOW"

    else:
        return "SAFE"


# ============================================================
# CREATE FUTURE TIME GRID
# ============================================================

def create_time_grid():

    now = datetime.now(timezone.utc)

    times = []

    total_steps = int(
        (PREDICTION_HOURS * 60) / STEP_MINUTES
    )

    for step in range(total_steps + 1):

        future_time = (
            now
            + timedelta(minutes=step * STEP_MINUTES)
        )

        times.append(future_time)

    return times


# ============================================================
# GET FUTURE POSITIONS
# ============================================================

def propagate_satellite(satellite, times, ts):

    skyfield_times = ts.from_datetimes(times)

    geocentric = satellite.at(skyfield_times)

    positions = geocentric.position.km

    velocities = geocentric.velocity.km_per_s

    return positions, velocities


# ============================================================
# CALCULATE CLOSE APPROACH
# ============================================================

def analyze_pair(
    name_a,
    norad_a,
    positions_a,
    velocities_a,
    name_b,
    norad_b,
    positions_b,
    velocities_b,
    times,
):

    # Difference between satellite positions
    differences = positions_a - positions_b

    # Distance at every time
    distances = np.linalg.norm(
        differences,
        axis=0
    )

    # Find minimum distance
    minimum_index = int(
        np.argmin(distances)
    )

    minimum_distance = float(
        distances[minimum_index]
    )

    # Relative velocity at closest approach
    relative_velocity_vector = (
        velocities_a[:, minimum_index]
        -
        velocities_b[:, minimum_index]
    )

    relative_velocity = float(
        np.linalg.norm(
            relative_velocity_vector
        )
    )

    # Current distance
    current_distance = float(
        distances[0]
    )

    return {

        "name_a": name_a,
        "name_b": name_b,

        "norad_a": norad_a,
        "norad_b": norad_b,

        "current_distance": current_distance,

        "minimum_distance": minimum_distance,

        "relative_velocity": relative_velocity,

        "time_of_closest_approach":
            times[minimum_index],

    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SPACE COLLISION WARNING SYSTEM")
    print("=" * 70)

    # --------------------------------------------------------
    # DOWNLOAD DATA
    # --------------------------------------------------------

    print()
    print("Downloading satellite data...")

    satellites = get_available_satellites()

    print()
    print(
        f"Satellites available: "
        f"{len(satellites)}"
    )

    # --------------------------------------------------------
    # SKYFIELD SETUP
    # --------------------------------------------------------

    print()
    print("Preparing orbital models...")

    ts = load.timescale()

    # --------------------------------------------------------
    # CREATE SKYFIELD SATELLITES
    # --------------------------------------------------------

    satellite_objects = []

    print()
    print("Creating satellite orbital models...")
    print("-" * 40)

    for index, sat_data in enumerate(
        satellites,
        start=1
    ):

        try:

            satellite = create_satellite(
                sat_data
            )

            satellite_objects.append(
                {
                    "name": sat_data["name"],
                    "norad_id": sat_data["norad_id"],
                    "satellite": satellite
                }
            )

            print(
                f"[{index}/{len(satellites)}] "
                f"{sat_data['norad_id']} - "
                f"{sat_data['name']}"
            )

        except Exception as e:

            print(
                f"[{index}/{len(satellites)}] "
                f"{sat_data['norad_id']} - "
                f"ERROR: {e}"
            )

    print()
    print(
        f"Successfully created orbital models "
        f"for {len(satellite_objects)} satellites."
    )

    # --------------------------------------------------------
    # CREATE FUTURE TIME GRID
    # --------------------------------------------------------

    times = create_time_grid()

    print()
    print("=" * 70)
    print("PROPAGATING SATELLITE TRAJECTORIES")
    print("=" * 70)

    print()
    print(
        f"Prediction window: "
        f"{PREDICTION_HOURS} hours"
    )

    print(
        f"Prediction step: "
        f"{STEP_MINUTES} minutes"
    )

    print(
        f"Time samples: "
        f"{len(times)}"
    )

    # --------------------------------------------------------
    # PROPAGATE ALL SATELLITES
    # --------------------------------------------------------

    trajectories = []

    for index, sat in enumerate(
        satellite_objects,
        start=1
    ):

        print(
            f"[{index}/{len(satellite_objects)}] "
            f"Propagating "
            f"{sat['name']}"
        )

        try:

            positions, velocities = (
                propagate_satellite(
                    sat["satellite"],
                    times,
                    ts
                )
            )

            trajectories.append(
                {
                    "name": sat["name"],
                    "norad_id": sat["norad_id"],
                    "positions": positions,
                    "velocities": velocities
                }
            )

        except Exception as e:

            print(
                f"  ERROR: {e}"
            )

    print()
    print(
        f"Successfully propagated "
        f"{len(trajectories)} satellites."
    )

    # --------------------------------------------------------
    # ANALYZE ALL PAIRS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ANALYZING SATELLITE PAIRS")
    print("=" * 70)

    total_pairs = 0

    results = []

    for i in range(len(trajectories)):

        for j in range(i + 1, len(trajectories)):

            total_pairs += 1

            sat_a = trajectories[i]
            sat_b = trajectories[j]

            result = analyze_pair(
                sat_a["name"],
                sat_a["norad_id"],
                sat_a["positions"],
                sat_a["velocities"],

                sat_b["name"],
                sat_b["norad_id"],
                sat_b["positions"],
                sat_b["velocities"],

                times
            )

            # Only keep pairs that become close
            # during the prediction window.
            if (
                result["minimum_distance"]
                <= SCREENING_DISTANCE_KM
            ):

                result["risk"] = classify_risk(
                    result["minimum_distance"]
                )

                results.append(result)

    # --------------------------------------------------------
    # SORT RESULTS
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["minimum_distance"]
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FUTURE COLLISION ANALYSIS RESULTS")
    print("=" * 70)

    print()
    print(
        f"Total pairs analyzed: "
        f"{total_pairs}"
    )

    print(
        f"Pairs predicted within "
        f"{SCREENING_DISTANCE_KM} km: "
        f"{len(results)}"
    )

    # --------------------------------------------------------
    # DISPLAY TOP RESULTS
    # --------------------------------------------------------

    for index, result in enumerate(
        results[:20],
        start=1
    ):

        print()
        print(
            f"{index}. "
            f"{result['name_a']} "
            f"<-> "
            f"{result['name_b']}"
        )

        print(
            f"   NORAD: "
            f"{result['norad_a']} "
            f"<-> "
            f"{result['norad_b']}"
        )

        print(
            f"   Current distance: "
            f"{result['current_distance']:.3f} km"
        )

        print(
            f"   Minimum predicted distance: "
            f"{result['minimum_distance']:.3f} km"
        )

        print(
            f"   Relative velocity: "
            f"{result['relative_velocity']:.3f} km/s"
        )

        print(
            f"   Time of closest approach: "
            f"{result['time_of_closest_approach']}"
        )

        print(
            f"   Risk: "
            f"{result['risk']}"
        )

        if result["risk"] == "HIGH":

            print(
                "   !!! WARNING: "
                "VERY CLOSE APPROACH !!!"
            )

        elif result["risk"] == "MEDIUM":

            print(
                "   WARNING: CLOSE APPROACH"
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    high = sum(
        1
        for r in results
        if r["risk"] == "HIGH"
    )

    medium = sum(
        1
        for r in results
        if r["risk"] == "MEDIUM"
    )

    low = sum(
        1
        for r in results
        if r["risk"] == "LOW"
    )

    safe = sum(
        1
        for r in results
        if r["risk"] == "SAFE"
    )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"High risk:   {high}"
    )

    print(
        f"Medium risk: {medium}"
    )

    print(
        f"Low risk:    {low}"
    )

    print(
        f"Safe:        {safe}"
    )

    print()
    print(
        f"Prediction window: "
        f"{PREDICTION_HOURS} hours"
    )

    print(
        f"Prediction step: "
        f"{STEP_MINUTES} minutes"
    )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()