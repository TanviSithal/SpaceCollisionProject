"""
SPACE COLLISION DETECTION ENGINE
================================

Reads propagated satellite states from propagated_satellites.json,
analyzes every satellite pair, calculates current distance,
relative velocity, closest approach and risk, and saves
collision_results.csv.

IMPORTANT:
This is a prototype screening system.
It is NOT an operational collision probability system.
"""

import json
import csv
import os
import sys
import itertools
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)
from collision_analysis import analyze_collision


# ============================================================
# FILE PATHS
# ============================================================

DATA_FILE = os.path.join(
    "data",
    "propagated_satellites.json"
)

RESULTS_FILE = "collision_results.csv"


# ============================================================
# LOAD PROPAGATED DATA
# ============================================================

def load_propagated_data():

    print()
    print("=" * 70)
    print("LOADING PROPAGATED SATELLITE DATA")
    print("=" * 70)

    print()
    print("File:")
    print(os.path.abspath(DATA_FILE))

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    satellites = data["satellites"]

    print()
    print(
        f"Total satellites: "
        f"{len(satellites)}"
    )

    print(
        f"Successful propagation: "
        f"{data['successful']}"
    )

    print(
        f"Failed propagation: "
        f"{data['failed']}"
    )

    return satellites


# ============================================================
# CONVERT SATELLITE DATA
# ============================================================

def prepare_satellite(satellite):

    return {
        "name": satellite["object_name"],
        "norad_id": satellite["norad_id"],

        "position": np.array([
            satellite["position"]["x"],
            satellite["position"]["y"],
            satellite["position"]["z"]
        ], dtype=float),

        "velocity": np.array([
            satellite["velocity"]["vx"],
            satellite["velocity"]["vy"],
            satellite["velocity"]["vz"]
        ], dtype=float)
    }


# ============================================================
# ANALYZE ALL PAIRS
# ============================================================

def analyze_all_pairs(satellites):

    print()
    print("=" * 70)
    print("SATELLITE PAIR ANALYSIS")
    print("=" * 70)

    total_pairs = (
        len(satellites)
        * (len(satellites) - 1)
        // 2
    )

    print()
    print(
        f"Total possible pairs: "
        f"{total_pairs:,}"
    )

    results = []

    for index, (sat_a, sat_b) in enumerate(
        itertools.combinations(satellites, 2),
        start=1
    ):

        try:

            result = analyze_collision(
                sat_a["position"],
                sat_a["velocity"],
                sat_b["position"],
                sat_b["velocity"]
            )

            results.append({

                "satellite_a":
                    sat_a["name"],

                "satellite_b":
                    sat_b["name"],

                "norad_a":
                    sat_a["norad_id"],

                "norad_b":
                    sat_b["norad_id"],

                "distance_km":
                    result["distance"],

                "relative_speed_km_s":
                    result["relative_speed"],

                "time_to_cpa_seconds":
                    result["time_to_cpa"],

                "minimum_distance_km":
                    result["minimum_distance"],

                "risk":
                    result["risk"],

                "risk_score":
                    result["probability"]
            })

        except Exception as error:

            print(
                f"Pair analysis error: "
                f"{error}"
            )

        # Progress display
        if (
            index % 1000 == 0
            or index == total_pairs
        ):

            print(
                f"Processed "
                f"{index:,}/{total_pairs:,}"
            )

    return results


# ============================================================
# SORT RESULTS
# ============================================================

def sort_results(results):

    return sorted(
        results,
        key=lambda x: x["minimum_distance_km"]
    )


# ============================================================
# DISPLAY CLOSE APPROACHES
# ============================================================

def display_results(results):

    print()
    print("=" * 70)
    print("CLOSE APPROACH RESULTS")
    print("=" * 70)

    # Only show potentially dangerous results
    close_results = [
        result
        for result in results
        if result["minimum_distance_km"] <= 500
    ]

    if not close_results:

        print()
        print(
            "No potential close approaches "
            "within 500 km."
        )

        return

    print()
    print(
        f"Potential close approaches: "
        f"{len(close_results)}"
    )

    for index, result in enumerate(
        close_results[:20],
        start=1
    ):

        print()
        print(
            f"{index}. "
            f"{result['satellite_a']} "
            f"<-> "
            f"{result['satellite_b']}"
        )

        print(
            f"   NORAD: "
            f"{result['norad_a']} "
            f"<-> "
            f"{result['norad_b']}"
        )

        print(
            f"   Current distance: "
            f"{result['distance_km']:.3f} km"
        )

        print(
            f"   Minimum predicted distance: "
            f"{result['minimum_distance_km']:.3f} km"
        )

        print(
            f"   Relative speed: "
            f"{result['relative_speed_km_s']:.3f} km/s"
        )

        print(
            f"   Time to CPA: "
            f"{result['time_to_cpa_seconds']:.2f} s"
        )

        print(
            f"   Risk: "
            f"{result['risk']}"
        )

        print(
            f"   Risk score: "
            f"{result['risk_score']}%"
        )


# ============================================================
# SAVE CSV
# ============================================================

def save_results(results):

    print()
    print("=" * 70)
    print("SAVING COLLISION RESULTS")
    print("=" * 70)

    fieldnames = [
        "satellite_a",
        "satellite_b",
        "norad_a",
        "norad_b",
        "distance_km",
        "relative_speed_km_s",
        "time_to_cpa_seconds",
        "minimum_distance_km",
        "risk",
        "risk_score"
    ]

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    print()
    print(
        "Collision results saved to:"
    )

    print(
        os.path.abspath(
            RESULTS_FILE
        )
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(results):

    print()
    print("=" * 70)
    print("COLLISION ANALYSIS SUMMARY")
    print("=" * 70)

    high = sum(
        1
        for r in results
        if r["risk"] == "HIGH"
    )

    critical = sum(
        1
        for r in results
        if r["risk"] == "CRITICAL"
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

    print()
    print(
        f"Total pairs analyzed: "
        f"{len(results):,}"
    )

    print(
        f"CRITICAL: {critical}"
    )

    print(
        f"HIGH:     {high}"
    )

    print(
        f"MEDIUM:   {medium}"
    )

    print(
        f"LOW:      {low}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SPACE COLLISION DETECTION ENGINE")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load propagated data
    # --------------------------------------------------------

    satellites_raw = load_propagated_data()

    # --------------------------------------------------------
    # 2. Prepare satellite states
    # --------------------------------------------------------

    satellites = []

    for satellite in satellites_raw:

        try:

            satellites.append(
                prepare_satellite(
                    satellite
                )
            )

        except Exception as error:

            print(
                "Skipping satellite:",
                error
            )

    print()
    print(
        f"Prepared {len(satellites)} "
        f"satellite states."
    )

    # --------------------------------------------------------
    # 3. Analyze every pair
    # --------------------------------------------------------

    results = analyze_all_pairs(
        satellites
    )

    # --------------------------------------------------------
    # 4. Sort
    # --------------------------------------------------------

    results = sort_results(
        results
    )

    # --------------------------------------------------------
    # 5. Display
    # --------------------------------------------------------

    display_results(
        results
    )

    # --------------------------------------------------------
    # 6. Save CSV
    # --------------------------------------------------------

    save_results(
        results
    )

    # --------------------------------------------------------
    # 7. Summary
    # --------------------------------------------------------

    print_summary(
        results
    )

    print()
    print("=" * 70)
    print("COLLISION ENGINE COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()