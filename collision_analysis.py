"""
SPACE COLLISION WARNING SYSTEM
================================

Downloads active satellite orbital data from CelesTrak,
propagates satellite orbits using Skyfield/SGP4,
screens satellite pairs for possible close approaches,
performs detailed analysis only on candidate pairs,
and exports results to CSV.

IMPORTANT:
This is a screening/prediction system.

A predicted close approach is NOT a confirmed collision.
Operational collision avoidance requires high-quality tracking
data, covariance/uncertainty information, and professional
conjunction assessment.
"""

import csv
import json
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from skyfield.api import EarthSatellite, load


# ============================================================
# CONFIGURATION
# ============================================================

# CelesTrak OMM JSON endpoint.
#
# JSON/OMM is preferred over the old TLE endpoint because
# modern catalog numbers can exceed the 5-digit TLE limit.
CELESTRAK_URL = (
    "https://celestrak.org/NORAD/elements/gp.php"
    "?GROUP=active&FORMAT=json"
)

# Local cache.
CACHE_FILE = "satellite_cache.json"

# Cache validity.
# If the cache is newer than this many hours, reuse it.
CACHE_HOURS = 12

# Prediction.
PREDICTION_HOURS = 24
STEP_MINUTES = 10

# Candidate screening distance.
#
# We first screen satellites using a larger spatial cell
# before doing detailed pair analysis.
SCREENING_DISTANCE_KM = 700.0

# Final reporting distance.
ANALYSIS_DISTANCE_KM = 500.0

# Risk thresholds.
HIGH_RISK_KM = 5.0
MEDIUM_RISK_KM = 25.0

# Output.
RESULTS_CSV = "collision_results.csv"
SUMMARY_CSV = "collision_summary.csv"


# ============================================================
# DISPLAY
# ============================================================

def print_separator(char="=", length=70):
    print(char * length)


# ============================================================
# NORAD NUMBER
# ============================================================

def extract_norad_number(value):
    """
    Convert a catalog/NORAD number to an integer when possible.
    """

    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return "UNKNOWN"


# ============================================================
# DOWNLOAD SATELLITE DATA
# ============================================================

def download_satellite_data():
    """
    Download active satellite OMM JSON data from CelesTrak.

    If downloading fails, a recent local cache is used.
    """

    print("Downloading satellite data from CelesTrak...")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/131.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
    }

    try:

        response = requests.get(
            CELESTRAK_URL,
            headers=headers,
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, list):
            raise ValueError(
                "CelesTrak did not return a JSON satellite list."
            )

        # Keep only records containing the required orbital fields.
        satellites = []

        for item in data:

            if not isinstance(item, dict):
                continue

            if (
                "TLE_LINE1" not in item
                or "TLE_LINE2" not in item
            ):
                continue

            name = (
                item.get("OBJECT_NAME")
                or item.get("OBJECT_ID")
                or "UNKNOWN"
            )

            line1 = item["TLE_LINE1"].strip()
            line2 = item["TLE_LINE2"].strip()

            if not (
                line1.startswith("1 ")
                and line2.startswith("2 ")
            ):
                continue

            satellites.append(
                {
                    "name": str(name).strip(),
                    "norad": extract_norad_number(
                        item.get("NORAD_CAT_ID")
                    ),
                    "line1": line1,
                    "line2": line2,
                }
            )

        if len(satellites) < 2:
            raise ValueError(
                "CelesTrak returned too few valid satellites."
            )

        # Save cache.
        with open(
            CACHE_FILE,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                satellites,
                file,
                indent=2,
            )

        print(
            f"Downloaded {len(satellites)} satellites."
        )

        print(
            f"Satellite data cached to: "
            f"{os.path.abspath(CACHE_FILE)}"
        )

        return satellites

    except Exception as exc:

        print()
        print(
            "WARNING: CelesTrak download failed."
        )

        print(
            f"Reason: {exc}"
        )

        # Try cache.
        if os.path.exists(CACHE_FILE):

            try:

                cache_age = (
                    datetime.now().timestamp()
                    - os.path.getmtime(CACHE_FILE)
                ) / 3600.0

                if cache_age <= CACHE_HOURS:

                    print(
                        f"Using local satellite cache "
                        f"({cache_age:.1f} hours old)."
                    )

                    with open(
                        CACHE_FILE,
                        "r",
                        encoding="utf-8",
                    ) as file:

                        satellites = json.load(file)

                    print(
                        f"Loaded {len(satellites)} "
                        f"satellites from cache."
                    )

                    return satellites

                else:

                    print(
                        f"Cache exists but is "
                        f"{cache_age:.1f} hours old."
                    )

            except Exception as cache_exc:

                print(
                    f"WARNING: Could not read cache: "
                    f"{cache_exc}"
                )

        print()
        print(
            "ERROR: No usable satellite data available."
        )

        return []


# ============================================================
# CREATE ORBITAL MODELS
# ============================================================

def create_satellite_models(satellite_data, ts):

    print()
    print_separator()
    print("CREATING ORBITAL MODELS")
    print_separator()

    satellites = []

    total = len(satellite_data)

    for index, item in enumerate(
        satellite_data,
        start=1,
    ):

        name = item["name"]

        print(
            f"[{index}/{total}] {name}"
        )

        try:

            satellite = EarthSatellite(
                item["line1"],
                item["line2"],
                name,
                ts,
            )

            satellites.append(
                {
                    "name": name,
                    "norad": item["norad"],
                    "satellite": satellite,
                }
            )

        except Exception as exc:

            print(
                f"   WARNING: orbital model failed: {exc}"
            )

    print()
    print(
        f"Successfully created orbital models for "
        f"{len(satellites)} satellites."
    )

    return satellites


# ============================================================
# CREATE PREDICTION TIMES
# ============================================================

def create_prediction_times(ts, start_time):

    number_of_steps = int(
        (PREDICTION_HOURS * 60)
        / STEP_MINUTES
    )

    datetimes = [
        start_time
        + timedelta(
            minutes=i * STEP_MINUTES
        )
        for i in range(
            number_of_steps + 1
        )
    ]

    times = ts.from_datetimes(datetimes)

    return times, datetimes


# ============================================================
# PROPAGATE SATELLITES
# ============================================================

def propagate_satellites(
    satellites,
    times,
):

    print()
    print_separator()
    print("PROPAGATING SATELLITE TRAJECTORIES")
    print_separator()

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
        f"Time samples: {len(times)}"
    )

    trajectories = {}

    total = len(satellites)

    for index, item in enumerate(
        satellites,
        start=1,
    ):

        name = item["name"]

        print(
            f"[{index}/{total}] "
            f"Propagating {name}"
        )

        try:

            geocentric = item["satellite"].at(times)

            positions = np.asarray(
                geocentric.position.km,
                dtype=np.float64,
            ).T

            velocities = np.asarray(
                geocentric.velocity.km_per_s,
                dtype=np.float64,
            ).T

            # Ignore non-finite propagation results.
            if not (
                np.isfinite(positions).all()
                and np.isfinite(velocities).all()
            ):
                print(
                    "   WARNING: non-finite propagation."
                )
                continue

            trajectories[name] = {
                "norad": item["norad"],
                "positions": positions,
                "velocities": velocities,
            }

        except Exception as exc:

            print(
                f"   WARNING: propagation failed: {exc}"
            )

    print()
    print(
        f"Successfully propagated "
        f"{len(trajectories)} satellites."
    )

    return trajectories


# ============================================================
# SPATIAL SCREENING
# ============================================================

def build_spatial_candidates(
    trajectories,
    screening_distance_km,
):
    """
    Find candidate satellite pairs using spatial grid cells.

    This prevents the program from performing expensive detailed
    calculations for every possible pair.

    The grid is rebuilt for every time sample.
    """

    print()
    print_separator()
    print("SPATIAL PAIR SCREENING")
    print_separator()

    names = list(trajectories.keys())

    if len(names) < 2:
        return []

    number_of_times = len(
        trajectories[names[0]]["positions"]
    )

    # Grid cell size.
    cell_size = screening_distance_km

    candidate_pairs = set()

    for time_index in range(number_of_times):

        if (
            time_index == 0
            or time_index % 12 == 0
        ):
            print(
                f"Screening time sample "
                f"{time_index + 1}/{number_of_times}..."
            )

        grid = {}

        # ----------------------------------------------------
        # Put each satellite into a spatial cell.
        # ----------------------------------------------------

        for name in names:

            position = trajectories[name][
                "positions"
            ][time_index]

            x, y, z = position

            cell = (
                math.floor(x / cell_size),
                math.floor(y / cell_size),
                math.floor(z / cell_size),
            )

            grid.setdefault(
                cell,
                [],
            ).append(name)

        # ----------------------------------------------------
        # Compare only neighboring cells.
        # ----------------------------------------------------

        for cell, cell_names in grid.items():

            cx, cy, cz = cell

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):

                        neighbor = (
                            cx + dx,
                            cy + dy,
                            cz + dz,
                        )

                        other_names = grid.get(
                            neighbor
                        )

                        if not other_names:
                            continue

                        for name1 in cell_names:

                            pos1 = trajectories[
                                name1
                            ]["positions"][
                                time_index
                            ]

                            for name2 in other_names:

                                if name1 >= name2:
                                    continue

                                pos2 = trajectories[
                                    name2
                                ]["positions"][
                                    time_index
                                ]

                                distance = np.linalg.norm(
                                    pos1 - pos2
                                )

                                if (
                                    distance
                                    <= screening_distance_km
                                ):

                                    candidate_pairs.add(
                                        (
                                            name1,
                                            name2,
                                        )
                                    )

    print()
    print(
        f"Candidate pairs after spatial screening: "
        f"{len(candidate_pairs)}"
    )

    return list(candidate_pairs)


# ============================================================
# RISK
# ============================================================

def classify_risk(distance_km):

    if distance_km < HIGH_RISK_KM:
        return "HIGH"

    if distance_km < MEDIUM_RISK_KM:
        return "MEDIUM"

    if distance_km < ANALYSIS_DISTANCE_KM:
        return "LOW"

    return "SAFE"


# ============================================================
# ANALYZE ONE PAIR
# ============================================================

def analyze_pair(
    name1,
    data1,
    name2,
    data2,
    datetimes,
):

    positions1 = data1["positions"]
    positions2 = data2["positions"]

    velocities1 = data1["velocities"]
    velocities2 = data2["velocities"]

    differences = (
        positions1
        - positions2
    )

    distances = np.linalg.norm(
        differences,
        axis=1,
    )

    minimum_index = int(
        np.argmin(distances)
    )

    minimum_distance = float(
        distances[minimum_index]
    )

    current_distance = float(
        distances[0]
    )

    current_relative_velocity = float(
        np.linalg.norm(
            velocities1[0]
            - velocities2[0]
        )
    )

    closest_relative_velocity = float(
        np.linalg.norm(
            velocities1[minimum_index]
            - velocities2[minimum_index]
        )
    )

    closest_time = datetimes[
        minimum_index
    ]

    return {
        "satellite_1": name1,
        "satellite_2": name2,

        "norad_1": data1["norad"],
        "norad_2": data2["norad"],

        "current_distance_km":
            current_distance,

        "minimum_distance_km":
            minimum_distance,

        "relative_velocity_at_closest_km_s":
            closest_relative_velocity,

        "current_relative_velocity_km_s":
            current_relative_velocity,

        "closest_approach_time":
            closest_time,

        "risk":
            classify_risk(
                minimum_distance
            ),
    }


# ============================================================
# ANALYZE CANDIDATE PAIRS
# ============================================================

def analyze_candidate_pairs(
    trajectories,
    candidate_pairs,
    datetimes,
):

    print()
    print_separator()
    print("DETAILED CLOSE-APPROACH ANALYSIS")
    print_separator()

    print()
    print(
        f"Candidate pairs to analyze: "
        f"{len(candidate_pairs)}"
    )

    results = []

    total = len(candidate_pairs)

    for index, (
        name1,
        name2,
    ) in enumerate(
        candidate_pairs,
        start=1,
    ):

        if (
            index == 1
            or index % 1000 == 0
            or index == total
        ):
            print(
                f"[{index}/{total}] "
                f"Analyzing pairs..."
            )

        result = analyze_pair(
            name1,
            trajectories[name1],
            name2,
            trajectories[name2],
            datetimes,
        )

        if (
            result["minimum_distance_km"]
            <= ANALYSIS_DISTANCE_KM
        ):

            results.append(result)

    results.sort(
        key=lambda item:
        item["minimum_distance_km"]
    )

    print()
    print(
        f"Pairs within "
        f"{ANALYSIS_DISTANCE_KM:.1f} km: "
        f"{len(results)}"
    )

    return results


# ============================================================
# SAVE DETAILED CSV
# ============================================================

def save_results_csv(results):

    fieldnames = [
        "rank",
        "satellite_1",
        "satellite_2",
        "norad_1",
        "norad_2",
        "current_distance_km",
        "minimum_distance_km",
        "relative_velocity_at_closest_km_s",
        "current_relative_velocity_km_s",
        "closest_approach_time_utc",
        "risk",
    ]

    with open(
        RESULTS_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for rank, result in enumerate(
            results,
            start=1,
        ):

            writer.writerow(
                {
                    "rank": rank,
                    "satellite_1":
                        result["satellite_1"],
                    "satellite_2":
                        result["satellite_2"],
                    "norad_1":
                        result["norad_1"],
                    "norad_2":
                        result["norad_2"],
                    "current_distance_km":
                        f'{result["current_distance_km"]:.3f}',
                    "minimum_distance_km":
                        f'{result["minimum_distance_km"]:.3f}',
                    "relative_velocity_at_closest_km_s":
                        f'{result["relative_velocity_at_closest_km_s"]:.3f}',
                    "current_relative_velocity_km_s":
                        f'{result["current_relative_velocity_km_s"]:.3f}',
                    "closest_approach_time_utc":
                        result[
                            "closest_approach_time"
                        ].isoformat(),
                    "risk":
                        result["risk"],
                }
            )

    print()
    print(
        f"Detailed results saved to:"
    )

    print(
        os.path.abspath(
            RESULTS_CSV
        )
    )


# ============================================================
# SUMMARY CSV
# ============================================================

def save_summary_csv(
    results,
    total_satellites,
    total_pairs,
):

    counts = {
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0,
    }

    for result in results:

        risk = result["risk"]

        if risk in counts:
            counts[risk] += 1

    safe = (
        total_pairs
        - len(results)
    )

    rows = [
        (
            "Satellites monitored",
            total_satellites,
        ),
        (
            "Total possible satellite pairs",
            total_pairs,
        ),
        (
            "Pairs within analysis distance",
            len(results),
        ),
        (
            "High risk",
            counts["HIGH"],
        ),
        (
            "Medium risk",
            counts["MEDIUM"],
        ),
        (
            "Low risk",
            counts["LOW"],
        ),
        (
            "Outside analysis distance",
            safe,
        ),
        (
            "Prediction window hours",
            PREDICTION_HOURS,
        ),
        (
            "Prediction step minutes",
            STEP_MINUTES,
        ),
        (
            "Screening distance km",
            SCREENING_DISTANCE_KM,
        ),
        (
            "Analysis distance km",
            ANALYSIS_DISTANCE_KM,
        ),
    ]

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["Metric", "Value"]
        )

        writer.writerows(rows)

    print()
    print(
        f"Summary saved to:"
    )

    print(
        os.path.abspath(
            SUMMARY_CSV
        )
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):

    print()
    print_separator()
    print("CLOSE APPROACH RESULTS")
    print_separator()

    if not results:

        print()
        print(
            "No satellite pairs were predicted "
            f"within {ANALYSIS_DISTANCE_KM:.1f} km."
        )

        return

    display_count = min(
        20,
        len(results),
    )

    for index in range(
        display_count
    ):

        result = results[index]

        print()
        print(
            f"{index + 1}. "
            f'{result["satellite_1"]} '
            f"<-> "
            f'{result["satellite_2"]}'
        )

        print(
            f'   NORAD: '
            f'{result["norad_1"]} '
            f'<-> '
            f'{result["norad_2"]}'
        )

        print(
            f'   Current distance: '
            f'{result["current_distance_km"]:.3f} km'
        )

        print(
            f'   Minimum predicted distance: '
            f'{result["minimum_distance_km"]:.3f} km'
        )

        print(
            f'   Relative velocity: '
            f'{result["relative_velocity_at_closest_km_s"]:.3f} km/s'
        )

        print(
            f'   Closest approach time: '
            f'{result["closest_approach_time"].isoformat()}'
        )

        print(
            f'   Risk: '
            f'{result["risk"]}'
        )

    if len(results) > display_count:

        print()
        print(
            f"... {len(results) - display_count} "
            f"additional results are in "
            f"{RESULTS_CSV}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_summary(
    results,
    total_satellites,
    total_pairs,
):

    counts = {
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0,
    }

    for result in results:

        if result["risk"] in counts:
            counts[result["risk"]] += 1

    print()
    print_separator()
    print("FINAL SUMMARY")
    print_separator()

    print(
        f"Satellites monitored: "
        f"{total_satellites}"
    )

    print(
        f"Total possible pairs: "
        f"{total_pairs}"
    )

    print(
        f"Pairs within "
        f"{ANALYSIS_DISTANCE_KM:.1f} km: "
        f"{len(results)}"
    )

    print()

    print(
        f"HIGH:   {counts['HIGH']}"
    )

    print(
        f"MEDIUM: {counts['MEDIUM']}"
    )

    print(
        f"LOW:    {counts['LOW']}"
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
    print(
        "IMPORTANT: These are TLE/OMM-based screening "
        "predictions, not confirmed collision probabilities."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_separator()
    print("SPACE COLLISION WARNING SYSTEM")
    print_separator()

    print()

    # --------------------------------------------------------
    # 1. DOWNLOAD DATA
    # --------------------------------------------------------

    satellite_data = (
        download_satellite_data()
    )

    if len(satellite_data) < 2:

        print()
        print(
            "Program stopped because usable "
            "satellite data was not available."
        )

        return

    print()
    print(
        f"Satellites available: "
        f"{len(satellite_data)}"
    )

    # --------------------------------------------------------
    # 2. START TIME
    # --------------------------------------------------------

    start_time = datetime.now(
        timezone.utc
    )

    print()
    print(
        f"Prediction start time: "
        f"{start_time.isoformat()}"
    )

    # --------------------------------------------------------
    # 3. SKYFIELD
    # --------------------------------------------------------

    print()
    print(
        "Preparing Skyfield timescale..."
    )

    ts = load.timescale()

    # --------------------------------------------------------
    # 4. ORBITAL MODELS
    # --------------------------------------------------------

    satellites = (
        create_satellite_models(
            satellite_data,
            ts,
        )
    )

    if len(satellites) < 2:

        print()
        print(
            "ERROR: Not enough satellites "
            "for analysis."
        )

        return

    # --------------------------------------------------------
    # 5. TIME ARRAY
    # --------------------------------------------------------

    times, datetimes = (
        create_prediction_times(
            ts,
            start_time,
        )
    )

    # --------------------------------------------------------
    # 6. PROPAGATION
    # --------------------------------------------------------

    trajectories = (
        propagate_satellites(
            satellites,
            times,
        )
    )

    if len(trajectories) < 2:

        print()
        print(
            "ERROR: Not enough satellites "
            "were successfully propagated."
        )

        return

    # --------------------------------------------------------
    # 7. TOTAL POSSIBLE PAIRS
    # --------------------------------------------------------

    total_satellites = len(
        trajectories
    )

    total_pairs = (
        total_satellites
        * (total_satellites - 1)
        // 2
    )

    print()
    print(
        f"Total possible satellite pairs: "
        f"{total_pairs:,}"
    )

    # --------------------------------------------------------
    # 8. SPATIAL SCREENING
    # --------------------------------------------------------

    candidate_pairs = (
        build_spatial_candidates(
            trajectories,
            SCREENING_DISTANCE_KM,
        )
    )

    # --------------------------------------------------------
    # 9. DETAILED ANALYSIS
    # --------------------------------------------------------

    results = (
        analyze_candidate_pairs(
            trajectories,
            candidate_pairs,
            datetimes,
        )
    )

    # --------------------------------------------------------
    # 10. RESULTS
    # --------------------------------------------------------

    print_results(
        results
    )

    # --------------------------------------------------------
    # 11. SAVE CSV
    # --------------------------------------------------------

    save_results_csv(
        results
    )

    save_summary_csv(
        results,
        total_satellites,
        total_pairs,
    )

    # --------------------------------------------------------
    # 12. SUMMARY
    # --------------------------------------------------------

    print_summary(
        results,
        total_satellites,
        total_pairs,
    )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print()
    print_separator()
    print("ANALYSIS COMPLETE")
    print_separator()

    print()
    print(
        f"Detailed report: "
        f"{RESULTS_CSV}"
    )

    print(
        f"Summary report:  "
        f"{SUMMARY_CSV}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()