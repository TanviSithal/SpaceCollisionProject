import itertools

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

def load_satellite_states():

    satellites = get_available_satellites()

    print(
        f"Found {len(satellites)} satellites."
    )

    states = []

    print()
    print("Getting satellite states...")
    print("---------------------------")

    for index, satellite in enumerate(satellites, start=1):

        try:

            state = get_current_state(
                satellite["norad_id"]
            )

            states.append(state)

            print(
                f"[{index}/{len(satellites)}] "
                f"{state['norad_id']} - "
                f"{state['name']}"
            )

        except Exception as e:

            print(
                f"Skipping "
                f"{satellite['norad_id']} "
                f"because of error: {e}"
            )

    return states


# ============================================================
# FIND CLOSEST PAIRS
# ============================================================

def find_closest_pairs(states):

    results = []

    total_pairs = (
        len(states) * (len(states) - 1) // 2
    )

    print()
    print(
        f"Analyzing {total_pairs} satellite pairs..."
    )

    for satellite_a, satellite_b in itertools.combinations(
        states,
        2
    ):

        try:

            result = analyze_pair(
                satellite_a,
                satellite_b
            )

            results.append(result)

        except Exception as e:

            print(
                "Could not analyze pair:",
                e
            )

    # Sort by current distance
    results.sort(
        key=lambda x: x["distance_km"]
    )

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    results,
    number_of_results=10
):

    print()
    print("=" * 70)
    print("CLOSEST SATELLITE PAIRS")
    print("=" * 70)

    for index, result in enumerate(
        results[:number_of_results],
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
            f"   Distance: "
            f"{result['distance_km']:.3f} km"
        )

        print(
            f"   Relative velocity: "
            f"{result['relative_velocity_km_s']:.3f} km/s"
        )

        print(
            f"   Altitude difference: "
            f"{result['altitude_difference_km']:.3f} km"
        )

        print(
            f"   Risk: "
            f"{result['risk']}"
        )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    states = load_satellite_states()

    if len(states) < 2:

        print(
            "Not enough satellites "
            "to perform collision analysis."
        )

        raise SystemExit

    results = find_closest_pairs(
        states
    )

    display_results(
        results,
        number_of_results=10
    )