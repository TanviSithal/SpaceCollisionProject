# satellite_data.py

import requests
import numpy as np

from datetime import datetime, timezone

from skyfield.api import load, EarthSatellite


# ============================================================
# CELESTRAK
# ============================================================

CELESTRAK_URL = (
    "https://celestrak.org/NORAD/elements/"
    "gp.php?GROUP=last-30-days&FORMAT=json"
)

USER_AGENT = "SpaceCollisionProject/1.0"


# ============================================================
# CACHE
# ============================================================

_satellite_cache = None
_satellite_index = None
_timescale = None
_orbital_model_cache = {}


# ============================================================
# GET SKYFIELD TIMESCALE
# ============================================================

def get_timescale():
    """
    Return one shared Skyfield timescale object.
    """

    global _timescale

    if _timescale is None:
        _timescale = load.timescale()

    return _timescale


# ============================================================
# DOWNLOAD SATELLITE DATA
# ============================================================

def get_satellite_list():
    """
    Download satellite OMM data from CelesTrak.

    The downloaded data is cached in memory so that repeated
    calls do not download the data again.
    """

    global _satellite_cache

    if _satellite_cache is not None:
        return _satellite_cache

    print("Downloading CelesTrak data...")

    headers = {
        "User-Agent": USER_AGENT
    }

    try:
        response = requests.get(
            CELESTRAK_URL,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, list):
            raise ValueError(
                "CelesTrak returned unexpected data format."
            )

        _satellite_cache = data

        print(
            f"Downloaded {len(_satellite_cache)} satellites."
        )

        return _satellite_cache

    except requests.exceptions.RequestException as e:

        print(
            "ERROR: Could not download CelesTrak data."
        )
        print(e)
        raise

    except ValueError as e:

        print(
            "ERROR: Invalid satellite data received "
            "from CelesTrak."
        )
        print(e)
        raise


# ============================================================
# PREPARE SATELLITE INDEX
# ============================================================

def _build_satellite_index():
    """
    Convert the raw CelesTrak list into a dictionary indexed
    by NORAD catalog ID.

    This makes satellite lookup much faster than repeatedly
    scanning the complete list.
    """

    global _satellite_index

    if _satellite_index is not None:
        return _satellite_index

    data = get_satellite_list()

    index = {}

    for sat in data:

        if (
            "NORAD_CAT_ID" not in sat
            or "OBJECT_NAME" not in sat
        ):
            continue

        try:

            norad_id = int(sat["NORAD_CAT_ID"])

        except (ValueError, TypeError):

            continue

        index[norad_id] = {
            "name": str(sat["OBJECT_NAME"]),
            "norad_id": norad_id,
            "data": sat
        }

    _satellite_index = index

    return _satellite_index


# ============================================================
# GET AVAILABLE SATELLITES
# ============================================================

def get_available_satellites():
    """
    Return all satellites available in the CelesTrak dataset.
    """

    index = _build_satellite_index()

    return list(index.values())


# ============================================================
# FIND SATELLITE
# ============================================================

def find_satellite(norad_id):
    """
    Find a satellite using its NORAD catalog ID.
    """

    index = _build_satellite_index()

    try:
        norad_id = int(norad_id)
    except (ValueError, TypeError):

        return None

    return index.get(norad_id)


# ============================================================
# CREATE SKYFIELD SATELLITE
# ============================================================

def create_satellite(sat_data):
    """
    Convert CelesTrak OMM data into a Skyfield EarthSatellite.

    The resulting orbital model is cached by NORAD ID.
    """

    norad_id = int(sat_data["norad_id"])

    if norad_id in _orbital_model_cache:
        return _orbital_model_cache[norad_id]

    d = sat_data["data"]

    ts = get_timescale()

    satellite = EarthSatellite.from_omm(
        ts,
        d
    )

    _orbital_model_cache[norad_id] = satellite

    return satellite


# ============================================================
# CREATE ALL ORBITAL MODELS
# ============================================================

def create_all_satellites():
    """
    Create Skyfield orbital models for every available
    satellite.

    Returns:
        list of dictionaries containing:
            name
            norad_id
            data
            satellite
    """

    satellites = get_available_satellites()

    orbital_satellites = []

    print()
    print("Creating satellite orbital models...")
    print("-" * 40)

    for index, sat in enumerate(satellites, start=1):

        try:

            satellite = create_satellite(sat)

            orbital_satellites.append({
                "name": sat["name"],
                "norad_id": sat["norad_id"],
                "data": sat["data"],
                "satellite": satellite
            })

            print(
                f"[{index}/{len(satellites)}] "
                f"{sat['norad_id']} - {sat['name']}"
            )

        except Exception as e:

            print(
                f"[{index}/{len(satellites)}] "
                f"{sat['norad_id']} - ERROR: {e}"
            )

    return orbital_satellites


# ============================================================
# GET CURRENT SATELLITE STATE
# ============================================================

def get_current_state(norad_id, now=None):
    """
    Calculate the current state of a satellite.

    Returns:

        name
        norad_id
        position       km
        velocity       km/s
        latitude       degrees
        longitude      degrees
        altitude       km
        timestamp      UTC datetime
    """

    sat_data = find_satellite(norad_id)

    if sat_data is None:

        raise ValueError(
            f"Satellite {norad_id} was not found "
            "in the CelesTrak dataset."
        )

    satellite = create_satellite(sat_data)

    ts = get_timescale()

    # --------------------------------------------------------
    # Use supplied time if provided.
    # This is important because all satellites should use
    # exactly the same timestamp during collision analysis.
    # --------------------------------------------------------

    if now is None:

        now = datetime.now(timezone.utc)

    else:

        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        else:
            now = now.astimezone(timezone.utc)

    t = ts.from_datetime(now)

    # --------------------------------------------------------
    # Propagate orbit
    # --------------------------------------------------------

    geocentric = satellite.at(t)

    # Position in km
    position = np.asarray(
        geocentric.position.km,
        dtype=float
    )

    # Velocity in km/s
    velocity = np.asarray(
        geocentric.velocity.km_per_s,
        dtype=float
    )

    # --------------------------------------------------------
    # Geographic information
    # --------------------------------------------------------

    subpoint = geocentric.subpoint()

    latitude = float(
        subpoint.latitude.degrees
    )

    longitude = float(
        subpoint.longitude.degrees
    )

    altitude = float(
        subpoint.elevation.km
    )

    return {
        "name": sat_data["name"],

        "norad_id": sat_data["norad_id"],

        "position": position,

        "velocity": velocity,

        "latitude": latitude,

        "longitude": longitude,

        "altitude": altitude,

        "timestamp": now
    }


# ============================================================
# GET MULTIPLE CURRENT STATES
# ============================================================

def get_current_states(satellites=None, now=None):
    """
    Get current states for multiple satellites.

    All satellites are evaluated at exactly the same UTC time.

    Returns:
        list of state dictionaries
    """

    if satellites is None:
        satellites = get_available_satellites()

    if now is None:
        now = datetime.now(timezone.utc)

    else:
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        else:
            now = now.astimezone(timezone.utc)

    states = []

    for index, satellite in enumerate(
        satellites,
        start=1
    ):

        norad_id = satellite["norad_id"]

        try:

            state = get_current_state(
                norad_id,
                now=now
            )

            states.append(state)

            print(
                f"[{index}/{len(satellites)}] "
                f"{norad_id} - {state['name']}"
            )

        except Exception as e:

            print(
                f"[{index}/{len(satellites)}] "
                f"{norad_id} - ERROR: {e}"
            )

    return states


# ============================================================
# CALCULATE DISTANCE
# ============================================================

def calculate_distance(
    position_a,
    position_b
):
    """
    Calculate 3D distance between two positions.

    Input:
        positions in km

    Output:
        distance in km
    """

    position_a = np.asarray(
        position_a,
        dtype=float
    )

    position_b = np.asarray(
        position_b,
        dtype=float
    )

    return float(
        np.linalg.norm(
            position_b - position_a
        )
    )


# ============================================================
# CALCULATE SPEED
# ============================================================

def calculate_speed(velocity):
    """
    Calculate speed from velocity vector.

    Input:
        velocity in km/s

    Output:
        speed in km/s
    """

    velocity = np.asarray(
        velocity,
        dtype=float
    )

    return float(
        np.linalg.norm(velocity)
    )


# ============================================================
# CALCULATE RELATIVE VELOCITY
# ============================================================

def calculate_relative_velocity(
    velocity_a,
    velocity_b
):
    """
    Calculate relative velocity vector.

    Input:
        velocity in km/s

    Output:
        relative velocity vector in km/s
    """

    velocity_a = np.asarray(
        velocity_a,
        dtype=float
    )

    velocity_b = np.asarray(
        velocity_b,
        dtype=float
    )

    return velocity_b - velocity_a


# ============================================================
# RELATIVE SPEED
# ============================================================

def calculate_relative_speed(
    velocity_a,
    velocity_b
):
    """
    Calculate magnitude of relative velocity.

    Output:
        km/s
    """

    relative_velocity = (
        calculate_relative_velocity(
            velocity_a,
            velocity_b
        )
    )

    return float(
        np.linalg.norm(relative_velocity)
    )


# ============================================================
# CLEAR CACHE
# ============================================================

def clear_cache():
    """
    Clear all downloaded and orbital-model caches.

    Useful if you want to force a fresh CelesTrak download.
    """

    global _satellite_cache
    global _satellite_index
    global _orbital_model_cache

    _satellite_cache = None
    _satellite_index = None
    _orbital_model_cache = {}


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SATELLITE DATA MODULE TEST")
    print("=" * 70)

    satellites = get_available_satellites()

    print()
    print(
        f"Satellites available: "
        f"{len(satellites)}"
    )

    if satellites:

        first_satellite = satellites[0]

        print()
        print(
            f"Testing satellite: "
            f"{first_satellite['name']}"
        )

        state = get_current_state(
            first_satellite["norad_id"]
        )

        print()
        print("Current state:")
        print(
            f"Name:       {state['name']}"
        )
        print(
            f"NORAD ID:   {state['norad_id']}"
        )
        print(
            f"Position:   {state['position']} km"
        )
        print(
            f"Velocity:   {state['velocity']} km/s"
        )
        print(
            f"Latitude:   {state['latitude']:.4f}°"
        )
        print(
            f"Longitude:  {state['longitude']:.4f}°"
        )
        print(
            f"Altitude:   {state['altitude']:.3f} km"
        )
        print(
            f"Timestamp:  {state['timestamp']}"
        )

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)