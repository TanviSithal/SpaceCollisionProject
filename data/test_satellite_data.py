import json

# Location of downloaded CelesTrak data
DATA_FILE = "data/satellites.json"


# Open the JSON file
with open(DATA_FILE, "r", encoding="utf-8") as file:
    satellites = json.load(file)


# Check that data was loaded
print("=" * 50)
print("CELESTRAK DATA TEST")
print("=" * 50)

print("Satellite data loaded successfully!")
print("Number of satellites:", len(satellites))


# Display information about the first satellite
if len(satellites) > 0:

    satellite = satellites[0]

    print()
    print("First satellite:")
    print("Name:", satellite.get("OBJECT_NAME"))
    print("NORAD ID:", satellite.get("NORAD_CAT_ID"))
    print("Epoch:", satellite.get("EPOCH"))
    print("Mean Motion:", satellite.get("MEAN_MOTION"))
    print("Eccentricity:", satellite.get("ECCENTRICITY"))
    print("Inclination:", satellite.get("INCLINATION"))

else:
    print("No satellite records found.")

print("=" * 50)