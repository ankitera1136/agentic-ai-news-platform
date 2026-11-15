"""Deterministic place-name normalization and a static coordinate table. No network, no LLM.

Rules:
- Known aliases fold to one canonical name ("Bangalore" -> "Bengaluru", "USA" -> "United States").
- Unknown names keep their cleaned original spelling.
- Coordinates come only from the tables below. A city that is not in CITY_COORDS gets NO
  coordinates (it is reported as unmapped). Coordinates are never guessed or interpolated.
- When only a country is stated, the country centroid is used and the precision is "country".
"""
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from backend.services.profile.text import clean_text

# ---------------------------------------------------------------------------
# Countries
# ---------------------------------------------------------------------------

# canonical name -> aliases (matched case-insensitively, punctuation and extra spaces ignored)
_COUNTRY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "India": ("india", "in", "ind", "bharat", "republic of india"),
    "United States": (
        "united states", "united states of america", "us", "usa", "u.s.", "u.s.a.", "america",
    ),
    "United Kingdom": ("united kingdom", "uk", "u.k.", "gb", "great britain", "britain"),
    "United Arab Emirates": ("united arab emirates", "uae", "u.a.e.", "emirates"),
    "Singapore": ("singapore",),
    "Germany": ("germany", "deutschland"),
    "Canada": ("canada",),
    "Australia": ("australia",),
    "France": ("france",),
    "Netherlands": ("netherlands", "the netherlands", "holland"),
    "Ireland": ("ireland", "republic of ireland"),
    "Israel": ("israel",),
    "Japan": ("japan",),
    "China": ("china", "prc", "peoples republic of china", "people's republic of china"),
    "South Korea": ("south korea", "korea", "republic of korea"),
    "Malaysia": ("malaysia",),
    "Indonesia": ("indonesia",),
    "Brazil": ("brazil", "brasil"),
    "Sweden": ("sweden",),
    "Switzerland": ("switzerland",),
    "Spain": ("spain",),
    "Italy": ("italy",),
    "Poland": ("poland",),
    "Saudi Arabia": ("saudi arabia", "ksa"),
    "Taiwan": ("taiwan",),
    "Vietnam": ("vietnam", "viet nam"),
    "Philippines": ("philippines", "the philippines"),
    "Thailand": ("thailand",),
    "Mexico": ("mexico",),
    "South Africa": ("south africa",),
    "Nigeria": ("nigeria",),
    "Kenya": ("kenya",),
    "Egypt": ("egypt",),
    "Pakistan": ("pakistan",),
    "Bangladesh": ("bangladesh",),
    "Sri Lanka": ("sri lanka",),
    "Hong Kong": ("hong kong",),
}

# Country centroids (lat, lng). Used only when a signal states a country but no city.
COUNTRY_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "India": (20.5937, 78.9629),
    "United States": (39.8283, -98.5795),
    "United Kingdom": (54.0, -2.0),
    "United Arab Emirates": (23.4241, 53.8478),
    "Singapore": (1.3521, 103.8198),
    "Germany": (51.1657, 10.4515),
    "Canada": (56.1304, -106.3468),
    "Australia": (-25.2744, 133.7751),
    "France": (46.2276, 2.2137),
    "Netherlands": (52.1326, 5.2913),
    "Ireland": (53.1424, -7.6921),
    "Israel": (31.0461, 34.8516),
    "Japan": (36.2048, 138.2529),
    "China": (35.8617, 104.1954),
    "South Korea": (35.9078, 127.7669),
    "Malaysia": (4.2105, 101.9758),
    "Indonesia": (-0.7893, 113.9213),
    "Brazil": (-14.2350, -51.9253),
    "Sweden": (60.1282, 18.6435),
    "Switzerland": (46.8182, 8.2275),
    "Spain": (40.4637, -3.7492),
    "Italy": (41.8719, 12.5674),
    "Poland": (51.9194, 19.1451),
    "Saudi Arabia": (23.8859, 45.0792),
    "Taiwan": (23.6978, 120.9605),
    "Vietnam": (14.0583, 108.2772),
    "Philippines": (12.8797, 121.7740),
    "Thailand": (15.8700, 100.9925),
    "Mexico": (23.6345, -102.5528),
    "South Africa": (-30.5595, 22.9375),
    "Nigeria": (9.0820, 8.6753),
    "Kenya": (-0.0236, 37.9062),
    "Egypt": (26.8206, 30.8025),
    "Pakistan": (30.3753, 69.3451),
    "Bangladesh": (23.6850, 90.3563),
    "Sri Lanka": (7.8731, 80.7718),
    "Hong Kong": (22.3193, 114.1694),
}

# ---------------------------------------------------------------------------
# Cities: (canonical name, country, lat, lng)
# ---------------------------------------------------------------------------

_CITIES = (
    # India
    ("Bengaluru", "India", 12.9716, 77.5946),
    ("Mumbai", "India", 19.0760, 72.8777),
    ("Delhi", "India", 28.6139, 77.2090),
    ("Hyderabad", "India", 17.3850, 78.4867),
    ("Chennai", "India", 13.0827, 80.2707),
    ("Pune", "India", 18.5204, 73.8567),
    ("Kolkata", "India", 22.5726, 88.3639),
    ("Gurugram", "India", 28.4595, 77.0266),
    ("Noida", "India", 28.5355, 77.3910),
    ("Ahmedabad", "India", 23.0225, 72.5714),
    ("Jaipur", "India", 26.9124, 75.7873),
    ("Chandigarh", "India", 30.7333, 76.7794),
    ("Kochi", "India", 9.9312, 76.2673),
    ("Thiruvananthapuram", "India", 8.5241, 76.9366),
    ("Coimbatore", "India", 11.0168, 76.9558),
    ("Indore", "India", 22.7196, 75.8577),
    ("Lucknow", "India", 26.8467, 80.9462),
    ("Nagpur", "India", 21.1458, 79.0882),
    ("Bhubaneswar", "India", 20.2961, 85.8245),
    ("Visakhapatnam", "India", 17.6868, 83.2185),
    ("Mysuru", "India", 12.2958, 76.6394),
    ("Vadodara", "India", 22.3072, 73.1812),
    ("Surat", "India", 21.1702, 72.8311),
    ("Mohali", "India", 30.7046, 76.7179),
    ("Faridabad", "India", 28.4089, 77.3178),
    ("Ghaziabad", "India", 28.6692, 77.4538),
    ("Greater Noida", "India", 28.4744, 77.5040),
    ("Navi Mumbai", "India", 19.0330, 73.0297),
    ("Thane", "India", 19.2183, 72.9781),
    ("Nashik", "India", 19.9975, 73.7898),
    ("Kanpur", "India", 26.4499, 80.3319),
    ("Patna", "India", 25.5941, 85.1376),
    ("Ranchi", "India", 23.3441, 85.3096),
    ("Guwahati", "India", 26.1445, 91.7362),
    ("Dehradun", "India", 30.3165, 78.0322),
    ("Vijayawada", "India", 16.5062, 80.6480),
    ("Madurai", "India", 9.9252, 78.1198),
    ("Mangaluru", "India", 12.9141, 74.8560),
    ("Hubballi", "India", 15.3647, 75.1240),
    ("Tiruchirappalli", "India", 10.7905, 78.7047),
    ("Bhopal", "India", 23.2599, 77.4126),
    ("Raipur", "India", 21.2514, 81.6296),
    ("Jodhpur", "India", 26.2389, 73.0243),
    ("Udaipur", "India", 24.5854, 73.7125),
    ("Panaji", "India", 15.4909, 73.8278),
    # United States
    ("San Francisco", "United States", 37.7749, -122.4194),
    ("Seattle", "United States", 47.6062, -122.3321),
    ("Austin", "United States", 30.2672, -97.7431),
    ("New York", "United States", 40.7128, -74.0060),
    ("Boston", "United States", 42.3601, -71.0589),
    ("Los Angeles", "United States", 34.0522, -118.2437),
    ("San Diego", "United States", 32.7157, -117.1611),
    ("San Jose", "United States", 37.3382, -121.8863),
    ("Mountain View", "United States", 37.3861, -122.0839),
    ("Palo Alto", "United States", 37.4419, -122.1430),
    ("Sunnyvale", "United States", 37.3688, -122.0363),
    ("Redmond", "United States", 47.6740, -122.1215),
    ("Chicago", "United States", 41.8781, -87.6298),
    ("Denver", "United States", 39.7392, -104.9903),
    ("Atlanta", "United States", 33.7490, -84.3880),
    # Canada
    ("Toronto", "Canada", 43.6532, -79.3832),
    ("Vancouver", "Canada", 49.2827, -123.1207),
    ("Montreal", "Canada", 45.5017, -73.5673),
    # Europe
    ("London", "United Kingdom", 51.5074, -0.1278),
    ("Dublin", "Ireland", 53.3498, -6.2603),
    ("Paris", "France", 48.8566, 2.3522),
    ("Berlin", "Germany", 52.5200, 13.4050),
    ("Munich", "Germany", 48.1351, 11.5820),
    ("Amsterdam", "Netherlands", 52.3676, 4.9041),
    ("Stockholm", "Sweden", 59.3293, 18.0686),
    ("Zurich", "Switzerland", 47.3769, 8.5417),
    # Middle East, Asia-Pacific, Latin America
    ("Tel Aviv", "Israel", 32.0853, 34.7818),
    ("Dubai", "United Arab Emirates", 25.2048, 55.2708),
    ("Abu Dhabi", "United Arab Emirates", 24.4539, 54.3773),
    ("Singapore", "Singapore", 1.3521, 103.8198),
    ("Tokyo", "Japan", 35.6762, 139.6503),
    ("Seoul", "South Korea", 37.5665, 126.9780),
    ("Shenzhen", "China", 22.5431, 114.0579),
    ("Beijing", "China", 39.9042, 116.4074),
    ("Shanghai", "China", 31.2304, 121.4737),
    ("Hong Kong", "Hong Kong", 22.3193, 114.1694),
    ("Sydney", "Australia", -33.8688, 151.2093),
    ("Melbourne", "Australia", -37.8136, 144.9631),
    ("Kuala Lumpur", "Malaysia", 3.1390, 101.6869),
    ("Sao Paulo", "Brazil", -23.5558, -46.6396),
)

CITY_COORDS: Dict[str, Tuple[float, float]] = {name: (lat, lng) for name, _, lat, lng in _CITIES}
CITY_COUNTRY: Dict[str, str] = {name: country for name, country, _, _ in _CITIES}

# canonical city -> extra spellings (the canonical name itself is always accepted)
_CITY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Bengaluru": ("bangalore", "bengalooru", "bangaluru"),
    "Gurugram": ("gurgaon",),
    "Mumbai": ("bombay",),
    "Chennai": ("madras",),
    "Kolkata": ("calcutta",),
    "Delhi": ("new delhi", "delhi ncr", "ncr", "national capital region", "nct of delhi"),
    "Pune": ("poona",),
    "Thiruvananthapuram": ("trivandrum",),
    "Kochi": ("cochin", "ernakulam"),
    "Visakhapatnam": ("vizag", "vishakhapatnam"),
    "Mysuru": ("mysore",),
    "Vadodara": ("baroda",),
    "Mangaluru": ("mangalore",),
    "Hubballi": ("hubli",),
    "Tiruchirappalli": ("trichy", "tiruchirapalli"),
    "Panaji": ("panjim",),
    "Noida": ("gautam buddha nagar",),
    "San Francisco": ("sf", "san francisco bay", "san fransisco"),
    "New York": ("new york city", "nyc", "new york, ny", "manhattan"),
    "Los Angeles": ("la",),
    "Sao Paulo": ("s\u00e3o paulo", "sao paolo"),
    "Zurich": ("z\u00fcrich",),
    "Munich": ("m\u00fcnchen",),
    "Tel Aviv": ("tel aviv-yafo", "tel-aviv"),
    "Hong Kong": ("hongkong",),
    "Mountain View": ("mountain view, ca",),
    "Montreal": ("montr\u00e9al",),
    "Kuala Lumpur": ("kl",),
}


def _key(value: str) -> str:
    text = value.casefold().replace("&", " and ")
    text = re.sub(r"[.']", "", text)           # "u.s." -> "us", "people's" -> "peoples"
    text = re.sub(r"[^\w\s,-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


_COUNTRY_LOOKUP: Dict[str, str] = {}
for _canonical, _aliases in _COUNTRY_ALIASES.items():
    _COUNTRY_LOOKUP[_key(_canonical)] = _canonical
    for _alias in _aliases:
        _COUNTRY_LOOKUP[_key(_alias)] = _canonical

_CITY_LOOKUP: Dict[str, str] = {}
for _canonical in CITY_COORDS:
    _CITY_LOOKUP[_key(_canonical)] = _canonical
for _canonical, _aliases in _CITY_ALIASES.items():
    for _alias in _aliases:
        _CITY_LOOKUP[_key(_alias)] = _canonical


def normalize_country(raw: Optional[str]) -> Optional[str]:
    """Canonical country name for known aliases, else the cleaned original. None when empty."""
    if raw is None or not isinstance(raw, str):
        return None
    text = clean_text(raw)
    if not text:
        return None
    return _COUNTRY_LOOKUP.get(_key(text), text)


def normalize_city(raw: Optional[str]) -> Optional[str]:
    """Canonical city name for known aliases, else the cleaned original. None when empty."""
    if raw is None or not isinstance(raw, str):
        return None
    text = clean_text(raw)
    if not text:
        return None
    known = _CITY_LOOKUP.get(_key(text))
    if known:
        return known
    # "Bengaluru, Karnataka" -> try the part before the first comma.
    head = text.split(",")[0].strip()
    if head and head != text:
        known = _CITY_LOOKUP.get(_key(head))
        if known:
            return known
    return text


@dataclass(frozen=True)
class Location:
    """A resolved place for one signal."""
    city: Optional[str]
    country: Optional[str]
    lat: Optional[float]
    lng: Optional[float]
    precision: Optional[str]          # "city" | "country" | None (cannot be plotted)
    stated: bool                      # the signal named a city or a country
    country_inferred: bool = False    # country came from the city table, not from the signal

    @property
    def plottable(self) -> bool:
        return self.lat is not None and self.lng is not None


def resolve_location(city_raw: Optional[str], country_raw: Optional[str]) -> Location:
    """Normalize a signal's stated place and attach coordinates only from the static tables."""
    city = normalize_city(city_raw)
    country = normalize_country(country_raw)
    stated = bool(city or country)

    if city:
        coords = CITY_COORDS.get(city)
        table_country = CITY_COUNTRY.get(city)
        if coords is None:
            return Location(city, country, None, None, None, stated)
        if country and country != table_country:
            # The signal states a different country for a name we know (e.g. "Hyderabad, Pakistan").
            # It may be another place with the same name, so do not plot it.
            return Location(city, country, None, None, None, stated)
        return Location(
            city, country or table_country, coords[0], coords[1], "city", stated,
            country_inferred=country is None,
        )

    if country:
        centroid = COUNTRY_CENTROIDS.get(country)
        if centroid is None:
            return Location(None, country, None, None, None, stated)
        return Location(None, country, centroid[0], centroid[1], "country", stated)

    return Location(None, None, None, None, None, False)
