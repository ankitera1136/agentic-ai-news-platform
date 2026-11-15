"""Deterministic company-name canonicalization for CareerRadar.

The LLM returns the company name as written. This module maps known aliases to one
canonical display name and strips legal entity suffixes from unknown names. It never
uses an LLM, so the result is testable and auditable.
"""

import html
import re
from dataclasses import dataclass
from typing import Optional

# Legal entity suffixes only. Business descriptors such as "Technologies",
# "Solutions", and "Services" are intentionally preserved.
_LEGAL_SUFFIXES = [
    "private limited",
    "pvt ltd",
    "pvt. ltd.",
    "pvt. ltd",
    "pvt limited",
    "public limited",
    "limited",
    "incorporated",
    "corporation",
    "company",
    "holdings",
    "holding",
    "gmbh",
    "plc",
    "llp",
    "llc",
    "ltd",
    "inc",
    "corp",
    "co",
    "pvt",
    "private",
    "pte",
    "ag",
    "sa",
    "bv",
    "nv",
]

# Canonical display names. Keys are normalized with _key().
_ALIASES = {
    # Indian IT services
    "tcs": "Tata Consultancy Services",
    "tata consultancy services": "Tata Consultancy Services",
    "tata consultancy service": "Tata Consultancy Services",
    "infosys": "Infosys",
    "infosys technologies": "Infosys",
    "infosys bpm": "Infosys BPM",
    "wipro": "Wipro",
    "wipro technologies": "Wipro",
    "hcl": "HCL Technologies",
    "hcl technologies": "HCL Technologies",
    "hcltech": "HCL Technologies",
    "accenture": "Accenture",
    "cognizant": "Cognizant",
    "cognizant technology solutions": "Cognizant",
    "tech mahindra": "Tech Mahindra",
    "techmahindra": "Tech Mahindra",
    "ibm": "IBM",
    "international business machines": "IBM",
    "capgemini": "Capgemini",
    "ltimindtree": "LTIMindtree",
    "l&t infotech": "LTIMindtree",
    "l and t infotech": "LTIMindtree",
    "mindtree": "LTIMindtree",
    "mphasis": "Mphasis",
    "persistent systems": "Persistent Systems",
    "zensar": "Zensar",
    "hexaware": "Hexaware",
    "coforge": "Coforge",
    "niit technologies": "Coforge",
    "birlasoft": "Birlasoft",
    "tata elxsi": "Tata Elxsi",
    "l&t technology services": "L&T Technology Services",
    "l and t technology services": "L&T Technology Services",
    # Indian startups / product companies
    "flipkart": "Flipkart",
    "myntra": "Myntra",
    "zomato": "Zomato",
    "eternal": "Zomato",
    "swiggy": "Swiggy",
    "ola": "Ola",
    "ola electric": "Ola Electric",
    "paytm": "Paytm",
    "one97 communications": "Paytm",
    "phonepe": "PhonePe",
    "razorpay": "Razorpay",
    "cred": "CRED",
    "meesho": "Meesho",
    "zepto": "Zepto",
    "byju's": "Byju's",
    "byjus": "Byju's",
    "think & learn": "Byju's",
    "think and learn": "Byju's",
    "unacademy": "Unacademy",
    "dream11": "Dream11",
    "sporta technologies": "Dream11",
    "nykaa": "Nykaa",
    "fsn e-commerce": "Nykaa",
    "fsn e commerce": "Nykaa",
    "policybazaar": "PolicyBazaar",
    "pb fintech": "PolicyBazaar",
    "zoho": "Zoho",
    "zoho corporation": "Zoho",
    "freshworks": "Freshworks",
    "chargebee": "Chargebee",
    "browserstack": "BrowserStack",
    "postman": "Postman",
    "zerodha": "Zerodha",
    "groww": "Groww",
    "jupiter": "Jupiter",
    "navi": "Navi",
    "pine labs": "Pine Labs",
    "bharatpe": "BharatPe",
    "sharechat": "ShareChat",
    "mohalla tech": "ShareChat",
    "dailyhunt": "Dailyhunt",
    "verse innovation": "Dailyhunt",
    "glance": "Glance",
    "makemytrip": "MakeMyTrip",
    "oyo": "OYO",
    "urban company": "Urban Company",
    "delhivery": "Delhivery",
    "lenskart": "Lenskart",
    "mamaearth": "Mamaearth",
    "honasa consumer": "Mamaearth",
    "boat": "boAt",
    "imagine marketing": "boAt",
    "physicswallah": "PhysicsWallah",
    "pw": "PhysicsWallah",
    "vedantu": "Vedantu",
    "upgrad": "upGrad",
    "simplilearn": "Simplilearn",
    "great learning": "Great Learning",
    "scaler": "Scaler",
    "interviewbit": "InterviewBit",
    "masai school": "Masai School",
    "newton school": "Newton School",
    "turing": "Turing",
    # Global technology companies
    "microsoft": "Microsoft",
    "google": "Google",
    "alphabet": "Alphabet",
    "amazon": "Amazon",
    "amazon web services": "Amazon Web Services",
    "aws": "Amazon Web Services",
    "apple": "Apple",
    "meta": "Meta",
    "facebook": "Meta",
    "netflix": "Netflix",
    "nvidia": "Nvidia",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "intel": "Intel",
    "amd": "AMD",
    "qualcomm": "Qualcomm",
    "samsung": "Samsung",
    "adobe": "Adobe",
    "salesforce": "Salesforce",
    "oracle": "Oracle",
    "sap": "SAP",
    "uber": "Uber",
    "airbnb": "Airbnb",
    "tesla": "Tesla",
    "spacex": "SpaceX",
    "stripe": "Stripe",
    "shopify": "Shopify",
    "twitter": "X",
    "x": "X",
}


@dataclass(frozen=True)
class NormalizedCompany:
    canonical: str
    raw: str
    matched_alias: bool


def clean_company(raw: Optional[str]) -> str:
    """Strip HTML, normalize whitespace and surrounding punctuation."""
    if not raw:
        return ""
    text = html.unescape(str(raw))
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" \t\r\n,;:-–—|()[]{}\"'")
    return text


def _key(value: str) -> str:
    text = value.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_legal_suffix(value: str) -> str:
    text = value.strip()
    # Repeat to handle "Pvt. Ltd." and similar chains.
    changed = True
    while changed:
        changed = False
        lowered = text.lower().strip(" .,")
        for suffix in _LEGAL_SUFFIXES:
            if lowered == suffix:
                return ""
            if lowered.endswith(" " + suffix):
                text = text[: -(len(suffix) + 1)].strip(" .,")
                changed = True
                break
    return text


def normalize_company(raw: Optional[str]) -> NormalizedCompany:
    """Return the canonical company name and whether a curated alias matched."""
    cleaned = clean_company(raw)
    if not cleaned:
        return NormalizedCompany("", cleaned, False)

    candidates = [cleaned]
    without_suffix = _strip_legal_suffix(cleaned)
    if without_suffix and without_suffix != cleaned:
        candidates.append(without_suffix)
    without_the = re.sub(r"^the\s+", "", cleaned, flags=re.IGNORECASE)
    if without_the != cleaned:
        candidates.append(without_the)

    for candidate in candidates:
        canonical = _ALIASES.get(_key(candidate))
        if canonical:
            return NormalizedCompany(canonical, cleaned, True)

    fallback = without_suffix or cleaned
    return NormalizedCompany(fallback, cleaned, False)
