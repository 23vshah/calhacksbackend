import re
from typing import Dict, Any

def standardize_county_name(county: str) -> str:
    """Standardize county names for consistent storage"""
    # Remove common suffixes and standardize
    county = county.strip()
    county = re.sub(r'\s+County$', '', county, flags=re.IGNORECASE)
    county = re.sub(r'\s+', ' ', county)  # Normalize whitespace
    return county.title()

def normalize_geographic_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize geographic data for consistent processing"""
    if 'county' in data:
        data['county'] = standardize_county_name(data['county'])
    return data

def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between different units"""
    # Basic unit conversions
    conversions = {
        ('percent', 'decimal'): lambda x: x / 100,
        ('decimal', 'percent'): lambda x: x * 100,
        ('thousands', 'units'): lambda x: x * 1000,
        ('units', 'thousands'): lambda x: x / 1000,
    }
    
    key = (from_unit, to_unit)
    if key in conversions:
        return conversions[key](value)
    
    return value  # No conversion needed

def validate_metric_value(value: Any) -> float:
    """Validate and convert metric values to float"""
    try:
        if isinstance(value, str):
            # Remove common formatting
            value = value.replace(',', '').replace('$', '').replace('%', '')
        return float(value)
    except (ValueError, TypeError):
        return 0.0

