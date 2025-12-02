#!/usr/bin/env python3
"""
Generate common names using Wikidata P1843 (taxon common name).

P1843 is the specific Wikidata property for common names of organisms.

Usage:
    python generate_common_names_wikidata.py
"""

import json
import os
import time
import requests
from typing import Dict, Optional

# Hardcoded overrides (trusted sources for most common plants)
COMMON_NAME_OVERRIDES = {
    "Solanum lycopersicum": "Tomato",
    "Solanum tuberosum": "Potato",
    "Cucurbita pepo": "Zucchini",
    "Lactuca sativa": "Lettuce",
    "Daucus carota": "Carrot",
    "Fragaria vesca": "Strawberry",
    "Lavandula angustifolia": "English Lavender",
    "Ocimum basilicum": "Sweet Basil",
    "Monstera deliciosa": "Swiss Cheese Plant",
    "Anthurium andraeanum": "Flamingo Flower",
    "Ficus lyrata": "Fiddle Leaf Fig",
    "Acer palmatum": "Japanese Maple",
    "Aloe vera": "Aloe Vera",
    "Rosmarinus officinalis": "Rosemary",
    "Thymus vulgaris": "Thyme",
    "Origanum vulgare": "Oregano",
    "Mentha piperita": "Peppermint",
    "Rosa damascena": "Damask Rose",
    "Pinus sylvestris": "Scots Pine",
    "Quercus robur": "English Oak",
}

# Wikidata SPARQL endpoint
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
CACHE_FILE = "wikidata_cache.json"


def load_cache() -> Dict[str, str]:
    """Load cached Wikidata results."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_cache(cache: Dict[str, str]) -> None:
    """Save cache to file."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def extract_genus(scientific_name: str) -> str:
    """Extract genus as fallback."""
    parts = scientific_name.split()
    return parts[0].capitalize() if parts else scientific_name


def query_wikidata_p1843(scientific_name: str) -> Optional[str]:
    """Query Wikidata for taxon common name using P1843."""
    try:
        # Query P1843 - get ALL results and filter for English
        query = f"""
        SELECT ?commonName WHERE {{
          ?item wdt:P225 "{scientific_name}".
          ?item wdt:P1843 ?commonName.
        }}
        """
        
        response = requests.post(
            WIKIDATA_SPARQL,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", {}).get("bindings", [])
            
            # Filter for English entries
            if results:
                for result in results:
                    common_name = result["commonName"]["value"]
                    lang = result["commonName"].get("xml:lang", "")
                    
                    # Return first English result
                    if lang == "en" and common_name and common_name.strip():
                        return common_name
        
        return None
        
    except Exception as e:
        return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    species_json = os.path.join(script_dir, "models", "species_to_common_name.json")
    output_json = os.path.join(script_dir, "models", "species_to_common_name2.json")
    
    print("🌍 PlantNet Common Names Generator (Wikidata P1843)")
    print("=" * 60)
    print(f"\nInput:  {species_json}")
    print(f"Output: {output_json}\n")
    
    # Check if input exists
    if not os.path.exists(species_json):
        print(f"❌ File not found: {species_json}")
        print("   Place plantnet300K_species_id_2_name.json in the meta/ folder")
        return
    
    # Load species data
    with open(species_json, 'r') as f:
        species_data = json.load(f)
    
    print(f"📊 Found {len(species_data)} species\n")
    
    # Load cache
    cache = load_cache()
    print(f"💾 Loaded {len(cache)} cached entries\n")
    
    # Query each species
    common_names = {}
    found_via_override = 0
    found_via_wikidata = 0
    found_via_genus = 0
    
    species_list = list(species_data.items())
    
    for i, (species_id, scientific_name) in enumerate(species_list):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(species_data)} | Wikidata: {found_via_wikidata} | Cached: {len(cache)}")
        
        # Check override first
        if scientific_name in COMMON_NAME_OVERRIDES:
            common_names[scientific_name] = COMMON_NAME_OVERRIDES[scientific_name]
            found_via_override += 1
            continue
        
        # Check cache
        if scientific_name in cache:
            result = cache[scientific_name]
            common_names[scientific_name] = result
            if result != extract_genus(scientific_name):
                found_via_wikidata += 1
            else:
                found_via_genus += 1
            continue
        
        # Query Wikidata P1843
        common_name = query_wikidata_p1843(scientific_name)
        
        if common_name and common_name.strip():
            common_names[scientific_name] = common_name
            cache[scientific_name] = common_name
            found_via_wikidata += 1
        else:
            # Fallback to genus
            fallback = extract_genus(scientific_name)
            common_names[scientific_name] = fallback
            cache[scientific_name] = fallback
            found_via_genus += 1
        
        # Be nice to Wikidata - rate limit
        if i % 5 == 0:
            time.sleep(0.3)
    
    # Save cache
    save_cache(cache)
    print(f"\n💾 Saved cache with {len(cache)} entries")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    # Save results
    with open(output_json, 'w') as f:
        json.dump(common_names, f, indent=2)
    
    print(f"\n✅ Done! Generated {len(common_names)} common names")
    print(f"   🎯 From hardcoded overrides: {found_via_override}")
    print(f"   🌍 From Wikidata P1843: {found_via_wikidata}")
    print(f"   📚 Fallback (genus name): {found_via_genus}")
    print(f"\n📁 Saved to: {output_json}")
    
    # Show samples
    print(f"\n📊 Sample mappings:")
    samples = []
    for sci, comm in common_names.items():
        if sci in COMMON_NAME_OVERRIDES:
            samples.append((sci, comm, "🎯"))
        elif comm != extract_genus(sci):
            samples.append((sci, comm, "🌍"))
        else:
            samples.append((sci, comm, "📚"))
        if len(samples) >= 15:
            break
    
    for sci, comm, marker in samples:
        print(f"   {marker} {sci:40} → {comm}")


if __name__ == "__main__":
    main()