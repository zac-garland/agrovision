#!/usr/bin/env python3
"""
Debug script to inspect all Wikidata properties for a species.

Usage:
    python debug_wikidata.py "Lactuca virosa"
"""

import sys
import requests
import json

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

def debug_species(scientific_name: str):
    """Get all available data for a species from Wikidata."""
    
    print(f"\n🔍 Debugging: {scientific_name}")
    print("=" * 70)
    
    # Query 1: Get the Wikidata item
    query1 = f"""
    SELECT ?item ?itemLabel WHERE {{
      ?item wdt:P225 "{scientific_name}".
      ?item rdfs:label ?itemLabel.
      FILTER(LANG(?itemLabel) = "en")
    }}
    LIMIT 1
    """
    
    response = requests.post(
        WIKIDATA_SPARQL,
        data={"query": query1},
        headers={"Accept": "application/sparql-results+json"},
        timeout=10
    )
    
    data = response.json()
    results = data.get("results", {}).get("bindings", [])
    
    if not results:
        print(f"❌ Not found in Wikidata!")
        return
    
    wikidata_id = results[0]["item"]["value"].split("/")[-1]
    item_label = results[0]["itemLabel"]["value"]
    
    print(f"\n✅ Found: {item_label}")
    print(f"📍 Wikidata ID: {wikidata_id}")
    print(f"🔗 URL: https://www.wikidata.org/wiki/{wikidata_id}")
    
    # Query 2: Get all P1843 values (taxon common name)
    query2 = f"""
    SELECT ?commonName ?lang WHERE {{
      ?item wdt:P225 "{scientific_name}".
      ?item wdt:P1843 ?commonName.
      BIND(LANG(?commonName) as ?lang)
    }}
    ORDER BY ?lang
    """
    
    response = requests.post(
        WIKIDATA_SPARQL,
        data={"query": query2},
        headers={"Accept": "application/sparql-results+json"},
        timeout=10
    )
    
    data = response.json()
    results = data.get("results", {}).get("bindings", [])
    
    print(f"\n📌 P1843 (Taxon Common Names):")
    if results:
        for i, result in enumerate(results, 1):
            common_name = result["commonName"]["value"]
            lang = result["lang"]["value"]
            print(f"   {i}. [{lang}] {common_name}")
    else:
        print("   ❌ None found")
    
    # Query 3: Get all English labels
    query3 = f"""
    SELECT ?label WHERE {{
      ?item wdt:P225 "{scientific_name}".
      ?item rdfs:label ?label.
      FILTER(LANG(?label) = "en")
    }}
    """
    
    response = requests.post(
        WIKIDATA_SPARQL,
        data={"query": query3},
        headers={"Accept": "application/sparql-results+json"},
        timeout=10
    )
    
    data = response.json()
    results = data.get("results", {}).get("bindings", [])
    
    print(f"\n📌 rdfs:label (English):")
    if results:
        for i, result in enumerate(results, 1):
            label = result["label"]["value"]
            print(f"   {i}. {label}")
    else:
        print("   ❌ None found")
    
    # Query 4: Get all description
    query4 = f"""
    SELECT ?desc WHERE {{
      ?item wdt:P225 "{scientific_name}".
      ?item schema:description ?desc.
      FILTER(LANG(?desc) = "en")
    }}
    """
    
    response = requests.post(
        WIKIDATA_SPARQL,
        data={"query": query4},
        headers={"Accept": "application/sparql-results+json"},
        timeout=10
    )
    
    data = response.json()
    results = data.get("results", {}).get("bindings", [])
    
    print(f"\n📌 schema:description (English):")
    if results:
        for i, result in enumerate(results, 1):
            desc = result["desc"]["value"]
            print(f"   {i}. {desc}")
    else:
        print("   ❌ None found")
    
    # Query 5: Get all Wikipedia articles
    query5 = f"""
    SELECT ?wiki ?title WHERE {{
      ?item wdt:P225 "{scientific_name}".
      ?wiki schema:about ?item;
            schema:inLanguage "en";
            schema:name ?title.
    }}
    """
    
    response = requests.post(
        WIKIDATA_SPARQL,
        data={"query": query5},
        headers={"Accept": "application/sparql-results+json"},
        timeout=10
    )
    
    data = response.json()
    results = data.get("results", {}).get("bindings", [])
    
    print(f"\n📌 Wikipedia Articles (English):")
    if results:
        for i, result in enumerate(results, 1):
            title = result["title"]["value"]
            wiki = result["wiki"]["value"]
            print(f"   {i}. {title}")
            print(f"      {wiki}")
    else:
        print("   ❌ None found")
    
    # Query 6: Get all claims
    query6 = f"""
    SELECT ?predicate ?predicateLabel ?object ?objectLabel WHERE {{
      ?item wdt:P225 "{scientific_name}".
      ?item ?predicate ?object.
      FILTER(STRSTARTS(STR(?predicate), "http://www.wikidata.org/prop/direct/"))
      ?prop wikibase:directClaim ?predicate.
      ?prop rdfs:label ?predicateLabel.
      FILTER(LANG(?predicateLabel) = "en")
      OPTIONAL {{
        ?object rdfs:label ?objectLabel.
        FILTER(LANG(?objectLabel) = "en")
      }}
    }}
    ORDER BY ?predicateLabel
    LIMIT 50
    """
    
    response = requests.post(
        WIKIDATA_SPARQL,
        data={"query": query6},
        headers={"Accept": "application/sparql-results+json"},
        timeout=10
    )
    
    data = response.json()
    results = data.get("results", {}).get("bindings", [])
    
    print(f"\n📌 All Wikidata Properties (first 50):")
    if results:
        for i, result in enumerate(results, 1):
            pred = result["predicateLabel"]["value"]
            obj = result.get("objectLabel", {}).get("value") or result["object"]["value"]
            # Truncate long URLs
            if obj.startswith("http"):
                obj = obj.split("/")[-1]
            print(f"   {i}. {pred}: {obj}")
    else:
        print("   ❌ None found")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default test
        species = "Lactuca virosa"
    else:
        species = " ".join(sys.argv[1:])
    
    try:
        debug_species(species)
    except Exception as e:
        print(f"❌ Error: {e}")
