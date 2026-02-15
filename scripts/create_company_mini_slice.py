import argparse
import json
import time
from pathlib import Path
from typing import Iterable, List

import requests

PREFIXES = """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX spb: <http://www.ldbcouncil.org/spb#>
PREFIX gn: <http://www.geonames.org/ontology#>
PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
"""

SELECT_COMPANIES = """
{prefixes}
SELECT ?c
WHERE {{
  ?c a dbo:Company .
}}
LIMIT {limit}
OFFSET {offset}
"""

COMPANY_CONSTRUCT = """
{prefixes}
CONSTRUCT {{
  ?c a dbo:Company .
  ?c rdfs:label ?cLabel .
  ?c spb:prefLabel ?cPref .
  ?c dbo:location ?loc .
  ?c dbo:industry ?industry .
  ?c dbo:keyPerson ?person .
  ?c dbo:foundingYear ?foundingYear .
  ?c dbo:numberOfEmployees ?numEmployees .
}}
WHERE {{
  VALUES ?c {{
    {company_values}
  }}
  OPTIONAL {{ ?c rdfs:label ?cLabel FILTER (LANG(?cLabel) = "en" || LANG(?cLabel) = "") }}
  OPTIONAL {{ ?c spb:prefLabel ?cPref }}
  OPTIONAL {{ ?c dbo:location ?loc }}
  OPTIONAL {{ ?c dbo:industry ?industry }}
  OPTIONAL {{ ?c dbo:keyPerson ?person }}
  OPTIONAL {{ ?c dbo:foundingYear ?foundingYear }}
  OPTIONAL {{ ?c dbo:numberOfEmployees ?numEmployees }}
}}
"""

LOCATION_CONSTRUCT = """
{prefixes}
CONSTRUCT {{
  ?loc a gn:Feature .
  ?loc rdfs:label ?locLabel .
  ?loc spb:prefLabel ?locPref .
  ?loc gn:name ?locName .
  ?loc geo:lat ?lat .
  ?loc geo:long ?long .
  ?loc gn:countryCode ?countryCode .
}}
WHERE {{
  VALUES ?loc {{
    {loc_values}
  }}
  OPTIONAL {{ ?loc a gn:Feature }}
  OPTIONAL {{ ?loc rdfs:label ?locLabel FILTER (LANG(?locLabel) = "en" || LANG(?locLabel) = "") }}
  OPTIONAL {{ ?loc spb:prefLabel ?locPref }}
  OPTIONAL {{ ?loc gn:name ?locName }}
  OPTIONAL {{ ?loc geo:lat ?lat }}
  OPTIONAL {{ ?loc geo:long ?long }}
  OPTIONAL {{ ?loc gn:countryCode ?countryCode }}
}}
"""

INDUSTRY_CONSTRUCT = """
{prefixes}
CONSTRUCT {{
  ?industry rdfs:label ?industryLabel .
  ?industry spb:prefLabel ?industryPref .
}}
WHERE {{
  VALUES ?industry {{
    {industry_values}
  }}
  OPTIONAL {{ ?industry rdfs:label ?industryLabel FILTER (LANG(?industryLabel) = "en" || LANG(?industryLabel) = "") }}
  OPTIONAL {{ ?industry spb:prefLabel ?industryPref }}
}}
"""

PERSON_CONSTRUCT = """
{prefixes}
CONSTRUCT {{
  ?person a foaf:Person .
  ?person rdfs:label ?personLabel .
  ?person foaf:name ?personName .
  ?person spb:prefLabel ?personPref .
}}
WHERE {{
  VALUES ?person {{
    {person_values}
  }}
  OPTIONAL {{ ?person a foaf:Person }}
  OPTIONAL {{ ?person rdfs:label ?personLabel FILTER (LANG(?personLabel) = "en" || LANG(?personLabel) = "") }}
  OPTIONAL {{ ?person foaf:name ?personName }}
  OPTIONAL {{ ?person spb:prefLabel ?personPref }}
}}
"""

SELECT_LOCATIONS = """
{prefixes}
SELECT DISTINCT ?loc
WHERE {{
  VALUES ?c {{
    {company_values}
  }}
  ?c dbo:location ?loc .
}}
"""

SELECT_INDUSTRIES = """
{prefixes}
SELECT DISTINCT ?industry
WHERE {{
  VALUES ?c {{
    {company_values}
  }}
  ?c dbo:industry ?industry .
}}
"""

SELECT_PERSONS = """
{prefixes}
SELECT DISTINCT ?person
WHERE {{
  VALUES ?c {{
    {company_values}
  }}
  ?c dbo:keyPerson ?person .
}}
"""


def chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def sparql_request(endpoint: str, query: str, accept: str, timeout: int) -> str:
    resp = requests.post(
        endpoint,
        data={"query": query},
        headers={"Accept": accept},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.text


def append_text(path: Path, text: str) -> None:
    if not text.strip():
        return
    with path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://localhost:7200/repositories/spb_1m")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--company-page", type=int, default=500)
    parser.add_argument("--entity-chunk", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--out", default="artifacts/slices/spb_company_mini_slice.ttl")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    companies: List[str] = []
    offset = 0
    while len(companies) < args.limit:
        page = min(args.company_page, args.limit - len(companies))
        select_query = SELECT_COMPANIES.format(prefixes=PREFIXES, limit=page, offset=offset)
        data = None
        for attempt in range(1, args.retries + 1):
            try:
                data = sparql_request(args.source, select_query, "application/sparql-results+json", args.timeout)
                break
            except Exception:
                if attempt >= args.retries:
                    raise
                time.sleep(2 * attempt)
        rows = json.loads(data).get("results", {}).get("bindings", []) if data else []
        if not rows:
            break
        companies.extend([row["c"]["value"] for row in rows])
        offset += page

    companies = companies[: args.limit]
    if not companies:
        raise SystemExit("No companies returned from SELECT; aborting.")

    out_path.write_text("", encoding="utf-8")
    total = 0
    for chunk in chunked(companies, args.chunk_size):
        values = "\n    ".join(f"<{c}>" for c in chunk)
        company_query = COMPANY_CONSTRUCT.format(prefixes=PREFIXES, company_values=values)
        for attempt in range(1, args.retries + 1):
            try:
                text = sparql_request(args.source, company_query, "text/turtle", args.timeout)
                append_text(out_path, text)
                break
            except Exception:
                if attempt >= args.retries:
                    raise
                time.sleep(2 * attempt)

        for select_tpl, construct_tpl, key in (
            (SELECT_LOCATIONS, LOCATION_CONSTRUCT, "loc"),
            (SELECT_INDUSTRIES, INDUSTRY_CONSTRUCT, "industry"),
            (SELECT_PERSONS, PERSON_CONSTRUCT, "person"),
        ):
            select_q = select_tpl.format(prefixes=PREFIXES, company_values=values)
            data = None
            for attempt in range(1, args.retries + 1):
                try:
                    data = sparql_request(args.source, select_q, "application/sparql-results+json", args.timeout)
                    break
                except Exception:
                    if attempt >= args.retries:
                        raise
                    time.sleep(2 * attempt)
            rows = json.loads(data).get("results", {}).get("bindings", []) if data else []
            if not rows:
                continue
            entities = [row[key]["value"] for row in rows]
            for echunk in chunked(entities, args.entity_chunk):
                ev = "\n    ".join(f"<{e}>" for e in echunk)
                q = construct_tpl.format(
                    prefixes=PREFIXES,
                    loc_values=ev,
                    industry_values=ev,
                    person_values=ev,
                )
                for attempt in range(1, args.retries + 1):
                    try:
                        text = sparql_request(args.source, q, "text/turtle", args.timeout)
                        append_text(out_path, text)
                        break
                    except Exception:
                        if attempt >= args.retries:
                            raise
                        time.sleep(2 * attempt)

        total += len(chunk)
        print(f"Wrote chunk: {total}/{len(companies)} companies")

    print(f"Wrote slice to {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
