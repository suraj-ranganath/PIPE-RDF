import argparse
from pathlib import Path
from typing import Iterable, List, Set, Tuple
import glob

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
DBO_COMPANY = "http://dbpedia.org/ontology/Company"
FOAF_PERSON = "http://xmlns.com/foaf/0.1/Person"
GN_FEATURE = "http://www.geonames.org/ontology#Feature"

P_COMPANY = {
    RDF_TYPE,
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.ldbcouncil.org/spb#prefLabel",
    "http://dbpedia.org/ontology/location",
    "http://dbpedia.org/ontology/industry",
    "http://dbpedia.org/ontology/keyPerson",
    "http://dbpedia.org/ontology/foundingYear",
    "http://dbpedia.org/ontology/numberOfEmployees",
}
P_LOCATION = {
    RDF_TYPE,
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.ldbcouncil.org/spb#prefLabel",
    "http://www.geonames.org/ontology#name",
    "http://www.w3.org/2003/01/geo/wgs84_pos#lat",
    "http://www.w3.org/2003/01/geo/wgs84_pos#long",
    "http://www.geonames.org/ontology#countryCode",
}
P_INDUSTRY = {
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.ldbcouncil.org/spb#prefLabel",
}
P_PERSON = {
    RDF_TYPE,
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://xmlns.com/foaf/0.1/name",
    "http://www.ldbcouncil.org/spb#prefLabel",
}


def parse_line(line: str) -> Tuple[str, str, str | None]:
    if not line.startswith("<"):
        return "", "", None
    s_end = line.find(">")
    if s_end == -1:
        return "", "", None
    subj = line[1:s_end]
    rest = line[s_end + 1 :].lstrip()
    if not rest.startswith("<"):
        return subj, "", None
    p_end = rest.find(">")
    if p_end == -1:
        return subj, "", None
    pred = rest[1:p_end]
    rest2 = rest[p_end + 1 :].lstrip()
    if rest2.startswith("<"):
        o_end = rest2.find(">")
        if o_end == -1:
            return subj, pred, None
        obj = rest2[1:o_end]
        return subj, pred, obj
    return subj, pred, None


def scan_files(files: List[str], limit: int) -> Set[str]:
    companies: Set[str] = set()
    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                subj, pred, obj = parse_line(line)
                if pred == RDF_TYPE and obj == DBO_COMPANY:
                    companies.add(subj)
                    if len(companies) >= limit:
                        return companies
    return companies


def write_matches(
    files: List[str],
    out_path: Path,
    companies: Set[str],
    locations: Set[str],
    industries: Set[str],
    persons: Set[str],
) -> Tuple[Set[str], Set[str], Set[str]]:
    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                subj, pred, obj = parse_line(line)
                if not pred:
                    continue
                if subj in companies and pred in P_COMPANY:
                    with out_path.open("a", encoding="utf-8") as out:
                        out.write(line)
                    if pred == "http://dbpedia.org/ontology/location" and obj:
                        locations.add(obj)
                    elif pred == "http://dbpedia.org/ontology/industry" and obj:
                        industries.add(obj)
                    elif pred == "http://dbpedia.org/ontology/keyPerson" and obj:
                        persons.add(obj)
    return locations, industries, persons


def write_entities(files: List[str], out_path: Path, entities: Set[str], predicates: Set[str]) -> None:
    if not entities:
        return
    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                subj, pred, obj = parse_line(line)
                if subj in entities and pred in predicates:
                    with out_path.open("a", encoding="utf-8") as out:
                        out.write(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default="external/ldbc_spb_bm_2.0/dist/generated_data/*.nq")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--out", default="artifacts/slices/spb_company_mini_slice.nq")
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise SystemExit("No input N-Quads found.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")

    companies = scan_files(files, args.limit)
    if not companies:
        raise SystemExit("No companies found in input.")

    locations: Set[str] = set()
    industries: Set[str] = set()
    persons: Set[str] = set()

    locations, industries, persons = write_matches(files, out_path, companies, locations, industries, persons)

    # Add explicit types for related entities if present
    write_entities(files, out_path, locations, P_LOCATION)
    write_entities(files, out_path, industries, P_INDUSTRY)
    write_entities(files, out_path, persons, P_PERSON)

    print(f"Companies: {len(companies)} | Locations: {len(locations)} | Industries: {len(industries)} | Persons: {len(persons)}")
    print(f"Wrote slice to {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
