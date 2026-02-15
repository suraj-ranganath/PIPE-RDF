from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rdflib import Graph, RDF, RDFS, URIRef
from rdflib.namespace import Namespace


@dataclass
class PropertyInfo:
    uri: URIRef
    domains: List[URIRef]
    ranges: List[URIRef]


@dataclass
class Schema:
    graph: Graph
    classes: List[URIRef]
    properties: List[PropertyInfo]
    ns: Dict[str, Namespace]

    @classmethod
    def from_ttl(cls, path: str) -> "Schema":
        g = Graph()
        g.parse(path, format="turtle")
        classes = sorted(set(g.subjects(RDF.type, RDFS.Class)), key=str)
        props = []
        for p in g.subjects(RDF.type, RDF.Property):
            domains = list(g.objects(p, RDFS.domain))
            ranges = list(g.objects(p, RDFS.range))
            props.append(PropertyInfo(uri=p, domains=domains, ranges=ranges))
        ns = {
            "mv": Namespace("http://data.linkedmdb.org/movie/"),
            "remv": Namespace("http://data.linkedmdb.org/resource/movie/"),
            "dcterms": Namespace("http://purl.org/dc/terms/"),
            "foaf": Namespace("http://xmlns.com/foaf/0.1/"),
            "rdfs": Namespace("http://www.w3.org/2000/01/rdf-schema#"),
            "xsd": Namespace("http://www.w3.org/2001/XMLSchema#"),
        }
        return cls(graph=g, classes=classes, properties=props, ns=ns)

    def properties_for_domain(self, domain: URIRef) -> List[PropertyInfo]:
        return [p for p in self.properties if domain in p.domains]

    def find_property_by_local(self, local_name: str) -> Optional[URIRef]:
        for p in self.properties:
            if str(p.uri).split("/")[-1] == local_name:
                return p.uri
        return None

    def label_property_for_class(self, class_uri: URIRef) -> Optional[URIRef]:
        candidates = []
        for p in self.properties:
            if class_uri in p.domains:
                local = str(p.uri).split("/")[-1]
                if local.endswith("_name") or local.endswith("_title"):
                    candidates.append(p.uri)
        return candidates[0] if candidates else None

    def id_property_for_class(self, class_uri: URIRef) -> Optional[URIRef]:
        for p in self.properties:
            if class_uri in p.domains:
                local = str(p.uri).split("/")[-1]
                if local.endswith("id"):
                    return p.uri
        return None

    def compact(self, uri: URIRef) -> str:
        for prefix, ns in self.ns.items():
            ns_str = str(ns)
            if str(uri).startswith(ns_str):
                return f"{prefix}:{str(uri)[len(ns_str):]}"
        return str(uri)

    def resolve(self, qname: str) -> URIRef:
        prefix, local = qname.split(":", 1)
        if prefix not in self.ns:
            raise ValueError(f"Unknown prefix: {prefix}")
        return self.ns[prefix][local]
