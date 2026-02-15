from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

from rdflib import Graph, Literal, RDF, RDFS, URIRef
from rdflib.namespace import XSD

from .schema import Schema


@dataclass
class SyntheticKG:
    schema: Schema
    graph: Graph
    label_map: Dict[URIRef, str]

    @classmethod
    def build(cls, schema: Schema, sizes: Dict[str, int]) -> "SyntheticKG":
        g = Graph()
        # bind namespaces
        for prefix, ns in schema.ns.items():
            g.bind(prefix, ns)

        mv = schema.ns["mv"]
        remv = schema.ns["remv"]
        dcterms = schema.ns["dcterms"]

        label_map: Dict[URIRef, str] = {}

        def add_entity(class_uri: URIRef, name: str, idx: int) -> URIRef:
            uri = remv[f"{name}_{idx:03d}"]
            g.add((uri, RDF.type, class_uri))
            label_prop = schema.label_property_for_class(class_uri) or RDFS.label
            g.add((uri, label_prop, Literal(name.replace("_", " ") + f" {idx:03d}")))
            label_map[uri] = name.replace("_", " ") + f" {idx:03d}"
            id_prop = schema.id_property_for_class(class_uri)
            if id_prop is not None:
                g.add((uri, id_prop, Literal(idx, datatype=XSD.int)))
            return uri

        # Create entities
        films = [add_entity(mv.film, "film", i + 1) for i in range(sizes["films"])]
        directors = [add_entity(mv.director, "director", i + 1) for i in range(sizes["directors"])]
        actors = [add_entity(mv.actor, "actor", i + 1) for i in range(sizes["actors"])]
        producers = [add_entity(mv.producer, "producer", i + 1) for i in range(sizes["producers"])]
        genres = [add_entity(mv.film_genre, "genre", i + 1) for i in range(sizes["genres"])]
        countries = [add_entity(mv.country, "country", i + 1) for i in range(sizes["countries"])]

        # Properties for film
        p_title = dcterms.title
        p_director = remv.has_director
        p_actor = remv.has_actor
        p_producer = remv.has_producer
        p_genre = mv.genre
        p_country = remv.has_country
        p_release = mv.initial_release_date
        p_runtime = mv.runtime

        for idx, film in enumerate(films, start=1):
            g.add((film, p_title, Literal(f"Film {idx:03d}")))
            label_map[film] = f"Film {idx:03d}"

            director = random.choice(directors)
            producer = random.choice(producers)
            genre = random.choice(genres)
            country = random.choice(countries)
            film_actors = random.sample(actors, k=random.randint(2, 4))

            g.add((film, p_director, director))
            g.add((film, p_producer, producer))
            g.add((film, p_genre, genre))
            g.add((film, p_country, country))
            for actor in film_actors:
                g.add((film, p_actor, actor))

            year = random.randint(1990, 2023)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            g.add((film, p_release, Literal(f"{year:04d}-{month:02d}-{day:02d}", datatype=XSD.date)))
            g.add((film, p_runtime, Literal(random.randint(70, 180), datatype=XSD.int)))

        return cls(schema=schema, graph=g, label_map=label_map)

    def label_of(self, uri: URIRef) -> str:
        if uri in self.label_map:
            return self.label_map[uri]
        for label_prop in (RDFS.label,):
            for obj in self.graph.objects(uri, label_prop):
                return str(obj)
        return uri.split("/")[-1]

    def run_select(self, query: str) -> List[Dict[str, URIRef]]:
        results = []
        for row in self.graph.query(query):
            binding = {}
            for var, val in row.asdict().items():
                binding[str(var)] = val
            results.append(binding)
        return results

    def run_query_answers(self, query: str) -> List[str]:
        results = self.graph.query(query)
        answers = []
        for row in results:
            if hasattr(row, "asdict"):
                row_dict = row.asdict()
                for val in row_dict.values():
                    answers.append(self._format_value(val))
            else:
                answers.append(self._format_value(row))
        return answers

    def _format_value(self, val) -> str:
        if isinstance(val, URIRef):
            return self.label_of(val)
        return str(val)
