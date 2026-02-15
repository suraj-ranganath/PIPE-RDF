from dataclasses import dataclass
from typing import List


@dataclass
class QuestionTemplate:
    name: str
    category: str
    question_template: str
    sparql_template: str
    slot_query: str
    slot_vars: List[str]
    answer_vars: List[str]


PREFIXES = """
PREFIX mv: <http://data.linkedmdb.org/movie/>
PREFIX remv: <http://data.linkedmdb.org/resource/movie/>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
""".strip()


def build_templates() -> List[QuestionTemplate]:
    templates = []

    templates.append(
        QuestionTemplate(
            name="genre_of_film",
            category="generic",
            question_template="What is the genre of {film}?",
            sparql_template=PREFIXES
            + "\nSELECT ?genre_name WHERE { {film} mv:genre ?genre . ?genre mv:film_genre_name ?genre_name . }",
            slot_query=PREFIXES
            + "\nSELECT ?film WHERE { ?film a mv:film . ?film mv:genre ?g . }",
            slot_vars=["film"],
            answer_vars=["genre_name"],
        )
    )

    templates.append(
        QuestionTemplate(
            name="release_date_of_film",
            category="generic",
            question_template="What is the release date of {film}?",
            sparql_template=PREFIXES
            + "\nSELECT ?date WHERE { {film} mv:initial_release_date ?date . }",
            slot_query=PREFIXES + "\nSELECT ?film WHERE { ?film a mv:film . ?film mv:initial_release_date ?d . }",
            slot_vars=["film"],
            answer_vars=["date"],
        )
    )

    templates.append(
        QuestionTemplate(
            name="count_films_by_director",
            category="counting",
            question_template="How many films did {director} direct?",
            sparql_template=PREFIXES
            + "\nSELECT (COUNT(?film) AS ?count) WHERE { ?film remv:has_director {director} . }",
            slot_query=PREFIXES + "\nSELECT ?director WHERE { ?film remv:has_director ?director . }",
            slot_vars=["director"],
            answer_vars=["count"],
        )
    )

    templates.append(
        QuestionTemplate(
            name="count_films_by_actor",
            category="counting",
            question_template="How many films did {actor} act in?",
            sparql_template=PREFIXES
            + "\nSELECT (COUNT(?film) AS ?count) WHERE { ?film remv:has_actor {actor} . }",
            slot_query=PREFIXES + "\nSELECT ?actor WHERE { ?film remv:has_actor ?actor . }",
            slot_vars=["actor"],
            answer_vars=["count"],
        )
    )

    templates.append(
        QuestionTemplate(
            name="actors_by_director",
            category="multi-hop",
            question_template="Which actors starred in films directed by {director}?",
            sparql_template=PREFIXES
            + "\nSELECT DISTINCT ?actor_name WHERE { ?film remv:has_director {director} ; remv:has_actor ?actor . ?actor mv:actor_name ?actor_name . }",
            slot_query=PREFIXES
            + "\nSELECT ?director WHERE { ?film remv:has_director ?director ; remv:has_actor ?actor . }",
            slot_vars=["director"],
            answer_vars=["actor_name"],
        )
    )

    templates.append(
        QuestionTemplate(
            name="films_by_genre_and_producer",
            category="intersection",
            question_template="Which films are in genre {genre} and produced by {producer}?",
            sparql_template=PREFIXES
            + "\nSELECT ?title WHERE { ?film mv:genre {genre} ; remv:has_producer {producer} ; dcterms:title ?title . }",
            slot_query=PREFIXES
            + "\nSELECT ?genre ?producer WHERE { ?film mv:genre ?genre ; remv:has_producer ?producer . }",
            slot_vars=["genre", "producer"],
            answer_vars=["title"],
        )
    )

    templates.append(
        QuestionTemplate(
            name="films_not_produced_by",
            category="difference",
            question_template="Which films directed by {director} were not produced by {producer}?",
            sparql_template=PREFIXES
            + "\nSELECT ?title WHERE { ?film remv:has_director {director} ; dcterms:title ?title . FILTER NOT EXISTS { ?film remv:has_producer {producer} . } }",
            slot_query=PREFIXES
            + "\nSELECT ?director ?producer WHERE { ?film remv:has_director ?director ; remv:has_producer ?producer . }",
            slot_vars=["director", "producer"],
            answer_vars=["title"],
        )
    )

    templates.append(
        QuestionTemplate(
            name="yesno_actor_in_film",
            category="yesno",
            question_template="Did {actor} act in {film}?",
            sparql_template=PREFIXES
            + "\nASK WHERE { {film} remv:has_actor {actor} . }",
            slot_query=PREFIXES
            + "\nSELECT ?film ?actor WHERE { ?film remv:has_actor ?actor . }",
            slot_vars=["film", "actor"],
            answer_vars=[],
        )
    )

    templates.append(
        QuestionTemplate(
            name="superlative_runtime",
            category="superlative",
            question_template="Which film has the longest runtime?",
            sparql_template=PREFIXES
            + "\nSELECT ?title WHERE { ?film mv:runtime ?rt ; dcterms:title ?title . } ORDER BY DESC(?rt) LIMIT 1",
            slot_query=PREFIXES + "\nSELECT ?film WHERE { ?film mv:runtime ?rt . } LIMIT 1",
            slot_vars=[],
            answer_vars=["title"],
        )
    )
    templates.append(
        QuestionTemplate(
            name="superlative_shortest_runtime",
            category="superlative",
            question_template="Which film has the shortest runtime?",
            sparql_template=PREFIXES
            + "\nSELECT ?title WHERE { ?film mv:runtime ?rt ; dcterms:title ?title . } ORDER BY ASC(?rt) LIMIT 1",
            slot_query=PREFIXES + "\nSELECT ?film WHERE { ?film mv:runtime ?rt . } LIMIT 1",
            slot_vars=[],
            answer_vars=["title"],
        )
    )
    templates.append(
        QuestionTemplate(
            name="superlative_latest_release",
            category="superlative",
            question_template="Which film has the most recent release date?",
            sparql_template=PREFIXES
            + "\nSELECT ?title WHERE { ?film mv:initial_release_date ?d ; dcterms:title ?title . } ORDER BY DESC(?d) LIMIT 1",
            slot_query=PREFIXES + "\nSELECT ?film WHERE { ?film mv:initial_release_date ?d . } LIMIT 1",
            slot_vars=[],
            answer_vars=["title"],
        )
    )
    templates.append(
        QuestionTemplate(
            name="superlative_runtime_by_genre",
            category="superlative",
            question_template="Which film in genre {genre} has the longest runtime?",
            sparql_template=PREFIXES
            + "\nSELECT ?title WHERE { ?film mv:genre {genre} ; mv:runtime ?rt ; dcterms:title ?title . } ORDER BY DESC(?rt) LIMIT 1",
            slot_query=PREFIXES + "\nSELECT ?genre WHERE { ?film mv:genre ?genre . }",
            slot_vars=["genre"],
            answer_vars=["title"],
        )
    )
    templates.append(
        QuestionTemplate(
            name="superlative_runtime_by_director",
            category="superlative",
            question_template="Which film directed by {director} has the longest runtime?",
            sparql_template=PREFIXES
            + "\nSELECT ?title WHERE { ?film remv:has_director {director} ; mv:runtime ?rt ; dcterms:title ?title . } ORDER BY DESC(?rt) LIMIT 1",
            slot_query=PREFIXES + "\nSELECT ?director WHERE { ?film remv:has_director ?director . }",
            slot_vars=["director"],
            answer_vars=["title"],
        )
    )
    templates.append(
        QuestionTemplate(
            name="superlative_director_most_films",
            category="superlative",
            question_template="Which director has directed the most films?",
            sparql_template=PREFIXES
            + "\nSELECT ?director_name WHERE { ?film remv:has_director ?director . ?director mv:director_name ?director_name . } GROUP BY ?director ?director_name ORDER BY DESC(COUNT(?film)) LIMIT 1",
            slot_query=PREFIXES + "\nSELECT ?director WHERE { ?film remv:has_director ?director . } LIMIT 1",
            slot_vars=[],
            answer_vars=["director_name"],
        )
    )

    templates.append(
        QuestionTemplate(
            name="ordinal_third_film",
            category="ordinal",
            question_template="What is the third film directed by {director} by release date?",
            sparql_template=PREFIXES
            + "\nSELECT ?title WHERE { ?film remv:has_director {director} ; mv:initial_release_date ?d ; dcterms:title ?title . } ORDER BY ?d OFFSET 2 LIMIT 1",
            slot_query=PREFIXES
            + "\nSELECT ?director WHERE { ?film remv:has_director ?director . }",
            slot_vars=["director"],
            answer_vars=["title"],
        )
    )
    templates.append(
        QuestionTemplate(
            name="ordinal_first_film",
            category="ordinal",
            question_template="What is the first film directed by {director} by release date?",
            sparql_template=PREFIXES
            + "\nSELECT ?title WHERE { ?film remv:has_director {director} ; mv:initial_release_date ?d ; dcterms:title ?title . } ORDER BY ?d OFFSET 0 LIMIT 1",
            slot_query=PREFIXES
            + "\nSELECT ?director WHERE { ?film remv:has_director ?director . }",
            slot_vars=["director"],
            answer_vars=["title"],
        )
    )
    templates.append(
        QuestionTemplate(
            name="ordinal_second_film",
            category="ordinal",
            question_template="What is the second film directed by {director} by release date?",
            sparql_template=PREFIXES
            + "\nSELECT ?title WHERE { ?film remv:has_director {director} ; mv:initial_release_date ?d ; dcterms:title ?title . } ORDER BY ?d OFFSET 1 LIMIT 1",
            slot_query=PREFIXES
            + "\nSELECT ?director WHERE { ?film remv:has_director ?director . }",
            slot_vars=["director"],
            answer_vars=["title"],
        )
    )

    templates.append(
        QuestionTemplate(
            name="comparative_directors",
            category="comparative",
            question_template="Which director has more films, {director1} or {director2}?",
            sparql_template=PREFIXES
            + "\nSELECT ?winner_name WHERE {\n"
            + "  VALUES ?director { {director1} {director2} }\n"
            + "  ?film remv:has_director ?director .\n"
            + "  ?director mv:director_name ?winner_name .\n"
            + "}\n"
            + "GROUP BY ?director ?winner_name\n"
            + "ORDER BY DESC(COUNT(?film))\n"
            + "LIMIT 1",
            slot_query=PREFIXES
            + "\nSELECT ?director1 ?director2 WHERE { ?film1 remv:has_director ?director1 . ?film2 remv:has_director ?director2 . FILTER(?director1 != ?director2) }",
            slot_vars=["director1", "director2"],
            answer_vars=["winner_name"],
        )
    )

    templates.append(
        QuestionTemplate(
            name="films_by_director_and_country",
            category="multi-hop",
            question_template="Which films directed by {director} are associated with {country}?",
            sparql_template=PREFIXES
            + "\nSELECT ?title WHERE { ?film remv:has_director {director} ; remv:has_country {country} ; dcterms:title ?title . }",
            slot_query=PREFIXES
            + "\nSELECT ?director ?country WHERE { ?film remv:has_director ?director ; remv:has_country ?country . }",
            slot_vars=["director", "country"],
            answer_vars=["title"],
        )
    )

    return templates
