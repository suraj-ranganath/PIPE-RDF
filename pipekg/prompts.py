SYSTEM_NL_TEMPLATE = """You are an expert knowledge graph engineer. Output must be STRICT JSON only. Return a JSON array (even if length 1). No markdown, no extra text."""

SYSTEM_SPARQL_GEN = """You are an expert SPARQL engineer. Produce a SPARQL query for the given question using only the provided prefixes and schema. Return only SPARQL."""

SYSTEM_REPAIR = """You are an expert SPARQL debugger. Fix the query given the error message and schema. Return only corrected SPARQL."""

SYSTEM_PARAPHRASE = """You are a paraphrasing assistant. Output must be STRICT JSON only. Return a JSON array of strings. No markdown, no extra text."""

TEMPLATE_REQUEST = """
Schema (prefixes + key classes/properties):
{schema}

Generate {n} diverse natural-language question templates for category: {category}.
Category hints (use these relations if helpful): {hints}
Use ONLY relations and entity types that appear in the schema summary (especially the Allowed predicates/types list). Do not invent relations.
Each template must be fillable by real entities in the graph and easy to answer with a simple SPARQL query.
Limit slots to at most 2 entities (0 or 1 preferred).
Only place entities where it makes sense in the question; avoid over-specifying.
Questions must sound like realistic user questions asked about the knowledge graph.
Keep questions natural and human-like, but ensure they are directly and unambiguously answerable from the graph.
Avoid schema-probing or meaningless questions (e.g., "type of", "RDF rank", "most famous", "most popular").
Avoid placeholders like "metric", "feature", or "rdfRank" unless they are actual entities in the schema.
Slot names must be unique. If you need two entities of the same type, use distinct names like person1/person2.
Include some templates with zero entity slots (slots: []) when the category supports it (e.g., counting/superlative/ordinal/yesno).
Slots must correspond to entity IRIs. Literal slots are allowed ONLY when the schema explicitly includes literal-valued predicates
(e.g., dbo:foundingYear, dbo:numberOfEmployees). In that case, name the literal slot as foundingYear or numberOfEmployees (or year/number).
For comparative questions, always use two distinct entity slots (e.g., person1/person2, location1/location2).
Placeholders must only be used where a concrete entity is needed, not as a generic class noun (e.g., "the tallest {{person}}" is invalid; use "the tallest person in {{location}}" instead).
Avoid repeating any of these templates if provided: {avoid}
Use slot placeholders in braces, e.g., {{person}}, {{company}}, {{location}}.
Return ONLY valid JSON (no markdown). Always return a JSON array (even if n=1), with objects having keys:
- template: the question template
- slots: list of slot names used in the template
Do not include any explanation or extra text.
Example JSON:
[
  {{"template": "Which {{person}} works at {{company}}?", "slots": ["person", "company"]}}
]
""".strip()

SPARQL_REQUEST = """
Prefixes:
{prefixes}

Schema summary:
{schema}

Question:
{question}

Use entity hints (IRIs + types) if provided in the schema summary.
Include rdf:type constraints for entity variables when type hints are known and relevant.
Use ONLY predicates/classes that appear in the schema summary (Allowed predicates/types). Do not invent new relations.
Bind hinted entities explicitly using VALUES at the start of the WHERE block, e.g., VALUES ?person {{ <IRI> }}.
Avoid subqueries, GROUP BY, and aggregates. Use ORDER BY only when the question explicitly requires a superlative or ordinal.
Do not add label/name constraints unless the question explicitly asks for a label, name, or description.

SPARQL best practices (follow strictly):
- For yes/no questions, use ASK {{ ... }} (not SELECT with boolean).
- For counting questions, use SELECT (COUNT(DISTINCT ?var) AS ?count) WHERE {{ ... }}.
- For set-returning queries (intersection, difference, multi-hop), use SELECT DISTINCT.
- For superlative/ordinal questions, always include ORDER BY with LIMIT 1. Add a tie-breaking secondary sort (e.g., ORDER BY DESC(?count) ?company).
- Place VALUES clauses immediately after WHERE {{ before triple patterns.
- Cast literal comparisons explicitly when needed (e.g., xsd:integer).
Return a SPARQL query.
""".strip()

REVERSE_QUERY_REQUEST = """
Prefixes:
{prefixes}

Schema summary:
{schema}

Question template:
{template}

Slot names (variables must match exactly):
{slots}

Slot type hints (use when slot name matches; if provided):
{slot_type_hints}

Generate a SPARQL SELECT query that returns bindings for each slot variable.
Requirements:
- Use variables named exactly as the slot names, prefixed with '?' (e.g., ?person, ?company).
- Include rdf:type constraints ONLY for slot variables that have explicit type hints above.
- Do NOT invent rdf:type constraints for other variables (e.g., industry) unless the schema summary explicitly lists that type.
- If a slot is a literal (e.g., foundingYear/numberOfEmployees), bind it directly in a triple pattern (no rdf:type) and return it in SELECT.
- Prefer instances (not classes); avoid binding rdf:Class/owl:Class as slot values.
- The SELECT clause MUST include every slot variable (e.g., SELECT ?person ?company).
- Only include triple patterns that directly bind slot variables or connect slot variables to each other. Do NOT add extra constraints (e.g., foundingYear, numberOfEmployees) unless required by the template.
- Avoid functions unless absolutely required by the template. If you must use them, prefer LCASE/UCASE (not LOWER/UPPER), and keep them inside FILTER or BIND.
- Keep reverse queries simple: only basic triple patterns. Avoid FILTER, BIND, ORDER BY, GROUP BY, aggregates, subqueries, OPTIONAL, and inline comments.
- Include LIMIT 85.
- Do not invent predicates or literals; only use predicates visible in the schema summary and avoid hardcoding example entities (e.g., "London") unless they appear in the template.
- Do not use label/name predicates (rdfs:label, foaf:name, spb:prefLabel, gn:hasName) unless the template explicitly asks for a name/label.
- Do not include inline comments.
- Return only SPARQL.
""".strip()

REPAIR_REQUEST = """
Schema summary:
{schema}

Original question:
{question}

SPARQL attempt:
{sparql}

Execution error or issue:
{error}

Return corrected SPARQL only.
""".strip()

PARAPHRASE_REQUEST = """
Question:
{question}

Return 2 paraphrases as a JSON array of strings. Response must start with '[' and end with ']'.
""".strip()
