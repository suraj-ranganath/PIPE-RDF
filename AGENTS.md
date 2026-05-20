# Agent Instructions

- The agent owns GraphDB/SPARQL availability for PIPE-RDF runs. On `ds-serv6`, set up, start, stop, restart, and health-check GraphDB directly instead of asking the user to restart it.
- If GraphDB or a SPARQL endpoint times out or becomes unresponsive, first restart the GraphDB service/container yourself, then rerun the endpoint health checks before continuing experiments.
- Only ask the user for help when the server denies the required permissions or credentials and there is no user-space fallback.
