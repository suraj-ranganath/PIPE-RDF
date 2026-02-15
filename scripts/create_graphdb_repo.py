import argparse
from pathlib import Path
import requests

CONFIG_TEMPLATE = """@prefix rep: <http://www.openrdf.org/config/repository#>.
@prefix sr: <http://www.openrdf.org/config/repository/sail#>.
@prefix sail: <http://www.openrdf.org/config/sail#>.
@prefix graphdb: <http://www.ontotext.com/config/graphdb#>.

[] a rep:Repository ;
   rep:repositoryID "{repo_id}" ;
   rep:repositoryImpl [
      rep:repositoryType "graphdb:SailRepository" ;
      sr:sailImpl [
         sail:sailType "graphdb:Sail" ;
         graphdb:ruleset "{ruleset}" ;
         graphdb:storageFolder "{repo_id}" ;
         graphdb:enable-context-index "false" ;
         graphdb:enablePredicateList "false" ;
      ]
   ].
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphdb", default="http://localhost:7200")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--ruleset", default="empty")
    parser.add_argument("--out", default="artifacts/graphdb")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / f"{args.repo}.ttl"
    cfg_path.write_text(CONFIG_TEMPLATE.format(repo_id=args.repo, ruleset=args.ruleset), encoding="utf-8")

    url = args.graphdb.rstrip("/") + "/rest/repositories"
    with cfg_path.open("rb") as f:
        files = {"config": (cfg_path.name, f, "text/turtle")}
        resp = requests.post(url, files=files, timeout=60)
    if resp.status_code not in (200, 201, 204):
        raise SystemExit(f"Repo create failed: {resp.status_code} {resp.text}")
    print(f"Repository created: {args.repo}")


if __name__ == "__main__":
    main()
