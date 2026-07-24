from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import urllib.parse
import webbrowser
from pathlib import Path


class ExperimentRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(
        self,
        *arguments: object,
        web_dir: Path,
        experiment_dir: Path,
        experiment_name: str,
        **keyword_arguments: object,
    ) -> None:
        self._web_dir = web_dir.resolve()
        self._experiment_dir = experiment_dir.resolve()
        self._experiment_name = experiment_name
        super().__init__(*arguments, **keyword_arguments)

    def translate_path(self, path: str) -> str:
        parsed_path = urllib.parse.urlparse(path).path
        decoded_path = urllib.parse.unquote(parsed_path)
        if decoded_path.startswith("/web/"):
            relative_path = Path(decoded_path.removeprefix("/web/"))
            return str((self._web_dir / relative_path).resolve())
        experiment_prefix = f"/experiments/{self._experiment_name}/"
        if decoded_path.startswith(experiment_prefix):
            relative_path = Path(decoded_path.removeprefix(experiment_prefix))
            return str((self._experiment_dir / relative_path).resolve())
        return str(self._web_dir / "aligner.html")

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def main() -> None:
    arguments = parse_arguments()
    experiment_dir = arguments.experiment_dir.resolve()
    if not (experiment_dir / "experiment.json").exists():
        raise ValueError(f"Missing experiment.json in {experiment_dir}")

    web_dir = (Path(__file__).resolve().parent.parent / "web").resolve()
    experiment_name = experiment_dir.name
    manifest_url = f"/experiments/{experiment_name}/experiment.json"
    query = urllib.parse.urlencode({"manifest": manifest_url, "case": arguments.case})
    url = f"http://localhost:{arguments.port}/web/aligner.html?{query}"

    handler = functools.partial(
        ExperimentRequestHandler,
        web_dir=web_dir,
        experiment_dir=experiment_dir,
        experiment_name=experiment_name,
    )
    with ReusableThreadingTCPServer(("localhost", arguments.port), handler) as server:
        print(f"Serving aligner: {url}")
        webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Stopping aligner server.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the browser alignment tool.")
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--case", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    main()
