"""Start Walk Studio: python -m studio [--port 8770] [--no-browser]"""
import argparse

from studio.server import DEFAULT_PORT, serve


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Walk Studio in your web browser.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser.")
    arguments = parser.parse_args()
    serve(arguments.port, open_browser=not arguments.no_browser)


if __name__ == "__main__":
    main()
