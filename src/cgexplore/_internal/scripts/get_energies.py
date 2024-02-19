import argparse
import pathlib


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=pathlib.Path)
    parser.add_argument("csv_results", type=pathlib.Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(args)


if __name__ == "__main__":
    main()
