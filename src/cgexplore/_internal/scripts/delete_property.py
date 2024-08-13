import argparse
import logging
import pathlib

from cgexplore._internal.utilities.databases import AtomliteDatabase


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "database",
        type=pathlib.Path,
        help="path to the database you want to delete property from",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="all",
        help="name of entry in database to delete value of",
    )
    parser.add_argument(
        "--path",
        type=str,
        help=(
            "path to value to delete (see atomlite documentation for valid "
            "paths)"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    database = AtomliteDatabase(args.database)

    if args.name == "all":
        count = 0
        logging.info("deleting all values for path: %s", args.path)

        for entry in database.get_entries():
            name = entry.key
            database.remove_property(key=name, property_path=args.path)
            count += 1
        logging.info("removed %s properties", count)
    else:
        database.remove_property(key=args.name, property_path=args.path)
        logging.info("removed %s of %s", args.path, args.name)


if __name__ == "__main__":
    main()
