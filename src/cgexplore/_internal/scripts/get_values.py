import argparse
import logging
import pathlib
import pprint

import atomlite

from cgexplore._internal.utilities.databases import AtomliteDatabase


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "database",
        type=pathlib.Path,
        help="path to the database you want to read",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="all",
        help="name of entry in database to get value of",
    )
    parser.add_argument("--path", type=str, help="path to value", nargs="*")
    parser.add_argument(
        "--paths",
        action="store_true",
        help="show paths available in database",
    )
    return parser.parse_args()


def extract_property(path: list[str], properties: dict) -> atomlite.Json:
    if len(path) == 1:
        value = properties[path[0]]
    elif len(path) == 2:  # noqa: PLR2004
        value = properties[path[0]][path[1]]
    elif len(path) == 3:  # noqa: PLR2004
        value = properties[path[0]][path[1]][path[2]]
    elif len(path) == 4:  # noqa: PLR2004
        value = properties[path[0]][path[1]][path[2]][path[3]]
    else:
        msg = f"{path} is too deep ({len(path)})."
        raise RuntimeError(msg)
    return value


def main() -> None:
    args = _parse_args()

    database = AtomliteDatabase(args.database)
    if args.paths:
        keys = set()
        for entry in database.get_entries():
            properties = entry.properties
            for key in properties:
                keys.add(key)
                if isinstance(properties[key], dict):
                    pdict = properties[key]
                    for new_key in pdict:
                        keys.add(f"{key}.{new_key}")
        logging.info("showing all keys in database:")
        pprint.pprint(keys)  # noqa: T203

    elif args.name == "all":
        count = 0
        logging.info(f"showing all values for path: {args.path}")

        for entry in database.get_entries():
            properties = entry.properties
            name = entry.key
            try:
                value = extract_property(path=args.path, properties=properties)
            except KeyError:
                continue

            logging.info(f"energy of {name} is {value} ")
            count += 1
        logging.info(f"showed {count} values")
    else:
        entry = database.get_entry(key=args.name)
        try:
            value = extract_property(
                path=args.path, properties=entry.properties
            )

        except KeyError as ex:
            ex.add_note(
                f"path {args.path} not in database for entry {args.name}"
            )
            raise
        logging.info(f"{args.path} of {args.name} is {value}")


if __name__ == "__main__":
    main()
