import argparse
import logging
import pathlib

from cgexplore._internal.utilities.databases import AtomliteDatabase

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "database",
        type=pathlib.Path,
        help="path to the database you want to delete entry from",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="name of entry in database to delete ",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    database = AtomliteDatabase(args.database)

    database.remove_entry(key=args.name)
    logger.info("removed %s", args.name)


if __name__ == "__main__":
    main()
