import argparse
import logging
import pathlib

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
        help="name of entry in database to get energy of",
    )
    parser.add_argument("--min", type=float, help="min energy to show")
    parser.add_argument("--max", type=float, help="max energy to show")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    max_energy = 1e24 if args.max is None else args.max
    min_energy = -1e24 if args.min is None else args.min

    database = AtomliteDatabase(args.database)

    if args.name == "all":
        count = 0
        logging.info(f"showing all energies in ({args.min}, {args.max})")
        for entry in database.get_entries():
            properties = entry.properties
            if "energy_per_bb" not in properties:
                continue
            name = entry.key
            energy = properties["energy_per_bb"]
            if energy > min_energy and energy < max_energy:
                logging.info(f"energy of {name} is {round(energy, 3)} kJmol-1")
                count += 1
        logging.info(f"showed {count} energies")
    else:
        entry = database.get_entry(key=args.name)
        energy = entry.properties["energy_per_bb"]
        logging.info(f"energy of {args.name} is {round(energy, 3)} kJmol-1")


if __name__ == "__main__":

    main()
