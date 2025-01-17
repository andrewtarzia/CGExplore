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
        logging.info("showing all energies in (%s, %s)", args.min, args.max)
        for entry in database.get_entries():
            properties = entry.properties
            if "energy_per_bb" not in properties:
                continue
            name = entry.key
            energy: float = properties["energy_per_bb"]  # type: ignore[assignment]
            if energy > min_energy and energy < max_energy:
                logging.info(
                    "energy of %s is %s kJmol-1",
                    name,
                    round(energy, 3),  # type: ignore[arg-type]
                )
                count += 1
        logging.info("showed %s energies", count)
    else:
        entry = database.get_entry(key=args.name)
        energy = entry.properties["energy_per_bb"]  # type: ignore[assignment]
        logging.info("energy of %s is %s kJmol-1", args.name, round(energy, 3))  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
