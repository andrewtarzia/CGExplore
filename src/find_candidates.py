#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to find candidates with high fitness from a dataset.

Author: Andrew Tarzia

"""

import sys
import json
import logging
import pandas as pd


from env_set import fourplussix


def survey_existing_data_distributions(
    calculation_dir,
    target_diameter,
):
    all_jsons = list(calculation_dir.glob("*_res.json"))
    logging.info(f"there are {len(all_jsons)} existing data points")

    scores = {}
    for json_file in all_jsons:
        with open(json_file, "r") as f:
            res_dict = json.load(f)

        name = json_file.name.replace("_res.json", "")
        tg_str = name.split("_")[0]

        pore_radius = res_dict["opt_pore_data"]["pore_max_rad"]
        pore_size_diff = (
            abs(target_diameter - pore_radius * 2) / target_diameter
        )
        score = (
            1
            / (
                res_dict["fin_energy"]
                + res_dict["oh6_measure"] * 100
                + pore_size_diff * 100
            )
            * 10
        )
        scores[name] = {
            "score": score,
            "energy": res_dict["fin_energy"],
            "oh6": res_dict["oh6_measure"],
            "pore_radius": pore_radius,
            "tg": tg_str,
        }

    return scores


def main():
    first_line = f"Usage: {__file__}.py"
    if not len(sys.argv) == 2:
        print(f"{first_line} target_diameter")
        sys.exit()
    else:
        target_diameter = float(sys.argv[1])

    calculation_output = fourplussix() / "calculations"

    N = 10

    scores = survey_existing_data_distributions(
        calculation_dir=calculation_output,
        target_diameter=target_diameter,
    )

    scores = pd.DataFrame.from_dict(scores, orient="index")
    sorted_scores = scores.sort_values(by=["score"], ascending=False)
    top_N = sorted_scores.head(N)
    print(top_N)

    viz_command = "pymol "
    for i, row in top_N.iterrows():
        viz_command += f"{i}_opted2.mol "

    print(viz_command)

    # Per topology graph.
    for tg in ("FourPlusSix", "FourPlusSix2"):
        print(tg)
        tg_df = sorted_scores[sorted_scores["tg"] == tg]
        top_N = tg_df.head(N)
        print(top_N)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
