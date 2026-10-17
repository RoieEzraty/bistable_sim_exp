from __future__ import annotations

import csv
from pathlib import Path
import numpy as np


def load_pos_force(path: str | Path) -> list[dict[str, object]]:
    """Load position-force dataset rows from CSV.
    used in experiments.sweep_measurement_fixed_origami() and also in plot inside main Meca500 notebook.

    Supported schemas
    -----------------
    1. Measurement/training dataset:
       columns compatible with
       ``x_tip, y_tip, tip_angle_deg, F_x, F_y``
       or equivalent aliases.

       Returns rows of the form:
       ``{"pos": (x, y, theta_deg), "force": (Fx, Fy), "t_unix": ...}``

    2. Predetermined-training dataset:
       columns including
       ``upd_x_tip, upd_y_tip, upd_tip_angle, Fx_meas, Fy_meas, Fx_des, Fy_des``.

       Returns rows of the form:
       ``{"pos_meas": ..., "force_meas": ..., "pos_update": ..., "force_des": ..., "t_unix": ...}``

    Parameters
    ----------
    path : str | Path. Path to CSV file.

    Returns
    -------
    list[dict[str, object]]. Parsed dataset rows.
    """
    rows: list[dict[str, object]] = []

    with Path(path).open(newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)

        if reader.fieldnames is None:
            return rows

        fieldnames = set(reader.fieldnames)

        has_time = ("t" in fieldnames) or ("t_unix" in fieldnames)
        has_deg = "tip_angle_deg" in fieldnames
        has_rad = "tip_angle_rad" in fieldnames

        has_meas = any(col in fieldnames for col in ("pos_x", "x_tip", "Px"))
        has_force = any(col in fieldnames for col in ("force_x", "F_x", "Fx"))

        has_update = "upd_x_tip" in fieldnames
        has_force_meas = ("Fx_meas" in fieldnames) and ("Fy_meas" in fieldnames)
        has_force_des = ("Fx_des" in fieldnames) and ("Fy_des" in fieldnames)

        for record in reader:
            row: dict[str, object] = {}

            if has_meas:
                x = _get_first_in_file(record, ["pos_x", "x_tip", "Px"], name="x")
                y = _get_first_in_file(record, ["pos_y", "y_tip", "Py"], name="y")

                if has_deg and record["tip_angle_deg"] != "":
                    theta_deg = float(record["tip_angle_deg"])
                elif has_rad and record["tip_angle_rad"] != "":
                    theta_deg = float(np.rad2deg(float(record["tip_angle_rad"])))

                if has_update:
                    row["pos_meas"] = (x, y, theta_deg)
                else:
                    row["pos"] = (x, y, theta_deg)

            if has_force:
                Fx = _get_first_in_file(record, ["force_x", "F_x", "Fx"], name="Fx")
                Fy = _get_first_in_file(record, ["force_y", "F_y", "Fy"], name="Fy")
                row["force"] = (Fx, Fy)

            if has_force_meas:
                row["force_meas"] = (float(record["Fx_meas"]), float(record["Fy_meas"]))

            if has_update:
                row["pos_update"] = (float(record["upd_x_tip"]),
                                     float(record["upd_y_tip"]),
                                     float(record["upd_tip_angle"]))

            if has_force_des:
                row["force_des"] = (float(record["Fx_des"]), float(record["Fy_des"]))

            if has_time:
                t = _get_first_in_file(record, ["t", "t_unix"], name="t")
                row["t"] = float(t) if t != "" else None
            else:
                row["t"] = None

            rows.append(row)

    return rows


def _get_first_in_file(r: Mapping[str, Union[str, float, int, None]], keys: Iterable[str], *,
                       name: str = "", allow_missing: bool = False) -> Optional[float]:
    """Extract the first valid scalar value from a row-like mapping.
    Used in load_pos_force()

    Parameters
    ----------
    r             : Mapping[str, str | float | int | None]. Record-like object, typically a CSV row dictionary.
    keys          : Iterable[str]. Ordered candidate keys to search for.
    name          : str, optional. Human-readable field name used only in error reporting.
    allow_missing : bool, optional. If ``True``, return ``None`` when no valid key is found.

    Returns
    -------
    float | None. First successfully parsed scalar value.
    """
    for key in keys:
        if key in r and r[key] not in ("", None):
            return float(r[key])

    # return None if not found
    if allow_missing:
        return None

    raise KeyError(f"None of {list(keys)} found for {name}")