import ast
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from IPython.display import HTML
from matplotlib import patches
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
from matplotlib.patches import Ellipse, FancyArrowPatch
from collections import Counter

from typing import Tuple, List, Union
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

import colors


def padded_lims(values: List[NDArray[np.float64]], pad_fraction: float = 0.08) -> Tuple[float, float]:
    finite_values = [np.asarray(value, dtype=float).reshape(-1) for value in values]
    finite_values = [value[np.isfinite(value)] for value in finite_values if np.any(np.isfinite(value))]
    if not finite_values:
        return -1.0, 1.0
    combined = np.concatenate(finite_values)
    lim_min = float(np.min(combined))
    lim_max = float(np.max(combined))
    delta = lim_max - lim_min
    if delta <= 0:
        pad = max(abs(lim_min) * pad_fraction, 1.0)
        return lim_min - pad, lim_max + pad
    return lim_min - delta * pad_fraction, lim_max + delta * pad_fraction


def centered_lims(center_values: NDArray[np.float64], values: List[NDArray[np.float64]],
                  pad_fraction: float = 0.08) -> Tuple[float, float]:
    center_arr = np.asarray(center_values, dtype=float).reshape(-1)
    center_arr = center_arr[np.isfinite(center_arr)]
    if len(center_arr) == 0:
        return padded_lims(values, pad_fraction)
    center = float(np.mean(center_arr))
    finite_values = [np.asarray(value, dtype=float).reshape(-1) for value in values]
    finite_values = [value[np.isfinite(value)] for value in finite_values if np.any(np.isfinite(value))]
    if not finite_values:
        return center - 1.0, center + 1.0
    combined = np.concatenate(finite_values)
    half_range = float(np.max(np.abs(combined - center)))
    if half_range <= 0:
        half_range = max(abs(center) * pad_fraction, 1.0)
    else:
        half_range *= 1.0 + pad_fraction
    return center - half_range, center + half_range


def time_step_lims(t_values: NDArray[np.float64], pad: float = 0.25) -> Tuple[float, float]:
    t_arr = np.asarray(t_values, dtype=float).reshape(-1)
    t_arr = t_arr[np.isfinite(t_arr)]
    if len(t_arr) == 0:
        return -pad, pad
    return float(t_arr[0] - pad), float(t_arr[-1] + pad)


def update_position_angle_lims(
    x_update: NDArray[np.float64],
    y_update: NDArray[np.float64],
    angle_update: NDArray[np.float64],
) -> dict:
    angle_arr = np.asarray(angle_update, dtype=float).reshape(-1)
    finite_angle = angle_arr[np.isfinite(angle_arr)]
    center_values = finite_angle[-1:] if len(finite_angle) else angle_arr[-1:]
    return {
        "position": padded_lims([x_update, y_update]),
        "angle": centered_lims(center_values, [angle_update]),
    }


def video_writer_fps(output_path: Path, fps: int, playback_speed: float = 2.0) -> float:
    if output_path.suffix.lower() == ".mp4":
        return float(fps) * float(playback_speed)
    return float(fps)


def crop_frame_edges(
    frame: NDArray[np.uint8],
    left_fraction: float = 0.0,
    bottom_fraction: float = 0.0,
) -> NDArray[np.uint8]:
    height, width = frame.shape[:2]
    left_px = int(round(width * max(0.0, min(left_fraction, 0.85))))
    bottom_px = int(round(height * max(0.0, min(bottom_fraction, 0.85))))
    bottom_px = height - bottom_px
    if left_px >= width - 1 or bottom_px <= 1:
        return frame
    return frame[:bottom_px, left_px:]


def plot_compare_sim_exp_training(exp_file_path: str, sim_file_path: str,
                                  final_t: Optional[int] = None, save: bool = False) -> None:
    """
    Plot experimental and simulated training for comparison of a full chain simulation.

    Parameters
    ----------
    exp_dfs : List[pandas.DataFrame]
        A list of experimental dataframes. Each dataframe must contain
        the columns:
            - "Position (mm)" : tip position in millimeters
            - "Load2 (N)"    : measured load (force) in Newtons

    sim_df : pandas.DataFrame
        Simulation results. Must contain:
            - "x_tip" : simulated tip x-position
            - "Fx"    : simulated x-direction force

    translate_ratio : float
        Factor converting displacement units (e.g., mm). Applied as:
            (x_tip - x_tip_initial) * translate_ratio

    Returns
    -------
    None
        matplotlib figure

    Notes
    -----
    - Experimental curves are smoothed using a Savitzky–Golay filter
      with window length 16 and polynomial order 4.
    - Simulation force is plotted as -Fx to match the experimental sign
      convention.
    """
    colors_lst, red, custom_cmap = colors.color_scheme()
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors_lst)
    font_size = 14

    # read experimental dataframe and extract sizes
    exp_df = pd.read_csv(exp_file_path)

    F_exp_meas = np.vstack([exp_df["F_x_meas"].to_numpy(dtype=float),
                            exp_df["F_y_meas"].to_numpy(dtype=float)])  # shape (2, T)
    F_exp_des = np.vstack([exp_df["F_x_des"].to_numpy(dtype=float),
                           exp_df["F_y_des"].to_numpy(dtype=float)])  # shape (2, T)
    loss_MSE_exp = exp_df["loss_MSE"].to_numpy(dtype=float)
    if "F_err" in exp_df.columns:  # optional force error margin from file
        F_err = np.vstack([exp_df["F_err"].to_numpy(dtype=float),
                           exp_df["F_err"].to_numpy(dtype=float)])  # same error for x and y
    else:
        F_err = None

    # read simulation dataframe
    sim_df = pd.read_csv(sim_file_path)

    F_sim_meas = np.vstack([sim_df["Fx_meas"].to_numpy(dtype=float),
                            sim_df["Fy_meas"].to_numpy(dtype=float)])  # shape (2, T)
    F_sim_des = np.vstack([sim_df["Fx_des"].to_numpy(dtype=float),
                           sim_df["Fy_des"].to_numpy(dtype=float)])  # shape (2, T)
    loss_MSE_sim = sim_df["loss_MSE"].to_numpy(dtype=float)

    # time steps
    T = min(int(F_sim_meas.shape[1]), int(F_exp_meas.shape[1]))
    if final_t is None:
        final_t = T-1
    t = np.arange(T-1, dtype=int)

    # limits
    force_max = np.max([np.max(F_exp_meas), np.max(F_exp_des), np.max(F_sim_meas), np.max(F_sim_des)])
    force_min = np.min([np.min(F_exp_meas), np.min(F_exp_des), np.min(F_sim_meas), np.min(F_sim_des)])
    delta_force = force_max-force_min
    force_lims = [force_min-delta_force*0.15, force_max+delta_force*0.05]
    loss_max = np.max([np.max(loss_MSE_exp), np.max(loss_MSE_sim)])
    loss_min = np.min([np.min(loss_MSE_exp), np.min(loss_MSE_sim)])
    delta_loss = loss_max - loss_min
    loss_lims = [loss_min-delta_loss*0.1, loss_max-delta_loss*0.05]

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex="col", sharey="row", figsize=(8, 3),
                            gridspec_kw={"height_ratios": [1.4, 1]})

    # ====== top: forces ======
    markersize = 10.0

    # left panel - experiment
    axs[0, 0].plot(F_exp_meas[0, 1:], marker=".", linestyle="None", markersize=markersize,
                   color=colors_lst[2], label=r"$F_x$")
    axs[0, 0].plot(t, F_exp_des[0, 1:], marker="None", linestyle="--", markersize=markersize,
                   color=colors_lst[2], label=r"$\hat{F}_x$")

    axs[0, 0].plot(F_exp_meas[1, 1:], marker=".", linestyle="None", markersize=markersize,
                   color=colors_lst[1], label=r"$F_y$")
    axs[0, 0].plot(t, F_exp_des[1, 1:], marker="None", linestyle="--", markersize=markersize,
                   color=colors_lst[1], label=r"$\hat{F}_y$")

    if F_err is not None:
        axs[0, 0].fill_between(t, F_exp_meas[0, 1:] - F_err[0, 1:],
                               F_exp_meas[0, 1:] + F_err[0, 1:],
                               color=colors_lst[2], alpha=0.5, linewidth=0)

        axs[0, 0].fill_between(t, F_exp_meas[1, 1:] - F_err[1, 1:],
                               F_exp_meas[1, 1:] + F_err[1, 1:],
                               color=colors_lst[1], alpha=0.5, linewidth=0)

    # shared axis
    axs[0, 0].set_ylabel(r"$F\left[mN\right]$", fontsize=font_size)
    axs[0, 0].set_ylim(force_lims)

    # right panel - simulation
    axs[0, 1].plot(F_sim_meas[0, 1:], color=colors_lst[2], label=r"$F_x$")
    axs[0, 1].plot(F_sim_des[0, 1:], color=colors_lst[2], linestyle="--", label=r"$\hat{F}_x$")
    axs[0, 1].plot(F_sim_meas[1, 1:], color=colors_lst[1], label=r"$F_y$")
    axs[0, 1].plot(F_sim_des[1, 1:], color=colors_lst[1], linestyle="--", label=r"$\hat{F}_y$")

    # ====== bottom: MSE loss ======
    # left panel - experiment
    axs[1, 0].plot(loss_MSE_exp[1:], marker=".", linestyle="None", markersize=markersize, color=colors_lst[0], 
                   label=r"$\mathcal{L}$")
    axs[1, 0].plot(t, np.zeros(len(t)), color=colors_lst[0], linestyle="--")

    axs[1, 0].set_xlabel("t", fontsize=font_size)
    axs[1, 0].set_ylabel("Loss", fontsize=font_size)
    axs[1, 0].set_ylim(loss_lims)

    # right panel - simulation
    axs[1, 1].plot(loss_MSE_sim[1:], color=colors_lst[0], label=r"$\mathcal{L}$")
    axs[1, 1].plot(t, np.zeros(len(t)), color=colors_lst[0], linestyle="--")
    axs[1, 1].set_xlabel("t", fontsize=font_size)

    # ====== titles ======
    axs[0, 0].set_title("Experiment", fontsize=font_size)
    axs[0, 1].set_title("Simulation", fontsize=font_size)

    # ====== legends ======
    legend_kw = dict(loc="best", ncol=2, fontsize=11.5, handlelength=1.2,
                     handletextpad=0.3, columnspacing=0.18, borderpad=0.08,
                     labelspacing=0.08, markerscale=0.8, frameon=True)
    axs[0, 0].legend(**legend_kw)
    axs[0, 1].legend(**legend_kw)
    axs[1, 0].legend(**legend_kw)
    axs[1, 1].legend(**legend_kw)

    # ====== locator + layout ======
    axs[-1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[-1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    if save:
        plt.savefig("importants.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_sim_or_exp(file_path: str, mod: str = "summary", final_t: Optional[int] = None, save: bool = False) -> None:
    """
    Plot the main signals from a single simulation/experiment update CSV.

    Expected columns:
        - upd_x_tip, upd_y_tip, upd_tip_angle
        - buckle_arr_meas, buckle_arr_update
        - Fx_update, Fy_update
        - loss_MSE

    With ``mod="pos"``, the force panel shows only update forces. Other
    modes also include measured and desired force references when available.
    """
    mod = mod.lower()
    colors_lst, red, custom_cmap = colors.color_scheme()
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors_lst)
    font_size = 13

    df = pd.read_csv(file_path)
    required_cols = [
        "upd_x_tip", "upd_y_tip", "upd_tip_angle",
        "buckle_arr_meas", "buckle_arr_update",
        "Fx_update", "Fy_update", "loss_MSE",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in {file_path}: {missing_cols}")

    if final_t is not None:
        if "t" in df.columns:
            df = df[df["t"] <= final_t].copy()
        else:
            df = df.iloc[:final_t + 1].copy()

    if "t" in df.columns:
        df = df[df["t"] >= 1].copy()
    else:
        df = df.iloc[1:].copy()

    if df.empty:
        raise ValueError("No rows available to plot after applying final_t.")

    t = df["t"].to_numpy(dtype=float) if "t" in df.columns else np.arange(len(df), dtype=float)

    def parse_buckle_array(value: object) -> NDArray[np.float64]:
        try:
            return np.asarray(ast.literal_eval(str(value)), dtype=float).reshape(-1)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Could not parse buckle array: {value!r}") from exc

    def stack_buckle_arrays(series: pd.Series) -> NDArray[np.float64]:
        arrays = [parse_buckle_array(value) for value in series]
        max_len = max(len(arr) for arr in arrays)
        stacked = np.full((len(arrays), max_len), np.nan, dtype=float)
        for row, arr in enumerate(arrays):
            stacked[row, :len(arr)] = arr
        return stacked

    buckle_update = stack_buckle_arrays(df["buckle_arr_update"])

    fig, axs = plt.subplots(
        nrows=4, ncols=1, sharex=True, figsize=(9, 7.0),
        gridspec_kw={"height_ratios": [1.35, 0.9, 1.15, 0.9]},
    )
    fig.suptitle(Path(file_path).name, fontsize=font_size + 1)

    # ====== update position and angle ======
    ax_update = axs[0]
    ax_angle = ax_update.twinx()
    ax_update.plot(t, df["upd_x_tip"].to_numpy(dtype=float), color=colors_lst[2], label=r"$x_{tip}$ update")
    ax_update.plot(t, df["upd_y_tip"].to_numpy(dtype=float), color=colors_lst[1], label=r"$y_{tip}$ update")
    ax_angle.plot(t, df["upd_tip_angle"].to_numpy(dtype=float), color=red, label=r"$\theta_{tip}$ update")
    ax_update.set_ylabel("Position", fontsize=font_size)
    ax_angle.set_ylabel("Angle", fontsize=font_size)

    pos_lines, pos_labels = ax_update.get_legend_handles_labels()
    angle_lines, angle_labels = ax_angle.get_legend_handles_labels()
    ax_update.legend(pos_lines + angle_lines, pos_labels + angle_labels, loc="best", ncol=3, fontsize=10)

    # ====== buckle states ======
    buckle_cmap = ListedColormap([colors_lst[1], colors_lst[4]])
    buckle_cmap.set_bad("#e5e5e5")
    buckle_norm = BoundaryNorm([-1.5, 0, 1.5], buckle_cmap.N)

    def plot_buckle_heatmap(ax: plt.Axes, buckle_values: NDArray[np.float64], title: str) -> None:
        ax.imshow(
            buckle_values.T,
            aspect="auto",
            interpolation="nearest",
            cmap=buckle_cmap,
            norm=buckle_norm,
            extent=[t[0] - 0.5, t[-1] + 0.5, buckle_values.shape[1] - 0.5, -0.5],
        )
        ax.set_ylabel(title, fontsize=font_size)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plot_buckle_heatmap(axs[1], buckle_update, "Buckle update")

    buckle_handles = [
        patches.Patch(color=colors_lst[1], label="-1"),
        patches.Patch(color=colors_lst[4], label="1"),
    ]
    axs[1].legend(handles=buckle_handles, loc="upper right", ncol=2, fontsize=10,
                  handlelength=1.0, columnspacing=0.8, frameon=True)

    # ====== forces during update ======
    ax_force = axs[2]
    ax_force.plot(t, df["Fx_update"].to_numpy(dtype=float), color=colors_lst[2], label=r"$F_x$ update")
    ax_force.plot(t, df["Fy_update"].to_numpy(dtype=float), color=colors_lst[1], label=r"$F_y$ update")

    if mod != "pos":
        optional_force_cols = [
            ("Fx_meas", colors_lst[2], ":", r"$F_x$ meas."),
            ("Fy_meas", colors_lst[1], ":", r"$F_y$ meas."),
            ("Fx_des", colors_lst[2], "--", r"$F_x$ des."),
            ("Fy_des", colors_lst[1], "--", r"$F_y$ des."),
        ]
        for col, color, linestyle, label in optional_force_cols:
            if col in df.columns:
                ax_force.plot(t, df[col].to_numpy(dtype=float), color=color, linestyle=linestyle,
                              alpha=0.75, label=label)

    ax_force.set_ylabel(r"$F\,\left[mN\right]$", fontsize=font_size)
    ax_force.legend(loc="best", ncol=2 if mod == "pos" else 3, fontsize=10)

    # ====== loss ======
    ax_loss = axs[3]
    loss_mse = df["loss_MSE"].to_numpy(dtype=float)
    ax_loss.plot(t, loss_mse, color=colors_lst[0], label=r"$\mathcal{L}$")
    ax_loss.plot(t, np.zeros(len(t)), color=colors_lst[0], linestyle="--")
    loss_min = np.min(loss_mse)
    loss_max = np.max(loss_mse)
    delta_loss = loss_max - loss_min
    if delta_loss > 0:
        ax_loss.set_ylim(loss_min - delta_loss * 0.1, loss_max + delta_loss * 0.05)
    ax_loss.set_xlabel("t", fontsize=font_size)
    ax_loss.set_ylabel(r"$\mathcal{L}$", fontsize=font_size)
    ax_loss.legend(loc="best", fontsize=10)
    ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    if save:
        output_path = f"{Path(file_path).stem}_important_sizes.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_force_along_traj(
    csv_file_path: Union[str, Path],
    vid_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    initial_time_s: float = 0.0,
    final_time_s: Optional[float] = None,
    mean_line_mode: str = "des",
    csv_file_path_des: Optional[Union[str, Path]] = None,
    fps: int = 2,
    save: bool = True,
    dpi: int = 180,
    video_crop_left_fraction: float = 0.12,
    video_crop_bottom_fraction: float = 0.12,
) -> Path:
    """
    Create a video with the measured trajectory video on the left and
    ``F_x``/``F_y`` as a function of ``y_tip`` on the right.

    The force curves start growing at ``initial_time_s`` in the source video and
    finish at ``final_time_s``. Once all trajectory points are shown, mean-force
    lines are overlaid: dotted for ``mean_line_mode="des"`` and solid for
    ``mean_line_mode="meas"``. When ``mean_line_mode="meas"`` and
    ``csv_file_path_des`` is supplied, dotted desired mean-force lines are
    overlaid too.
    """
    csv_path = Path(csv_file_path)
    video_path = Path(vid_path)
    if output_path is None:
        output_path = csv_path.with_name(f"{csv_path.stem}_force_along_traj.mp4")
    output_path = Path(output_path)

    if not save:
        raise ValueError("plot_force_along_traj creates a video file; call it with save=True.")
    mean_line_mode = mean_line_mode.lower()
    if mean_line_mode not in {"des", "meas"}:
        raise ValueError('mean_line_mode must be either "des" or "meas".')
    mean_linestyle = ":" if mean_line_mode == "des" else "-"

    col_candidates = {
        "y": ("y_tip", "y", "Position_y"),
        "fx": ("F_x", "Fx", "F_x_meas", "Fx_meas"),
        "fy": ("F_y", "Fy", "F_y_meas", "Fy_meas"),
    }

    def find_col(df_in: pd.DataFrame, kind: str, path: Path) -> str:
        for col in col_candidates[kind]:
            if col in df_in.columns:
                return col
        raise KeyError(f"Missing {kind} column in {path}. Tried {col_candidates[kind]}.")

    def read_force_traj(path: Path) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        df_in = pd.read_csv(path)
        y_col = find_col(df_in, "y", path)
        fx_col = find_col(df_in, "fx", path)
        fy_col = find_col(df_in, "fy", path)
        df_in = df_in[[y_col, fx_col, fy_col]].dropna().reset_index(drop=True)
        if df_in.empty:
            raise ValueError(f"No valid force trajectory rows found in {path}.")
        return (
            df_in[y_col].to_numpy(dtype=float),
            df_in[fx_col].to_numpy(dtype=float),
            df_in[fy_col].to_numpy(dtype=float),
        )

    y, fx, fy = read_force_traj(csv_path)
    fx_mean = float(np.mean(fx))
    fy_mean = float(np.mean(fy))
    des_means = None
    if csv_file_path_des is not None:
        _, fx_des, fy_des = read_force_traj(Path(csv_file_path_des))
        des_means = (float(np.mean(fx_des)), float(np.mean(fy_des)))

    colors_lst, red, custom_cmap = colors.color_scheme()

    x_lims = padded_lims([y])
    force_lim_values = [fx, fy, np.asarray([fx_mean, fy_mean], dtype=float)]
    if des_means is not None:
        force_lim_values.append(np.asarray(des_means, dtype=float))
    force_lims = padded_lims(force_lim_values)

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Saving MP4 requires opencv-python (cv2).") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    input_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(input_fps) or input_fps <= 0:
        input_fps = float(fps)
    sample_every = max(1, int(round(input_fps / fps)))
    initial_time_s = max(0.0, float(initial_time_s))
    if final_time_s is None:
        final_time_s = initial_time_s + max(len(y) - 1, 1)
    final_time_s = float(final_time_s)
    if final_time_s < initial_time_s:
        raise ValueError("final_time_s must be greater than or equal to initial_time_s.")

    fig = plt.figure(figsize=(11.5, 5.2), dpi=dpi, constrained_layout=True)

    def draw_frame(video_frame_bgr: NDArray[np.uint8], plot_idx: int, show_means: bool) -> NDArray[np.uint8]:
        fig.clear()
        grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
        ax_video = fig.add_subplot(grid[0, 0])
        ax_force = fig.add_subplot(grid[0, 1])

        video_frame_bgr = crop_frame_edges(
            video_frame_bgr,
            left_fraction=video_crop_left_fraction,
            bottom_fraction=video_crop_bottom_fraction,
        )
        video_frame_rgb = cv2.cvtColor(video_frame_bgr, cv2.COLOR_BGR2RGB)
        ax_video.imshow(video_frame_rgb)
        ax_video.set_xticks([])
        ax_video.set_yticks([])
        for spine in ax_video.spines.values():
            spine.set_visible(False)

        upto = max(0, min(plot_idx + 1, len(y)))
        current_suffix = "des" if mean_line_mode == "des" else "meas"
        ax_force.plot(y[:upto], fx[:upto], color=colors_lst[2], marker="o")
        ax_force.plot(y[:upto], fy[:upto], color=colors_lst[1], marker="o")
        if mean_line_mode == "meas" and des_means is not None:
            ax_force.axhline(des_means[0], color=colors_lst[2], linestyle=":", linewidth=2.0)
            ax_force.axhline(des_means[1], color=colors_lst[1], linestyle=":", linewidth=2.0)
        if show_means:
            ax_force.axhline(fx_mean, color=colors_lst[2], linestyle=mean_linestyle, linewidth=2.0)
            ax_force.axhline(fy_mean, color=colors_lst[1], linestyle=mean_linestyle, linewidth=2.0)

        ax_force.set_xlim(x_lims)
        ax_force.set_ylim(force_lims)
        ax_force.set_xlabel(r"$y_{tip}$")
        ax_force.set_ylabel(r"$F\,\left[mN\right]$")
        legend_handles = [
            Line2D([0], [0], color=colors_lst[2], linestyle=mean_linestyle, marker="o",
                   label=rf"$F_{{x}}\ \mathrm{{{'measured' if current_suffix == 'meas' else 'desired'}}}$"),
            Line2D([0], [0], color=colors_lst[1], linestyle=mean_linestyle, marker="o",
                   label=rf"$F_{{y}}\ \mathrm{{{'measured' if current_suffix == 'meas' else 'desired'}}}$"),
        ]
        if mean_line_mode == "meas" and des_means is not None:
            legend_handles.extend([
                Line2D([0], [0], color=colors_lst[2], linestyle=":",
                       label=r"$F_{x}\ \mathrm{desired}$"),
                Line2D([0], [0], color=colors_lst[1], linestyle=":",
                       label=r"$F_{y}\ \mathrm{desired}$"),
            ])
        ax_force.legend(handles=legend_handles, loc="best", fontsize=9)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        return rgba[:, :, :3].copy()

    def plot_index_for_video_time(video_time_s: float) -> Tuple[int, bool]:
        if video_time_s < initial_time_s:
            return -1, False
        if final_time_s == initial_time_s:
            return len(y) - 1, True
        if video_time_s >= final_time_s:
            return len(y) - 1, True
        progress = (video_time_s - initial_time_s) / (final_time_s - initial_time_s)
        plot_idx = int(np.floor(progress * (len(y) - 1)))
        return plot_idx, False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    output_frame_idx = 0
    input_frame_idx = 0
    last_frame_bgr = None

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            last_frame_bgr = frame_bgr
            if input_frame_idx % sample_every != 0:
                input_frame_idx += 1
                continue

            video_time_s = input_frame_idx / input_fps
            plot_idx, show_means = plot_index_for_video_time(video_time_s)
            frame_rgb = draw_frame(frame_bgr, plot_idx, show_means)
            height, width = frame_rgb.shape[:2]
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*("mp4v" if output_path.suffix.lower() == ".mp4" else "XVID"))
                writer = cv2.VideoWriter(str(output_path), fourcc, video_writer_fps(output_path, fps), (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer for {output_path}.")

            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            output_frame_idx += 1
            input_frame_idx += 1

        if writer is None:
            raise RuntimeError(f"No frames were read from {video_path}.")

        if last_frame_bgr is not None:
            final_rgb = draw_frame(last_frame_bgr, len(y) - 1, True)
            final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
            for _ in range(max(1, int(round(3 * fps)))):
                writer.write(final_bgr)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        plt.close(fig)

    return output_path


def plot_force_during_zero_force(
    csv_file_path: Union[str, Path],
    vid_path: Union[str, Path],
    initial_time_vid: float,
    final_time_vid: float,
    final_time_csv: Optional[Union[int, float]] = None,
    fps: int = 2,
    mean_line_mode: str = "meas",
    csv_file_path_des: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    save: bool = True,
    dpi: int = 180,
    video_crop_left_fraction: float = 0.12,
    video_crop_bottom_fraction: float = 0.12,
) -> Path:
    """
    Create a video with the source video on the left and force/update values on the right.

    The source video plays for its full duration. Force plotting starts at
    ``initial_time_vid`` and reaches ``final_time_csv`` by ``final_time_vid``.
    The final plotted force sample is emphasized with a larger marker.
    """
    csv_path = Path(csv_file_path)
    video_path = Path(vid_path)
    if output_path is None:
        output_path = csv_path.with_name(f"{csv_path.stem}_zero_force_vid.mp4")
    output_path = Path(output_path)

    if not save:
        raise ValueError("plot_force_during_zero_force creates a video file; call it with save=True.")
    mean_line_mode = mean_line_mode.lower()
    if mean_line_mode not in {"des", "meas"}:
        raise ValueError('mean_line_mode must be either "des" or "meas".')

    col_candidates = {
        "t": ("t", "time", "time_s"),
        "fx": ("F_x", "Fx", "F_x_meas", "Fx_meas"),
        "fy": ("F_y", "Fy", "F_y_meas", "Fy_meas"),
        "x": ("upd_x_tip", "x_tip", "x", "update_x"),
        "y": ("upd_y_tip", "y_tip", "y", "update_y"),
        "angle": ("upd_tip_angle", "angle_tip", "theta", "angle", "update_angle"),
        "x_des": ("x_desired", "x_rest_des", "x_des", "x_tip_des", "x"),
        "y_des": ("y_desired", "y_rest_des", "y_des", "y_tip_des", "y"),
        "angle_des": ("theta_desired", "angle_desired", "theta_rest_des", "theta_des", "angle_des", "tip_angle_des", "theta"),
    }

    def find_col(df_in: pd.DataFrame, kind: str, path: Path, required: bool = True) -> Optional[str]:
        for col in col_candidates[kind]:
            if col in df_in.columns:
                return col
        if required:
            raise KeyError(f"Missing {kind} column in {path}. Tried {col_candidates[kind]}.")
        return None

    def read_zero_force_data(path: Path) -> Tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
        NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
    ]:
        df_in = pd.read_csv(path)
        fx_col = find_col(df_in, "fx", path)
        fy_col = find_col(df_in, "fy", path)
        x_col = find_col(df_in, "x", path)
        y_col = find_col(df_in, "y", path)
        angle_col = find_col(df_in, "angle", path)
        t_col = find_col(df_in, "t", path, required=False)
        use_cols = [fx_col, fy_col, x_col, y_col, angle_col]
        if t_col is not None:
            use_cols = [t_col] + use_cols
        df_in = df_in[use_cols].dropna().reset_index(drop=True)
        if df_in.empty:
            raise ValueError(f"No valid zero-force rows found in {path}.")
        t_values = (
            df_in[t_col].to_numpy(dtype=float)
            if t_col is not None else np.arange(len(df_in), dtype=float)
        )
        return (
            t_values,
            df_in[fx_col].to_numpy(dtype=float),
            df_in[fy_col].to_numpy(dtype=float),
            df_in[x_col].to_numpy(dtype=float),
            df_in[y_col].to_numpy(dtype=float),
            df_in[angle_col].to_numpy(dtype=float),
        )

    def read_desired_position_data(path: Path) -> Optional[Tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
    ]]:
        df_in = pd.read_csv(path)
        x_des_col = find_col(df_in, "x_des", path, required=False)
        y_des_col = find_col(df_in, "y_des", path, required=False)
        angle_des_col = find_col(df_in, "angle_des", path, required=False)
        if x_des_col is None or y_des_col is None or angle_des_col is None:
            return None
        t_col = find_col(df_in, "t", path, required=False)
        use_cols = [x_des_col, y_des_col, angle_des_col]
        if t_col is not None:
            use_cols = [t_col] + use_cols
        df_in = df_in[use_cols].dropna().reset_index(drop=True)
        if df_in.empty:
            return None
        t_values = (
            df_in[t_col].to_numpy(dtype=float)
            if t_col is not None else np.arange(len(df_in), dtype=float)
        )
        return (
            t_values,
            df_in[x_des_col].to_numpy(dtype=float),
            df_in[y_des_col].to_numpy(dtype=float),
            df_in[angle_des_col].to_numpy(dtype=float),
        )

    t_csv, fx, fy, upd_x, upd_y, upd_angle = read_zero_force_data(csv_path)
    if final_time_csv is not None:
        final_time_csv = float(final_time_csv)
        if "t" in pd.read_csv(csv_path, nrows=0).columns:
            keep = t_csv <= final_time_csv
            if not np.any(keep):
                raise ValueError("No CSV rows remain after applying final_time_csv.")
            t_csv, fx, fy = t_csv[keep], fx[keep], fy[keep]
            upd_x, upd_y, upd_angle = upd_x[keep], upd_y[keep], upd_angle[keep]
        else:
            final_count = max(1, min(len(t_csv), int(final_time_csv)))
            t_csv, fx, fy = t_csv[:final_count], fx[:final_count], fy[:final_count]
            upd_x, upd_y, upd_angle = upd_x[:final_count], upd_y[:final_count], upd_angle[:final_count]

    desired_position = (
        read_desired_position_data(Path(csv_file_path_des))
        if csv_file_path_des is not None else None
    )

    line_style = ":" if mean_line_mode == "des" else "-"
    colors_lst, red, custom_cmap = colors.color_scheme()

    x_lims = time_step_lims(t_csv)
    force_lims = padded_lims([fx, fy])
    update_pos_values = [upd_x, upd_y]
    update_angle_values = [upd_angle]
    if desired_position is not None:
        _, x_des, y_des, angle_des = desired_position
        update_pos_values.extend([x_des, y_des])
        update_angle_values.append(angle_des)
    update_lims = {
        "position": padded_lims(update_pos_values),
        "angle": centered_lims(update_angle_values[-1][-1:], update_angle_values),
    }

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Saving MP4 requires opencv-python (cv2).") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    input_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(input_fps) or input_fps <= 0:
        input_fps = float(fps)
    sample_every = max(1, int(round(input_fps / fps)))
    initial_time_vid = max(0.0, float(initial_time_vid))
    final_time_vid = float(final_time_vid)
    if final_time_vid < initial_time_vid:
        raise ValueError("final_time_vid must be greater than or equal to initial_time_vid.")

    fig = plt.figure(figsize=(11.5, 6.2), dpi=dpi, constrained_layout=True)

    def plot_index_for_video_time(video_time_s: float) -> int:
        if video_time_s < initial_time_vid:
            return -1
        if final_time_vid == initial_time_vid or video_time_s >= final_time_vid:
            return len(t_csv) - 1
        progress = (video_time_s - initial_time_vid) / (final_time_vid - initial_time_vid)
        return int(np.floor(progress * (len(t_csv) - 1)))

    def draw_frame(video_frame_bgr: NDArray[np.uint8], plot_idx: int) -> NDArray[np.uint8]:
        fig.clear()
        grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1, 1])
        ax_video = fig.add_subplot(grid[:, 0])
        ax_force = fig.add_subplot(grid[0, 1])
        ax_update = fig.add_subplot(grid[1, 1])
        ax_update_angle = ax_update.twinx()

        video_frame_bgr = crop_frame_edges(
            video_frame_bgr,
            left_fraction=video_crop_left_fraction,
            bottom_fraction=video_crop_bottom_fraction,
        )
        ax_video.imshow(cv2.cvtColor(video_frame_bgr, cv2.COLOR_BGR2RGB))
        ax_video.set_xticks([])
        ax_video.set_yticks([])
        for spine in ax_video.spines.values():
            spine.set_visible(False)

        upto = max(0, min(plot_idx + 1, len(t_csv)))
        if upto > 0:
            ax_force.plot(t_csv[:upto], fx[:upto], color=colors_lst[2],
                          linestyle=line_style, marker="o", markersize=4)
            ax_force.plot(t_csv[:upto], fy[:upto], color=colors_lst[1],
                          linestyle=line_style, marker="o", markersize=4)
            # if plot_idx >= len(t_csv) - 1:
            #     ax_force.plot(t_csv[-1], fx[-1], color=colors_lst[2], marker="o",
            #                   markersize=10, linestyle="None")
            #     ax_force.plot(t_csv[-1], fy[-1], color=colors_lst[1], marker="o",
            #                   markersize=10, linestyle="None")

        ax_force.set_xlim(x_lims)
        ax_force.set_ylim(force_lims)
        ax_force.set_xlabel("step" if final_time_csv is not None else "CSV sample")
        ax_force.set_ylabel(r"$F\,\left[mN\right]$")
        legend_handles = [
            Line2D([0], [0], color=colors_lst[2], linestyle=line_style, marker="o",
                   label=r"$F_{x}$"),
            Line2D([0], [0], color=colors_lst[1], linestyle=line_style, marker="o",
                   label=r"$F_{y}$"),
        ]
        ax_force.legend(handles=legend_handles, loc="best", fontsize=9)

        if upto > 0:
            ax_update.plot(t_csv[:upto], upd_x[:upto], color=colors_lst[2], marker="o",
                           label=r"$x$ update")
            ax_update.plot(t_csv[:upto], upd_y[:upto], color=colors_lst[1], marker="o",
                           label=r"$y$ update")
            ax_update_angle.plot(t_csv[:upto], upd_angle[:upto], color=red, marker="o",
                                 label=r"$\theta$ update")
            if plot_idx >= len(t_csv) - 1:
                ax_update.plot(t_csv[-1], upd_x[-1], color=colors_lst[2], marker="o",
                              markersize=10, linestyle="None")
                ax_update.plot(t_csv[-1], upd_y[-1], color=colors_lst[1], marker="o",
                              markersize=10, linestyle="None")
                ax_update_angle.plot(t_csv[-1], upd_angle[-1], color=red, marker="o",
                                     markersize=10, linestyle="None")
        if desired_position is not None:
            t_des, x_des, y_des, angle_des = desired_position
            ax_update.axhline(x_des[-1], color=colors_lst[2], linestyle="--", linewidth=2.0)
            ax_update.axhline(y_des[-1], color=colors_lst[1], linestyle="--", linewidth=2.0)
            ax_update_angle.axhline(angle_des[-1], color=red, linestyle="--", linewidth=2.0)
        ax_update.set_xlim(x_lims)
        ax_update.set_ylim(update_lims["position"])
        ax_update_angle.set_ylim(update_lims["angle"])
        ax_update.set_xlabel("step" if final_time_csv is not None else "CSV sample", fontsize=10)
        ax_update.set_ylabel(r"pos $\left[m\right]$", fontsize=10)
        ax_update_angle.set_ylabel(r"angle $\left[\degree\right]$", fontsize=10)
        if desired_position is not None:
            update_handles = [Line2D([0], [0], color=colors_lst[2], linestyle="-", marker="o", label=r"$x$"),
                            Line2D([0], [0], color=colors_lst[2], linestyle="--", label=r"$x$ des."),
                            Line2D([0], [0], color=colors_lst[1], linestyle="-", marker="o", label=r"$y$"),
                            Line2D([0], [0], color=colors_lst[1], linestyle="--", label=r"$y$ des."),
                            Line2D([0], [0], color=red, linestyle="-", marker="o", label=r"$\theta$"),
                            Line2D([0], [0], color=red, linestyle="--", label=r"$\theta$ des.")]
        else:
            update_handles = [Line2D([0], [0], color=colors_lst[2], linestyle="-", marker="o", label=r"$x$"),
                            Line2D([0], [0], color=colors_lst[1], linestyle="-", marker="o", label=r"$y$"),
                            Line2D([0], [0], color=red, linestyle="-", marker="o", label=r"$\theta$")]
        ax_update.legend(handles=update_handles, loc="best", ncol=3, fontsize=8)

        for ax in (ax_force, ax_update):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(labelsize=8)
        ax_update_angle.tick_params(labelsize=8)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        return rgba[:, :, :3].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    input_frame_idx = 0
    last_frame_bgr = None

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            last_frame_bgr = frame_bgr
            if input_frame_idx % sample_every != 0:
                input_frame_idx += 1
                continue

            video_time_s = input_frame_idx / input_fps
            plot_idx = plot_index_for_video_time(video_time_s)
            frame_rgb = draw_frame(frame_bgr, plot_idx)
            height, width = frame_rgb.shape[:2]
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*("mp4v" if output_path.suffix.lower() == ".mp4" else "XVID"))
                writer = cv2.VideoWriter(str(output_path), fourcc, video_writer_fps(output_path, fps), (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer for {output_path}.")
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            input_frame_idx += 1

        if writer is None:
            raise RuntimeError(f"No frames were read from {video_path}.")
        if last_frame_bgr is not None:
            final_rgb = draw_frame(last_frame_bgr, len(t_csv) - 1)
            final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
            for _ in range(max(1, int(round(3 * fps)))):
                writer.write(final_bgr)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        plt.close(fig)

    return output_path


def training_force_data_and_vid(
    training_dir: Union[str, Path] = r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI\Meca500\data\training\June19_fullTraining",
    csv_file_path: Optional[Union[str, Path]] = None,
    image_dir: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    mod: str = "summary",
    final_t: Optional[int] = None,
    fps: int = 2,
    dpi: int = 180,
) -> List[Path]:
    """
    Create video(s) with the training image on the left and CSV plots up to the
    corresponding training time on the right.

    If ``csv_file_path`` and ``image_dir`` are not supplied, all matching pairs
    in ``training_dir`` are rendered. Pairing is based on the common prefix, for
    example ``0001to0000_pos1.csv`` with ``0001to0000Training_Full``.

    The CSV starts from ``t=1``. If the number of images does not equal the
    number of plotted time steps, images are sampled uniformly across the sorted
    image list.
    """
    mod = mod.lower()
    training_dir = Path(training_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def natural_key(path: Path) -> List[Union[int, str]]:
        parts = re.split(r"(\d+)", path.name.lower())
        return [int(part) if part.isdigit() else part for part in parts]

    def image_files_in_dir(path: Path) -> List[Path]:
        return sorted(
            [item for item in path.iterdir() if item.is_file() and item.suffix.lower() in image_exts],
            key=natural_key,
        )

    def find_image_dir_for_csv(csv_path: Path, fallback_dirs: List[Path]) -> Optional[Path]:
        if image_files_in_dir(csv_path.parent):
            return csv_path.parent

        child_dirs = [path for path in csv_path.parent.iterdir() if path.is_dir()]
        child_dirs_with_images = [path for path in child_dirs if image_files_in_dir(path)]
        if child_dirs_with_images:
            return sorted(child_dirs_with_images, key=natural_key)[0]

        csv_prefix = csv_path.stem.split("_pos", 1)[0]
        matching_dirs = [path for path in fallback_dirs if path.name.startswith(csv_prefix)]
        if matching_dirs:
            return sorted(matching_dirs, key=natural_key)[0]
        return None

    def find_pairs() -> List[Tuple[Path, Path, Path]]:
        if csv_file_path is not None or image_dir is not None:
            if csv_file_path is None or image_dir is None:
                raise ValueError("csv_file_path and image_dir must be supplied together.")
            csv_path = Path(csv_file_path)
            img_dir = Path(image_dir)
            out_path = Path(output_path) if output_path is not None else csv_path.with_name(f"{csv_path.stem}_training_vid.mp4")
            return [(csv_path, img_dir, out_path)]

        csv_paths = [
            path for path in training_dir.rglob("*.csv")
            if "dataset" not in path.stem.lower() and re.search(r"[01]+to[01]+", path.stem)
        ]
        csv_paths = sorted(csv_paths, key=natural_key)
        image_dirs = [
            path for path in training_dir.rglob("*")
            if path.is_dir() and "training" in path.name.lower() and image_files_in_dir(path)
        ]
        pairs = []
        for csv_path in csv_paths:
            img_dir = find_image_dir_for_csv(csv_path, image_dirs)
            if img_dir is None:
                continue
            out_path = csv_path.with_name(f"{csv_path.stem}_training_data_vid.mp4")
            pairs.append((csv_path, img_dir, out_path))
        return pairs

    def prepare_dataframe(csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        if final_t is not None:
            if "t" in df.columns:
                df = df[df["t"] <= final_t].copy()
            else:
                df = df.iloc[:final_t + 1].copy()
        if "t" in df.columns:
            df = df[df["t"] >= 1].copy()
        else:
            df = df.iloc[1:].copy()
        if df.empty:
            raise ValueError(f"No rows available to plot in {csv_path} after applying final_t.")
        return df.reset_index(drop=True)

    def parse_buckle_array(value: object) -> NDArray[np.float64]:
        try:
            return np.asarray(ast.literal_eval(str(value)), dtype=float).reshape(-1)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Could not parse buckle array: {value!r}") from exc

    def stack_buckle_arrays(series: pd.Series) -> NDArray[np.float64]:
        arrays = [parse_buckle_array(value) for value in series]
        max_len = max(len(arr) for arr in arrays)
        stacked = np.full((len(arrays), max_len), np.nan, dtype=float)
        for row, arr in enumerate(arrays):
            stacked[row, :len(arr)] = arr
        return stacked

    def build_axis_lims(df_full: pd.DataFrame) -> dict:
        t_full = df_full["t"].to_numpy(dtype=float) if "t" in df_full.columns else np.arange(1, len(df_full) + 1, dtype=float)
        measured_desired_cols = [col for col in ["F_x_meas", "F_y_meas", "F_x_des", "F_y_des"] if col in df_full.columns]
        update_cols = ["F_x_update", "F_y_update"]
        update_lims = update_position_angle_lims(
            df_full["upd_x_tip"].to_numpy(dtype=float),
            df_full["upd_y_tip"].to_numpy(dtype=float),
            df_full["upd_tip_angle"].to_numpy(dtype=float),
        )

        lims = {
            "x": time_step_lims(t_full),
            "position": update_lims["position"],
            "angle": update_lims["angle"],
            "measured_desired_force": padded_lims([df_full[col].to_numpy(dtype=float) for col in measured_desired_cols]),
            "update_force": padded_lims([df_full[col].to_numpy(dtype=float) for col in update_cols]),
            "loss": padded_lims([
                df_full["loss_MSE"].to_numpy(dtype=float),
                np.zeros(len(df_full), dtype=float),
            ]),
        }
        if "buckle_arr_update" in df_full.columns:
            buckle_update = stack_buckle_arrays(df_full["buckle_arr_update"])
            lims["buckle_y"] = (buckle_update.shape[1] - 0.5, -0.5)
        return lims

    def plot_training_snapshot(fig: plt.Figure, axes: List[plt.Axes], df_full: pd.DataFrame, frame_idx: int,
                               csv_name: str, axis_lims: dict) -> None:
        colors_lst, red, custom_cmap = colors.color_scheme()
        df = df_full.iloc[:frame_idx + 1]
        t = df["t"].to_numpy(dtype=float) if "t" in df.columns else np.arange(1, len(df) + 1, dtype=float)
        font_size = 10
        title_size = 11

        for ax in axes:
            ax.clear()

        fig.suptitle(r"$t={}$".format(f"{t[-1]:g}"), fontsize=title_size)

        ax_force = axes[0]
        measured_desired_cols = [
            ("F_x_meas", colors_lst[2], "-", r"$F_x$ meas."),
            ("F_y_meas", colors_lst[1], "-", r"$F_y$ meas."),
            ("F_x_des", colors_lst[2], "--", r"$F_x$ des."),
            ("F_y_des", colors_lst[1], "--", r"$F_y$ des."),
        ]
        for col, color, linestyle, label in measured_desired_cols:
            if col in df.columns:
                is_desired = col.endswith("_des")
                ax_force.plot(t, df[col].to_numpy(dtype=float), color=color, linestyle=linestyle,
                              marker="o", markerfacecolor="none" if is_desired else color,
                              markeredgecolor=color, alpha=0.8, label=label)
        ax_force.set_ylabel(r"$F_{meas}\,\left[mN\right]$", fontsize=font_size)
        ax_force.set_ylim(axis_lims["measured_desired_force"])
        ax_force.legend(loc="best", ncol=2, fontsize=8)

        ax_loss = axes[1]
        loss_mse = df["loss_MSE"].to_numpy(dtype=float)
        ax_loss.plot(t, loss_mse, color=colors_lst[0], marker="o", label=r"$\mathcal{L}$")
        ax_loss.plot(t, np.zeros(len(t)), color=colors_lst[0], linestyle="--")
        ax_loss.set_ylim(axis_lims["loss"])
        ax_loss.set_ylabel(r"$\mathcal{L}$", fontsize=font_size)
        ax_loss.legend(loc="best", fontsize=8)

        ax_update = axes[2]
        ax_angle = ax_update.twinx()
        ax_update.plot(t, df["upd_x_tip"].to_numpy(dtype=float), color=colors_lst[2], marker="o",
                       label=r"$x_{tip}$ update")
        ax_update.plot(t, df["upd_y_tip"].to_numpy(dtype=float), color=colors_lst[1], marker="o",
                       label=r"$y_{tip}$ update")
        ax_angle.plot(t, df["upd_tip_angle"].to_numpy(dtype=float), color=red, marker="o",
                      label=r"$\theta_{tip}$ update")
        ax_update.set_ylabel(r"update pos $\left[m\right]$", fontsize=font_size)
        ax_angle.set_ylabel(r"update angle $\left[\degree\right]$", fontsize=font_size)
        ax_update.set_ylim(axis_lims["position"])
        ax_angle.set_ylim(axis_lims["angle"])
        pos_lines, pos_labels = ax_update.get_legend_handles_labels()
        angle_lines, angle_labels = ax_angle.get_legend_handles_labels()
        ax_update.legend(pos_lines + angle_lines, pos_labels + angle_labels, loc="best", ncol=3, fontsize=8)

        next_axis_idx = 3
        ax_update_force = axes[next_axis_idx]
        ax_update_force.plot(t, df["F_x_update"].to_numpy(dtype=float), color=colors_lst[2], marker="o",
                             label=r"$F_x$ update")
        ax_update_force.plot(t, df["F_y_update"].to_numpy(dtype=float), color=colors_lst[1], marker="o",
                             label=r"$F_y$ update")
        ax_update_force.set_ylabel(r"$F_{update}\,\left[mN\right]$", fontsize=font_size)
        ax_update_force.set_ylim(axis_lims["update_force"])
        ax_update_force.legend(loc="best", ncol=2, fontsize=8)
        next_axis_idx += 1

        if "buckle_y" in axis_lims:
            buckle_update = stack_buckle_arrays(df["buckle_arr_update"])
            buckle_cmap = ListedColormap([colors_lst[1], colors_lst[4]])
            buckle_cmap.set_bad("#e5e5e5")
            buckle_norm = BoundaryNorm([-1.5, 0, 1.5], buckle_cmap.N)
            axes[next_axis_idx].imshow(
                buckle_update.T,
                aspect="auto",
                interpolation="nearest",
                cmap=buckle_cmap,
                norm=buckle_norm,
                extent=[t[0] - 0.5, t[-1] + 0.5, buckle_update.shape[1] - 0.5, -0.5],
            )
            axes[next_axis_idx].set_ylabel("Buckle update", fontsize=font_size)
            axes[next_axis_idx].set_ylim(axis_lims["buckle_y"])
            axes[next_axis_idx].yaxis.set_major_locator(MaxNLocator(integer=True))
            next_axis_idx += 1

        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(labelsize=8)
            ax.set_xlim(axis_lims["x"])

    saved_paths = []
    pairs = find_pairs()
    if not pairs:
        raise FileNotFoundError(f"No matching CSV/image-folder pairs found in {training_dir}.")

    for csv_path, img_dir, out_path in pairs:
        df = prepare_dataframe(csv_path)
        required_cols = ["upd_x_tip", "upd_y_tip", "upd_tip_angle", "F_x_update", "F_y_update", "loss_MSE"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns in {csv_path}: {missing_cols}")

        image_paths = image_files_in_dir(img_dir)
        if not image_paths:
            raise FileNotFoundError(f"No image files found in {img_dir}.")

        original_csv_rows = len(df)
        original_image_count = len(image_paths)
        frame_count = min(original_csv_rows, original_image_count)
        df = df.iloc[:frame_count].copy().reset_index(drop=True)
        image_indices = np.linspace(0, len(image_paths) - 1, frame_count).round().astype(int)
        row_indices = np.arange(frame_count, dtype=int)
        frame_t_values = df["t"].to_numpy(dtype=int) if "t" in df.columns else np.arange(1, frame_count + 1, dtype=int)
        if original_image_count != original_csv_rows:
            print(f"{csv_path.name}: using {frame_count} frames from {original_image_count} images and {original_csv_rows} CSV rows.")

        axis_lims = build_axis_lims(df)
        has_buckle = "buckle_y" in axis_lims
        height_ratios = [1.2, 0.8, 1.0, 0.85]
        if has_buckle:
            height_ratios.append(0.75)
        n_plot_rows = len(height_ratios)
        fig = plt.figure(figsize=(12, 6.6), dpi=dpi, constrained_layout=True)

        frames_dir = out_path.with_name(f"{out_path.stem}_frames")
        frames_dir.mkdir(parents=True, exist_ok=True)

        def draw_frame(frame: int, jpg_path: Optional[Path] = None) -> NDArray[np.uint8]:
            fig.clear()
            grid = fig.add_gridspec(n_plot_rows, 2, width_ratios=[1.2, 1.0], height_ratios=height_ratios)
            ax_image = fig.add_subplot(grid[:, 0])
            plot_axes = [fig.add_subplot(grid[row, 1]) for row in range(n_plot_rows)]
            ax_image.clear()
            image = plt.imread(image_paths[image_indices[frame]])
            ax_image.imshow(image)
            ax_image.axis("off")
            plot_training_snapshot(fig, plot_axes, df, int(row_indices[frame]), csv_path.name, axis_lims)
            fig.canvas.draw()
            if jpg_path is not None:
                fig.savefig(jpg_path, dpi=max(dpi, 300), bbox_inches="tight")
            rgba = np.asarray(fig.canvas.buffer_rgba())
            return rgba[:, :, :3].copy()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".gif":
            from PIL import Image

            pil_frames = [Image.fromarray(draw_frame(frame)) for frame in range(frame_count)]
            for frame in range(frame_count):
                draw_frame(frame, frames_dir / f"t_{int(frame_t_values[frame]):03d}.jpg")
            durations = [int(1000 / fps)] * frame_count
            durations[-1] = 3000
            pil_frames[0].save(
                out_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=durations,
                loop=0,
            )
        else:
            try:
                import cv2
            except ImportError as exc:
                raise ImportError("Saving MP4/AVI requires either ffmpeg or opencv-python (cv2).") from exc

            frame_rgb = draw_frame(0)
            height, width = frame_rgb.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*("mp4v" if out_path.suffix.lower() == ".mp4" else "XVID"))
            writer = cv2.VideoWriter(str(out_path), fourcc, video_writer_fps(out_path, fps), (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {out_path}.")
            try:
                for frame in range(frame_count):
                    jpg_path = frames_dir / f"t_{int(frame_t_values[frame]):03d}.jpg"
                    frame_rgb = draw_frame(frame, jpg_path)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    repeats = max(1, int(round(3 * fps))) if frame == frame_count - 1 else 1
                    for _ in range(repeats):
                        writer.write(frame_bgr)
            finally:
                writer.release()
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def training_pos_data_and_vid(
    training_dir: Union[str, Path] = r"C:\Users\SMR_Admin\OneDrive - huji.ac.il\ORIGAMI\Meca500\data\training\June21_fromPos",
    csv_file_path: Optional[Union[str, Path]] = None,
    pics_dir: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    infer_image_sequence: bool = False,
    final_t: Optional[int] = None,
    fps: int = 2,
    dpi: int = 180,
) -> List[Path]:
    """
    Create position-training video(s) from desired/measured/update state images.

    Expected image names are ``des.jpg``, ``meas1.jpg``, ``meas2.jpg``, ...
    and either ``up1.jpg``/``up2.jpg`` or ``update1.jpg``/``update2.jpg``.
    With ``infer_image_sequence=True``, image names are ignored and natural file
    order is used as: desired, meas1, update1, meas2, update2, ...
    This is separate from ``training_data_and_vid`` and writes
    ``*_position_states_vid.mp4`` by default.
    """
    training_dir = Path(training_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def natural_key(path: Path) -> List[Union[int, str]]:
        parts = re.split(r"(\d+)", path.name.lower())
        return [int(part) if part.isdigit() else part for part in parts]

    def numbered_image_map(path: Path, prefixes: Tuple[str, ...]) -> dict:
        images = {}
        prefix_regex = "|".join(re.escape(prefix) for prefix in prefixes)
        pattern = re.compile(rf"^({prefix_regex})(\d+)$", re.IGNORECASE)
        for item in path.iterdir():
            if not item.is_file() or item.suffix.lower() not in image_exts:
                continue
            match = pattern.match(item.stem)
            if match:
                images[int(match.group(2))] = item
        return images

    def image_files_in_dir(path: Path) -> List[Path]:
        return sorted(
            [item for item in path.iterdir() if item.is_file() and item.suffix.lower() in image_exts],
            key=natural_key,
        )

    def infer_sequence_images(path: Path) -> Tuple[Optional[Path], List[Tuple[int, Path, Path]]]:
        image_paths = image_files_in_dir(path)
        if len(image_paths) < 3:
            return None, []
        des_image = image_paths[0]
        state_images = image_paths[1:]
        paired_state_count = len(state_images) - (len(state_images) % 2)
        frame_entries = []
        for pair_idx in range(0, paired_state_count, 2):
            t_number = pair_idx // 2 + 1
            frame_entries.append((t_number, state_images[pair_idx], state_images[pair_idx + 1]))
        return des_image, frame_entries

    def named_sequence_images(path: Path) -> Tuple[Optional[Path], List[Tuple[int, Path, Path]]]:
        des_image = find_des_image(path)
        meas_images = numbered_image_map(path, ("meas",))
        update_images = numbered_image_map(path, ("up", "update"))
        frame_numbers = sorted(set(meas_images).intersection(update_images))
        frame_entries = [(frame_number, meas_images[frame_number], update_images[frame_number])
                         for frame_number in frame_numbers]
        return des_image, frame_entries

    def get_state_images(path: Path) -> Tuple[Optional[Path], List[Tuple[int, Path, Path]]]:
        if infer_image_sequence:
            return infer_sequence_images(path)
        return named_sequence_images(path)

    def find_des_image(path: Path) -> Optional[Path]:
        for suffix in image_exts:
            candidate = path / f"des{suffix}"
            if candidate.exists():
                return candidate
        return None

    def find_pics_dir_for_csv(csv_path: Path) -> Optional[Path]:
        candidates = [csv_path.parent, csv_path.parent / "pics"]
        candidates.extend([path for path in csv_path.parent.iterdir() if path.is_dir()])
        for candidate in candidates:
            des_image, frame_entries = get_state_images(candidate)
            if des_image is not None and frame_entries:
                return candidate
        return None

    def find_pairs() -> List[Tuple[Path, Path, Path]]:
        if csv_file_path is not None or pics_dir is not None:
            if csv_file_path is None or pics_dir is None:
                raise ValueError("csv_file_path and pics_dir must be supplied together.")
            csv_path = Path(csv_file_path)
            image_dir = Path(pics_dir)
            out_path = Path(output_path) if output_path is not None else csv_path.with_name(f"{csv_path.stem}_position_states_vid.mp4")
            return [(csv_path, image_dir, out_path)]

        csv_paths = [
            path for path in training_dir.rglob("*.csv")
            if "oldver" not in path.stem.lower() and "dataset" not in path.stem.lower()
        ]
        pairs = []
        for csv_path in sorted(csv_paths, key=natural_key):
            image_dir = find_pics_dir_for_csv(csv_path)
            if image_dir is None:
                continue
            out_path = csv_path.with_name(f"{csv_path.stem}_position_states_vid.mp4")
            pairs.append((csv_path, image_dir, out_path))
        return pairs

    def prepare_dataframe(csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        if final_t is not None:
            if "t" in df.columns:
                df = df[df["t"] <= final_t].copy()
            else:
                df = df.iloc[:final_t + 1].copy()
        if "t" in df.columns:
            df = df[df["t"] >= 1].copy()
        else:
            df = df.iloc[1:].copy()
        if df.empty:
            raise ValueError(f"No rows available to plot in {csv_path} after applying final_t.")
        return df.reset_index(drop=True)

    def build_axis_lims(df_full: pd.DataFrame) -> dict:
        t_full = df_full["t"].to_numpy(dtype=float) if "t" in df_full.columns else np.arange(1, len(df_full) + 1, dtype=float)
        return {
            "x": time_step_lims(t_full),
            "measured_desired_position": padded_lims([
                df_full["x_rest_meas"].to_numpy(dtype=float),
                df_full["y_rest_meas"].to_numpy(dtype=float),
                df_full["x_rest_des"].to_numpy(dtype=float),
                df_full["y_rest_des"].to_numpy(dtype=float),
            ]),
            "update_position": padded_lims([
                df_full["upd_x_tip"].to_numpy(dtype=float),
                df_full["upd_y_tip"].to_numpy(dtype=float),
            ]),
            "measured_desired_angle": centered_lims(
                df_full["theta_rest_des"].to_numpy(dtype=float),
                [
                df_full["theta_rest_meas"].to_numpy(dtype=float),
                df_full["theta_rest_des"].to_numpy(dtype=float),
                ],
            ),
            "update_angle": padded_lims([df_full["upd_tip_angle"].to_numpy(dtype=float)]),
            "loss": padded_lims([
                df_full["loss_MSE"].to_numpy(dtype=float),
                np.zeros(len(df_full), dtype=float),
            ]),
        }

    def plot_position_snapshot(fig: plt.Figure, axes: List[plt.Axes], df_full: pd.DataFrame,
                               frame_idx: int, axis_lims: dict) -> None:
        colors_lst, red, custom_cmap = colors.color_scheme()
        df = df_full.iloc[:frame_idx + 1]
        t = df["t"].to_numpy(dtype=float) if "t" in df.columns else np.arange(1, len(df) + 1, dtype=float)
        font_size = 10

        fig.suptitle(r"$t={}$".format(f"{t[-1]:g}"), fontsize=11)

        ax_meas_des = axes[0]
        ax_meas_des_angle = ax_meas_des.twinx()
        ax_meas_des.plot(t, df["x_rest_meas"].to_numpy(dtype=float), color=colors_lst[2],
                         marker="o", markerfacecolor=colors_lst[2], markeredgecolor=colors_lst[2])
        ax_meas_des.plot(t, df["y_rest_meas"].to_numpy(dtype=float), color=colors_lst[1],
                         marker="o", markerfacecolor=colors_lst[1], markeredgecolor=colors_lst[1])
        ax_meas_des_angle.plot(t, df["theta_rest_meas"].to_numpy(dtype=float), color=red,
                               marker="o", markerfacecolor=red, markeredgecolor=red)
        ax_meas_des.plot(t, df["x_rest_des"].to_numpy(dtype=float), color=colors_lst[2], linestyle="--",
                         marker="o", markerfacecolor="none", markeredgecolor=colors_lst[2])
        ax_meas_des.plot(t, df["y_rest_des"].to_numpy(dtype=float), color=colors_lst[1], linestyle="--",
                         marker="o", markerfacecolor="none", markeredgecolor=colors_lst[1])
        ax_meas_des_angle.plot(t, df["theta_rest_des"].to_numpy(dtype=float), color=red, linestyle="--",
                               marker="o", markerfacecolor="none", markeredgecolor=red)
        ax_meas_des.set_ylabel(r"pos $\left[m\right]$", fontsize=font_size)
        ax_meas_des_angle.set_ylabel(r"angle $\left[\degree\right]$", fontsize=font_size)
        ax_meas_des.set_ylim(axis_lims["measured_desired_position"])
        ax_meas_des_angle.set_ylim(axis_lims["measured_desired_angle"])
        meas_des_handles = [
            Line2D([0], [0], color=colors_lst[2], linestyle="-", marker="o",
                   markerfacecolor=colors_lst[2], markeredgecolor=colors_lst[2], label=r"$x$ meas."),
            Line2D([0], [0], color=colors_lst[2], linestyle="--", marker="o",
                   markerfacecolor="none", markeredgecolor=colors_lst[2], label=r"$x$ des."),
            Line2D([0], [0], color=colors_lst[1], linestyle="-", marker="o",
                   markerfacecolor=colors_lst[1], markeredgecolor=colors_lst[1], label=r"$y$ meas."),
            Line2D([0], [0], color=colors_lst[1], linestyle="--", marker="o",
                   markerfacecolor="none", markeredgecolor=colors_lst[1], label=r"$y$ des."),
            Line2D([0], [0], color=red, linestyle="-", marker="o",
                   markerfacecolor=red, markeredgecolor=red, label=r"$\theta$ meas."),
            Line2D([0], [0], color=red, linestyle="--", marker="o",
                   markerfacecolor="none", markeredgecolor=red, label=r"$\theta$ des.")
        ]
        ax_meas_des.legend(handles=meas_des_handles, loc="best", ncol=3, fontsize=8)

        ax_loss = axes[1]
        ax_loss.plot(t, df["loss_MSE"].to_numpy(dtype=float), color=colors_lst[0], marker="o",
                     label=r"$\mathcal{L}$")
        ax_loss.plot(t, np.zeros(len(t)), color=colors_lst[0], linestyle="--")
        ax_loss.set_ylabel(r"$\mathcal{L}$", fontsize=font_size)
        ax_loss.set_ylim(axis_lims["loss"])
        ax_loss.legend(loc="best", fontsize=8)

        ax_update = axes[2]
        ax_update_angle = ax_update.twinx()
        ax_update.plot(t, df["upd_x_tip"].to_numpy(dtype=float), color=colors_lst[2], marker="o",
                       label=r"$x$ update")
        ax_update.plot(t, df["upd_y_tip"].to_numpy(dtype=float), color=colors_lst[1], marker="o",
                       label=r"$y$ update")
        ax_update_angle.plot(t, df["upd_tip_angle"].to_numpy(dtype=float), color=red, marker="o",
                             label=r"$\theta$ update")
        ax_update.set_xlabel("t", fontsize=font_size)
        ax_update.set_ylabel(r"pos $\left[m\right]$", fontsize=font_size)
        ax_update_angle.set_ylabel(r"angle $\left[\degree\right]$", fontsize=font_size)
        ax_update.set_ylim(axis_lims["update_position"])
        ax_update_angle.set_ylim(axis_lims["update_angle"])
        lines, labels = ax_update.get_legend_handles_labels()
        angle_lines, angle_labels = ax_update_angle.get_legend_handles_labels()
        ax_update.legend(lines + angle_lines, labels + angle_labels, loc="best", ncol=3, fontsize=8)

        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(labelsize=8)
            ax.set_xlim(axis_lims["x"])

    def show_image(ax: plt.Axes, image_path: Path, title: str) -> None:
        ax.imshow(plt.imread(image_path))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylabel(title, fontsize=11, rotation=90, labelpad=12)

    saved_paths = []
    pairs = find_pairs()
    if not pairs:
        raise FileNotFoundError(f"No matching position CSV/pics pairs found in {training_dir}.")

    for csv_path, image_dir, out_path in pairs:
        df = prepare_dataframe(csv_path)
        required_cols = [
            "upd_x_tip", "upd_y_tip", "upd_tip_angle", "loss_MSE",
            "x_rest_meas", "y_rest_meas", "theta_rest_meas",
            "x_rest_des", "y_rest_des", "theta_rest_des",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns in {csv_path}: {missing_cols}")

        des_image, frame_entries = get_state_images(image_dir)
        if des_image is None:
            raise FileNotFoundError(f"No des image found in {image_dir}.")
        if not frame_entries:
            if infer_image_sequence:
                raise FileNotFoundError(f"No inferred desired/measured/update image sequence found in {image_dir}.")
            raise FileNotFoundError(f"No matching measN/upN or measN/updateN image pairs found in {image_dir}.")

        original_frame_count = len(frame_entries)
        frame_count = min(len(df), original_frame_count)
        frame_entries = frame_entries[:frame_count]
        df = df.iloc[:frame_count].copy().reset_index(drop=True)
        if original_frame_count != len(df):
            print(f"{csv_path.name}: using {frame_count} frames from {original_frame_count} image pairs and {len(df)} CSV rows.")

        axis_lims = build_axis_lims(df)
        fig = plt.figure(figsize=(13.2, 7.2), dpi=dpi, constrained_layout=True)
        frames_dir = out_path.with_name(f"{out_path.stem}_frames")
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_t_values = df["t"].to_numpy(dtype=int) if "t" in df.columns else np.arange(1, frame_count + 1, dtype=int)

        def draw_frame(frame: int, jpg_path: Optional[Path] = None) -> NDArray[np.uint8]:
            fig.clear()
            grid = fig.add_gridspec(3, 2, width_ratios=[1.65, 1.0], height_ratios=[1, 1, 1])
            ax_meas = fig.add_subplot(grid[0, 0])
            ax_update = fig.add_subplot(grid[1, 0])
            ax_des = fig.add_subplot(grid[2, 0])
            plot_axes = [
                fig.add_subplot(grid[0, 1]),
                fig.add_subplot(grid[1, 1]),
                fig.add_subplot(grid[2, 1]),
            ]

            frame_number, meas_image, update_image = frame_entries[frame]
            show_image(ax_meas, meas_image, "measured")
            show_image(ax_update, update_image, "update")
            show_image(ax_des, des_image, "desired")
            plot_position_snapshot(fig, plot_axes, df, frame, axis_lims)
            fig.canvas.draw()
            if jpg_path is not None:
                fig.savefig(jpg_path, dpi=max(dpi, 300), bbox_inches="tight")
            rgba = np.asarray(fig.canvas.buffer_rgba())
            return rgba[:, :, :3].copy()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".gif":
            from PIL import Image

            pil_frames = [Image.fromarray(draw_frame(frame)) for frame in range(frame_count)]
            for frame in range(frame_count):
                draw_frame(frame, frames_dir / f"t_{int(frame_t_values[frame]):03d}.jpg")
            durations = [int(1000 / fps)] * frame_count
            durations[-1] = 3000
            pil_frames[0].save(
                out_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=durations,
                loop=0,
            )
        else:
            try:
                import cv2
            except ImportError as exc:
                raise ImportError("Saving MP4/AVI requires opencv-python (cv2).") from exc

            frame_rgb = draw_frame(0)
            height, width = frame_rgb.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*("mp4v" if out_path.suffix.lower() == ".mp4" else "XVID"))
            writer = cv2.VideoWriter(str(out_path), fourcc, video_writer_fps(out_path, fps), (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {out_path}.")
            try:
                for frame in range(frame_count):
                    jpg_path = frames_dir / f"t_{int(frame_t_values[frame]):03d}.jpg"
                    frame_rgb = draw_frame(frame, jpg_path)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    repeats = max(1, int(round(3 * fps))) if frame == frame_count - 1 else 1
                    for _ in range(repeats):
                        writer.write(frame_bgr)
            finally:
                writer.release()
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths
