"""
Code for testing the Fourier Series Filter (FSF)
Same as se2_door_filter_legacy.py, but uses the FSF
instead of the histogram filter
"""

from typing import Optional
from omegaconf import DictConfig, OmegaConf

import json
import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tqdm import tqdm
from copy import deepcopy

from src.spectral.se2_fft import SE2_FFT
from src.simulators.se2_door_dataset import SE2DoorDataset
from src.distributions.se2_distributions import SE2, SE2Gaussian

from src.filters.hef import HarmonicExponentialFilter
from src.filters.range_ekf import BearingEKF
from src.filters.range_pf import BearingPF
from src.filters.fsf import FourierSeriesFilter

from src.sampler.se2_sampler import se2_grid_samples
from src.utils.create_video import create_mp4
from src.utils.numpy_json import NumpyEncoder
from src.utils.se2_plotting import (
    plot_se2_mean_filters,
    plot_error_xy_trajectory,
    plot_se2_filters,
    plot_neg_log_likelihood,
)
from src.utils.statistics import compute_weighted_mean, compute_mode
from src.utils import se2_plot_configs as plt_cfg
from src.utils.logging import seed_everything, get_logger, extras

log = get_logger(__name__)

for i in range(len(plt_cfg.CONFIG_MEAN_SE2_UWB)):
    if plt_cfg.CONFIG_MEAN_SE2_UWB[i]['label'] == 'HistF':
        plt_cfg.CONFIG_MEAN_SE2_UWB.pop(i)
        break


def main(cfg: DictConfig) -> Optional[float]:
    make_video = cfg.get("make_video")
    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed)
    results_path = os.path.join(cfg.results_path, datetime.datetime.now().isoformat())
    figures_path = os.path.join(results_path, "figures")
    others_path = os.path.join(results_path, "others")
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)
        os.makedirs(others_path)

    # Store config
    extras(cfg, others_path)

    # Hyperparameters
    grid_size = cfg.filter.grid_size
    pose_grid, x, y, theta = se2_grid_samples(grid_size)

    fft = SE2_FFT(
        spatial_grid_size=grid_size,
        interpolation_method="spline",
        spline_order=2,
        # TODO: Set this to 1 (or undersample) to ensure fair comparison
        oversampling_factor=3,
    )

    scaling_factor = cfg.filter.scaling_factor
    d_door2pose = cfg.filter.d_door2pose
    doors_blacklist = cfg.filter.doors_blacklist
    offset_x, offset_y = cfg.filter.offset_x, cfg.filter.offset_y
    # Define motion noise and measurement noise
    var_motion = cfg.filter.var_motion
    motion_noise = np.ones(3) * np.sqrt(var_motion)
    measurement_noise = np.sqrt(cfg.filter.var_measurement)
    simulator = SE2DoorDataset(
        data_path=cfg.data_dir,
        fft=fft,
        d_door2pose=d_door2pose,
        scaling_factor=scaling_factor,
        offset_x=offset_x,
        offset_y=offset_y,
        doors_blacklist=doors_blacklist,
        motion_noise=motion_noise,
        pose_grid=pose_grid,
        measurement_noise=measurement_noise,
    )

    # Define prior
    mu_prior = simulator.position.parameters()
    # mu_prior = np.array([0.0, 0.0, 0.0])
    cov_prior = np.diag(cfg.filter.var_prior)

    prior = SE2Gaussian(mu_prior, cov_prior, samples=pose_grid, fft=fft)
    prior.normalize()

    # Initialize filters
    hef = HarmonicExponentialFilter(distribution=SE2, prior=prior)
    ekf = BearingEKF(prior=mu_prior, prior_cov=cov_prior)
    pf = BearingPF(
        prior=mu_prior,
        prior_cov=cov_prior,
        n_particles=np.prod(grid_size),
        d_door2pose=d_door2pose,
    )
    fsf = FourierSeriesFilter(
        prior=mu_prior,
        prior_cov=cov_prior,
        grid_samples=pose_grid,
        grid_size=grid_size
    )

    # For logging
    mean_trajectory = dict(
        HEF=np.zeros((simulator.n_samples, 3)),
        EKF=np.zeros((simulator.n_samples, 3)),
        FSF=np.zeros((simulator.n_samples, 3)),
        PF=np.zeros((simulator.n_samples, 3)),
        Measurement=np.zeros((simulator.n_samples, 3)),
        GT=np.zeros((simulator.n_samples, 3)),
    )
    mode_trajectory = deepcopy(mean_trajectory)
    nll = dict(HEF=[], EKF=[], FSF=[], PF=[], Measurement=[])

    # Populate the first entry with prior pose
    gt_pose = simulator.position.parameters()
    for key in mean_trajectory.keys():
        mean_trajectory[key][0] = gt_pose
        mode_trajectory[key][0] = gt_pose

    nll["Measurement"].append(-np.log(0.5))
    nll['HEF'].append(hef.neg_log_likelihood2(prior.energy, prior.l_n_z, 
                                              gt_pose, grid_size).item())
    nll["EKF"].append(ekf.neg_log_likelihood(gt_pose))
    nll["PF"].append(pf.neg_log_likelihood(gt_pose, (-0.5, 0.5), grid_size))
    nll["FSF"].append(fsf.neg_log_likelihood(gt_pose))

    for it in tqdm(range(100), desc="Filtering door dataset..."):
        ### Predict step ###
        motion_distribution = simulator.motion()

        # For the HEF, we use the same parameters as were used in their paper
        # The other filters seem to work better with a lower process noise
        hef_pred = hef.prediction(motion_model=motion_distribution)

        ekf.prediction(
            step=motion_distribution.mu,
            step_cov=motion_distribution.cov*0.25,
        )
        pf.prediction(
            step=motion_distribution.mu,
            step_cov=motion_distribution.cov*0.25,
        )
        fsf.prediction(
            step=motion_distribution.mu,
            diffusion_coefficient=motion_distribution.cov,
            dt=simulator.timestep_bins[simulator.iteration],
        )

        ### Update step ###
        # Set mean/mode and nll for each filter
        measurement_distribution = simulator.measurement()
        gt_pose = simulator.gt_bins[simulator.iteration]
        mean_trajectory["GT"][it] = mode_trajectory["GT"][it] = gt_pose

        # "Expected/Maximum Likelihood Filter" (no prior info.)
        mean_trajectory["Measurement"][it] = compute_weighted_mean(
            measurement_distribution.prob, pose_grid, x, y, theta)
        mode_trajectory["Measurement"][it] = compute_mode(measurement_distribution.prob, pose_grid)
        nll["Measurement"].append(simulator.neg_log_likelihood(gt_pose))
        # Harmonic Exponential Filter (HEF)
        hef_posterior, _ = hef.update(
            measurement_model=measurement_distribution
        )
        hef_mean = compute_weighted_mean(
            hef_posterior.prob, pose_grid, x, y, theta
        )
        mean_trajectory["HEF"][it] = hef_mean
        hef_mode = compute_mode(hef_posterior.prob, pose_grid)
        mode_trajectory["HEF"][it] = hef_mode
        nll['HEF'].append(hef.neg_log_likelihood2(hef_posterior.energy, hef_posterior.l_n_z, gt_pose, grid_size).item())
        # Extended Kalman Filter (EKF)
        ekf_mean, ekf_pos_cov = ekf.update(
            landmarks=np.array(simulator.doors),
            observations=np.array(simulator.bearing_bins[simulator.iteration]),
            observations_cov=np.ones(len(simulator.doors)) * simulator.measurement_cov,
        )
        mean_trajectory["EKF"][it] = mode_trajectory["EKF"][it] = ekf_mean
        nll["EKF"].append(ekf.neg_log_likelihood(gt_pose))
        # Particle Filter (PF)
        pf_mean = pf.update(
            landmarks=np.array(simulator.doors),
            map_mask=simulator.map_mask_unprocessed,
            observations=simulator.bearing_bins[simulator.iteration],
            observations_cov=np.ones(len(simulator.doors)) * simulator.measurement_cov,
        )
        mean_trajectory["PF"][it] = pf_mean
        pf_mode = pf.compute_mode()
        mode_trajectory["PF"][it] = pf_mode
        nll["PF"].append(pf.neg_log_likelihood(gt_pose, (-0.5, 0.5), grid_size))
        # Fourier Series Filter (FSF)
        # TODO
        fsf.update(
            landmarks=np.array(simulator.doors),
            map_mask=simulator.map_mask,
            observations=simulator.bearing_bins[simulator.iteration],
            observations_cov=np.ones(len(simulator.doors)) * simulator.measurement_cov,
        )
        fsf_mean = fsf.compute_mean()
        fsf_mode = fsf.compute_mode()
        mean_trajectory["FSF"][it] = fsf_mean
        mode_trajectory["FSF"][it] = fsf_mode
        nll["FSF"].append(fsf.neg_log_likelihood(gt_pose))

        if make_video:
            ### Plotting ###
            # Axes for the plot at media/reference/axes_means.png
            legend = [rf"Predicted belief", rf"Measurement Likelihood", 
                    rf"Posterior belief"]

            axes_means = plot_se2_mean_filters(
                [hef_pred.prob.real, measurement_distribution.prob.real, hef_posterior.prob.real],
                x, y, theta,
                samples=mean_trajectory,
                iteration=it,
                beacons=np.array(simulator.doors),
                level_contours=False,
                contour_titles=legend,
                config=plt_cfg.CONFIG_MEAN_SE2_UWB,
            )
            axes_modes = plot_se2_mean_filters(
                [hef_pred.prob.real, measurement_distribution.prob.real, hef_posterior.prob.real],
                x, y, theta,
                samples=mode_trajectory,
                iteration=it,
                beacons=np.array(simulator.doors),
                level_contours=False,
                contour_titles=legend,
                config=plt_cfg.CONFIG_MEAN_SE2_UWB,
            )
            # Axes for the plot at media/reference/axes_filters.png
            axes_filters = plot_se2_filters(
                {
                    "HEF": [hef_mean, hef_posterior.prob.real, hef_mode],
                    "EKF": [ekf_mean, ekf_pos_cov, ekf_mean],
                    "PF": [pf_mean, pf.particles, pf_mode, pf.weights],
                    "FSF": [fsf_mean, fsf.prior.reshape(grid_size), fsf_mode],
                    "GT": [gt_pose, None],
                },
                x, y, theta,
                np.array(simulator.doors),
                titles=[
                    f"Harmonic Exponential Filter",
                    f"Extended Kalman Filter",
                    f"Particle Filter",
                    f"Fourier Series Filter",
                ],
                config=plt_cfg.CONFIG_FILTERS_SE2_UWB,
            )
            map_alpha = 0.7
            for ax_filter in axes_filters:
                # Plot map
                ax_filter.imshow(
                    simulator.map_array[2],
                    extent=[
                        simulator.map_array[0].min(),
                        simulator.map_array[0].max(),
                        simulator.map_array[1].min(),
                        simulator.map_array[1].max(),
                    ],
                    origin="upper",
                    cmap=plt.cm.Greys_r,
                    alpha=map_alpha,
                    zorder=-11,
                )

            for ax in axes_means + axes_modes + axes_filters:
                ax.set_xlim(-0.45, 0.45)
                ax.set_ylim(-0.45, 0.45)
            
            axes_means[3].imshow(
                simulator.map_array[2],
                extent=[
                    simulator.map_array[0].min(),
                    simulator.map_array[0].max(),
                    simulator.map_array[1].min(),
                    simulator.map_array[1].max(),
                ],
                origin="upper",
                cmap=plt.cm.Greys_r,
                alpha=map_alpha,
                zorder=-11,
            )
            for c in plt_cfg.CONFIG_FILTERS_SE2_LF:
                if c.get('label') == 'Ground Truth':
                    break
            axes_means[3].scatter(gt_pose[0], gt_pose[1], **c)
            axes_modes[3].scatter(gt_pose[0], gt_pose[1], **c)
            
            axes_modes[3].imshow(
                simulator.map_array[2],
                extent=[
                    simulator.map_array[0].min(),
                    simulator.map_array[0].max(),
                    simulator.map_array[1].min(),
                    simulator.map_array[1].max(),
                ],
                origin="upper",
                cmap=plt.cm.Greys_r,
                alpha=map_alpha,
                zorder=-11
            )
            axes_means[3].set_title(f"EAP (Mean) - Timestep {it}", fontdict={'fontsize': 18})
            axes_modes[3].set_title(f"MAP (Mode) - Timestep {it}", fontdict={'fontsize': 18})
            
            # Save means' figure
            plt.figure(1)
            plt.savefig(f"{figures_path}/se2_main{it:03d}.png")
            plt.close()
            plt.figure(2)
            plt.savefig(f"{figures_path}/se2_map{it:03d}.png")
            plt.close()
            # Save filters' figure
            plt.figure(3)
            plt.savefig(f"{figures_path}/se2_filters{it:03d}.png")
            plt.close()
            # plt.show()
        
        pass  # next iteration

    # Plot log-likelihood of each estimator and the ground truth
    plot_neg_log_likelihood(nll, config=plt_cfg.CONFIG_LL_SE2_LF)
    plt.savefig(f"{others_path}/se2_nll.png")
    plt.close()
    # Plot trajectory
    ax, metrics = plot_error_xy_trajectory(
        mean_trajectory,
        scaling_factor,
        offset_x,
        offset_y,
        landmarks=np.array(simulator.doors),
        config=plt_cfg.CONFIG_TRAJ_SE2_UWB,
        x_y_limits=[-18.5, 8.5, -8.0, 16.5],
    )
    _, metrics_mode = plot_error_xy_trajectory(
        mode_trajectory,
        scaling_factor,
        offset_x,
        offset_y,
        landmarks=np.array(simulator.doors),
        config=plt_cfg.CONFIG_TRAJ_SE2_UWB,
        x_y_limits=[-18.5, 8.5, -8.0, 16.5],
    )
    # Scale map coordinates and add map to plot
    map_x = (simulator.map_array[0] / scaling_factor) - offset_x
    map_y = (simulator.map_array[1] / scaling_factor) - offset_y
    ax.imshow(
        simulator.map_array[2],
        extent=[map_x.min(), map_x.max(), map_y.min(), map_y.max()],
        origin="upper",
        cmap=plt.cm.Greys_r,
        alpha=0.8,
        zorder=0,
    )
    ax.set_title(f"Trajectory Estimates - {simulator.n_samples} Timesteps", fontdict={'fontsize': 18})
    # Print metrics
    table = PrettyTable()
    table.float_format = "6.3"
    table.title = 'Results w/ mean'
    table.field_names = ["Filter", "RMSE", "Mean", "Std", "MeanNLL"]
    table_mode = deepcopy(table)
    table_mode.title = 'Results w/ mode'
    metrics_dict = {}
    for k, v in metrics.items():
        # Dead reckoning does not have ll
        map_filter = metrics_mode[k]
        table.add_row([k, v[0], v[1], v[2], np.mean(nll[k])])
        table_mode.add_row([k, map_filter[0], map_filter[1], map_filter[2], np.mean(nll[k])])
        metrics_dict[k] = {
            table.field_names[1]: v[0],
            table.field_names[2]: v[1],
            table.field_names[3]: v[2],
            "RMSE_MAP": map_filter[0],
            "Mean_MAP": map_filter[1],
            "Std_MAP": map_filter[2],
            table.field_names[4]: np.mean(nll[k]),
        }

    with open(f"{others_path}/results.json", "w") as f:
        json.dump(
            {
                "trajectories": mean_trajectory,
                "trajectoris_mode": mode_trajectory,
                "metrics": metrics_dict,
                "neg_log_likelihood": nll,
                "params": OmegaConf.to_container(cfg),
            },
            f,
            cls=NumpyEncoder,
            indent=4,
        )

    print(table)
    print(table_mode)
    plt.savefig(f"{others_path}/se2_traj.png")
    plt.close()
    # Create video
    if cfg.get("duration"):
        create_mp4(results_path, duration=cfg.duration)
    
    # Log information to the logger (if available)
    # log_experiment_info(cfg, results_path)

    import subprocess
    subprocess.run(["open", results_path + "/means.mp4"])

    return 0.0
