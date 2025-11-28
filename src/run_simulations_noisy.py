import os
import math
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

from src.model import ThermalModel2D
import src.parameters as params


def generate_noise_sequences(steps: int, dt: float, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    time = np.linspace(0.0, dt * (steps - 1), steps)
    sequences = {}

    sequences["white"] = rng.normal(0.0, 1.0, size=steps)

    amplitude = 5.0
    period = 120.0
    sequences["cosine"] = amplitude * np.cos(2.0 * np.pi * time / period)

    delta_amplitude = 15.0
    delta_center = 150.0
    delta_width = 5.0
    sequences["delta"] = np.where(np.abs(time - delta_center) <= delta_width, delta_amplitude, 0.0)

    return sequences


def build_category_indices(nx: int, ny: int) -> Tuple[int, List[int], List[int]]:
    center_coord = (ny // 2, nx // 2)
    center_idx = center_coord[0] * nx + center_coord[1]

    inner_ring = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            y = center_coord[0] + dy
            x = center_coord[1] + dx
            if 0 <= y < ny and 0 <= x < nx:
                inner_ring.append(y * nx + x)

    all_indices = set(range(nx * ny))
    edge_indices = sorted(all_indices.difference({center_idx}).difference(inner_ring))
    return center_idx, sorted(inner_ring), edge_indices


def plot_categories(ax: plt.Axes,
                    time: np.ndarray,
                    data: np.ndarray,
                    center_idx: int,
                    inner_ring: List[int],
                    edge_indices: List[int],
                    target_temp: float):
    edge_color = (0.0, 0.35, 0.9)
    inner_color = (0.5, 0.2, 0.8)
    center_color = (0.85, 0.1, 0.1)

    def _plot_group(indices: List[int], color, label: str, alpha: float, lw: float):
        label_used = False
        for idx in indices:
            ax.plot(time, data[:, idx], color=color, alpha=alpha, linewidth=lw,
                    label=label if not label_used else None)
            label_used = True

    _plot_group(edge_indices, edge_color, "Edge Voxels", 0.45, 0.9)
    _plot_group(inner_ring, inner_color, "Inner Ring Voxels", 0.75, 1.1)
    ax.plot(time, data[:, center_idx], color=center_color, linewidth=1.6, label="Center Voxel")
    ax.axhline(target_temp, color='k', linestyle='--', linewidth=1.0, label=f'Target {target_temp:.0f}°C')
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True)


def simulate_lqr_with_noise(model: ThermalModel2D,
                            K: np.ndarray,
                            y0: np.ndarray,
                            T_target: np.ndarray,
                            noise_values: np.ndarray,
                            noise_mask: np.ndarray,
                            dt: float,
                            duration: float) -> Tuple[np.ndarray, np.ndarray]:
    N = model.sys.B.shape[1]
    steps = int(duration / dt)
    time = np.linspace(0.0, duration, steps)

    y_star, u_star = model.get_steady_state(T_target)
    y_current = y0.copy()
    y_hist = np.zeros((steps, 2 * N))

    for i in range(steps):
        y_tilde = y_current - y_star
        delta_u = -K @ y_tilde
        u_applied = np.clip(u_star + delta_u, 0.0, params.P_MAX)

        noise_val = noise_values[i]
        env_noise = np.zeros_like(y_current)
        env_noise[:N] = noise_mask * noise_val

        dy = model.sys.A @ y_current + model.sys.B @ u_applied + model.sys.E + env_noise
        y_current += dy * dt
        y_hist[i] = y_current

    return time, y_hist


def simulate_mpc_with_noise(model: ThermalModel2D,
                            y0: np.ndarray,
                            T_target: np.ndarray,
                            horizon: float,
                            noise_values: np.ndarray,
                            noise_mask: np.ndarray,
                            q_diag_override: np.ndarray,
                            dt: float,
                            duration: float,
                            record_interval: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Adapted from run_simulations.simulate_mpc with noise injection."""
    N = model.sys.B.shape[1]
    steps = int(duration / dt)
    sample_interval = max(1, int(record_interval))
    sample_count = math.ceil(steps / sample_interval)
    time = np.zeros(sample_count)

    y_star, u_star = model.get_steady_state(T_target)
    y_current = y0.copy()
    u_current = np.zeros(N)
    y_hist = np.zeros((sample_count, 2 * N))

    q_diag = q_diag_override
    Q_matrix = np.diag(q_diag)
    S_matrix = model.get_step_response_matrix(horizon)
    QS = Q_matrix @ S_matrix
    if QS.shape[0] == QS.shape[1]:
        try:
            QS_solver = np.linalg.inv(QS)
        except np.linalg.LinAlgError:
            QS_solver = np.linalg.pinv(QS)
    else:
        QS_solver = np.linalg.pinv(QS)

    sample_idx = 0
    for i in range(steps):
        delta_u = model.mpc_step(
            y_current,
            u_current,
            y_star,
            horizon=horizon,
            q_diag=q_diag,
            p_max=params.P_MAX,
            step_response=S_matrix,
            q_matrix=Q_matrix,
            qs_solver=QS_solver,
            u_target=u_star,
        )
        u_current = u_current + delta_u

        noise_val = noise_values[i]
        env_noise = np.zeros_like(y_current)
        env_noise[:N] = noise_mask * noise_val

        if i % sample_interval == 0 and sample_idx < sample_count:
            time[sample_idx] = i * dt
            y_hist[sample_idx] = y_current
            sample_idx += 1

        dy = model.sys.A @ y_current + model.sys.B @ u_current + model.sys.E + env_noise
        y_current += dy * dt

    if sample_idx < sample_count:
        time[sample_idx:] = time[sample_idx - 1]
        y_hist[sample_idx:] = y_current

    return time, y_hist


def run_simulations_noisy():
    os.makedirs("figures_writting/sim_outputs", exist_ok=True)

    NX = NY = 5
    model = ThermalModel2D(NX, NY, coupling_mode="nearest")
    N = NX * NY

    Q_T_W, Q_E_W, R_W = 100.0, 0.1, 0.001
    K, _ = model.design_lqr(Q_T_W, Q_E_W, R_W)

    dt = 0.05
    duration = 360.0

    center_idx, inner_ring, edge_indices = build_category_indices(NX, NY)

    T_target = np.ones((NY, NX)) * params.TEMP_AMBIENT
    T_target_flat = T_target.flatten()
    target_indices = [center_idx] + inner_ring
    for idx in target_indices:
        y, x = divmod(idx, NX)
        T_target[y, x] = 80.0

    q_diag = np.zeros(2 * N)
    q_diag[target_indices] = 1.0

    noise_mask = np.zeros(N)
    noise_mask[edge_indices] = 1.0

    rng = np.random.default_rng(7)
    y0_amb, _ = model.get_steady_state(np.ones((NY, NX)) * params.TEMP_AMBIENT)

    horizons = [5.0, 125.0]

    steps = int(duration / dt)
    noise_sequences = generate_noise_sequences(steps, dt, rng)

    for noise_label, noise_values in noise_sequences.items():
        fig, axes = plt.subplots(1 + len(horizons), 1, figsize=(8, 10), sharex=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        t_lqr, y_lqr = simulate_lqr_with_noise(
            model=model,
            K=K,
            y0=y0_amb,
            T_target=T_target,
            noise_values=noise_values,
            noise_mask=noise_mask,
            dt=dt,
            duration=duration,
        )
        axes[0].set_title(f"LQR with {noise_label} noise")
        plot_categories(axes[0], t_lqr, y_lqr[:, :N], center_idx, inner_ring, edge_indices, target_temp=80.0)

        for ax, horizon in zip(axes[1:], horizons):
            t_mpc, y_mpc = simulate_mpc_with_noise(
                model=model,
                y0=y0_amb,
                T_target=T_target,
                horizon=horizon,
                noise_values=noise_values,
                noise_mask=noise_mask,
                q_diag_override=q_diag,
                dt=dt,
                duration=duration,
                record_interval=1,
            )
            ax.set_title(f"MPC (H={horizon:.0f}s) with {noise_label} noise")
            plot_categories(ax, t_mpc, y_mpc[:, :N], center_idx, inner_ring, edge_indices, target_temp=80.0)

        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        fig.savefig(f"figures_writting/sim_outputs/sim_noise_{noise_label}.png")
        plt.close(fig)


if __name__ == "__main__":
    run_simulations_noisy()

