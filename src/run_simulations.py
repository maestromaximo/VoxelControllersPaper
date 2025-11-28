import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import math
from typing import Callable, Optional
from src.model import ThermalModel2D
import src.parameters as params

DEFAULT_MPC_HORIZON = 1.0
MPC_HORIZON_AVG = 5.0
MPC_HORIZON_MAX = 60.0
ALEJANDRO_M_VALUES = [5, 120]
DAVIS_HORIZON_VALUES = [5.0, 25.0, 125.0]
DAVIS_ZOOM_WINDOW = 200.0

PredictionOverrideFn = Callable[[ThermalModel2D, np.ndarray, np.ndarray, float, float], np.ndarray]


def _uniform_weights(count: int) -> np.ndarray:
    weights = np.ones(count, dtype=float)
    return weights / weights.sum()


def _exp_weights(count: int) -> np.ndarray:
    idx = np.arange(1, count + 1, dtype=float)
    raw = np.exp((idx / count) - 1.0)
    return raw / raw.sum()


def make_weighted_prediction_fn(count: int,
                                weight_builder: Callable[[int], np.ndarray]) -> PredictionOverrideFn:
    weights = weight_builder(count)

    def _predict(model: ThermalModel2D,
                 y_current: np.ndarray,
                 u_current: np.ndarray,
                 horizon: float,
                 dt: float) -> np.ndarray:
        lookahead_times = np.linspace(dt, horizon, count)
        weighted_prediction = np.zeros_like(y_current)
        for weight, lookahead in zip(weights, lookahead_times):
            weighted_prediction += weight * model.predict_state(y_current, u_current, lookahead)
        return weighted_prediction

    return _predict

def run_simulation():
    os.makedirs("figures_writting/sim_outputs", exist_ok=True)
    
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.0,
        'legend.fontsize': 12,
    })
    
    NX, NY = 5, 5
    model_nearest = ThermalModel2D(NX, NY, coupling_mode='nearest')
    model_distance = ThermalModel2D(NX, NY, coupling_mode='distance')
    
    N = NX * NY
    
    Q_T_W, Q_E_W, R_W = 100.0, 0.1, 0.001
    K_nearest, _ = model_nearest.design_lqr(Q_T_W, Q_E_W, R_W)
    K_distance, _ = model_distance.design_lqr(Q_T_W, Q_E_W, R_W)
    MPC_HORIZON = DEFAULT_MPC_HORIZON
    
    dt = 0.05 
    duration = 1500.0
    steps = int(duration / dt)
    time = np.linspace(0, duration, steps)
    
    def simulate(model, K, y0, T_target_map, saturation=True):
        
        y_star, u_star = model.get_steady_state(T_target_map)
        y_current = y0.copy()
        y_hist = np.zeros((steps, 2*N))
        u_hist_sat = np.zeros((steps, N))
        u_hist_unsat = np.zeros((steps, N))
        
        for i in range(steps):
            y_tilde = y_current - y_star
            delta_u = -K @ y_tilde
            
            u_unsat = u_star + delta_u
            if saturation:
                u_applied = np.clip(u_unsat, 0.0, params.P_MAX)
            else:
                u_applied = u_unsat
            
            y_hist[i] = y_current
            u_hist_sat[i] = u_applied
            u_hist_unsat[i] = u_unsat
            
            dy = model.sys.A @ y_current + model.sys.B @ u_applied + model.sys.E
            y_current += dy * dt
            
        return time, y_hist, u_hist_sat, u_hist_unsat, y_star, u_star

    def simulate_mpc(model,
                     y0,
                     T_target_map,
                     horizon=MPC_HORIZON,
                     q_temp_weight=1.0,
                     q_energy_weight=0.05,
                     steps_override=None,
                     dt_override=None,
                     record_interval=1,
                     debug: bool = False,
                     debug_label: str = "",
                     prediction_override_fn: Optional[PredictionOverrideFn] = None,
                     capture_diagnostics: bool = False,
                     q_diag_override: Optional[np.ndarray] = None,
                     zero_ambient_q: bool = True):
        
        y_star, u_star = model.get_steady_state(T_target_map)
        local_steps = steps_override if steps_override is not None else steps
        local_dt = dt_override if dt_override is not None else dt
        sample_interval = max(1, int(record_interval))
        sample_count = math.ceil(local_steps / sample_interval)
        local_time = np.zeros(sample_count)

        y_current = y0.copy()
        u_current = np.zeros(N)
        y_hist = np.zeros((sample_count, 2 * N))
        u_hist = np.zeros((sample_count, N))
        if capture_diagnostics:
            eC_hist = np.zeros((sample_count, 2 * N))
            delta_hist = np.zeros((sample_count, N))
        else:
            eC_hist = None
            delta_hist = None

        if q_diag_override is not None:
            q_diag = q_diag_override
        else:
            if zero_ambient_q:
                temp_mask = (np.abs(T_target_map.flatten() - params.TEMP_AMBIENT) > 1e-6).astype(float)
            else:
                temp_mask = np.ones(N)
            q_temp_diag = temp_mask * q_temp_weight
            q_energy_diag = np.zeros(N)
            q_diag = np.concatenate([q_temp_diag, q_energy_diag])
        if q_diag.shape[0] != 2 * N:
            raise ValueError("q_diag_override must have length 2N (temperature + energy states).")
        Q_matrix = np.diag(q_diag)
        S_matrix = model.get_step_response_matrix(horizon)
        QS = Q_matrix @ S_matrix
        if QS.shape[0] == QS.shape[1]:
            try:
                QS_solver = np.linalg.inv(QS)
                print("Inverse found")
            except np.linalg.LinAlgError:
                print("Singular matrix, using pseudoinverse")
                QS_solver = np.linalg.pinv(QS)
        else:
            print("Rectangular QS; using pseudoinverse")
            QS_solver = np.linalg.pinv(QS)

        sample_idx = 0
        request_error = debug or capture_diagnostics
        for i in range(local_steps):
            if prediction_override_fn is not None:
                y_pred_override = prediction_override_fn(
                    model=model,
                    y_current=y_current,
                    u_current=u_current,
                    horizon=horizon,
                    dt=local_dt,
                )
            else:
                y_pred_override = None
            if request_error:
                delta_u, e_C = model.mpc_step(
                    y_current,
                    u_current,
                    y_star,
                    horizon=horizon,
                    q_diag=q_diag,
                    p_max=params.P_MAX,
                    step_response=S_matrix,
                    q_matrix=Q_matrix,
                    qs_solver=QS_solver,
                    y_pred_override=y_pred_override,
                    u_target=u_star,
                    return_error=True,
                )
                if debug and i < 5:
                    label = debug_label or "MPC"
                    print(
                        f"[{label}] step {i}: max|e_C|={np.abs(e_C).max():.2f}, "
                        f"max delta={delta_u.max():.2f}, min delta={delta_u.min():.2f}"
                    )
            else:
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
                    y_pred_override=y_pred_override,
                    u_target=u_star,
                )
                e_C = None
            u_candidate = u_current + delta_u
            # u_current = np.clip(u_candidate, 0.0, params.P_MAX)
            u_current = u_candidate

            if i % sample_interval == 0 and sample_idx < sample_count:
                y_hist[sample_idx] = y_current
                u_hist[sample_idx] = u_current
                local_time[sample_idx] = i * local_dt
                if capture_diagnostics and e_C is not None:
                    eC_hist[sample_idx] = e_C
                    delta_hist[sample_idx] = delta_u
                sample_idx += 1

            dy = model.sys.A @ y_current + model.sys.B @ u_current + model.sys.E
            y_current += dy * local_dt

        y_hist[-1] = y_current
        u_hist[-1] = u_current
        local_time[-1] = (local_steps - 1) * local_dt

        if capture_diagnostics:
            diag_data = {
                "time": local_time.copy(),
                "error": eC_hist,
                "delta": delta_hist,
            }
            return local_time, y_hist, u_hist, y_star, diag_data

        return local_time, y_hist, u_hist, y_star

    y0_amb, _ = model_nearest.get_steady_state(np.ones((NY, NX)) * params.TEMP_AMBIENT)
    
    print("Re-running Standard Sims (Nearest)...")
    T_target_1 = np.ones((NY, NX)) * params.TEMP_AMBIENT
    center_idx = (NY // 2, NX // 2)
    T_target_1[center_idx] = 100.0
    t1, y1, u1_sat, u1_unsat, _, _ = simulate(model_nearest, K_nearest, y0_amb, T_target_1)
    
    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(8, 10))
    center_flat = center_idx[0]*NX + center_idx[1]
    neighbor_flat = center_flat + 1
    ax1a.plot(t1, y1[:, center_flat], label='Center Voxel')
    ax1a.plot(t1, y1[:, neighbor_flat], label='Neighbor Voxel')
    ax1a.axhline(100, color='k', linestyle=':', alpha=0.5)
    ax1a.set_title('Sim 1: Nearest Neighbor - Center Step Response')
    ax1a.set_ylabel('Temperature (°C)')
    ax1a.set_xlabel('Time (s)')
    ax1a.legend()
    ax1a.grid(True)
    
    ax1b.plot(t1, u1_sat[:, center_flat], label='Center Input')
    ax1b.plot(t1, u1_sat[:, neighbor_flat], label='Neighbor Input')
    ax1b.set_title('Control Inputs')
    ax1b.set_ylabel('Power (W)')
    ax1b.set_xlabel('Time (s)')
    ax1b.legend()
    ax1b.grid(True)
    fig1.tight_layout()
    fig1.savefig('figures_writting/sim_outputs/sim1_center_step.png')
    plt.close(fig1)

    t1_mpc, y1_mpc, u1_mpc, _ = simulate_mpc(model_nearest, y0_amb, T_target_1)
    fig1_mpc, (ax1m_a, ax1m_b) = plt.subplots(2, 1, figsize=(8, 10))
    ax1m_a.plot(t1_mpc, y1_mpc[:, center_flat], label='Center Voxel (MPC)')
    ax1m_a.plot(t1_mpc, y1_mpc[:, neighbor_flat], label='Neighbor Voxel (MPC)')
    ax1m_a.axhline(100, color='k', linestyle=':', alpha=0.5)
    ax1m_a.set_title('Sim 1 (MPC): Center Step Response')
    ax1m_a.set_ylabel('Temperature (°C)')
    ax1m_a.set_xlabel('Time (s)')
    ax1m_a.legend()
    ax1m_a.grid(True)

    ax1m_b.plot(t1_mpc, u1_mpc[:, center_flat], label='Center Input (MPC)')
    ax1m_b.plot(t1_mpc, u1_mpc[:, neighbor_flat], label='Neighbor Input (MPC)')
    ax1m_b.set_title('MPC Control Inputs')
    ax1m_b.set_ylabel('Power (W)')
    ax1m_b.set_xlabel('Time (s)')
    ax1m_b.legend()
    ax1m_b.grid(True)
    fig1_mpc.tight_layout()
    fig1_mpc.savefig('figures_writting/sim_outputs/sim1_center_step_mpc.png')
    plt.close(fig1_mpc)

    T_target_2 = np.ones((NY, NX)) * 100.0
    t2, y2, u2_sat, u2_unsat, _, _ = simulate(model_nearest, K_nearest, y0_amb, T_target_2)
    
    fig2, (ax2a, ax2b, ax2c) = plt.subplots(3, 1, figsize=(8, 12))
    ax2a.plot(t2, y2[:, 0], label='Corner Voxel')
    ax2a.plot(t2, y2[:, center_flat], label='Center Voxel', linestyle='--')
    ax2a.axhline(100, color='k', linestyle=':', alpha=0.5)
    ax2a.set_title('Sim 2: Nearest Neighbor - Uniform Step (20°C -> 100°C)')
    ax2a.set_ylabel('Temperature (°C)')
    ax2a.set_xlabel('Time (s)')
    ax2a.legend()
    ax2a.grid(True)
    
    ax2b.plot(t2, u2_sat[:, 0], label='Corner Input (Saturated)')
    ax2b.plot(t2, u2_sat[:, center_flat], label='Center Input (Saturated)')
    ax2b.axhline(params.P_MAX, color='r', linestyle='--', label='Saturation Limit')
    ax2b.set_title('Control Inputs with Saturation')
    ax2b.set_ylabel('Power (W)')
    ax2b.set_xlabel('Time (s)')
    ax2b.legend()
    ax2b.grid(True)

    ax2c.plot(t2, u2_unsat[:, 0], label='Corner Input (Unsaturated)')
    ax2c.plot(t2, u2_unsat[:, center_flat], label='Center Input (Unsaturated)')
    ax2c.set_title('Unconstrained LQR Inputs (Not Physically Realizable)')
    ax2c.set_ylabel('Power (W)')
    ax2c.set_xlabel('Time (s)')
    ax2c.legend()
    ax2c.grid(True)

    fig2.tight_layout()
    fig2.savefig('figures_writting/sim_outputs/sim2_uniform_sat.png')
    plt.close(fig2)

    t2_mpc, y2_mpc, u2_mpc, _ = simulate_mpc(model_nearest, y0_amb, T_target_2)
    fig2_mpc, (ax2m_a, ax2m_b) = plt.subplots(2, 1, figsize=(8, 10))
    ax2m_a.plot(t2_mpc, y2_mpc[:, 0], label='Corner Voxel (MPC)')
    ax2m_a.plot(t2_mpc, y2_mpc[:, center_flat], label='Center Voxel (MPC)', linestyle='--')
    ax2m_a.axhline(100, color='k', linestyle=':', alpha=0.5)
    ax2m_a.set_title('Sim 2 (MPC): Uniform Step (20°C -> 100°C)')
    ax2m_a.set_ylabel('Temperature (°C)')
    ax2m_a.set_xlabel('Time (s)')
    ax2m_a.legend()
    ax2m_a.grid(True)

    ax2m_b.plot(t2_mpc, u2_mpc[:, 0], label='Corner Input (MPC)')
    ax2m_b.plot(t2_mpc, u2_mpc[:, center_flat], label='Center Input (MPC)')
    ax2m_b.axhline(params.P_MAX, color='r', linestyle='--', label='Saturation Limit')
    ax2m_b.set_title('MPC Control Inputs with Saturation')
    ax2m_b.set_ylabel('Power (W)')
    ax2m_b.set_xlabel('Time (s)')
    ax2m_b.legend()
    ax2m_b.grid(True)

    fig2_mpc.tight_layout()
    fig2_mpc.savefig('figures_writting/sim_outputs/sim2_uniform_sat_mpc.png')
    plt.close(fig2_mpc)

    T_target_3 = np.ones((NY, NX)) * params.TEMP_AMBIENT
    for x in range(NX):
        if x < NX // 2:
            T_target_3[:, x] = 100.0
        else:
            T_target_3[:, x] = 20.0
    t3, y3, u3_sat, u3_unsat, _, _ = simulate(model_nearest, K_nearest, y0_amb, T_target_3)
    
    fig3, axes = plt.subplots(3, 2, figsize=(10, 12))
    T_final = y3[-1, :N].reshape((NY, NX))
    
    def plot_heatmap(ax, data, title, cbar_label, vmin=None, vmax=None, cmap='inferno'):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('Voxel X')
        ax.set_ylabel('Voxel Y')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
        return im

    plot_heatmap(axes[0,0], T_final, 'Final Temp (Nearest)', 'Temp (°C)', 20, 100)
    plot_heatmap(axes[0,1], T_target_3, 'Target Temp', 'Temp (°C)', 20, 100)

    plot_heatmap(axes[1,0], u3_sat[-1, :].reshape((NY, NX)),
                 'Steady State Power (Saturated)', 'Power (W)',
                 0, params.P_MAX, cmap='plasma')
    plot_heatmap(axes[1,1], T_final - T_target_3,
                 'Error (Actual - Target)', 'Error (°C)',
                 -5, 5, cmap='RdBu_r')

    plot_heatmap(axes[2,0], u3_unsat[-1, :].reshape((NY, NX)),
                 'Unconstrained LQR Power', 'Power (W)',
                 None, None, cmap='plasma')
    plot_heatmap(axes[2,1], (u3_unsat[-1, :] - u3_sat[-1, :]).reshape((NY, NX)),
                 'Clipping (Unsat - Sat)', 'Power (W)',
                 None, None, cmap='RdBu_r')
    
    fig3.tight_layout()
    fig3.savefig('figures_writting/sim_outputs/sim3_gradient.png')
    plt.close(fig3)

    t3_mpc, y3_mpc, u3_mpc, _ = simulate_mpc(model_nearest, y0_amb, T_target_3)
    fig3_mpc, axes_mpc = plt.subplots(3, 2, figsize=(10, 12))
    T_final_mpc = y3_mpc[-1, :N].reshape((NY, NX))

    plot_heatmap(axes_mpc[0,0], T_final_mpc, 'Final Temp (MPC)', 'Temp (°C)', 20, 100)
    plot_heatmap(axes_mpc[0,1], T_target_3, 'Target Temp', 'Temp (°C)', 20, 100)
    plot_heatmap(axes_mpc[1,0], u3_mpc[-1, :].reshape((NY, NX)),
                 'Steady State Power (MPC)', 'Power (W)', 0, params.P_MAX, cmap='plasma')
    plot_heatmap(axes_mpc[1,1], T_final_mpc - T_target_3,
                 'Error (Actual - Target)', 'Error (°C)', -5, 5, cmap='RdBu_r')
    plot_heatmap(axes_mpc[2,0], u3_mpc[-1, :].reshape((NY, NX)),
                 'MPC Power Map', 'Power (W)', None, None, cmap='plasma')
    plot_heatmap(axes_mpc[2,1], T_final_mpc - T_final,
                 'Temp Diff (MPC - LQR)', 'Temp (°C)', -5, 5, cmap='PRGn')

    fig3_mpc.tight_layout()
    fig3_mpc.savefig('figures_writting/sim_outputs/sim3_gradient_mpc.png')
    plt.close(fig3_mpc)

    print("Running Sim 3b: Long-Term Gradient (30 min)...")
    duration_30m = 1800.0
    steps_30m = int(duration_30m / dt)
    
    y_current = y0_amb.copy()
    y_star, u_star = model_nearest.get_steady_state(T_target_3)
    
    for i in range(steps_30m):
        y_tilde = y_current - y_star
        delta_u = -K_nearest @ y_tilde
        u_applied = u_star + delta_u
        u_applied = np.clip(u_applied, 0.0, params.P_MAX)
        dy = model_nearest.sys.A @ y_current + model_nearest.sys.B @ u_applied + model_nearest.sys.E
        y_current += dy * dt

    fig3b, axes = plt.subplots(2, 2, figsize=(10, 8))
    T_final_30m = y_current[:N].reshape((NY, NX))
    
    plot_heatmap(axes[0,0], T_final_30m, 'Final Temp (30 min)', 'Temp (°C)', 20, 100)
    plot_heatmap(axes[0,1], T_target_3, 'Target Temp', 'Temp (°C)', 20, 100)
    y_tilde = y_current - y_star
    u_final_30m = np.clip(u_star - K_nearest @ y_tilde, 0.0, params.P_MAX)
    plot_heatmap(axes[1,0], u_final_30m.reshape((NY, NX)), 'Steady State Power', 'Power (W)', 0, params.P_MAX, cmap='plasma')
    plot_heatmap(axes[1,1], T_final_30m - T_target_3, 'Error (30 min)', 'Error (°C)', -5, 5, cmap='RdBu_r')
    
    fig3b.tight_layout()
    fig3b.savefig('figures_writting/sim_outputs/sim3b_gradient_30m.png')
    plt.close(fig3b)

    t3b_mpc, y3b_mpc, u3b_mpc, _ = simulate_mpc(model_nearest, y0_amb, T_target_3,
                                                steps_override=steps_30m, dt_override=dt)
    T_final_mpc_30m = y3b_mpc[-1, :N].reshape((NY, NX))
    fig3b_mpc, axes_b = plt.subplots(2, 2, figsize=(10, 8))
    plot_heatmap(axes_b[0,0], T_final_mpc_30m, 'Final Temp (MPC, 30 min)', 'Temp (°C)', 20, 100)
    plot_heatmap(axes_b[0,1], T_target_3, 'Target Temp', 'Temp (°C)', 20, 100)
    plot_heatmap(axes_b[1,0], u3b_mpc[-1, :].reshape((NY, NX)),
                 'Steady State Power (MPC)', 'Power (W)', 0, params.P_MAX, cmap='plasma')
    plot_heatmap(axes_b[1,1], T_final_mpc_30m - T_target_3, 'Error (30 min)', 'Error (°C)',
                 -5, 5, cmap='RdBu_r')
    fig3b_mpc.tight_layout()
    fig3b_mpc.savefig('figures_writting/sim_outputs/sim3b_gradient_30m_mpc.png')
    plt.close(fig3b_mpc)

    print("Running Distance-Based Sims...")
    
    t4, y4, u4_sat, u4_unsat, _, _ = simulate(model_distance, K_distance, y0_amb, T_target_1)
    t4_mpc, y4_mpc, u4_mpc, _ = simulate_mpc(model_distance, y0_amb, T_target_1)
    
    print("Running Long Duration Sim...")
    duration_long = 1500.0 # 25 minutes
    steps_long = int(duration_long / dt)
    time_long = np.linspace(0, duration_long, steps_long)
    
    def simulate_long(model, K, y0, T_target_map):
        y_star, u_star = model.get_steady_state(T_target_map)
        y_current = y0.copy()
        y_hist = np.zeros((steps_long, 2*N))
        
        for i in range(steps_long):
            y_tilde = y_current - y_star
            delta_u = -K @ y_tilde
            u_applied = u_star + delta_u
            u_applied = np.clip(u_applied, 0.0, params.P_MAX)
            y_hist[i] = y_current
            dy = model.sys.A @ y_current + model.sys.B @ u_applied + model.sys.E
            y_current += dy * dt
        return time_long, y_hist

    t_long_near, y_long_near = simulate_long(model_nearest, K_nearest, y0_amb, T_target_1)
    t_long_dist, y_long_dist = simulate_long(model_distance, K_distance, y0_amb, T_target_1)
    t_long_near_mpc, y_long_near_mpc, _, _ = simulate_mpc(model_nearest, y0_amb, T_target_1,
                                                          steps_override=steps_long, dt_override=dt)
    t_long_dist_mpc, y_long_dist_mpc, _, _ = simulate_mpc(model_distance, y0_amb, T_target_1,
                                                          steps_override=steps_long, dt_override=dt)

    fig6, ax6 = plt.subplots(figsize=(8, 6))
    ax6.plot(t_long_near, y_long_near[:, 0], 'b-', label='Nearest: Corner')
    ax6.plot(t_long_dist, y_long_dist[:, 0], 'r--', label='Distance: Corner')
    ax6.set_title('Long Duration (1500s) Far-Field Response')
    ax6.set_ylabel('Temperature (°C)')
    ax6.set_xlabel('Time (s)')
    ax6.legend()
    ax6.grid(True)
    fig6.tight_layout()
    fig6.savefig('figures_writting/sim_outputs/sim6_long_duration.png')
    plt.close(fig6)

    fig6_mpc, ax6_mpc = plt.subplots(figsize=(8, 6))
    ax6_mpc.plot(t_long_near_mpc, y_long_near_mpc[:, 0], 'b-', label='Nearest: Corner (MPC)')
    ax6_mpc.plot(t_long_dist_mpc, y_long_dist_mpc[:, 0], 'r--', label='Distance: Corner (MPC)')
    ax6_mpc.set_title('MPC Long Duration (1500s) Far-Field Response')
    ax6_mpc.set_ylabel('Temperature (°C)')
    ax6_mpc.set_xlabel('Time (s)')
    ax6_mpc.legend()
    ax6_mpc.grid(True)
    fig6_mpc.tight_layout()
    fig6_mpc.savefig('figures_writting/sim_outputs/sim6_long_duration_mpc.png')
    plt.close(fig6_mpc)

    print("Running Sim 7: Impossible Cold Trap...")
    T_target_7 = np.ones((NY, NX)) * 100.0
    T_target_7[center_idx] = 20.0
    q_diag_sim7 = np.zeros(2 * N)
    q_diag_sim7[:N] = 1.0

    y_star_7, u_star_7 = model_nearest.get_steady_state(T_target_7)
    y_current = y0_amb.copy()
    y7 = np.zeros((steps, 2 * N))
    u7_sat = np.zeros((steps, N))
    u7_unsat = np.zeros((steps, N))

    for i in range(steps):
        y_tilde = y_current - y_star_7
        delta_u = -K_nearest @ y_tilde

        u_unsat = u_star_7 + delta_u
        u_sat = np.clip(u_unsat, 0.0, params.P_MAX)

        y7[i] = y_current
        u7_sat[i] = u_sat
        u7_unsat[i] = u_unsat

        dy = model_nearest.sys.A @ y_current + model_nearest.sys.B @ u_sat + model_nearest.sys.E
        y_current += dy * dt

    t7 = time
    
    fig7, (ax7a, ax7b, ax7c) = plt.subplots(3, 1, figsize=(8, 12))
    ax7a.plot(t7, y7[:, center_flat], label='Center Voxel (Target 20°C)')
    ax7a.plot(t7, y7[:, neighbor_flat], label='Neighbor Voxel (Target 100°C)')
    ax7a.axhline(20.0, color='g', linestyle='--', label='Center Target')
    ax7a.axhline(100.0, color='r', linestyle='--', label='Neighbor Target')
    ax7a.set_title('Sim 7: The "Impossible" Cold Center')
    ax7a.set_ylabel('Temperature (°C)')
    ax7a.set_xlabel('Time (s)')
    ax7a.legend()
    ax7a.grid(True)
    
    ax7b.plot(t7, u7_sat[:, center_flat], label='Center Input (Saturated)')
    ax7b.plot(t7, u7_sat[:, neighbor_flat], label='Neighbor Input (Saturated)')
    ax7b.set_ylabel('Power (W)')
    ax7b.set_xlabel('Time (s)')
    ax7b.set_title('Control Inputs with Saturation (Center rails at 0W)')
    ax7b.legend()
    ax7b.grid(True)

    ax7c.plot(t7, u7_unsat[:, center_flat], label='Center Input (Unsaturated)')
    ax7c.plot(t7, u7_unsat[:, neighbor_flat], label='Neighbor Input (Unsaturated)')
    ax7c.set_ylabel('Power (W)')
    ax7c.set_xlabel('Time (s)')
    ax7c.set_title('Unconstrained LQR Inputs (Not Physically Realizable)')
    ax7c.legend()
    ax7c.grid(True)
    
    fig7.tight_layout()
    fig7.savefig('figures_writting/sim_outputs/sim7_impossible_trap.png')
    plt.close(fig7)

    t7_mpc, y7_mpc, u7_mpc, _ = simulate_mpc(
        model_nearest,
        y0_amb,
        T_target_7,
        horizon=60.0,
        debug=True,
        debug_label="Sim7",
        q_diag_override=q_diag_sim7,
    )
    center_idx_flat = center_flat
    neighbor_idx_flat = neighbor_flat
    neighbor_coords = divmod(neighbor_idx_flat, NX)
    print(
        f"Sim 7 LQR final temps: center={y7[-1, center_idx_flat]:.2f}°C, "
        f"neighbor={y7[-1, neighbor_idx_flat]:.2f}°C; "
        f"inputs center={u7_sat[-1, center_idx_flat]:.2f}W, "
        f"neighbor={u7_sat[-1, neighbor_idx_flat]:.2f}W"
    )
    print(
        f"Sim 7 MPC final temps: center={y7_mpc[-1, center_idx_flat]:.2f}°C, "
        f"neighbor={y7_mpc[-1, neighbor_idx_flat]:.2f}°C; "
        f"inputs center={u7_mpc[-1, center_idx_flat]:.2f}W, "
        f"neighbor={u7_mpc[-1, neighbor_idx_flat]:.2f}W"
    )
    print(
        f"Sim 7 targets: center={T_target_7[center_idx]:.1f}°C, "
        f"neighbor={T_target_7[neighbor_coords]:.1f}°C"
    )
    print(
        f"Sim 7 MPC peak inputs: center={u7_mpc[:, center_idx_flat].max():.2f}W, "
        f"neighbor={u7_mpc[:, neighbor_idx_flat].max():.2f}W"
    )
    fig7_mpc, (ax7m_a, ax7m_b) = plt.subplots(2, 1, figsize=(8, 10))
    center_color = (0.85, 0.1, 0.1)
    inner_color = (0.48, 0.24, 0.8)
    edge_color = (0.0, 0.35, 0.9)
    legend_flags_temp = set()

    def _voxel_category(idx: int) -> str:
        y, x = divmod(idx, NX)
        cy, cx = center_idx
        if (y, x) == center_idx:
            return "center"
        if max(abs(y - cy), abs(x - cx)) == 1:
            return "inner"
        return "edge"

    for voxel_idx in range(N):
        category = _voxel_category(voxel_idx)
        if category == "center":
            color = center_color
            label = 'Center Voxel'
        elif category == "inner":
            color = inner_color
            label = 'Inner Ring Voxel'
        else:
            color = edge_color
            label = 'Edge Voxel'
        if label in legend_flags_temp:
            label = None
        else:
            legend_flags_temp.add(label if label is not None else "")
        ax7m_a.plot(
            t7_mpc,
            y7_mpc[:, voxel_idx],
            color=color,
            alpha=0.95 if category == "center" else 0.75 if category == "inner" else 0.6,
            linewidth=1.6 if category == "center" else 1.2 if category == "inner" else 0.9,
            label=label,
        )
    ax7m_a.axhline(20.0, color='g', linestyle='--', label='Center Target')
    ax7m_a.axhline(100.0, color='r', linestyle='--', label='Neighbor Target')
    ax7m_a.set_title('Sim 7 (MPC): Cold Trap')
    ax7m_a.set_ylabel('Temperature (°C)')
    ax7m_a.set_xlabel('Time (s)')
    ax7m_a.legend()
    ax7m_a.grid(True)

    legend_flags_power = set()
    for voxel_idx in range(N):
        category = _voxel_category(voxel_idx)
        if category == "center":
            color = center_color
            label = 'Center Input'
        elif category == "inner":
            color = inner_color
            label = 'Inner Ring Input'
        else:
            color = edge_color
            label = 'Edge Input'
        if label in legend_flags_power:
            label = None
        else:
            legend_flags_power.add(label if label is not None else "")
        ax7m_b.plot(
            t7_mpc,
            u7_mpc[:, voxel_idx],
            color=color,
            alpha=0.95 if category == "center" else 0.75 if category == "inner" else 0.6,
            linewidth=1.6 if category == "center" else 1.2 if category == "inner" else 0.9,
            label=label,
        )
    ax7m_b.set_ylabel('Power (W)')
    ax7m_b.set_xlabel('Time (s)')
    ax7m_b.set_title('MPC Control Inputs')
    ax7m_b.legend()
    ax7m_b.grid(True)

    fig7_mpc.tight_layout()
    fig7_mpc.savefig('figures_writting/sim_outputs/sim7_impossible_trap_mpc.png')
    plt.close(fig7_mpc)

    print("Running Sim 8: Long-Term Stability (1 Hour)...")
    duration_stability = 3600.0 # 1 hour
    steps_stab = int(duration_stability / dt)
    time_stab = np.linspace(0, duration_stability, steps_stab)
    save_interval = 100
    
    T_target_8 = np.zeros((NY, NX))
    for x in range(NX):
        if x % 2 == 0:
            T_target_8[:, x] = 80.0
        else:
            T_target_8[:, x] = 40.0
                
    # Reuse simulate_long logic but for this duration/target
    def simulate_stability(model, K, y0, T_target_map):
        y_star, u_star = model.get_steady_state(T_target_map)
        y_current = y0.copy()
        y_hist = np.zeros((steps_stab // save_interval, 2*N))
        t_hist = time_stab[::save_interval]
        
        for i in range(steps_stab):
            y_tilde = y_current - y_star
            delta_u = -K @ y_tilde
            u_applied = u_star + delta_u
            u_applied = np.clip(u_applied, 0.0, params.P_MAX)
            
            if i % save_interval == 0:
                idx = i // save_interval
                if idx < len(y_hist):
                    y_hist[idx] = y_current
            
            dy = model.sys.A @ y_current + model.sys.B @ u_applied + model.sys.E
            y_current += dy * dt
            
        return t_hist, y_hist

    t8, y8 = simulate_stability(model_nearest, K_nearest, y0_amb, T_target_8)
    
    fig8, (ax8a, ax8b) = plt.subplots(2, 1, figsize=(8, 10))
    # Plot a few random voxels
    ax8a.plot(t8, y8[:, 0], label='Corner (Target 80°C)')
    ax8a.plot(t8, y8[:, 1], label='Neighbor (Target 40°C)')
    ax8a.plot(t8, y8[:, 12], label='Center (Target 80°C)')
    ax8a.set_title('Sim 8: Long-Term Stability (1 Hour)')
    ax8a.set_ylabel('Temperature (°C)')
    ax8a.set_xlabel('Time (s)')
    ax8a.legend()
    ax8a.grid(True)
    ax8a.set_ylim(35, 85) # Zoom in to see steadiness
    
    # Plot Error Norm
    T_target_flat = T_target_8.flatten()
    T_hist = y8[:, :N]
    error_norm = np.linalg.norm(T_hist - T_target_flat, axis=1) / np.sqrt(N) # RMS Error
    
    ax8b.plot(t8, error_norm, 'k-')
    ax8b.set_title('RMS Temperature Error over 1 Hour')
    ax8b.set_ylabel('RMS Error (°C)')
    ax8b.set_xlabel('Time (s)')
    ax8b.grid(True)
    
    fig8.tight_layout()
    fig8.savefig('figures_writting/sim_outputs/sim8_stability.png')
    plt.close(fig8)

    # MPC counterpart for Sim 8
    t8_mpc, y8_mpc, _, _ = simulate_mpc(
        model_nearest,
        y0_amb,
        T_target_8,
        steps_override=steps_stab,
        dt_override=dt,
        record_interval=save_interval,
    )

    fig8_mpc, (ax8m_a, ax8m_b) = plt.subplots(2, 1, figsize=(8, 10))
    ax8m_a.plot(t8_mpc, y8_mpc[:, 0], label='Corner (MPC)')
    ax8m_a.plot(t8_mpc, y8_mpc[:, 1], label='Neighbor (MPC)')
    ax8m_a.plot(t8_mpc, y8_mpc[:, 12], label='Center (MPC)')
    ax8m_a.set_title('Sim 8 (MPC): Long-Term Stability')
    ax8m_a.set_ylabel('Temperature (°C)')
    ax8m_a.set_xlabel('Time (s)')
    ax8m_a.legend()
    ax8m_a.grid(True)
    ax8m_a.set_ylim(35, 85)

    T_hist_mpc = y8_mpc[:, :N]
    error_norm_mpc = np.linalg.norm(T_hist_mpc - T_target_flat, axis=1) / np.sqrt(N)
    ax8m_b.plot(t8_mpc, error_norm_mpc, 'k-')
    ax8m_b.set_title('MPC RMS Temperature Error (1 Hour)')
    ax8m_b.set_ylabel('RMS Error (°C)')
    ax8m_b.set_xlabel('Time (s)')
    ax8m_b.grid(True)

    fig8_mpc.tight_layout()
    fig8_mpc.savefig('figures_writting/sim_outputs/sim8_stability_mpc.png')
    plt.close(fig8_mpc)

    print("Running Sim 9: LQR Energy Cost Analysis (q_E > 0 vs q_E = 0)...")

    T_target_energy = np.ones((NY, NX)) * params.TEMP_AMBIENT
    T_target_energy[center_idx] = 100.0
    y_star_energy, u_star_energy = model_nearest.get_steady_state(T_target_energy)

    Q_E_zero = 0.0
    Q_E_heavy = 0.01
    K_no_energy, _ = model_nearest.design_lqr(Q_T_W, Q_E_zero, R_W)
    K_with_energy, _ = model_nearest.design_lqr(Q_T_W, Q_E_heavy, R_W)

    tE, y_noE, _, u_noE, _, _ = simulate(
        model_nearest, K_no_energy, y0_amb, T_target_energy, saturation=False
    )
    _, y_withE, _, u_withE, _, _ = simulate(
        model_nearest, K_with_energy, y0_amb, T_target_energy, saturation=False
    )

    center_idx_flat = center_flat
    T_center_noE = y_noE[:, center_idx_flat]
    T_center_withE = y_withE[:, center_idx_flat]

    figE, axE_temp = plt.subplots(figsize=(7.5, 4.5))
    axE_temp.plot(tE, T_center_noE, label=r"$q_E = 0$")
    axE_temp.plot(tE, T_center_withE, label=rf"$q_E = {Q_E_heavy:.2f}$")
    axE_temp.axhline(100.0, color='k', linestyle=':', alpha=0.6, label='Target 100°C')
    axE_temp.set_title('Sim 9: Center Step LQR Tracking vs Energy Cost')
    axE_temp.set_ylabel('Center Temperature (°C)')
    axE_temp.set_xlabel('Time (s)')
    axE_temp.set_xlim(0.0, 5.0)
    axE_temp.set_ylim(99.0, 105.5)
    axE_temp.grid(True)
    axE_temp.legend()

    figE.tight_layout()
    figE.savefig('figures_writting/sim_outputs/sim_lqr_energy_cost_analysis.png')
    plt.close(figE)

    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(8, 10))
    
    # Center & Neighbor Response
    ax4a.plot(t1, y1[:, center_flat], 'b-', label='Nearest: Center')
    ax4a.plot(t4, y4[:, center_flat], 'r--', label='Distance: Center')
    ax4a.plot(t1, y1[:, neighbor_flat], 'b:', alpha=0.6, label='Nearest: Neighbor')
    ax4a.plot(t4, y4[:, neighbor_flat], 'r:', alpha=0.6, label='Distance: Neighbor')
    
    ax4a.set_title('Sim 4: Center Step Comparison')
    ax4a.set_ylabel('Temperature (°C)')
    ax4a.set_xlabel('Time (s)')
    ax4a.legend()
    ax4a.grid(True)
    
    # Far Field Response
    corner_flat = 0
    ax4b.plot(t1, y1[:, corner_flat], 'b-', label='Nearest: Corner')
    ax4b.plot(t4, y4[:, corner_flat], 'r--', label='Distance: Corner')
    ax4b.set_title('Far-Field Response (Corner Voxel)')
    ax4b.set_ylabel('Temperature (°C)')
    ax4b.set_xlabel('Time (s)')
    ax4b.legend()
    ax4b.grid(True)
    
    fig4.tight_layout()
    fig4.savefig('figures_writting/sim_outputs/sim4_comparison_step.png')
    plt.close(fig4)

    fig4_mpc, (ax4m_a, ax4m_b) = plt.subplots(2, 1, figsize=(8, 10))
    ax4m_a.plot(t1_mpc, y1_mpc[:, center_flat], 'b-', label='Nearest (MPC): Center')
    ax4m_a.plot(t4_mpc, y4_mpc[:, center_flat], 'r--', label='Distance (MPC): Center')
    ax4m_a.plot(t1_mpc, y1_mpc[:, neighbor_flat], 'b:', alpha=0.6, label='Nearest (MPC): Neighbor')
    ax4m_a.plot(t4_mpc, y4_mpc[:, neighbor_flat], 'r:', alpha=0.6, label='Distance (MPC): Neighbor')
    ax4m_a.set_title('Sim 4 (MPC): Center Step Comparison')
    ax4m_a.set_ylabel('Temperature (°C)')
    ax4m_a.set_xlabel('Time (s)')
    ax4m_a.legend()
    ax4m_a.grid(True)

    ax4m_b.plot(t1_mpc, y1_mpc[:, 0], 'b-', label='Nearest (MPC): Corner')
    ax4m_b.plot(t4_mpc, y4_mpc[:, 0], 'r--', label='Distance (MPC): Corner')
    ax4m_b.set_title('Far-Field Response (MPC)')
    ax4m_b.set_ylabel('Temperature (°C)')
    ax4m_b.set_xlabel('Time (s)')
    ax4m_b.legend()
    ax4m_b.grid(True)

    fig4_mpc.tight_layout()
    fig4_mpc.savefig('figures_writting/sim_outputs/sim4_comparison_step_mpc.png')
    plt.close(fig4_mpc)
    
    t5, y5, u5_sat, u5_unsat, _, _ = simulate(model_distance, K_distance, y0_amb, T_target_3)
    t5_mpc, y5_mpc, u5_mpc, _ = simulate_mpc(model_distance, y0_amb, T_target_3)
    
    fig5, axes = plt.subplots(2, 2, figsize=(10, 8))
    T_final_dist = y5[-1, :N].reshape((NY, NX))
    
    plot_heatmap(axes[0,0], T_final_dist, 'Final Temp (Distance)', 'Temp (°C)', 20, 100)
    
    err_dist = T_final_dist - T_target_3
    plot_heatmap(axes[0,1], err_dist, 'Error (Distance)', 'Error (°C)', -5, 5, cmap='RdBu_r')
    
    err_nearest = y3[-1, :N].reshape((NY, NX)) - T_target_3
    plot_heatmap(axes[1,0], err_nearest, 'Error (Nearest)', 'Error (°C)', -5, 5, cmap='RdBu_r')
    
    diff_map = T_final_dist - y3[-1, :N].reshape((NY, NX))
    plot_heatmap(axes[1,1], diff_map, 'Diff (Distance - Nearest)', 'Temp Diff (°C)', -2, 2, cmap='PRGn')
    
    fig5.tight_layout()
    fig5.savefig('figures_writting/sim_outputs/sim5_comparison_gradient.png')
    plt.close(fig5)

    fig5_mpc, axes_m = plt.subplots(2, 2, figsize=(10, 8))
    T_final_dist_mpc = y5_mpc[-1, :N].reshape((NY, NX))

    plot_heatmap(axes_m[0,0], T_final_dist_mpc, 'Final Temp (Distance, MPC)', 'Temp (°C)', 20, 100)
    err_dist_mpc = T_final_dist_mpc - T_target_3
    plot_heatmap(axes_m[0,1], err_dist_mpc, 'Error (Distance, MPC)', 'Error (°C)', -5, 5, cmap='RdBu_r')
    plot_heatmap(axes_m[1,0], T_final - T_target_3, 'Error (Nearest, LQR)', 'Error (°C)', -5, 5, cmap='RdBu_r')
    diff_map_mpc = T_final_dist_mpc - T_final_mpc
    plot_heatmap(axes_m[1,1], diff_map_mpc, 'Diff (Distance - Nearest, MPC)', 'Temp Diff (°C)', -2, 2, cmap='PRGn')

    fig5_mpc.tight_layout()
    fig5_mpc.savefig('figures_writting/sim_outputs/sim5_comparison_gradient_mpc.png')
    plt.close(fig5_mpc)

    print("Running Alejandro-requested MPC studies...")
    horizon_configs = [
        ("avg", MPC_HORIZON_AVG),
        ("max", MPC_HORIZON_MAX),
    ]
    horizon_results = []
    for label, horizon_value in horizon_configs:
        t_h, y_h, u_h, _ = simulate_mpc(
            model_nearest,
            y0_amb,
            T_target_1,
            horizon=horizon_value,
            record_interval=1,
            zero_ambient_q=False,
        )
        horizon_results.append((label, horizon_value, t_h, y_h, u_h))

    figH, (axH_temp, axH_power) = plt.subplots(2, 1, figsize=(8, 10))
    for label, horizon_value, t_h, y_h, u_h in horizon_results:
        legend_label = f"{label.title()} Horizon (H={horizon_value:.1f}s)"
        axH_temp.plot(t_h, y_h[:, center_flat], label=legend_label)
        axH_power.plot(t_h, u_h[:, center_flat], label=legend_label)
    axH_temp.set_title('MPC Center Voxel Temperature vs. Horizon Choice')
    axH_temp.set_ylabel('Temperature (°C)')
    axH_temp.set_xlabel('Time (s)')
    axH_temp.grid(True)
    axH_temp.legend()

    axH_power.set_title('MPC Center Heater Power vs. Horizon Choice')
    axH_power.set_ylabel('Power (W)')
    axH_power.set_xlabel('Time (s)')
    axH_power.grid(True)
    axH_power.legend()
    figH.tight_layout()
    figH.savefig('figures_writting/sim_outputs/sim_alejandro_horizon_compare_mpc.png')
    plt.close(figH)

    print("Running Davis-requested MPC horizon sweep (distance coupling)...")
    q_diag_davis = np.zeros(2 * N)
    q_diag_davis[center_flat] = 1.0
    davis_results = []
    for horizon_value in DAVIS_HORIZON_VALUES:
        sim_result = simulate_mpc(
            model_distance,
            y0_amb,
            T_target_1,
            horizon=horizon_value,
            record_interval=1,
            capture_diagnostics=True,
            q_diag_override=q_diag_davis,
        )
        t_d, y_d, u_d, _, diag_data = sim_result
        davis_results.append((horizon_value, t_d, y_d, u_d, diag_data))

    figD, (axD_full, axD_zoom) = plt.subplots(2, 1, figsize=(9, 10), sharey=True)
    for horizon_value, t_d, y_d, _, _ in davis_results:
        label = f"H={horizon_value:.0f}s"
        axD_full.plot(t_d, y_d[:, center_flat], label=label)

    axD_full.axhline(100.0, color='k', linestyle=':', alpha=0.6, label='Target 100°C')
    axD_full.set_title('MPC Center Temperature vs Horizon')
    axD_full.set_ylabel('Temperature (°C)')
    axD_full.set_xlabel('Time (s)')
    axD_full.grid(True)
    axD_full.legend()

    for horizon_value, t_d, y_d, _, _ in davis_results:
        label = f"H={horizon_value:.0f}s"
        zoom_start = max(t_d[0], t_d[-1] - DAVIS_ZOOM_WINDOW)
        mask = t_d >= zoom_start
        axD_zoom.plot(t_d[mask], y_d[mask, center_flat], label=label)

    axD_zoom.axhline(100.0, color='k', linestyle=':', alpha=0.6)
    axD_zoom.set_title(f'Zoom on Final {DAVIS_ZOOM_WINDOW:.0f} s')
    axD_zoom.set_ylabel('Temperature (°C)')
    axD_zoom.set_xlabel('Time (s)')
    axD_zoom.grid(True)
    axD_zoom.legend()

    figD.tight_layout()
    figD.savefig('figures_writting/sim_outputs/sim_davisrequest_horizon_scan.png')
    plt.close(figD)

    figD_power, axD_power = plt.subplots(figsize=(9, 5))
    for horizon_value, t_d, _, u_d, _ in davis_results:
        label = f"H={horizon_value:.0f}s"
        axD_power.plot(t_d, u_d[:, center_flat], label=label)
    axD_power.axhline(params.P_MAX, color='r', linestyle='--', alpha=0.6, label='Saturation')
    axD_power.set_title('Davis Request: Center Heater Power vs Horizon')
    axD_power.set_ylabel('Power (W)')
    axD_power.set_xlabel('Time (s)')
    axD_power.grid(True)
    axD_power.legend()
    figD_power.tight_layout()
    figD_power.savefig('figures_writting/sim_outputs/sim_davisrequest_horizon_scan_power.png')
    plt.close(figD_power)

    figD_error, axD_error = plt.subplots(figsize=(9, 5))
    figD_delta, axD_delta = plt.subplots(figsize=(9, 5))
    for horizon_value, _, _, _, diag in davis_results:
        label = f"H={horizon_value:.0f}s"
        diag_time = diag["time"]
        axD_error.plot(diag_time, diag["error"][:, center_flat], label=label)
        axD_delta.plot(diag_time, diag["delta"][:, center_flat], label=label)
    axD_error.set_title('Davis Request: Center Error Component $e_C$')
    axD_error.set_ylabel('Error Component')
    axD_error.set_xlabel('Time (s)')
    axD_error.grid(True)
    axD_error.legend()
    figD_error.tight_layout()
    figD_error.savefig('figures_writting/sim_outputs/sim_davisrequest_horizon_scan_error.png')
    plt.close(figD_error)

    axD_delta.axhline(0.0, color='k', linestyle='--', alpha=0.4)
    axD_delta.set_title('Davis Request: Center Δu (QS⁻¹ e_C)')
    axD_delta.set_ylabel('Δu (W)')
    axD_delta.set_xlabel('Time (s)')
    axD_delta.grid(True)
    axD_delta.legend()
    figD_delta.tight_layout()
    figD_delta.savefig('figures_writting/sim_outputs/sim_davisrequest_horizon_scan_delta.png')
    plt.close(figD_delta)

    q_diag_alejandro = np.zeros(2 * N)
    q_diag_alejandro[center_flat] = 1.0

    display_horizons = ALEJANDRO_M_VALUES
    figW, axesW = plt.subplots(len(display_horizons), 1, figsize=(8, 8), sharex=True)
    if len(display_horizons) == 1:
        axes_iter = [axesW]
    else:
        axes_iter = axesW

    for ax, horizon_value in zip(axes_iter, display_horizons):
        horizon_seconds = float(horizon_value)
        M_samples = max(1, int(round(horizon_seconds)))
        prediction_fn = make_weighted_prediction_fn(M_samples, _uniform_weights)
        t_weighted, y_weighted, _, _ = simulate_mpc(
            model_nearest,
            y0_amb,
            T_target_1,
            horizon=horizon_seconds,
            prediction_override_fn=prediction_fn,
            record_interval=1,
            q_diag_override=q_diag_alejandro,
        )
        t_base, y_base, _, _ = simulate_mpc(
            model_nearest,
            y0_amb,
            T_target_1,
            horizon=horizon_seconds,
            record_interval=1,
            q_diag_override=q_diag_alejandro,
        )
        ax.plot(t_weighted, y_weighted[:, center_flat], label='Uniform Weighted MPC')
        ax.plot(t_base, y_base[:, center_flat], label='Baseline MPC')
        ax.set_title(f'Uniform vs Baseline (H={horizon_seconds:.0f}s)')
        ax.set_ylabel('Temperature (°C)')
        ax.grid(True)
        ax.legend()
        ax.set_xlim(0, 360.0)
        ax.set_ylim(85.0, 125.0)

    axes_iter[-1].set_xlabel('Time (s)')

    figW.tight_layout()
    figW.savefig('figures_writting/sim_outputs/sim_alejandro_weighted_error_mpc.png')
    plt.close(figW)

    print("Running Sim 10 & 11: Linear Temperature Gradient...")
    print(f"Target gradient range: 40°C to 100°C")
    
    T_target_gradient = np.zeros((NY, NX))
    for y in range(NY):
        for x in range(NX):
            normalized_pos = (y + x) / ((NY - 1) + (NX - 1))
            T_target_gradient[y, x] = 40.0 + 60.0 * normalized_pos
    
    t10, y10, u10_sat, u10_unsat, y_star_10, u_star_10 = simulate(
        model_distance, K_distance, y0_amb, T_target_gradient
    )
    
    fig10, axes10 = plt.subplots(2, 2, figsize=(10, 8))
    T_final_10 = y10[-1, :N].reshape((NY, NX))
    
    plot_heatmap(axes10[0, 0], T_target_gradient, 'Target Gradient', 'Temp (°C)', 40, 100)
    plot_heatmap(axes10[0, 1], T_final_10, 'Final Temp (LQR)', 'Temp (°C)', 40, 100)
    plot_heatmap(axes10[1, 0], u10_sat[-1, :].reshape((NY, NX)),
                 'Steady State Power', 'Power (W)', None, None, cmap='plasma')
    error_10 = T_final_10 - T_target_gradient
    err_max_10 = max(abs(error_10.min()), abs(error_10.max()), 0.01) 
    plot_heatmap(axes10[1, 1], error_10,
                 'Error (Actual - Target)', 'Error (°C)', -err_max_10, err_max_10, cmap='RdBu_r')
    
    fig10.suptitle('Sim 10: LQR Linear Gradient (40°C → 100°C, distance coupling)')
    fig10.tight_layout()
    fig10.savefig('figures_writting/sim_outputs/sim10_gradient_linear_lqr.png')
    plt.close(fig10)
    print(f"Sim 10 LQR power range: {u10_sat[-1, :].min():.3f}W to {u10_sat[-1, :].max():.3f}W")
    print(f"Sim 10 LQR error range: {error_10.min():.6f}°C to {error_10.max():.6f}°C")
    
    q_diag_sim11 = np.zeros(2 * N)
    q_diag_sim11[:N] = 1.0  
    
    t11_mpc, y11_mpc, u11_mpc, _ = simulate_mpc(
        model_distance,
        y0_amb,
        T_target_gradient,
        horizon=MPC_HORIZON_MAX,
        q_diag_override=q_diag_sim11,
    )
    
    fig11, axes11 = plt.subplots(2, 2, figsize=(10, 8))
    T_final_11 = y11_mpc[-1, :N].reshape((NY, NX))
    
    plot_heatmap(axes11[0, 0], T_target_gradient, 'Target Gradient', 'Temp (°C)', 40, 100)
    plot_heatmap(axes11[0, 1], T_final_11, 'Final Temp (MPC)', 'Temp (°C)', 40, 100)
    plot_heatmap(axes11[1, 0], u11_mpc[-1, :].reshape((NY, NX)),
                 'Steady State Power (MPC)', 'Power (W)', None, None, cmap='plasma')
    error_11 = T_final_11 - T_target_gradient
    err_max_11 = max(abs(error_11.min()), abs(error_11.max()), 0.01)  
    plot_heatmap(axes11[1, 1], error_11,
                 'Error (Actual - Target)', 'Error (°C)', -err_max_11, err_max_11, cmap='RdBu_r')
    
    fig11.suptitle('Sim 11: MPC Linear Gradient (40°C → 100°C, distance coupling)')
    fig11.tight_layout()
    fig11.savefig('figures_writting/sim_outputs/sim11_gradient_linear_mpc.png')
    plt.close(fig11)
    print(f"Sim 11 MPC power range: {u11_mpc[-1, :].min():.3f}W to {u11_mpc[-1, :].max():.3f}W")
    print(f"Sim 11 MPC error range: {error_11.min():.6f}°C to {error_11.max():.6f}°C")

    print("Simulations Complete.")

if __name__ == "__main__":
    run_simulation()
