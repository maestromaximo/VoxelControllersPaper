
import numpy as np
import scipy.linalg
from dataclasses import dataclass
from typing import Tuple, List, Optional

import src.parameters as params

@dataclass
class SystemMatrices:
    A: np.ndarray
    B: np.ndarray
    E: np.ndarray
    C_matrices: Tuple[np.ndarray, np.ndarray] 

class ThermalModel2D:
    def __init__(self, nx: int, ny: int, coupling_mode: str = 'nearest', config: Optional[dict] = None):
        """
        coupling_mode: 'nearest' (default) or 'distance' (inverse distance decay)
        """
        self.nx = nx
        self.ny = ny
        self.num_voxels = nx * ny
        self.num_states = 2 * self.num_voxels 
        self.coupling_mode = coupling_mode
        
        if config is None:
            self.config = {
                'k_neighbor': params.K_NEIGHBOR,
                'kappa': params.KAPPA,
                'k_env': params.K_ENV,
                'c_v': params.VOXEL_HEAT_CAPACITY,
                'c_h': params.HEATER_HEAT_CAPACITY,
                'temp_ambient': params.TEMP_AMBIENT,
                'voxel_side_length': params.VOXEL_SIDE_LENGTH
            }
        else:
            self.config = config

        self.idx_map = np.arange(self.num_voxels).reshape((ny, nx))
        
        self.sys = self._build_matrices()
        self._A_inv = np.linalg.inv(self.sys.A)
        self.selector_T = np.hstack([
            np.eye(self.num_voxels),
            np.zeros((self.num_voxels, self.num_voxels))
        ]) #not used anymore
        self._phi_cache = {}
        self._step_response_cache = {}
        
    def _get_voxel_coords(self, idx: int) -> Tuple[int, int]:
        return divmod(idx, self.nx)

    def _get_neighbors_nearest(self, idx: int) -> List[Tuple[int, float]]:
        """Get 4-connected neighbors with coupling strength."""
        neighbors = []
        cy, cx = self._get_voxel_coords(idx)
        
        k_neighbor = self.config['k_neighbor']

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < self.ny and 0 <= nx < self.nx:
                n_idx = self.idx_map[ny, nx]
                neighbors.append((n_idx, k_neighbor))
        return neighbors
        
    def _get_neighbors_distance(self, idx: int) -> List[Tuple[int, float]]:
        """Get ALL other voxels with 1/r^2 coupling (Gauss's Law approximation)."""
        neighbors = []
        cy, cx = self._get_voxel_coords(idx)
        
        
        # k(d) = K_NEIGHBOR * (L / d)^2
        
        L = self.config['voxel_side_length']
        k_neighbor = self.config['k_neighbor']
        
        
        for i in range(self.num_voxels):
            if i == idx:
                continue
                
            iy, ix = self._get_voxel_coords(i)
            
            
            dist_grid = np.sqrt((cy-iy)**2 + (cx-ix)**2)
            dist_meters = dist_grid * L
            
            # k = k0 * (L/d)^2
            k_val = k_neighbor * (L / dist_meters)**2
            neighbors.append((i, k_val))
            
        return neighbors

    def _build_matrices(self) -> SystemMatrices:
        """Construct A, B, E matrices based on the paper's Appendix."""
        N = self.num_voxels
        
        Lambda = np.zeros((N, N))       # T -> T (coupling)
        Lambda_prime = np.zeros((N, N)) # T -> T (self loss)
        Psi = np.zeros((N, N))          # E -> T
        Phi = np.zeros((N, N))          # T -> E
        Delta = np.zeros((N, N))        # E -> E
        
        B_top = np.zeros((N, N))
        B_bot = np.eye(N)
        
        # environmental 
        E_vec = np.zeros(2 * N)
        
        kappa = self.config['kappa']
        k_env = self.config['k_env']
        c_v = self.config['c_v']
        c_h = self.config['c_h']
        temp_ambient = self.config['temp_ambient']
        
        for i in range(N):
            if self.coupling_mode == 'nearest':
                neighbors = self._get_neighbors_nearest(i)
            elif self.coupling_mode == 'distance':
                neighbors = self._get_neighbors_distance(i)
            else:
                raise ValueError(f"Unknown coupling mode: {self.coupling_mode}")
            
            sum_k_neighbors = 0.0
            
            for j_idx, k_val in neighbors:
                Lambda[i, j_idx] = k_val
                sum_k_neighbors += k_val
            
            Lambda_prime[i, i] = -kappa - k_env - sum_k_neighbors
            
            Psi[i, i] = kappa / c_h
            
            Phi[i, i] = kappa * c_h
            
            Delta[i, i] = -kappa
            
            E_vec[i] = k_env * temp_ambient
            
        A_top = np.hstack([Lambda + Lambda_prime, Psi])
        A_bot = np.hstack([Phi, Delta])
        A = np.vstack([A_top, A_bot])
        
        B = np.vstack([B_top, B_bot])
        
        return SystemMatrices(A, B, E_vec, (np.eye(N)*c_v, np.eye(N)*c_h))

    def get_steady_state(self, T_targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        
            u_i^* = c_i^* ( Σ_{j≠i} k_{ij}(T_i^* - T_j^*) + k_i^e (T_i^* - T_i^e) )
            E_i^* = c_i^* ( T_i^* + (1/κ_i)( Σ_{j≠i} k_{ij}(T_i^* - T_j^*) + k_i^e (T_i^* - T_i^e) ) )
        """
        T_star = T_targets.flatten()
        N = self.num_voxels

        kappa = self.config['kappa']
        k_env = self.config['k_env']
        c_h = self.config['c_h']
        temp_env = self.config['temp_ambient']

        if self.coupling_mode == 'nearest':
            neighbor_fn = self._get_neighbors_nearest
        elif self.coupling_mode == 'distance':
            neighbor_fn = self._get_neighbors_distance
        else:
            raise ValueError(f"Unknown coupling mode: {self.coupling_mode}")

        E_star = np.zeros(N)
        u_star = np.zeros(N)

        for idx in range(N):
            T_i_star = T_star[idx]
            coupling_sum = 0.0

            for j_idx, k_val in neighbor_fn(idx):
                coupling_sum += k_val * (T_i_star - T_star[j_idx])

            env_term = k_env * (T_i_star - temp_env)
            total_term = coupling_sum + env_term

            u_star[idx] = c_h * total_term
            E_star[idx] = c_h * (T_i_star + total_term / kappa)

        y_star = np.concatenate([T_star, E_star])
        return y_star, u_star

    def design_lqr(self, Q_temp_weight=1.0, Q_energy_weight=0.0, R_weight=0.01):
        """
        Compute LQR matrix K.
        """
        N = self.num_voxels
        
        Q_T = np.eye(N) * Q_temp_weight
        Q_E = np.eye(N) * Q_energy_weight
        Q = scipy.linalg.block_diag(Q_T, Q_E)
        
        R = np.eye(N) * R_weight
        
        P = scipy.linalg.solve_continuous_are(self.sys.A, self.sys.B, Q, R)
        
        R_inv = np.linalg.inv(R)
        K = R_inv @ self.sys.B.T @ P
        
        return K, P

    @staticmethod
    def bryson_weights(temp_max, energy_max=None, power_max=1.0):
        """
        Compute diagonal weight entries using Bryson's rule.
        """
        temp_max_arr = np.asarray(temp_max, dtype=float)
        if energy_max is None:
            energy_max_arr = temp_max_arr.copy()
        else:
            energy_max_arr = np.asarray(energy_max, dtype=float)
        power_max_arr = np.asarray(power_max, dtype=float)

        q_temp = 1.0 / np.square(temp_max_arr)
        q_energy = 1.0 / np.square(energy_max_arr)
        r_weight = 1.0 / np.square(power_max_arr)

        if q_temp.shape == ():
            q_temp = float(q_temp)
        if q_energy.shape == ():
            q_energy = float(q_energy)
        if r_weight.shape == ():
            r_weight = float(r_weight)

        return q_temp, q_energy, r_weight

    def _phi(self, horizon: float) -> np.ndarray:
        """
        Matrix exponential Φ(H) = e^{A H}.
        """
        if horizon not in self._phi_cache:
            self._phi_cache[horizon] = scipy.linalg.expm(self.sys.A * horizon)
        return self._phi_cache[horizon]

    def get_step_response_matrix(self, horizon: float) -> np.ndarray:
        """
        Cached step response matrix S = B A^{-1} (Φ(H) - I) B.
        """
        if horizon not in self._step_response_cache:
            Phi = self._phi(horizon)
            identity = np.eye(self.sys.A.shape[0])
            self._step_response_cache[horizon] = (
                self._A_inv @ ((Phi - identity) @ self.sys.B)
            )
        return self._step_response_cache[horizon]

    def predict_state(self, y_current: np.ndarray, u_current: np.ndarray, horizon: float) -> np.ndarray:
        """
            y_traj[t+H] = A^{-1}(Φ(H) - I)(B u + E) + Φ(H) y(t)
        """
        Phi = self._phi(horizon)
        identity = np.eye(self.sys.A.shape[0])
        forcing = self.sys.B @ u_current + self.sys.E
        return self._A_inv @ ((Phi - identity) @ forcing) + Phi @ y_current

    def predict_error_state(self,
                            delta_y_current: np.ndarray,
                            delta_u_current: np.ndarray,
                            horizon: float) -> np.ndarray:
        """
       
            δy[t+H] = A^{-1}(Φ(H) - I) B δu(t) + Φ(H) δy(t)
        """
        Phi = self._phi(horizon)
        identity = np.eye(self.sys.A.shape[0])
        forcing = self.sys.B @ delta_u_current
        return self._A_inv @ ((Phi - identity) @ forcing) + Phi @ delta_y_current

    def compute_step_response_matrix(self, horizon: float) -> np.ndarray:
        """
            S = B A^{-1} (Φ(H) - I) B
        """
        Phi = self._phi(horizon)
        identity = np.eye(self.sys.A.shape[0])
        return self._A_inv @ ((Phi - identity) @ self.sys.B)

    def mpc_step(self,
                 y_current: np.ndarray,
                 u_current: np.ndarray,
                 y_target: np.ndarray,
                 horizon: float,
                 q_diag: np.ndarray,
                 p_max: float,
                 step_response: Optional[np.ndarray] = None,
                 q_matrix: Optional[np.ndarray] = None,
                 qs_solver: Optional[np.ndarray] = None,
                 y_pred_override: Optional[np.ndarray] = None,
                 u_target: Optional[np.ndarray] = None,
                 return_error: bool = False):
        """
        
            Δu = (Q S)^{-1} e_C
            e_C = Q (y* - y_traj[t+H])
        """
        if step_response is None:
            step_response = self.get_step_response_matrix(horizon)
        if q_matrix is None:
            q_matrix = np.diag(q_diag)

        if u_target is None:
            raise ValueError("mpc_step requires u_target to compute deviation coordinates.")

        delta_y_current = y_current - y_target
        delta_u_current = u_current - u_target

        if y_pred_override is not None:
            delta_y_pred = y_pred_override - y_target
        else:
            delta_y_pred = self.predict_error_state(
                delta_y_current=delta_y_current,
                delta_u_current=delta_u_current,
                horizon=horizon,
            )

        e_C = -q_matrix @ delta_y_pred

        if qs_solver is not None:
            delta_u = qs_solver @ e_C
        else:
            QS = q_matrix @ step_response
            if QS.shape[0] == QS.shape[1]:
                try:
                    delta_u = np.linalg.solve(QS, e_C)
                except np.linalg.LinAlgError:
                    delta_u = np.linalg.pinv(QS) @ e_C
            else:
                delta_u = np.linalg.pinv(QS) @ e_C

        lower = -u_target - delta_u_current
        upper = params.P_MAX - u_target - delta_u_current
        delta_u = np.clip(delta_u, lower, upper)
        
        if return_error:
            return delta_u, e_C
        return delta_u
