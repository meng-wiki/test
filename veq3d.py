import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import time
from types import SimpleNamespace
from typing import Callable, Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------------
# 径向坐标 ρ（与 DESC / PEST 类 flux 坐标一致）
#   ψ_N := Ψ/Ψ_a  归一化环向磁通（磁轴 ψ_N=0，最外闭合面 ψ_N=1）
#   ρ := √ψ_N      本文件中所有 ρ、RHO、径向谱变量均基于此定义，故 ψ_N = ρ²
#   Ψ(ρ) = Ψ_a·ρ²（取轴上 Ψ=0），dΨ/dρ = 2 Ψ_a ρ，与 _get_profiles 的 Phip 一致
# 径向 Chebyshev 节点映射 x = 2ρ²−1 = 2ψ_N−1，即在 ψ_N 上谱离散。
# ---------------------------------------------------------------------------

# 边界目标面 R,Z 为 (theta, zeta) 的函数；
# 默认使用结构化参数字典 + 生成器函数（由 VEQ RZ 解析式在 rho=1 处生成 LCFS）。
_DEFAULT_BOUNDARY_PARAMS = {
    "R0": 10,
    "Z0": 0,
    "c0R": [0, 0, -1.3e-04],  # [n=0, cos(zeta), sin(zeta)]
    "c0Z": [0, 0, -2.9e-01],
    "h": [0, -2.67e-01, 0],
    "v": [0, 0, 0],
    "k": [-1, 2.8e-01, 0],
    "a": [1, 3.15e-03, 0],
    # (m,n,typ): coeff, typ in {"c","s"}, phase = m*th - n*ze
    "tR_modes": {
        (1, -1, "c"): 0,
        (1, -1, "s"): 5.4e-01,
        (1, 0, "c"): 0,
        (1, 0, "s"): 2.4e-05,
        (1, 1, "c"): 0,
        (1, 1, "s"): 1.45e-02,
    },
    "tZ_modes": {
        (1, -1, "c"): 0,
        (1, -1, "s"): 0,
        (1, 0, "c"): 0,
        (1, 0, "s"): 0,
        (1, 1, "c"): 0,
        (1, 1, "s"): 0,
    },
}


def _eval_1d_series(coeffs, ze):
    """评估 1D zeta 级数：c0 + c1c*cos(ze) + c1s*sin(ze) + ..."""
    c = np.asarray(coeffs, dtype=float).reshape(-1)
    out = np.zeros_like(ze, dtype=float)
    if c.size == 0:
        return out
    out = out + c[0]
    idx = 1
    n = 1
    while idx + 1 < c.size:
        out = out + c[idx] * np.cos(n * ze) + c[idx + 1] * np.sin(n * ze)
        idx += 2
        n += 1
    return out


def _eval_2d_modes(mode_dict, th, ze):
    """评估 2D 模态和：sum c_mn * cos/sin(m*th - n*ze)"""
    out = np.zeros_like(th, dtype=float)
    for (m, n, typ), coeff in mode_dict.items():
        phase = m * th - n * ze
        if typ == "c":
            out = out + coeff * np.cos(phase)
        else:
            out = out + coeff * np.sin(phase)
    return out


def _build_boundary_pair_from_params(params):
    """由结构化边界参数生成 (R(th,ze), Z(th,ze)) 函数对。"""
    p = dict(params)

    def R(th: np.ndarray, ze: np.ndarray) -> np.ndarray:
        c0R = _eval_1d_series(p["c0R"], ze)
        h = _eval_1d_series(p["h"], ze)
        a = _eval_1d_series(p["a"], ze)
        tR = _eval_2d_modes(p.get("tR_modes", {}), th, ze)
        thR = th + c0R + tR
        return p["R0"] + a * (h - np.cos(thR))

    def Z(th: np.ndarray, ze: np.ndarray) -> np.ndarray:
        c0Z = _eval_1d_series(p["c0Z"], ze)
        v = _eval_1d_series(p["v"], ze)
        k = _eval_1d_series(p["k"], ze)
        a = _eval_1d_series(p["a"], ze)
        tZ = _eval_2d_modes(p.get("tZ_modes", {}), th, ze)
        thZ = th + c0Z + tZ
        return p["Z0"] + a * (v - k * np.sin(thZ))

    return R, Z


def _compile_boundary_pair(
    expr_R: str, expr_Z: str
) -> Tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """将字符串解析为 (R(th,ze), Z(th,ze))，可用变量：th, ze（别名 theta, zeta），以及 np.*。"""
    globs = {"np": np, "__builtins__": {}}
    code_R = compile(expr_R.strip(), "<boundary_R>", "eval")
    code_Z = compile(expr_Z.strip(), "<boundary_Z>", "eval")

    def R(th: np.ndarray, ze: np.ndarray) -> np.ndarray:
        loc = {"th": th, "ze": ze, "theta": th, "zeta": ze}
        return eval(code_R, globs, loc)

    def Z(th: np.ndarray, ze: np.ndarray) -> np.ndarray:
        loc = {"th": th, "ze": ze, "theta": th, "zeta": ze}
        return eval(code_Z, globs, loc)

    return R, Z


class VEQ3D_Solver:
    """3D 力平衡求解；径向标签 ρ 统一定义为归一化环向磁通的平方根 ρ=√(Ψ/Ψ_a)。"""

    def __init__(
        self,
        boundary_R_expr: Optional[str] = None,
        boundary_Z_expr: Optional[str] = None,
        boundary_fns: Optional[Tuple[Callable, Callable]] = None,
    ):
        if boundary_fns is not None:
            self._boundary_R_fn, self._boundary_Z_fn = boundary_fns
            self.boundary_R_expr = None
            self.boundary_Z_expr = None
        else:
            # 显式给了表达式时，保持旧接口；否则走参数字典生成默认边界
            if boundary_R_expr is not None or boundary_Z_expr is not None:
                R_def, Z_def = _build_boundary_pair_from_params(_DEFAULT_BOUNDARY_PARAMS)
                self.boundary_R_expr = boundary_R_expr
                self.boundary_Z_expr = boundary_Z_expr
                if boundary_R_expr is not None and boundary_Z_expr is not None:
                    self._boundary_R_fn, self._boundary_Z_fn = _compile_boundary_pair(boundary_R_expr, boundary_Z_expr)
                elif boundary_R_expr is not None:
                    R_user, _ = _compile_boundary_pair(boundary_R_expr, "0.0*th")
                    self._boundary_R_fn, self._boundary_Z_fn = R_user, Z_def
                else:
                    _, Z_user = _compile_boundary_pair("0.0*th", boundary_Z_expr)
                    self._boundary_R_fn, self._boundary_Z_fn = R_def, Z_user
            else:
                self.boundary_R_expr = None
                self.boundary_Z_expr = None
                self._boundary_R_fn, self._boundary_Z_fn = _build_boundary_pair_from_params(_DEFAULT_BOUNDARY_PARAMS)

        # =========================================================
        # [自由度扩容] 提升表达能力以包容 3D 边界调制
        # =========================================================
        self.M_pol = 2 
        self.N_tor = 2
        self.L_rad = 4
        # =========================================================
        
        self.N_fp = 19
        # Ψ_a：最外闭合面内总环向磁通（与 ψ_N=Ψ/Ψ_a、ρ=√ψ_N 配套；单位与 Phip=dΨ/dρ 一致）
        self.Phi_a = 1.0
        self.mu_0 = 4 * np.pi * 1e-7
        
        self.target_Nr = 24
        self.target_Nt = 24
        self.target_Nz = 16
        
        self.p_edge = None  
        # det 手性方向（+1/-1）：用于兼容输入 LCFS 与 RZ 参数化方向差异
        self.det_chirality = 1.0
        # 正则项总开关缩放，避免其在物理残差前喧宾夺主
        self.regularization_scale = 0.5
        # 主残差中的剖面约束权重（默认关闭，仅在 _run_optimization 中按 phase 注入）
        self.main_dp_weight = 0.0
        self.main_pressure_edge_weight = 0.0
        self.main_profile_scale_floor = 1.0
        # 各残差组分总权重（可在 projection_cfg 中按 phase 覆盖）
        self.main_phys_weight = 1.0
        self.profile_penalty_weight = 1.0
        self.reg_tR_weight = 1.0
        self.reg_tZ_weight = 1.0
        self.reg_lam_weight = 1.0
        # 线性约束 A x = b（LinearConstraintProjection）；默认无约束
        self.linear_constraint_A = None
        self.linear_constraint_b = None
        self.linear_constraint_tol = 1e-10
        
        self._setup_modes()
        
        self.update_grid(self.target_Nr, self.target_Nt, self.target_Nz)
        self.fit_boundary()  
        self._initialize_scaling()
        self._reset_core_freeze_state()

    def _reset_core_freeze_state(self):
        self.core_fixed_mask = np.zeros(self.num_core_params, dtype=bool)
        self.core_fixed_values = np.zeros(self.num_core_params, dtype=float)

    def _apply_fixed_core(self, x_core):
        x = np.asarray(x_core, dtype=float).copy()
        if np.any(self.core_fixed_mask):
            x[self.core_fixed_mask] = self.core_fixed_values[self.core_fixed_mask]
        return x

    def _iter_core_blocks(self):
        offset = 0
        block_specs = [
            ("c0R", self.len_1d),
            ("c0Z", self.len_1d),
            ("h", self.len_1d),
            ("v", self.len_1d),
            ("k", self.len_1d),
            ("a", self.len_1d),
            ("tR", self.len_2d),
            ("tZ", self.len_2d),
            ("lam", self.len_lam),
        ]
        for name, width in block_specs:
            size = self.L_rad * width
            yield name, offset, width
            offset += size

    def _one_d_mode_map(self):
        out = {(0, "0"): 0}
        idx = 1
        for n in range(1, self.N_tor + 1):
            out[(n, "c")] = idx
            out[(n, "s")] = idx + 1
            idx += 2
        return out

    def _format_core_mode_label(self, block_name, mode_idx):
        if block_name in ("c0R", "c0Z", "h", "v", "k", "a"):
            if mode_idx == 0:
                return f"{block_name}_0"
            n = (mode_idx + 1) // 2
            typ = "c" if (mode_idx % 2 == 1) else "s"
            return f"{block_name}_{n}{typ}"
        if block_name in ("tR", "tZ"):
            m, n, typ = self.modes_2d[mode_idx]
            return f"{block_name}_{m}_{n}{typ}"
        if block_name == "lam":
            m, n = self.lambda_modes[mode_idx]
            return f"L_{m}_{n}"
        return f"{block_name}_{mode_idx}"

    def _collect_frozen_parameter_details(self):
        details = []
        if not np.any(self.core_fixed_mask):
            return details

        for block_name, offset, width in self._iter_core_blocks():
            if width == 0:
                continue
            block_mask = self.core_fixed_mask[offset:offset + self.L_rad * width].reshape(self.L_rad, width)
            block_vals = self.core_fixed_values[offset:offset + self.L_rad * width].reshape(self.L_rad, width)
            for j in range(width):
                fixed_L = np.flatnonzero(block_mask[:, j]).tolist()
                if len(fixed_L) == 0:
                    continue
                label = self._format_core_mode_label(block_name, j)
                if len(fixed_L) == self.L_rad and np.all(np.abs(block_vals[:, j]) < 1e-15):
                    details.append(f"{label:<16} | L=0~{self.L_rad-1} -> 0")
                else:
                    l_desc = ", ".join([f"L={L}" for L in fixed_L])
                    details.append(f"{label:<16} | {l_desc}")
        return details

    def _print_frozen_parameter_details(self):
        details = self._collect_frozen_parameter_details()
        print(">>> 冻结参数明细:")
        if len(details) == 0:
            print("    (无冻结参数)")
            return
        print("-" * 70)
        for line in details:
            print(f"    {line}")
        print("-" * 70)

    def _regularization_active_stats(self):
        stats = {"tR_active": 0, "tR_total": 0, "tZ_active": 0, "tZ_total": 0, "lam_active": 0, "lam_total": 0}
        if self.num_core_params <= 0:
            return stats
        offset = 0
        for name, width in [("c0R", self.len_1d), ("c0Z", self.len_1d), ("h", self.len_1d), ("v", self.len_1d), ("k", self.len_1d), ("a", self.len_1d), ("tR", self.len_2d), ("tZ", self.len_2d), ("lam", self.len_lam)]:
            size = self.L_rad * width
            if name in ("tR", "tZ", "lam") and size > 0:
                m = self.core_fixed_mask[offset:offset + size]
                active = int(np.sum(~m))
                total = int(size)
                if name == "tR":
                    stats["tR_active"], stats["tR_total"] = active, total
                elif name == "tZ":
                    stats["tZ_active"], stats["tZ_total"] = active, total
                else:
                    stats["lam_active"], stats["lam_total"] = active, total
            offset += size
        return stats

    def _build_prefilter_freeze_mask(self, pilot_solver, pilot_x_core, tol=1e-6):
        full_mask = np.zeros(self.num_core_params, dtype=bool)
        full_vals = np.zeros(self.num_core_params, dtype=float)
        pilot_L_check = min(3, pilot_solver.L_rad)

        def should_freeze(pilot_arr, pilot_idx):
            return bool(np.all(np.abs(pilot_arr[:pilot_L_check, pilot_idx]) < tol))

        (
            p_c0R,
            p_c0Z,
            p_h,
            p_v,
            p_k,
            p_a,
            p_tR,
            p_tZ,
            p_lam,
        ) = pilot_solver.unpack_core(pilot_x_core)
        pilot_1d = {
            "c0R": p_c0R,
            "c0Z": p_c0Z,
            "h": p_h,
            "v": p_v,
            "k": p_k,
            "a": p_a,
        }

        pilot_1d_modes = pilot_solver._one_d_mode_map()
        full_1d_modes = self._one_d_mode_map()
        pilot_2d_modes = {mode: i for i, mode in enumerate(pilot_solver.modes_2d)}
        full_2d_modes = {mode: i for i, mode in enumerate(self.modes_2d)}
        pilot_lam_modes = {mode: i for i, mode in enumerate(pilot_solver.lambda_modes)}
        full_lam_modes = {mode: i for i, mode in enumerate(self.lambda_modes)}
        zero_seed_1d = {k: set() for k in pilot_1d.keys()}  # block -> {(n, typ)}
        zero_seed_2d = {"tR": [], "tZ": []}                 # block -> [(m, n, typ)]
        zero_seed_lam = []                                   # [(m, n)]

        # 第一层：对 pilot 与 full 共有的模式做直接判零映射
        for name, offset, width in self._iter_core_blocks():
            if width == 0:
                continue
            block_mask = full_mask[offset:offset + self.L_rad * width].reshape(self.L_rad, width)
            block_vals = full_vals[offset:offset + self.L_rad * width].reshape(self.L_rad, width)

            if name in pilot_1d:
                p_arr = pilot_1d[name]
                for mode_key, j_full in full_1d_modes.items():
                    j_pilot = pilot_1d_modes.get(mode_key)
                    if j_pilot is None:
                        continue
                    if should_freeze(p_arr, j_pilot):
                        block_mask[:, j_full] = True
                        block_vals[:, j_full] = 0.0
                        if mode_key == (0, "0"):
                            zero_seed_1d[name].add((0, "0"))
                        else:
                            n_mode, typ_mode = mode_key
                            zero_seed_1d[name].add((n_mode, typ_mode))
            elif name in ("tR", "tZ"):
                p_arr = p_tR if name == "tR" else p_tZ
                for mode_key, j_full in full_2d_modes.items():
                    j_pilot = pilot_2d_modes.get(mode_key)
                    if j_pilot is None:
                        continue
                    if should_freeze(p_arr, j_pilot):
                        block_mask[:, j_full] = True
                        block_vals[:, j_full] = 0.0
                        zero_seed_2d[name].append(mode_key)
            elif name == "lam":
                for mode_key, j_full in full_lam_modes.items():
                    j_pilot = pilot_lam_modes.get(mode_key)
                    if j_pilot is None:
                        continue
                    if should_freeze(p_lam, j_pilot):
                        block_mask[:, j_full] = True
                        block_vals[:, j_full] = 0.0
                        zero_seed_lam.append(mode_key)

        # 第二层：m,n 外推冻结（把低阶零种子传播到后续高阶 m,n）
        # 1D: 同 typ 下，n_seed 为 0 的低阶项若固定，则 n>=n_seed 的同 typ 也固定。
        for name in pilot_1d.keys():
            seeds = zero_seed_1d[name]
            if len(seeds) == 0:
                continue
            for block_name, offset, width in self._iter_core_blocks():
                if block_name != name or width == 0:
                    continue
                block_mask = full_mask[offset:offset + self.L_rad * width].reshape(self.L_rad, width)
                block_vals = full_vals[offset:offset + self.L_rad * width].reshape(self.L_rad, width)
                for mode_key, j_full in full_1d_modes.items():
                    if mode_key == (0, "0"):
                        if (0, "0") in seeds:
                            block_mask[:, j_full] = True
                            block_vals[:, j_full] = 0.0
                        continue
                    n_full, typ_full = mode_key
                    for n_seed, typ_seed in seeds:
                        if typ_seed == "0":
                            continue
                        if typ_full == typ_seed and n_full >= n_seed:
                            block_mask[:, j_full] = True
                            block_vals[:, j_full] = 0.0
                            break

        # 2D/Lambda 的 n 分支匹配：n=0 仅传播到 n=0；n!=0 按符号分支传播到更高 |n|
        def n_branch_match(n_full, n_seed):
            if n_seed == 0:
                return n_full == 0
            return (np.sign(n_full) == np.sign(n_seed)) and (abs(n_full) >= abs(n_seed))

        # 2D: 同 typ 且 m>=m_seed 且 n 在同分支并更高阶 -> 固定
        for name in ("tR", "tZ"):
            seeds = zero_seed_2d[name]
            if len(seeds) == 0:
                continue
            for block_name, offset, width in self._iter_core_blocks():
                if block_name != name or width == 0:
                    continue
                block_mask = full_mask[offset:offset + self.L_rad * width].reshape(self.L_rad, width)
                block_vals = full_vals[offset:offset + self.L_rad * width].reshape(self.L_rad, width)
                for (m_full, n_full, typ_full), j_full in full_2d_modes.items():
                    for m_seed, n_seed, typ_seed in seeds:
                        if typ_full != typ_seed:
                            continue
                        if (m_full >= m_seed) and n_branch_match(n_full, n_seed):
                            block_mask[:, j_full] = True
                            block_vals[:, j_full] = 0.0
                            break

        # Lambda: 与 2D 相同传播规则（无 typ）
        if len(zero_seed_lam) > 0:
            for block_name, offset, width in self._iter_core_blocks():
                if block_name != "lam" or width == 0:
                    continue
                block_mask = full_mask[offset:offset + self.L_rad * width].reshape(self.L_rad, width)
                block_vals = full_vals[offset:offset + self.L_rad * width].reshape(self.L_rad, width)
                for (m_full, n_full), j_full in full_lam_modes.items():
                    for m_seed, n_seed in zero_seed_lam:
                        if (m_full >= m_seed) and n_branch_match(n_full, n_seed):
                            block_mask[:, j_full] = True
                            block_vals[:, j_full] = 0.0
                            break

        # 第三层：径向 L 阶传播（若低 L 被固定为 0，则后续高 L 也固定）
        for name, offset, width in self._iter_core_blocks():
            if width == 0:
                continue
            block_mask = full_mask[offset:offset + self.L_rad * width].reshape(self.L_rad, width)
            block_vals = full_vals[offset:offset + self.L_rad * width].reshape(self.L_rad, width)
            for j in range(width):
                fixed_from = None
                for L in range(self.L_rad):
                    if block_mask[L, j] and abs(block_vals[L, j]) < 1e-15:
                        fixed_from = L
                        break
                if fixed_from is not None:
                    block_mask[fixed_from:, j] = True
                    block_vals[fixed_from:, j] = 0.0

        return full_mask, full_vals

    def update_grid(self, Nr, Nt_grid, Nz_grid):
        self.Nr = Nr
        self.Nt_grid = Nt_grid
        self.Nz_grid = Nz_grid
        
        rho_nodes, self.rho_weights = self._get_chebyshev_nodes_and_weights(self.Nr)
        # ρ = √ψ_N ∈ (0,1]：Chebyshev 原点在 x∈[-1,1]，映射到 ρ；最外面对应 ψ_N=1、ρ=1
        self.rho = 0.5 * (rho_nodes + 1)
        self.rho_weights *= 0.5

        self.D_matrix = self._get_spectral_diff_matrix(self.rho)
        
        self.k_th = np.fft.fftfreq(self.Nt_grid, d=1.0/self.Nt_grid)[None, :, None]
        self.k_ze = np.fft.fftfreq(self.Nz_grid, d=1.0/self.Nz_grid)[None, None, :]
        
        self.theta = np.linspace(0, 2*np.pi, self.Nt_grid, endpoint=False)
        self.zeta = np.linspace(0, 2*np.pi, self.Nz_grid, endpoint=False)
        self.dtheta = 2 * np.pi / self.Nt_grid
        self.dzeta = 2 * np.pi / self.Nz_grid
        
        self.RHO, self.TH, self.ZE = np.meshgrid(self.rho, self.theta, self.zeta, indexing='ij')
        self.weights_3d = self.rho_weights[:, None, None]

        self._precompute_radial_factors() 
        self._build_basis_matrices()

    def _precompute_radial_factors(self):
        # x = 2ψ_N − 1 = 2ρ² − 1，径向 Chebyshev 自变量为归一化环向磁通 ψ_N=ρ²
        x = 2.0 * self.rho**2 - 1.0
        T_list = []
        dTdx_list = []
        if self.L_rad > 0:
            T_list.append(np.ones_like(x))
            dTdx_list.append(np.zeros_like(x))
        if self.L_rad > 1:
            T_list.append(x)
            dTdx_list.append(np.ones_like(x))
        for l in range(2, self.L_rad):
            T_new = 2.0 * x * T_list[l-1] - T_list[l-2]
            dTdx_new = 2.0 * T_list[l-1] + 2.0 * x * dTdx_list[l-1] - dTdx_list[l-2]
            T_list.append(T_new)
            dTdx_list.append(dTdx_new)
            
        self.T = np.stack(T_list, axis=0) if self.L_rad > 0 else np.empty((0,) + x.shape)
        self.dTdx = np.stack(dTdx_list, axis=0) if self.L_rad > 0 else np.empty((0,) + x.shape)
        
        self.fac_rad = (1.0 - self.rho**2) * self.T
        self.dfac_rad = -2.0 * self.rho * self.T + 4.0 * self.rho * (1.0 - self.rho**2) * self.dTdx
        
        self.fac_lam_eval = (1.0 - self.rho**2)**2 * self.T
        self.fac_lam_proj = (1.0 - self.rho**2) * self.T

        if self.len_lam > 0:
            lam_m_vals = np.array([m for m, n in self.lambda_modes])
            self.rho_m_lam = self.rho[None, :] ** lam_m_vals[:, None]
        else:
            self.rho_m_lam = np.array([])
            
        if self.len_2d > 0:
            m_vals_2d = np.array([m for m, n, typ in self.modes_2d])
            n_vals_2d = np.array([n for m, n, typ in self.modes_2d])
            self.reg_weight_2d = np.sqrt(m_vals_2d**2 + n_vals_2d**2)
        else:
            self.reg_weight_2d = np.array([])

    def _setup_modes(self):
        self.len_1d = 1 + 2 * self.N_tor
        
        self.modes_2d = []
        for m in range(1, self.M_pol + 1):
            for n in range(-self.N_tor, self.N_tor + 1):
                self.modes_2d.append((m, n, 'c'))
                self.modes_2d.append((m, n, 's'))
        self.len_2d = len(self.modes_2d)
        
        self.lambda_modes = []
        for n in range(1, self.N_tor + 1):
            self.lambda_modes.append((0, n))
        for m in range(1, self.M_pol + 2):
            for n in range(-self.N_tor, self.N_tor + 1):
                self.lambda_modes.append((m, n))
        self.len_lam = len(self.lambda_modes)
        
        self.num_geom_params = (6 * self.len_1d + 2 * self.len_2d) * self.L_rad
        self.num_core_params = self.num_geom_params + self.len_lam * self.L_rad
        self.num_edge_params = 2 + 6 * self.len_1d + 2 * self.len_2d

    def _build_basis_matrices(self):
        self.basis_1d_val = np.zeros((self.len_1d, 1, 1, self.Nz_grid))
        self.basis_1d_dz  = np.zeros((self.len_1d, 1, 1, self.Nz_grid))
        self.basis_1d_val[0, 0, 0, :] = 1.0
        idx = 1
        for n in range(1, self.N_tor + 1):
            self.basis_1d_val[idx, 0, 0, :] = np.cos(n * self.ZE[0,0,:]); self.basis_1d_dz[idx, 0, 0, :] = -n * np.sin(n * self.ZE[0,0,:]); idx+=1
            self.basis_1d_val[idx, 0, 0, :] = np.sin(n * self.ZE[0,0,:]); self.basis_1d_dz[idx, 0, 0, :] =  n * np.cos(n * self.ZE[0,0,:]); idx+=1
            
        self.basis_2d_val = np.zeros((self.len_2d, self.Nr, self.Nt_grid, self.Nz_grid))
        self.basis_2d_dr  = np.zeros((self.len_2d, self.Nr, self.Nt_grid, self.Nz_grid))
        self.basis_2d_dth = np.zeros((self.len_2d, self.Nr, self.Nt_grid, self.Nz_grid))
        self.basis_2d_dze = np.zeros((self.len_2d, self.Nr, self.Nt_grid, self.Nz_grid))
        self.basis_2d_edge = np.zeros((self.len_2d, self.Nt_grid, self.Nz_grid))
        
        for i, (m, n, typ) in enumerate(self.modes_2d):
            phase = m * self.TH - n * self.ZE
            rho_m = self.RHO ** m
            phase_edge = m * self.TH[0,:,:] - n * self.ZE[0,:,:]
            
            drho_m = np.zeros_like(self.RHO) if m == 0 else m * (self.RHO ** (m - 1))
                
            if typ == 'c':
                self.basis_2d_val[i, :, :, :] = rho_m * np.cos(phase)
                self.basis_2d_dr[i, :, :, :]  = drho_m * np.cos(phase)
                self.basis_2d_dth[i, :, :, :] = rho_m * (-m * np.sin(phase))
                self.basis_2d_dze[i, :, :, :] = rho_m * (n * np.sin(phase))
                self.basis_2d_edge[i] = np.cos(phase_edge)
            else:
                self.basis_2d_val[i, :, :, :] = rho_m * np.sin(phase)
                self.basis_2d_dr[i, :, :, :]  = drho_m * np.sin(phase)
                self.basis_2d_dth[i, :, :, :] = rho_m * (m * np.cos(phase))
                self.basis_2d_dze[i, :, :, :] = rho_m * (-n * np.cos(phase))
                self.basis_2d_edge[i] = np.sin(phase_edge)
                
        self.basis_lam_val = np.zeros((self.len_lam, 1, self.Nt_grid, self.Nz_grid))
        self.basis_lam_dth = np.zeros((self.len_lam, 1, self.Nt_grid, self.Nz_grid))
        self.basis_lam_dze = np.zeros((self.len_lam, 1, self.Nt_grid, self.Nz_grid))
        for i, (m, n) in enumerate(self.lambda_modes):
            phase = m * self.TH[0,:,:] - n * self.ZE[0,:,:]
            self.basis_lam_val[i, 0, :, :] = np.sin(phase)
            self.basis_lam_dth[i, 0, :, :] = m * np.cos(phase)
            self.basis_lam_dze[i, 0, :, :] = -n * np.cos(phase)

    def unpack_edge(self):
        p = self.p_edge; idx = 2
        def get(L): nonlocal idx; c = p[idx:idx+L]; idx+=L; return c
        return p[0], p[1], get(self.len_1d), get(self.len_1d), get(self.len_1d), get(self.len_1d), get(self.len_1d), get(self.len_1d), get(self.len_2d), get(self.len_2d)

    def unpack_core(self, x_core):
        idx = 0
        def get(L): 
            nonlocal idx
            c = x_core[idx:idx+L*self.L_rad].reshape((self.L_rad, L))
            idx += L * self.L_rad
            return c
        return get(self.len_1d), get(self.len_1d), get(self.len_1d), get(self.len_1d), get(self.len_1d), get(self.len_1d), get(self.len_2d), get(self.len_2d), get(self.len_lam)

    def _get_chebyshev_nodes_and_weights(self, N):
        k = np.arange(N)
        theta = (2 * (N - 1 - k) + 1) * np.pi / (2 * N)
        x_nodes = np.cos(theta)
        w_nodes = np.zeros(N)
        for i in range(N):
            sum_term = 0.0
            for j in range(1, N // 2 + 1):
                sum_term += 2.0 * np.cos(2 * j * theta[i]) / (4 * j**2 - 1)
            w_nodes[i] = (2.0 / N) * (1.0 - sum_term)
        return x_nodes, w_nodes

    def _get_spectral_diff_matrix(self, x):
        n = len(x); D = np.zeros((n, n)); w = np.ones(n)
        for i in range(n):
            for j in range(n):
                if i != j: w[i] *= (x[i] - x[j])
        w = 1.0 / w
        for i in range(n):
            for j in range(n):
                if i != j: D[i, j] = (w[j] / w[i]) / (x[i] - x[j])
            D[i, i] = -np.sum(D[i, :])
        return D

    def _initialize_scaling(self):
        print(f">>> 参数体系已重构: 空间维度 (M={self.M_pol}, N={self.N_tor}, L={self.L_rad})")
        print(f">>> 总核心参数量: {self.num_core_params} 个")
        # 不再使用超大数值缩放，避免梯度容差误判导致提前停止
        self.res_scales = np.ones(self.num_core_params)

    def compute_psi(self, rho):
        """极向磁通 ψ(ρ)，与 _get_profiles 中 psip=dψ/dρ 保持一致。"""
        # 约定：phi(ρ) 为环向磁通，dphi/dρ = Phip = 2*Phi_a*ρ
        #      psi(ρ) 为极向磁通，dpsi/dρ = psip = iota(ρ)*Phip
        # 这里 iota(ρ)=1+1.5ρ²，积分并取 psi(0)=0 得
        # psi(ρ)=Phi_a*(ρ²+0.75ρ⁴)
        return self.Phi_a * (rho**2 + 0.75 * rho**4)

    def _get_profiles(self, rho, pressure_scale_factor):
        """剖面输入：P(ρ), dP/dρ, dphi/dρ(Phip), dpsi/dρ(psip)。"""
        P_scale = 1.8e4
        # p 仅为 ψ_N 的函数：(ψ_N−1)² 在边界 ψ_N=1 处为零
        P = pressure_scale_factor * P_scale * (rho**2 - 1)**2
        dP_drho = pressure_scale_factor * P_scale * 4 * rho * (rho**2 - 1)

        iota = 1.0 + 1.5 * rho**2
        Phip = 2 * rho * self.Phi_a
        psip = iota * Phip
        return P, dP_drho, Phip, psip

    def configure_linear_constraints(self, A: Optional[np.ndarray], b: Optional[np.ndarray]):
        """配置线性等式约束 A x = b；传 None 则清空。"""
        if A is None or b is None:
            self.linear_constraint_A = None
            self.linear_constraint_b = None
            return
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        if A.ndim != 2 or A.shape[1] != self.num_core_params:
            raise ValueError("A 形状必须为 (m, num_core_params)")
        if b.shape[0] != A.shape[0]:
            raise ValueError("b 长度必须与 A 行数一致")
        self.linear_constraint_A = A
        self.linear_constraint_b = b

    def _linear_constraint_project(self, x_core):
        """将 x 投影到 A x = b（最小二乘意义下最近点）。"""
        if self.linear_constraint_A is None or self.linear_constraint_b is None:
            return np.asarray(x_core, dtype=float)
        A = self.linear_constraint_A
        b = self.linear_constraint_b
        x = np.asarray(x_core, dtype=float)
        r = A @ x - b
        if np.linalg.norm(r) <= self.linear_constraint_tol:
            return x
        AA_t = A @ A.T
        lam, *_ = np.linalg.lstsq(AA_t, r, rcond=None)
        return x - A.T @ lam

    def _radial_integrate_forward(self, y, y0=0.0):
        out = np.zeros_like(y, dtype=float)
        out[0] = float(y0)
        for i in range(1, len(y)):
            dr = self.rho[i] - self.rho[i - 1]
            out[i] = out[i - 1] + 0.5 * (y[i] + y[i - 1]) * dr
        return out

    def _surface_mean_tz(self, f_rtz):
        """对每个 rho 磁面做 (theta, zeta) 平均，避免广播分母陷阱。"""
        return np.mean(f_rtz, axis=(1, 2))

    def _profile_projection_data(self, x_core, pressure_scale_factor):
        """构造 proximal 投影所需的重构剖面与目标剖面。"""
        s = self._compute_state_numpy(x_core, pressure_scale_factor)

        dP_est = self._surface_mean_tz(
            s["sqrt_g"] * (s["Jt_sup"] * s["Bz_sup"] - s["Jz_sup"] * s["Bt_sup"]) / self.mu_0
        )
        dP_tar = self._surface_mean_tz(s["dP"])
        P_tar = self._surface_mean_tz(s["P"])

        Phip_est_3d = 2 * np.pi * s["sqrt_g"] * s["Bz_sup"] - s["Lt"]
        psip_est_3d = self.N_fp * (2 * np.pi * s["sqrt_g"] * s["Bt_sup"] + s["Lz"])
        Phip_est = self._surface_mean_tz(Phip_est_3d)
        psip_est = self._surface_mean_tz(psip_est_3d)

        Phip_tar = self._surface_mean_tz(s["Phip"])
        psip_tar = self._surface_mean_tz(s["psip"])

        # rho 网格从 rho[0] > 0 开始，补偿 [0, rho0] 的梯形面积可减小常数偏置
        phi_est = self._radial_integrate_forward(Phip_est, y0=0.5 * float(Phip_est[0]) * self.rho[0])
        psi_est = self._radial_integrate_forward(psip_est, y0=0.5 * float(psip_est[0]) * self.rho[0])
        phi_tar = self.Phi_a * self.rho**2
        psi_tar = self.compute_psi(self.rho)

        # 用目标轴压强作为锚点，由 dP_est 前向积分检查边界 P(1)=0 约束
        P_est = self._radial_integrate_forward(dP_est, y0=float(P_tar[0]))

        return {
            "dP_est": dP_est,
            "dP_tar": dP_tar,
            "P_tar": P_tar,
            "P_est": P_est,
            "Phip_est": Phip_est,
            "Phip_tar": Phip_tar,
            "psip_est": psip_est,
            "psip_tar": psip_tar,
            "phi_est": phi_est,
            "phi_tar": phi_tar,
            "psi_est": psi_est,
            "psi_tar": psi_tar,
        }

    def _apply_proximal_projection(self, x_core, pressure_scale_factor, projection_cfg: Optional[Dict[str, Any]] = None):
        """非线性约束投影：匹配 Phip/psip/dP + 关键边界条件。"""
        cfg = dict(projection_cfg or {})
        if not bool(cfg.get("enabled", False)):
            return np.asarray(x_core, dtype=float), {"applied": False}

        strength = float(cfg.get("strength", 0.1))
        boundary_strength = float(cfg.get("boundary_strength", 0.2))
        anchor_weight = float(cfg.get("anchor_weight", 1e-3))
        use_flux_integral_constraints = bool(cfg.get("use_flux_integral_constraints", False))
        flux_strength = float(cfg.get("flux_strength", strength))
        dp_strength = float(cfg.get("dp_strength", strength))
        p_profile_strength = float(cfg.get("p_profile_strength", 0.0))
        pressure_edge_strength = float(cfg.get("pressure_edge_strength", boundary_strength))
        pressure_axis_strength = float(cfg.get("pressure_axis_strength", 0.0))
        profile_scale_floor = float(cfg.get("profile_scale_floor", 1.0))
        prox_max_nfev = int(cfg.get("prox_max_nfev", 10))
        prox_ftol = float(cfg.get("prox_ftol", 1e-8))

        x_ref = np.asarray(x_core, dtype=float).copy()

        def prox_residual(x_trial):
            d = self._profile_projection_data(x_trial, pressure_scale_factor)
            out = []

            s_p = max(np.sqrt(np.mean(d["Phip_tar"] ** 2)), profile_scale_floor) + 1e-14
            s_s = max(np.sqrt(np.mean(d["psip_tar"] ** 2)), profile_scale_floor) + 1e-14
            s_d = max(np.sqrt(np.mean(d["dP_tar"] ** 2)), profile_scale_floor) + 1e-14
            s_P = max(np.sqrt(np.mean(d["P_tar"] ** 2)), profile_scale_floor) + 1e-14

            out.extend((flux_strength * (d["Phip_est"] - d["Phip_tar"]) / s_p).tolist())
            out.extend((flux_strength * (d["psip_est"] - d["psip_tar"]) / s_s).tolist())
            out.extend((dp_strength * (d["dP_est"] - d["dP_tar"]) / s_d).tolist())

            # 可选：积分型通量约束（默认关闭，避免离散积分误差反向污染 proximal）
            if use_flux_integral_constraints:
                out.extend((strength * (d["phi_est"] - d["phi_tar"]) / (abs(self.Phi_a) + 1e-14)).tolist())
                out.extend((strength * (d["psi_est"] - d["psi_tar"]) / (abs(d["psi_tar"][-1]) + 1e-14)).tolist())
                out.append(boundary_strength * (d["phi_est"][-1] - self.Phi_a) / (abs(self.Phi_a) + 1e-14))
                out.append(boundary_strength * d["psi_est"][0])
            # 推荐策略：强约束 dP + 边界锚点 P(1)=0；P 全剖面仅可选弱约束
            out.append(pressure_edge_strength * d["P_est"][-1] / s_P)
            if pressure_axis_strength > 0:
                out.append(pressure_axis_strength * (d["P_est"][0] - d["P_tar"][0]) / s_P)
            if p_profile_strength > 0:
                out.extend((p_profile_strength * (d["P_est"] - d["P_tar"]) / s_P).tolist())

            if anchor_weight > 0:
                x_scale = np.maximum(np.abs(x_ref), 1.0)
                out.extend((np.sqrt(anchor_weight) * (x_trial - x_ref) / x_scale).tolist())

            return np.asarray(out, dtype=float)

        prox_res0 = prox_residual(x_ref)
        prox_sol = least_squares(
            prox_residual,
            x_ref,
            method="trf",
            xtol=prox_ftol,
            ftol=prox_ftol,
            max_nfev=prox_max_nfev,
            verbose=0,
        )
        x_proj = np.asarray(prox_sol.x, dtype=float)
        prox_res1 = prox_residual(x_proj)
        info = {
            "applied": True,
            "nfev": int(prox_sol.nfev),
            "before_norm": float(np.linalg.norm(prox_res0)),
            "after_norm": float(np.linalg.norm(prox_res1)),
        }
        return x_proj, info

    def fit_boundary(self):
        TH_F, ZE_F = self.TH[0], self.ZE[0]
        R_target = np.asarray(self._boundary_R_fn(TH_F, ZE_F), dtype=float)
        Z_target = np.asarray(self._boundary_Z_fn(TH_F, ZE_F), dtype=float)
        
        def eval_1d_edge(coeffs): return np.sum(coeffs[:, None, None] * self.basis_1d_val[:, 0, ...], axis=0)
        def eval_2d_edge(coeffs): return np.sum(coeffs[:, None, None] * self.basis_2d_edge, axis=0) if self.len_2d > 0 else 0.0

        def boundary_residuals(p):
            R0, Z0 = p[0], p[1]
            idx = 2
            def get(L): nonlocal idx; c = p[idx:idx+L]; idx+=L; return c
            c0R_v = eval_1d_edge(get(self.len_1d))
            c0Z_v = eval_1d_edge(get(self.len_1d))
            h_c, v_c = get(self.len_1d), get(self.len_1d)
            h_v, v_v = eval_1d_edge(h_c), eval_1d_edge(v_c)
            k_v, a_v = eval_1d_edge(get(self.len_1d)), eval_1d_edge(get(self.len_1d))
            tR_v, tZ_v = eval_2d_edge(get(self.len_2d)), eval_2d_edge(get(self.len_2d))
            
            thR = TH_F + c0R_v + tR_v
            thZ = TH_F + c0Z_v + tZ_v
            
            R_mod = R0 + a_v * (h_v - np.cos(thR))
            Z_mod = Z0 + a_v * (v_v - k_v * np.sin(thZ))
            
            res_geom = np.concatenate([(R_mod - R_target).flatten(), (Z_mod - Z_target).flatten()])
            res_reg = np.array([h_c[0], v_c[0]]) * 100.0
            return np.concatenate([res_geom, res_reg])
            
        if self.p_edge is not None and len(self.p_edge) == self.num_edge_params:
            p0 = self.p_edge.copy()
        else:
            p0 = np.zeros(self.num_edge_params)
            p0[0] = 10.0 
            p0[2] = 0 
            p0[2 + self.len_1d] = 0 
            p0[2 + 4 * self.len_1d] = 1.0
            p0[2 + 5 * self.len_1d] = 1.0
        
        res = least_squares(boundary_residuals, p0, method='trf', ftol=1e-12)
        self.p_edge = res.x
        self.det_chirality = self._infer_det_chirality_from_boundary(R_target, Z_target)
        print(f"    >>> 边界拟合完成 (Max Residual: {np.max(np.abs(res.fun)):.3e})")
        print(f"    >>> det 手性方向已锁定: {self.det_chirality:+.0f}")
        x_ref = np.zeros(self.num_core_params)
        ref_state = self._compute_state_numpy(x_ref, pressure_scale_factor=0.0)
        det_ref = ref_state["det_phys"].reshape(-1)
        det_eff_ref = ref_state["det_eff"].reshape(-1)
        print(f"    >>> 参考态 det_phys 中位数: {np.median(det_ref):+.3e}, det_eff 中位数: {np.median(det_eff_ref):+.3e}")

    def _signed_area_theta_loop(self, R_curve, Z_curve):
        R1 = np.asarray(R_curve).reshape(-1)
        Z1 = np.asarray(Z_curve).reshape(-1)
        Rp = np.roll(R1, -1)
        Zp = np.roll(Z1, -1)
        return 0.5 * np.sum(R1 * Zp - Rp * Z1)

    def _infer_det_chirality_from_boundary(self, R_target, Z_target):
        e_R0, e_Z0, e_c0R, e_c0Z, e_h, e_v, e_k, e_a, e_tR, e_tZ = self.unpack_edge()
        TH_F = self.TH[0]
        ZE_F = self.ZE[0]

        c0R_v = np.sum(e_c0R[:, None, None] * self.basis_1d_val[:, 0, ...], axis=0)
        c0Z_v = np.sum(e_c0Z[:, None, None] * self.basis_1d_val[:, 0, ...], axis=0)
        h_v = np.sum(e_h[:, None, None] * self.basis_1d_val[:, 0, ...], axis=0)
        v_v = np.sum(e_v[:, None, None] * self.basis_1d_val[:, 0, ...], axis=0)
        k_v = np.sum(e_k[:, None, None] * self.basis_1d_val[:, 0, ...], axis=0)
        a_v = np.sum(e_a[:, None, None] * self.basis_1d_val[:, 0, ...], axis=0)
        if self.len_2d > 0:
            tR_v = np.sum(e_tR[:, None, None] * self.basis_2d_edge, axis=0)
            tZ_v = np.sum(e_tZ[:, None, None] * self.basis_2d_edge, axis=0)
        else:
            tR_v = 0.0
            tZ_v = 0.0

        thR = TH_F + c0R_v + tR_v
        thZ = TH_F + c0Z_v + tZ_v
        R_fit = e_R0 + a_v * (h_v - np.cos(thR))
        Z_fit = e_Z0 + a_v * (v_v - k_v * np.sin(thZ))

        area_t, area_f = [], []
        for j in range(self.Nz_grid):
            at = self._signed_area_theta_loop(R_target[:, j], Z_target[:, j])
            af = self._signed_area_theta_loop(R_fit[:, j], Z_fit[:, j])
            area_t.append(at)
            area_f.append(af)
        area_t = np.asarray(area_t)
        area_f = np.asarray(area_f)

        mean_t = np.mean(area_t[np.abs(area_t) > 1e-12]) if np.any(np.abs(area_t) > 1e-12) else np.mean(area_t)
        mean_f = np.mean(area_f[np.abs(area_f) > 1e-12]) if np.any(np.abs(area_f) > 1e-12) else np.mean(area_f)
        s_t = 1.0 if mean_t >= 0.0 else -1.0
        s_f = 1.0 if mean_f >= 0.0 else -1.0
        boundary_match = 1.0 if s_t * s_f > 0.0 else -1.0

        # 关键：det 的天然符号由参数化映射决定，不能只看边界曲线同向性
        # 用“当前边界 + 零 core”参考态估计 det 主符号，避免将天然负号误判为坏网格。
        x_ref = np.zeros(self.num_core_params)
        s_ref = self._compute_state_numpy(x_ref, pressure_scale_factor=0.0)
        det_ref = np.asarray(s_ref["det_phys"]).reshape(-1)
        det_ref_nz = det_ref[np.abs(det_ref) > 1e-12]
        if det_ref_nz.size == 0:
            det_sign_ref = 1.0
        else:
            det_sign_ref = 1.0 if np.median(det_ref_nz) >= 0.0 else -1.0

        # 若边界方向存在反手性，则翻转一次；否则沿用参考 det 符号
        return det_sign_ref * boundary_match

    def _build_jax_residual_fn(self, pressure_scale_factor=1.0):
        RHO, TH, ZE = jnp.array(self.RHO), jnp.array(self.TH), jnp.array(self.ZE)
        rho_1d, D_matrix = jnp.array(self.rho), jnp.array(self.D_matrix)
        
        fac_rad, dfac_rad = jnp.array(self.fac_rad), jnp.array(self.dfac_rad)
        fac_lam_eval, fac_lam_proj = jnp.array(self.fac_lam_eval), jnp.array(self.fac_lam_proj)
        rho_m_lam, reg_weight_2d = jnp.array(self.rho_m_lam), jnp.array(self.reg_weight_2d)

        basis_1d_val_slice, basis_1d_dz_slice = jnp.array(self.basis_1d_val[:, 0, 0, :]), jnp.array(self.basis_1d_dz[:, 0, 0, :])
        basis_2d_val, basis_2d_dr = jnp.array(self.basis_2d_val), jnp.array(self.basis_2d_dr)
        basis_2d_dth, basis_2d_dth = jnp.array(self.basis_2d_dth), jnp.array(self.basis_2d_dth) # Typo guard
        basis_2d_dth, basis_2d_dze = jnp.array(self.basis_2d_dth), jnp.array(self.basis_2d_dze)
        
        basis_lam_tz, basis_lam_dth, basis_lam_dze = jnp.array(self.basis_lam_val[:, 0, :, :]), jnp.array(self.basis_lam_dth[:, 0, :, :]), jnp.array(self.basis_lam_dze[:, 0, :, :])
        
        k_th, k_ze, weights_3d = jnp.array(self.k_th), jnp.array(self.k_ze), jnp.array(self.weights_3d)
        res_scales, p_edge = jnp.array(self.res_scales), jnp.array(self.p_edge)
        regularization_scale = float(self.regularization_scale)
        main_dp_weight = float(self.main_dp_weight)
        main_pressure_edge_weight = float(self.main_pressure_edge_weight)
        main_profile_scale_floor = float(self.main_profile_scale_floor)
        main_phys_weight = float(self.main_phys_weight)
        profile_penalty_weight = float(self.profile_penalty_weight)
        reg_tR_weight = float(self.reg_tR_weight)
        reg_tZ_weight = float(self.reg_tZ_weight)
        reg_lam_weight = float(self.reg_lam_weight)
        det_chirality = jnp.array(self.det_chirality)
        
        L_rad, len_1d, len_2d, len_lam = self.L_rad, self.len_1d, self.len_2d, self.len_lam
        N_fp, mu_0, dtheta, dzeta = self.N_fp, self.mu_0, self.dtheta, self.dzeta
        core_fixed_mask_np = np.asarray(self.core_fixed_mask, dtype=bool)
        if core_fixed_mask_np.size != self.num_core_params:
            core_fixed_mask_np = np.zeros(self.num_core_params, dtype=bool)
        off_tR = (6 * len_1d) * L_rad
        off_tZ = off_tR + len_2d * L_rad
        off_lam = off_tZ + len_2d * L_rad
        if len_2d > 0:
            reg_active_tR = jnp.array((~core_fixed_mask_np[off_tR:off_tR + len_2d * L_rad]).reshape((L_rad, len_2d)).astype(float))
            reg_active_tZ = jnp.array((~core_fixed_mask_np[off_tZ:off_tZ + len_2d * L_rad]).reshape((L_rad, len_2d)).astype(float))
        else:
            reg_active_tR = jnp.zeros((L_rad, 0))
            reg_active_tZ = jnp.zeros((L_rad, 0))
        if len_lam > 0:
            reg_active_lam = jnp.array((~core_fixed_mask_np[off_lam:off_lam + len_lam * L_rad]).reshape((L_rad, len_lam)).astype(float))
        else:
            reg_active_lam = jnp.zeros((L_rad, 0))
            
        def jax_unpack_edge(p):
            idx = 2
            def get(L): nonlocal idx; c = p[idx:idx+L]; idx += L; return c
            return p[0], p[1], get(len_1d), get(len_1d), get(len_1d), get(len_1d), get(len_1d), get(len_1d), get(len_2d), get(len_2d)

        def jax_unpack_core(x_core):
            idx = 0
            def get(L): nonlocal idx; c = x_core[idx:idx+L*L_rad].reshape((L_rad, L)); idx += L * L_rad; return c
            return get(len_1d), get(len_1d), get(len_1d), get(len_1d), get(len_1d), get(len_1d), get(len_2d), get(len_2d), get(len_lam)

        def spectral_grad_th(f): return jnp.real(jnp.fft.ifft(1j * k_th * jnp.fft.fft(f, axis=1), axis=1))
        def spectral_grad_ze(f): return jnp.real(jnp.fft.ifft(1j * k_ze * jnp.fft.fft(f, axis=2), axis=2))

        def jax_res_fn(x_core, apply_scaling=True, apply_group_weights=True):
            e_R0, e_Z0, e_c0R, e_c0Z, e_h, e_v, e_k, e_a, e_tR, e_tZ = jax_unpack_edge(p_edge)
            c_c0R, c_c0Z, c_h, c_v, c_k, c_a, c_tR, c_tZ, c_lam = jax_unpack_core(x_core)

            def eval_1d(c_e, c_c):
                core_contrib = jnp.dot(c_c.T, fac_rad)  
                ce_eff = c_e[:, None] + core_contrib    
                val = jnp.dot(ce_eff.T, basis_1d_val_slice)[:, None, :] 
                dz  = jnp.dot(ce_eff.T, basis_1d_dz_slice)[:, None, :]  
                dr_contrib = jnp.dot(c_c.T, dfac_rad)   
                dr  = jnp.dot(dr_contrib.T, basis_1d_val_slice)[:, None, :]
                return val, dr, dz

            def eval_2d(c_e, c_c):
                if len_2d == 0: return 0.0, 0.0, 0.0, 0.0
                core_contrib = jnp.dot(c_c.T, fac_rad)  
                ce_eff = c_e[:, None] + core_contrib    
                val = jnp.sum(ce_eff[:, :, None, None] * basis_2d_val, axis=0)
                dth = jnp.sum(ce_eff[:, :, None, None] * basis_2d_dth, axis=0)
                dz  = jnp.sum(ce_eff[:, :, None, None] * basis_2d_dze, axis=0)
                dr_contrib = jnp.dot(c_c.T, dfac_rad)   
                dr = jnp.sum(dr_contrib[:, :, None, None] * basis_2d_val + ce_eff[:, :, None, None] * basis_2d_dr, axis=0)
                return val, dr, dth, dz

            c0R, c0Rr, c0Rz = eval_1d(e_c0R, c_c0R)
            c0Z, c0Zr, c0Zz = eval_1d(e_c0Z, c_c0Z)
            h, hr, hz = eval_1d(e_h, c_h)
            v, vr, vz = eval_1d(e_v, c_v)
            k, kr, kz = eval_1d(e_k, c_k)
            a, ar, az = eval_1d(e_a, c_a)
            
            tR, tRr, tRth, tRz = eval_2d(e_tR, c_tR)
            tZ, tZr, tZth, tZz = eval_2d(e_tZ, c_tZ)
            
            thR = TH + c0R + tR
            thZ = TH + c0Z + tZ
            thR_r, thR_th, thR_z = c0Rr + tRr, 1.0 + tRth, c0Rz + tRz
            thZ_r, thZ_th, thZ_z = c0Zr + tZr, 1.0 + tZth, c0Zz + tZz
            
            R = e_R0 + a * (h - RHO * jnp.cos(thR))
            Rr = ar * (h - RHO * jnp.cos(thR)) + a * (hr - jnp.cos(thR) + RHO * jnp.sin(thR) * thR_r)
            Rt = a * RHO * jnp.sin(thR) * thR_th
            Rz = az * (h - RHO * jnp.cos(thR)) + a * (hz + RHO * jnp.sin(thR) * thR_z)
            
            Z = e_Z0 + a * (v - k * RHO * jnp.sin(thZ))
            Zr = ar * (v - k * RHO * jnp.sin(thZ)) + a * (vr - kr * RHO * jnp.sin(thZ) - k * jnp.sin(thZ) - k * RHO * jnp.cos(thZ) * thZ_r)
            Zt = -a * k * RHO * jnp.cos(thZ) * thZ_th
            Zz = az * (v - k * RHO * jnp.sin(thZ)) + a * (vz - kz * RHO * jnp.sin(thZ) - k * RHO * jnp.cos(thZ) * thZ_z)

            det_phys = Rr * Zt - Rt * Zr
            det_eff = det_chirality * det_phys
            sqrt_g = (R / N_fp) * det_eff 
            
            g_rr, g_tt = Rr**2 + Zr**2, Rt**2 + Zt**2
            g_zz = Rz**2 + (R/N_fp)**2 + Zz**2 
            g_rt, g_rz, g_tz = Rr*Rt+Zr*Zt, Rr*Rz+Zr*Zz, Rt*Rz+Zt*Zz

            if len_lam > 0:
                lam_ce = jnp.dot(c_lam.T, fac_lam_eval) * rho_m_lam
                Lt = jnp.einsum('mr,mtz->rtz', lam_ce, basis_lam_dth)
                Lz = jnp.einsum('mr,mtz->rtz', lam_ce, basis_lam_dze)
            else:
                Lt, Lz = 0.0, 0.0
            
            P, dP, Phip, psip = self._get_profiles(RHO, pressure_scale_factor)

            Bt_sup = (psip / N_fp - Lz) / (2 * jnp.pi * sqrt_g)
            Bz_sup = (Phip + Lt) / (2 * jnp.pi * sqrt_g)

            Br_sub = g_rt * Bt_sup + g_rz * Bz_sup
            Bt_sub = g_tt * Bt_sup + g_tz * Bz_sup
            Bz_sub = g_tz * Bt_sup + g_zz * Bz_sup

            dBt_drho = jnp.tensordot(D_matrix, Bt_sub, axes=(1, 0))
            dBz_drho = jnp.tensordot(D_matrix, Bz_sub, axes=(1, 0))

            Jz_sup = (dBt_drho - spectral_grad_th(Br_sub)) / sqrt_g
            Jt_sup = (spectral_grad_ze(Br_sub) - dBz_drho) / sqrt_g
            Jr_sup = (spectral_grad_th(Bz_sub) - spectral_grad_ze(Bt_sub)) / sqrt_g

            Jr_phys = Jr_sup / mu_0
            G_rho = dP - sqrt_g * (Jt_sup * Bz_sup - Jz_sup * Bt_sup) / mu_0

            # =================================================================
            # [物理闭环终极修复 1]：完全解析抵消 det_phys，消除磁轴除零奇点
            # =================================================================
            F_rho = G_rho
            F_beta = (Jr_phys / (2 * jnp.pi)) * (Phip + Lt)

            # 提取剔除了 det_phys 后的纯几何积分权重
            metric_w = (R / N_fp) * weights_3d * (dtheta * dzeta)
            
            # 直接构造解析平滑的加权笛卡尔力分量
            GR_w = (Zt * F_rho - Zr * F_beta) * metric_w
            GZ_w = (-Rt * F_rho + Rr * F_beta) * metric_w
            
            term1 = GR_w * (a * RHO * jnp.sin(thR))
            term2 = GZ_w * (-a * k * RHO * jnp.cos(thZ))
            term3 = GR_w * a
            term4 = GZ_w * a
            term5 = GZ_w * (-a * RHO * jnp.sin(thZ))
            term6 = GR_w * (h - RHO * jnp.cos(thR)) + GZ_w * (v - k * RHO * jnp.sin(thZ))
            
            terms_1d = jnp.stack([term1, term2, term3, term4, term5, term6], axis=0) 
            terms_1d_t = jnp.sum(terms_1d, axis=2) 
            terms_1d_tz = jnp.einsum('vrz,mz->vrm', terms_1d_t, basis_1d_val_slice) 
            res_1d = jnp.einsum('lr,vrm->lvm', fac_rad, terms_1d_tz).reshape((L_rad, 6 * len_1d)) 
            
            if len_2d > 0:
                term7 = GR_w * (a * RHO * jnp.sin(thR))
                term8 = GZ_w * (-a * k * RHO * jnp.cos(thZ))
                terms_2d = jnp.stack([term7, term8], axis=0)
                terms_2d_r = jnp.einsum('vrtz,mrtz->vmr', terms_2d, basis_2d_val)
                res_2d = jnp.einsum('lr,vmr->lvm', fac_rad, terms_2d_r).reshape((L_rad, 2 * len_2d))
                res_geom = jnp.concatenate([res_1d, res_2d], axis=1).flatten()
            else:
                res_geom = res_1d.flatten()
                
            res_list = [res_geom]
            
            if len_lam > 0:
                vol_w = metric_w * det_eff
                term_lam = Jr_phys * vol_w
                term_lam_tz = jnp.einsum('rtz,mtz->rm', term_lam, basis_lam_tz) 
                res_lam = jnp.dot(fac_lam_proj, rho_m_lam.T * term_lam_tz)
                res_list.append(res_lam.flatten())
                
            phys_res = jnp.concatenate(res_list)

            if apply_scaling:
                phys_res = phys_res / res_scales
            if apply_group_weights:
                phys_res = main_phys_weight * phys_res

            # 将 dP 剖面约束直接并入主残差，避免“主优化-投影”循环互相抵消
            profile_res_list = []
            if (main_dp_weight > 0.0) or (main_pressure_edge_weight > 0.0):
                dP_recon_3d = sqrt_g * (Jt_sup * Bz_sup - Jz_sup * Bt_sup) / mu_0
                dP_recon_surf = jnp.mean(dP_recon_3d, axis=(1, 2))
                dP_tar_surf = jnp.mean(dP, axis=(1, 2))
                s_d = jnp.maximum(jnp.sqrt(jnp.mean(dP_tar_surf**2)), main_profile_scale_floor) + 1e-14

                if main_dp_weight > 0.0:
                    profile_res_list.append(main_dp_weight * (dP_recon_surf - dP_tar_surf) / s_d)

                if main_pressure_edge_weight > 0.0:
                    P_tar_surf = jnp.mean(P, axis=(1, 2))
                    dr = rho_1d[1:] - rho_1d[:-1]
                    dP_mid = 0.5 * (dP_recon_surf[1:] + dP_recon_surf[:-1])
                    P_edge_est = P_tar_surf[0] + jnp.sum(dP_mid * dr)
                    s_P = jnp.maximum(jnp.sqrt(jnp.mean(P_tar_surf**2)), main_profile_scale_floor) + 1e-14
                    profile_res_list.append(jnp.array([main_pressure_edge_weight * (P_edge_est / s_P)]))
                
            final_res_list = [phys_res]
            if len(profile_res_list) > 0:
                profile_res = jnp.concatenate(profile_res_list)
                if apply_group_weights:
                    profile_res = profile_penalty_weight * profile_res
                final_res_list.append(profile_res)
            
            if len_2d > 0:
                reg_tR_res = (c_tR * reg_weight_2d[None, :] * reg_active_tR * regularization_scale).flatten()
                reg_tZ_res = (c_tZ * reg_weight_2d[None, :] * reg_active_tZ * regularization_scale).flatten()
                if apply_group_weights:
                    reg_tR_res = reg_tR_weight * reg_tR_res
                    reg_tZ_res = reg_tZ_weight * reg_tZ_res
                final_res_list.append(reg_tR_res)
                final_res_list.append(reg_tZ_res)
                
            if len_lam > 0:
                reg_lam_res = (c_lam * reg_active_lam * regularization_scale).flatten()
                if apply_group_weights:
                    reg_lam_res = reg_lam_weight * reg_lam_res
                final_res_list.append(reg_lam_res)
                
            return jnp.concatenate(final_res_list)
            
        return jax_res_fn

    def _run_optimization(self, x0, max_nfev, ftol, pressure_scale_factor=1.0, projection_cfg: Optional[Dict[str, Any]] = None):
        # =================================================================
        # [物理闭环终极修复 3]：卸载冗余 AL 外壳，交付原生雅可比推土机
        # =================================================================
        proj_cfg = dict(projection_cfg or {})
        self._last_projection_cfg = proj_cfg.copy()
        self.main_dp_weight = float(proj_cfg.get("main_dp_weight", 0.0))
        self.main_pressure_edge_weight = float(proj_cfg.get("main_pressure_edge_weight", 0.0))
        self.main_profile_scale_floor = float(proj_cfg.get("main_profile_scale_floor", 1.0))
        self.main_phys_weight = float(proj_cfg.get("main_phys_weight", 1.0))
        self.profile_penalty_weight = float(proj_cfg.get("profile_penalty_weight", 1.0))
        self.reg_tR_weight = float(proj_cfg.get("reg_tR_weight", 1.0))
        self.reg_tZ_weight = float(proj_cfg.get("reg_tZ_weight", 1.0))
        self.reg_lam_weight = float(proj_cfg.get("reg_lam_weight", 1.0))
        jax_res_fn = self._build_jax_residual_fn(pressure_scale_factor)
        use_prox = bool(proj_cfg.get("enabled", False))
        outer_loops = int(proj_cfg.get("outer_loops", 1 if not use_prox else 3))
        lsq_chunk_nfev = int(proj_cfg.get("lsq_chunk_nfev", max_nfev if outer_loops <= 1 else max(20, max_nfev // outer_loops)))
        use_exact_jacobian = bool(proj_cfg.get("use_exact_jacobian", True))
        
        start_time = time.time()
        print(f"    >>> 启动优化求解引擎 (网格: {self.Nr}x{self.Nt_grid}x{self.Nz_grid}, 纯物理驱动模式)")
        print(f"    >>> 雅可比模式: {'JAX 精确雅可比' if use_exact_jacobian else 'SciPy 2-point 数值雅可比'}")
        eval_hist = []
        self._last_eval_history = eval_hist
        if self.linear_constraint_A is None:
            print("    >>> 线性约束投影: 未配置线性 A x = b，跳过 Layer-1")
        else:
            print(f"    >>> 线性约束投影: 启用 Layer-1 (约束数={self.linear_constraint_A.shape[0]})")

        active_idx = np.flatnonzero(~self.core_fixed_mask)
        n_active = int(active_idx.size)
        active_idx_jnp = jnp.array(active_idx, dtype=jnp.int32)
        fixed_mask_jnp = jnp.array(self.core_fixed_mask, dtype=bool)
        fixed_values_jnp = jnp.array(self.core_fixed_values, dtype=float)

        def merge_active_to_full(x_active):
            x_full = self.core_fixed_values.copy() if np.any(self.core_fixed_mask) else np.zeros(self.num_core_params)
            x_full[active_idx] = np.asarray(x_active, dtype=float)
            return x_full

        @jax.jit
        def fun_active_wrapped(x_active_arr):
            x_full = jnp.where(fixed_mask_jnp, fixed_values_jnp, 0.0)
            x_full = x_full.at[active_idx_jnp].set(x_active_arr)
            return jax_res_fn(x_full, apply_scaling=True)

        @jax.jit
        def jac_active_wrapped(x_active_arr):
            return jax.jacfwd(fun_active_wrapped)(x_active_arr)

        def fun_logged(x_arr):
            r = np.array(fun_active_wrapped(jnp.array(x_arr)))
            eval_hist.append(float(np.linalg.norm(r)))
            return r

        def jac_logged(x_arr):
            return np.array(jac_active_wrapped(jnp.array(x_arr)))

        iter_counter = 0

        def iter_callback(intermediate_result):
            nonlocal iter_counter
            iter_counter += 1
            if (iter_counter % 10) != 0:
                return
            x_iter = merge_active_to_full(np.array(intermediate_result.x, dtype=float))
            en_iter = self.metric_global_energy_terms(x_iter, pressure_scale_factor)
            print(
                f"    [Iter {iter_counter:03d}] "
                f"W_mhd={en_iter['total_energy']:.6e} "
                f"(W_B={en_iter['magnetic_energy']:.6e}, W_p={en_iter['pressure_energy']:.6e})"
            )
            rb_iter = self.metric_residual_group_breakdown(x_iter, pressure_scale_factor)
            print("    当前残差组分统计 (L2 范数):")
            for item in rb_iter["groups"]:
                print(
                    f"      - {item['name_cn']:<18}: "
                    f"未加权raw={item['norm_raw']:.6e}, "
                    f"加权后={item['norm_weighted']:.6e}, "
                    f"权重系数={item['effective_weight']:.6e}, "
                    f"维度={item['size']}"
                )
        
        x_curr = self._apply_fixed_core(self._linear_constraint_project(np.array(x0, dtype=float)))
        total_nfev = 0
        res = None

        if n_active == 0:
            print("    >>> 核心参数冻结: 0 个自由变量（全部固定），跳过 least_squares。")
            res = SimpleNamespace(x=x_curr.copy(), nfev=0, success=True, status=1, message="all variables fixed")

        for outer in range(outer_loops if n_active > 0 else 0):
            budget_left = max_nfev - total_nfev
            if budget_left <= 0:
                break
            this_chunk = budget_left if outer == outer_loops - 1 else min(lsq_chunk_nfev, budget_left)
            print(f"    >>> [Outer {outer + 1}/{outer_loops}] 主优化步 (max_nfev={this_chunk})")
            try:
                x_active0 = x_curr[active_idx]
                res = least_squares(
                    fun_logged,
                    x_active0,
                    jac=jac_logged if use_exact_jacobian else '2-point',
                    method='trf',
                    xtol=ftol,
                    ftol=ftol,
                    max_nfev=this_chunk,
                    verbose=2,
                    callback=iter_callback
                )
            except Exception as exc:
                err_msg = str(exc)
                oom_like = (
                    "RESOURCE_EXHAUSTED" in err_msg
                    or "Out of memory" in err_msg
                    or "out of memory" in err_msg
                )
                if use_exact_jacobian and oom_like:
                    use_exact_jacobian = False
                    print("    >>> [JacobianFallback] 检测到 JAX 雅可比内存不足，降级到 SciPy 2-point 数值雅可比继续。")
                    res = least_squares(
                        fun_logged,
                        x_active0,
                        jac='2-point',
                        method='trf',
                        xtol=ftol,
                        ftol=ftol,
                        max_nfev=this_chunk,
                        verbose=2,
                        callback=iter_callback
                    )
                else:
                    raise
            x_curr = merge_active_to_full(np.asarray(res.x, dtype=float))
            total_nfev += int(res.nfev)

            x_curr = self._apply_fixed_core(self._linear_constraint_project(x_curr))

            if use_prox:
                x_proj, pinfo = self._apply_proximal_projection(
                    x_curr,
                    pressure_scale_factor=pressure_scale_factor,
                    projection_cfg=proj_cfg,
                )
                x_curr = self._apply_fixed_core(self._linear_constraint_project(x_proj))
                if pinfo.get("applied", False):
                    print(
                        "    >>> [ProximalProjection] "
                        f"nfev={pinfo['nfev']}, ||r||: {pinfo['before_norm']:.3e} -> {pinfo['after_norm']:.3e}"
                    )

        end_time = time.time()
        if res is None:
            raise RuntimeError("优化未执行：max_nfev 预算不足")
        res.x = x_curr
        res.nfev = total_nfev
        phys_res_final = jax_res_fn(jnp.array(x_curr), apply_scaling=True)
        phys_len = self.num_core_params 
        phys_res_only = phys_res_final[:phys_len]
        fb_rel_final = self.metric_force_balance_rel(x_curr, pressure_scale_factor)
        
        print(f"    当前网格总耗时: {end_time - start_time:.2f} s | 总函数计算: {total_nfev} 次")
        print(f"    纯主物理残差 L2 范数: {np.linalg.norm(phys_res_only):.4e}")
        print(f"    归一化MHD力平衡误差: {fb_rel_final:.6e}")
        
        return res

    def _spectral_grad_th_np(self, f):
        return np.real(np.fft.ifft(1j * self.k_th * np.fft.fft(f, axis=1), axis=1))

    def _spectral_grad_ze_np(self, f):
        return np.real(np.fft.ifft(1j * self.k_ze * np.fft.fft(f, axis=2), axis=2))

    def _compute_state_numpy(self, x_core, pressure_scale_factor=1.0):
        RHO, TH, ZE = self.RHO, self.TH, self.ZE
        D_matrix = self.D_matrix

        e_R0, e_Z0, e_c0R, e_c0Z, e_h, e_v, e_k, e_a, e_tR, e_tZ = self.unpack_edge()
        c_c0R, c_c0Z, c_h, c_v, c_k, c_a, c_tR, c_tZ, c_lam = self.unpack_core(x_core)

        basis_1d_val_slice = self.basis_1d_val[:, 0, 0, :]
        basis_1d_dz_slice = self.basis_1d_dz[:, 0, 0, :]

        def eval_1d(c_e, c_c):
            core_contrib = np.dot(c_c.T, self.fac_rad)
            ce_eff = c_e[:, None] + core_contrib
            val = np.dot(ce_eff.T, basis_1d_val_slice)[:, None, :]
            dz = np.dot(ce_eff.T, basis_1d_dz_slice)[:, None, :]
            dr_contrib = np.dot(c_c.T, self.dfac_rad)
            dr = np.dot(dr_contrib.T, basis_1d_val_slice)[:, None, :]
            return val, dr, dz

        def eval_2d(c_e, c_c):
            if self.len_2d == 0:
                z = np.zeros((self.Nr, self.Nt_grid, self.Nz_grid))
                return z, z, z, z
            core_contrib = np.dot(c_c.T, self.fac_rad)
            ce_eff = c_e[:, None] + core_contrib
            val = np.sum(ce_eff[:, :, None, None] * self.basis_2d_val, axis=0)
            dth = np.sum(ce_eff[:, :, None, None] * self.basis_2d_dth, axis=0)
            dz = np.sum(ce_eff[:, :, None, None] * self.basis_2d_dze, axis=0)
            dr_contrib = np.dot(c_c.T, self.dfac_rad)
            dr = np.sum(
                dr_contrib[:, :, None, None] * self.basis_2d_val
                + ce_eff[:, :, None, None] * self.basis_2d_dr,
                axis=0,
            )
            return val, dr, dth, dz

        c0R, c0Rr, c0Rz = eval_1d(e_c0R, c_c0R)
        c0Z, c0Zr, c0Zz = eval_1d(e_c0Z, c_c0Z)
        h, hr, hz = eval_1d(e_h, c_h)
        v, vr, vz = eval_1d(e_v, c_v)
        k, kr, kz = eval_1d(e_k, c_k)
        a, ar, az = eval_1d(e_a, c_a)
        tR, tRr, tRth, tRz = eval_2d(e_tR, c_tR)
        tZ, tZr, tZth, tZz = eval_2d(e_tZ, c_tZ)

        thR = TH + c0R + tR
        thZ = TH + c0Z + tZ
        thR_r, thR_th, thR_z = c0Rr + tRr, 1.0 + tRth, c0Rz + tRz
        thZ_r, thZ_th, thZ_z = c0Zr + tZr, 1.0 + tZth, c0Zz + tZz

        R = e_R0 + a * (h - RHO * np.cos(thR))
        Rr = ar * (h - RHO * np.cos(thR)) + a * (hr - np.cos(thR) + RHO * np.sin(thR) * thR_r)
        Rt = a * RHO * np.sin(thR) * thR_th
        Rz = az * (h - RHO * np.cos(thR)) + a * (hz + RHO * np.sin(thR) * thR_z)

        Z = e_Z0 + a * (v - k * RHO * np.sin(thZ))
        Zr = ar * (v - k * RHO * np.sin(thZ)) + a * (
            vr - kr * RHO * np.sin(thZ) - k * np.sin(thZ) - k * RHO * np.cos(thZ) * thZ_r
        )
        Zt = -a * k * RHO * np.cos(thZ) * thZ_th
        Zz = az * (v - k * RHO * np.sin(thZ)) + a * (
            vz - kz * RHO * np.sin(thZ) - k * RHO * np.cos(thZ) * thZ_z
        )

        det_phys = Rr * Zt - Rt * Zr
        det_eff = self.det_chirality * det_phys
        sqrt_g = (R / self.N_fp) * det_eff

        g_rr = Rr**2 + Zr**2
        g_tt = Rt**2 + Zt**2
        g_zz = Rz**2 + (R / self.N_fp) ** 2 + Zz**2
        g_rt = Rr * Rt + Zr * Zt
        g_rz = Rr * Rz + Zr * Zz
        g_tz = Rt * Rz + Zt * Zz

        if self.len_lam > 0:
            lam_ce = np.dot(c_lam.T, self.fac_lam_eval) * self.rho_m_lam
            Lt = np.einsum("mr,mtz->rtz", lam_ce, self.basis_lam_dth[:, 0, :, :])
            Lz = np.einsum("mr,mtz->rtz", lam_ce, self.basis_lam_dze[:, 0, :, :])
        else:
            Lt = np.zeros_like(R)
            Lz = np.zeros_like(R)

        P, dP, Phip, psip = self._get_profiles(RHO, pressure_scale_factor)

        Bt_sup = (psip / self.N_fp - Lz) / (2 * np.pi * sqrt_g)
        Bz_sup = (Phip + Lt) / (2 * np.pi * sqrt_g)
        Br_sup = np.zeros_like(Bt_sup)

        Br_sub = g_rt * Bt_sup + g_rz * Bz_sup
        Bt_sub = g_tt * Bt_sup + g_tz * Bz_sup
        Bz_sub = g_tz * Bt_sup + g_zz * Bz_sup

        dBt_drho = np.tensordot(D_matrix, Bt_sub, axes=(1, 0))
        dBz_drho = np.tensordot(D_matrix, Bz_sub, axes=(1, 0))

        Jz_sup = (dBt_drho - self._spectral_grad_th_np(Br_sub)) / sqrt_g
        Jt_sup = (self._spectral_grad_ze_np(Br_sub) - dBz_drho) / sqrt_g
        Jr_sup = (self._spectral_grad_th_np(Bz_sub) - self._spectral_grad_ze_np(Bt_sub)) / sqrt_g
        Jr_phys = Jr_sup / self.mu_0

        G_rho = dP - sqrt_g * (Jt_sup * Bz_sup - Jz_sup * Bt_sup) / self.mu_0
        F_rho = G_rho
        F_beta = (Jr_phys / (2 * np.pi)) * (Phip + Lt)

        return {
            "R": R, "Z": Z, "Rr": Rr, "Rt": Rt, "Rz": Rz, "Zr": Zr, "Zt": Zt, "Zz": Zz,
            "det_phys": det_phys, "det_eff": det_eff, "sqrt_g": sqrt_g, "P": P, "dP": dP, "Phip": Phip, "psip": psip,
            "Lt": Lt, "Lz": Lz, "Bt_sup": Bt_sup, "Bz_sup": Bz_sup, "Br_sup": Br_sup,
            "Br_sub": Br_sub, "Bt_sub": Bt_sub, "Bz_sub": Bz_sub,
            "Jr_sup": Jr_sup, "Jt_sup": Jt_sup, "Jz_sup": Jz_sup, "Jr_phys": Jr_phys,
            "G_rho": G_rho, "F_rho": F_rho, "F_beta": F_beta,
            "g_rr": g_rr, "g_tt": g_tt, "g_zz": g_zz, "g_rt": g_rt, "g_rz": g_rz, "g_tz": g_tz
        }

    def _check_rho_derivative_stability(
        self,
        x_core=None,
        pressure_scale_factor=0.0,
        stage_label="当前模式",
        raise_on_fail=True,
    ):
        """检查当前 M/N/L 下 R,Z 对 rho 的一阶导数在 rho=0 附近是否稳定。"""
        x_eval = np.zeros(self.num_core_params, dtype=float) if x_core is None else np.asarray(x_core, dtype=float)
        x_eval = self._apply_fixed_core(x_eval)
        s = self._compute_state_numpy(x_eval, pressure_scale_factor=pressure_scale_factor)

        rho_arr = np.asarray(self.rho, dtype=float)
        Nr = int(rho_arr.size)
        if Nr == 0:
            raise RuntimeError("rho 网格为空，无法进行 rho=0 导数稳定性检查。")

        axis_idx = int(np.argmin(np.abs(rho_arr)))
        near_idx1 = min(axis_idx + 1, Nr - 1)
        near_idx2 = min(axis_idx + 2, Nr - 1)
        chk_idx = sorted(set([axis_idx, near_idx1, near_idx2]))

        Rr = np.asarray(s["Rr"], dtype=float)
        Zr = np.asarray(s["Zr"], dtype=float)
        dmag = np.sqrt(Rr**2 + Zr**2)
        sel = dmag[chk_idx, :, :]
        finite_ok = bool(np.isfinite(sel).all())

        axis_mag = dmag[axis_idx, :, :]
        axis_mean = float(np.mean(np.abs(axis_mag)))
        axis_std = float(np.std(axis_mag))
        axis_rel_std = float(axis_std / (axis_mean + 1e-14))
        near1_mean = float(np.mean(np.abs(dmag[near_idx1, :, :])))
        near2_mean = float(np.mean(np.abs(dmag[near_idx2, :, :])))
        jump_ratio_01 = float(near1_mean / (axis_mean + 1e-14))
        jump_ratio_12 = float(near2_mean / (near1_mean + 1e-14))
        max_abs = float(np.max(np.abs(sel)))

        # 稳定性判据：避免 NaN/Inf 与明显数值爆炸。
        stable = finite_ok and np.isfinite(max_abs) and (max_abs < 1e8)

        print(
            f">>> [rho-derivative-check][{stage_label}] "
            f"(M={self.M_pol}, N={self.N_tor}, L={self.L_rad})"
        )
        print(
            "    "
            f"rho_idx={chk_idx}, |dX/drho|_axis_mean={axis_mean:.3e}, "
            f"axis_rel_std={axis_rel_std:.3e}, jump01={jump_ratio_01:.3e}, jump12={jump_ratio_12:.3e}, "
            f"max_abs={max_abs:.3e}, finite={finite_ok}"
        )
        if stable:
            print("    >>> rho=0 一阶导数稳定性检查: 通过")
            return {
                "stable": True,
                "axis_mean": axis_mean,
                "axis_rel_std": axis_rel_std,
                "jump_ratio_01": jump_ratio_01,
                "jump_ratio_12": jump_ratio_12,
                "max_abs": max_abs,
                "indices": chk_idx,
            }

        msg = (
            "rho=0 一阶导数稳定性检查失败："
            f"finite={finite_ok}, max_abs={max_abs:.3e}, "
            f"axis_rel_std={axis_rel_std:.3e}, jump01={jump_ratio_01:.3e}, jump12={jump_ratio_12:.3e}"
        )
        if raise_on_fail:
            raise RuntimeError(msg)
        print(f"    >>> [WARN] {msg}")
        return {
            "stable": False,
            "axis_mean": axis_mean,
            "axis_rel_std": axis_rel_std,
            "jump_ratio_01": jump_ratio_01,
            "jump_ratio_12": jump_ratio_12,
            "max_abs": max_abs,
            "indices": chk_idx,
        }

    def _volume_weights(self, state):
        # 验证指标使用正权重，避免手性符号导致的积分抵消或负权重 sqrt 数值问题
        return np.abs(state["sqrt_g"]) * self.weights_3d * self.dtheta * self.dzeta

    def metric_force_balance_L2(self, x_core, pressure_scale_factor=1.0):
        s = self._compute_state_numpy(x_core, pressure_scale_factor)
        w = self._volume_weights(s)
        v = np.sqrt(s["F_rho"] ** 2 + s["F_beta"] ** 2)
        return float(np.sqrt(np.sum(w * v**2)))

    def metric_force_balance_rel(self, x_core, pressure_scale_factor=1.0):
        s = self._compute_state_numpy(x_core, pressure_scale_factor)
        w = self._volume_weights(s)
        lhs = np.sqrt(np.sum(w * (s["F_rho"] ** 2 + s["F_beta"] ** 2)))
        rhs = np.sqrt(np.sum(w * (s["dP"] ** 2))) + 1e-14
        return float(lhs / rhs)

    def metric_divB_L2(self, x_core, pressure_scale_factor=1.0):
        s = self._compute_state_numpy(x_core, pressure_scale_factor)
        sqrtg = s["sqrt_g"]
        divB = (self._spectral_grad_th_np(sqrtg * s["Bt_sup"]) + self._spectral_grad_ze_np(sqrtg * s["Bz_sup"])) / sqrtg
        w = self._volume_weights(s)
        return float(np.sqrt(np.sum(w * divB**2)))

    def metric_divB_rel(self, x_core, pressure_scale_factor=1.0):
        s = self._compute_state_numpy(x_core, pressure_scale_factor)
        sqrtg = s["sqrt_g"]
        divB = (self._spectral_grad_th_np(sqrtg * s["Bt_sup"]) + self._spectral_grad_ze_np(sqrtg * s["Bz_sup"])) / sqrtg
        B_mag = np.sqrt(s["Bt_sup"] * s["Bt_sub"] + s["Bz_sup"] * s["Bz_sub"])
        w = self._volume_weights(s)
        lhs = np.sqrt(np.sum(w * divB**2))
        rhs = np.sqrt(np.sum(w * B_mag**2)) + 1e-14
        return float(lhs / rhs)

    def metric_jacobian_stats(self, x_core, pressure_scale_factor=1.0):
        s = self._compute_state_numpy(x_core, pressure_scale_factor)
        d = s["det_phys"].flatten()
        de = s["det_eff"].flatten()
        return {
            "min": float(np.min(d)),
            "p1": float(np.percentile(d, 1)),
            "p50": float(np.percentile(d, 50)),
            "p99": float(np.percentile(d, 99)),
            "negative_fraction": float(np.mean(d < 0.0)),
            "eff_min": float(np.min(de)),
            "eff_negative_fraction": float(np.mean(de < 0.0)),
        }

    def metric_current_closure_L2(self, x_core, pressure_scale_factor=1.0):
        s = self._compute_state_numpy(x_core, pressure_scale_factor)
        sqrtg = s["sqrt_g"]
        divJ = (
            np.tensordot(self.D_matrix, sqrtg * s["Jr_sup"], axes=(1, 0))
            + self._spectral_grad_th_np(sqrtg * s["Jt_sup"])
            + self._spectral_grad_ze_np(sqrtg * s["Jz_sup"])
        ) / sqrtg
        w = self._volume_weights(s)
        return float(np.sqrt(np.sum(w * divJ**2)))

    def metric_axis_location(self, x_core, nz_samples=128, nt_samples=128):
        zeta = np.linspace(0, 2 * np.pi, nz_samples, endpoint=False)
        theta = np.linspace(0, 2 * np.pi, nt_samples, endpoint=False)
        R_axis = np.zeros_like(zeta)
        Z_axis = np.zeros_like(zeta)
        for i, zv in enumerate(zeta):
            Rv, Zv = self.compute_geometry(x_core, 0.0, theta, zv)[:2]
            R_axis[i] = np.mean(np.asarray(Rv))
            Z_axis[i] = np.mean(np.asarray(Zv))
        out = {
            "zeta": zeta,
            "R_axis": R_axis,
            "Z_axis": Z_axis,
            "R_std": float(np.std(R_axis)),
            "Z_std": float(np.std(Z_axis)),
            "R_mean": float(np.mean(R_axis)),
            "Z_mean": float(np.mean(Z_axis)),
        }
        return out

    def metric_profile_reconstruction_error(self, x_core, pressure_scale_factor=1.0):
        s = self._compute_state_numpy(x_core, pressure_scale_factor)
        dP_est = self._surface_mean_tz(
            s["sqrt_g"] * (s["Jt_sup"] * s["Bz_sup"] - s["Jz_sup"] * s["Bt_sup"]) / self.mu_0
        )
        dP_tar = self._surface_mean_tz(s["dP"])
        P_tar = self._surface_mean_tz(s["P"])
        P_est = np.zeros_like(P_tar)
        for i in range(1, len(self.rho)):
            dr = self.rho[i] - self.rho[i - 1]
            P_est[i] = P_est[i - 1] + 0.5 * (dP_est[i] + dP_est[i - 1]) * dr
        P_est = P_est - P_est[-1]
        P_tar = P_tar - P_tar[-1]
        err = np.sqrt(np.mean((P_est - P_tar) ** 2))
        rel = err / (np.sqrt(np.mean(P_tar**2)) + 1e-14)
        return {
            "rho": self.rho.copy(),
            "P_target": P_tar,
            "P_recon": P_est,
            "dP_target": dP_tar,
            "dP_recon": dP_est,
            "rmse": float(err),
            "rel_rmse": float(rel),
        }

    def metric_rotational_transform_error(self, x_core, pressure_scale_factor=1.0):
        s = self._compute_state_numpy(x_core, pressure_scale_factor)
        Phip_est_3d = 2 * np.pi * s["sqrt_g"] * s["Bz_sup"] - s["Lt"]
        psip_est_3d = self.N_fp * (2 * np.pi * s["sqrt_g"] * s["Bt_sup"] + s["Lz"])
        Phip_est = self._surface_mean_tz(Phip_est_3d)
        psip_est = self._surface_mean_tz(psip_est_3d)
        iota_est = psip_est / (Phip_est + 1e-14)
        iota_tar = 1.0 + 1.5 * self.rho**2
        rmse = np.sqrt(np.mean((iota_est - iota_tar) ** 2))
        rel = rmse / (np.sqrt(np.mean(iota_tar**2)) + 1e-14)
        return {
            "rho": self.rho.copy(),
            "iota_target": iota_tar,
            "iota_recon": iota_est,
            "rmse": float(rmse),
            "rel_rmse": float(rel),
            "Phip_recon": Phip_est,
            "psip_recon": psip_est,
        }

    def metric_toroidal_flux_error(self, x_core, pressure_scale_factor=1.0):
        iota_data = self.metric_rotational_transform_error(x_core, pressure_scale_factor)
        Phip_est = iota_data["Phip_recon"]
        phi_est = np.zeros_like(Phip_est)
        phi_est[0] = 0.5 * float(Phip_est[0]) * self.rho[0]
        for i in range(1, len(self.rho)):
            dr = self.rho[i] - self.rho[i - 1]
            phi_est[i] = phi_est[i - 1] + 0.5 * (Phip_est[i] + Phip_est[i - 1]) * dr
        edge_abs = abs(phi_est[-1] - self.Phi_a)
        edge_rel = edge_abs / (abs(self.Phi_a) + 1e-14)
        phi_target = self.Phi_a * self.rho**2
        return {
            "rho": self.rho.copy(),
            "Phi_target": phi_target,
            "Phi_recon": phi_est,
            "edge_abs_error": float(edge_abs),
            "edge_rel_error": float(edge_rel),
        }

    def metric_poloidal_flux_error(self, x_core, pressure_scale_factor=1.0):
        iota_data = self.metric_rotational_transform_error(x_core, pressure_scale_factor)
        psip_est = iota_data["psip_recon"]
        psi_est = np.zeros_like(psip_est)
        psi_est[0] = 0.5 * float(psip_est[0]) * self.rho[0]
        for i in range(1, len(self.rho)):
            dr = self.rho[i] - self.rho[i - 1]
            psi_est[i] = psi_est[i - 1] + 0.5 * (psip_est[i] + psip_est[i - 1]) * dr
        psi_target = self.compute_psi(self.rho)
        edge_abs = abs(psi_est[-1] - psi_target[-1])
        edge_rel = edge_abs / (abs(psi_target[-1]) + 1e-14)
        rmse = np.sqrt(np.mean((psi_est - psi_target) ** 2))
        return {
            "rho": self.rho.copy(),
            "Psi_target": psi_target,
            "Psi_recon": psi_est,
            "edge_abs_error": float(edge_abs),
            "edge_rel_error": float(edge_rel),
            "rmse": float(rmse),
        }

    def metric_global_energy_terms(self, x_core, pressure_scale_factor=1.0):
        s = self._compute_state_numpy(x_core, pressure_scale_factor)
        w = self._volume_weights(s)
        B2 = s["Bt_sup"] * s["Bt_sub"] + s["Bz_sup"] * s["Bz_sub"]
        magnetic_energy = 0.5 / self.mu_0 * np.sum(w * B2)
        pressure_energy = np.sum(w * s["P"])
        total_energy = magnetic_energy + pressure_energy
        return {
            "magnetic_energy": float(magnetic_energy),
            "pressure_energy": float(pressure_energy),
            "total_energy": float(total_energy),
        }

    def metric_spectral_tail_ratio(self, x_core):
        c_c0R, c_c0Z, c_h, c_v, c_k, c_a, c_tR, c_tZ, c_lam = self.unpack_core(x_core)

        def tail_ratio(arr):
            if arr.shape[0] < 2:
                return 0.0
            e_all = np.sum(arr**2)
            e_tail = np.sum(arr[-1:, :] ** 2)
            return float(e_tail / (e_all + 1e-14))

        ratios = {
            "c0R_tail": tail_ratio(c_c0R),
            "c0Z_tail": tail_ratio(c_c0Z),
            "h_tail": tail_ratio(c_h),
            "v_tail": tail_ratio(c_v),
            "k_tail": tail_ratio(c_k),
            "a_tail": tail_ratio(c_a),
            "tR_tail": tail_ratio(c_tR) if self.len_2d > 0 else 0.0,
            "tZ_tail": tail_ratio(c_tZ) if self.len_2d > 0 else 0.0,
            "lam_tail": tail_ratio(c_lam) if self.len_lam > 0 else 0.0,
        }
        ratios["max_tail_ratio"] = float(max(ratios.values()) if len(ratios) > 0 else 0.0)
        return ratios

    def metric_residual_drop_history(self):
        hist = np.array(getattr(self, "_all_eval_history", []), dtype=float)
        if hist.size == 0:
            return {"history": hist, "drop_ratio": np.nan, "final": np.nan}
        drop_ratio = hist[-1] / (hist[0] + 1e-14)
        return {"history": hist, "drop_ratio": float(drop_ratio), "final": float(hist[-1])}

    def metric_conditioning_proxy(self, x_core, pressure_scale_factor=1.0):
        jax_res_fn = self._build_jax_residual_fn(pressure_scale_factor)

        @jax.jit
        def fun_wrapped(x_arr):
            return jax_res_fn(x_arr, apply_scaling=True)

        @jax.jit
        def jac_wrapped(x_arr):
            return jax.jacfwd(fun_wrapped)(x_arr)

        J = np.array(jac_wrapped(jnp.array(x_core)))
        col_norms = np.linalg.norm(J, axis=0)
        min_col = np.min(col_norms) + 1e-14
        max_col = np.max(col_norms)
        try:
            svals = np.linalg.svd(J, compute_uv=False)
            cond = float(svals[0] / (svals[-1] + 1e-14))
            smin = float(svals[-1])
            smax = float(svals[0])
        except Exception:
            cond = np.inf
            smin = np.nan
            smax = np.nan
        return {
            "column_norm_ratio": float(max_col / min_col),
            "svd_condition_est": cond,
            "svd_smin": smin,
            "svd_smax": smax,
        }

    def metric_penalty_ratio(self, x_core, pressure_scale_factor=1.0):
        """物理残差与惩罚/正则残差比例（禁用 scaling）。"""
        breakdown = self.metric_penalty_breakdown(x_core, pressure_scale_factor)
        norm_phys = breakdown["norm_phys_unscaled"]
        norm_pen = breakdown["norm_penalty_unscaled"]
        ratio = norm_pen / (norm_phys + 1e-14)
        return {
            "norm_phys_unscaled": norm_phys,
            "norm_penalty_unscaled": norm_pen,
            "penalty_ratio": ratio,
        }

    def metric_penalty_breakdown(self, x_core, pressure_scale_factor=1.0):
        """将惩罚项拆分为 profile/reg_tR/reg_tZ/reg_lam 四类（禁用 scaling）。"""
        jax_res_fn = self._build_jax_residual_fn(pressure_scale_factor)
        x_eff = self._apply_fixed_core(x_core)
        res_all = np.array(jax_res_fn(jnp.array(x_eff), apply_scaling=False), dtype=float)
        phys_len = self.num_core_params
        res_phys = res_all[:phys_len]
        res_pen_all = res_all[phys_len:]

        profile_len = (self.Nr if self.main_dp_weight > 0.0 else 0) + (1 if self.main_pressure_edge_weight > 0.0 else 0)
        reg_tR_len = self.L_rad * self.len_2d
        reg_tZ_len = self.L_rad * self.len_2d
        reg_lam_len = self.L_rad * self.len_lam

        idx = 0
        res_profile = res_pen_all[idx:idx + profile_len]
        idx += profile_len
        res_reg_tR = res_pen_all[idx:idx + reg_tR_len]
        idx += reg_tR_len
        res_reg_tZ = res_pen_all[idx:idx + reg_tZ_len]
        idx += reg_tZ_len
        res_reg_lam = res_pen_all[idx:idx + reg_lam_len]

        res_pen = np.concatenate([res_profile, res_reg_tR, res_reg_tZ, res_reg_lam]) if (
            (res_profile.size + res_reg_tR.size + res_reg_tZ.size + res_reg_lam.size) > 0
        ) else np.array([], dtype=float)
        norm_phys = float(np.linalg.norm(res_phys))
        norm_pen = float(np.linalg.norm(res_pen))
        return {
            "norm_phys_unscaled": norm_phys,
            "norm_profile_penalty_unscaled": float(np.linalg.norm(res_profile)),
            "norm_reg_tR_unscaled": float(np.linalg.norm(res_reg_tR)),
            "norm_reg_tZ_unscaled": float(np.linalg.norm(res_reg_tZ)),
            "norm_reg_lam_unscaled": float(np.linalg.norm(res_reg_lam)),
            "norm_penalty_unscaled": norm_pen,
        }

    def metric_scaling_masking(self, x_core, pressure_scale_factor=1.0):
        """对比 scaled 与 unscaled 残差差距，监控缩放掩蔽。"""
        jax_res_fn = self._build_jax_residual_fn(pressure_scale_factor)
        res_scaled = np.array(jax_res_fn(jnp.array(x_core), apply_scaling=True), dtype=float)
        res_unscaled = np.array(jax_res_fn(jnp.array(x_core), apply_scaling=False), dtype=float)
        norm_scaled = float(np.linalg.norm(res_scaled))
        norm_unscaled = float(np.linalg.norm(res_unscaled))
        return {
            "norm_scaled": norm_scaled,
            "norm_unscaled": norm_unscaled,
            "masking_factor": norm_unscaled / (norm_scaled + 1e-14),
        }

    def metric_projection_disruption(self, x_core, projection_cfg=None, pressure_scale_factor=1.0):
        """评估 proximal 投影对力平衡的扰动程度。"""
        cfg = dict(projection_cfg or {})
        if not cfg or not bool(cfg.get("enabled", False)):
            return {"fb_before": 0.0, "fb_after": 0.0, "disruption_ratio": 1.0}
        fb_before = self.metric_force_balance_rel(x_core, pressure_scale_factor)
        x_proj, _ = self._apply_proximal_projection(x_core, pressure_scale_factor, cfg)
        fb_after = self.metric_force_balance_rel(x_proj, pressure_scale_factor)
        return {
            "fb_before": float(fb_before),
            "fb_after": float(fb_after),
            "disruption_ratio": float(fb_after / (fb_before + 1e-14)),
        }

    def metric_residual_group_breakdown(self, x_core, pressure_scale_factor=1.0):
        """按组分拆分残差，返回未加权 raw 与加权后 residual。"""
        jax_res_fn = self._build_jax_residual_fn(pressure_scale_factor)
        x_eff = self._apply_fixed_core(x_core)
        res_raw = np.array(
            jax_res_fn(jnp.array(x_eff), apply_scaling=False, apply_group_weights=False),
            dtype=float,
        )
        res_weighted = np.array(
            jax_res_fn(jnp.array(x_eff), apply_scaling=False, apply_group_weights=True),
            dtype=float,
        )

        phys_len = self.num_core_params
        profile_len = (self.Nr if self.main_dp_weight > 0.0 else 0) + (1 if self.main_pressure_edge_weight > 0.0 else 0)
        reg_tR_len = self.L_rad * self.len_2d
        reg_tZ_len = self.L_rad * self.len_2d
        reg_lam_len = self.L_rad * self.len_lam

        groups_meta = [
            ("main_phys", "主物理残差", phys_len),
            ("profile_penalty", "剖面约束惩罚", profile_len),
            ("reg_tR", "正则项 tR", reg_tR_len),
            ("reg_tZ", "正则项 tZ", reg_tZ_len),
            ("reg_lam", "正则项 Lambda", reg_lam_len),
        ]

        idx = 0
        groups = []
        for key, name_cn, glen in groups_meta:
            glen = int(max(0, glen))
            g_raw = res_raw[idx:idx + glen]
            g_weighted = res_weighted[idx:idx + glen]
            idx += glen
            norm_raw = float(np.linalg.norm(g_raw))
            norm_weighted = float(np.linalg.norm(g_weighted))
            groups.append({
                "key": key,
                "name_cn": name_cn,
                "size": glen,
                "norm_raw": norm_raw,
                "norm_weighted": norm_weighted,
                "effective_weight": float(norm_weighted / (norm_raw + 1e-14)),
            })

        if idx < res_raw.size:
            tail_raw = res_raw[idx:]
            tail_weighted = res_weighted[idx:]
            norm_raw = float(np.linalg.norm(tail_raw))
            norm_weighted = float(np.linalg.norm(tail_weighted))
            groups.append({
                "key": "tail_unknown",
                "name_cn": "未分类尾部",
                "size": int(res_raw.size - idx),
                "norm_raw": norm_raw,
                "norm_weighted": norm_weighted,
                "effective_weight": float(norm_weighted / (norm_raw + 1e-14)),
            })

        return {
            "groups": groups,
            "norm_total_raw": float(np.linalg.norm(res_raw)),
            "norm_total_weighted": float(np.linalg.norm(res_weighted)),
        }

    def metric_directional_spectral_blocking(self, x_core):
        """分离监控径向/角向谱阻塞，定位自由度不足方向。"""
        c_c0R, c_c0Z, c_h, c_v, c_k, c_a, c_tR, c_tZ, c_lam = self.unpack_core(x_core)

        def check_radial_blocking(c_arr):
            if c_arr.shape[0] < 3:
                return 0.0
            e_all = np.sum(c_arr**2)
            e_tail = np.sum(c_arr[-2:, :]**2)
            return float(e_tail / (e_all + 1e-14))

        def check_2d_angular_blocking(c_arr2d):
            if self.len_2d == 0:
                return 0.0
            e_all = np.sum(c_arr2d**2)
            tail_idx = []
            for i, (m, n, typ) in enumerate(self.modes_2d):
                if m == self.M_pol or abs(n) == self.N_tor:
                    tail_idx.append(i)
            if len(tail_idx) == 0:
                return 0.0
            e_tail = np.sum(c_arr2d[:, tail_idx] ** 2)
            return float(e_tail / (e_all + 1e-14))

        return {
            "radial_blocking_R": check_radial_blocking(c_c0R) + check_radial_blocking(c_tR) if self.len_2d > 0 else check_radial_blocking(c_c0R),
            "radial_blocking_Z": check_radial_blocking(c_c0Z) + check_radial_blocking(c_tZ) if self.len_2d > 0 else check_radial_blocking(c_c0Z),
            "angular_blocking_R": check_2d_angular_blocking(c_tR) if self.len_2d > 0 else 0.0,
            "angular_blocking_Z": check_2d_angular_blocking(c_tZ) if self.len_2d > 0 else 0.0,
        }

    def metric_resolution_convergence(self, x_core, pressure_scale_factor=1.0, levels=None):
        if levels is None:
            levels = [
                (max(10, self.target_Nr - 6), max(10, self.target_Nt - 6), max(10, self.target_Nz - 6)),
                (max(12, self.target_Nr - 2), max(12, self.target_Nt - 2), max(12, self.target_Nz - 2)),
                (self.target_Nr, self.target_Nt, self.target_Nz),
            ]
        old = (self.Nr, self.Nt_grid, self.Nz_grid)
        results = []
        for Nr, Nt, Nz in levels:
            self.update_grid(int(Nr), int(Nt), int(Nz))
            fb = self.metric_force_balance_rel(x_core, pressure_scale_factor)
            db = self.metric_divB_rel(x_core, pressure_scale_factor)
            js = self.metric_jacobian_stats(x_core, pressure_scale_factor)
            results.append({
                "Nr": self.Nr,
                "Nt": self.Nt_grid,
                "Nz": self.Nz_grid,
                "force_balance_rel": fb,
                "divB_rel": db,
                "det_min": js["min"],
            })
        self.update_grid(*old)
        return results

    def _print_scalar_metric(self, name, value, target_text):
        print(f"    - {name:<32}: {value:>12.6e} | 目标最优值: {target_text}")

    def run_validation_suite(self, x_core, pressure_scale_factor=1.0, projection_cfg=None):
        print("\n" + "=" * 70)
        print(">>> 运行平衡正确性验证指标套件")
        print("=" * 70)

        fb_l2 = self.metric_force_balance_L2(x_core, pressure_scale_factor)
        fb_rel = self.metric_force_balance_rel(x_core, pressure_scale_factor)
        db_l2 = self.metric_divB_L2(x_core, pressure_scale_factor)
        db_rel = self.metric_divB_rel(x_core, pressure_scale_factor)
        jac_stats = self.metric_jacobian_stats(x_core, pressure_scale_factor)
        jcl = self.metric_current_closure_L2(x_core, pressure_scale_factor)
        axis = self.metric_axis_location(x_core)
        prof = self.metric_profile_reconstruction_error(x_core, pressure_scale_factor)
        iota_m = self.metric_rotational_transform_error(x_core, pressure_scale_factor)
        phi_m = self.metric_toroidal_flux_error(x_core, pressure_scale_factor)
        psi_m = self.metric_poloidal_flux_error(x_core, pressure_scale_factor)
        en_m = self.metric_global_energy_terms(x_core, pressure_scale_factor)
        tail = self.metric_spectral_tail_ratio(x_core)
        hist = self.metric_residual_drop_history()
        cond = self.metric_conditioning_proxy(x_core, pressure_scale_factor)
        ppr = self.metric_penalty_ratio(x_core, pressure_scale_factor)
        pbd = self.metric_penalty_breakdown(x_core, pressure_scale_factor)
        smf = self.metric_scaling_masking(x_core, pressure_scale_factor)
        proj_cfg = projection_cfg if projection_cfg is not None else getattr(self, "_last_projection_cfg", None)
        pdi = self.metric_projection_disruption(x_core, proj_cfg, pressure_scale_factor)
        block = self.metric_directional_spectral_blocking(x_core)
        conv = self.metric_resolution_convergence(x_core, pressure_scale_factor)

        print(">>> 标量指标（直接数值 + 目标最优值）")
        self._print_scalar_metric("force_balance_L2", fb_l2, "0 (越小越好)")
        self._print_scalar_metric("force_balance_rel", fb_rel, "0 (越小越好)")
        self._print_scalar_metric("divB_L2", db_l2, "0 (越小越好)")
        self._print_scalar_metric("divB_rel", db_rel, "0 (越小越好)")
        self._print_scalar_metric("current_closure_L2", jcl, "0 (越小越好)")
        self._print_scalar_metric("det_phys_min", jac_stats["min"], "> 0 (越大越安全)")
        self._print_scalar_metric("det_phys_negative_fraction", jac_stats["negative_fraction"], "0 (越小越好)")
        self._print_scalar_metric("det_eff_min", jac_stats["eff_min"], "> 0 (越大越安全)")
        self._print_scalar_metric("det_eff_negative_fraction", jac_stats["eff_negative_fraction"], "0 (越小越好)")
        self._print_scalar_metric("axis_R_std", axis["R_std"], "0 (越小越好)")
        self._print_scalar_metric("axis_Z_std", axis["Z_std"], "0 (越小越好)")
        self._print_scalar_metric("profile_rmse", prof["rmse"], "0 (越小越好)")
        self._print_scalar_metric("profile_rel_rmse", prof["rel_rmse"], "0 (越小越好)")
        self._print_scalar_metric("iota_rmse", iota_m["rmse"], "0 (越小越好)")
        self._print_scalar_metric("iota_rel_rmse", iota_m["rel_rmse"], "0 (越小越好)")
        self._print_scalar_metric("phi_edge_abs_error", phi_m["edge_abs_error"], "0 (越小越好)")
        self._print_scalar_metric("phi_edge_rel_error", phi_m["edge_rel_error"], "0 (越小越好)")
        self._print_scalar_metric("psi_edge_abs_error", psi_m["edge_abs_error"], "0 (越小越好)")
        self._print_scalar_metric("psi_edge_rel_error", psi_m["edge_rel_error"], "0 (越小越好)")
        self._print_scalar_metric("psi_profile_rmse", psi_m["rmse"], "0 (越小越好)")
        self._print_scalar_metric("magnetic_energy", en_m["magnetic_energy"], "有限正值 (物理可接受)")
        self._print_scalar_metric("pressure_energy", en_m["pressure_energy"], "有限值 (与剖面一致)")
        self._print_scalar_metric("total_energy", en_m["total_energy"], "有限正值")
        self._print_scalar_metric("spectral_max_tail_ratio", tail["max_tail_ratio"], "接近 0 (越小越好)")
        self._print_scalar_metric("residual_drop_ratio", hist["drop_ratio"], "接近 0 (越小越好)")
        self._print_scalar_metric("residual_final", hist["final"], "接近 0 (越小越好)")
        self._print_scalar_metric("conditioning_col_ratio", cond["column_norm_ratio"], "尽可能小")
        self._print_scalar_metric("conditioning_svd_est", cond["svd_condition_est"], "尽可能小")

        print(">>> 优化健康度与求解器诊断指标")
        ppr_flag = "PASS" if ppr["penalty_ratio"] < 0.1 else ("WARN" if ppr["penalty_ratio"] < 1.0 else "FAIL")
        smf_flag = "PASS" if smf["masking_factor"] < 1e2 else ("WARN" if smf["masking_factor"] < 1e4 else "FAIL")
        pdi_flag = "PASS" if pdi["disruption_ratio"] < 1.2 else ("WARN" if pdi["disruption_ratio"] < 1.5 else "FAIL")
        rb_max = max(block["radial_blocking_R"], block["radial_blocking_Z"])
        ab_max = max(block["angular_blocking_R"], block["angular_blocking_Z"])
        rb_flag = "PASS" if rb_max < 1e-3 else ("WARN" if rb_max < 1e-2 else "FAIL")
        ab_flag = "PASS" if ab_max < 1e-3 else ("WARN" if ab_max < 1e-2 else "FAIL")

        print(
            f"    - physics_to_penalty_ratio        : {ppr['penalty_ratio']:.6e} "
            f"[{ppr_flag}: 推荐 < 1e-1]"
        )
        print(f"    - profile_penalty                 : {pbd['norm_profile_penalty_unscaled']:.6e}")
        print(f"    - reg_tR                          : {pbd['norm_reg_tR_unscaled']:.6e}")
        print(f"    - reg_tZ                          : {pbd['norm_reg_tZ_unscaled']:.6e}")
        print(f"    - reg_lam                         : {pbd['norm_reg_lam_unscaled']:.6e}")
        print(
            f"    - scaling_masking_factor          : {smf['masking_factor']:.6e} "
            f"[{smf_flag}: 推荐 < 1e2]"
        )
        print(
            f"    - projection_disruption_ratio     : {pdi['disruption_ratio']:.6e} "
            f"[{pdi_flag}: 推荐 ~ 1]"
        )
        print(
            f"    - radial_spectral_blocking_max    : {rb_max:.6e} "
            f"[{rb_flag}: 推荐 < 1e-3]"
        )
        print(
            f"    - angular_spectral_blocking_max   : {ab_max:.6e} "
            f"[{ab_flag}: 推荐 < 1e-3]"
        )

        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(3, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(axis["zeta"], axis["R_axis"], label="R_axis(zeta)")
        ax1.plot(axis["zeta"], axis["Z_axis"], label="Z_axis(zeta)")
        ax1.set_title("Axis location vs zeta")
        ax1.set_xlabel("zeta")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(prof["rho"], prof["P_target"], label="P target")
        ax2.plot(prof["rho"], prof["P_recon"], "--", label="P recon")
        ax2.plot(prof["rho"], prof["dP_target"], label="dP/drho target")
        ax2.plot(prof["rho"], prof["dP_recon"], "--", label="dP/drho recon")
        ax2.set_title("Profile reconstruction")
        ax2.set_xlabel("rho")
        ax2.legend(fontsize="small")
        ax2.grid(alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(iota_m["rho"], iota_m["iota_target"], label="iota target")
        ax3.plot(iota_m["rho"], iota_m["iota_recon"], "--", label="iota recon")
        ax3.set_title("Rotational transform")
        ax3.set_xlabel("rho")
        ax3.legend()
        ax3.grid(alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(phi_m["rho"], phi_m["Phi_target"], label="Phi target")
        ax4.plot(phi_m["rho"], phi_m["Phi_recon"], "--", label="Phi recon")
        ax4.plot(psi_m["rho"], psi_m["Psi_target"], label="Psi target")
        ax4.plot(psi_m["rho"], psi_m["Psi_recon"], "--", label="Psi recon")
        ax4.set_title("Flux consistency")
        ax4.set_xlabel("rho")
        ax4.legend(fontsize="small")
        ax4.grid(alpha=0.3)

        ax5 = fig.add_subplot(gs[2, 0])
        if hist["history"].size > 0:
            ax5.semilogy(np.arange(len(hist["history"])), hist["history"], label="||res||")
        ax5.set_title("Residual drop history")
        ax5.set_xlabel("function eval index")
        ax5.legend()
        ax5.grid(alpha=0.3)

        ax6 = fig.add_subplot(gs[2, 1])
        labels = [k for k in tail.keys() if k != "max_tail_ratio"]
        vals = [tail[k] for k in labels]
        ax6.bar(np.arange(len(labels)), vals)
        ax6.set_xticks(np.arange(len(labels)))
        ax6.set_xticklabels(labels, rotation=45, ha="right")
        ax6.set_title("Spectral tail ratios")
        ax6.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(14, 5))
        grids = [f"{c['Nr']}x{c['Nt']}x{c['Nz']}" for c in conv]
        bx1.semilogy(grids, [c["force_balance_rel"] for c in conv], marker="o", label="force_balance_rel")
        bx1.semilogy(grids, [c["divB_rel"] for c in conv], marker="s", label="divB_rel")
        bx1.set_title("Resolution convergence (relative residuals)")
        bx1.set_xlabel("grid")
        bx1.legend()
        bx1.grid(alpha=0.3)

        bx2.plot(grids, [c["det_min"] for c in conv], marker="^", color="tab:red")
        bx2.set_title("Resolution convergence (det_phys min)")
        bx2.set_xlabel("grid")
        bx2.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        print("=" * 70 + "\n")

    def _run_three_phase_optimization(self, x_guess=None, phase_title_prefix=""):
        def make_even(x): return x + (x % 2)
        c_Nr_raw = make_even(max(10, 4 * self.L_rad + 2))
        c_Nt_raw = make_even(max(12, 4 * self.M_pol + 4))
        c_Nz_raw = make_even(max(8, 4 * self.N_tor + 2))
        m_Nr_raw = make_even(c_Nr_raw + 4)
        m_Nt_raw = make_even(c_Nt_raw + 6)
        m_Nz_raw = make_even(c_Nz_raw + 4)

        h_Nr = make_even(max(4, int(self.target_Nr)))
        h_Nt = make_even(max(4, int(self.target_Nt)))
        h_Nz = make_even(max(4, int(self.target_Nz)))

        m_Nr = min(m_Nr_raw, h_Nr)
        m_Nt = min(m_Nt_raw, h_Nt)
        m_Nz = min(m_Nz_raw, h_Nz)

        c_Nr = min(c_Nr_raw, m_Nr)
        c_Nt = min(c_Nt_raw, m_Nt)
        c_Nz = min(c_Nz_raw, m_Nz)

        phase_total_start_time = time.time()
        self._all_eval_history = []
        x0 = np.zeros(self.num_core_params) if x_guess is None else np.asarray(x_guess, dtype=float)
        x0 = self._apply_fixed_core(x0)

        print("\n" + "="*70)
        print(f">>> {phase_title_prefix}[Phase 1/3]: 粗网格 & 零压无力矩冷启动 (Nr={c_Nr}, Nt={c_Nt}, Nz={c_Nz}, P=0.0)")
        print("="*70)
        self.update_grid(c_Nr, c_Nt, c_Nz)
        proj_phase1 = {
            "enabled": True,
            "outer_loops": 2,
            "lsq_chunk_nfev": 90,
            "strength": 0.03,
            "boundary_strength": 0.05,
            "anchor_weight": 1e-2,
            "dp_strength": 0.0,
            "pressure_edge_strength": 0.0,
            "pressure_axis_strength": 0.0,
            "p_profile_strength": 0.0,
            "main_dp_weight": 0.0,
            "main_pressure_edge_weight": 0.0,
            "prox_max_nfev": 6,
            "prox_ftol": 1e-7,
        }
        res_phase1 = self._run_optimization(
            x0,
            max_nfev=200,
            ftol=1e-3,
            pressure_scale_factor=0.0,
            projection_cfg=proj_phase1,
        )
        self._all_eval_history.extend(getattr(self, "_last_eval_history", []))

        print("\n" + "="*70)
        print(f">>> {phase_title_prefix}[Phase 2/3]: 中网格 & 低压等离子体过渡 (Nr={m_Nr}, Nt={m_Nt}, Nz={m_Nz}, P=0.1)")
        print("="*70)
        self.update_grid(m_Nr, m_Nt, m_Nz)
        proj_phase2 = {
            "enabled": True,
            "outer_loops": 3,
            "lsq_chunk_nfev": 110,
            "strength": 0.08,
            "boundary_strength": 0.12,
            "anchor_weight": 5e-3,
            "dp_strength": 0.35,
            "pressure_edge_strength": 0.35,
            "pressure_axis_strength": 0.0,
            "p_profile_strength": 0.0,
            "main_dp_weight": 0.15,
            "main_pressure_edge_weight": 0.05,
            "prox_max_nfev": 8,
            "prox_ftol": 1e-8,
        }
        res_phase2 = self._run_optimization(
            res_phase1.x,
            max_nfev=300,
            ftol=1e-5,
            pressure_scale_factor=0.1,
            projection_cfg=proj_phase2,
        )
        self._all_eval_history.extend(getattr(self, "_last_eval_history", []))

        print("\n" + "="*70)
        print(f">>> {phase_title_prefix}[Phase 3/3]: 高保真网格 & 目标高压极限收敛 (Nr={h_Nr}, Nt={h_Nt}, Nz={h_Nz}, P=1.0)")
        print("="*70)
        self.update_grid(h_Nr, h_Nt, h_Nz)
        proj_phase3 = {
            "enabled": False,
            "outer_loops": 2,
            "lsq_chunk_nfev": 800,
            "strength": 0.05,
            "boundary_strength": 0.1,
            "anchor_weight": 5e-2,
            "dp_strength": 0.1,
            "pressure_edge_strength": 0.1,
            "pressure_axis_strength": 0.0,
            "p_profile_strength": 0.0,
            "main_dp_weight": 0.5,
            "main_pressure_edge_weight": 0.2,
            "main_phys_weight": 10.0,
            "profile_penalty_weight": 0.1,
            "reg_tR_weight": 0.1,
            "reg_tZ_weight": 0.1,
            "reg_lam_weight": 0.1,
            "prox_max_nfev": 5,
            "prox_ftol": 1e-6,
        }
        res_fine = self._run_optimization(
            res_phase2.x,
            max_nfev=1600,
            ftol=1e-12,
            pressure_scale_factor=1.0,
            projection_cfg=proj_phase3,
        )
        self._all_eval_history.extend(getattr(self, "_last_eval_history", []))
        phase_total_end_time = time.time()
        return res_phase1, res_phase2, res_fine, phase_total_end_time - phase_total_start_time

    def _prefilter_from_low_order_case(self, tol=1e-6):
        print("\n" + "=" * 70)
        print(">>> 预运行判零: 先执行 (M,N,L)=(1,1,3) 低阶平衡，用于筛除后续完整运行中的零参数")
        print("=" * 70)
        pilot = VEQ3D_Solver(boundary_fns=(self._boundary_R_fn, self._boundary_Z_fn))
        # 强制低阶预运行维度，避免继承当前主求解器的 M/N/L 配置
        pilot.M_pol = 1
        pilot.N_tor = 1
        pilot.L_rad = 3
        pilot._setup_modes()
        pilot.p_edge = None
        pilot.linear_constraint_A = None
        pilot.linear_constraint_b = None
        pilot._reset_core_freeze_state()

        pilot.N_fp = self.N_fp
        pilot.Phi_a = self.Phi_a
        pilot.mu_0 = self.mu_0
        pilot.det_chirality = self.det_chirality
        pilot.regularization_scale = self.regularization_scale
        pilot.target_Nr = self.target_Nr
        pilot.target_Nt = self.target_Nt
        pilot.target_Nz = self.target_Nz
        print(f">>> 预运行实际维度: (M={pilot.M_pol}, N={pilot.N_tor}, L={pilot.L_rad})")
        pilot.update_grid(pilot.target_Nr, pilot.target_Nt, pilot.target_Nz)
        pilot.fit_boundary()
        pilot._initialize_scaling()
        pilot._check_rho_derivative_stability(
            x_core=np.zeros(pilot.num_core_params),
            pressure_scale_factor=0.0,
            stage_label="PreFilter-Pilot",
            raise_on_fail=True,
        )

        _, _, pilot_res, pilot_elapsed = pilot._run_three_phase_optimization(phase_title_prefix="[PreFilter] ")
        freeze_mask, freeze_vals = self._build_prefilter_freeze_mask(pilot, pilot_res.x, tol=tol)
        self.core_fixed_mask = freeze_mask
        self.core_fixed_values = freeze_vals

        n_fixed = int(np.sum(self.core_fixed_mask))
        n_total = int(self.num_core_params)
        print(f">>> 预运行完成: 耗时 {pilot_elapsed:.2f} s")
        print(f">>> 判零阈值: |Chebyshev L=0,1,2| < {tol:.1e}")
        print(f">>> 参数冻结结果: 固定 {n_fixed}/{n_total} 个核心参数，自由参数 {n_total - n_fixed} 个")
        reg_stats = self._regularization_active_stats()
        print(
            ">>> 正则有效自由度: "
            f"tR {reg_stats['tR_active']}/{reg_stats['tR_total']}, "
            f"tZ {reg_stats['tZ_active']}/{reg_stats['tZ_total']}, "
            f"Lambda {reg_stats['lam_active']}/{reg_stats['lam_total']}"
        )
        self._print_frozen_parameter_details()
        print("=" * 70)

    def solve(self):
        print(">>> 启动 VEQ-3D 谱精度平衡求解器 (奇点相消无障碍版)...")
        self._reset_core_freeze_state()
        self._check_rho_derivative_stability(
            x_core=np.zeros(self.num_core_params),
            pressure_scale_factor=0.0,
            stage_label="Main-Solver-Precheck",
            raise_on_fail=True,
        )
        self._prefilter_from_low_order_case(tol=1e-6)

        res_phase1, res_phase2, res_fine, elapsed = self._run_three_phase_optimization()
        total_nfev = int(res_phase1.nfev + res_phase2.nfev + res_fine.nfev)
        print("\n" + "=" * 70)
        print(">>> 三阶段收敛总耗时统计:")
        print(f"    总耗时(Phase 1~3, 至收敛): {elapsed:.2f} s")
        print(f"    总函数计算次数(nfev):       {total_nfev}")
        print("=" * 70)

        self.print_final_parameters(res_fine.x)
        self.plot_equilibrium(res_fine.x)
        self.run_validation_suite(res_fine.x, pressure_scale_factor=1.0)
        return res_fine.x

    def compute_geometry(self, x_core, rho, theta, zeta):
        rho, theta, zeta = np.atleast_1d(rho), np.atleast_1d(theta), np.atleast_1d(zeta)
        base_grid = rho + theta + zeta
        x = 2.0 * rho**2 - 1.0  # ψ_N = ρ²
        T = np.zeros((self.L_rad,) + x.shape)
        if self.L_rad > 0: T[0] = 1.0
        if self.L_rad > 1: T[1] = x
        for l in range(2, self.L_rad):
            T[l] = 2.0 * x * T[l-1] - T[l-2]
            
        fac_rad = (1.0 - rho**2) * T  
        
        e_R0, e_Z0, e_c0R, e_c0Z, e_h, e_v, e_k, e_a, e_tR, e_tZ = self.unpack_edge()
        c_c0R, c_c0Z, c_h, c_v, c_k, c_a, c_tR, c_tZ, c_lam = self.unpack_core(x_core)
        
        def ev_1d(c_e, c_c):
            val = c_e[0] + np.tensordot(c_c[:, 0], fac_rad, axes=(0, 0))
            val = val + np.zeros_like(base_grid)  
            idx = 1
            for n in range(1, self.N_tor + 1):
                c_n = c_e[idx]   + np.tensordot(c_c[:, idx],   fac_rad, axes=(0, 0))
                s_n = c_e[idx+1] + np.tensordot(c_c[:, idx+1], fac_rad, axes=(0, 0))
                val = val + c_n * np.cos(n * zeta) + s_n * np.sin(n * zeta) 
                idx += 2
            return val
            
        def ev_2d(c_e, c_c):
            val = np.zeros_like(base_grid)
            if self.len_2d == 0: return val
            for i, (m, n, typ) in enumerate(self.modes_2d):
                b = np.cos(m * theta - n * zeta) if typ == 'c' else np.sin(m * theta - n * zeta)
                b = b * (rho ** m)
                val = val + (c_e[i] + np.tensordot(c_c[:, i], fac_rad, axes=(0, 0))) * b
            return val

        c0R, c0Z = ev_1d(e_c0R, c_c0R), ev_1d(e_c0Z, c_c0Z)
        h, v = ev_1d(e_h, c_h), ev_1d(e_v, c_v)
        k, a = ev_1d(e_k, c_k), ev_1d(e_a, c_a)
        tR, tZ = ev_2d(e_tR, c_tR), ev_2d(e_tZ, c_tZ)
        
        thR, thZ = theta + c0R + tR, theta + c0Z + tZ
        
        R = e_R0 + a * (h - rho * np.cos(thR))
        Z = e_Z0 + a * (v - k * rho * np.sin(thZ))
        
        lam = np.zeros_like(base_grid)
        for i, (m, n) in enumerate(self.lambda_modes):
            L_fac_rad_m = (rho ** m) * (1 - rho**2)**2 * T
            lam = lam + np.tensordot(c_lam[:, i], L_fac_rad_m, axes=(0, 0)) * np.sin(m * theta - n * zeta)
            
        return R, Z, thR, thZ, a, k, lam

    def print_final_parameters(self, x_core):
        table_width = max(110, 46 + self.L_rad * 25)
        
        print("\n" + "=" * table_width)
        print(f"{f'VEQ-3D 动态高维参数报告 (M={self.M_pol}, N={self.N_tor}, L={self.L_rad}, N_fp={self.N_fp})':^{table_width}}")
        print("=" * table_width)
        
        edge_R0, edge_Z0, e_c0R, e_c0Z, e_h, e_v, e_k, e_a, e_tR, e_tZ = self.unpack_edge()
        c_c0R, c_c0Z, c_h, c_v, c_k, c_a, c_tR, c_tZ, c_lam = self.unpack_core(x_core)
        
        print(f"R0 (大半径中心) = {edge_R0:>15.8e}")
        print(f"Z0 (垂直中心)   = {edge_Z0:>15.8e}")
        print("-" * table_width)
        
        header_cols = [f"Chebyshev L={L} 演化系数" for L in range(self.L_rad)]
        header_str = f"{'参数标识':<15} | {'Edge (ρ=1, ψ_N=1)':<25} | " + " | ".join([f"{h:<22}" for h in header_cols])
        print(header_str)
        print("-" * table_width)
        
        def print_1d(name, e_arr, c_arr):
            h_str = f"{name+'0':<15} | {e_arr[0]:>25.8e} | " + " | ".join([f"{c_arr[L, 0]:>22.8e}" for L in range(self.L_rad)])
            print(h_str)
            idx = 1
            for n in range(1, self.N_tor + 1):
                c_str = f"{f'{name}{n}c':<15} | {e_arr[idx]:>25.8e} | " + " | ".join([f"{c_arr[L, idx]:>22.8e}" for L in range(self.L_rad)])
                print(c_str)
                s_str = f"{f'{name}{n}s':<15} | {e_arr[idx+1]:>25.8e} | " + " | ".join([f"{c_arr[L, idx+1]:>22.8e}" for L in range(self.L_rad)])
                print(s_str)
                idx+=2

        print_1d("c0R_", e_c0R, c_c0R); print_1d("c0Z_", e_c0Z, c_c0Z)
        print_1d("h_", e_h, c_h); print_1d("v_", e_v, c_v)
        print_1d("k_", e_k, c_k); print_1d("a_", e_a, c_a)
        
        if self.len_2d > 0:
            print("-" * table_width)
            print(">>> 极向高阶摄动分量 (theta_R & theta_Z) [受谱压缩正则化约束]:")
            for i, (m, n, typ) in enumerate(self.modes_2d):
                tR_str = f"{f'tR_{m}_{n}{typ}':<15} | {e_tR[i]:>25.8e} | " + " | ".join([f"{c_tR[L, i]:>22.8e}" for L in range(self.L_rad)])
                print(tR_str)
            for i, (m, n, typ) in enumerate(self.modes_2d):
                tZ_str = f"{f'tZ_{m}_{n}{typ}':<15} | {e_tZ[i]:>25.8e} | " + " | ".join([f"{c_tZ[L, i]:>22.8e}" for L in range(self.L_rad)])
                print(tZ_str)
                
        if self.len_lam > 0:
            print("-" * table_width)
            print(">>> 磁流函数 (Lambda) 谐波分量 [受二次最小化惩罚约束]:") 
            for i, (m, n) in enumerate(self.lambda_modes):
                L_str = f"{f'L_{m}_{n}':<15} | {'-- Null --':>25} | " + " | ".join([f"{c_lam[L, i]:>22.8e}" for L in range(self.L_rad)])
                print(L_str)
        print("=" * table_width + "\n")

    def _wrap_to_2pi(self, ang):
        return np.mod(np.asarray(ang, dtype=float), 2 * np.pi)

    def _angle_distance(self, a, b):
        da = self._wrap_to_2pi(a) - self._wrap_to_2pi(b)
        return np.abs((da + np.pi) % (2 * np.pi) - np.pi)

    def _desc_phi_to_solver_zeta(self, phi_like):
        """
        DESC 对比数据第三列统一转换层：
        约定文件列是 phi（即使列名写成 zeta），统一映射为求解器内部 zeta = N_fp * phi。
        """
        phi_arr = np.asarray(phi_like, dtype=float)
        # 使用原位运算减少临时数组，降低大数据量转换耗时
        np.multiply(phi_arr, self.N_fp, out=phi_arr)
        np.remainder(phi_arr, 2 * np.pi, out=phi_arr)
        return phi_arr

    def plot_equilibrium(self, x_core):
        import os
        zetas = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
        
        # 尝试加载 DESC 的对比数据
        desc_data = None
        unique_zetas = None
        zeta_match_tol = 0.1
        if os.path.exists("RZ_data.txt"):
            try:
                t0 = time.time()
                desc_data = np.loadtxt("RZ_data.txt", delimiter=',', skiprows=1)
                t1 = time.time()
                # 统一将第三列角度转换为求解器内部 zeta
                desc_data[:, 2] = self._desc_phi_to_solver_zeta(desc_data[:, 2])
                t2 = time.time()
                unique_zetas = np.unique(desc_data[:, 2])
                if len(unique_zetas) > 1:
                    zeta_step = np.min(np.diff(unique_zetas))
                    # 自适应阈值：至少覆盖半个离散步长，避免 zeta=pi 这类临界漏匹配
                    zeta_match_tol = max(0.1, 0.55 * zeta_step)
                print("    >>> 成功加载 DESC 对比数据 RZ_data.txt")
                print("    >>> 已应用角度统一转换: zeta = N_fp * phi (mod 2pi)")
                print(f"    >>> DESC 数据读取耗时: {t1 - t0:.3f} s | 角度转换耗时: {t2 - t1:.3f} s")
            except Exception as e:
                print(f"    >>> 无法加载 DESC 数据: {e}")

        fig, axes = plt.subplots(2, 3, figsize=(15, 12))
        axes = axes.flatten()
        rp = np.linspace(0, 1, 50); tp = np.linspace(0, 2*np.pi, 100)
        R_P, T_P = np.meshgrid(rp, tp); PSI_P = self.compute_psi(R_P)
        
        # 与 DESC 数据对比误差累计量
        total_squared_error = 0.0
        total_matched_points = 0
        total_squared_true = 0.0
        total_sum_true_r = 0.0
        total_sum_true_z = 0.0
        for i, zv in enumerate(zetas):
            ax = axes[i]; Rm, Zm = [], []
            for r, t in zip(R_P.flatten(), T_P.flatten()):
                rg = self.compute_geometry(x_core, r, t, zv)
                Rm.append(rg[0]); Zm.append(rg[1])
            Rm = np.array(Rm).reshape(R_P.shape); Zm = np.array(Zm).reshape(R_P.shape)
            ax.tripcolor(Rm.flatten(), Zm.flatten(), PSI_P.flatten(), shading='gouraud', cmap='magma', alpha=0.9)
            
            # 绘制 VEQ3D 内部磁面
            for r_lev in [0.2, 0.4, 0.6, 0.8, 1.0]:
                rl, zl = self.compute_geometry(x_core, r_lev, np.linspace(0, 2*np.pi, 100), zv)[:2]
                ax.plot(rl, zl, color='white', lw=1.0, alpha=0.5, label='VEQ3D Surfaces' if r_lev == 1.0 and i == 0 else "")
            
            th_t = np.linspace(0, 2*np.pi, 200)
            ze_line = np.full_like(th_t, zv)
            r_lcfs = np.asarray(self._boundary_R_fn(th_t, ze_line), dtype=float)
            z_lcfs = np.asarray(self._boundary_Z_fn(th_t, ze_line), dtype=float)
            ax.plot(r_lcfs, z_lcfs, 'r--', lw=1.5, label='Input LCFS')
            
            rl_e, zl_e = self.compute_geometry(x_core, 1.0, np.linspace(0, 2*np.pi, 100), zv)[:2]
            ax.plot(rl_e, zl_e, color='#FFD700', lw=2.0, label='VEQ3D Boundary')
            
            # ==========================================
            # 叠加绘制 DESC 结果
            # ==========================================
            if desc_data is not None and unique_zetas is not None:
                if len(unique_zetas) > 0:
                    # 统一转换后，仅按求解器 zeta 直接匹配
                    err_field = self._angle_distance(unique_zetas, zv)
                    closest_zv = unique_zetas[np.argmin(err_field)]  # 该值来自 DESC 原始切片
                    
                    if np.min(err_field) <= zeta_match_tol:
                        # 严格使用 DESC 文件中“完全相等”的 zeta 切片数据
                        slice_data = desc_data[desc_data[:, 2] == closest_zv]
                        unique_rhos = np.unique(slice_data[:, 0])
                        
                        # 只绘制特定的 rho 层
                        target_rhos = [0.2, 0.4, 0.6, 0.8, 1.0]
                        for r_target in target_rhos:
                            if len(unique_rhos) == 0:
                                continue
                            idx_rho = np.argmin(np.abs(unique_rhos - r_target))
                            r_val = unique_rhos[idx_rho]
                            
                            # 容差内匹配 rho
                            if np.abs(r_val - r_target) < 0.05:
                                # 提取等磁面坐标并按极向角排序
                                r_data = slice_data[np.abs(slice_data[:, 0] - r_val) < 1e-5]
                                r_data = r_data[np.argsort(r_data[:, 1])]
                                if len(r_data) > 0:
                                    # 闭合曲线处理
                                    r_plot = np.append(r_data[:, 3], r_data[0, 3])
                                    z_plot = np.append(r_data[:, 4], r_data[0, 4])
                                    # 暂时关闭 DESC 叠加可视化输出（保留误差统计）
                                    ax.plot(r_plot, z_plot, color='cyan', linestyle=':', lw=1.5,
                                            label='DESC Data' if r_target == 1.0 and i == 0 else "")
                                    
                                    # 统计该层对应点上的误差（按 DESC 给定 theta 逐点比较）
                                    th_desc = r_data[:, 1]
                                    R_desc = r_data[:, 3]
                                    Z_desc = r_data[:, 4]
                                    # 统计点使用 DESC 切片的精确 zeta，保证 zeta 完全相等
                                    R_calc, Z_calc, _, _, _, _, _ = self.compute_geometry(x_core, r_val, th_desc, closest_zv)
                                    
                                    total_squared_error += np.sum((R_calc - R_desc)**2 + (Z_calc - Z_desc)**2)
                                    total_matched_points += len(th_desc)
                                    total_squared_true += np.sum(R_desc**2 + Z_desc**2)
                                    total_sum_true_r += np.sum(R_desc)
                                    total_sum_true_z += np.sum(Z_desc)
            # ==========================================

            ax.set_aspect('equal'); ax.set_title(fr'Field Period Angle $\zeta={zv:.2f}$')
            ax.set_xlabel('R (m)')
            ax.set_ylabel('Z (m)')
            if i == 0: ax.legend(loc='upper right', fontsize='xx-small')

        # 输出 RMSE 与两类相对误差
        if desc_data is not None and total_matched_points > 0:
            rmse = np.sqrt(total_squared_error / total_matched_points)
            rms_true = np.sqrt(total_squared_true / total_matched_points)
            rel_err_l2 = rmse / rms_true if rms_true > 0 else 0.0

            mean_r = total_sum_true_r / total_matched_points
            mean_z = total_sum_true_z / total_matched_points
            mean_dist = np.sqrt(mean_r**2 + mean_z**2)
            rel_err_mean = rmse / mean_dist if mean_dist > 0 else 0.0

            print("\n" + "="*70)
            print(">>> 与 DESC 数据对比的误差统计:")
            print(f"    绝对均方根误差 (RMSE): {rmse:.6e}")
            print(f"    L2 相对误差:           {rel_err_l2:.6e} ({rel_err_l2*100:.4f}%)")
            print(f"    基于均值的相对误差:    {rel_err_mean:.6e} ({rel_err_mean*100:.4f}%)")
            print(f"    (基于统计的共计 {total_matched_points} 个对应网格点)")
            print("="*70 + "\n")
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    VEQ3D_Solver().solve()
