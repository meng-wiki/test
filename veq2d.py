import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import time
import json

class VEQ2D_Tokamak_Solver:
    def __init__(self, json_filepath="demo-2-packed.json"):
        # =========================================================
        # [DESC降维逻辑] 剔除环向自由度，大幅提升极向和径向解析度
        # =========================================================
        self.M_pol = 8      # 极向模数 (2D 需要较高的极向分辨率)
        self.N_tor = 0      # 环向模数强制为 0 (Axisymmetry)
        self.L_rad = 10     # 径向 Zernike 类多项式阶数
        # =========================================================
        
        self.N_fp = 1       # 托卡马克场周期为 1
        self.mu_0 = 4 * np.pi * 1e-7
        
        self.target_Nr = 20
        self.target_Nt = 32
        self.target_Nz = 1  # 环向网格点固定为 1
        
        self.p_edge = None  
        
        # 1. 加载并解析目标 JSON 数据
        self._load_reference_data(json_filepath)
        
        # 2. 构建模态与网格
        self._setup_modes()
        self.update_grid(self.target_Nr, self.target_Nt, self.target_Nz)
        
        # 3. 拟合边界 & 初始化
        self.fit_boundary()  
        self._initialize_scaling()

    def _load_reference_data(self, filepath):
        """解析 demo-2-packed.json 中的物理剖面与几何边界"""
        print(f">>> 正在加载基准数据: {filepath} ...")
        with open(filepath, 'r') as f:
            self.demo_data = json.load(f)
            
        self.rho_ref = np.array(self.demo_data['profiles']['rho'])
        self.psi_ref = np.array(self.demo_data['profiles']['psi'])
        self.q_ref = np.array(self.demo_data['profiles']['q'])
        self.P_psi_ref = np.array(self.demo_data['profiles']['P_psi'])
        
        # 使用三次样条计算 dpsi/drho
        psi_spline = CubicSpline(self.rho_ref, self.psi_ref)
        self.psip_ref = psi_spline(self.rho_ref, 1) # 求一阶导
        
        # Phip = q * dpsi/drho (托卡马克磁流学核心关系)
        self.Phip_ref = self.q_ref * self.psip_ref
        
        # 压力梯度 dP/drho = (dP/dpsi) * (dpsi/drho)
        self.dP_drho_ref = self.P_psi_ref * self.psip_ref
        
        # 逆向积分获取实际压力 P(rho)，边界处 P(1.0) = 0
        self.P_ref = np.zeros_like(self.rho_ref)
        for i in range(len(self.rho_ref)-2, -1, -1):
            d_rho = self.rho_ref[i+1] - self.rho_ref[i]
            avg_dP = 0.5 * (self.dP_drho_ref[i+1] + self.dP_drho_ref[i])
            self.P_ref[i] = self.P_ref[i+1] - d_rho * avg_dP

    def update_grid(self, Nr, Nt_grid, Nz_grid):
        self.Nr = Nr
        self.Nt_grid = Nt_grid
        self.Nz_grid = Nz_grid  # 在2D模式下，这将被强制为1
        
        rho_nodes, self.rho_weights = self._get_chebyshev_nodes_and_weights(self.Nr)
        self.rho = 0.5 * (rho_nodes + 1)
        self.rho_weights *= 0.5
        
        self.D_matrix = self._get_spectral_diff_matrix(self.rho)
        
        self.k_th = np.fft.fftfreq(self.Nt_grid, d=1.0/self.Nt_grid)[None, :, None]
        self.k_ze = np.fft.fftfreq(self.Nz_grid, d=1.0/self.Nz_grid)[None, None, :]
        
        self.theta = np.linspace(0, 2*np.pi, self.Nt_grid, endpoint=False)
        self.zeta = np.linspace(0, 2*np.pi, self.Nz_grid, endpoint=False)
        self.dtheta = 2 * np.pi / self.Nt_grid
        self.dzeta = 2 * np.pi / self.Nz_grid if self.Nz_grid > 1 else 1.0 # 防止除0
        
        self.RHO, self.TH, self.ZE = np.meshgrid(self.rho, self.theta, self.zeta, indexing='ij')
        self.weights_3d = self.rho_weights[:, None, None]

        self._precompute_radial_factors() 
        self._build_basis_matrices()

    def _precompute_radial_factors(self):
        # 纯粹径向的正交展开 (类 Zernike 多项式的径向部分)
        x = 2.0 * self.rho**2 - 1.0
        T_list, dTdx_list = [], []
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
            self.reg_weight_2d = np.sqrt(1e-6 * (m_vals_2d**2 + n_vals_2d**2))
        else:
            self.reg_weight_2d = np.array([])

    def _setup_modes(self):
        self.len_1d = 1 + 2 * self.N_tor  # 2D时 = 1
        
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
        print(f">>> 2D 参数体系已重构: 空间维度 (M={self.M_pol}, N={self.N_tor}, L={self.L_rad})")
        print(f">>> 总核心参数量: {self.num_core_params} 个")
        self.res_scales = np.ones(self.num_core_params)
        self.res_scales[:self.num_geom_params] = 1e5
        self.res_scales[self.num_geom_params:] = 1e6

    def compute_psi(self, rho):
        """解析解：直接插值实际数据的极向磁通"""
        return np.interp(rho, self.rho_ref, self.psi_ref)

    def fit_boundary(self):
        """完全匹配 demo-2-packed.json 提供的最外层封闭磁面(LCFS)"""
        theta_ref = np.array(self.demo_data['surface_grid']['theta'])
        R_bnd = np.array(self.demo_data['surface_grid']['R'])[-1, :]
        Z_bnd = np.array(self.demo_data['surface_grid']['Z'])[-1, :]
        
        # 将参考数据延拓以支持安全的周期插值
        theta_ext = np.concatenate([theta_ref - 2*np.pi, theta_ref, theta_ref + 2*np.pi])
        R_ext = np.concatenate([R_bnd, R_bnd, R_bnd])
        Z_ext = np.concatenate([Z_bnd, Z_bnd, Z_bnd])
        
        TH_F = self.TH[0, :, 0] # 极向网格一维化
        R_target_1D = np.interp(TH_F, theta_ext, R_ext)
        Z_target_1D = np.interp(TH_F, theta_ext, Z_ext)
        
        # 恢复维度以适用原求解器形状
        R_target = R_target_1D[:, None]
        Z_target = Z_target_1D[:, None]
        
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
            
            # DESC 边界对准公式
            thR = self.TH[0] + c0R_v + tR_v
            thZ = self.TH[0] + c0Z_v + tZ_v
            
            R_mod = R0 + a_v * (h_v + np.cos(thR))
            Z_mod = Z0 + a_v * (v_v - k_v * np.sin(thZ))
            
            res_geom = np.concatenate([(R_mod - R_target).flatten(), (Z_mod - Z_target).flatten()])
            res_reg = np.array([h_c[0], v_c[0]]) * 100.0
            return np.concatenate([res_geom, res_reg])
            
        p0 = np.zeros(self.num_edge_params)
        p0[0] = 1.17 # R_center
        p0[2 + 4 * self.len_1d] = 1.0 # k0
        p0[2 + 5 * self.len_1d] = 0.3 # a0 边界短轴近似
        
        res = least_squares(boundary_residuals, p0, method='trf', ftol=1e-12)
        self.p_edge = res.x
        print(f"    >>> JSON真实边界对准完成 (Max Residual: {np.max(np.abs(res.fun)):.3e})")

    def _build_jax_residual_fn(self, pressure_scale_factor=1.0):
        RHO, TH, ZE = jnp.array(self.RHO), jnp.array(self.TH), jnp.array(self.ZE)
        rho_1d, D_matrix = jnp.array(self.rho), jnp.array(self.D_matrix)
        
        fac_rad, dfac_rad = jnp.array(self.fac_rad), jnp.array(self.dfac_rad)
        fac_lam_eval, fac_lam_proj = jnp.array(self.fac_lam_eval), jnp.array(self.fac_lam_proj)
        rho_m_lam, reg_weight_2d = jnp.array(self.rho_m_lam), jnp.array(self.reg_weight_2d)

        basis_1d_val_slice, basis_1d_dz_slice = jnp.array(self.basis_1d_val[:, 0, 0, :]), jnp.array(self.basis_1d_dz[:, 0, 0, :])
        basis_2d_val, basis_2d_dr = jnp.array(self.basis_2d_val), jnp.array(self.basis_2d_dr)
        basis_2d_dth, basis_2d_dze = jnp.array(self.basis_2d_dth), jnp.array(self.basis_2d_dze)
        
        basis_lam_tz, basis_lam_dth, basis_lam_dze = jnp.array(self.basis_lam_val[:, 0, :, :]), jnp.array(self.basis_lam_dth[:, 0, :, :]), jnp.array(self.basis_lam_dze[:, 0, :, :])
        
        k_th, k_ze, weights_3d = jnp.array(self.k_th), jnp.array(self.k_ze), jnp.array(self.weights_3d)
        res_scales, p_edge = jnp.array(self.res_scales), jnp.array(self.p_edge)
        
        L_rad, len_1d, len_2d, len_lam = self.L_rad, self.len_1d, self.len_2d, self.len_lam
        N_fp, mu_0, dtheta, dzeta = self.N_fp, self.mu_0, self.dtheta, self.dzeta
            
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

        # 闭包注入样条插值
        j_rho_ref = jnp.array(self.rho_ref)
        j_P_ref = jnp.array(self.P_ref)
        j_dP_drho_ref = jnp.array(self.dP_drho_ref)
        j_Phip_ref = jnp.array(self.Phip_ref)
        j_psip_ref = jnp.array(self.psip_ref)

        def jax_res_fn(x_core, apply_scaling=True):
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
            
            R = e_R0 + a * (h + RHO * jnp.cos(thR))
            Rr = ar * (h + RHO * jnp.cos(thR)) + a * (hr + jnp.cos(thR) - RHO * jnp.sin(thR) * thR_r)
            Rt = -a * RHO * jnp.sin(thR) * thR_th
            Rz = az * (h + RHO * jnp.cos(thR)) + a * (hz - RHO * jnp.sin(thR) * thR_z)
            
            Z = e_Z0 + a * (v - k * RHO * jnp.sin(thZ))
            Zr = ar * (v - k * RHO * jnp.sin(thZ)) + a * (vr - kr * RHO * jnp.sin(thZ) - k * jnp.sin(thZ) - k * RHO * jnp.cos(thZ) * thZ_r)
            Zt = -a * k * RHO * jnp.cos(thZ) * thZ_th
            Zz = az * (v - k * RHO * jnp.sin(thZ)) + a * (vz - kz * RHO * jnp.sin(thZ) - k * RHO * jnp.cos(thZ) * thZ_z)

            # [修改]：翻转雅可比行列式计算公式，适配顺时针(CW)坐标映射，避免生成负面积触发天价惩罚项
            det_phys = Rt * Zr - Rr * Zt
            sqrt_g = (R / N_fp) * det_phys 
            
            g_rr, g_tt = Rr**2 + Zr**2, Rt**2 + Zt**2
            g_zz = Rz**2 + (R/N_fp)**2 + Zz**2 
            g_rt, g_rz, g_tz = Rr*Rt+Zr*Zt, Rr*Rz+Zr*Zz, Rt*Rz+Zt*Zz

            if len_lam > 0:
                lam_ce = jnp.dot(c_lam.T, fac_lam_eval) * rho_m_lam
                Lt = jnp.einsum('mr,mtz->rtz', lam_ce, basis_lam_dth)
                Lz = jnp.einsum('mr,mtz->rtz', lam_ce, basis_lam_dze)
            else:
                Lt, Lz = 0.0, 0.0
            
            # 使用从 JSON 导入的 JAX 插值函数替代硬编码的二次型剖面
            P = jnp.interp(RHO, j_rho_ref, j_P_ref) * pressure_scale_factor
            dP = jnp.interp(RHO, j_rho_ref, j_dP_drho_ref) * pressure_scale_factor
            Phip = jnp.interp(RHO, j_rho_ref, j_Phip_ref)
            psip = jnp.interp(RHO, j_rho_ref, j_psip_ref)

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

            F_rho = G_rho
            F_beta = (Jr_phys / (2 * jnp.pi)) * (Phip + Lt)

            metric_w = (R / N_fp) * weights_3d * (dtheta * dzeta)
            
            GR_w = (Zt * F_rho - Zr * F_beta) * metric_w
            GZ_w = (-Rt * F_rho + Rr * F_beta) * metric_w
            
            term1 = GR_w * (-a * RHO * jnp.sin(thR))
            term2 = GZ_w * (-a * k * RHO * jnp.cos(thZ))
            term3 = GR_w * a
            term4 = GZ_w * a
            term5 = GZ_w * (-a * RHO * jnp.sin(thZ))
            term6 = GR_w * (h + RHO * jnp.cos(thR)) + GZ_w * (v - k * RHO * jnp.sin(thZ))
            
            terms_1d = jnp.stack([term1, term2, term3, term4, term5, term6], axis=0) 
            terms_1d_t = jnp.sum(terms_1d, axis=2) 
            terms_1d_tz = jnp.einsum('vrz,mz->vrm', terms_1d_t, basis_1d_val_slice) 
            res_1d = jnp.einsum('lr,vrm->lvm', fac_rad, terms_1d_tz).reshape((L_rad, 6 * len_1d)) 
            
            if len_2d > 0:
                term7 = GR_w * (-a * RHO * jnp.sin(thR))
                term8 = GZ_w * (-a * k * RHO * jnp.cos(thZ))
                terms_2d = jnp.stack([term7, term8], axis=0)
                terms_2d_r = jnp.einsum('vrtz,mrtz->vmr', terms_2d, basis_2d_val)
                res_2d = jnp.einsum('lr,vmr->lvm', fac_rad, terms_2d_r).reshape((L_rad, 2 * len_2d))
                res_geom = jnp.concatenate([res_1d, res_2d], axis=1).flatten()
            else:
                res_geom = res_1d.flatten()
                
            res_list = [res_geom]
            
            if len_lam > 0:
                vol_w = metric_w * det_phys
                term_lam = Jr_phys * vol_w
                term_lam_tz = jnp.einsum('rtz,mtz->rm', term_lam, basis_lam_tz) 
                res_lam = jnp.dot(fac_lam_proj, rho_m_lam.T * term_lam_tz)
                res_list.append(res_lam.flatten())
                
            phys_res = jnp.concatenate(res_list)

            if apply_scaling:
                phys_res = phys_res / res_scales
                
            penalty_res = jnp.where(det_phys < 1e-6, 1e4 * (1e-6 - det_phys), 0.0).flatten()
                
            final_res_list = [phys_res, penalty_res]
            
            if len_2d > 0:
                final_res_list.append((c_tR * reg_weight_2d[None, :]).flatten())
                final_res_list.append((c_tZ * reg_weight_2d[None, :]).flatten())
                
            if len_lam > 0:
                final_res_list.append((c_lam * jnp.sqrt(1e-6)).flatten())
                
            return jnp.concatenate(final_res_list)
            
        return jax_res_fn

    def _run_optimization(self, x0, max_nfev, ftol, pressure_scale_factor=1.0):
        jax_res_fn = self._build_jax_residual_fn(pressure_scale_factor)
        
        @jax.jit
        def fun_wrapped(x_arr):
            return jax_res_fn(x_arr, apply_scaling=True)
            
        @jax.jit
        def jac_wrapped(x_arr):
            return jax.jacfwd(fun_wrapped)(x_arr)

        start_time = time.time()
        print(f"    >>> JAX直接优化引擎启动 (网格: {self.Nr}x{self.Nt_grid}x{self.Nz_grid}, 标度={pressure_scale_factor})")
        
        res = least_squares(
            fun_wrapped, 
            np.array(x0), 
            jac=jac_wrapped, 
            method='trf', 
            xtol=ftol, 
            ftol=ftol, 
            max_nfev=max_nfev,
            verbose=2 
        )

        end_time = time.time()
        phys_res_final = jax_res_fn(jnp.array(res.x), apply_scaling=True)
        phys_len = self.num_core_params 
        phys_res_only = phys_res_final[:phys_len]
        
        print(f"    当前网格总耗时: {end_time - start_time:.2f} s | 总函数计算: {res.nfev} 次")
        print(f"    纯主物理残差 L2 范数: {np.linalg.norm(phys_res_only):.4e}")
        
        return res

    def solve(self):
        print(">>> 启动 VEQ-2D 托卡马克谱平衡求解器 (完全对接 DESC Json 数据)...")

        # ======================================================================
        # [Phase 1/2]: 全压预收敛 (固定高保真网格，防止混叠震荡)
        # ======================================================================
        # 必须满足 Nyquist 定理，Nr > 2*L, Nt > 2*M，所以 42x36 是安全的抗混叠网格
        c_Nr, c_Nt = 42, 36  
        c_Nz = 1
        
        print("\n" + "="*70)
        print(f">>> [Phase 1/2]: 密网格全压预收敛 (Nr={c_Nr}, Nt={c_Nt}, Nz={c_Nz}, P=1.0)")
        print("="*70)
        self.update_grid(c_Nr, c_Nt, c_Nz)
        
        x_guess = np.zeros(self.num_core_params)
        
        # 第一阶段：较松的容差，让优化器快速找到宏观力平衡区间
        res_phase1 = self._run_optimization(x_guess, max_nfev=150, ftol=1e-3, pressure_scale_factor=1.0)

        # ======================================================================
        # [Phase 2/2]: 全物理极限收敛 (保持相同网格，收紧容差)
        # ======================================================================
        print("\n" + "="*70)
        print(f">>> [Phase 2/2]: 全物理极限精确求解 (Nr={c_Nr}, Nt={c_Nt}, Nz={c_Nz}, P=1.0)")
        print("="*70)
        
        # 【关键】：千万不要再调用 self.update_grid() 改变网格，锁定积分权重！
        # 第二阶段：极严苛的容差，压榨出 1e-11 级别的谱精度极限
        res_fine = self._run_optimization(res_phase1.x, max_nfev=500, ftol=1e-11, pressure_scale_factor=1.0)

        self.plot_equilibrium(res_fine.x)
        return res_fine.x

    def compute_geometry(self, x_core, rho, theta, zeta):
        rho, theta, zeta = np.atleast_1d(rho), np.atleast_1d(theta), np.atleast_1d(zeta)
        base_grid = rho + theta + zeta
        x = 2.0 * rho**2 - 1.0
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
        
        R = e_R0 + a * (h + rho * np.cos(thR))
        Z = e_Z0 + a * (v - k * rho * np.sin(thZ))
        
        lam = np.zeros_like(base_grid)
        for i, (m, n) in enumerate(self.lambda_modes):
            L_fac_rad_m = (rho ** m) * (1 - rho**2)**2 * T
            lam = lam + np.tensordot(c_lam[:, i], L_fac_rad_m, axes=(0, 0)) * np.sin(m * theta - n * zeta)
            
        return R, Z, thR, thZ, a, k, lam

    def plot_equilibrium(self, x_core):
        """对比输出 VEQ-2D 计算结果与 demo-2-packed 的基准值"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        target_rhos = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        th_plot = np.linspace(0, 2*np.pi, 150)
        
        # =========================================================
        # 1. 绘制 VEQ-2D 计算得到的等磁面 (Psi(R,Z) 轮廓线)
        # =========================================================
        for i, r_lev in enumerate(target_rhos):
            rl, zl = self.compute_geometry(x_core, r_lev, th_plot, 0.0)[:2]
            label = r'VEQ-2D Calculated $\psi(R,Z)$' if i == 0 else ""
            ax.plot(rl, zl, color='blue', lw=2.0, alpha=0.8, label=label)
            
        # =========================================================
        # 2. 叠加绘制 demo-2-packed.json 提供的精确解进行对比
        # =========================================================
        demo_R = np.array(self.demo_data['surface_grid']['R'])
        demo_Z = np.array(self.demo_data['surface_grid']['Z'])
        
        for i, r_lev in enumerate(target_rhos):
            idx = np.argmin(np.abs(self.rho_ref - r_lev))
            # 闭合曲线
            r_plot = np.append(demo_R[idx, :], demo_R[idx, 0])
            z_plot = np.append(demo_Z[idx, :], demo_Z[idx, 0])
            label = r'DESC Target $\psi(R,Z)$' if i == 0 else ""
            ax.plot(r_plot, z_plot, color='red', linestyle='--', lw=1.8, label=label)

        # 绘制背景底色表达 Psi(R,Z) 强度
        R_bg, Z_bg, Psi_bg = [], [], []
        r_grid, t_grid = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 2*np.pi, 60))
        for rg, tg in zip(r_grid.flatten(), t_grid.flatten()):
            rl, zl = self.compute_geometry(x_core, rg, tg, 0.0)[:2]
            R_bg.append(rl)
            Z_bg.append(zl)
            Psi_bg.append(self.compute_psi(rg))
            
        ax.tricontourf(np.array(R_bg).flatten(), np.array(Z_bg).flatten(), np.array(Psi_bg).flatten(), 
                       levels=30, cmap='magma', alpha=0.3)

        ax.set_aspect('equal')
        ax.set_title("2D Tokamak Equilibrium Contour: VEQ-2D vs JSON Target", fontsize=14)
        ax.set_xlabel("R [m]", fontsize=12)
        ax.set_ylabel("Z [m]", fontsize=12)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    solver = VEQ2D_Tokamak_Solver("demo-2-packed.json")
    solver.solve()
