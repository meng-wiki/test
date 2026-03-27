import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import time

class VEQ3D_Solver:
    def __init__(self):
        # =========================================================
        # [核心可调参数区] 自由调节极向 M、环向 N 与径向 L 阶数
        # =========================================================
        self.M_pol = 1
        self.N_tor = 1
        self.L_rad = 2
        # =========================================================
        
        self.Nt = 19
        self.Phi_a = 1.0
        self.mu_0 = 4 * np.pi * 1e-7
        
        # 保存目标高精度网格尺寸
        self.target_Nr = 32
        self.target_Nt = 32
        self.target_Nz = 32
        
        self.p_edge = None  # 增加初始化，用于后续的边界热启动
        
        self._setup_modes()
        
        # 初始化基础参数 (先以目标网格初始化，以确定参数长度)
        self.update_grid(self.target_Nr, self.target_Nt, self.target_Nz)
        self._initialize_scaling()

    def update_grid(self, Nr, Nt_grid, Nz_grid):
        """动态网格重构引擎：支持在优化过程中无缝切换网格分辨率"""
        self.Nr = Nr
        self.Nt_grid = Nt_grid
        self.Nz_grid = Nz_grid
        
        rho_nodes, self.rho_weights = self._get_chebyshev_nodes_and_weights(self.Nr)
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

        self._build_basis_matrices()
        self.fit_boundary()

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
        # --- 1D 模式基底构建 ---
        self.basis_1d_val = np.zeros((self.len_1d, 1, 1, self.Nz_grid))
        self.basis_1d_dz  = np.zeros((self.len_1d, 1, 1, self.Nz_grid))
        self.basis_1d_val[0, 0, 0, :] = 1.0
        idx = 1
        for n in range(1, self.N_tor + 1):
            self.basis_1d_val[idx, 0, 0, :] = np.cos(n * self.ZE[0,0,:]); self.basis_1d_dz[idx, 0, 0, :] = -n * np.sin(n * self.ZE[0,0,:]); idx+=1
            self.basis_1d_val[idx, 0, 0, :] = np.sin(n * self.ZE[0,0,:]); self.basis_1d_dz[idx, 0, 0, :] =  n * np.cos(n * self.ZE[0,0,:]); idx+=1
            
        # --- 2D 模式基底构建 (包含 rho^m 因子) ---
        self.basis_2d_val = np.zeros((self.len_2d, self.Nr, self.Nt_grid, self.Nz_grid))
        self.basis_2d_dr  = np.zeros((self.len_2d, self.Nr, self.Nt_grid, self.Nz_grid))
        self.basis_2d_dth = np.zeros((self.len_2d, self.Nr, self.Nt_grid, self.Nz_grid))
        self.basis_2d_dze = np.zeros((self.len_2d, self.Nr, self.Nt_grid, self.Nz_grid))
        self.basis_2d_edge = np.zeros((self.len_2d, self.Nt_grid, self.Nz_grid))
        
        for i, (m, n, typ) in enumerate(self.modes_2d):
            phase = m * self.TH - n * self.ZE
            rho_m = self.RHO ** m
            
            # 边界上的无 rho 影响基底(此时 rho=1)
            phase_edge = m * self.TH[0,:,:] - n * self.ZE[0,:,:]
            
            # 解析径向导数 d(rho^m)/drho
            if m == 0:
                drho_m = np.zeros_like(self.RHO)
            else:
                drho_m = m * (self.RHO ** (m - 1))
                
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
                
        # --- 磁流函数 Lambda 模式基底 ---
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
        print(f">>> 总优化参数量: {self.num_core_params} 个 (几何: {self.num_geom_params}, 流函数: {self.len_lam * self.L_rad})")
        self.res_scales = np.ones(self.num_core_params)
        self.res_scales[:self.num_geom_params] = 1e5
        self.res_scales[self.num_geom_params:] = 1e6

    def compute_psi(self, rho):
        return self.Phi_a * (rho**2 + 0.75 * rho**4)

    def fit_boundary(self):
        TH_F, ZE_F = self.TH[0], self.ZE[0]
        R_target = 10 - np.cos(TH_F) - 0.3 * np.cos(TH_F + ZE_F)
        Z_target = np.sin(TH_F) - 0.3 * np.sin(TH_F + ZE_F)
        
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
            R_mod = R0 + h_v + a_v * np.cos(thR)
            Z_mod = Z0 + v_v + k_v * a_v * np.sin(thZ)
            
            res_geom = np.concatenate([(R_mod - R_target).flatten(), (Z_mod - Z_target).flatten()])
            res_reg = np.array([h_c[0], v_c[0]]) * 100.0
            return np.concatenate([res_geom, res_reg])
            
        if self.p_edge is not None and len(self.p_edge) == self.num_edge_params:
            p0 = self.p_edge.copy()
        else:
            p0 = np.zeros(self.num_edge_params)
            p0[0] = 10.0 
            p0[2] = np.pi 
            p0[2 + 4 * self.len_1d] = 1.0 
            p0[2 + 5 * self.len_1d] = 1.0 
        
        res = least_squares(boundary_residuals, p0, method='trf', ftol=1e-12)
        self.p_edge = res.x

    def _build_jax_residual_fn(self):
        RHO = jnp.array(self.RHO)
        TH = jnp.array(self.TH)
        ZE = jnp.array(self.ZE)
        rho_1d = jnp.array(self.rho)
        D_matrix = jnp.array(self.D_matrix)
        basis_1d_val = jnp.array(self.basis_1d_val)
        basis_1d_dz = jnp.array(self.basis_1d_dz)
        basis_2d_val = jnp.array(self.basis_2d_val)
        basis_2d_dr = jnp.array(self.basis_2d_dr)
        basis_2d_dth = jnp.array(self.basis_2d_dth)
        basis_2d_dze = jnp.array(self.basis_2d_dze)
        basis_lam_val = jnp.array(self.basis_lam_val)
        basis_lam_dth = jnp.array(self.basis_lam_dth)
        basis_lam_dze = jnp.array(self.basis_lam_dze)
        k_th = jnp.array(self.k_th)
        k_ze = jnp.array(self.k_ze)
        weights_3d = jnp.array(self.weights_3d)
        res_scales = jnp.array(self.res_scales)
        p_edge = jnp.array(self.p_edge)
        
        L_rad = self.L_rad
        len_1d = self.len_1d
        len_2d = self.len_2d
        len_lam = self.len_lam
        Nt = self.Nt
        mu_0 = self.mu_0
        Phi_a = self.Phi_a
        dtheta = self.dtheta
        dzeta = self.dzeta

        # 提取各个Lambda模式的极向阶数m，用于构造 rho^m
        if self.len_lam > 0:
            lam_m_vals = jnp.array([m for m, n in self.lambda_modes])
        else:
            lam_m_vals = jnp.array([])

        def jax_unpack_edge(p):
            idx = 2
            def get(L):
                nonlocal idx
                c = p[idx:idx+L]
                idx += L
                return c
            return p[0], p[1], get(len_1d), get(len_1d), get(len_1d), get(len_1d), get(len_1d), get(len_1d), get(len_2d), get(len_2d)

        def jax_unpack_core(x_core):
            idx = 0
            def get(L):
                nonlocal idx
                c = x_core[idx:idx+L*L_rad].reshape((L_rad, L))
                idx += L * L_rad
                return c
            return get(len_1d), get(len_1d), get(len_1d), get(len_1d), get(len_1d), get(len_1d), get(len_2d), get(len_2d), get(len_lam)

        def spectral_grad_th(f): 
            return jnp.real(jnp.fft.ifft(1j * k_th * jnp.fft.fft(f, axis=1), axis=1))
            
        def spectral_grad_ze(f): 
            return jnp.real(jnp.fft.ifft(1j * k_ze * jnp.fft.fft(f, axis=2), axis=2))

        def jax_res_fn(x_core, apply_scaling=True):
            x = 2.0 * rho_1d**2 - 1.0
            
            T_list = []
            dTdx_list = []
            if L_rad > 0:
                T_list.append(jnp.ones_like(x))
                dTdx_list.append(jnp.zeros_like(x))
            if L_rad > 1:
                T_list.append(x)
                dTdx_list.append(jnp.ones_like(x))
            for l in range(2, L_rad):
                T_new = 2.0 * x * T_list[l-1] - T_list[l-2]
                dTdx_new = 2.0 * T_list[l-1] + 2.0 * x * dTdx_list[l-1] - dTdx_list[l-2]
                T_list.append(T_new)
                dTdx_list.append(dTdx_new)
                
            T = jnp.stack(T_list, axis=0) if L_rad > 0 else jnp.empty((0,) + x.shape)
            dTdx = jnp.stack(dTdx_list, axis=0) if L_rad > 0 else jnp.empty((0,) + x.shape)
            
            fac_rad = (1.0 - rho_1d**2) * T
            dfac_rad = -2.0 * rho_1d * T + 4.0 * rho_1d * (1.0 - rho_1d**2) * dTdx
            
            e_R0, e_Z0, e_c0R, e_c0Z, e_h, e_v, e_k, e_a, e_tR, e_tZ = jax_unpack_edge(p_edge)
            c_c0R, c_c0Z, c_h, c_v, c_k, c_a, c_tR, c_tZ, c_lam = jax_unpack_core(x_core)

            # ========================================================
            # [核心提速优化] 1D 和 2D 求解器全部改用 CPU 最爱的内积+Einsum
            # 彻底摒弃内存开销巨大的多维张量 Broadcasting
            # ========================================================
            def eval_1d(c_e, c_c):
                # c_c: (L_rad, len_1d) -> .T -> (len_1d, L_rad)
                # fac_rad: (L_rad, Nr) 
                core_contrib = jnp.dot(c_c.T, fac_rad)  # (len_1d, Nr)
                ce_eff = c_e[:, None] + core_contrib    # (len_1d, Nr)
                
                b_val = basis_1d_val[:, 0, 0, :]        # (len_1d, Nz)
                b_dz  = basis_1d_dz[:, 0, 0, :]         # (len_1d, Nz)
                
                val = jnp.einsum('mr,mz->rz', ce_eff, b_val)[:, None, :] # (Nr, 1, Nz)
                dz  = jnp.einsum('mr,mz->rz', ce_eff, b_dz)[:, None, :]  # (Nr, 1, Nz)
                
                dr_contrib = jnp.dot(c_c.T, dfac_rad)   # (len_1d, Nr)
                dr  = jnp.einsum('mr,mz->rz', dr_contrib, b_val)[:, None, :]
                return val, dr, dz

            def eval_2d(c_e, c_c):
                if len_2d == 0: 
                    return 0.0, 0.0, 0.0, 0.0
                core_contrib = jnp.dot(c_c.T, fac_rad)  # (len_2d, Nr)
                ce_eff = c_e[:, None] + core_contrib    # (len_2d, Nr)
                
                # 张量收缩: (len_2d, Nr) x (len_2d, Nr, Nt, Nz) -> (Nr, Nt, Nz)
                val = jnp.einsum('mr,mrtz->rtz', ce_eff, basis_2d_val)
                dth = jnp.einsum('mr,mrtz->rtz', ce_eff, basis_2d_dth)
                dz  = jnp.einsum('mr,mrtz->rtz', ce_eff, basis_2d_dze)
                
                dr_contrib = jnp.dot(c_c.T, dfac_rad)   # (len_2d, Nr)
                dr = jnp.einsum('mr,mrtz->rtz', dr_contrib, basis_2d_val) + \
                     jnp.einsum('mr,mrtz->rtz', ce_eff, basis_2d_dr)
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
            R = e_R0 + h + RHO * a * jnp.cos(thR)
            Z = e_Z0 + v + k * RHO * a * jnp.sin(thZ)
            
            thR_r, thR_th, thR_z = c0Rr + tRr, 1.0 + tRth, c0Rz + tRz
            thZ_r, thZ_th, thZ_z = c0Zr + tZr, 1.0 + tZth, c0Zz + tZz
            
            Rr = hr + a * jnp.cos(thR) + RHO * ar * jnp.cos(thR) - RHO * a * jnp.sin(thR) * thR_r
            Rt = -RHO * a * jnp.sin(thR) * thR_th
            Rz = hz + RHO * az * jnp.cos(thR) - RHO * a * jnp.sin(thR) * thR_z
            
            Zr = vr + kr * RHO * a * jnp.sin(thZ) + k * a * jnp.sin(thZ) + k * RHO * ar * jnp.sin(thZ) + k * RHO * a * jnp.cos(thZ) * thZ_r
            Zt = k * RHO * a * jnp.cos(thZ) * thZ_th
            Zz = vz + kz * RHO * a * jnp.sin(thZ) + k * RHO * az * jnp.sin(thZ) + k * RHO * a * jnp.cos(thZ) * thZ_z

            det_phys = Rr * Zt - Rt * Zr
            # 防御机制
            det_safe = jnp.where(jnp.abs(det_phys) < 1e-13, -1e-13, det_phys)
            
            sqrt_g = (R / Nt) * det_safe
            
            g_rr, g_tt = Rr**2 + Zr**2, Rt**2 + Zt**2
            g_zz = Rz**2 + (R/Nt)**2 + Zz**2 
            g_rt, g_rz, g_tz = Rr*Rt+Zr*Zt, Rr*Rz+Zr*Zz, Rt*Rz+Zt*Zz

            if len_lam > 0:
                # [向量化更新]: 构造针对各Lambda模式具有 rho^m 衰减特性的系数
                rho_m_lam = rho_1d[None, :] ** lam_m_vals[:, None] # (len_lam, Nr)
                lam_ce = jnp.dot(c_lam.T, T) * rho_m_lam * (1.0 - rho_1d**2)**2
                
                b_dth = basis_lam_dth[:, 0, :, :]    # (len_lam, Nt, Nz)
                b_dze = basis_lam_dze[:, 0, :, :]    # (len_lam, Nt, Nz)
                Lt = jnp.einsum('mr,mtz->rtz', lam_ce, b_dth)
                Lz = jnp.einsum('mr,mtz->rtz', lam_ce, b_dze)
            else:
                Lt = 0.0
                Lz = 0.0
            
            P_scale = 1.8e4 
            P = P_scale * (RHO**2 - 1)**2
            dP = P_scale * 4 * RHO * (RHO**2 - 1)
            Phip = 2 * RHO * Phi_a
            iota = 1.0 + 1.5 * RHO**2
            psip = iota * Phip

            Bt_sup = (psip - Lz) / (2 * jnp.pi * sqrt_g)
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
            rho_R, rho_Z = Zt/det_safe, -Rt/det_safe
            th_R, th_Z = -Zr/det_safe, Rr/det_safe
            
            GR = (rho_R * G_rho + (Jr_phys / (2 * jnp.pi)) * (th_R * (Phip + Lt)))
            GZ = (rho_Z * G_rho + (Jr_phys / (2 * jnp.pi)) * (th_Z * (Phip + Lt)))

            dV = dtheta * dzeta
            vol_w = sqrt_g * weights_3d * dV
            
            term1 = GR * (-RHO * a * jnp.sin(thR)) * vol_w
            term2 = GZ * (k * RHO * a * jnp.cos(thZ)) * vol_w
            term3 = GR * vol_w
            term4 = GZ * vol_w
            term5 = GZ * (RHO * a * jnp.sin(thZ)) * vol_w
            term6 = (GR * RHO * jnp.cos(thR) + GZ * k * RHO * jnp.sin(thZ)) * vol_w
            
            basis_1d_tz = basis_1d_val[:, 0, 0, :]
            
            def integ_1d(term):
                term_t = jnp.sum(term, axis=1) # (Nr, Nz)
                term_tz = jnp.dot(term_t, basis_1d_tz.T) # (Nr, len_1d)
                return jnp.dot(fac_rad, term_tz) # (L_rad, len_1d)
                
            res1 = integ_1d(term1)
            res2 = integ_1d(term2)
            res3 = integ_1d(term3)
            res4 = integ_1d(term4)
            res5 = integ_1d(term5)
            res6 = integ_1d(term6)
            
            geom_res_list = [res1, res2, res3, res4, res5, res6]
            
            if len_2d > 0:
                term7 = GR * (-RHO * a * jnp.sin(thR)) * vol_w
                term8 = GZ * (k * RHO * a * jnp.cos(thZ)) * vol_w
                
                # [内存/速度优化点]: 同样用 einsum 替换原先巨大的 5维 张量广播
                def integ_2d(term):
                    # term: (Nr, Nt, Nz), basis_2d_val: (len_2d, Nr, Nt, Nz)
                    term_r = jnp.einsum('rtz,mrtz->mr', term, basis_2d_val) # (len_2d, Nr)
                    return jnp.dot(fac_rad, term_r.T) # (L_rad, len_2d)
                    
                geom_res_list.append(integ_2d(term7))
                geom_res_list.append(integ_2d(term8))
                
            res_geom_concat = jnp.concatenate(geom_res_list, axis=1)
            final_res_list = [res_geom_concat.flatten()]
            
            if len_lam > 0:
                term_lam = Jr_phys * vol_w
                basis_lam_tz = basis_lam_val[:, 0, :, :]
                term_lam_tz = jnp.tensordot(term_lam, basis_lam_tz, axes=([1, 2], [1, 2])) # (Nr, len_lam)
                
                # [向量化更新]: 使用更新后且包含 rho^m 缩放规律的基底执行内积残差投影
                fac_lam = (1.0 - rho_1d**2) * T
                res_lam = jnp.dot(fac_lam, rho_m_lam.T * term_lam_tz)
                final_res_list.append(res_lam.flatten())
                
            final_res = jnp.concatenate(final_res_list)

            if apply_scaling:
                final_res = final_res / res_scales
                
            penalty = jnp.sum(jnp.where(det_phys < 1e-5, 100.0 * (1e-5 - det_phys)**2, 0.0))
            final_res = final_res * (1.0 + penalty)  
                
            return final_res
            
        return jax_res_fn

    def _run_optimization(self, x0, max_nfev, ftol):
        jax_res_fn = self._build_jax_residual_fn()
        
        @jax.jit
        def res_compiled(x):
            return jax_res_fn(x, apply_scaling=True)
            
        @jax.jit
        def jac_compiled(x):
            return jax.jacfwd(lambda x_: jax_res_fn(x_, apply_scaling=True))(x)
            
        _ = res_compiled(jnp.array(x0))
        _ = jac_compiled(jnp.array(x0))
        
        def fun_wrapped(x):
            return np.array(res_compiled(jnp.array(x)))
            
        def jac_wrapped(x):
            return np.array(jac_compiled(jnp.array(x)))

        start_time = time.time()
        res = least_squares(
            fun_wrapped, 
            x0, 
            jac=jac_wrapped, 
            method='trf', 
            xtol=ftol, 
            ftol=ftol, 
            max_nfev=max_nfev
        )
        end_time = time.time()
        
        print(f"    当前网格求解耗时: {end_time - start_time:.4f} 秒")
        print(f"    函数评估次数: {res.nfev} 次")
        print(f"    最终残差范数: {np.linalg.norm(res.fun):.4e}")
        return res

    def solve(self):
        print(">>> 启动 VEQ-3D 谱精度平衡求解器 (极简纯净三级火箭版)...")
        
        def make_even(x): return x + (x % 2)
        c_Nr = make_even(max(8, 4 * self.L_rad + 2))
        c_Nt = make_even(max(12, 4 * self.M_pol + 4))
        c_Nz = make_even(max(8, 4 * self.N_tor + 2))

        m_Nr = make_even(c_Nr + 6)
        m_Nt = make_even(c_Nt + 8)
        m_Nz = make_even(c_Nz + 4)

        print("\n" + "="*70)
        print(f">>> [Phase 1/3]: 极粗网格冷启动寻向 (Nr={c_Nr}, Nt={c_Nt}, Nz={c_Nz})")
        print("="*70)
        self.update_grid(Nr=c_Nr, Nt_grid=c_Nt, Nz_grid=c_Nz)
        x_guess = np.zeros(self.num_core_params)
        
        res_very_coarse = self._run_optimization(x_guess, max_nfev=30, ftol=1e-2)

        print("\n" + "="*70)
        print(f">>> [Phase 2/3]: 中等网格形貌精炼 (Nr={m_Nr}, Nt={m_Nt}, Nz={m_Nz})")
        print("="*70)
        self.update_grid(Nr=m_Nr, Nt_grid=m_Nt, Nz_grid=m_Nz)
        
        res_coarse = self._run_optimization(res_very_coarse.x, max_nfev=60, ftol=1e-5)

        print("\n" + "="*70)
        print(f">>> [Phase 3/3]: 高保真网格极限收敛 (Nr={self.target_Nr}, Nt={self.target_Nt}, Nz={self.target_Nz})")
        print("="*70)
        self.update_grid(self.target_Nr, self.target_Nt, self.target_Nz)
        
        res_fine = self._run_optimization(res_coarse.x, max_nfev=1000, ftol=1e-12)

        self.print_final_parameters(res_fine.x)
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
        R = e_R0 + h + rho * a * np.cos(thR)
        Z = e_Z0 + v + k * rho * a * np.sin(thZ)
        
        lam = np.zeros_like(base_grid)
        for i, (m, n) in enumerate(self.lambda_modes):
            # [评估时更新]: 针对特定的 (m, n) 模式应用 rho^m 进行计算展开
            L_fac_rad_m = (rho ** m) * (1 - rho**2) * T
            lam = lam + np.tensordot(c_lam[:, i], L_fac_rad_m, axes=(0, 0)) * np.sin(m * theta - n * zeta)
            
        return R, Z, thR, thZ, a, k, lam

    def print_final_parameters(self, x_core):
        print("\n" + "="*110)
        print(f"{f'VEQ-3D 动态高维参数报告 (M={self.M_pol}, N={self.N_tor}, L={self.L_rad})':^110}")
        print("="*110)
        
        edge_R0, edge_Z0, e_c0R, e_c0Z, e_h, e_v, e_k, e_a, e_tR, e_tZ = self.unpack_edge()
        c_c0R, c_c0Z, c_h, c_v, c_k, c_a, c_tR, c_tZ, c_lam = self.unpack_core(x_core)
        
        print(f"R0 (大半径中心) = {edge_R0:>15.8e}")
        print(f"Z0 (垂直中心)   = {edge_Z0:>15.8e}")
        print("-" * 110)
        
        header_cols = [f"Chebyshev L={L} 演化系数" for L in range(self.L_rad)]
        header_str = f"{'参数标识':<15} | {'Edge 边界常量 (rho=1)':<25} | " + " | ".join([f"{h:<22}" for h in header_cols])
        print(header_str)
        print("-" * 110)
        
        def print_1d(name, e_arr, c_arr):
            h_str = f"{name+'0':<15} | {e_arr[0]:>25.8e} | " + " | ".join([f"{c_arr[L, 0]:>22.8e}" for L in range(self.L_rad)])
            print(h_str)
            idx = 1
            for n in range(1, self.N_tor + 1):
                c_str = f"{name+str(n)+'c':<15} | {e_arr[idx]:>25.8e} | " + " | ".join([f"{c_arr[L, idx]:>22.8e}" for L in range(self.L_rad)])
                print(c_str)
                s_str = f"{name+str(n)+'s':<15} | {e_arr[idx+1]:>25.8e} | " + " | ".join([f"{c_arr[L, idx+1]:>22.8e}" for L in range(self.L_rad)])
                print(s_str)
                idx+=2

        print_1d("c0R_", e_c0R, c_c0R); print_1d("c0Z_", e_c0Z, c_c0Z)
        print_1d("h_", e_h, c_h); print_1d("v_", e_v, c_v)
        print_1d("k_", e_k, c_k); print_1d("a_", e_a, c_a)
        
        if self.len_2d > 0:
            print("-" * 110); print(">>> 极向高阶摄动分量 (theta_R & theta_Z):")
            for i, (m, n, typ) in enumerate(self.modes_2d):
                tR_str = f"tR_{m}_{n}{typ:<10} | {e_tR[i]:>25.8e} | " + " | ".join([f"{c_tR[L, i]:>22.8e}" for L in range(self.L_rad)])
                print(tR_str)
            for i, (m, n, typ) in enumerate(self.modes_2d):
                tZ_str = f"tZ_{m}_{n}{typ:<10} | {e_tZ[i]:>25.8e} | " + " | ".join([f"{c_tZ[L, i]:>22.8e}" for L in range(self.L_rad)])
                print(tZ_str)
                
        print("-" * 110); print(f">>> 磁流函数 (Lambda) 谐波分量 ( 比例因子: rho^m * T_l(x) ):") # 更新了此处的提示文案
        for i, (m, n) in enumerate(self.lambda_modes):
            L_str = f"L_{m}_{n:<12} | {'-- Null --':>25} | " + " | ".join([f"{c_lam[L, i]:>22.8e}" for L in range(self.L_rad)])
            print(L_str)
        print("="*110 + "\n")

    def plot_equilibrium(self, x_core):
        zetas = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
        fig, axes = plt.subplots(2, 3, figsize=(15, 12))
        axes = axes.flatten()
        rp = np.linspace(0, 1, 50); tp = np.linspace(0, 2*np.pi, 100)
        R_P, T_P = np.meshgrid(rp, tp); PSI_P = self.compute_psi(R_P)
        for i, zv in enumerate(zetas):
            ax = axes[i]; Rm, Zm = [], []
            for r, t in zip(R_P.flatten(), T_P.flatten()):
                rg = self.compute_geometry(x_core, r, t, zv)
                Rm.append(rg[0]); Zm.append(rg[1])
            Rm = np.array(Rm).reshape(R_P.shape); Zm = np.array(Zm).reshape(R_P.shape)
            ax.tripcolor(Rm.flatten(), Zm.flatten(), PSI_P.flatten(), shading='gouraud', cmap='magma', alpha=0.9)
            for r_lev in [0.2, 0.4, 0.6, 0.8, 1.0]:
                rl, zl = self.compute_geometry(x_core, r_lev, np.linspace(0, 2*np.pi, 100), zv)[:2]
                ax.plot(rl, zl, color='white', lw=1.0, alpha=0.5)
            
            th_t = np.linspace(0, 2*np.pi, 200)
            ax.plot(10 - np.cos(th_t) - 0.3*np.cos(th_t+zv), np.sin(th_t) - 0.3*np.sin(th_t+zv), 'r--', lw=1.5, label='Input LCFS')
            
            rl_e, zl_e = self.compute_geometry(x_core, 1.0, np.linspace(0, 2*np.pi, 100), zv)[:2]
            ax.plot(rl_e, zl_e, color='#FFD700', lw=2.0, label='Solved Boundary')
            ax.set_aspect('equal'); ax.set_title(f'Toroidal Angle $\zeta={zv:.2f}$'); 
            if i == 0: ax.legend(loc='upper right', fontsize='xx-small')
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    VEQ3D_Solver().solve()
