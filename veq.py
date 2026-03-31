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
        self.L_rad = 3
        # =========================================================
        
        self.Nt = 19
        self.Phi_a = 1.0
        self.mu_0 = 4 * np.pi * 1e-7
        
        # 保存目标高精度网格尺寸
        self.target_Nr = 16
        self.target_Nt = 16
        self.target_Nz = 16
        
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
            
            # 使用全新的公式：rho在边界处为1
            R_mod = R0 + a_v * (h_v + np.cos(thR))
            Z_mod = Z0 + a_v * (v_v - k_v * np.sin(thZ))
            
            res_geom = np.concatenate([(R_mod - R_target).flatten(), (Z_mod - Z_target).flatten()])
            res_reg = np.array([h_c[0], v_c[0]]) * 100.0
            return np.concatenate([res_geom, res_reg])
            
        if self.p_edge is not None and len(self.p_edge) == self.num_edge_params:
            p0 = self.p_edge.copy()
        else:
            p0 = np.zeros(self.num_edge_params)
            p0[0] = 10.0 
            p0[2] = np.pi # c0R 初始偏置 
            p0[2 + self.len_1d] = np.pi # c0Z 初始偏置，用于配平 sin 中增加的负号
            p0[2 + 4 * self.len_1d] = 1.0 # k
            p0[2 + 5 * self.len_1d] = 1.0 # a
        
        res = least_squares(boundary_residuals, p0, method='trf', ftol=1e-12)
        self.p_edge = res.x

    def _build_jax_residual_fn(self, pressure_scale_factor=1.0):
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

        if self.len_lam > 0:
            lam_m_vals = jnp.array([m for m, n in self.lambda_modes])
        else:
            lam_m_vals = jnp.array([])

        # --- 为新增正则化预计算模式惩罚权重 ---
        if self.len_2d > 0:
            m_vals_2d = jnp.array([m for m, n, typ in self.modes_2d])
            n_vals_2d = jnp.array([n for m, n, typ in self.modes_2d])
            # 正则化机制 1 权重：为避免与边界形变发生拉扯冲突，将基础权重下调到 1e-6，
            # 并乘以 (m^2+n^2) 迫使高频模式尽量压缩。
            reg_weight_2d = jnp.sqrt(1e-6 * (m_vals_2d**2 + n_vals_2d**2))
            
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

            def eval_1d(c_e, c_c):
                core_contrib = jnp.dot(c_c.T, fac_rad)  
                ce_eff = c_e[:, None] + core_contrib    
                b_val = basis_1d_val[:, 0, 0, :]        
                b_dz  = basis_1d_dz[:, 0, 0, :]         
                val = jnp.einsum('mr,mz->rz', ce_eff, b_val)[:, None, :] 
                dz  = jnp.einsum('mr,mz->rz', ce_eff, b_dz)[:, None, :]  
                dr_contrib = jnp.dot(c_c.T, dfac_rad)   
                dr  = jnp.einsum('mr,mz->rz', dr_contrib, b_val)[:, None, :]
                return val, dr, dz

            def eval_2d(c_e, c_c):
                if len_2d == 0: 
                    return 0.0, 0.0, 0.0, 0.0
                core_contrib = jnp.dot(c_c.T, fac_rad)  
                ce_eff = c_e[:, None] + core_contrib    
                val = jnp.einsum('mr,mrtz->rtz', ce_eff, basis_2d_val)
                dth = jnp.einsum('mr,mrtz->rtz', ce_eff, basis_2d_dth)
                dz  = jnp.einsum('mr,mrtz->rtz', ce_eff, basis_2d_dze)
                dr_contrib = jnp.dot(c_c.T, dfac_rad)   
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
            
            R = e_R0 + a * (h + RHO * jnp.cos(thR))
            Z = e_Z0 + a * (v - k * RHO * jnp.sin(thZ))
            
            thR_r, thR_th, thR_z = c0Rr + tRr, 1.0 + tRth, c0Rz + tRz
            thZ_r, thZ_th, thZ_z = c0Zr + tZr, 1.0 + tZth, c0Zz + tZz
            
            Rr = ar * (h + RHO * jnp.cos(thR)) + a * (hr + jnp.cos(thR) - RHO * jnp.sin(thR) * thR_r)
            Rt = -a * RHO * jnp.sin(thR) * thR_th
            Rz = az * (h + RHO * jnp.cos(thR)) + a * (hz - RHO * jnp.sin(thR) * thR_z)
            
            Zr = ar * (v - k * RHO * jnp.sin(thZ)) + a * (vr - kr * RHO * jnp.sin(thZ) - k * jnp.sin(thZ) - k * RHO * jnp.cos(thZ) * thZ_r)
            Zt = -a * k * RHO * jnp.cos(thZ) * thZ_th
            Zz = az * (v - k * RHO * jnp.sin(thZ)) + a * (vz - kz * RHO * jnp.sin(thZ) - k * RHO * jnp.cos(thZ) * thZ_z)

            det_phys = Rr * Zt - Rt * Zr
            det_safe = jnp.where(jnp.abs(det_phys) < 1e-13, -1e-13, det_phys)
            
            sqrt_g = (R / Nt) * det_safe
            
            g_rr, g_tt = Rr**2 + Zr**2, Rt**2 + Zt**2
            g_zz = Rz**2 + (R/Nt)**2 + Zz**2 
            g_rt, g_rz, g_tz = Rr*Rt+Zr*Zt, Rr*Rz+Zr*Zz, Rt*Rz+Zt*Zz

            if len_lam > 0:
                rho_m_lam = rho_1d[None, :] ** lam_m_vals[:, None] 
                lam_ce = jnp.dot(c_lam.T, T) * rho_m_lam * (1.0 - rho_1d**2)**2
                
                b_dth = basis_lam_dth[:, 0, :, :]    
                b_dze = basis_lam_dze[:, 0, :, :]    
                Lt = jnp.einsum('mr,mtz->rtz', lam_ce, b_dth)
                Lz = jnp.einsum('mr,mtz->rtz', lam_ce, b_dze)
            else:
                Lt = 0.0
                Lz = 0.0
            
            P_scale = 1.8e4 
            P = pressure_scale_factor * P_scale * (RHO**2 - 1)**2
            dP = pressure_scale_factor * P_scale * 4 * RHO * (RHO**2 - 1)
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
            
            term1 = GR * (-a * RHO * jnp.sin(thR)) * vol_w
            term2 = GZ * (-a * k * RHO * jnp.cos(thZ)) * vol_w
            term3 = GR * a * vol_w
            term4 = GZ * a * vol_w
            term5 = GZ * (-a * RHO * jnp.sin(thZ)) * vol_w
            term6 = (GR * (h + RHO * jnp.cos(thR)) + GZ * (v - k * RHO * jnp.sin(thZ))) * vol_w
            
            basis_1d_tz = basis_1d_val[:, 0, 0, :]
            
            def integ_1d(term):
                term_t = jnp.sum(term, axis=1) 
                term_tz = jnp.dot(term_t, basis_1d_tz.T) 
                return jnp.dot(fac_rad, term_tz) 
                
            res1 = integ_1d(term1)
            res2 = integ_1d(term2)
            res3 = integ_1d(term3)
            res4 = integ_1d(term4)
            res5 = integ_1d(term5)
            res6 = integ_1d(term6)
            
            geom_res_list = [res1, res2, res3, res4, res5, res6]
            
            if len_2d > 0:
                term7 = GR * (-a * RHO * jnp.sin(thR)) * vol_w
                term8 = GZ * (-a * k * RHO * jnp.cos(thZ)) * vol_w
                
                def integ_2d(term):
                    term_r = jnp.einsum('rtz,mrtz->mr', term, basis_2d_val) 
                    return jnp.dot(fac_rad, term_r.T) 
                    
                geom_res_list.append(integ_2d(term7))
                geom_res_list.append(integ_2d(term8))
                
            res_geom_concat = jnp.concatenate(geom_res_list, axis=1)
            final_res_list = [res_geom_concat.flatten()]
            
            if len_lam > 0:
                term_lam = Jr_phys * vol_w
                basis_lam_tz = basis_lam_val[:, 0, :, :]
                term_lam_tz = jnp.tensordot(term_lam, basis_lam_tz, axes=([1, 2], [1, 2])) 
                
                fac_lam = (1.0 - rho_1d**2) * T
                res_lam = jnp.dot(fac_lam, rho_m_lam.T * term_lam_tz)
                final_res_list.append(res_lam.flatten())
                
            # =======================================================
            # [解耦修改 1]：组装基础物理残差并进行物理缩放与屏障惩罚
            # =======================================================
            phys_res = jnp.concatenate(final_res_list)

            if apply_scaling:
                phys_res = phys_res / res_scales
                
            # 防坐标系自交/折叠的屏障惩罚 (必须且仅能应用于物理残差，防止引发优化器作弊)
            penalty = jnp.sum(jnp.where(det_phys < 1e-5, 100.0 * (1e-5 - det_phys)**2, 0.0))
            phys_res = phys_res * (1.0 + penalty)  
                
            # =======================================================
            # [解耦修改 2]：非线性纯净物理正则化约束扩展 (独立拼接)
            # =======================================================
            reg_residuals = []
            
            # [机制 1] 谱压缩正则化：惩罚 tR 和 tZ
            # 迫使优化器在等效物理场中寻找一个高频分量最小的平滑 theta 坐标映射
            if len_2d > 0:
                # 给 t_R 和 t_Z 的每个系数独立乘以 sqrt(权重) (此时权重自身已经是无量纲合理的量级)
                reg_tR = (c_tR * reg_weight_2d[None, :]).flatten()
                reg_tZ = (c_tZ * reg_weight_2d[None, :]).flatten()
                reg_residuals.extend([reg_tR, reg_tZ])
                
            # [机制 2] 流函数 Lambda 绝对最小化
            # 通过 1e-6 的极小二次惩罚防止无意义的 Lambda 分支漂移
            if len_lam > 0:
                reg_lam = (c_lam * jnp.sqrt(1e-6)).flatten()
                reg_residuals.append(reg_lam)
                
            # 将缩放后的物理残差和正则化辅助残差直接平级合并 (构建最终无量纲的超定系统)
            # 注意：正则项不参与 res_scales 的缩放，也不参与坐标壁垒 penalty 的乘法作弊。
            if reg_residuals:
                final_res = jnp.concatenate([phys_res] + reg_residuals)
            else:
                final_res = phys_res

            return final_res
            
        return jax_res_fn

    def _run_optimization(self, x0, max_nfev, ftol, pressure_scale_factor=1.0):
        jax_res_fn = self._build_jax_residual_fn(pressure_scale_factor)
        
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
        
        # 计算纯物理残差（剥离末尾拼接的正则项）
        reg_len = (2 * self.len_2d + self.len_lam) * self.L_rad
        phys_len = len(res.fun) - reg_len
        phys_res = res.fun[:phys_len]
        
        print(f"    当前网格求解耗时: {end_time - start_time:.4f} 秒")
        print(f"    函数评估次数: {res.nfev} 次")
        print(f"    纯物理残差范数 (无正则项): {np.linalg.norm(phys_res):.4e}")
        print(f"    最终合成残差范数 (含正则项): {np.linalg.norm(res.fun):.4e}")
        return res

    def solve(self):
        print(">>> 启动 VEQ-3D 谱精度平衡求解器 (网格-压力双重延拓完美版)...")
        
        # 动态计算不同阶段的安全网格尺寸
        def make_even(x): return x + (x % 2)
        c_Nr = make_even(max(8, 4 * self.L_rad + 2))
        c_Nt = make_even(max(12, 4 * self.M_pol + 4))
        c_Nz = make_even(max(8, 4 * self.N_tor + 2))

        m_Nr = make_even(c_Nr + 6)
        m_Nt = make_even(c_Nt + 8)
        m_Nz = make_even(c_Nz + 4)

        print("\n" + "="*70)
        print(f">>> [Phase 1/3]: 粗网格 & 零压无力矩冷启动 (Nr={c_Nr}, Nt={c_Nt}, Nz={c_Nz}, P=0.0)")
        print("="*70)
        self.update_grid(c_Nr, c_Nt, c_Nz)
        x_guess = np.zeros(self.num_core_params)
        
        # 放开探索步数至 200，在低分辨率下计算极快，不会耽误时间
        res_phase1 = self._run_optimization(x_guess, max_nfev=200, ftol=1e-3, pressure_scale_factor=0.0)

        print("\n" + "="*70)
        print(f">>> [Phase 2/3]: 中网格 & 低压等离子体过渡 (Nr={m_Nr}, Nt={m_Nt}, Nz={m_Nz}, P=0.1)")
        print("="*70)
        self.update_grid(m_Nr, m_Nt, m_Nz)
        
        res_phase2 = self._run_optimization(res_phase1.x, max_nfev=200, ftol=1e-5, pressure_scale_factor=0.1)

        print("\n" + "="*70)
        print(f">>> [Phase 3/3]: 高保真网格 & 目标高压极限收敛 (Nr={self.target_Nr}, Nt={self.target_Nt}, Nz={self.target_Nz}, P=1.0)")
        print("="*70)
        self.update_grid(self.target_Nr, self.target_Nt, self.target_Nz)
        
        res_fine = self._run_optimization(res_phase2.x, max_nfev=1000, ftol=1e-12, pressure_scale_factor=1.0)

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
        
        R = e_R0 + a * (h + rho * np.cos(thR))
        Z = e_Z0 + a * (v - k * rho * np.sin(thZ))
        
        lam = np.zeros_like(base_grid)
        for i, (m, n) in enumerate(self.lambda_modes):
            L_fac_rad_m = (rho ** m) * (1 - rho**2) * T
            lam = lam + np.tensordot(c_lam[:, i], L_fac_rad_m, axes=(0, 0)) * np.sin(m * theta - n * zeta)
            
        return R, Z, thR, thZ, a, k, lam

    def print_final_parameters(self, x_core):
        # 动态计算表格排版所需的总宽度 (基础宽度46 + 每一阶L需要25个字符)
        table_width = max(110, 46 + self.L_rad * 25)
        
        print("\n" + "=" * table_width)
        print(f"{f'VEQ-3D 动态高维参数报告 (M={self.M_pol}, N={self.N_tor}, L={self.L_rad})':^{table_width}}")
        print("=" * table_width)
        
        edge_R0, edge_Z0, e_c0R, e_c0Z, e_h, e_v, e_k, e_a, e_tR, e_tZ = self.unpack_edge()
        c_c0R, c_c0Z, c_h, c_v, c_k, c_a, c_tR, c_tZ, c_lam = self.unpack_core(x_core)
        
        print(f"R0 (大半径中心) = {edge_R0:>15.8e}")
        print(f"Z0 (垂直中心)   = {edge_Z0:>15.8e}")
        print("-" * table_width)
        
        header_cols = [f"Chebyshev L={L} 演化系数" for L in range(self.L_rad)]
        header_str = f"{'参数标识':<15} | {'Edge 边界常量 (rho=1)':<25} | " + " | ".join([f"{h:<22}" for h in header_cols])
        print(header_str)
        print("-" * table_width)
        
        def print_1d(name, e_arr, c_arr):
            h_str = f"{name+'0':<15} | {e_arr[0]:>25.8e} | " + " | ".join([f"{c_arr[L, 0]:>22.8e}" for L in range(self.L_rad)])
            print(h_str)
            idx = 1
            for n in range(1, self.N_tor + 1):
                # 修复了 f-string 嵌套对齐问题
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
                # 修复了变量名长度变化导致的列错位
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
            ax.set_aspect('equal'); ax.set_title(f'Toroidal Angle $\zeta={zv:.2f}$') 
            if i == 0: ax.legend(loc='upper right', fontsize='xx-small')
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    VEQ3D_Solver().solve()
