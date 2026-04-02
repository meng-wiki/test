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
        # [宏观物理目标约束区 (DESC/VMEC 风格软约束)]
        # =========================================================
        self.target_beta = 0.03   # 目标体平均比压 (3%)
        self.target_Ip = 0.5      # 目标无量纲环向总电流 (与物理电流成正比)
        self.Phi_a_phys = 1.0     # 物理真空磁通输入 (Wb)
        self.B0 = 3.0             # 输入真空环向磁场 (T)
        self.mu_0 = 4 * np.pi * 1e-7
        # =========================================================
        
        self.Nt = 19
        
        # 保存目标高精度网格尺寸
        self.target_Nr = 16
        self.target_Nt = 16
        self.target_Nz = 16
        
        self.p_edge = None 
        
        self._setup_modes()
        
        # 初始化基础参数 
        self.update_grid(self.target_Nr, self.target_Nt, self.target_Nz)
        self._initialize_scaling()

    def update_grid(self, Nr, Nt_grid, Nz_grid):
        """动态网格重构引擎"""
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

        self._precompute_radial_factors() 
        self._build_basis_matrices()
        self.fit_boundary()

    def _precompute_radial_factors(self):
        """预计算与网格相关、与参数无关的 Chebyshev 和各种修正因子"""
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
            self.reg_weight_2d = np.sqrt(1e-6 * (m_vals_2d**2 + n_vals_2d**2))
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
        
        # 增加 2 个 DESC 风格宏观物理缩放变量：C_beta (压强幅值) 和 C_I (总极向磁通缩放/Ip)
        self.num_macro_params = 2
        self.num_core_params = self.num_geom_params + self.len_lam * self.L_rad + self.num_macro_params
        
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
        c_c0R, c_c0Z, c_h, c_v, c_k, c_a = get(self.len_1d), get(self.len_1d), get(self.len_1d), get(self.len_1d), get(self.len_1d), get(self.len_1d)
        c_tR, c_tZ = get(self.len_2d), get(self.len_2d)
        c_lam = get(self.len_lam)
        
        # 提取最后两个宏观缩放标量
        C_beta = x_core[idx]
        C_I = x_core[idx+1]
        
        return c_c0R, c_c0Z, c_h, c_v, c_k, c_a, c_tR, c_tZ, c_lam, C_beta, C_I

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
        print(f">>> 完全引入 DESC/VMEC 级别软约束：将物理缩放常数 C_beta, C_I 合并进入自适应求解空间。")
        # 修复：res_scales 的长度应当精确匹配纯物理力学残差的长度，而不应包含最后的 2 个宏观约束参数
        self.res_scales = np.ones(self.num_geom_params + self.len_lam * self.L_rad)

    def fit_boundary(self):
        TH_F, ZE_F = self.TH[0], self.ZE[0]
        R_target_phys = 10 - np.cos(TH_F) - 0.3 * np.cos(TH_F + ZE_F)
        Z_target_phys = np.sin(TH_F) - 0.3 * np.sin(TH_F + ZE_F)
        
        self.R0_phys = float(np.mean(R_target_phys))
        self.R_target = R_target_phys / self.R0_phys
        self.Z_target = Z_target_phys / self.R0_phys
        self.bar_Phi_a = self.Phi_a_phys / (self.B0 * self.R0_phys**2)
        
        def eval_1d_edge(coeffs): return np.sum(coeffs[:, None, None] * self.basis_1d_val[:, 0, ...], axis=0)
        def eval_2d_edge(coeffs): return np.sum(coeffs[:, None, None] * self.basis_2d_edge, axis=0) if self.len_2d > 0 else 0.0

        def boundary_residuals(p):
            R0, Z0 = p[0], p[1]
            idx = 2
            def get(L): nonlocal idx; c = p[idx:idx+L]; idx+=L; return c
            c0R_v, c0Z_v = eval_1d_edge(get(self.len_1d)), eval_1d_edge(get(self.len_1d))
            h_c, v_c = get(self.len_1d), get(self.len_1d)
            h_v, v_v = eval_1d_edge(h_c), eval_1d_edge(v_c)
            k_v, a_v = eval_1d_edge(get(self.len_1d)), eval_1d_edge(get(self.len_1d))
            tR_v, tZ_v = eval_2d_edge(get(self.len_2d)), eval_2d_edge(get(self.len_2d))
            
            thR = TH_F + c0R_v + tR_v
            thZ = TH_F + c0Z_v + tZ_v
            
            R_mod = R0 + a_v * (h_v + np.cos(thR))
            Z_mod = Z0 + a_v * (v_v - k_v * np.sin(thZ))
            
            res_geom = np.concatenate([(R_mod - self.R_target).flatten(), (Z_mod - self.Z_target).flatten()])
            res_reg = np.array([h_c[0], v_c[0]]) * 100.0
            return np.concatenate([res_geom, res_reg])
            
        if self.p_edge is not None and len(self.p_edge) == self.num_edge_params:
            p0 = self.p_edge.copy()
        else:
            p0 = np.zeros(self.num_edge_params)
            p0[0] = 1.0 
            p0[2] = np.pi 
            p0[2 + self.len_1d] = np.pi 
            p0[2 + 4 * self.len_1d] = 1.0 
            p0[2 + 5 * self.len_1d] = 1.0 / self.R0_phys 
        
        res = least_squares(boundary_residuals, p0, method='trf', ftol=1e-12)
        self.p_edge = res.x

    def _build_jax_residual_fn(self, pressure_scale_factor=1.0):
        # 抓取常量节点
        RHO, TH, ZE = jnp.array(self.RHO), jnp.array(self.TH), jnp.array(self.ZE)
        rho_1d, D_matrix = jnp.array(self.rho), jnp.array(self.D_matrix)
        
        fac_rad, dfac_rad = jnp.array(self.fac_rad), jnp.array(self.dfac_rad)
        fac_lam_eval, fac_lam_proj = jnp.array(self.fac_lam_eval), jnp.array(self.fac_lam_proj)
        rho_m_lam, reg_weight_2d = jnp.array(self.rho_m_lam), jnp.array(self.reg_weight_2d)

        basis_1d_val_slice, basis_1d_dz_slice = jnp.array(self.basis_1d_val[:, 0, 0, :]), jnp.array(self.basis_1d_dz[:, 0, 0, :])
        basis_2d_val, basis_2d_dr = jnp.array(self.basis_2d_val), jnp.array(self.basis_2d_dr)
        basis_2d_dth, basis_2d_dze = jnp.array(self.basis_2d_dth), jnp.array(self.basis_2d_dze)
        
        basis_lam_tz = jnp.array(self.basis_lam_val[:, 0, :, :])
        basis_lam_dth = jnp.array(self.basis_lam_dth[:, 0, :, :])
        basis_lam_dze = jnp.array(self.basis_lam_dze[:, 0, :, :])
        
        k_th, k_ze = jnp.array(self.k_th), jnp.array(self.k_ze)
        weights_3d, res_scales = jnp.array(self.weights_3d), jnp.array(self.res_scales)
        p_edge = jnp.array(self.p_edge)
        
        bar_Phi_a = jnp.array(self.bar_Phi_a)
        
        # 将用户物理目标暴露入计算图
        target_beta_host = jnp.array(self.target_beta)
        target_Ip_host = jnp.array(self.target_Ip)
        
        L_rad, len_1d, len_2d, len_lam = self.L_rad, self.len_1d, self.len_2d, self.len_lam
        Nt, dtheta, dzeta = self.Nt, self.dtheta, self.dzeta
            
        def jax_unpack_edge(p):
            idx = 2
            def get(L): nonlocal idx; c = p[idx:idx+L]; idx += L; return c
            return p[0], p[1], get(len_1d), get(len_1d), get(len_1d), get(len_1d), get(len_1d), get(len_1d), get(len_2d), get(len_2d)

        def jax_unpack_core(x_core):
            idx = 0
            def get(L): nonlocal idx; c = x_core[idx:idx+L*L_rad].reshape((L_rad, L)); idx += L * L_rad; return c
            c_c0R, c_c0Z, c_h, c_v, c_k, c_a = get(len_1d), get(len_1d), get(len_1d), get(len_1d), get(len_1d), get(len_1d)
            c_tR, c_tZ = get(len_2d), get(len_2d)
            c_lam = get(len_lam)
            C_beta = x_core[idx]
            C_I = x_core[idx+1]
            return c_c0R, c_c0Z, c_h, c_v, c_k, c_a, c_tR, c_tZ, c_lam, C_beta, C_I

        def spectral_grad_th(f): return jnp.real(jnp.fft.ifft(1j * k_th * jnp.fft.fft(f, axis=1), axis=1))
        def spectral_grad_ze(f): return jnp.real(jnp.fft.ifft(1j * k_ze * jnp.fft.fft(f, axis=2), axis=2))

        def jax_res_fn(x_core, apply_scaling=True):
            e_R0, e_Z0, e_c0R, e_c0Z, e_h, e_v, e_k, e_a, e_tR, e_tZ = jax_unpack_edge(p_edge)
            c_c0R, c_c0Z, c_h, c_v, c_k, c_a, c_tR, c_tZ, c_lam, C_beta, C_I = jax_unpack_core(x_core)

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
            
            thR, thZ = TH + c0R + tR, TH + c0Z + tZ
            
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
                lam_ce = jnp.dot(c_lam.T, fac_lam_eval) * rho_m_lam
                Lt = jnp.einsum('mr,mtz->rtz', lam_ce, basis_lam_dth)
                Lz = jnp.einsum('mr,mtz->rtz', lam_ce, basis_lam_dze)
            else:
                Lt, Lz = 0.0, 0.0
            
            # --- DESC 式剖面缩放重构机制 ---
            # 压强由优化器控制的幅值系数 C_beta 完全接管
            P_nd = C_beta * (RHO**2 - 1)**2 * pressure_scale_factor
            dP_nd = C_beta * 4 * RHO * (RHO**2 - 1) * pressure_scale_factor
            
            bar_Phip = 2 * RHO * bar_Phi_a
            
            # 电流扭曲由优化器控制的极向通量放大器 C_I 完全接管
            iota = C_I * (1.0 + 1.5 * RHO**2)
            psip = iota * bar_Phip

            Bt_sup = (psip - Lz) / (2 * jnp.pi * sqrt_g)
            Bz_sup = (bar_Phip + Lt) / (2 * jnp.pi * sqrt_g)

            Br_sub = g_rt * Bt_sup + g_rz * Bz_sup
            Bt_sub = g_tt * Bt_sup + g_tz * Bz_sup
            Bz_sub = g_tz * Bt_sup + g_zz * Bz_sup

            dBt_drho = jnp.tensordot(D_matrix, Bt_sub, axes=(1, 0))
            dBz_drho = jnp.tensordot(D_matrix, Bz_sub, axes=(1, 0))

            bar_Jz_sup = (dBt_drho - spectral_grad_th(Br_sub)) / sqrt_g
            bar_Jt_sup = (spectral_grad_ze(Br_sub) - dBz_drho) / sqrt_g
            bar_Jr_sup = (spectral_grad_th(Bz_sub) - spectral_grad_ze(Bt_sub)) / sqrt_g

            G_rho = dP_nd - sqrt_g * (bar_Jt_sup * Bz_sup - bar_Jz_sup * Bt_sup)
            rho_R, rho_Z = Zt/det_safe, -Rt/det_safe
            th_R, th_Z = -Zr/det_safe, Rr/det_safe
            
            GR = (rho_R * G_rho + (bar_Jr_sup / (2 * jnp.pi)) * (th_R * (bar_Phip + Lt)))
            GZ = (rho_Z * G_rho + (bar_Jr_sup / (2 * jnp.pi)) * (th_Z * (bar_Phip + Lt)))

            dV = dtheta * dzeta
            vol_w = sqrt_g * weights_3d * dV
            
            term1 = GR * (-a * RHO * jnp.sin(thR)) * vol_w
            term2 = GZ * (-a * k * RHO * jnp.cos(thZ)) * vol_w
            term3 = GR * a * vol_w
            term4 = GZ * a * vol_w
            term5 = GZ * (-a * RHO * jnp.sin(thZ)) * vol_w
            term6 = (GR * (h + RHO * jnp.cos(thR)) + GZ * (v - k * RHO * jnp.sin(thZ))) * vol_w
            
            terms_1d = jnp.stack([term1, term2, term3, term4, term5, term6], axis=0) 
            terms_1d_t = jnp.sum(terms_1d, axis=2) 
            terms_1d_tz = jnp.einsum('vrz,mz->vrm', terms_1d_t, basis_1d_val_slice) 
            res_1d = jnp.einsum('lr,vrm->lvm', fac_rad, terms_1d_tz).reshape((L_rad, 6 * len_1d)) 
            
            if len_2d > 0:
                term7 = GR * (-a * RHO * jnp.sin(thR)) * vol_w
                term8 = GZ * (-a * k * RHO * jnp.cos(thZ)) * vol_w
                terms_2d = jnp.stack([term7, term8], axis=0)
                terms_2d_r = jnp.einsum('vrtz,mrtz->vmr', terms_2d, basis_2d_val)
                res_2d = jnp.einsum('lr,vmr->lvm', fac_rad, terms_2d_r).reshape((L_rad, 2 * len_2d))
                res_geom = jnp.concatenate([res_1d, res_2d], axis=1).flatten()
            else:
                res_geom = res_1d.flatten()
                
            res_list = [res_geom]
            
            if len_lam > 0:
                term_lam = bar_Jr_sup * vol_w
                term_lam_tz = jnp.einsum('rtz,mtz->rm', term_lam, basis_lam_tz) 
                res_lam = jnp.dot(fac_lam_proj, rho_m_lam.T * term_lam_tz)
                res_list.append(res_lam.flatten())
                
            phys_res = jnp.concatenate(res_list)

            if apply_scaling:
                phys_res = phys_res / res_scales
                
            penalty = jnp.sum(jnp.where(det_phys < 1e-6, 100.0 * (1e-6 - det_phys)**2, 0.0))
            phys_res = phys_res * (1.0 + penalty)  
                
            final_res_list = [phys_res]
            
            # --- 网格平滑正则化 ---
            w_jac = 1e-3
            mean_sqrt_g = jnp.mean(sqrt_g, axis=(1, 2), keepdims=True)
            jac_smooth_res = (sqrt_g - mean_sqrt_g).flatten() * w_jac
            final_res_list.append(jac_smooth_res)
            
            # --- DESC 宏观优化目标追加区 (Beta 与 Ip 约束) ---
            vol_total = jnp.sum(vol_w)
            mean_P_nd = jnp.sum(P_nd * vol_w) / vol_total
            current_beta = 2.0 * mean_P_nd
            
            # 积分获取三维空间的真正总环向电流
            current_Ip = jnp.sum(bar_Jt_sup * vol_w) / (2.0 * jnp.pi)
            
            # 防止在压力延拓初期 (冷启动) 除以0，并使得目标 beta 随延拓因子同步过渡
            eff_target_beta = target_beta_host * pressure_scale_factor
            res_beta = jnp.where(
                eff_target_beta > 1e-7,
                (current_beta - eff_target_beta) / eff_target_beta,
                C_beta - 0.05  # 当目标无压力时，只要求 C_beta 固定在极小预设值
            )
            res_Ip = (current_Ip - target_Ip_host) / target_Ip_host
            
            weight_macro = 100.0  # 给予极高的惩罚权重，强制符合电流和 Beta 目标
            final_res_list.append(jnp.array([res_beta * weight_macro, res_Ip * weight_macro]))
            
            # --- 其它基础高阶正则化 ---
            if len_2d > 0:
                final_res_list.append((c_tR * reg_weight_2d[None, :]).flatten() * 1e-1)
                final_res_list.append((c_tZ * reg_weight_2d[None, :]).flatten() * 1e-1)
                
            if len_lam > 0:
                final_res_list.append((c_lam * jnp.sqrt(1e-6)).flatten() * 1e-1)
                
            return jnp.concatenate(final_res_list)
            
        return jax_res_fn

    def _run_optimization(self, x0, max_nfev, ftol, pressure_scale_factor=1.0):
        jax_res_fn = self._build_jax_residual_fn(pressure_scale_factor)
        
        @jax.jit
        def res_compiled(x): return jax_res_fn(x, apply_scaling=True)
            
        @jax.jit
        def jac_compiled(x): return jax.jacfwd(lambda x_: jax_res_fn(x_, apply_scaling=True))(x)
            
        _ = res_compiled(jnp.array(x0))
        _ = jac_compiled(jnp.array(x0))
        
        def fun_wrapped(x): return np.array(res_compiled(jnp.array(x)))
        def jac_wrapped(x): return np.array(jac_compiled(jnp.array(x)))

        start_time = time.time()
        res = least_squares(
            fun_wrapped, x0, jac=jac_wrapped, method='trf', 
            xtol=ftol, ftol=ftol, max_nfev=max_nfev
        )
        end_time = time.time()
        
        # 剥离所有正则与惩罚项进行报告
        jac_reg_len = self.Nr * self.Nt_grid * self.Nz_grid
        geom_lam_reg_len = (2 * self.len_2d + self.len_lam) * self.L_rad
        macro_reg_len = 2 # Beta 和 Ip 约束
        reg_len = geom_lam_reg_len + jac_reg_len + macro_reg_len
        phys_len = len(res.fun) - reg_len
        phys_res = res.fun[:phys_len]
        
        print(f"    当前网格求解耗时: {end_time - start_time:.4f} 秒")
        print(f"    函数评估次数: {res.nfev} 次")
        print(f"    纯物理力学残差: {np.linalg.norm(phys_res):.4e}")
        return res

    def solve(self):
        print(">>> 启动 VEQ-3D 谱精度平衡求解器 (整合DESC动态宏观逼近约束)...")
        
        def make_even(x): return x + (x % 2)
        c_Nr = make_even(max(8, 4 * self.L_rad + 2))
        c_Nt = make_even(max(12, 4 * self.M_pol + 4))
        c_Nz = make_even(max(8, 4 * self.N_tor + 2))

        m_Nr = make_even(c_Nr + 6)
        m_Nt = make_even(c_Nt + 8)
        m_Nz = make_even(c_Nz + 4)

        self.update_grid(c_Nr, c_Nt, c_Nz)
        
        # 将最新的 C_beta 和 C_I 追加于求解猜想序列末尾
        x_guess = np.zeros(self.num_core_params)
        x_guess[-2] = 0.05  # 初始 C_beta (压强系数猜想)
        x_guess[-1] = 1.0   # 初始 C_I (通量/极向磁场系数猜想)

        print("\n" + "="*70)
        print(f">>> [Phase 1/3]: 粗网格 & 零压无力矩冷启动 (Nr={c_Nr}, Nt={c_Nt}, Nz={c_Nz}, P=0.0)")
        print("="*70)
        res_phase1 = self._run_optimization(x_guess, max_nfev=200, ftol=1e-4, pressure_scale_factor=0.0)

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
        c_c0R, c_c0Z, c_h, c_v, c_k, c_a, c_tR, c_tZ, c_lam, C_beta, C_I = self.unpack_core(x_core) # 解包包括多出来的系数
        
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
        table_width = max(115, 46 + self.L_rad * 25)
        
        print("\n" + "=" * table_width)
        print(f"{f'VEQ-3D 宏观与几何联合参数报告 (物理重构输出)':^{table_width}}")
        print("=" * table_width)
        
        edge_R0, edge_Z0, e_c0R, e_c0Z, e_h, e_v, e_k, e_a, e_tR, e_tZ = self.unpack_edge()
        c_c0R, c_c0Z, c_h, c_v, c_k, c_a, c_tR, c_tZ, c_lam, C_beta, C_I = self.unpack_core(x_core)
        
        # 物理量推导
        L_scale = self.R0_phys
        F_scale = self.B0 * (self.R0_phys ** 2)
        Ip_phys_MA = self.target_Ip * (self.B0 * self.R0_phys) / self.mu_0 / 1e6
        
        print(f"提取参考大半径 R0_phys = {self.R0_phys:>10.5f} m")
        print(f"输入真空环向磁场 B0    = {self.B0:>10.5f} T")
        print(f"中心 R 坐标 [R_0]      = {edge_R0 * L_scale:>15.8e} [m]")
        print(f"中心 Z 坐标 [Z_0]      = {edge_Z0 * L_scale:>15.8e} [m]")
        print(f"动态演化压强系数 C_b   = {C_beta:>15.8e} [达到目标 Beta={self.target_beta*100:.2f}%]")
        print(f"动态极向通量放大器 C_I = {C_I:>15.8e} [推算达成总电流 Ip={Ip_phys_MA:.3f} MA]")
        print("-" * table_width)
        
        header_cols = [f"Chebyshev L={L} 演化" for L in range(self.L_rad)]
        header_str = f"{'参数标识':<15} | {'Edge 边界常量 (rho=1)':<25} | " + " | ".join([f"{h:<22}" for h in header_cols])
        print(header_str)
        print("-" * table_width)
        
        def print_1d(name, e_arr, c_arr, mult=1.0):
            h_str = f"{name+'0':<15} | {e_arr[0]*mult:>25.8e} | " + " | ".join([f"{c_arr[L, 0]*mult:>22.8e}" for L in range(self.L_rad)])
            print(h_str)
            idx = 1
            for n in range(1, self.N_tor + 1):
                c_str = f"{f'{name}{n}c':<15} | {e_arr[idx]*mult:>25.8e} | " + " | ".join([f"{c_arr[L, idx]*mult:>22.8e}" for L in range(self.L_rad)])
                print(c_str)
                s_str = f"{f'{name}{n}s':<15} | {e_arr[idx+1]*mult:>25.8e} | " + " | ".join([f"{c_arr[L, idx+1]*mult:>22.8e}" for L in range(self.L_rad)])
                print(s_str)
                idx+=2

        print_1d("c0R_ (angle)", e_c0R, c_c0R); print_1d("c0Z_ (angle)", e_c0Z, c_c0Z)
        print_1d("h_   (nd-len)", e_h, c_h);     print_1d("v_   (nd-len)", e_v, c_v)
        print_1d("k_   (nd-shp)", e_k, c_k);     print_1d("a_   [meter]", e_a, c_a, mult=L_scale)
        
        if self.len_2d > 0:
            print("-" * table_width)
            print(">>> 极向高阶摄动分量 (theta_R & theta_Z) [无量纲角度]:")
            for i, (m, n, typ) in enumerate(self.modes_2d):
                tR_str = f"{f'tR_{m}_{n}{typ}':<15} | {e_tR[i]:>25.8e} | " + " | ".join([f"{c_tR[L, i]:>22.8e}" for L in range(self.L_rad)])
                print(tR_str)
            for i, (m, n, typ) in enumerate(self.modes_2d):
                tZ_str = f"{f'tZ_{m}_{n}{typ}':<15} | {e_tZ[i]:>25.8e} | " + " | ".join([f"{c_tZ[L, i]:>22.8e}" for L in range(self.L_rad)])
                print(tZ_str)
                
        if self.len_lam > 0:
            print("-" * table_width)
            print(">>> 磁流函数 (Lambda) 谐波分量 [还原为物理磁通量纲 Wb]:") 
            for i, (m, n) in enumerate(self.lambda_modes):
                L_str = f"{f'L_{m}_{n}':<15} | {'-- Null --':>25} | " + " | ".join([f"{c_lam[L, i] * F_scale:>22.8e}" for L in range(self.L_rad)])
                print(L_str)
        print("=" * table_width + "\n")

    def plot_equilibrium(self, x_core):
        zetas = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
        fig, axes = plt.subplots(2, 3, figsize=(15, 12))
        axes = axes.flatten()
        rp = np.linspace(0, 1, 50); tp = np.linspace(0, 2*np.pi, 100)
        R_P, T_P = np.meshgrid(rp, tp)
        PSI_P = self.compute_psi(R_P) * (self.B0 * self.R0_phys**2)
        
        for i, zv in enumerate(zetas):
            ax = axes[i]; Rm, Zm = [], []
            for r, t in zip(R_P.flatten(), T_P.flatten()):
                rg = self.compute_geometry(x_core, r, t, zv)
                Rm.append(rg[0] * self.R0_phys); Zm.append(rg[1] * self.R0_phys)
            Rm = np.array(Rm).reshape(R_P.shape); Zm = np.array(Zm).reshape(R_P.shape)
            ax.tripcolor(Rm.flatten(), Zm.flatten(), PSI_P.flatten(), shading='gouraud', cmap='magma', alpha=0.9)
            
            for r_lev in [0.2, 0.4, 0.6, 0.8, 1.0]:
                rl, zl = self.compute_geometry(x_core, r_lev, np.linspace(0, 2*np.pi, 100), zv)[:2]
                ax.plot(rl * self.R0_phys, zl * self.R0_phys, color='white', lw=1.0, alpha=0.5)
            
            th_t = np.linspace(0, 2*np.pi, 200)
            ax.plot(10 - np.cos(th_t) - 0.3*np.cos(th_t+zv), np.sin(th_t) - 0.3*np.sin(th_t+zv), 'r--', lw=1.5, label='Input LCFS')
            
            rl_e, zl_e = self.compute_geometry(x_core, 1.0, np.linspace(0, 2*np.pi, 100), zv)[:2]
            ax.plot(rl_e * self.R0_phys, zl_e * self.R0_phys, color='#FFD700', lw=2.0, label='Solved Boundary')
            ax.set_aspect('equal'); ax.set_title(f'Toroidal Angle $\zeta={zv:.2f}$') 
            if i == 0: ax.legend(loc='upper right', fontsize='xx-small')
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    VEQ3D_Solver().solve()
