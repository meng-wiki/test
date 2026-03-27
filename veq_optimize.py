import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse.linalg import gmres, LinearOperator
import time

# 启用JAX的64位浮点数以保证MHD平衡计算的精度
jax.config.update("jax_enable_x64", True)

class Veq3D:
    def __init__(self, config=None):
        if config is None:
            config = {}
        
        # --- 1. 严格锁定为用户指定的物理剖面与标量 ---
        self.NFP = config.get('NFP', 19)
        self.Phi_edge = config.get('Phi_edge', 1.0)
        
        # P = 18000 * (rho^2 - 1)^2 = 18000 - 36000*rho^2 + 18000*rho^4
        self.p_coeffs = jnp.array([18000.0, 0.0, -36000.0, 0.0, 18000.0])
        
        # iota = 1 + 1.5*rho^2
        self.iota_coeffs = jnp.array([1.0, 0.0, 1.5])
        
        self.mu_0 = config.get('mu_0', 4.0 * jnp.pi * 1e-7)
        
        # --- 2. 谱展开截断阶数 ---
        self.L_rad = config.get('L_rad', 4)  
        self.M_pol = config.get('M_pol', 3)  
        self.N_tor = config.get('N_tor', 2)  

        self.L_rad_lam = config.get('L_rad_lam', 2)
        self.M_pol_lam = config.get('M_pol_lam', 2)
        self.N_tor_lam = config.get('N_tor_lam', 1)
        
        # --- 3. 物理配点网格数 (环向仅需解析单个 NFP 周期) ---
        self.N_rad = config.get('N_rad', 12)
        self.N_pol = config.get('N_pol', 12)
        self.N_tor_grid = config.get('N_tor_grid', 6)
        
        self.shape_rz = (self.L_rad, 2 * self.M_pol + 1, 2 * self.N_tor + 1)
        self.shape_lam = (self.L_rad_lam, 2 * self.M_pol_lam + 1, 2 * self.N_tor_lam + 1)
        
        self.size_rz = np.prod(self.shape_rz)
        self.size_lam = np.prod(self.shape_lam)
        self.total_params = 2 * self.size_rz + self.size_lam
        
        print("--- VEQ-3D 初始化 (自动微分能量变分终极版) ---")
        print(f"R, Z   展开模式: {self.shape_rz} -> 自由度: {self.size_rz}")
        print(f"Lambda 降阶模式: {self.shape_lam} -> 自由度: {self.size_lam}")
        print(f"总优化参数维度 : {self.total_params}")
        print(f"NFP (场周期)   : {self.NFP}")
        
        self._build_grid()
        self._build_basis()
        self._init_guess_and_base_fields()
        
        print("正在 JIT 编译目标梯度(残差)函数...")
        t0 = time.time()
        self.res_fn_jit = jax.jit(self._build_jax_residual_fn())
        _ = self.res_fn_jit(self.params_1d)
        print(f"编译完成，耗时: {time.time() - t0:.2f} 秒\n")

    def _build_grid(self):
        i = np.arange(self.N_rad)
        x_cgl = np.cos(np.pi * i / (self.N_rad - 1))
        self.rho_1d = 0.5 * (1.0 - x_cgl)
        self.theta_1d = np.linspace(0, 2 * np.pi, self.N_pol, endpoint=False)
        
        if self.N_tor_grid <= 1:
            self.zeta_1d = np.array([0.0])
        else:
            self.zeta_1d = np.linspace(0, 2 * np.pi / self.NFP, self.N_tor_grid, endpoint=False)
            
        self.rho_3d, self.theta_3d, self.zeta_3d = np.meshgrid(
            self.rho_1d, self.theta_1d, self.zeta_1d, indexing='ij'
        )

    def _build_cheb_matrix(self, x, L):
        T = np.zeros((len(x), L))
        dT = np.zeros((len(x), L))
        
        T[:, 0] = 1.0
        if L > 1:
            T[:, 1] = x
            dT[:, 1] = 1.0
            
        for n in range(2, L):
            T[:, n] = 2 * x * T[:, n-1] - T[:, n-2]
            dT[:, n] = 2 * T[:, n-1] + 2 * x * dT[:, n-1] - dT[:, n-2]
            
        return T, dT

    def _build_fourier_matrix(self, angle, M, is_toroidal=False):
        num_modes = 2 * M + 1
        F = np.zeros((len(angle), num_modes))
        dF = np.zeros((len(angle), num_modes))
        
        F[:, 0] = 1.0
        for m in range(1, M + 1):
            cos_idx = 2 * m - 1
            sin_idx = 2 * m
            k = m * self.NFP if is_toroidal else m
            
            F[:, cos_idx] = np.cos(k * angle)
            F[:, sin_idx] = np.sin(k * angle)
            dF[:, cos_idx] = -k * np.sin(k * angle)
            dF[:, sin_idx] =  k * np.cos(k * angle)
            
        return F, dF

    def _build_basis(self):
        x_mapped = 2.0 * self.rho_1d**2 - 1.0
        rho_col = self.rho_1d[:, None]
        
        max_L = max(self.L_rad, self.L_rad_lam)
        max_M = max(self.M_pol, self.M_pol_lam)
        max_N = max(self.N_tor, self.N_tor_lam)
        
        T, dT_dx = self._build_cheb_matrix(x_mapped, max_L)
        
        # --- Lambda 基底 (标准切比雪夫) ---
        dT_drho_lam = dT_dx * 4.0 * rho_col
        master_T_lam = jnp.array([T, dT_drho_lam])
        
        # --- R 和 Z 扰动基底 (自带 (1-rho) 衰减项，严格锁死边界) ---
        T_RZ = (1.0 - rho_col) * T
        dT_drho_RZ = -T + (1.0 - rho_col) * dT_dx * 4.0 * rho_col
        master_T_RZ = jnp.array([T_RZ, dT_drho_RZ])
        
        F, dF_dtheta = self._build_fourier_matrix(self.theta_1d, max_M, is_toroidal=False)
        H, dH_dzeta = self._build_fourier_matrix(self.zeta_1d, max_N, is_toroidal=True)
        
        master_F = jnp.array([F, dF_dtheta])
        master_H = jnp.array([H, dH_dzeta])
        
        self.d_T_mat = jnp.array([m[:, :self.L_rad] for m in master_T_RZ])
        self.d_F_mat = jnp.array([m[:, :2*self.M_pol+1] for m in master_F])
        self.d_H_mat = jnp.array([m[:, :2*self.N_tor+1] for m in master_H])
        
        self.d_T_mat_lam = jnp.array([m[:, :self.L_rad_lam] for m in master_T_lam])
        self.d_F_mat_lam = jnp.array([m[:, :2*self.M_pol_lam+1] for m in master_F])
        self.d_H_mat_lam = jnp.array([m[:, :2*self.N_tor_lam+1] for m in master_H])

    def _init_guess_and_base_fields(self):
        rho = self.rho_3d
        theta = self.theta_3d
        zeta = self.zeta_3d
        z_nfp = self.NFP * zeta
        
        # 固定的解析基底场 (对应公式中的 a_00 且满足原点连续性)
        self.R_base = 10.0 - rho * np.cos(theta) - 0.3 * rho * np.cos(theta + z_nfp)
        self.R_rho_base = -np.cos(theta) - 0.3 * np.cos(theta + z_nfp)
        self.R_theta_base = rho * np.sin(theta) + 0.3 * rho * np.sin(theta + z_nfp)
        self.R_zeta_base = 0.3 * rho * np.sin(theta + z_nfp) * self.NFP

        self.Z_base = rho * np.sin(theta) - 0.3 * rho * np.sin(theta + z_nfp)
        self.Z_rho_base = np.sin(theta) - 0.3 * np.sin(theta + z_nfp)
        self.Z_theta_base = rho * np.cos(theta) - 0.3 * rho * np.cos(theta + z_nfp)
        self.Z_zeta_base = -0.3 * rho * np.cos(theta + z_nfp) * self.NFP

        R_c = np.zeros(self.shape_rz)
        Z_c = np.zeros(self.shape_rz)
        Lam_c = np.zeros(self.shape_lam)
                
        self.params_1d = np.concatenate([R_c.flatten(), Z_c.flatten(), Lam_c.flatten()])

    def _build_jax_residual_fn(self):
        d_T_mat = self.d_T_mat; d_F_mat = self.d_F_mat; d_H_mat = self.d_H_mat
        d_T_mat_lam = self.d_T_mat_lam; d_F_mat_lam = self.d_F_mat_lam; d_H_mat_lam = self.d_H_mat_lam
        
        size_rz = self.size_rz; shape_rz = self.shape_rz; shape_lam = self.shape_lam
        rho_3d = jnp.array(self.rho_3d)
        
        R_base = jnp.array(self.R_base); R_rho_base = jnp.array(self.R_rho_base)
        R_theta_base = jnp.array(self.R_theta_base); R_zeta_base = jnp.array(self.R_zeta_base)
        Z_base = jnp.array(self.Z_base); Z_rho_base = jnp.array(self.Z_rho_base)
        Z_theta_base = jnp.array(self.Z_theta_base); Z_zeta_base = jnp.array(self.Z_zeta_base)

        def eval_poly(coeffs, x):
            res = jnp.zeros_like(x)
            for i in range(len(coeffs)):
                res += coeffs[i] * (x ** i)
            return res

        def unpack_params(params_1d):
            R_c = params_1d[:size_rz].reshape(shape_rz)
            Z_c = params_1d[size_rz:2*size_rz].reshape(shape_rz)
            Lam_c = params_1d[2*size_rz:].reshape(shape_lam)
            return R_c, Z_c, Lam_c

        def eval_scalar(coeffs, d_rad=0, d_pol=0, d_tor=0, is_lam=False):
            T = d_T_mat_lam[d_rad] if is_lam else d_T_mat[d_rad]
            F = d_F_mat_lam[d_pol] if is_lam else d_F_mat[d_pol]
            H = d_H_mat_lam[d_tor] if is_lam else d_H_mat[d_tor]
            return jnp.einsum('il, jm, kn, lmn -> ijk', T, F, H, coeffs)

        # 【核心重构】：计算整个系统的标量势能，让 JAX 自动生成严格的残差梯度！
        def compute_energy(params_1d):
            R_c, Z_c, Lam_c = unpack_params(params_1d)
            
            # 1. 运动学评估
            R       = eval_scalar(R_c) + R_base
            R_rho   = eval_scalar(R_c, d_rad=1) + R_rho_base
            R_theta = eval_scalar(R_c, d_pol=1) + R_theta_base
            R_zeta  = eval_scalar(R_c, d_tor=1) + R_zeta_base
            
            Z       = eval_scalar(Z_c) + Z_base
            Z_rho   = eval_scalar(Z_c, d_rad=1) + Z_rho_base
            Z_theta = eval_scalar(Z_c, d_pol=1) + Z_theta_base
            Z_zeta  = eval_scalar(Z_c, d_tor=1) + Z_zeta_base
            
            Lam_rho   = eval_scalar(Lam_c, d_rad=1, is_lam=True)
            Lam_theta = eval_scalar(Lam_c, d_pol=1, is_lam=True)
            Lam_zeta  = eval_scalar(Lam_c, d_tor=1, is_lam=True)
            
            R_safe = jnp.where(R < 1e-4, 1e-4, R)
            jac = R_safe * (R_theta * Z_rho - R_rho * Z_theta)
            
            # --- 【消除磁轴奇点 (0/0) 的数学修正】 ---
            # 因为 Phi' ~ rho 且 jac ~ rho，这会导致计算 B 场时出现 0/0。
            # 解决方法：提出 rho 因子，用 jac/rho 代替 jac
            rho_safe = jnp.where(rho_3d < 1e-8, 1e-8, rho_3d)
            jac_over_rho = jac / rho_safe
            
            jac_over_rho_safe = jnp.where(jnp.abs(jac_over_rho) < 1e-5, 
                                          jnp.where(jac_over_rho >= 0, 1e-5, -1e-5), 
                                          jac_over_rho)

            # 2. 物理量评估
            p_val = eval_poly(self.p_coeffs, rho_3d)
            iota = eval_poly(self.iota_coeffs, rho_3d)
            
            # 提出 rho 因子的通量梯度
            Phi_prime_over_rho = 2.0 * self.Phi_edge * jnp.ones_like(rho_3d)
            psi_prime_over_rho = iota * Phi_prime_over_rho
            
            g_rho_rho = R_rho**2 + Z_rho**2
            g_rho_theta = R_rho * R_theta + Z_rho * Z_theta
            g_theta_theta = R_theta**2 + Z_theta**2
            g_rho_zeta = R_rho * R_zeta + Z_rho * Z_zeta
            g_theta_zeta = R_theta * R_zeta + Z_theta * Z_zeta
            g_zeta_zeta = R_zeta**2 + Z_zeta**2 + R_safe**2
            
            # 3. 磁场构造 (由于使用了 over_rho，此处完全无奇点解析！)
            fac = 1.0 / (2 * jnp.pi * jac_over_rho_safe)
            B_rho   = fac * (Phi_prime_over_rho * Lam_theta - psi_prime_over_rho * Lam_zeta)
            B_theta = fac * (Phi_prime_over_rho * (1.0 - Lam_rho))
            B_zeta  = fac * (psi_prime_over_rho * (1.0 + Lam_rho))
            
            B_sub_rho = g_rho_rho * B_rho + g_rho_theta * B_theta + g_rho_zeta * B_zeta
            B_sub_theta = g_rho_theta * B_rho + g_theta_theta * B_theta + g_theta_zeta * B_zeta
            B_sub_zeta = g_rho_zeta * B_rho + g_theta_zeta * B_theta + g_zeta_zeta * B_zeta
            
            # B^2 标量
            B2 = B_rho * B_sub_rho + B_theta * B_sub_theta + B_zeta * B_sub_zeta
            
            # 4. MHD 离散能量积分: W = \int (B^2 / 2\mu_0 - P) dV
            W_density = B2 / (2.0 * self.mu_0) - p_val
            # 积分体积元使用绝对值保护符号反转
            W_total = jnp.mean(W_density * jnp.abs(jac))
            
            # 5. Lambda 规范消除惩罚 (惩罚常数项防止漂移)
            gauge_penalty = 1.0e2 * (Lam_c[0, 0, 0] ** 2)
            
            return W_total + gauge_penalty
            
        # 变分法的核心奥义：能量求偏导即为精确的物理受力残差！
        return jax.grad(compute_energy)

    def run_optimization(self, max_iter=30, tol=1e-8):
        print("--- 启动 JFNK 求解器 (带 LM/PTC 稳定器与能量梯度诊断) ---")
        x_n = np.copy(self.params_1d)
        mu = 10.0  
        
        idx_R = self.size_rz
        idx_Z = 2 * self.size_rz
        
        for iteration in range(max_iter):
            res_n = np.array(self.res_fn_jit(x_n))
            res_norm = np.linalg.norm(res_n)
            
            norm_R = np.linalg.norm(res_n[:idx_R])
            norm_Z = np.linalg.norm(res_n[idx_R:idx_Z])
            norm_Lam = np.linalg.norm(res_n[idx_Z:])
            
            print(f"\nNewton 迭代 {iteration:02d}: 总梯度残差 = {res_norm:.4e}, 阻尼 mu = {mu:.1e}")
            print(f"  ├─ 梯度范数 | dW/dR: {norm_R:.3e} | dW/dZ: {norm_Z:.3e} | dW/dLam: {norm_Lam:.3e}")
            
            if res_norm < tol:
                print("==> 成功收敛！")
                break
                
            def matvec(v):
                v_jnp = jnp.asarray(v, dtype=x_n.dtype)
                _, jv = jax.jvp(self.res_fn_jit, (x_n,), (v_jnp,))
                return np.array(jv) + mu * np.array(v)
                
            N = len(x_n)
            J_op = LinearOperator((N, N), matvec=matvec)
            
            dx, info = gmres(J_op, -res_n, rtol=1e-2, maxiter=min(N, 150)) 
            
            max_dx = np.max(np.abs(dx))
            print(f"  ├─ GMRES结果 | 状态: {'成功' if info==0 else f'未收敛(info={info})'} | 最大参数更新步长 max(|dx|) = {max_dx:.2e}")
            
            alpha = 1.0
            step_success = False
            for ls_step in range(6):
                x_new = x_n + alpha * dx
                res_new_norm = np.linalg.norm(self.res_fn_jit(x_new))
                
                if res_new_norm < res_norm: 
                    x_n = x_new
                    step_success = True
                    if ls_step > 0:
                        print(f"  [线搜索] 步长折半 {ls_step} 次, alpha={alpha:.3f}, 新残差={res_new_norm:.4e}")
                    break
                alpha *= 0.5
            
            if step_success:
                mu = max(1e-4, mu * 0.3)
            else:
                print("  [警告] 线搜索失败，残差上升！增大阻尼并强制微小步进...")
                mu = min(1e4, mu * 5.0)
                x_n = x_n + 0.01 * dx
                
        self.params_1d = x_n
        return self.params_1d

if __name__ == "__main__":
    config = {
        'NFP': 19,
        'Phi_edge': 1.0, 
        'L_rad': 4, 'M_pol': 3, 'N_tor': 2, 
        'L_rad_lam': 2, 'M_pol_lam': 2, 'N_tor_lam': 1, 
        'N_rad': 12, 'N_pol': 12, 'N_tor_grid': 6 
    }
    
    veq = Veq3D(config)
    final_params = veq.run_optimization(max_iter=30)
