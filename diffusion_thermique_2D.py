"""
============================================================
 PROJET 3a : Équation de la chaleur 2D par différences finies
 ----------------------------------------------------------
 Auteur : Fafadji Ambroise YELOUH    |    Date : Novembre 2025
 ----------------------------------------------------------
 Schémas :
   • Explicite (Euler avant)    -> condition de stabilité CFL
   • Crank–Nicolson (implicite) -> inconditionnellement stable
 Test : source chaude au centre, parois froides, condition
 initiale gaussienne, comparaison quantitative des deux schémas
============================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

# ----------------------------------------------------------
# Paramètres physiques et numériques
# ----------------------------------------------------------
alpha = 1.0e-4           # diffusivité thermique (m²/s)
Lx, Ly = 0.20, 0.20      # dimensions (m)
Nx, Ny = 60, 60          # points de grille
dx, dy = Lx/(Nx-1), Ly/(Ny-1)
dt     = 0.4             # s (choix compatible stabilité explicite)
Nt     = 400             # nombre d'itérations temporelles
T0     = 293.15          # K
Tc     = 373.15          # K (bord chaud)

# ----------------------------------------------------------
# Condition initiale : gaussienne centrée
# ----------------------------------------------------------
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
T_init = T0 + 40.0*np.exp(-((X-Lx/2)**2 + (Y-Ly/2)**2)/(2*(Lx/10)**2))

# Bordure : condition de Dirichlet Tc sur les 4 parois
def apply_bc(T):
    T[0,:]  = Tc; T[-1,:] = Tc
    T[:,0]  = Tc; T[:,-1] = Tc
    return T

# ----------------------------------------------------------
# Vérification de la condition de stabilité (explicite)
# ----------------------------------------------------------
dt_max = 1.0/(2*alpha*(1/dx**2 + 1/dy**2))
print(f"dt_max (explicite) = {dt_max:.4e} s  |  dt = {dt:.4e} s")
assert dt < dt_max, "Attention : schéma explicite instable !"

# ----------------------------------------------------------
# Schéma EXPLICITE
# ----------------------------------------------------------
def scheme_explicit(T, Nt):
    T = T.copy()
    for n in range(Nt):
        d2x = (T[1:-1,2:] - 2*T[1:-1,1:-1] + T[1:-1,:-2])/dx**2
        d2y = (T[2:,1:-1] - 2*T[1:-1,1:-1] + T[:-2,1:-1])/dy**2
        T[1:-1,1:-1] += alpha*dt*(d2x + d2y)
        T = apply_bc(T)
    return T

# ----------------------------------------------------------
# Schéma CRANK–NICOLSON (matrice 2D aplatie)
# ----------------------------------------------------------
def build_laplacian(Nx, Ny, dx, dy):
    N = Nx*Ny
    diag_main = -2.0/dx**2 - 2.0/dy**2
    diag_x    =  1.0/dx**2
    diag_y    =  1.0/dy**2
    offsets   = [0, 1, -1, Nx, -Nx]
    data      = [np.full(N, diag_main),
                 np.full(N-1, diag_x),
                 np.full(N-1, diag_x),
                 np.full(N-Nx, diag_y),
                 np.full(N-Nx, diag_y)]
    L = diags(data, offsets, shape=(N, N), format='csr')
    # blocage des bords (Dirichlet) : on annule les lignes/colonnes
    for j in range(Ny):
        for i in (0, Nx-1):
            k = j*Nx + i
            L.data[L.indptr[k]:L.indptr[k+1]] = 0
    for i in range(Nx):
        for j in (0, Ny-1):
            k = j*Nx + i
            L.data[L.indptr[k]:L.indptr[k+1]] = 0
    return L

def scheme_crank_nicolson(T, Nt, dt):
    T = T.copy()
    N = Nx*Ny
    L = build_laplacian(Nx, Ny, dx, dy)
    I = eye(N, format='csr')
    A = (I - 0.5*alpha*dt*L).tocsc()
    B = (I + 0.5*alpha*dt*L).tocsr()
    for n in range(Nt):
        T_flat = T.ravel()
        T_flat = spsolve(A, B @ T_flat)
        T = T_flat.reshape((Ny, Nx))
        T = apply_bc(T)
    return T

# ----------------------------------------------------------
# Exécution & visualisation
# ----------------------------------------------------------
T_exp = scheme_explicit(T_init, Nt)
T_cn  = scheme_crank_nicolson(T_init, Nt, dt)

# Comparaison quantitative
err = np.max(np.abs(T_exp - T_cn))
print(f"Écart max explicite vs Crank–Nicolson : {err:.4f} K")

# ----------------------------------------------------------
# Tracés
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15,4.5))
for ax, T, title in zip(axes, [T_init, T_exp, T_cn],
                        ["t = 0", f"t = {Nt*dt:.0f} s (explicite)",
                         f"t = {Nt*dt:.0f} s (Crank–Nicolson)"]):
    c = ax.contourf(X, Y, T, levels=20, cmap='inferno')
    ax.set_title(title); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect('equal')
    plt.colorbar(c, ax=ax, label="T (K)")
plt.tight_layout(); plt.savefig("chaleur_2d.png", dpi=150); plt.show()

# ----------------------------------------------------------
# Profil selon y = Ly/2
# ----------------------------------------------------------
i_y = Ny//2
plt.figure(figsize=(8,4))
plt.plot(x, T_exp[i_y,:], 'o-', label='Explicite', mfc='none')
plt.plot(x, T_cn[i_y,:],  's-', label='Crank–Nicolson', mfc='none')
plt.xlabel("x (m)"); plt.ylabel("T (K)")
plt.title("Profil T(x) à y = Ly/2")
plt.grid(True); plt.legend()
plt.tight_layout(); plt.savefig("profil_T.png", dpi=150); plt.show()
