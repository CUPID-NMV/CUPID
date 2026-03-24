#!/usr/bin/env python3
import numpy as np
import math

# Costanti fisiche
alpha = 1 / 137.035999  # costante di struttura fine
c = 3e8                 # velocità della luce [m/s]

def cherenkov_photons_in_water(
    beta,
    track_length,
    n=1.33,
    z_charge=1,
    lambda_min_nm=300,
    lambda_max_nm=600,
    r0=np.array([0.0, 0.0, 0.1]),
    u_part=np.array([0.0, 0.0, -1.0]),
    rng=None
):
    """
    Genera fotoni Cherenkov per una particella in acqua.
    
    beta: v/c della particella
    track_length: lunghezza del cammino in metri
    n: indice di rifrazione
    z_charge: carica in unità di e (1 per mu/e-)
    lambda_min_nm, lambda_max_nm: range spettrale in nm
    r0: posizione iniziale della particella (3D)
    u_part: direzione unitaria della particella (3D)
    """
    if rng is None:
        rng = np.random.default_rng()

    u_part = np.array(u_part, dtype=float)
    u_part = u_part / np.linalg.norm(u_part)

    # Controllo soglia Cherenkov
    if beta <= 1.0 / n:
        return {
            "positions": np.zeros((0, 3)),
            "directions": np.zeros((0, 3)),
            "wavelengths_nm": np.zeros(0),
        }

    # Converti lambda in metri
    lambda_min = lambda_min_nm * 1e-9
    lambda_max = lambda_max_nm * 1e-9

    # Angolo di Cherenkov
    cos_theta_c = 1.0 / (n * beta)
    if cos_theta_c > 1.0:  # sicurezza numerica
        return {
            "positions": np.zeros((0, 3)),
            "directions": np.zeros((0, 3)),
            "wavelengths_nm": np.zeros(0),
        }
    theta_c = math.acos(cos_theta_c)
    sin_theta_c = math.sin(theta_c)

    # Fattore K della formula di Frank-Tamm
    K = 2 * math.pi * alpha * z_charge**2 * (1 - 1 / (beta**2 * n**2))

    # dN/dx integrato su [lambda_min, lambda_max]
    dNdx = K * (1.0 / lambda_min - 1.0 / lambda_max)

    # Numero medio di fotoni lungo il track
    mu = dNdx * track_length

    # Estrai il numero reale di fotoni (Poisson)
    if mu <= 0:
        return {
            "positions": np.zeros((0, 3)),
            "directions": np.zeros((0, 3)),
            "wavelengths_nm": np.zeros(0),
        }

    N_gamma = rng.poisson(mu)
    if N_gamma == 0:
        return {
            "positions": np.zeros((0, 3)),
            "directions": np.zeros((0, 3)),
            "wavelengths_nm": np.zeros(0),
        }

    # --- Campiona le lunghezze d'onda (distribuzione ~ 1/lambda^2) ---
    u = rng.random(N_gamma)
    inv_lambda_min = 1.0 / lambda_min
    inv_lambda_max = 1.0 / lambda_max
    # 1/lambda = 1/lambda_min - u*(1/lambda_min - 1/lambda_max)
    inv_lambda = inv_lambda_min - u * (inv_lambda_min - inv_lambda_max)
    lambdas = 1.0 / inv_lambda  # in metri

    # --- Posizione di emissione lungo il track (uniforme) ---
    s = rng.random(N_gamma) * track_length
    r0 = np.array(r0, dtype=float)
    positions = r0[None, :] + s[:, None] * u_part[None, :]

    # --- Base ortonormale per il cono (e1, e2, u_part) ---
    if abs(u_part[2]) < 0.9:
        tmp = np.array([0.0, 0.0, 1.0])
    else:
        tmp = np.array([1.0, 0.0, 0.0])

    e1 = np.cross(u_part, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u_part, e1)

    # --- Direzioni dei fotoni sul cono di Cherenkov ---
    phi = rng.random(N_gamma) * 2.0 * math.pi
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    directions = (
        cos_theta_c * u_part[None, :] +
        sin_theta_c * (cos_phi[:, None] * e1[None, :] +
                       sin_phi[:, None] * e2[None, :])
    )

    return {
        "positions": positions,          # (N_gamma, 3)
        "directions": directions,        # (N_gamma, 3)
        "wavelengths_nm": lambdas * 1e9  # in nm
    }

def simulate_tank_events(
    n_events=1000,
    beta=0.99,
    tank_Lz=0.10,  # spessore (10 cm)
    tank_Lx=0.30,  # 30 cm
    tank_Ly=1.00,  # 100 cm
    qe=0.25,
    n_index=1.33,
    lambda_min_nm=300,
    lambda_max_nm=600,
    rng=None
):
    """
    Simula n_events particelle che attraversano la vasca.
    
    Geometria:
      0 <= x <= tank_Lx
      0 <= y <= tank_Ly
      0 <= z <= tank_Lz
      
    Particella:
      entra al centro della faccia superiore (z = tank_Lz)
      direzione verticale verso il basso (0, 0, -1)
      
    Fotocatodo:
      intera faccia inferiore (z = 0).
    """
    if rng is None:
        rng = np.random.default_rng()

    pe_counts = []

    for _ in range(n_events):
        # Punto di ingresso al centro della faccia superiore
        r0 = np.array([tank_Lx / 2, tank_Ly / 2, tank_Lz])
        u_part = np.array([0.0, 0.0, -1.0])
        track_length = tank_Lz / abs(u_part[2])  # = tank_Lz

        photons = cherenkov_photons_in_water(
            beta=beta,
            track_length=track_length,
            n=n_index,
            z_charge=1,
            lambda_min_nm=lambda_min_nm,
            lambda_max_nm=lambda_max_nm,
            r0=r0,
            u_part=u_part,
            rng=rng
        )

        pos = photons["positions"]
        dire = photons["directions"]
        N_gamma = pos.shape[0]
        if N_gamma == 0:
            pe_counts.append(0)
            continue

        # Considera solo fotoni che vanno verso il basso (uz < 0)
        uz = dire[:, 2]
        mask_down = uz < 0
        pos_d = pos[mask_down]
        dire_d = dire[mask_down]
        uz_d = uz[mask_down]

        if pos_d.shape[0] == 0:
            pe_counts.append(0)
            continue

        # Intersezione con il piano z = 0: r(t) = pos + t * dire
        # t_hit = -z / uz  (per z_final = 0)
        t_hit = -pos_d[:, 2] / uz_d
        hit_pos = pos_d + t_hit[:, None] * dire_d

        x_hit = hit_pos[:, 0]
        y_hit = hit_pos[:, 1]

        # Fotoni che arrivano alla faccia inferiore dentro i bordi
        mask_in = (
            (t_hit > 0) &
            (x_hit >= 0) & (x_hit <= tank_Lx) &
            (y_hit >= 0) & (y_hit <= tank_Ly)
        )

        n_photons_on_cathode = np.count_nonzero(mask_in)

        # Fotoelettroni con probabilità QE per fotone
        n_pe = rng.binomial(n=n_photons_on_cathode, p=qe)
        pe_counts.append(n_pe)

    return np.array(pe_counts)

if __name__ == "__main__":
    rng = np.random.default_rng(123)

    # Esempio: muone relativistico (beta ~ 0.995)
    beta_mu = 0.995
    pe_mu = simulate_tank_events(
        n_events=1000,
        beta=beta_mu,
        qe=0.25,
        rng=rng
    )
    print("MUONE:")
    print("  <N_pe> =", pe_mu.mean())
    print("  sigma(N_pe) =", pe_mu.std())

    # Esempio: elettrone relativistico (beta ~ 0.999)
    beta_e = 0.999
    pe_e = simulate_tank_events(
        n_events=1000,
        beta=beta_e,
        qe=0.25,
        rng=rng
    )
    print("\nELETTRONE:")
    print("  <N_pe> =", pe_e.mean())
    print("  sigma(N_pe) =", pe_e.std())
