import os
import json
import numpy as np
import matplotlib.pyplot as plt
import utilsKinematics
import utils as ut
from scipy.interpolate import interp1d
from utils import storage_to_numpy

def detect_upforce(signal, threshold_factor=0.1):
    """
    Detecta cundo hay el primer cambio de fuerza en la plataforma
    - signal: array, la señal de fuerza de las plataformas.
    - threshold_factor: float, factor del umbral basado en la desviación estándar.
    """
    baseline = np.mean(signal[:10])  # Nivel base del talón antes del movimiento
    
    # Encontrar el primer despegue (cuando el talón se eleva significativamente)
    # plt.axhline(y=baseline + np.std(signal) * threshold_factor, color='r', linestyle='--', label='Baseline')
    for i in range(1, len(signal)):
        if signal[i] > baseline + np.std(signal) * threshold_factor:
            break

    for j in range(i, len(signal)):
        if signal[j] < baseline + np.std(signal) * threshold_factor*0.5:
            break
    
    return i,j
        

# Setup
participant_id = 'P25'
data_folder = os.path.abspath(os.path.join("./Data", participant_id))
# Listar los nombres de los trials en la carpeta de Kinematics
kinematics_folder = os.path.join(data_folder, "OpenSimData", "Kinematics")
trial_names = [f.replace(".mot", "") for f in os.listdir(kinematics_folder) if f.endswith(".mot")]

modelName = "LaiUhlrich2022_scaled"

gravity = 9.81
rmse_results = {}

for trial_name in trial_names:
    # Obtener aceleración y masa
    kinematics = utilsKinematics.kinematics(data_folder, trial_name, modelName=modelName)
    acc = kinematics.get_center_of_mass_accelerations(lowpass_cutoff_frequency=4)
    opencap_metadata = ut.import_metadata(os.path.join(data_folder, 'sessionMetadata.yaml'))
    mass = opencap_metadata['mass_kg']  # O extraer del YAML con ut.import_metadata()

    # Simulación de fuerza desde aceleración: F = m(a + g)
    simulated_vgrf = (acc['y'] + gravity) / gravity  # N
    time_simulate = kinematics.time


    # Cargar fuerza real
    try:
        force_path = os.path.join(data_folder, "MeasuredForces", trial_name, f"{trial_name}_syncd_forces.mot")
        forces = storage_to_numpy(force_path)
        force_data = forces.view(np.float64).reshape(forces.shape + (-1,))
        headers = forces.dtype.names
        time_real = force_data[:, 0]  # Tiempo de la fuerza
        # Extraer señal de fuerza vertical (R + L)
        vy_idx = [i for i, name in enumerate(headers) if "ground_force_vy" in name]
        real_vgrf = np.sum(force_data[:, vy_idx], axis=1)  # N
        real_vgrf = real_vgrf / ( gravity*mass)  # Normalizar por el peso corporal

        

        # Normalizar
        start_idx = np.argmin(np.abs(time_simulate - time_real[0]))

        end_idx = np.argmin(np.abs(time_simulate - time_real[-1])) + 1

        new_simulate = simulated_vgrf[start_idx:end_idx]        
        new_time_simulate = time_simulate[start_idx:end_idx]

        framerate_forces = 1 / np.diff(time_real[:2])[0]
        framerate_kinematics = 1 / np.diff(new_time_simulate[:2])[0]

        # simulated_vgrf = (new_simulate - np.min(new_simulate)) / (np.max(new_simulate) - np.min(new_simulate))
        # real_vgrf = (real_vgrf - np.min(real_vgrf)) / (np.max(real_vgrf) - np.min(real_vgrf))
        
        # Interpolación para igualar longitudes
        interp_real = interp1d(time_real, real_vgrf, kind='linear', bounds_error=False, fill_value="extrapolate")
        real_resampled = interp_real(new_time_simulate)

        idx_up, idx_down = detect_upforce(real_resampled)
        
        # simulated_vgrf[:idx_up] = 0
        # simulated_vgrf[idx_down:] = 0
        simulated_vgrf = simulated_vgrf.reset_index(drop=True)

        # plt.plot(simulated_vgrf, label="Simulado") 
        plt.plot(real_resampled, label="Real")
        
        # plt.axvline(x=new_time_simulate[idx_up],linestyle='--')
        # plt.axvline(x=new_time_simulate[idx_down],linestyle='--')
        
        plt.title(f"Fuerza vertical normalizada - {trial_name}")
        plt.xlabel("Muestras")
        plt.ylabel("vGRF (bodyweights)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


        # Evaluación de sincronización: desplazar y calcular RMSE
#         max_lag = 100  # muestras
#         lags = np.arange(-max_lag, max_lag + 1,50)
#         rmses = []

#         for lag in lags:
#             if lag < 0:
#                 shifted = simulated_vgrf[-lag:]
#                 ref = real_resampled[:len(shifted)]
#             else:
#                 shifted = simulated_vgrf[:-lag] if lag != 0 else simulated_vgrf
#                 ref = real_resampled[lag:] if lag != 0 else real_resampled

#             min_len = min(len(shifted), len(ref))  # Evita errores si hay desbalance
#             rmse = np.sqrt(np.mean((shifted[:] - ref[:]) ** 2))
#             rmses.append(rmse)
#             print(f"RMSE para {trial_name} con lag {lag}: {rmse:.4f}")
#         # Guardar y graficar
#         rmse_results[trial_name] = {"lags": lags.tolist(), "rmses": rmses}
#         plt.plot(lags, rmses, label=trial_name)
    except FileNotFoundError:
        pass

# plt.xlabel("Lag (frames)")
# plt.ylabel("RMSE (vGRF normalizado)")
# plt.title("Evaluación de sincronización por prueba")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
