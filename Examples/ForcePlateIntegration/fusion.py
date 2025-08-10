import c3d
import pandas as pd
import numpy as np



# ---------- CONFIGURACIÓN ----------
TRC_PATH = r"C:\Users\Diego Alejandro\Downloads\motive\motive\sentadilla_60_1.trc"
MOT_PATH = r"C:\Users\Diego Alejandro\Downloads\motive\motive\sentadilla_60_1_plat.mot"
OUTPUT_PATH = "sentadilla_60_1_combined.c3d"

# ---------- 1. CARGAR TRC (MARCADORES) ----------
# El archivo tiene 5 líneas de cabecera
trc = pd.read_csv(TRC_PATH, sep='\t', skiprows=5)
frame_rate = 100.0  # Según tu archivo
n_frames = trc.shape[0]
marker_names = trc.columns[2::3].tolist()
n_markers = len(marker_names)

# Extraer marcadores y reorganizar shape: (frames, markers, xyz)
marker_data = trc.iloc[:, 2:].to_numpy()
markers = marker_data.reshape((n_frames, n_markers, 3))

# ---------- 2. CARGAR MOT (FUERZAS) ----------
with open(MOT_PATH) as f:
    for i, line in enumerate(f):
        if line.strip().lower() == 'endheader':
            skip = i + 1
            break

mot = pd.read_csv(
    MOT_PATH,
    sep=r'\s+',          # espacio o tab
    skiprows=skip,
    engine='python'
)
analog_labels = mot.columns[1:]
analog_data = mot.iloc[:, 1:].to_numpy()
analog_frame_rate = 1000
analogs_per_frame = analog_frame_rate // int(frame_rate)
# assert analog_data.shape[0] == n_frames * analogs_per_frame, "Fuerzas y marcadores no están sincronizados"

# ---------- 3. CONSTRUIR Y GUARDAR .C3D ----------
with open(OUTPUT_PATH, 'wb') as f:
    writer = c3d.Writer()
    writer.frame_rate = frame_rate
    writer.analog_sample_rate = analog_frame_rate

    # Etiquetas
    writer.set_point_labels(marker_names)
    writer.set_analog_labels(list(analog_labels))

    for i in range(n_frames):
        marker_frame = markers[i]
        analog_chunk = analog_data[i*analogs_per_frame:(i+1)*analogs_per_frame]
        writer.add_frames([marker_frame], analog_chunk.T)

    writer.write(f)

print(f"✅ Archivo exportado como: {OUTPUT_PATH}")

