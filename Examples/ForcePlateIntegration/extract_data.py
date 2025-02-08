import os
import sys
sys.path.append("./../../")
script_folder,_ = os.path.split(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import opensim
# import requests

import utils as ut
import utilsKinematics

def extract_data_t(session_id, trial_name, time):
    session_id = session_id
    trial_name = trial_name
    data_folder = os.path.abspath(os.path.join(script_folder, 'Data', session_id))

    if not os.path.exists(os.path.join(data_folder, 'MarkerData')):
        _, model_name = ut.download_kinematics(session_id, folder=data_folder)
    else:
        model_name, _ = os.path.splitext(ut.get_model_name_from_metadata(data_folder))

    # Iniciar análisis cinemático
    kinematics = utilsKinematics.kinematics(data_folder, trial_name, 
                                            modelName=model_name, 
                                            lowpass_cutoff_frequency_for_coordinate_values=10)

    opencap_settings = ut.get_main_settings(data_folder, trial_name)
    opencap_metadata = ut.import_metadata(os.path.join(data_folder, 'sessionMetadata.yaml'))

    # Ruta del archivo .mot
    mot_file_path = os.path.join(data_folder, 'OpenSimData', 'Kinematics', f'{trial_name}.mot')

    # Cargar el archivo .mot utilizando OpenSim
    mot_table = opensim.TimeSeriesTable(mot_file_path)

    # Convertir la tabla a un DataFrame de pandas
    mot_df = pd.DataFrame(mot_table.getMatrix().to_numpy())
    mot_df.columns = mot_table.getColumnLabels()
    mot_df['time'] = mot_table.getIndependentColumn()

    # Reordenar las columnas para que 'time' sea la primera columna
    mot_df = mot_df[['time'] + [col for col in mot_df.columns if col != 'time']]

    col = ['time','hip_flexion_r','hip_adduction_r','hip_rotation_r','knee_angle_r','ankle_angle_r','subtalar_angle_r',
        'hip_flexion_l',	'hip_adduction_l','hip_rotation_l','knee_angle_l','ankle_angle_l','subtalar_angle_l']

    df_mot = mot_df[col]
    # Mostrar las primeras filas del DataFrame
    print(df_mot[df_mot['time']==time].head())

    # Guardar el DataFrame en un archivo CSV si es necesario
    # output_csv_path = os.path.join(data_folder, 'OpenSimData', 'Kinematics', f'{trial_name}.csv')
    # mot_df.to_csv(output_csv_path, index=False)

