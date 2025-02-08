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
    session_id = session_id#'da9fcf0a-3ec2-4990-b1e5-7f0d8dec5d38'
    trial_name = trial_name#'escalon_derecho_1'
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
    print(df_mot[df_mot['time']==time])

    # Guardar el DataFrame en un archivo CSV si es necesario
    # output_csv_path = os.path.join(data_folder, 'OpenSimData', 'Kinematics', f'{trial_name}.csv')
    # mot_df.to_csv(output_csv_path, index=False)


data = [{'session_id': 'da9fcf0a-3ec2-4990-b1e5-7f0d8dec5d38', 'trial_names': ['escalon_derecho_1',], 'times': [4.65,]},
        {'session_id': 'a3724192-e2b6-4636-b176-3b3028d66230', 'trial_names': ['escalon_derecho_2',], 'times': [4.65,]},
        {'session_id': '0199dfe7-2595-471e-8f49-d5a887434357', 'trial_names': ['escalon_derecho_3',], 'times': [4.65,]},
        {'session_id': 'dd83a45d-85b0-4f5e-bd26-93d22c413ed9', 'trial_names': ['escalon_derecho_4',], 'times': [4.65,]},
        {'session_id': '34b7f090-1cbd-43ce-86a2-a40c50feec3f', 'trial_names': ['escalon_derecho_5',], 'times': [4.65,]},
        {'session_id': 'bfd679e8-b5e3-4185-b2f2-011554045077', 'trial_names': ['escalon_derecho_6',], 'times': [4.65,]},
        {'session_id': '37ad96c7-d786-4338-81ea-0d58104e9bb5', 'trial_names': ['escalon_derecho_7',], 'times': [4.65,]},
        {'session_id': '1a934006-0010-471a-9c38-fbd7edb6ffbc', 'trial_names': ['escalon_derecho_8',], 'times': [4.65,]},
        {'session_id': '03142c4c-922a-4d9e-8cf3-a2eb229baa14', 'trial_names': ['escalon_derecho_9',], 'times': [4.65,]},
        {'session_id': 'cb608973-5e67-4d57-8b64-2e73fbbc6361', 'trial_names': ['escalon_derecho_9',], 'times': [4.65,]},
        {'session_id': '39cb5ebb-153d-4aa0-ada7-a28c849eec67', 'trial_names': ['escalon_derecho_1',], 'times': [4.65,]},
        {'session_id': '3eee7aa7-cece-4ead-a822-554b95b05613', 'trial_names': ['escalon_derecho_2',], 'times': [4.65,]},
        #{'session_id': '0199dfe7-2595-471e-8f49-d5a887434357', 'trial_names': ['escalon_derecho_3',], 'times': [4.65,]},
        {'session_id': 'ad28c4e1-9f27-4554-9e19-1063888ab302', 'trial_names': ['escalon_derecho_4',], 'times': [4.65,]},
        {'session_id': '8cff7224-37bf-44d5-94d0-f7dfdea5bc36', 'trial_names': ['escalon_derecho_5',], 'times': [4.65,]},
        {'session_id': 'fe7e294e-b199-4c16-b9ca-c8c7841c42ba', 'trial_names': ['escalon_derecho_6',], 'times': [4.65,]},
        {'session_id': '634a9945-6598-4478-aa6f-3e88bdb2ab07', 'trial_names': ['escalon_derecho_7',], 'times': [4.65,]},
        {'session_id': '126c9b36-d8f4-4d25-83e3-cd3afbc04148', 'trial_names': ['escalon_derecho_8',], 'times': [4.65,]},
        {'session_id': '67a1f5a5-1723-4c3c-86a5-54a06a0eb878', 'trial_names': ['escalon_derecho_9',], 'times': [4.65,]},
        {'session_id': 'a3ba3d30-d5fd-4727-b460-b4310339a852', 'trial_names': ['escalon_derecho_9',], 'times': [4.65,]},
        #
        {'session_id': 'f8ab78be-81ee-451f-affa-80a56880f741', 'trial_names': ['escalon_derecho_1',], 'times': [4.65,]},
        {'session_id': 'a3724192-e2b6-4636-b176-3b3028d66230', 'trial_names': ['escalon_derecho_2',], 'times': [4.65,]},
        {'session_id': 'c71d092b-93ba-4dfb-afc3-57fb46ef0736', 'trial_names': ['escalon_derecho_3',], 'times': [4.65,]},
        {'session_id': '85faebd8-e10e-41a4-a0cd-4abea37e1948', 'trial_names': ['escalon_derecho_4',], 'times': [4.65,]},
        #falta del 25 al 31
        {'session_id': '34b7f090-1cbd-43ce-86a2-a40c50feec3f', 'trial_names': ['escalon_derecho_5',], 'times': [4.65,]},
        {'session_id': 'bfd679e8-b5e3-4185-b2f2-011554045077', 'trial_names': ['escalon_derecho_6',], 'times': [4.65,]},
        {'session_id': '37ad96c7-d786-4338-81ea-0d58104e9bb5', 'trial_names': ['escalon_derecho_7',], 'times': [4.65,]},
        {'session_id': '1a934006-0010-471a-9c38-fbd7edb6ffbc', 'trial_names': ['escalon_derecho_8',], 'times': [4.65,]},
        {'session_id': '03142c4c-922a-4d9e-8cf3-a2eb229baa14', 'trial_names': ['escalon_derecho_9',], 'times': [4.65,]},
        {'session_id': 'cb608973-5e67-4d57-8b64-2e73fbbc6361', 'trial_names': ['escalon_derecho_9',], 'times': [4.65,]},
        
]
extract_data_t('da9fcf0a-3ec2-4990-b1e5-7f0d8dec5d38', 'escalon_derecho_1', 4.65)