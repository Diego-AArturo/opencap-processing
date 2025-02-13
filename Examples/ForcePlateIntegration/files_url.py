def convertir_a_descarga(movimiento):
    enlace = movimiento["link"]
    # Extraer el ID del archivo del enlace original
    if "id=" in enlace:
        file_id = enlace.split("id=")[1].split("&")[0]
    else:
        raise ValueError("El enlace no tiene un formato válido de Google Drive")

    # Construir el enlace de descarga
    enlace_descarga = f"https://drive.usercontent.google.com/u/0/uc?id={file_id}&export=download"
    return {"move": movimiento["move"], "link": enlace_descarga}

# Lista de enlaces originales
enlaces_originales = [{
        "participant_id": "P1",
        "session_id": "da9fcf0a-3ec2-4990-b1e5-7f0d8dec5d38",
        "movements": [
            {"move": 'escalón_derecho_1', "link": 'https://drive.google.com/open?id=169loZMDfK_n3OaDMWBh6N4g0LA85eqR1&usp=drive_copy'},
            {"move": 'escalón_derecho_2', "link": 'https://drive.google.com/open?id=16CfUcL-P1XBfs4SAYL-EHsAaTuFeFWrV&usp=drive_copy'},
            {"move": 'escalón_izquierdo_1', "link": 'https://drive.google.com/open?id=16FXnbPWt79vrKNayDQm1UWSGa6haTbsW&usp=drive_copy'},
            {"move": 'escalón_izquierdo_2', "link": 'https://drive.google.com/open?id=16ae4kdFR07BguExsMx15YARK8dOmq6n4&usp=drive_copy'},
            {"move": 'estocada_deslizamiento_lateral_derecho_1', "link": 'https://drive.google.com/open?id=15M-gSfDxzu_FM-p9hbA2_c9lVV8UYobo&usp=drive_copy'},
            {"move": 'estocada_deslizamiento_lateral_derecho_2', "link": 'https://drive.google.com/open?id=15BqEnxKxCq08DDcn-ZNKIF4T9S28F9jX&usp=drive_copy'},
            {"move": 'estocada_deslizamiento_lateral_izquierdo_1', "link": 'https://drive.google.com/open?id=15pLYW6420yRutkDxEWBEJa-cUUuI0a10&usp=drive_copy'},
            {"move": 'estocada_deslizamiento_lateral_izquierdo_2', "link": 'https://drive.google.com/open?id=15lnbrjrLVHWYhinGWDARU4q-wpztY0JX&usp=drive_copy'},
            {"move": 'estocada_deslizamiento_posterior_derecho_1', "link": 'https://drive.google.com/open?id=15hKv91paucoHIz8iir_29OInPnyYt-cI&usp=drive_copy'},
            {"move": 'estocada_deslizamiento_posterior_derecho_2', "link": 'https://drive.google.com/open?id=15YvgYz921mHPYEkO1UNNIlO4BPMBtFBt&usp=drive_copy'},
            {"move": 'estocada_deslizamiento_posterior_izquierdo_1', "link": 'https://drive.google.com/open?id=15vIgIICgo_US6hd9KxW7_0egRf5CK1ZF&usp=drive_copy'},
            {"move": 'estocada_deslizamiento_posterior_izquierdo_2', "link": 'https://drive.google.com/open?id=161-8WisGZLHpau_SSQ8gPt_bETYq4MgW&usp=drive_copy'},
            {"move": 'estocada_izquierda_1', "link": 'https://drive.google.com/open?id=15SDdpjPusvdqYvfSvXZnj_eCWkrCorS7&usp=drive_copy'},
            {"move": 'estocada_izquierda_2', "link": 'https://drive.google.com/open?id=15VXaE16Qod3OOfefbBg7SFBBEEATPuhk&usp=drive_copy'},
            {"move": 'estocada_lateral_derecha_1', "link": 'https://drive.google.com/open?id=157Hqw6eetDbR-XS2-lEPjulXR5mtdX69&usp=drive_copy'},
            {"move": 'estocada_lateral_derecha_2', "link": 'https://drive.google.com/open?id=157njrRM9-xKl3_LA0atpJoL51w_dw721&usp=drive_copy'},
            {"move": 'estocada_lateral_izquierda_1', "link": 'https://drive.google.com/open?id=14vK21uYcaW8ZLKbsWt1OfFa3HKRITHWh&usp=drive_copy'},
            {"move": 'estocada_lateral_izquierda_2', "link": 'https://drive.google.com/open?id=14yRdUeOOZhxJJcJmLowfyZGg3PFJsE-W&usp=drive_copy'},
            {"move": 'sentadilla_60_1', "link": 'https://drive.google.com/open?id=14yvRQbhCKhkYDXzX0ePRTHH_q4YoySR_&usp=drive_copy'},
            {"move": 'sentadilla_60_2', "link": 'https://drive.google.com/open?id=14dQca3Q_50yDBYB65QvCQONSxJyFxU2a&usp=drive_copy'},
            {"move": 'sentadilla_90_1', "link": 'https://drive.google.com/open?id=14g5fso92QEBClGrnv7zW4-3G2ghgu9x0&usp=drive_copy'},
            {"move": 'sentadilla_90_2', "link": 'https://drive.google.com/open?id=14nN-BaDEsz8O1PsoLHfJgJm-zGQZymRE&usp=drive_copy'},
        ]
    }
]

# Convertir enlaces
enlaces_descarga = []
for original in enlaces_originales:
    movimientos_descarga = [convertir_a_descarga(movimiento) for movimiento in original["movements"]]
    enlaces_descarga.append({
        "participant_id": original["participant_id"],
        "session_id": original["session_id"],
        "movements": movimientos_descarga
    })

# Imprimir resultados
# for original, descarga in zip(enlaces_originales, enlaces_descarga):
#     print(f"Original: {original}\nDescarga: {descarga}\n")

print(enlaces_descarga)