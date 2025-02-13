
abreviaciones = {
    "escalon_derecho": "ESCALON D",
    "escalon_izquierdo": "ESCALON I",
    "estocada_deslizamiento_lateral_derecho": "EST.DESl.LATERAL D",
    "estocada_deslizamiento_lateral_izquierdo": "EST.DESl.LATERAL I",
    "estocada_deslizamiento_posterior_derecho": "EST.DESL.POSTERIOR D",
    "estocada_deslizamiento_posterior_izquierdo": "EST.DESL.POSTERIOR I",
    "estocada_derecha": "EST.POSTERIOR D",
    "estocada_izquierda": "EST.POSTERIOR I",
    "estocada_lateral_derecha": "EST.LATERAL D",
    "estocada_lateral_izquierda": "EST.LATERAL I",
    "sentadilla_60": "SENT 60",
    "sentadilla_90": "SENT 90"
}

trial_names =  [
            'escalon_derecho_1',
            'escalon_izquierdo_1',
            'estocada_deslizamiento_lateral_derecho_1',
            'estocada_deslizamiento_lateral_izquierdo_1',
            'estocada_deslizamiento_posterior_derecho_1',
            'estocada_deslizamiento_posterior_izquierdo_1',
            'estocada_izquierda_1',
            'estocada_lateral_derecha_1',
            'estocada_lateral_izquierda_1',
            'sentadilla_60_1',
            'sentadilla_90_1']

for trial_name in trial_names:
        
        sheet_name = abreviaciones.get(trial_name, trial_name)
        print(trial_name.split('_'),": ",len(trial_name.split('_')))