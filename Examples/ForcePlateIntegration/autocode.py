from funtion_integrate_forceplates import IntegrateForcepalte

records = [
    # {
    #     "move": 'sentadilla_60_afuera',
    #     "link": 'https://drive.usercontent.google.com/u/2/uc?id=1b2JsFs4wQH0O3ajHEYgQyPhrs6B6jJmM&export=download'
    # },
    {
        "move": 'sentadilla_60_adentro',
        "link": 'https://drive.usercontent.google.com/u/2/uc?id=1HspDM5ahC4Jg5TcrwUw8ZYLgtWP0jH7r&export=download',
    },
    {
        "move": 'estocada_lateral_derecha_afuera',
        "link": 'https://drive.usercontent.google.com/u/2/uc?id=1myIkZkCIJzdPaOpU_arXzbiFlxW4-OqJ&export=download'
    },
    {
        "move": 'estocada_lateral_derecha_afuera_modificada',
        "link": 'https://drive.usercontent.google.com/u/2/uc?id=1gaM8jziYdiobfgb1GFUB9iBdnnzsDXOw&export=download'
    },
    {
        "move": 'estocada_lateral_derecha_adentro',
        "link": 'https://drive.usercontent.google.com/u/2/uc?id=1-LhOuNPR2yMFr-O5RqzeQOKhqviT7-Sa&export=download'
    },
    {
        "move": 'estocada_lateral_derecha_adentro_modificada',
        "link": 'https://drive.usercontent.google.com/u/2/uc?id=1a7ig-WkmyEOjJx9JLBl_P3p98Pu7Pvhf&export=download'
    },
    {
        "move": 'estocada_deslizamiento_lateral_izquierdo_afuera',
        "link": 'https://drive.usercontent.google.com/u/2/uc?id=1-JdHY2Uv2W97rDPKsu_LR-GIYMboM25J&export=download'
    },
    {
        "move": 'estocada_deslizamiento_lateral_izquierdo_adentro',
        "link": 'https://drive.usercontent.google.com/u/2/uc?id=1-J3ZxSKuvnWdryfTtW3CvaAIPn1Y4pUp&export=download'
    },
    {
        "move": 'estocada_deslizamiento_lateral_derecho_afuera',
        "link": 'https://drive.usercontent.google.com/u/2/uc?id=1-4DiuoMtx3B2D_K6IpJ0CHAhaqj54nvq&export=download'
    },
    {
        "move": 'estocada_deslizamiento_lateral_derecho_adentro',
        "link": 'https://drive.usercontent.google.com/u/2/uc?id=1PDkj-IXDmNHTxKb3YF0UAv0SVfX4gaz0&export=download'
    },
    {
        "move": 'estocada_derecha_afuera',
        "link": 'https://drive.usercontent.google.com/u/2/uc?id=1r4bru0stEwS8xDxlxUFmFXNxMWXjIa6C&export=download'
    },
    {
        "move": 'estocada_derecha_adentro',
        "link": 'https://drive.usercontent.google.com/u/2/uc?id=1-44cn8k74fM39iJ0QEkub9L8BgFrX6Td&export=download'
    },
    # {
    #     "move": 'escalón_derecho_afuera ',
    #     "link": 'https://drive.usercontent.google.com/u/2/uc?id=16H3Y4e810F5UzTT1KH1PY8vB6phUAHbH&export=download'
    # },
]

fines = []
fails = []
es = []
c = 0

for rec in records:
    
    try:
        IntegrateForcepalte(session_id = '7200e234-c206-4945-aa5a-980b0ec502cf',
                    trial_name = rec['move'] ,
                    force_gdrive_url = rec['link'])
        print(f"funciono {rec['move']}")
    except Exception as e:
        fails.append(rec['move'])
        
        if e not in es:
            es.append(f'{c}.{e}')
        # print(f"Error en {rec['move']} error {e}")
        c += 1

print('correctos: ',fines)
print('errores: ',fails)
print("type_error: ",len(es))
print('errors\n', es)