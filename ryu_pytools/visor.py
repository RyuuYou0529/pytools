import os

IMG_NAME = 'image.tif'
DB_NAME = 'database.db'

RM009 = {
    'axon': {
        'num': 2,
    },
    'arbor': {
        'num': 1
    },
    'noisy': {
        'num': 2
    }
}

Z002 = {
    'dendrites': {
        'num': 1
    }
}

METADATA = {
    'RM009': RM009,
    'Z002': Z002
}

def get_path(brain:str, scenario:str, index:int, root_path:str='', *, IMG_NAME:str='image.tif', DB_NAME:str='database.db'):
    metadata = METADATA[brain][scenario]
    assert index>0 and index<=metadata['num'], 'invalid order number'
    img_path = os.path.join(root_path, brain, 'data', f'{scenario}_{index}', IMG_NAME)
    db_path = os.path.join(root_path, brain, 'data', f'{scenario}_{index}', DB_NAME)
    return img_path, db_path

def get_metadata():
    return METADATA