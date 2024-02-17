import os

STORAGE_DIR = os.path.join('project', 'storage')
RESOURCES_DIR = os.path.join(STORAGE_DIR, 'resources')


def experiment_path(module_path: str, name: str):
    return os.path.join('project', module_path, name)
