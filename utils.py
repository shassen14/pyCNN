import os

###############################################################################
def get_folder_path(folder: str) -> str:
    folder_path = os.path.join(os.path.dirname(__file__), folder)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


###############################################################################
def get_file_path(folder: str, file: str) -> str:
    file_path = os.path.join(get_folder_path(folder), file)
    return file_path