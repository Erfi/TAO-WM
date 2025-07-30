from pathlib import Path


def get_file_list(data_dir, extension=".npz", sort_list=False):
    """retrieve a list of files inside a folder"""
    dir_path = Path(data_dir)
    dir_path = dir_path.expanduser()
    assert dir_path.is_dir(), f"{data_dir} is not a valid dir path"
    file_list = []
    for x in dir_path.iterdir():
        if x.is_file() and extension in x.suffix:
            file_list.append(x)
        elif x.is_dir():
            file_list.extend(get_file_list(x, extension))
    if sort_list:
        file_list = sorted(file_list, key=lambda file: file.name)
    return file_list
