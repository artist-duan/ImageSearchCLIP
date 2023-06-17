import os


def traverse_path(root_path: str, extensions: set, complete=True) -> list:
    files = []
    if not os.path.exists(root_path):
        return files

    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension in extensions:
                image_path = os.path.join(dirpath, filename) if complete else filename
                files.append(image_path)
    return files
