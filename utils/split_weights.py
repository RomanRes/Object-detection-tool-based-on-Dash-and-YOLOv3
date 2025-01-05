import os


def split_file(file_path, chunk_size=25 * 1024 * 1024):
    """
    Teilt eine Datei in kleinere Teile.
    :param file_path: Pfad zur Originaldatei
    :param chunk_size: Größe jedes Teils in Bytes (25 MB Standard)
    """
    file_name = os.path.basename(file_path)
    output_dir = f"{file_path}_parts"
    os.makedirs(output_dir, exist_ok=True)

    with open(file_path, "rb") as f:
        part_num = 0
        while chunk := f.read(chunk_size):
            part_file_name = os.path.join(output_dir, f"{file_name}.part{part_num}")
            with open(part_file_name, "wb") as part_file:
                part_file.write(chunk)
            part_num += 1
    print(f"Datei aufgeteilt in {part_num} Teile.")
    return output_dir


split_file("D:\Python_projects\YOLOv3_dash\yolov3.weights")