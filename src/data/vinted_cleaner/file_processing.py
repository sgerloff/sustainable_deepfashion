import os

def get_all_images_path(image_dir):

    return [os.path.join(image_dir, path) for path in os.listdir(image_dir)]

def get_all_directories_path(directory_dir):

    return [os.path.join(directory_dir, path) for path in os.listdir(directory_dir)]

def delete_images(image_paths):
    """
    This function will delete our detected images to the desired location.
    :param image_paths: A list containing the paths to every images detected
    :return: Nothing
    """

    for path in image_paths:
        os.remove(path)