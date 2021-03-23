import os
import hashlib


def get_project_dir():
    file_path = os.path.abspath(__file__)  # <project_dir>/src/utility.py
    file_dir = os.path.dirname(file_path)  # <project_dir>/src
    return os.path.dirname(file_dir)  # <project_dir>


def remove_project_dir(path):
    return path.replace(get_project_dir(), "")[1:]


def get_hashsum_of_file(path_to_file):
    buffer = 64*1024  # 64k bytes
    sha256 = hashlib.sha256()
    with open(path_to_file, "rb") as f:
        while True:
            data = f.read(buffer)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def draw_bbox(path2image, bounding_box, color=(255, 255, 0), thickness=3):
    """    
    A simple function that draws a bounding box from a dataframe using the cv2.rectangle() method.
    ----------------------------------------------------------------------------------------------  
    Atributes:
    ---------
      path2image : str
        path to the image         
      bounding_box : list
        bounding box [x1, y1, x2, y2]
      color : tuple
        RGB code for the box line, default color is yellow (255, 255, 0) 
      thickness : int
        line thickness
    
    Methods:
    -------
       cv2.rectangle()
        draw a rectangle with a chosen color line borders of a chosen thickness 
    """                      
    start_point = (bounding_box[0], bounding_box[1]) # top left corner
    end_point = (bounding_box[2], bounding_box[3])   # botton right corner

    image = cv2.rectangle(path2image, start_point, end_point, color, thickness) 
    plt.imshow(image)

    return
