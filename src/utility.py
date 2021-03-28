import cv2
import os
import hashlib
import matplotlib.pyplot as plt
import plotly.express as px


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

def savely_unfreeze_layers_of_model(model, ratio):
    """
    Freezes the first 100*ratio percent of the layers
    of the model. The remaining layers are set to
    trainable. BatchNormalization layers are frozen to
    prevent loss of pretrained weights.
    """

    model.trainable = True

    ratio_index = int(ratio * len(model.layers))

    for layer in model.layers[:ratio_index]:
        layer.trainable = False
    for layer in model.layers[ratio_index:]:
        if layer.__class__.__name__ == "BatchNormalization":
            layer.trainable = False
        else:
            layer.trainable = True
    return model


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


def plot_bar(df, x, y, color, title, width=900, height=600, showlegend=False):
    '''
    Creates a bar plot with plotly
    '''
    fig = px.bar(df, 
                 x=x,
                 y=y, 
               #  title=title, 
                 width=width, 
                 height=height, 
                 color=color
    )
    fig.update_layout(
        title=dict(
        text=title,
        x=0.5,
        y=0.95,
        xanchor='center',
        yanchor='top',
    ),
    xaxis_title=x,
    yaxis_title=y,
    showlegend=False,
    legend_title="Legend",
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="RebeccaPurple",
    )
    )
    fig.show()


def plot_donut(df_col, labels, fig_title, plt, text_color='black', 
               font_size=16.0, explode=(0.02, 0.02, 0.02), 
               circle_radius=0.5, title_color='blue'):
    '''
    Creates a donut plot.
    '''
    labels = labels
    explode = explode
    plt.rcParams['text.color'] = text_color
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.titlecolor'] = title_color

    theme = plt.get_cmap('PiYG')
    colors =[theme(1. * i / len(labels)) for i in range(len(labels))]

    plt.pie(df_col.value_counts(), 
            labels=labels, 
            labeldistance=1.2,
            explode=explode, 
            shadow=True, 
            wedgeprops = {'linewidth': 15},
            pctdistance=0.7,  # position of % numbers
            autopct="%.1f%%",
            colors=colors
    )

    # add a circle at the center to transform it in a donut chart
    my_circle = plt.Circle( (0,0), circle_radius, color='white')
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    
    plt.title(fig_title, loc='center', y=1.1)
    plt.axis('equal')
    plt.tight_layout()
    #plt.show()