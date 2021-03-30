import datetime

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

import base64, os
from src.utility import get_project_dir

import numpy as np
import pandas as pd

from src.dash_app.inference import ModelInference, distance

from sklearn.decomposition import PCA


basename = "simple_conv2d_embedding_size_32-1"
model = ModelInference(f"{basename}.meta")
prediction_csv_path = os.path.join(get_project_dir(), "data", "processed", f"{basename}_predictions.csv")
prediction_df = pd.read_csv(prediction_csv_path)
prediction_df["prediction"] = prediction_df["prediction"].apply(lambda x: np.array(eval(x)).flatten())

NUMBER_OF_BEST_PREDICTIONS = 10

NUMBER_OF_PCA_SLIDERS = 10
pca = PCA(n_components=NUMBER_OF_PCA_SLIDERS)
pca.fit(prediction_df["prediction"].tolist())
pca_components = pca.components_

slider_input = []
for i in range(NUMBER_OF_PCA_SLIDERS):
    slider_input.append(Input(f"pca_slider_{i}", "value"))

external_stylesheets = ["https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def build_layout():
    children = []
    children.append(
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            multiple=False
        )
    )
    for i in range(NUMBER_OF_PCA_SLIDERS):
        children.append(
            dcc.Slider(
                id=f"pca_slider_{i}",
                min=-1., max=1., step=0.01, value=0.0
            )
        )
    children.append(
        html.Div(id='output-image-upload')
    )
    return children


app.layout = html.Div(build_layout())


def parse_contents(contents, values):
    embedding = model.predict(contents)

    values = np.array(values).flatten()
    embedding = embedding + np.dot(values, pca_components)
    prediction_df["distance"] = prediction_df["prediction"].apply(
        lambda x: distance(x, embedding, metric=model.get_metric())
    )
    top_5_pred = prediction_df.sort_values(by="distance", ascending=True)["image"].head(
        NUMBER_OF_BEST_PREDICTIONS).to_list()

    top_5_pred_base64 = [base64.b64encode(open(file, 'rb').read()).decode() for file in top_5_pred]

    children = [
        html.Img(src=contents, width=500),
        html.Hr(),
        html.Div('Embedding Vector:'),
        html.Pre(f"{embedding[0]}"),
    ]
    for i in range(NUMBER_OF_BEST_PREDICTIONS):
        children.append(
            html.Img(id="prediction-image", src='data:image/jpeg;base64,{}'.format(top_5_pred_base64[i]), height=300)
        )

    return html.Div(children=children)


@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              slider_input)
def update_output(contents, *values):
    if contents is not None:
        children = [
            parse_contents(contents, values)
        ]
        return children


if __name__ == '__main__':
    app.run_server(debug=True)
