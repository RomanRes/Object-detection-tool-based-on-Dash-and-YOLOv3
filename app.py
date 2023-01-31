import dash_bootstrap_components as dbc
import dash

from dash import dcc, html
from utils.plot import img_to_plotly_fig
from dash.dependencies import Input, Output, State
from utils.loadimage import load_image_pixels

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.LITERA],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

app.layout = html.Div([

    # header row
    dbc.Row(
        dbc.Col([html.H3("Dash detection tool based on  YOLOv3")],
                width={"size": 6, "offset": 1},
                ),
        align="center",
    ),

    # The row with a main figure to display the image and the boxes
    dbc.Row(
        [dbc.Col(
            dbc.Spinner(
                dcc.Graph(id="graph_figure",
                          figure={},
                          style={'width': '90vw', 'height': '75vh'}
                          )
            ),
        ),
        ], align="center"),

    # The row contains two sliders for choosing IoU and Confidence thresholds and submit bottom
    dbc.Row([
        dbc.Col([
            html.H6('Non-maximum Suppression (IoU) threshold', style={'textAlign': 'center'}),
            dcc.Slider(
                id="nms_thresh",
                min=0,
                max=1,
                marks={i / 10: str(i / 10) for i in range(0, 10)},
                value=0.5,
            ),
        ], width=4),
        dbc.Col([
            html.H6('Confidence threshold', style={'textAlign': 'center'}),
            dcc.Slider(
                id="class_threshold",
                min=0,
                max=1,
                marks={i / 10: str(i / 10) for i in range(0, 10)},
                value=0.5,
            ),
        ], width=4),
        dbc.Col([
            dbc.Button("SUBMIT",
                       id="submit_button",
                       n_clicks=0,
                       color="primary", className="d-grid gap-2 col-6 mx-auto"),
        ], width=4),
    ], align="center"),

    # upload box
    dbc.Row([
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["Drag and drop or click to select a file to upload."]
            ),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=False,
        ),
    ])
])


# this callback provides updating figure bei  click on the "submit_button"
@app.callback(Output("graph_figure", 'figure'),
              [Input("submit_button", 'n_clicks')],
              [State('upload-data', 'contents'),
               State("class_threshold", 'value'),
               State("nms_thresh", 'value')])
def updateTable(n_clicks, contents, class_threshold, nms_thresh):
    """
    This function will be
    :param n_clicks: int: this
    :param contents: is a base64 encoded string that contains the files contents
    :param class_threshold:
    :param nms_thresh:
    :return:

    """
    image = load_image_pixels(contents)
    image_w, image_h = image.size
    return img_to_plotly_fig(image, image_w, image_h, class_threshold=class_threshold, nms_thresh=nms_thresh)


if __name__ == '__main__':
    app.run_server(debug=False)
