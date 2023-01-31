import plotly.graph_objects as go

from detection.detection import predict_boxes
from itertools import cycle
from parameters.parameters import ANCHORS, IMG_SIZE


def img_to_plotly_fig(image, image_w, image_h, class_threshold=0.5, nms_thresh=0.5):
    """
    This function produced neu new graphic object of class 'Figure'

    1. predict boxes for an image by calling predict_boxes()
    2. formed new graphic object with image and add predicted boxes

    :param contents: string with image content base64 coded
    :param class_threshold: float
    :param nms_thresh: float
    :return:  object of class 'Figure'

    """
    # real width and height of image before resizing
    #img_width, img_height = img.size

    # detecting objects on the image
    boxes = predict_boxes(image, class_threshold, nms_thresh, ANCHORS, IMG_SIZE)

    # this list need for mapping colors to each class
    # list of found classes
    labels = [box.label for box in boxes]

    # create new objects of class 'Figure'
    fig = go.Figure()

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(x=[0, image_w],
                   y=[0, image_h],
                   mode="markers",
                   marker_opacity=0,
                   showlegend=False)
    )

    # Add image
    fig.add_layout_image(
        dict(
            source=image,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=image_w,
            sizey=image_h,
            opacity=1,
            layer="below"),
    )

    # Configure axes
    fig.update_xaxes(
        visible=False,
        showgrid=False,
        range=[0, image_w])

    fig.update_yaxes(
        visible=False,
        showgrid=False,
        scaleanchor="x",
        range=[image_h, 0])

    # change template
    fig.update_layout(template="ggplot2")

    # colors for boxes
    colors = ('#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
              '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52')

    # tuple of classes found in the picture
    exist_classes = (set(labels))

    # matching classes and colors
    color_map = dict(list(x) for x in zip(exist_classes, cycle(colors)))

    # add all boxes and annotation to the fig
    for i, box in enumerate(boxes):
        fig.add_trace(go.Scatter(
            x=[box.xmin, box.xmin, box.xmax, box.xmax, box.xmin],
            y=[box.ymin, box.ymax, box.ymax, box.ymin, box.ymin],
            mode="lines+text",
            fill="toself",
            opacity=0.7,
            marker_color=color_map[box.label],
            hoveron="fills",
            name=box.label,
            hoverlabel_namelength=0,
            showlegend=False
        ))

        fig.add_annotation(x=box.xmin, y=box.ymin,
                           text=f"{box.label} {box.score:.2f}",
                           showarrow=True,
                           arrowhead=1,
                           bordercolor=color_map[box.label],
                           bgcolor=color_map[box.label])

    # remove some options from the modebar
    fig.update_layout(modebar_remove=["select"])

    return fig
