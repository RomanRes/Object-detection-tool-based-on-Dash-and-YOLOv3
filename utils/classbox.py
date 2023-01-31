import numpy as np
from parameters.parameters import LABELS


class BoundBox:

    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.labels = LABELS
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = self.labels[np.argmax(self.classes)]
        self.score = np.max(self.classes)
