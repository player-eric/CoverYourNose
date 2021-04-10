from typing import List


class BoundingBox:
    """
    A utility class for bounding boxes.
    """

    def __init__(self, name, x1, y1, x2, y2, clip=None):
        """
        :param name: a name to identify the purpose of this bounding box in __repr__
        :param x1: the top left x coordinate
        :param y1: the top left y coordinate
        :param x2: the bottom right x coordinate
        :param y2: the bottom right y coordinate
        :param clip: an optional (img_width_px, img_height_px) tuple used to, if need be, constrain this rectangle
        """
        self.name = name
        self.class_id = None
        self.conf = None

        if clip is not None:
            # clip the coordinate, avoid the value exceed the image boundary.
            width, height = clip
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(x2, width)
            y2 = min(y2, height)

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def set_class_id(self, class_id):
        """
        Records the class id of this bounding box.
        """
        self.class_id = class_id

    def set_confidence(self, conf):
        """
        Records the confidence associated with this bounding box.
        """
        self.conf = conf

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def top_left(self):
        return self.x1, self.y1

    @property
    def bottom_right(self):
        return self.x2, self.y2

    @property
    def center(self):
        hw, hh = self.halves
        return self.x1 + hw, self.y1 + hh

    @property
    def halves(self):
        return self.width // 2, self.height // 2

    def intersects(self, other: 'BoundingBox'):
        return not (
                self.x1 > other.x2 or
                self.x2 < other.y1 or
                self.y1 > other.y2 or
                self.y2 < other.y1
        )

    def intersects_any(self, others: List['BoundingBox']):
        for o in others:
            if self.intersects(o):
                return True
        return False

    def contains(self, other: 'BoundingBox'):
        return (
                self.x1 <= other.x1 and
                self.y1 <= other.y1 and
                self.x2 >= other.x2 and
                self.y2 >= other.y2
        )

    def contains_any(self, others: List['BoundingBox']):
        for o in others:
            if self.contains(o):
                return True
        return False

    def crop(self, input_image):
        return input_image[self.x1:self.x2, self.y1:self.y2]

    def __repr__(self):
        return f"@{self.name}: TL ({self.x1}, {self.y1}) => BR ({self.x2}, {self.y2}) [{self.width} x {self.height}]"


def bbox_from_two_points(name, x1, y1, x2, y2, clip=None):
    """
    Instantiates a BoundingBox from the given two points.
    See BoundingBox docstring for other param description.
    """
    return BoundingBox(name, x1, y1, x2, y2, clip)


def bbox_from_anchor_dims(name, x, y, w, h, clip=None):
    """
    Instantiates a BoundingBox from a top left anchor point and the rectangle's desired width and height.
    See BoundingBox docstring for other param description.
    """
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return BoundingBox(name, x1, y1, x2, y2, clip)
