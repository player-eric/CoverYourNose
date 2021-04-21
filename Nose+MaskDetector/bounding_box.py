from typing import List, Tuple, Any, Optional


class BoundingBox:
    """
    A utility class for bounding boxes.
    """

    def __init__(self, name: str, x1: int, y1: int, x2: int, y2: int, clip: Optional[Tuple[int, int]] = None):
        """
        :param name: a name to identify the purpose of this bounding box in __repr__
        :param x1: the top left x coordinate
        :param y1: the top left y coordinate
        :param x2: the bottom right x coordinate
        :param y2: the bottom right y coordinate
        :param clip: an optional (img_width_px, img_height_px) tuple used to, if need be, constrain this rectangle
        """
        self.name = name
        self.dict = {}

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

    def set(self, attribute: str, value: Any) -> None:
        """
        Records arbitrary property of this bounding box.
        """
        self.dict[attribute] = value

    def get(self, attribute: str) -> Any:
        """
        Retrieve arbitrary property of this bounding box.
        """
        return self.dict[attribute]

    @property
    def points(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def top_left(self) -> Tuple[int, int]:
        return self.x1, self.y1

    @property
    def bottom_right(self) -> Tuple[int, int]:
        return self.x2, self.y2

    @property
    def center(self) -> Tuple[int, int]:
        hw, hh = self.halves
        return self.x1 + hw, self.y1 + hh

    @property
    def halves(self) -> Tuple[int, int]:
        return self.width // 2, self.height // 2

    def intersects_one(self, other: 'BoundingBox') -> bool:
        return not (
                self.x1 > other.x2 or
                self.x2 < other.x1 or
                self.y1 > other.y2 or
                self.y2 < other.y1
        )

    def intersects_many(self, others: List['BoundingBox']) -> List[bool]:
        return [self.intersects_one(o) for o in others]

    def contains_one(self, other: 'BoundingBox') -> bool:
        return (
                self.x1 <= other.x1 and
                self.y1 <= other.y1 and
                self.x2 >= other.x2 and
                self.y2 >= other.y2
        )

    def contains_many(self, others: List['BoundingBox']) -> List[bool]:
        return [self.contains_one(o) for o in others]

    def crop(self, input_image):
        return input_image[self.y1:self.y2, self.x1:self.x2]

    def to_global_coordinates(self, child: 'BoundingBox') -> None:
        child.x1 += self.x1
        child.y1 += self.y1
        child.x2 += self.x1
        child.y2 += self.y1

    def __repr__(self):
        return f"@{self.name}: TL ({self.x1}, {self.y1}) => BR ({self.x2}, {self.y2}) [{self.width} x {self.height}]"


def bbox_from_two_points(name: str, x1: int, y1: int, x2: int, y2: int, clip: Optional[Tuple[int, int]] = None):
    """
    Instantiates a BoundingBox from the given two points.
    See BoundingBox docstring for other param description.
    """
    return BoundingBox(name, x1, y1, x2, y2, clip)


def bbox_from_anchor_dims(name: str, x: int, y: int, w: int, h: int, clip: Optional[Tuple[int, int]] = None):
    """
    Instantiates a BoundingBox from a top left anchor point and the rectangle's desired width and height.
    See BoundingBox docstring for other param description.
    """
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return BoundingBox(name, x1, y1, x2, y2, clip)


def convert_to_global(name: str, positions: List[Tuple[int, int, int, int]], mask_box: BoundingBox, width: int,
                      height: int):
    """
    Convert all of the x, y, w, h positions, where x, y are in local
    coordinates, to BoundingBox instances with global image coordinates.
    """
    bboxes = []
    for x, y, w, h in positions:
        bbox = bbox_from_anchor_dims(name, x, y, w, h, (width, height))
        mask_box.to_global_coordinates(bbox)
        bboxes.append(bbox)
    assert all(mask_box.contains_many(bboxes))
    return bboxes


if __name__ == "__main__":
    a = BoundingBox("a", 50, 50, 250, 250)

    b = BoundingBox("b", 0, 0, 50, 50)
    # Intersection returns true even if intersection is only a single line (just borders overlap)
    assert a.intersects_one(b) and b.intersects_one(a)
    assert not (a.contains_one(b) or b.contains_one(a))

    b = BoundingBox("b", 0, 0, 49, 49)
    assert not (a.intersects_one(b) or b.intersects_one(a))
    assert not (a.contains_one(b) or b.contains_one(a))

    b = BoundingBox("b", 0, 75, 75, 100)
    # Left intersection
    assert a.intersects_one(b) and b.intersects_one(a)
    assert not (a.contains_one(b) or b.contains_one(a))

    b = BoundingBox("b", 100, 75, 300, 200)
    # Right intersection
    assert a.intersects_one(b) and b.intersects_one(a)
    assert not (a.contains_one(b) or b.contains_one(a))

    b = BoundingBox("b", 75, 0, 200, 200)
    # Top intersection
    assert a.intersects_one(b) and b.intersects_one(a)
    assert not (a.contains_one(b) or b.contains_one(a))

    b = BoundingBox("b", 75, 200, 200, 300)
    # Bottom intersection
    assert a.intersects_one(b) and b.intersects_one(a)
    assert not (a.contains_one(b) or b.contains_one(a))

    b = BoundingBox("b", 50, 50, 250, 250)
    # When a and b are the same box
    assert a.intersects_one(b) and b.intersects_one(a)
    assert a.contains_one(b) and b.contains_one(a)

    b = BoundingBox("b", 51, 51, 249, 249)
    # When b is a true child of a
    assert a.intersects_one(b) and b.intersects_one(a)
    assert a.contains_one(b) and not b.contains_one(a)
