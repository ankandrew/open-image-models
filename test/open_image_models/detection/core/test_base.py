from math import isclose

import pytest

from open_image_models.detection.core.base import BoundingBox

# pylint: disable=too-many-positional-arguments


@pytest.mark.parametrize(
    "x1, y1, x2, y2, expected_width, expected_height, expected_area, expected_center",
    [
        (0, 0, 10, 10, 10, 10, 100, (5.0, 5.0)),
        (2, 3, 5, 7, 3, 4, 12, (3.5, 5.0)),
        (10, 10, 15, 13, 5, 3, 15, (12.5, 11.5)),
    ],
)
def test_bounding_box_properties(x1, y1, x2, y2, expected_width, expected_height, expected_area, expected_center):
    bbox = BoundingBox(x1, y1, x2, y2)
    assert bbox.width == expected_width
    assert bbox.height == expected_height
    assert bbox.area == expected_area

    actual_center = bbox.center
    assert isclose(actual_center[0], expected_center[0], rel_tol=1e-6)
    assert isclose(actual_center[1], expected_center[1], rel_tol=1e-6)


@pytest.mark.parametrize(
    "bbox, expected_area, expected_aspect_ratio, expected_empty",
    [
        (BoundingBox(0, 0, 10, 5), 50, 2.0, False),
        (BoundingBox(0, 0, 0, 5), 0, 0.0, True),
        (BoundingBox(10, 10, 5, 5), 0, 0.0, True),
    ],
)
def test_bounding_box_empty_properties(bbox, expected_area, expected_aspect_ratio, expected_empty):
    assert bbox.area == expected_area
    assert bbox.aspect_ratio == expected_aspect_ratio
    assert bbox.is_empty is expected_empty


def test_bounding_box_xyxy():
    assert BoundingBox(1, 2, 3, 4).xyxy == (1, 2, 3, 4)


@pytest.mark.parametrize(
    "x, y, width, height, expected",
    [
        (0, 0, 10, 20, BoundingBox(0, 0, 10, 20)),
        (5, 7, 3, 4, BoundingBox(5, 7, 8, 11)),
        (-2, -3, 5, 6, BoundingBox(-2, -3, 3, 3)),
    ],
)
def test_from_xywh(x, y, width, height, expected):
    assert BoundingBox.from_xywh(x, y, width, height) == expected


@pytest.mark.parametrize(
    "center_x, center_y, width, height, expected",
    [
        (5.0, 10.0, 10.0, 20.0, BoundingBox(0, 0, 10, 20)),
        (5.5, 6.5, 3.0, 5.0, BoundingBox(4, 4, 7, 9)),
        (5.25, 6.75, 2.5, 3.5, BoundingBox(4, 5, 7, 9)),
    ],
)
def test_from_cxcywh(center_x, center_y, width, height, expected):
    assert BoundingBox.from_cxcywh(center_x, center_y, width, height) == expected


@pytest.mark.parametrize(
    "bbox, expected_xywh",
    [
        (BoundingBox(0, 0, 10, 10), (0, 0, 10, 10)),
        (BoundingBox(2, 3, 5, 7), (2, 3, 3, 4)),
        (BoundingBox(10, 10, 15, 13), (10, 10, 5, 3)),
    ],
)
def test_to_xywh(bbox, expected_xywh):
    xywh = bbox.to_xywh()
    assert xywh == expected_xywh


@pytest.mark.parametrize(
    "bbox, expected_cxcywh",
    [
        (BoundingBox(0, 0, 10, 20), (5.0, 10.0, 10, 20)),
        (BoundingBox(2, 3, 5, 7), (3.5, 5.0, 3, 4)),
    ],
)
def test_to_cxcywh(bbox, expected_cxcywh):
    assert bbox.to_cxcywh() == expected_cxcywh


def test_as_slices():
    bbox = BoundingBox(10, 20, 30, 40)
    assert bbox.as_slices() == (slice(20, 40), slice(10, 30))


@pytest.mark.parametrize(
    "bbox1, bbox2, expected_intersection",
    [
        # Overlapping case
        (
            BoundingBox(0, 0, 10, 10),
            BoundingBox(5, 5, 15, 15),
            BoundingBox(5, 5, 10, 10),
        ),
        # One box completely inside another
        (
            BoundingBox(0, 0, 10, 10),
            BoundingBox(2, 2, 5, 5),
            BoundingBox(2, 2, 5, 5),
        ),
        # Touching edges (should return None if there's no positive overlap)
        (
            BoundingBox(0, 0, 10, 10),
            BoundingBox(10, 10, 12, 12),
            None,
        ),
        # No overlap at all
        (
            BoundingBox(0, 0, 5, 5),
            BoundingBox(6, 6, 10, 10),
            None,
        ),
    ],
)
def test_intersection(bbox1, bbox2, expected_intersection):
    inter = bbox1.intersection(bbox2)
    assert inter == expected_intersection


@pytest.mark.parametrize(
    "bbox1, bbox2, expected",
    [
        (BoundingBox(0, 0, 10, 10), BoundingBox(5, 5, 15, 15), True),
        (BoundingBox(0, 0, 10, 10), BoundingBox(10, 0, 20, 10), False),
        (BoundingBox(0, 0, 10, 10), BoundingBox(11, 0, 20, 10), False),
    ],
)
def test_intersects(bbox1, bbox2, expected):
    assert bbox1.intersects(bbox2) is expected


@pytest.mark.parametrize(
    "bbox, x, y, expected",
    [
        (BoundingBox(0, 0, 10, 10), 0, 0, True),
        (BoundingBox(0, 0, 10, 10), 9.5, 9.5, True),
        (BoundingBox(0, 0, 10, 10), 10, 5, True),
        (BoundingBox(0, 0, 10, 10), 10.1, 5, False),
        (BoundingBox(0, 0, 10, 10), -1, 5, False),
        (BoundingBox(0, 0, 0, 10), 0, 5, False),
    ],
)
def test_contains_point(bbox, x, y, expected):
    assert bbox.contains_point(x, y) is expected


@pytest.mark.parametrize(
    "bbox, other, expected",
    [
        (BoundingBox(0, 0, 10, 10), BoundingBox(2, 2, 8, 8), True),
        (BoundingBox(0, 0, 10, 10), BoundingBox(0, 0, 10, 10), True),
        (BoundingBox(0, 0, 10, 10), BoundingBox(5, 5, 15, 15), False),
        (BoundingBox(0, 0, 10, 10), BoundingBox(5, 5, 5, 8), False),
    ],
)
def test_contains(bbox, other, expected):
    assert bbox.contains(other) is expected


@pytest.mark.parametrize(
    "bbox1, bbox2, expected_iou",
    [
        # Same box, IoU should equal 1.0
        (
            BoundingBox(0, 0, 10, 10),
            BoundingBox(0, 0, 10, 10),
            1.0,
        ),
        # Partial overlap
        (
            BoundingBox(0, 0, 10, 10),
            BoundingBox(5, 5, 15, 15),
            25 / 175,  # intersection=25, union=175, so IoU should equal 25/175
        ),
        # No overlap, IoU should equal 0.0
        (
            BoundingBox(0, 0, 5, 5),
            BoundingBox(6, 6, 10, 10),
            0.0,
        ),
        # One box inside another, ratio of areas
        (
            BoundingBox(0, 0, 10, 10),
            BoundingBox(2, 2, 8, 8),
            36 / 100,
        ),
    ],
)
def test_iou(bbox1, bbox2, expected_iou):
    iou_value = bbox1.iou(bbox2)
    assert isclose(iou_value, expected_iou, rel_tol=1e-5)


@pytest.mark.parametrize(
    "bbox, other, expected",
    [
        (BoundingBox(0, 0, 10, 10), BoundingBox(5, 5, 15, 15), 0.25),
        (BoundingBox(5, 5, 15, 15), BoundingBox(0, 0, 20, 20), 1.0),
        (BoundingBox(0, 0, 5, 5), BoundingBox(6, 6, 10, 10), 0.0),
        (BoundingBox(0, 0, 0, 5), BoundingBox(0, 0, 5, 5), 0.0),
    ],
)
def test_intersection_over_area(bbox, other, expected):
    assert isclose(bbox.intersection_over_area(other), expected, rel_tol=1e-5)


@pytest.mark.parametrize(
    "bbox, other, expected",
    [
        (BoundingBox(0, 0, 10, 10), BoundingBox(5, 5, 15, 20), BoundingBox(0, 0, 15, 20)),
        (BoundingBox(5, 5, 10, 10), BoundingBox(0, 0, 20, 20), BoundingBox(0, 0, 20, 20)),
        (BoundingBox(-5, -2, 0, 3), BoundingBox(1, 2, 4, 6), BoundingBox(-5, -2, 4, 6)),
    ],
)
def test_enclosing(bbox, other, expected):
    assert bbox.enclosing(other) == expected


def test_bounding_box_iter():
    bbox = BoundingBox(10, 20, 30, 40)
    assert tuple(bbox) == (10, 20, 30, 40)


@pytest.mark.parametrize(
    "bbox, max_width, max_height, expected",
    [
        # Some coords are below 0, should be clamped to 0
        (BoundingBox(-10, -10, 50, 50), 100, 100, BoundingBox(0, 0, 50, 50)),
        # Coords (x2, y2) exceed the frame size, should be clamped to the frame dimension
        (BoundingBox(10, 10, 150, 150), 100, 100, BoundingBox(10, 10, 100, 100)),
        # Both previous cases
        (BoundingBox(-10, 10, 150, 90), 120, 100, BoundingBox(0, 10, 120, 90)),
        # Valid bbox, should remain unchanged
        (BoundingBox(10, 10, 90, 90), 100, 100, BoundingBox(10, 10, 90, 90)),
    ],
)
def test_bounding_box_clamp(bbox, max_width, max_height, expected):
    assert bbox.clamp(max_width, max_height) == expected


@pytest.mark.parametrize(
    "bbox, frame_width, frame_height, expected_inside",
    [
        (BoundingBox(10, 20, 30, 40), 50, 50, True),
        (BoundingBox(0, 0, 50, 50), 50, 50, True),
        (BoundingBox(-1, 0, 30, 40), 50, 50, False),
        (BoundingBox(10, 20, 60, 40), 50, 50, False),
        (BoundingBox(10, 20, 5, 40), 50, 50, True),
    ],
)
def test_bounding_box_is_inside(bbox, frame_width, frame_height, expected_inside):
    assert bbox.is_inside(frame_width, frame_height) is expected_inside


@pytest.mark.parametrize(
    "bbox, frame_width, frame_height, expected_valid",
    [
        # Valid bbox inside frame
        (BoundingBox(10, 20, 30, 40), 50, 50, True),
        # x1 >= x2
        (BoundingBox(10, 20, 5, 40), 50, 50, False),
        # x2 exceeds frame width
        (BoundingBox(10, 20, 30, 40), 25, 50, False),
        # y2 exceeds frame height
        (BoundingBox(10, 20, 30, 40), 50, 35, False),
        # Exactly fills the frame
        (BoundingBox(0, 0, 50, 50), 50, 50, True),
        # Negative coord
        (BoundingBox(-1, 0, 30, 40), 50, 50, False),
        # x1 equals x2
        (BoundingBox(10, 20, 10, 40), 50, 50, False),
        # y1 equals y2
        (BoundingBox(10, 20, 30, 20), 50, 50, False),
    ],
)
def test_bounding_box_is_valid(bbox, frame_width, frame_height, expected_valid):
    assert bbox.is_valid(frame_width, frame_height) == expected_valid
