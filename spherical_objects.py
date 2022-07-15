from dataclasses import dataclass
from lib2to3.pytree import convert
import math
from typing import List, Tuple

# Lens constants assuming a 1080p image
f = 714.285714
center = [960, 540]
D = 1.082984  # For image-1, switch to 0.871413 for image-2


def cartesian2sphere(pt):
    x = (pt[0] - center[0]) / f
    y = (pt[1] - center[1]) / f

    r = math.sqrt(x * x + y * y)
    if r != 0:
        x /= r
        y /= r
    r *= D
    sin_theta = math.sin(r)
    x *= sin_theta
    y *= sin_theta
    z = math.cos(r)

    return [x, y, z]


def sphere2cartesian(pt):
    r = math.acos(pt[2])
    r /= D
    if pt[2] != 1:
        r /= math.sqrt(1 - pt[2] * pt[2])
    x = r * pt[0] * f + center[0]
    y = r * pt[1] * f + center[1]
    return [x, y]


def convert_point(point: List[int]) -> List[int]:
    """Convert single points between Cartesian and spherical coordinate systems"""
    if len(point) == 2:
        return cartesian2sphere(point)
    elif len(point) == 3:
        return sphere2cartesian(point)
    else:
        raise ValueError(f'Expected point to be 2 or 3D, got {len(point)} dimensions')


class CartesianBbox:

    def __init__(self, points: List[int], fmt: str):
        assert fmt in ['xyxy', 'xywh', 'cxcywh'], 'Invalid bbox format'
        assert len(points) == 4, 'Cartesian bbox must have 4 values'
        self.points = points
        self.fmt = fmt


@dataclass
class SphericalBbox:
    """
    Class representing a Spherical bounding box.
    The bbox parametrization is slightly based in CGNS:
    https://cgns.github.io/ProposedExtensions/CPEX-0042-boundingbox-v2.pdf

    Attributes
    points: List containing upper left and upper rigth corners of bbox
        expressed in spherical coordinates as [x1,y1,z1,x2,y2,z2].
    POINTS_LEN: Constant to validate the length of the new points.
    """
    POINTS_LEN = 6
    points: List[float]

    def __post_init__(self):
        assert len(self.points) == self.POINTS_LEN, \
            'Spherical bbox must have 6 values'


def _bbox_to_spherical(bbox: List[int]) -> SphericalBbox:

    upper_left = convert_point(bbox[:2])
    lower_right = convert_point(bbox[2:])

    return SphericalBbox([*upper_left, *lower_right])


def xywh_to_xxyy(points: List[int]) -> List[int]:
    x, y, w, h = points
    return x, y, x + w, y + h


def cxcywh_to_xxyy(points: List[int]) -> List[int]:
    cx, cy, w, h = points
    w, h = w // 2, h // 2
    return cx - w, cy - h, cx + w, cy + h


def bbox_to_spherical(cartesian: CartesianBbox) -> SphericalBbox:
    """
    Converts the current cartesian bbox format to xxyy
    and transforms it to spherical coordinates.

    Args:
        cartesian: CartesianBbox object. 
    Returns:
        Spherical bounding box representation as SphericalBbox object.
    """
    if cartesian.fmt is 'xywh':
        bbox_xxyy = xywh_to_xxyy(cartesian.points)
    elif cartesian.fmt is 'cxcywh':
        bbox_xxyy = cxcywh_to_xxyy(cartesian.points)
    else:
        bbox_xxyy = cartesian.points
    spherical = _bbox_to_spherical(bbox_xxyy)

    return spherical


@dataclass
class CartesianPolygon:
    """
    Class to represent a polygon in cartesian coordinates. Adopting opencv
    convention for the contour points. The polygon is a set of at least 3
    points which do not lie on the same line. This class only validates the
    existence of more than 2 points in the list. Other validations must be
    made by the user before initializing.

    Attributes:
        contour: List containing tuples of x,y coordinates of the polygon
                 boundary. This attribute follows the same convention as opencv
                 contours. More info in http://pdf.xuebalib.com:1262/xuebalib.com.17233.pdf
                 (OpenCV contour retrieval method is based on this paper)
        MIN_POLY_LEN: Constant to validate the length of the contour.
    """
    MIN_POLY_LEN = 3
    contour: List[Tuple[int, int]]

    def __post_init__(self):
        assert len(self.contour) >= self.MIN_POLY_LEN, \
            'We need at least 3 points to form a polygon'
        self.num_vertex = len(self.contour)


@dataclass
class SphericalPolygon:
    """
    Class to represent a polygon in spherical coordinates.

    Attributes:
        contour: List containing tuples of x,y,z for each point of the polygon
                 boundary in spherical coordinates.
        MIN_POLY_LEN: Constant to validate the length of the contour.
    """
    MIN_POLY_LEN = 3
    contour: List[Tuple[float, float, float]]

    def __post_init__(self):
        assert len(self.contour) >= self.MIN_POLY_LEN, \
            'We need at least 3 points to form a polygon'
        self.num_vertex = len(self.contour)


def polygon_to_spherical(cartesian: CartesianPolygon) -> SphericalPolygon:
    """
    Converts each point of cartesian polygon into spherical coordinates.

    Args:
        cartesian: CartesianPolygon object. 
    Returns:
        Spherical representation of the polygon points as SphericalPolygon object.
    """
    spherical_contour = [convert_point(point) for point in cartesian.contour]

    return SphericalPolygon(spherical_contour)


def main():

    # Test for bounding box convertion:
    # I create the same bbox but with the 3 different formats
    bbox1 = [CartesianBbox([100, 150, 150, 200], fmt='xyxy'),
             CartesianBbox([100, 150, 50, 50], fmt='xywh'),
             CartesianBbox([125, 175, 50, 50], fmt='cxcywh')]
    # As we can see in the result the spherical representation is the same
    for bbox in bbox1:
        print(bbox_to_spherical(bbox).points)

    # Test for polygons:
    # Creating a contour and a cartesian polygon instance
    contour = [(100, 100), (80, 150), (50, 75), (90, 25)]
    cart_poly = CartesianPolygon(contour)

    sphr_poly = polygon_to_spherical(cart_poly)
    print(sphr_poly.contour)
    return


if __name__ == "__main__":

    main()
