# Spherical Objects

Bounding boxes and polygons

## Motivation/Background

Fish eye lenses are useful for capturing large field of views from a static position. However, the optical distortion characteristics can vary widely between manufacturers/models and subtly between "identical" lenses due to manufacturing imperfections (see two different examples in the comments below).

Object bounding boxes are traditionally parameterized in a 2D Cartesian coordinate system. However, this space is subject to lens-specific distortions. Therefore, it is desirable to represent boxes on a 3D sphere instead which can be lens-agnostic. Spheres are also a natural way to model a wide FOV fisheye lenses.


## Provided Objects/Interfaces

In the Python snippet below you will find existing implementations for

* `convert_point()`: functional interface to conversions between coordinate systems. Points are parameterized as `(x, y)` for Cartesisan and `(x, y, z)` for spherical 
* `CartesianBbox`: a class to store traditional 2D bounding boxes. You may use whatever parameterization format(s) you find most appropriate

## Tasks

Given the code structure above

1. Complete the `SphericalBbox` class to provide a minimal bounding box representation in a 3D spherical coordinate system
2. Implement `bbox_to_spherical()` to convert a `CartesianBbox` to your 3D spherical parameterization
3. Create `CartesianPolygon` to represent arbitrary N-sided polygons

    You may assume a practical limit of 20 sides and optionally impose other restrictions as you see fit (i.e. convex, regular polygons only), but please document the reasoning behind any chosen assumptions and enforce them in your code

4. Generalize your conversion algorithm in `polygon_to_spherical()` to handle the new `CartesianPolygon` class 

    If you are unable to code a solution, please prepare thoughts on potential approaches to present later

For questions 3 and 4, you may modify the bounding box class/function rather than creating new ones if you prefer. 

Please complete all code within the `spherical_objects.py` script below and email your completed copy. We will schedule a follow-up discussion for you to present and explain the thought process behind your solution. Evaluation will be based on 

* Reason for choosing 3D parametrizations
* Computational complexity of algorithms
* Coding structure

