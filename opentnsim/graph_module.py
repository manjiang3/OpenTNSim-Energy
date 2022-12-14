# -*- coding: utf-8 -*-

"""Graph module."""

# package(s) related to time, space and id
import json
import logging
import math
import os
import uuid

import networkx as nx
import numpy as np

# spatial libraries
import pyproj
import shapely.geometry

# matplotlib
import matplotlib.pyplot as plt

# package(s) related to the simulation
import simpy

logger = logging.getLogger(__name__)

# Determine the wgs84 geoid
wgs84 = pyproj.Geod(ellps="WGS84")


class Graph:
    """General networkx object

    Initialize a nx.Graph() element
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = nx.Graph()
        self.graph_info = nx.info(self.graph)

    def from_shape(self, file_location, shapefile, simplify=True, strict=True):
        """Generate nx.Graph() from shapefile

        file_location: location on server of the shapefile
        shapefile: name of the shapefile (including .shp)
        """
        from osgeo import ogr, osr

        # Create graph
        self.graph = nx.read_shp(
            os.path.join(file_location, shapefile), simplify=simplify, strict=strict
        )
        self.graph_info = nx.info(self.graph)

        # Get spatial reference
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataset = driver.Open(os.path.join(file_location, shapefile))
        self.SpatialRef = dataset.GetLayer().GetSpatialRef()

    def transform_projection(self, to_EPSG):
        from osgeo import ogr, osr

        outSpatialRef = osr.SpatialReference()
        outSpatialRef.ImportFromEPSG(to_EPSG)

        # Transform the coordinates
        transform = osr.CoordinateTransformation(self.SpatialRef, outSpatialRef)

        return transform

    def change_projection(self, transform, point):
        from osgeo import ogr, osr

        point = ogr.CreateGeometryFromWkt(str(point))

        point.Transform(transform)
        point.ExportToWkt()

        return point.GetX(), point.GetY()

    def create_graph_new_projection(self, to_EPSG=4326):
        new_graph = nx.Graph()
        transform = self.transform_projection(to_EPSG)

        # Required to prevent loop-in-loop
        nodes_dict = {}

        # Add original nodes and edges to new graph
        for i, node in enumerate(self.graph.nodes(data=True)):
            coordinates = self.change_projection(
                transform,
                shapely.geometry.Point(
                    list(self.graph.nodes)[i][0], list(self.graph.nodes)[i][1]
                ),
            )
            name = "({:f}, {:f})".format(coordinates[0], coordinates[1])
            geometry = shapely.geometry.Point(coordinates[0], coordinates[1])

            nodes_dict[list(self.graph.nodes)[i]] = name
            new_graph.add_node(
                name, name=name, Position=coordinates, geometry=geometry, Old=node[1]
            )

        for edge in self.graph.edges(data=True):
            node_1 = nodes_dict[edge[0]]
            node_2 = nodes_dict[edge[1]]

            new_graph.add_edge(node_1, node_2, Info=edge[2])

        new_graph = new_graph.to_directed()

        if nx.info(new_graph) != self.graph_info:
            self.graph = new_graph
            self.graph_info = nx.info(new_graph)
        else:
            print("Conversion did not create an exact similar graph")

            print("")
            print("Original graph")
            print(self.graph_info)

            print("")
            print("New graph")
            print(nx.info(new_graph))

            self.graph = new_graph
            self.graph_info = nx.info(new_graph)

    def add_resources(self, edges, resources, environment):
        for i, edge in enumerate(edges):
            self.graph.edges[edge]["Resources"] = simpy.Resource(
                environment, capacity=resources[i]
            )

    def plot(
        self,
        size=[10, 10],
        with_labels=False,
        node_size=0.5,
        font_size=2,
        width=0.2,
        arrowsize=3,
    ):
        plt.figure(figsize=size)

        # If graph has positional attributes
        try:
            nx.draw(
                self.graph,
                nx.get_node_attributes(self.graph, "Position"),
                with_labels=with_labels,
                node_size=node_size,
                font_size=font_size,
                width=width,
                arrowsize=arrowsize,
            )
        # If graph does not have any positional information
        except:
            nx.draw(self.graph)

        plt.show()


def get_minimum_depth(graph, route):
    """return the minimum depth on the route based on the GeneralDepth in the Info dictionary"""
    # loop over the route
    depths = []
    # loop over all node pairs (e: edge numbers)
    for e in zip(route[:-1], route[1:]):
        # get the properties
        edge = graph.get_edge_data(e[0], e[1])
        # lookup the depth
        depth = edge['Info']['GeneralDepth']
        # remember
        depths.append(depth)
        # find the minimum
    h_min = np.min(depths)
    return h_min


def compute_distance(edge, orig, dest):
    """compute distance over edge, or if edge does not have a geometry over orig-dest"""
    if 'geometry' not in edge:
        distance = wgs84.inv(
            shapely.geometry.shape(orig).x,
            shapely.geometry.shape(orig).y,
            shapely.geometry.shape(dest).x,
            shapely.geometry.shape(dest).y,
        )[2]
        return distance

    edge_route = np.array(edge['geometry'].coords)

    # check if edge is in the sailing direction, otherwise flip it
    distance_from_start = wgs84.inv(
            orig.x,
            orig.y,
            edge_route[0][0],
            edge_route[0][1],
        )[2]
    distance_from_stop = wgs84.inv(
            orig.x,
            orig.y,
            edge_route[-1][0],
            edge_route[-1][1],
        )[2]
    if distance_from_start>distance_from_stop:
        # when the distance from the starting point is greater than from the end point
        edge_route = np.flipud(np.array(edge['geometry'].coords))

    distance = 0
    for index, pt in enumerate(edge_route[:-1]):
        sub_orig = shapely.geometry.Point(edge_route[index][0], edge_route[index][1])
        sub_dest = shapely.geometry.Point(edge_route[index+1][0], edge_route[index+1][1])

        distance += wgs84.inv(
            shapely.geometry.asShape(sub_orig).x,
            shapely.geometry.asShape(sub_orig).y,
            shapely.geometry.asShape(sub_dest).x,
            shapely.geometry.asShape(sub_dest).y,
        )[2]
    return distance
