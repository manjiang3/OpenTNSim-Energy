"""Main module."""

# package(s) related to time, space and id
import json
import logging
import uuid
import pathlib
import datetime
import time

# you need these dependencies (you can get these from anaconda)
# package(s) related to the simulation
import simpy
import random
import networkx as nx
import numpy as np
import math
import pandas as pd

# spatial libraries
import pyproj
import shapely.geometry

# additional packages

import opentnsim.energy
import opentnsim.graph_module

logger = logging.getLogger(__name__)


class SimpyObject:
    """General object which can be extended by any class requiring a simpy environment

    - env: a simpy Environment
    """

    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env


class HasResource(SimpyObject):
    """Something that has a resource limitation, a resource request must be granted before the object can be used.

    - nr_resources: nr of requests that can be handled simultaneously
    """

    def __init__(self, nr_resources=1, priority=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.resource = (
            simpy.PriorityResource(self.env, capacity=nr_resources)
            if priority
            else simpy.Resource(self.env, capacity=nr_resources)
        )


class Identifiable:
    """Mixin class: Something that has a name and id

    - name: a name
    - id: a unique id generated with uuid
    """

    def __init__(self, name, id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.name = name
        # generate some id, in this case based on m
        self.id = id if id else str(uuid.uuid1())


class Locatable:
    """Mixin class: Something with a geometry (geojson format)

    - geometry: can be a point as well as a polygon
    """

    def __init__(self, geometry, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.geometry = geometry
        self.node = None


class Neighbours:
    """Can be added to a locatable object (list)

    - travel_to: list of locatables to which can be travelled
    """

    def ___init(self, travel_to, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.neighbours = travel_to


class HasLength(SimpyObject):
    """Mixin class: Something with a storage capacity

    capacity: amount the container can hold
    level: amount the container holds initially
    total_requested: a counter that helps to prevent over requesting
    """

    def __init__(self, length, remaining_length=0, total_requested=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.length = simpy.Container(self.env, capacity=length, init=remaining_length)
        self.pos_length = simpy.Container(
            self.env, capacity=length, init=remaining_length
        )


class HasContainer(SimpyObject):
    """Mixin class: Something with a storage capacity
    capacity: amount the container can hold
    level: amount the container holds initially
    container: a simpy object that can hold stuff
    total_requested: a counter that helps to prevent over requesting"""

    def __init__(self, capacity, level=0, total_requested=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.container = simpy.Container(self.env, capacity, init=level)
        self.total_requested = total_requested

    @property
    def is_loaded(self):
        return True if self.container.level > 0 else False

    @property
    def filling_degree(self):
        return self.container.level / self.container.capacity


class Log(SimpyObject):
    """Mixin class: Something that has logging capability

    log: log message [format: 'start activity' or 'stop activity']
    t: timestamp
    value: a value can be logged as well
    geometry: value from locatable (lat, lon)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.log = {"Message": [], "Timestamp": [], "Value": [], "Geometry": []}

    def log_entry(self, log, t, value, geometry_log):
        """Log"""
        self.log["Message"].append(log)
        self.log["Timestamp"].append(datetime.datetime.fromtimestamp(t))
        self.log["Value"].append(value)
        self.log["Geometry"].append(geometry_log)

    def get_log_as_json(self):
        json = []
        for msg, t, value, geometry_log in zip(
            self.log["Message"],
            self.log["Timestamp"],
            self.log["Value"],
            self.log["Geometry"],
        ):
            json.append(
                dict(message=msg, time=t, value=value, geometry_log=geometry_log)
            )
        return json


class VesselProperties:
    """Mixin class: Something that has vessel properties
    This mixin is updated to better accommodate the ConsumesEnergy mixin

    - type: can contain info on vessel type (avv class, cemt_class or other)
    - B: vessel width
    - L: vessel length
    - h_min: vessel minimum water depth, can also be extracted from the network edges if they have the property ['Info']['GeneralDepth']
    - T: actual draught
    - C_B: block coefficient ('fullness') [-]
    - safety_margin : the water area above the waterway bed reserved to prevent ship grounding due to ship squatting during sailing, the value of safety margin depends on waterway bed material and ship types. For tanker vessel with rocky bed the safety margin is recommended as 0.3 m based on Van Dorsser et al. The value setting for safety margin depends on the risk attitude of the ship captain and shipping companies.
    - h_squat: the water depth considering ship squatting while the ship moving (if set to False, h_squat is disabled)
    - payload: cargo load [ton], the actual draught can be determined by knowing payload based on van Dorsser et al's method.(https://www.researchgate.net/publication/344340126_The_effect_of_low_water_on_loading_capacity_of_inland_ships)
    - vessel_type: vessel type can be selected from "Container","Dry_SH","Dry_DH","Barge","Tanker". ("Dry_SH" means dry bulk single hull, "Dry_DH" means dry bulk double hull), based on van Dorsser et al's paper.(https://www.researchgate.net/publication/344340126_The_effect_of_low_water_on_loading_capacity_of_inland_ships)
    Alternatively you can specify draught based on filling degree
    - H_e: vessel height unloaded
    - H_f: vessel height loaded
    - T_e: draught unloaded
    - T_f: draught loaded
    - renewable_fuel_mass: renewable fuel mass on board [kg]
    - renewable_fuel_volume: renewable fuel volume on board [m3]
    - renewable_fuel_required_space: renewable fuel required storage space (consider packaging factor) on board  [m3]
    """

    # TODO: add blockage factor S to vessel properties

    def __init__(
        self,
        type,
        B,
        L,
        h_min=None,
        T=None,
        C_B=None,
        H_e=None,
        H_f=None,
        T_e=None,
        T_f=None,
        safety_margin=None,
        h_squat=None,
        payload=None,
        vessel_type=None,
        renewable_fuel_mass=None,
        renewable_fuel_volume=None,
        renewable_fuel_required_space=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        """Initialization
        """
        self.type = type
        self.B = B
        self.L = L
        # hidden because these can also computed on the fly
        self._T = T
        self._h_min = h_min
        self.C_B = C_B
        # alternative  options
        self.H_e = H_e
        self.H_f = H_f
        self.T_e = T_e
        self.T_f = T_f
        self.safety_margin = safety_margin
        self.h_squat = h_squat
        self.payload = payload
        self.vessel_type = vessel_type
        self.renewable_fuel_mass = renewable_fuel_mass
        self.renewable_fuel_volume = renewable_fuel_volume
        self.renewable_fuel_required_space = renewable_fuel_required_space

    @property
    def T(self):
        """Compute the actual draught

        There are 3 ways to get actual draught
        - by directly providing actual draught values in the notebook
        - Or by providing ship draughts in fully loaded state and empty state, the actual draught will be computed based on filling degree


        """
        if self._T is not None:
            # if we were passed a T value, use tha one
            T = self._T
        elif self.T_f is not None and self.T_e is not None:
            # base draught on filling degree
            T = self.filling_degree * (self.T_f - self.T_e) + self.T_e
        elif self.payload is not None and self.vessel_type is not None:
            T = opentnsim.strategy.Payload2T(
                self,
                Payload_strategy=self.payload,
                vessel_type=self.vessel_type,
                bounds=(0, 40),
            )  # this need to be tested

        return T

    @property
    def h_min(self):
        if self._h_min is not None:
            h_min = self._h_min
        else:
            h_min = opentnsim.graph_module.get_minimum_depth(
                graph=self.env.FG, route=self.route
            )

        return h_min

    def calculate_max_sinkage(self, v, h_0):
        """Calculate the maximum sinkage of a moving ship

        the calculation equation is described in Barrass, B. & Derrett, R.'s book (2006), Ship Stability for Masters and Mates, chapter 42. https://doi.org/10.1016/B978-0-08-097093-6.00042-6

        some explanation for the variables in the equation:
        - h_0: water depth
        - v: ship velocity relative to the water
        - 150: Here we use the standard width 150 m as the waterway width

        """

        max_sinkage = (
            (self.C_B * ((self.B * self._T) / (150 * h_0)) ** 0.81)
            * ((v * 1.94) ** 2.08)
            / 20
        )

        return max_sinkage

    def calculate_h_squat(self, v, h_0):

        if self.h_squat:
            h_squat = h_0 - self.calculate_max_sinkage(v, h_0)

        else:
            h_squat = h_0

        return h_squat

    @property
    def H(self):
        """Calculate current height based on filling degree"""

        return self.filling_degree * (self.H_f - self.H_e) + self.H_e

    def get_route(
        self,
        origin,
        destination,
        graph=None,
        minWidth=None,
        minHeight=None,
        minDepth=None,
        randomSeed=4,
    ):
        """Calculate a path based on vessel restrictions"""

        graph = graph if graph else self.env.FG
        minWidth = minWidth if minWidth else 1.1 * self.B
        minHeight = minHeight if minHeight else 1.1 * self.H
        minDepth = minDepth if minDepth else 1.1 * self.T

        # Check if information on restrictions is added to the edges
        random.seed(randomSeed)
        edge = random.choice(list(graph.edges(data=True)))
        edge_attrs = list(edge[2].keys())

        # IMPROVE THIS TO CHECK ALL EDGES AND COMBINATIONS OF RESTRICTIONS

        if all(item in edge_attrs for item in ["Width", "Height", "Depth"]):
            edges = []
            nodes = []

            for edge in graph.edges(data=True):
                if (
                    edge[2]["Width"] >= minWidth
                    and edge[2]["Height"] >= minHeight
                    and edge[2]["Depth"] >= minDepth
                ):
                    edges.append(edge)

                    nodes.append(graph.nodes[edge[0]])
                    nodes.append(graph.nodes[edge[1]])

            subGraph = graph.__class__()

            for node in nodes:
                subGraph.add_node(
                    node["name"],
                    name=node["name"],
                    geometry=node["geometry"],
                    position=(node["geometry"].x, node["geometry"].y),
                )

            for edge in edges:
                subGraph.add_edge(edge[0], edge[1], attr_dict=edge[2])

            try:
                return nx.dijkstra_path(subGraph, origin, destination)
                # return nx.bidirectional_dijkstra(subGraph, origin, destination)
            except:
                raise ValueError(
                    "No path was found with the given boundary conditions."
                )

        # If not, return shortest path
        else:
            return nx.dijkstra_path(graph, origin, destination)


class Routeable:
    """Mixin class: Something with a route (networkx format)

    - route: a networkx path
    """

    def __init__(self, route, complete_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.route = route
        self.complete_path = complete_path


class Movable(Locatable, Routeable, Log):
    """Mixin class: Something can move

    Used for object that can move with a fixed speed

    - geometry: point used to track its current location
    - v: speed
    """

    def __init__(self, v, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.v = v
        self.on_pass_edge_functions = []
        self.wgs84 = pyproj.Geod(ellps="WGS84")

    def move(self):
        """determine distance between origin and destination, and
        yield the time it takes to travel it
        Assumption is that self.path is in the right order - vessel moves from route[0] to route[-1].
        """
        self.distance = 0
        speed = self.v

        # Check if vessel is at correct location - if not, move to location
        if (
            self.geometry
            != nx.get_node_attributes(self.env.FG, "geometry")[self.route[0]]
        ):
            orig = self.geometry
            dest = nx.get_node_attributes(self.env.FG, "geometry")[self.route[0]]

            logger.debug("Origin: {orig}")
            logger.debug("Destination: {dest}")

            self.distance += self.wgs84.inv(
                shapely.geometry.shape(orig).x,
                shapely.geometry.shape(orig).y,
                shapely.geometry.shape(dest).x,
                shapely.geometry.shape(dest).y,
            )[2]

            yield self.env.timeout(self.distance / self.current_speed)
            self.log_entry("Sailing to start", self.env.now, self.distance, dest)

        # Move over the path and log every step
        for edge in zip(self.route[:-1], self.route[1:]):
            origin, destination = edge
            self.node = origin
            yield from self.pass_edge(origin, destination)
            # we arrived at destination
            self.geometry = nx.get_node_attributes(self.env.FG, "geometry")[destination]

        logger.debug("  distance: " + "%4.2f" % self.distance + " m")
        if self.current_speed is not None:
            logger.debug("  sailing:  " + "%4.2f" % self.current_speed + " m/s")
            logger.debug(
                "  duration: "
                + "%4.2f" % ((self.distance / self.current_speed) / 3600)
                + " hrs"
            )
        else:
            logger.debug("  current_speed:  not set")

    def pass_edge(self, origin, destination):
        edge = self.env.FG.edges[origin, destination]
        orig = nx.get_node_attributes(self.env.FG, "geometry")[origin]
        dest = nx.get_node_attributes(self.env.FG, "geometry")[destination]

        for on_pass_edge_function in self.on_pass_edge_functions:
            on_pass_edge_function(origin, destination)

        if "geometry" in edge:
            edge_route = np.array(edge["geometry"].coords)

            # check if edge is in the sailing direction, otherwise flip it
            distance_from_start = self.wgs84.inv(
                orig.x,
                orig.y,
                edge_route[0][0],
                edge_route[0][1],
            )[2]
            distance_from_stop = self.wgs84.inv(
                orig.x,
                orig.y,
                edge_route[-1][0],
                edge_route[-1][1],
            )[2]
            if distance_from_start > distance_from_stop:
                # when the distance from the starting point is greater than from the end point
                edge_route = np.flipud(np.array(edge["geometry"].coords))

            for index, pt in enumerate(edge_route[:-1]):
                sub_orig = shapely.geometry.Point(
                    edge_route[index][0], edge_route[index][1]
                )
                sub_dest = shapely.geometry.Point(
                    edge_route[index + 1][0], edge_route[index + 1][1]
                )

                distance = self.wgs84.inv(
                    shapely.geometry.shape(sub_orig).x,
                    shapely.geometry.shape(sub_orig).y,
                    shapely.geometry.shape(sub_dest).x,
                    shapely.geometry.shape(sub_dest).y,
                )[2]
                self.distance += distance
                self.log_entry(
                    "Sailing from node {} to node {} sub edge {} start".format(
                        origin, destination, index
                    ),
                    self.env.now,
                    0,
                    sub_orig,
                )
                yield self.env.timeout(distance / self.current_speed)
                self.log_entry(
                    "Sailing from node {} to node {} sub edge {} stop".format(
                        origin, destination, index
                    ),
                    self.env.now,
                    0,
                    sub_dest,
                )
            self.geometry = dest
        else:
            distance = self.wgs84.inv(
                shapely.geometry.shape(orig).x,
                shapely.geometry.shape(orig).y,
                shapely.geometry.shape(dest).x,
                shapely.geometry.shape(dest).y,
            )[2]

            self.distance += distance

            value = 0

            # remember when we arrived at the edge
            arrival = self.env.now

            v = self.current_speed

            # This is the case if we are sailing on power
            if getattr(self, "P_tot_given", None) is not None:
                edge = self.env.FG.edges[origin, destination]
                depth = self.env.FG.get_edge_data(origin, destination)["Info"][
                    "GeneralDepth"
                ]

                # estimate 'grounding speed' as a useful upperbound
                (
                    upperbound,
                    selected,
                    results_df,
                ) = opentnsim.strategy.get_upperbound_for_power2v(
                    self, width=150, depth=depth, margin=0
                )
                v = self.power2v(self, edge, upperbound)
                # use computed power
                value = self.P_given

            # determine time to pass edge
            timeout = distance / v

            # Wait for edge resources to become available
            if "Resources" in edge.keys():
                with self.env.FG.edges[origin, destination][
                    "Resources"
                ].request() as request:
                    yield request
                    # we had to wait, log it
                    if arrival != self.env.now:
                        self.log_entry(
                            "Waiting to pass edge {} - {} start".format(
                                origin, destination
                            ),
                            arrival,
                            value,
                            orig,
                        )
                        self.log_entry(
                            "Waiting to pass edge {} - {} stop".format(
                                origin, destination
                            ),
                            self.env.now,
                            value,
                            orig,
                        )

            # default velocity based on current speed.
            self.log_entry(
                "Sailing from node {} to node {} start".format(origin, destination),
                self.env.now,
                value,
                orig,
            )
            yield self.env.timeout(timeout)
            self.log_entry(
                "Sailing from node {} to node {} stop".format(origin, destination),
                self.env.now,
                value,
                dest,
            )
        self.geometry = dest

    @property
    def current_speed(self):
        return self.v


class ContainerDependentMovable(Movable, HasContainer):
    """ContainerDependentMovable class
    Used for objects that move with a speed dependent on the container level
    compute_v: a function, given the fraction the container is filled (in [0,1]), returns the current speed"""

    def __init__(self, compute_v, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.compute_v = compute_v
        self.wgs84 = pyproj.Geod(ellps="WGS84")

    @property
    def current_speed(self):
        return self.compute_v(self.container.level / self.container.capacity)


class ExtraMetadata:
    """store all leftover keyword arguments as metadata property (use as last mixin)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        # store all other properties as metadata
        self.metadata = kwargs
