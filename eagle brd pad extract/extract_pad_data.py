#!/usr/bin/python3

import csv
import math
import os
import re
import sys
from collections import defaultdict
from copy import copy
from typing import Dict
from xml.etree import ElementTree

import svg_output
from data_structs import Element, Package, Pad, SMDPad, ViaPad, Layer

via_elongation_offset: int = 0


def get_general_pad_info(pad) -> Pad:
    x = float(pad.attrib["x"])
    y = float(pad.attrib["y"])

    return Pad(x, y)


def get_smd_pad_info(general_pad_info, smd) -> SMDPad:
    width = float(smd.attrib["dx"])
    length = float(smd.attrib["dy"])

    # 0 is horizontal, 90 degrees is vertical
    angle_degree = 0
    if width > length:
        width, length = length, width
        angle_degree += 90

    pad = SMDPad(general_pad_info.x, general_pad_info.y, length, width)

    if "rot" in smd.attrib:
        rotation = smd.attrib["rot"]

        digit_position = re.search(r"\d", rotation)
        angle_degree += int(rotation[digit_position.start():])

    # Limit rotation
    if angle_degree >= 180:
        angle_degree -= 180

    pad.rotation = angle_degree

    return pad


def get_via_pad_info(general_pad_info, via) -> ViaPad:
    drill = float(via.attrib["drill"])

    # 0 is horizontal, 90 degrees is vertical
    angle_degree = -90
    # if width > length:
    #   width, length = length, width
    #    angle_degree += 90

    if "rot" in via.attrib:
        rotation = via.attrib["rot"]

        digit_position = re.search(r"\d", rotation)
        angle_degree += int(rotation[digit_position.start():])

        # Limit rotation
    if angle_degree >= 180:
        angle_degree -= 180

    # pad.rotation = angle_degree

    width = drill + 0.5

    pad = ViaPad(general_pad_info.x, general_pad_info.y, width + width * (via_elongation_offset / 100), width,
                 angle_degree, drill=drill)

    return pad


def get_available_packages(xml_root):
    available_packages = dict()

    package_list = xml_root.findall(".//package")
    for item in package_list:
        package_name = item.attrib["name"]

        # Extract pad data, positions are offsets
        smd_pad_list = dict()
        via_pad_list = dict()

        smd_list = item.findall("smd")
        for smd in smd_list:
            pad = get_general_pad_info(smd)
            pad = get_smd_pad_info(pad, smd)
            pad_name = smd.attrib["name"]
            smd_pad_list[pad_name] = pad

        pad_list = item.findall("pad")
        for via in pad_list:
            pad = get_general_pad_info(via)
            pad = get_via_pad_info(pad, via)
            pad_name = via.attrib["name"]
            via_pad_list[pad_name] = pad

        package = Package(smd_pad_list, via_pad_list)
        available_packages[package_name] = package

    return available_packages


def extract_element_info(root, available_packages, layer: Layer):
    available_elements = dict()

    element_list = root.findall(".//element")
    for item in element_list:
        rotation_angle = 0
        if "rot" in item.attrib:
            rotation_angle = item.attrib["rot"]

            # Don't process opposite layer
            if (rotation_angle.startswith("MR") and layer == Layer.TOP) or (
                    rotation_angle.startswith("R") and layer == Layer.BOTTOM):
                continue

            # Find first digit position
            digit_position = re.search(r"\d", rotation_angle)
            rotation_angle = rotation_angle[digit_position.start():]
        elif layer == Layer.BOTTOM:
            continue

        package_name = item.attrib["package"]
        x = float(item.attrib["x"])
        if layer == Layer.BOTTOM:
            x = -x
        y = float(item.attrib["y"])
        element = Element(
            package_name, defaultdict(), defaultdict(), x, y)
        element_name = item.attrib["name"]

        available_elements[element_name] = element

        element.rotation = int(rotation_angle)

        element_package = available_packages[package_name]

        smd_pad_list = dict()
        for name, pad_item in element_package.smd_pads.items():
            pad = copy(pad_item)

            set_pad_data(element, pad, pad_item)

            smd_pad_list[name] = pad
        element.smd_pads = smd_pad_list

        via_pad_list = dict()
        for name, pad_item in element_package.via_pads.items():
            pad = copy(pad_item)
            # pad.width = float(pad_item.drill) + 0.5
            # pad.length = pad.drill * 2

            set_pad_data(element, pad, pad_item)

            via_pad_list[name] = pad
        element.via_pads = via_pad_list

    return available_elements


def set_pad_data(element, pad, pad_item):
    # Calculate rotation
    angle_degree = int(element.rotation)
    angle_rad = math.radians(angle_degree)
    rot_x = pad.x * math.cos(angle_rad) - pad.y * math.sin(angle_rad)
    rot_y = pad.x * math.sin(angle_rad) + pad.y * math.cos(angle_rad)

    pad.x = rot_x
    pad.y = rot_y

    angle_degree += pad_item.rotation
    if angle_degree >= 180:
        angle_degree -= 180
    pad.rotation = angle_degree

    # Calculate absolute position
    pad.x += element.x
    pad.y += element.y


def output_csv(file_name, element_list: Dict[str, Element]):
    f = open("./" + file_name, "w")

    with f:

        writer = csv.writer(f, delimiter=";")

        # Write CSV header
        header = ["INDEX", "PART_NAME", "PACKAGE_NAME", "PIN_INDEX", "PAD_NAME", "CENTER_X",
                  "CENTER_Y", "LONG_SIDE_LENGTH", "SHORT_SIDE_LENGTH", "ANGLE_DEGREE", "NET"]
        writer.writerow(header)

        # Write data rows
        overall_index = 0
        for element_name, element in element_list.items():
            pad_index = 0
            for pad_name, pad in element.smd_pads.items():
                writer.writerow([overall_index, element_name, element.package,
                                 pad_index, pad_name, round(pad.x, 3), round(pad.y, 3), pad.length, pad.width,
                                 pad.rotation, pad.net])

                pad_index += 1
                overall_index += 1

            pad_index = 0
            for pad_name, pad in element.via_pads.items():
                writer.writerow([overall_index, element_name, element.package,
                                 pad_index, pad_name, round(pad.x, 3), round(pad.y, 3), pad.length, pad.width,
                                 pad.rotation, pad.net])

                pad_index += 1
                overall_index += 1


def extract_nets(xml_root, elements):
    signal_root = xml_root.find(".//signals")
    signals = signal_root.findall("signal")

    for signal in signals:
        signal_name = signal.attrib["name"]
        contacts = signal.findall("contactref")

        for contact in contacts:
            element_name = contact.attrib["element"]
            pad_name = contact.attrib["pad"]

            if element_name in elements:
                element = elements[element_name]
                if pad_name in element.smd_pads:
                    element.smd_pads[pad_name].net = signal_name


def get_board_dimensions(xml_root):
    dimensions = xml_root.findall(".//wire[@layer=\"20\"]")

    min_x = 0.0
    min_y = 0.0
    max_x = 0.0
    max_y = 0.0

    for dimension in dimensions:
        if float(dimension.attrib["x1"]) < min_x:
            min_x = float(dimension.attrib["x1"])
        if float(dimension.attrib["y1"]) < min_y:
            min_y = float(dimension.attrib["y1"])
        if float(dimension.attrib["x2"]) > max_x:
            max_x = float(dimension.attrib["x2"])
        if float(dimension.attrib["y2"]) > max_y:
            max_y = float(dimension.attrib["y2"])

    return min_x, min_y, max_x - min_x, max_y - min_y


def export_layer(file_name_without_ext, layer_name, element_list, board_x, board_y, board_width, board_height):
    file_name = str.format("{0}_{1}", file_name_without_ext, layer_name.name.lower())
    if os.path.isfile(file_name + ".png"):
        svg_output.create_svg(file_name + ".png",
                              file_name + "_preview", element_list, board_x, board_y, board_width, board_height)

    output_csv(file_name + ".csv", element_list)


def get_general_board_info(xml_root):
    element = xml_root.find(".//param[@name=\"psElongationOffset\"]")
    global via_elongation_offset
    via_elongation_offset = int(element.attrib["value"])


def main():
    # Check argument count
    if len(sys.argv) != 2:
        print("Wrong argument count. Usage: extract_data_pad.py <eagle .brd>")
        sys.exit(1)

    brd_file_name = sys.argv[1]
    brd_file_path = os.path.dirname(brd_file_name)
    os.chdir(brd_file_path)
    brd_file_name = os.path.basename(brd_file_name)
    file_name_without_ext = os.path.splitext(brd_file_name)[0]

    tree = ElementTree.parse(brd_file_name)
    xml_root = tree.getroot()

    # Find board dimensions for visualization
    board_x, board_y, board_width, board_height = get_board_dimensions(
        xml_root)

    get_general_board_info(xml_root)

    # Make list of available packages
    available_packages = get_available_packages(xml_root)

    # Iterate through elements on the board
    available_elements_top = extract_element_info(
        xml_root, available_packages, Layer.TOP)

    available_elements_bottom = extract_element_info(
        xml_root, available_packages, Layer.BOTTOM)

    extract_nets(xml_root, available_elements_top)
    extract_nets(xml_root, available_elements_bottom)

    # Check if PCB image layer file exists, if yes create output
    export_layer(file_name_without_ext, Layer.TOP,
                 available_elements_top, board_x, board_y, board_width, board_height)
    export_layer(file_name_without_ext, Layer.BOTTOM,
                 available_elements_bottom, board_x, board_y, board_width, board_height)


main()
