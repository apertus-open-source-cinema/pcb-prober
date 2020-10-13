import argparse
import csv
from pathlib import Path
from xml.etree import ElementTree

import drawSvg
import pandas
from PIL import Image

OUTPUT_DIR = "output/"


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


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


def setup_args(parser):
    general_group = parser.add_argument_group('General')
    general_group.add_argument('eagle_brd_file', help='EAGLE .brd file',
                               type=str, default=1.0)
    general_group.add_argument('pad_extractor_csv_file', help='EAGLE pad extractor CSV output',
                               type=str, default=1.0)
    general_group.add_argument('background_image', help='PCB image for the background',
                               type=str, default=1.0)
    parser.add_argument('data_file', type=str, help='file with probed data')

    value_group = parser.add_argument_group('Values to subtract')
    value_group.add_argument('-rv', '--red_value', help='Value to subtract from red color (float)', type=float,
                             default=0.0)
    value_group.add_argument('-gv', '--green-value', help='Value to subtract from green color (float)',
                             type=float, default=0.0)
    value_group.add_argument('-bv', '--blue-value', help='Value to subtract from blue color (float)',
                             type=float, default=0.0)
    factor_group = parser.add_argument_group('Values to subtract')
    factor_group.add_argument('-rf', '--red_factor', help='Factor for red color (float)', type=float,
                              default=1.0)
    factor_group.add_argument('-gf', '--green-factor', help='Factor for green color (float)', type=float,
                              default=1.0)
    factor_group.add_argument('-bf', '--blue-factor', help='Factor for blue color(float)', type=float,
                              default=1.0)


def prepare_data(df):
    # Remove AIR and FID* entries
    df.drop(df[(df.PARTNAME == 'AIR') | (df.PARTNAME.str.startswith('FID'))].index, inplace=True)

    # Split and convert measured data
    df[['MEAS_1', 'MEAS_2', 'MEAS_3', 'MEAS_4', 'MEAS_5', 'MEAS_6']] = df['MEASUREMENT-RESULT'].str.split(expand=True)
    df['MEAS_3'] = df['MEAS_3'].apply(lambda x: int(x, 16))
    df['MEAS_5'] = df['MEAS_5'].apply(lambda x: int(x, 16))
    df['MEAS_6'] = df['MEAS_6'].apply(lambda x: int(x, 16))

    # Convert pad ID from float to integer
    df['PAD-ID'] = df['PAD-ID'].apply(lambda x: int(x))

    # Calculate median values
    agg_funcs = dict(MEAS_3='median', MEAS_5='median', MEAS_6='median')
    data = df.groupby(['PARTNAME', 'PAD-ID']).agg(agg_funcs)

    return data


def load_board_data(args, board_data):
    with open(args.pad_extractor_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                print(f'{row}')
            key = row[1] + '_' + row[3]
            board_data[key] = row
            line_count += 1


def draw_pads(board_data, d, data, factor):
    for index, element in data.iterrows():
        red = element['MEAS_3_CLAMPED']
        green = element['MEAS_5_CLAMPED']
        blue = element['MEAS_6_CLAMPED']
        color = rgb_to_hex((int(red / 4), int(green), int(blue)))
        part_name = element.name[0]
        pad_index = str(element.name[1])
        part = board_data[part_name + '_' + pad_index]
        element_x = float(part[5]) * factor
        element_y = float(part[6]) * factor

        pad_width = float(part[8]) * factor
        pad_length = float(part[7]) * factor

        rot_angle = int(part[9])

        rotation = str.format(
            "translate({1}, {2}) rotate({0})", str(-rot_angle), str(element_x), str(-element_y))

        d.append(drawSvg.Rectangle(-pad_width / 2, -pad_length / 2, pad_width,
                                   pad_length, stroke_width=1, stroke="black", fill=color, transform=rotation))


def main():
    # Setup
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate preview of probed PCB data.')
    setup_args(parser)
    args = parser.parse_args()

    # Find board dimensions for visualization
    tree = ElementTree.parse(args.eagle_brd_file)
    xml_root = tree.getroot()

    board_x, board_y, board_width, board_height = get_board_dimensions(
        xml_root)

    # Load background image
    image = Image.open(args.background_image)
    factor = (image.width / board_width + image.height / board_height) / 2
    d = drawSvg.Drawing(image.width, image.height, origin=(board_x * factor, board_y * factor))
    d.append(
        drawSvg.Image(board_x * factor, board_y * factor, image.width, image.height, args.background_image, embed=True))

    # Load measurement data
    df = pandas.read_csv(args.data_file, sep=';', header=3)
    data = prepare_data(df)

    # Load board data
    board_data = {}
    load_board_data(args, board_data)

    # Modify color values and apply factors
    data['MEAS_3_CLAMPED'] = data['MEAS_3'].apply(lambda x: (x - args.red_value) * args.red_factor)
    data['MEAS_5_CLAMPED'] = data['MEAS_5'].apply(lambda x: (x - args.green_value) * args.green_factor)
    data['MEAS_6_CLAMPED'] = data['MEAS_6'].apply(lambda x: (x - args.blue_value) * args.blue_factor)

    draw_pads(board_data, d, data, factor)

    d.savePng(OUTPUT_DIR + "measurement_result.png")
    d.saveSvg(OUTPUT_DIR + "measurement_result.svg")


if __name__ == "__main__":
    main()
