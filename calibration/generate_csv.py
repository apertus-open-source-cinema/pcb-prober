#!/usr/bin/env python3
#
# Copyright (C) 2020 Sebastian Pichelhofer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Notes:
# The AXIOM Beta Power Board requires 0,0 to 120,116 of ender space to operate properly (with probe and camera)

grid_distance_x = 2
grid_distance_y = 2
grid_count_x = 58
grid_count_y = 66

f = open("output.csv", "w")
index = 4

# first run ascending
for x in range(grid_count_x):
    for y in range(grid_count_y):
        # reference: 17;X20Y10;;0;0;20;10;0;0;0;
        f.write("%d;X%dY%d;;0;0;%d;%d;0;0;0;\r\n" % (
            index, x * grid_distance_x, y * grid_distance_y, x * grid_distance_x, y * grid_distance_y))
        index += 1

# second run descending
for x in range(grid_count_x - 1, -1, -1):
    for y in range(grid_count_y - 1, -1, -1):
        # reference: 17;X20Y10;;0;0;20;10;0;0;0;
        f.write("%d;X%dY%d;;0;0;%d;%d;0;0;0;\r\n" % (
            index, x * grid_distance_x, y * grid_distance_y, x * grid_distance_x, y * grid_distance_y))
        index += 1

# third run other axis first
for y in range(grid_count_y):
    for x in range(grid_count_x):
        # reference: 17;X20Y10;;0;0;20;10;0;0;0;
        f.write("%d;X%dY%d;;0;0;%d;%d;0;0;0;\r\n" % (
            index, x * grid_distance_x, y * grid_distance_y, x * grid_distance_x, y * grid_distance_y))
        index += 1

# forth run other axis first reverse
for y in range(grid_count_y - 1, -1, -1):
    for x in range(grid_count_x - 1, -1, -1):
        # reference: 17;X20Y10;;0;0;20;10;0;0;0;
        f.write("%d;X%dY%d;;0;0;%d;%d;0;0;0;\r\n" % (
            index, x * grid_distance_x, y * grid_distance_y, x * grid_distance_x, y * grid_distance_y))
        index += 1

f.close()
