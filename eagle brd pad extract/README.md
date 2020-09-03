# EAGLE .brd pad extractor

Extracts pad position and dimensions from EAGLE .brd file to .csv, for automated PCB inspection

## Limitations

* Some hardcoded values
* Messy code
* Optimizations missing

## Usage

Execute with following syntax:
`./extract_pad_data.py <EAGLE .brd file>`

Example:
`./extract_pad_data.py BetaPB/axiom_beta_power_board_v0.34.brd`

Output can be found next to `.brd` file with `.csv` extension
