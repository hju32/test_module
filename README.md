This project aims to be a robust algorithm to detect monocolor target.


## TO-DOs
Go-pro undistortion
fix white balance (find ways how)

run code on win-pc
run pipeline on sample pics
run tracking pipeline on vid

## Pipeline:(single image, no access to historic data)
Pre-adjustment:
(exp) sharpening?
Undistort

Tracking:
Mask out prevalent color pixels(histogram, greenest)
Mask small region of interest (area with unique colors)
Local normalization/re-whitebalance in each ROI (prevent bias,lighting, WB)
Edge detection, find lines, find crossing -> output.coordinate
One crossing each ROI

Then, score each crossing, based on following criterion:
Check length of crossed edges, does it look like our chessboard?
Check color near crossing, are they alternating black/white?
