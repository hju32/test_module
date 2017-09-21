Target detection code for Gatech DBF 2018 Medical Express Challenge

This project aims to be a robust algorithm to detect 2 by 2 chessboard pattern
### Author
Original Author: Eric Fang (CMU)
Current Author: Nick Zhang nickzhang@gatech.edu

## TO-DOs - Tickets

|Description    | Status    | comment|
|---|---|---|
|Reduce box drawing time| open ||
|Go-pro undistortion | Suspended| we will do that for real computer|
|fix white balance (find ways how)|closed| update: our GoPro can't do that|

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

## Issues
When close to target, whites in target will be considered mode, therefore ROI may filter target out.
This appears in clip1( 1:09-1:20 )

Some points get picked up in the pre-processing routine as flashing dots that appear intermittently from frame to frame
This appears in clip1( 20-30, 49-52 )

## Suggestions
Change target to non-directional pattern, like concentric circles

## Notes
Gopro 1080p delivers usable source at 100ft altitute. 
