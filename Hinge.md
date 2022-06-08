## Hinge Features

It capture information about the curvature and slant / angle of line at the point of intersection between two lines. This is done by computing the joint probability distribution of the orientations of the two legs. This extractor has 2 parameters. The length of each leg, and the number of angle bins. In our implementation we obtain legs by finding all contours in the image, then we filter out any contours shorter than 25 pixels. We then compute the angles between each two neighboring contours and construct a histogram using the angles 1, 2 (demonstrated in the above figure).

<p align="center">
  <img  width="350px" src="screenshots/hinge.png" alt="hinge">
</p>
