# Ball-Tracker
This is the python code for a cricket ball tracking app.

This is a project that I have been working on for about 2 years and this code has gone through many iterations. The code takes in a video of a cricket ball being bowled and tracks it through each frame. The final version of the project would take this data and then predict the path of the ball on the way up. I have written a function that does this but it does not fully work yet, so I have included it in the code but by default, it does not run.

The code currently outputs 2 images, the first "coordinates.png" displays all objects it identified from the frames and says 'could' be a ball. The second "pitchmap.png" displays the actual ballpatht that it has identified. 

This code has taken about 2 years working on it on and off to get to this current state. My first 'completed' version was done after about 6 months. Back then, I needed to input the colour of the ball, it would barely be able to identify the ball in perfect conditions and took 10 hours to run 10 frames.

Now it is much more robust and can analise a video of 40 frames in between 5 and 30 seconds (depending on the video)

Basically, the code works by going through the following steps:

1) Break down the image into frames
2) Identify all pixels that are different than the frame before (as the ball is moving)
3) Check each of those pixels and check around them to identify packs of pixels
4) Check that the pack is large enough (to elimate random noise)
5) Check how cirular the pack is by drawing a circle on top of it and comparing
6) Create a list of all packs over 60% circular (displayed on "coordinates.png")
8) Create a list of all sets of three packs that are in line with each other (less than 10 degrees)
9) Build chains of balls by combining these sets
10) The longest chain is detirmined to be the ball path
11) Try and find the longest path the starts at the bottom of the ball path (not always enough data on the way up to track this)
12) Now you have the full ball path!

Test the code by running main.py and changing the path when creating the Tracker object to test different videos
