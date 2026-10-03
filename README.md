[日本語](README.ja.md)

# mykawa

*my* + *kawa* (川, "river" in Japanese): a river of my own.

## Introduction

Virtual worlds are everywhere now. In a video game you can own a house, fight enemies, or start a business. Houses, enemies and businesses have become far more realistic and detailed than in early games. But what about the land they sit on?

When I play video games, my eyes keep going to the rivers. Rivers are a tough thing for computers. Water flows, so there must be a difference in height. But a river's slope is gentle: you go hundreds of meters and drop only a few. Showing such a small difference with limited computing resources is probably hard, and probably not needed either.

I fully understand that. Still, I would be happy if rivers were realistic and detailed too. A river where you can imagine the flow, not just a strip of water, makes a world feel closer to me. So I started this project. I want to spend the effort and the computation only on rivers, and make rivers with many different faces.

## What it does now

So far I have made terrain up to about 3000 m × 12000 m, at about 11.7 m per pixel. In my setting, 128 pixels make 1.5 km. That is roughly a mile, and one pixel is about 38 ft. The terrain is made in three stages, which produce valleys and ridges.

1. Draw the river network
2. Raise the ground, using the rivers as a guide
3. Finish by cutting fine detail into the hillslopes

The output is a 2D array of elevations. With ordinary plotting software you can draw a map colored by elevation and lay contour lines over it. I will show how to do that, too.

## One more thing

Today's AI can easily make photos of people that look real. If I trained it on a large number of maps from the Geospatial Information Authority of Japan, it would probably produce the kind of terrain I am trying to make, without much trouble.

Even so, I want to try the other way: adding rules little by little and getting closer to the real thing. I think that work will sharpen my eye for the complexity of nature. What AI draws is a picture that averages away the character of countless rivers. I want to put what I find into code, and get closer to a living river.

I hope I can share with you the fun of shaping a river of your own and enjoying it.
