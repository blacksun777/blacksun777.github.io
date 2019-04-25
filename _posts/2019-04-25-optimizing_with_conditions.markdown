---
layout: post
title:  "Optimizing with Karnush-Kuhn-Tucker conditions"
date:   2019-01-25 08:00:00 -0600
categories: update
---
# Spaceship to Mars
The goal is to optimize a non trivial model with multiple parametes and conditions using `pyTourch.optim` and **Karnush-Kuhn-Tucker** (KKT) 
We want to send a spacecraft from Earth to Mars in a straight line at constant speed, Ignoring gravity and Orbital mechanics ([see this otherwise](https://www.jpl.nasa.gov/edu/teach/activity/lets-go-to-mars-calculating-launch-windows/))
{% include videoPlayer.html file="tomars/toMars0.mp4" %}
All the code and additional details can be found [here](https://colab.research.google.com/drive/1TkazncDHYCHdxeyKk9e4eZnBvnTBW1JW)

#### Assumptions
* Traveling on a straight line with constant speed
* The path of Mars and Earth are perfectly circular around the sun and move on the same inclination

#### Requirements
* Select the time from now to launch the spaceship
* Tell how many days it will take the spaceship to travel
* Max speed allowed is 0.5 M km/day


