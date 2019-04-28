---
layout: post
title:  "Optimizing with Karnush-Kuhn-Tucker conditions"
date:   2019-04-25 08:00:00 -0600
categories: update
---
{% include mathJax.html %}
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

# Optimizing
How optimization works in torch, also in general with StochasticGradientDescent (SGD)
{% include videoPlayer.html file="tomars/toMars1.mp4" %}

How importantant starting positions are when using SGD
{% include videoPlayer.html file="tomars/toMars2.mp4" %}

# Unconstraint
{% include videoPlayer.html file="tomars/toMars3.mp4" %}
{% include videoPlayer.html file="tomars/toMars4.mp4" %}
{% include videoPlayer.html file="tomars/toMars5.mp4" %}
{% include videoPlayer.html file="tomars/toMars6.mp4" %}
{% include videoPlayer.html file="tomars/toMars7.mp4" %}
{% include videoPlayer.html file="tomars/toMars8.mp4" %}


# Constraint with Karnush-Kuhn-Tucker

\begin{equation}
   L(x,\lambda, \alpha) = f(x)+\sum_{i} \lambda_{i} g_{i}(x) + \sum_{j} \alpha_{j} h_{j}(x)
\end{equation}

{% include videoPlayer.html file="tomars/toMars9.mp4" %}
{% include videoPlayer.html file="tomars/toMars10.mp4" %}

## Solution

{% include videoPlayer.html file="tomars/toMars11.mp4" %}

# Conclutions

# Attribution
Original problem [presented here](https://colab.research.google.com/drive/15sg1s9WSkAvXaGJ5genkHi_SeXKT5xES) by:  b2ray2c@gmail.com
