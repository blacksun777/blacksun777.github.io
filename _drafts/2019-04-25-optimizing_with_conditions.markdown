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
The [`PyTourch.Optim`](https://pytorch.org/docs/stable/optim.html) package contains optimizers that update model parameters based on the computet gradient with StochasticGradientDescent (SGD).

Basically optimizers find minima for functions like $$f(x): x^2 -x -1$$

{% include videoPlayer.html file="tomars/toMars1.mp4" %}
By repeatetly going down the 'hill' we approch a local minima. We end up at a minima, in this case `0.5`

We will use this general structure to find solutions to our problem. We first define our problem and then iteratively improove our solution by going down the hill with a small step. 
{% highlight python %}
import torch
from torch.optim import SGD
from torch import nn
from torch import FloatTensor as ft

# The function that should be minimized
def lossFunction(x):
  return x**2-x-1

# Parameter x that should be minimized
param_x = nn.Parameter(ft([4]))

# Create optimizer and tell it what parameter to optimize and how
opt_sgd = SGD([param_x], lr = 0.1)

# Loop to let SGD converge
for i in range(20):
  
  # Remove previous gradient from parameters
  opt_sgd.zero_grad() 
  
  # Compute the loss (the thing that should be minized)
  loss = lossFunction(param_x)
  
  # Compute the gradient at current position (param_x)
  loss.backward()
  
  # Let Optimizer step down the gradient based on computed gradient (backpropagate)
  opt_sgd.step()
{% endhighlight %}

Our problem is quite a bit more complex than the previous loss function, there are many possible solutions. That is why a good starting location is very important. 
How importantant starting positions are when using optimizers can be illustrated with $$f(x): \cos(x)$$, we use two optimizers one with a slighly postive starting position and one with a slighly negative one.
{% include videoPlayer.html file="tomars/toMars2.mp4" %}
As we can see and hopefully predicted, the initial value causes us to end up in different local mimima. In the Animation we can also see the difference between an SGD and Adam optimizer.

# Unconstraint
With the understanding of how Optimizers work we can now model our problem. 

$$\begin{equation}
   calcPosXY(t,rad, r): (x,y) | x=cos(t, rad)r \land y=sin(t, rad)r \\
   distance(t_{launch}, t_{flight}): \sqrt{(x_{mars}-x_{earth})^2+(y_{mars}-y_{earth})^2} | \\
   (x_{mars}, y_{mars})=calcPosXY(t_{launch}+t_{flight}, rad_{mars}, r_{mars}) \\
   \land (x_{earth}, y_{earth})=calcPosXY(t_{launch}, rad_{earth}, r_{earth})
\end{equation}$$

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
