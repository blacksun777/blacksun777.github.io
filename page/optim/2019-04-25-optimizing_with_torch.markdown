---
layout: page
title:  "Optimizing with torch"
date:   2019-04-25 08:00:00 -0600
categories: update
---
{% include mathJax.html %}
# Optimizing

The [`PyTorch.Optim`](https://pytorch.org/docs/stable/optim.html) package contains optimizers that update model parameters based on the computed gradient with StochasticGradientDescent (SGD).

Basically optimizers find minima for functions like $$f(x): x^2 -x -1$$

{% include videoPlayer.html file="tomars/toMars1.mp4" %}
By repeatedly going down the 'hill' we approach a local minima. We end up at a minima, in this case `x=0.5`

We will use this general structure to find solutions to our problem. We first define our problem and then iteratively improve our solution by going down the hill with a small step. 
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
  
  # Compute the loss (the thing that should be minimized)
  loss = lossFunction(param_x)
  
  # Compute the gradient at current position (param_x)
  loss.backward()
  
  # Let Optimizer step down the gradient based on computed gradient (backpropagate)
  opt_sgd.step()
{% endhighlight %}

Our problem is quite a bit more complex than the previous loss function, there are many possible solutions. That is why a good starting location is very important. 
How important starting positions are when using optimizers can be illustrated with $$f(x): \cos(x)$$, we use two optimizers one with a slightly positive starting position and one with a slightly negative one.
{% include videoPlayer.html file="tomars/toMars2.mp4" %}
As we can see and hopefully predicted, the initial value causes us to end up in different local minima. In the Animation we can also see the difference between an SGD and Adam optimizer.

# Conclusion

If you have any input or feedback please drop me a mail: e.geisseler+blog_kkt@gmail.com 

