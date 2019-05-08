---
layout: post
title:  "Optimizing with Karnush-Kuhn-Tucker conditions"
date:   2019-04-25 08:00:00 -0600
categories: update
---
{% include mathJax.html %}
# Spaceship to Mars
The goal is to optimize a non trivial model with multiple parameters and conditions using `pyTourch.optim` and **Karnush-Kuhn-Tucker** (KKT) 

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
The [`PyTourch.Optim`](https://pytorch.org/docs/stable/optim.html) package contains optimizers that update model parameters based on the computed gradient with StochasticGradientDescent (SGD).

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

# Unconstrained
With the understanding of how optimizers work we can now model our problem. Here is some knowlege we got from wikipedia:
{% highlight python %}
import math

# Radius
rE = 149.6
rM = 227.9

# Periods
yearE = 365.25
yearM = 687.0

# Orbital speed
radE = 2*math.pi / yearE
radM = 2*math.pi / yearM

# Given max speed
max_v = .5

# minimun fly time
time_min = (rM - rE)/max_v
{% endhighlight %}

Given this data we can now calculate where every planet is given a point in time. 

$$\begin{equation}
   calcPosXY(t,rad, r)= (x,y) | x=cos(t, rad)r \land y=sin(t, rad)r \\
\end{equation}$$

If we have the launch date and the flight time we can also calculate the distance the rocket will travel.

$$\begin{equation}
   distance(t_{launch}, t_{flight})= \sqrt{(x_{mars}-x_{earth})^2+(y_{mars}-y_{earth})^2} | \\
   (x_{mars}, y_{mars})=calcPosXY(t_{launch}+t_{flight}, rad_{mars}, r_{mars}) \land  \\
   (x_{earth}, y_{earth})=calcPosXY(t_{launch}, rad_{earth}, r_{earth})
\end{equation}$$

We can use the distance as a loss function. The reasoning behind this is that if we have minimal distance to travel we can travel to mars the fastest. If the distance was large we might be even traveling near by the sun, we don't want that.

## Good Parameter initialization

To calculate the distance we need two parameters:
{% highlight python %}
# Launch date: days starting from now
Launch = nn.Parameter(ft([200]))
# time spent in flight
time = nn.Parameter(ft([time_min]))
{% endhighlight %}

now we can minimize the loss with a optimizer, similar to the situation before:
{% highlight python %}
opt = Adam([Launch,time],lr = 1e-1)

for i in range(300):
  opt.zero_grad()
  Distance = distance(Launch, time)
  Distance.backward()
  opt.step()

flyDistance = distance(Launch, time)
print('Launch date=%f \tflighttime=%f\t distance=%f \tspeed=%f'%(Launch, time, flyDistance, flyDistance/time))
{% endhighlight %}
`Launch date=189.591919 	flighttime=167.008087	 distance=78.299988 	speed=0.468839`

LINK TO CODE

{% include videoPlayer.html file="tomars/toMars3.mp4" %}
Nice, we got a** Launch date in 189 days!** And its also the shortest route to the target because the **distance** is actually the same as the **Closest approach**. Speed is within bounds as well! 

But there is a problem: **Its a lie!** Kind of a lie. The values work and are correct, but it's a manufactured situation because of the `Launch` initial value of `200` and `time` initial value of `time_min`.
> It works but not always.

## solution with bad starting point
lets do it again. Now with my favorite value and the only number I seem to remember
{% highlight python %}
# Launch date: days starting from now
Launch = nn.Parameter(ft([0]))
# time spent in flight
time = nn.Parameter(ft([0]))
{% endhighlight %}
`Launch date=0.000000 	flighttime=0.000000	 distance=78.299988 	speed=inf`

LINK TO CODE

{% include videoPlayer.html file="tomars/toMars4.mp4" %}

**Ouff! We should launch right now with an infinite speed!**
* But why now? Well turns out our planetary models starting position (at Launch day 0) is in position of closest approach, meaning there is no closer distance to optimize toward.
* But why infinite speed? Because we made the sin of dividing by 0 in the print function. 
lets do it again. Now lets chose a not perfect Launch day, and prevent a division by 0.

{% highlight python %}
# Launch date: days starting from now
Launch = nn.Parameter(ft([10]))
# time spent in flight
time = nn.Parameter(ft([0.1]))
{% endhighlight %}

LINK TO CODE

{% include videoPlayer.html file="tomars/toMars5.mp4" %}
Well this is better but not within the speed limit of *0.5 M km/day* 
We now have different options:
* Go on and search for good starting parameters where everything is within bounds, like the example with good parameter initialization. This is not really reliable but maybe good enough. 
* Set the speed to constant max, this might be okay or might cause us to miss a good solution.
* Change the Optimization target, not just optimize for minimal distance but also minimal speed. Since in space changing speed means spending fuel(delta v), maybe it is a good idea to minimize it too.
* We can add speed constraints to the optimization with Karnush-Kuhn-Tucker. And then try to get to mars at minimal time. 

> This solution is not useful. 

## Constant max speed
lets see what happens if we just use the max speed in our model.
Since we know we travel in a straight line and at a constant speed we get a distance. 
This distance needs to be the same as the one we get from the space between the planets. Therefore we need to minimize the delta between the two. 

{% highlight python %}
# Launch date: days starting from now
Launch = nn.Parameter(ft([0]))
# time spent in flight
time = nn.Parameter(ft([0]))
{% endhighlight %}
We will use zero initialization. 

{% highlight python %}
def distanceMaxSpeed(time):
  return time*max_v

opt = Adam([Launch,time],lr = 1)

for i in range(1000):
  opt.zero_grad()
  delta = torch.abs(distance(Launch, time) - distanceMaxSpeed(time))
  delta.backward()      
  opt.step()

flyDistance = distance(Launch, time)
print('Launch date=%f \tflighttime=%f\t distance=%f \tspeed=%f'%(Launch, time, flyDistance, flyDistance/time))
{% endhighlight %}
`Launch date=173.894073 	flighttime=157.322617	 distance=78.611275 	speed=0.499682`

LINK TO CODE

{% include videoPlayer.html file="tomars/toMars6.mp4" %}
Nice, it worked! Notice that this one is better than the original (10 days less flying). We can also start to search future opportunities. 
{% include videoPlayer.html file="tomars/toMars7.mp4" %}
Not too surprising since we use `sin`, `cos` like in the optimizer introduction. Or in other words: Planets tend to come back around.

> This solution is not guaranteed to be the best. 

## Minimal Speed
What if we want to save fuel? Fuel is used to accelerate and brake the spaceship, the less fuel we use the less speed we are going to have. 
We calculate the speed and add it to our loss function, so it should be minimized. Lets see what happens here.

{% highlight python %}
opt = Adam([Launch,time],lr = 1)

for i in range(20000):
  opt.zero_grad()
  Distance =distance(Launch, time)
  speed = Distance/time
  loss = Distance + speed
  loss.backward()      
  opt.step()

flyDistance = distance(Launch, time)
print('Launch date=%f \tflighttime=%f\t distance=%f \tspeed=%f'%(Launch, time, flyDistance, flyDistance/time))
{% endhighlight %}
`Launch date=1018.749268 	flighttime=897.419373	 distance=78.299988 	speed=0.087250`

LINK TO CODE

{% include videoPlayer.html file="tomars/toMars8.mp4" %}
A ton of iterations and still not at the optimum. What happened? Since the distance can not be less than the minimal distance between the planets *(78.299 M km)* the only gradient that can be minimized is the speed. Since planets tend to come around again we would just need light push and wait for a long long time eventually we would arrive. That's where this optimization is going. 
In reality there is gravity of all the celestial bodies causing this plan to fail. **This Strategy is not useful. **

> This solution is not useful. 

# Constraint with Karnush-Kuhn-Tucker (KKT)
While it's great to minimize functions with an optimizer we sometimes have constraints that need to be followed, like our speed limit of *0.5 M km/day*. 

There are different ways to do this. One way is to create or change our model so it is impossible to have values outside of our conditions. This is tricky and maybe not always possible. For example we could use speed instead of fly time in our model as a parameter. We then swap the speed with a function that binds it between 0 and 0.5 for any given meta speed parameter like sigmoid. 

$$ speed(metaSpeed) = \frac{maxSpeed}{1+e^{-metaSpeed}}$$ 

A general way to constrain an optimization is the Karnush-Kuhn-Tucker method. It's quite tricky to explain this simply so I would recommend the great chapter [4.4 Constrained Optimization](https://www.deeplearningbook.org/contents/numerical.html) of the [deeplearningbook](https://www.deeplearningbook.org/).

---

Instead of a generalized problem I want to try explain it in applied situations. So let us start with the function to optimize:

$$f(x) = x^2 -x -1 | 1\leq x\leq5$$

Without the constraints $$1\leq x\leq5$$ we would have had the following optimization:

$$ \min_{x} f(x) = \min_{x} x^2 -x -1 $$

We did this already in the introduction of optimization section. But now we have the following inequality constraints and no equality constants:

$$\begin{equation}
   h_0: 1 - x \leq 0 \\
   h_1: x - 5 \leq 0 
\end{equation}$$

---

Now let us look at the generalized Lagrangian:

$$\begin{equation}
   L(x,\lambda, \alpha) = f(x)+\sum_{i} \lambda_{i} g_{i}(x) + \sum_{j} \alpha_{j} h_{j}(x) \\
   f(x) \text{ is the function to optimize} \\
   \sum_{i} \lambda_{i} g_{i}(x) \text{ are the equality constants, we don't need them} \\
   \sum_{j} \alpha_{j} h_{j}(x) \text{ are the inequality constants, we have 2} 
\end{equation}$$

By throwing away the empty sum for the equality constants and filling in the inequality constants we get:

$$\begin{equation}
   L(x,\alpha) = f(x) + lh_{0}(x)+ lh_{1}(x) \\
   L(x,\alpha) = f(x) + \alpha_{0} h_{0}(x)+ \alpha_{1} h_{1}(x) \\
   L(x,\alpha) = x^2 -x -1 + \alpha_{0} (1 - x)+ \alpha_{1}(x-5) \\
\end{equation}$$

And instead of optimizing all:

$$\begin{equation}
   \min_{x} \max_{\lambda} \max_{\alpha,\alpha \geq 0} L(x,\lambda, \alpha)
\end{equation}$$

we only need to optimize:

$$\begin{equation}
   \min_{x} \max_{\alpha,\alpha \geq 0} L(x,\alpha)
\end{equation}$$

Great! I know this can be confusing at first so let us observe two situations with the equations. 

---

Once for within bounds `x=1` and once outside of them `x=0.5` 

$$\begin{equation}
\text{x=1}\\
   L(1,\alpha) = 1^2 -1 -1 + \alpha_{0} (1 - 1)+ \alpha_{1}(1-5)\\
   L(1,\alpha) = -1 + \alpha_{0} (0)+ \alpha_{1}(-4)
\end{equation}$$

Now let us chose an $$a$$ that makes this equation as big as possible:

$$\begin{equation}
   \max_{\alpha,\alpha \geq 0} L(1,\alpha) = \max_{\alpha,\alpha \geq 0} (-1 + \alpha_{0} (0)+ \alpha_{1}(-4) )\\
   \max_{\alpha,\alpha \geq 0} (\alpha_{0} * 0 )\to  \alpha_{0} = \infty \\
   \max_{\alpha,\alpha \geq 0} (\alpha_{1} * -4 )\to  \alpha_{1} = 0
\end{equation}$$

Both constraints have no way to increase the value of the term. Note $$\alpha_{1} = 0$$ because all $$\alpha$$ can only be positive.

But why $$\alpha_{0} = \infty$$? We plan to use gradient descent for our solution, this will cause our x to a bit under `x=1` causing the term to be slightly positive and that is why $$\alpha_{0}$$ will grow rather large, we will see later. 

$$\begin{equation}
   L(1,\alpha) = -1 + \infty*0+ 0*-4 \\
   L(1,\alpha) = -1 = f(1) = f(x)
\end{equation}$$

Both constraints are satisfied and cause the terms to be zero. This is called *complementary slack*. We are left with the original $$f(x)$$. 

---

Now let us observe the situation when a constraint is not satisfied anymore. `x=0.5` is actually the optimal solution without constraints. But with:
 
$$\begin{equation}
\text{x=0.5}\\
   L(0.5,\alpha) = 0.5^2 -0.5 -1 + \alpha_{0} (1 - 0.5)+ \alpha_{1}(0.5-5)\\
   L(0.5,\alpha) = -1.25 + \alpha_{0} (0.5)+ \alpha_{1}(-4.5)
\end{equation}$$

Now let us chose an $$a$$ that makes this equation as big as possible:

$$\begin{equation}
   \max_{\alpha,\alpha \geq 0} L(0.5,\alpha) = \max_{\alpha,\alpha \geq 0} (-1.25 + \alpha_{0} (0.5)+ \alpha_{1}(-4.5) )\\
   \max_{\alpha,\alpha \geq 0} (\alpha_{0} * 0.5 )\to  \alpha_{0} = \infty \\
   \max_{\alpha,\alpha \geq 0} (\alpha_{1} * -4.5 )\to  \alpha_{1} = 0
\end{equation}$$

Note $$\alpha_{1} = 0$$ because all $$\alpha$$ can only be positive.
Now $$\alpha_{0} = \infty$$ because this way the term can be maximized.

$$\begin{equation}
   L(0.5,\alpha) = -1.25 + \infty*0.5+ 0*-4.5 \\
   L(0.5,\alpha) = \infty
\end{equation}$$

The Lagrangian does not end up being equal to $$f(x)$$ because the first constraint is not satisfied. 

---

When using KKT to constrain a problem it is also important to check the following properties:
* The gradient of the generalized Lagrangian is 0. $$\frac{d}{d x} L(x,\lambda, \alpha) = 0$$
* The inequality constraints have _complementary slackness_. $$\alpha * h(x) = 0$$
* $$x$$ satisfies the conditions we set. 


## KKT applied
Let's apply the process of constraint optimization to code and then visualize it to see what happens to the KKT conditions. We will still use the same $$f(x) = x^2-x-1$$ with the constrains $$ 1\leq x \leq 5$$. 

$$\begin{equation}
   f(x) = x^2-x-1 \\
   lh_0(x)= \alpha_{0} (1 - x) \\
   lh_1(x)= \alpha_{1} (x - 5) \\
   \min_{x}  \max_{\alpha, \alpha \geq 0 }  x^2 -x -1 + \alpha_{0} *(1 -x) + \alpha_{1} *(x-5)
   
\end{equation}$$

{% highlight python %}
x = nn.Parameter(ft([3]))
a0 = nn.Parameter(ft([1]))
a1 = nn.Parameter(ft([1]))

def f(x):
  return x**2 -1*x -1

#coditions
def h0(x):
  return 1 -x
  
def h1(x):
  return x -5

#lagrange conditions
def lh0(x, a):
  return torch.abs(a) * h0(x)
  
def lh1(x, a):
  return torch.abs(a) *h1(x)
    
def lagrange(x, a0, a1):
  return lh0(x, a0) +lh1(x,a1)

f_opt = Adam([x], lr=5e-2)
lagrange_opt = SGD([a0, a1], lr=8e-3)

for i in range(120):
  # maxing the a
  for k in range(10):
    lagrange_opt.zero_grad()
    lagrange_loss = -lagrange(x, a0, a1)
    lagrange_loss.backward()
    lagrange_opt.step()
    
  # minimizing x
  for m in range(5):
    f_opt.zero_grad()
    loss = f(x) + lagrange(x, a0, a1)
    loss.backward()
    f_opt.step()
{% endhighlight%}
The following should be noted about the implementation:
* First we introduce all the parameters. In order to keep the `a`'s positive we wrap them in the torch.abs or square them. 
* Since we need to do two optimizations (min, max) we need two optimizers. One for the x and one for the a
* first we max a for a given x. *Notice the sign swap at `-lagrange()` as optimizes only minimize*
* We now minimize the whole generalized Lagrangian. *Notice the `+lagrange()`, no more sign swap*

---

The optimization is successful if we see the following:
* All x constraints satisfied. The first plot (*north*) will show us where x ended up and the lower limit is marked.
* The gradient of the general lagragian is 0. This is shown by the middle middle (*center*) diagram, and to it's side the active components that create it (`f(x)`(*west*) and the `lh0(x)`(*east*))
* Complementary slack = 0. We find this in the lowest part of the diagramm, here in blue. Along with all the gradients of the lagragian (*south*).  

{% include videoPlayer.html file="tomars/toMars9.mp4" %}
Focus on the north diagram.  See that the optimizer fist approaches `x=0.5` but once under the `x=1.0` line it goes back and finally balances on the `x=1.0` bound. Once the optimizer is settled our `x` satisfies all our requirements. 

The west diagram shows the gradient of $$f(x)$$. It shows the steepens of the function $$f()$$ we originally want to optimize. Notice that during the whole time and even at the end this gradient always points to down to the left (lower x).

The eastern diagram shows the gradient of `a0*h0`. 
* At first while x stays within the `x>=1` bound the gradient is 0, this means it has no influence on the optimization during that time. 
* Once x passes the bound it jumps up in an attempt to get x back within its bounds. 
* The x then goes back within its bounds and the gradient decreases again (if x would stay within the bounds it would go back to 0).
*  At the end the the gradient ends up to match the opposite of $$f(x)$$  

The center diagram shows the generalized Lagrangian. It is basically the west and eastern diagram added together. 
* At fist the gradient is following the gradient of the function f.
* Once over the bound it increases so much as to actually cause x to go back up.
* It then goes back and fourth around the bound.
* At the end the gradient settles at 0. This means it found a minima.

The south diagram shows all of the gradients, including $$\frac{d}{dx}lh_1(x)$$ which stays at 0 since x is always staying bellow 5. 
Notice the blue line `slack0:` $$lh_0(x)$$, it is one of the KKT conditions and should be 0 at the end, once x is out of bounds this one shoots up. The slack then settles at 0 once within bounds again.

### KKT applied to a different function
Now let us look at the same diagram but now with the function $$f(x)= x^2 -2.5x +0.5$$ and the constraints $$ 1\leq x \leq 5$$.

{% include videoPlayer.html file="tomars/toMars10.mp4" %}
Here we have a situation where we can see that constraints only influence the optimization while over the boundary. 

## Solution
Now let it put all together and solve our initial problem but now with speed constraints. 

$$0 \leq speed \leq 0.5$$

{% include videoPlayer.html file="tomars/toMars11.mp4" %}

# Conclusions

# Attribution
Original problem [presented here](https://colab.research.google.com/drive/15sg1s9WSkAvXaGJ5genkHi_SeXKT5xES) by:  b2ray2c@gmail.com
