# Tracking with Joint Probability Data Association

## Introduction to Tracking

Tracking is used to continuously associate and estimate the states of an existing set of targets to noisy incoming data over time. Tracking can become very difficult when trying to decide if a new measurement should be associated with an existing target or initiate a new target. We also need to consider the probability of missed detections and false alarms while managing track lifecycles. In order to do this, we want to have an idea of how targets should move with respect to time (formally a state transition function). To properly create a state transition function, we also measure the position and velocity of targets and model measurement uncertainty.

Let $X$ be the state space of the target (like position, velocity). Let $Z$ be the measurement space, space of possible sensor observations (think x,y). Let time represent discrete time steps. Let $x_i(t)$ be the state of target $i$ at time $t$. Let $z(t)$ be the set of measurements received at time $t$. Define a function $f$ where:

$$x_i(t+1) = f(x_i(t)) + w(t)$$

where $w(t)$ is process noise. Define $h: X \rightarrow Z$ as the measurement function, such that:

$$z_j(t) = H(x_i(t)) + v(t)$$

where $v(t)$ is measurement noise. Let $\alpha_t: \{1,\ldots,m\} \rightarrow \{0,\ldots,n\}$ be the data association function mapping measurements to targets at time $t$, where 0 represents false alarms or new targets.

Then we define the tracking problem as:
1. The number of targets at time $t$
2. The positions of targets at each time $t$
3. The data association at each time $t$

We want to maximize:

$$P(n(1:T), x(1:T), \alpha(1:T) \mid z(1:T))$$

## My approach
