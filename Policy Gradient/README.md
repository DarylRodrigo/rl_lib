# Policy Gradients

<!-- Policy gradients are a class of reinforcement learning where the policy (often a neural network controlling the agent) is directly updated. Essentially it tries to make actions which are more likely to deliver high rewards more likely by optimising the policy. -->

## Actor Critic

Policy gradient based methods update the policy based on how the agent has played. It will try and increase the probability of taking good actions and reduce the probability of taking bad actions. Unlike value based methods, which estimate the value of a state, and then choose the best action, policy based methods can handle continuous actions spaces (though also discrete by adding a argmax).

There are several ways of getting the experiences to update the agent with. In the REINFORCE algorithm a monte-carol approach is taken. We let the agent play out an entire game, and then work out what the expected reward was. This will have low bias, because they will be all true events, however high variability because they might lead to completely different outcomes.

A in-between method is to use a value-based approach to guess how much the following steps are worth. For example, you might let the agent take 2 steps and then guess what the expected reward would be. This will then allow you to update the policy. However, because you’re making a guess, you’ll probably have a high bias, because we do not know what a good state is yet. on the flip side, we’ll have low variance as what we think is a good state now we’ll likely also still find a good state later. This means we end up training the agent faster and has less problem converging.

Actor critic is a method where there are two neural networks within the agent:

- **Actor**: This network makes the decisions on what action to take
- **Critic**: This network decides HOW good an action is.

To illustrate this, consider the following example:


<p align="center">
  <img src="./img/intro/11s.png" alt="Drawing" height="300"/>
</p>


If you have an agent which is learning to run as fast as possible and ends up with a score of 11 (seconds to run to the finish line). Then how good is this? Is it the fastest every run because the distance was 1km or is it terrible because it only had to run 10m. This is the job of the critic to decipher, by looking at the past experiences it works out on average how good or bad an action is.

Over time the we’ll learn what a good value state is. This is what the actor citric method is. The actor makes the decisions and the critic tells us how good or bad those actions will be in the future.

## N-step Bootstrapping
