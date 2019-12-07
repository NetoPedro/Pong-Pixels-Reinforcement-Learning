# Pong-Pixels-Reinforcement-Learning

Using Q-Learning to teach an agent how to play Pong from the pixels. Final project to the course [ELEC-E8125 - Reinforcement learning](https://mycourses.aalto.fi/course/view.php?id=24753) at Aalto University, Finland.


## Overview 

Reinforcement learning has been growing for a few years and giving some amazing new techniques that were thought to be impossible or really hard. It has evolved from simple tasks requiring domain knowledge, to other more abstract ones, and finally it ended up beating humans on human games. For this project, the idea is to develop an agent that can learn how to play the game of Pong, not only that, but it shall start without any knowledge, about what is the ball, the paddles and how a point is scored. It shall learn from the frame pixels, similarly to the vision captured by the human eye from the game and decide between 3 possible actions: UP, DOWN and STAY.

![Agent Playing](https://raw.githubusercontent.com/NetoPedro/Pong-Pixels-Reinforcement-Learning/master/images/image_2.png)

## External Sources 

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)

- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)


## Approach and methods

In this section, the methods used will be described with some detail and explained why and how they work. Finally it presents also the final pseudo code used to train the agent. 

### Preprocessing

Most of the preprocessing was based on the [techniques presented by Deep Mind](https://arxiv.org/pdf/1312.5602.pdf), and it consists in receiving the raw pixels from the game converting them from RGB to a grayscale representation, while resizing it to 84x84. This resolution is thought to be used on the specific network architecture for this problem. 

This techniques allow the network to have less information to learn, since the colours are not important in this problem, and gray scale reduces the size of a frame by a factor of 3. Furthermore, the space reduction of the resizing allows the image to be compact and at same time to hold enough information so the network can predict from. 

However, these spatial reductions are not enough, a single static image has very low information that can be used by the network. Thus, by stacking several layers a few other physic laws are included in the input. For example, if we stack two consecutive frames together it is possible to determine the speed of the ball. Adding a third, allows to determine the acceleration. DeepMind stacks four consecutive layers, therefore that is also the number used here. 

### Deep Q Learning

Deep Q Learning is an off-policy algorithm that combines the use of regular Q learning with deep neural networks. The implementation is very similar, with the agent observing the environment, estimating the Q values for each possible action and selecting the action that maximizes that value.  Throughout the training process, these Q values are at each step compared with the discounted rewards until the end of the episode. This comparison is the ground truth used to perform an update in the way these Q values are calculated. With deep Q learning, a deep neural network is responsible to estimate the Q values, and the update step is done using automatic differentiation. These networks perform only an estimation of the value of taking each action, nevertheless, they also add the possibility to use continuous state spaces, something that is not possible with regular Q learning.  

Since the Q values are estimated by the network, the discounted rewards follow the same estimation pattern. However, it has been shown that this leads to an overestimation of the value of a specific action, possible leading the agent to be stuck on local optimums. Hence, and an alternative is to have a second deep neural network, a target network, dedicated to estimating the Q values for the next states. This target network should have its weights copied from the main network after some number of steps. 

### Experience Replay

A reinforcement learning agent is frequently challenging to train, and in some cases, it is also sample inefficient. To tackle this problem the update step can be performed not on consecutive, correlated samples, but on a batch of samples randomly selected from previous observations. This helps the model to learn, but it is only possible because the Deep Q learning algorithm proposed is an off-policy algorithm that does not try to enforce a policy, but the Q values, which are related to pairs state-action. 

### Convolutional Neural Network

Convolutional neural networks are a special case of Artificial Neural Networks that perform specially well with images. This is due to the fact that this networks usually have less parameters than one with only linear layers, and they can correlated pixels near. Each convolutional layer maps to the next one with the use of mask of a specific size. To that masks we call kernels or sliding windows. Our architecture is the [network proposed by DeepMind](https://arxiv.org/pdf/1509.06461.pdf), and uses an RMSprop optimizer to update the weights at each update step. 




## Results and performance analysis

| Colour  | Experience Replay Size | Target Update Frequency  | Exploration episodes | WinRate Simple AI  | 
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Pink  | 40000 | 1000  | 100000  | \[65;70\[  | 
| Green  | 40000  | 250  | 100000  | \[65;70\[   | 
| Gray  | 10000  | 1000  | 100000  | \[75;85\[  | 
| Orange  | 10000  | 250  | 100000  | \[65;70\[  | 
| **Blue**  | **100000**  | **250**  | **100000**  | **\[85;95\[**  | 
| Red  | 70000  | 250  | 250000  | \[75;80\[  | 

#### Other Hyperparameters 


- Learning Rate: 0.0001
- Batch Size: 32
- Frame Stack Size: 4
- Discount Factor: 0.99
- Frame Size: 84x84x1 (Gray Scale)
- Update Frequency: After 4 steps
- Random Play Initial Memory: 9900

#### Score Explained

- Score: -21; Agent: 0 points; SimpleAI: 21 points
- Score: -19; Agent: 1 points; SimpleAI: 20 points
- Score: -17; Agent: 2 points; SimpleAI: 19 points
- Score: -15; Agent: 3 points; SimpleAI: 18 points
- Score: -13; Agent: 4 points; SimpleAI: 17 points
- Score: -11; Agent: 5 points; SimpleAI: 16 points
- Score: -9; Agent: 6 points; SimpleAI: 15 points
- Score: -7; Agent: 7 points; SimpleAI: 14 points
- Score: -5; Agent: 8 points; SimpleAI: 13 points
- Score: -3; Agent: 9 points; SimpleAI: 12 points
- Score: -1; Agent: 10 points; SimpleAI: 11 points

On the list above is possible to see how points relate to the score obtained by SimpleAI and the Agent for a total of 21 points played. A further note to the fact that if the score is positive then the points of the SimpleAI and the Agent are swap (e.g. 17; Agent: 19 points; SimpleAI: 2 points).


![Agent Playing](https://raw.githubusercontent.com/NetoPedro/Pong-Pixels-Reinforcement-Learning/master/images/image_2.png)

#### Training plots 

![Train plot1](https://raw.githubusercontent.com/NetoPedro/Pong-Pixels-Reinforcement-Learning/master/images/Reward_score_steos.svg)

![Train plot2](https://raw.githubusercontent.com/NetoPedro/Pong-Pixels-Reinforcement-Learning/master/images/Reward_score.svg)
