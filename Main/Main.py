import pymunk
import pygame
import pymunk.pygame_util
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from collections import deque, namedtuple

class Env:
    def __init__(self):
        """
        This function initiates everything that is needed for the program
        It creates the window, the ship, the ground, the landing pad, and sets the needed flags for checking
        collisions later
        """
        pygame.init()
        self.width, self.height = 600, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock =  pygame.time.Clock()
        pygame.display.set_caption("Lander DQN Training")
        pymunk.pygame_util.positive_y_is_up = False
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        self.space  = pymunk.Space()
        self.space.gravity = (0, 900)

        self.obstacles = []

        # Ground
        self.ground = pymunk.Segment(self.space.static_body, (0, 550), (600, 550), 5)
        self.ground.elasticity = 0.9
        self.ground.collision_type = 2
        self.space.add(self.ground)

        # Obstacles
        self.generate_obstacles()

        # Ship
        self.ship_body = pymunk.Body(1, pymunk.moment_for_box(1, (40, 20)))
        self.ship_body.position = (300, 200)
        self.ship_shape = pymunk.Poly.create_box(self.ship_body, (40, 20))
        self.ship_shape.collision_type = 1
        self.space.add(self.ship_body, self.ship_shape)

        # Landing pad
        self.landing_pad = pymunk.Segment(self.space.static_body, (250, 530), (350, 530), 5)
        self.landing_pad.elasticity = 0.1
        self.landing_pad.collision_type = 3
        self.space.add(self.landing_pad)

        # Check Flags
        self.ground_collision = False
        self.obstacle_collision = False
        self.collision_happened = False
        self.landing_pad_collision = False
        self.done = False
        self.steps = 0
        self.max_episodes_steps = 1000


    def setupColHandlers(self):
        """
        This function sets up the collision handler needed for pymunk to recognize the collisions
        """
        handlerShipGround = self.space.on_collision(1, 2, begin=self.ship_ground_Collision)
        handlerShipPad = self.space.on_collision(1, 3, begin=self.ship_pad_Collision)
    
    def ship_ground_Collision(self, arbiter, space, data):
        self.ground_collision = True
        return True

    def ship_pad_Collision(self, arbiter, space, data):
        self.landing_pad_collision = True
        return True

    
    def generate_obstacles(self, count=4):
        """
        This will generate 4 obstacles every time the method is called
        using the pymunk library

        """
        for body, shape in getattr(self, "obstacles", []):
            self.space.remove(body, shape)
        self.obstacles = []

        for _ in range(count):
            width = random.randint(30, 80)
            height = random.randint(30, 120)

            while True:
                x = random.randint(width//2, 600 - width//2)
                break

            y = random.randint(350, 500)

            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = x, y
            shape = pymunk.Poly.create_box(body, (width, height))
            shape.elasticity = 0.4
            shape.collision_type = 2

            self.space.add(body, shape)
            self.obstacles.append((body, shape))

    def render(self):
        self.screen.fill((255, 255, 255))
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.clock.tick(60)

    def run(self):
        running = True
        thrusting_Up = False
        thrusting_Left = False
        thrusting_Right = False
    
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        thrusting_Left = True
                    elif event.type == pygame.K_w:
                        thrusting_Up = True
                    elif event.key == pygame.K_d:
                        thrusting_Right = True
                    elif event.key == pygame.K_r:
                        self.reset()
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_a:
                        thrusting_Left = False
                    elif event.key == pygame.K_d:
                        thrusting_Right = False
                    elif event.key == pygame.K_w:
                        thrusting_Up = False
            
            if thrusting_Up:
                self.step(1)
            elif thrusting_Left:
                self.step(2)
            elif thrusting_Right:
                self.step(3)


            print(self.calculateReward)
            self.space.step(1/60.0)
            self.render()
        pygame.quit()

    def reset(self):
        """
        Whenever this method is called it will generate a new environment with a
        random starting position for the ship and all the flags will be set to FALSE
        """
        start_x = random.randint(100, 500)
        self.ship_body.position =  (start_x, 100)
        self.ship_body.angular_velocity = 0

        self.generate_obstacles() 

        self.ground_collision = False
        self.obstacle_collision = False
        self.landing_pad_collision = False
        self.done = False
        self.steps = 0
        self.max_episodes_steps = 1000

    def step(self, action):
        """
        This function describes every possible action the ship can make
        """
        
        self.steps += 1

        if action == 0: #No thrust (Does not move)
            pass
        elif action == 1: # Main Thruster (Move straight)
            self.ship_body.apply_force_at_local_point((0, -1000), (0, 0))
        elif action == 2: # Rotate left
            self.ship_body.apply_force_at_local_point((0, -1000), (15, 0))
        elif action == 3: # Rotate right
            self.ship_body.apply_force_at_local_point((0, -1000), (-15, 0))

        x, y = self.ship_body.position
        vx, vy = self.ship_body.velocity
        ang = self.ship_body.angle
        angVel = self.ship_body.angular_velocity

        reward = self.calculateReward(x, y, vx, vy, ang, angVel)

        done = self.checkDone(x, y, vx, vy, ang)

        return self.getState(), reward, done, {}

    def getState(self):
        """
        This function returns a list of all the state values of the agent (position, velocity, angular velocity, and angle)
        """

        # Parameters for state
        pos = self.ship_body.position
        vel = self.ship_body.velocity
        angVel = self.ship_body.angular_velocity
        ang = self.ship_body.angle
        
        state = [
            pos.x / self.width,
            pos.y / self.height,
            vel.x / 200.0,
            vel.y / 200.0,
            ang / 3.14159,
            angVel / 10.0,
        ]
        min_obstacle_dist = float('inf')
        for body, shape in self.obstacles:
            obstacle_pos = body.position
            dist = ((pos.x - obstacle_pos.x)**2 + (pos.y - obstacle_pos.y)**2)**0.5
            min_obstacle_dist = min(min_obstacle_dist, dist)

        state.append(min_obstacle_dist / 600.0)

        print(min_obstacle_dist)

        return np.array(state, dtype = np.float32)
        
    def calculateReward(self, x, y, vx, vy, angVel, ang): 
        """
        This function calculates the reward from the specific state that the agent is in
        """
        reward = 0

        # Reward for surviving
        reward += 1

        # Penatly for being far away from the landing pad
        landing_pad_center = 300
        distancefromPad = abs(x - landing_pad_center)
        reward -= distancefromPad * 0.01

        # Penalty for high velocities
        reward -= abs(vx) * 0.01
        reward -= abs(vy) * 0.01

        # Penalty for tilting
        reward -= abs(ang) * 10
        reward -= abs(angVel) * 5

        # Rewards / Penalties for landing
        if self.landing_pad_collision:
            if abs(vx) < 30 and abs(vy) < 50 and abs(ang) < 0.3:
                reward += 200 # Landed succesfully
            else:
                reward -= 100 # Crashed
        elif self.ground_collision:
            reward -= 200 # Crashed on the ground
        
        # Out of bounds
        if x < 0 or x > self.width or y > self.height:
            reward -= 100

        # Time penalty
        if self.steps > 800:
            reward -= 2
        
        return reward

    def checkDone(self, x, y, vx, vy, angle):
        """
        This function checks if the episode is done or not
        """
        if (self.ground_collision or self.landing_pad_collision or x < 0 or x > self.width
             or y > self.height or self.steps >= self.max_episodes_steps):
            return True
        return False


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size = 256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class PriotizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0
    
    def push(self, transition):
        max_prio = self.priorities.max() if self.memory else 1.0

        if self.size < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.pos] = transition
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size, beta=0.4):
        if self.size < batch_size:
            return None, None, None
    
        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        # Importance sampling weights
        total = self.size
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
        
    def __len__(self):
        return self.size
    
# This is where the main AI training is done 
class TrainAI:
    def __init__(self):
        self.env = Env()
        self.state_size = len(self.env.getState())
        self.action_size = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Hyperparameters
        self.learning_rate = 0.0005
        self.epsilon_start = 1.0
        self.epsilon_end = 0.02
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start
        self.gamma = 0.99
        self.batch_size = 64
        self.buffer_size = 50000
        self.episodes = 2000
        self.target_update = 10

        # Tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.moving_avg_rewards = []

        # Networks and training
        self.replay_buffer = PriotizedReplayBuffer(self.buffer_size)
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item()
    
    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        samples, indices, weights = self.replay_buffer.sample(self.batch_size)
        if samples is None:
            return
        
        batch = Transition(*zip(*samples))

        state_batch = torch.FloatTensor(np.stack(batch.state)).to(self.device)
        action_batch = torch.FloatTensor(np.stack(action_batch)).to(self.device)
        reward_batch = torch.FloatTensor(np.stack(action_batch)).to(self.device)
        next_state_batch = torch.FloatTensor(np.stack(batch.next_state)).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)
        weights_batch = torch.FloatTensor(weights).to(self.device)

        # Generate current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Generate the  next Q values from target network
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)

        # Computer loss with importance sampling weights
        td_errors = current_q_values.squeeze() - target_q_values
        loss = (weights_batch * td_errors.pow(2)).mean()

        # Optimizer
        self.optimzer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameter(), 1)
        self.optimizer.step()

        # Update priorities
        priorities = torch.abs(td_errors).detach().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, priorities)

    def train(self):
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Target network to siya so make sure na naaupdate siya periodically
        """
        Make training function 
        - update hyperparameters as needed 
        - render (note: yung gameloop sa unahan, yung def run(), baka need mong palitan kasi ginamit ko lang pangtest sa controls yun)
        - tracking of episode rewards and episode lengths (may list na naka instantiate, yung self.episode_rewards tsaka self.episode_lengths, append mo na lang siguro para maupdate)
        """
    
    def save_model(self, filename):
        """
        Make save model function, may function yung torch na pang save (torch.save)
        """
        pass

    def load_model(self, filename):
        """
        As the name suggests, kaya mo na yarn!
        """
        pass

    # def plotting(self):
    """
    if gusto mong iplot pa yung results gaya nung una nating agentic nn, gamitin mo lang matplotlib (plt)
    """

def test_agent(model, env, num_episodes=5, render=True, max_steps=1000):
    """
    eto na yung masaya, kaya mo na yan
    
    """
if __name__ == "__main__":
    env = Env()
    env.run()