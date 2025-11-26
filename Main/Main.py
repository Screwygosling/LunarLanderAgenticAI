import pymunk
import pygame
import pymunk.pygame_util
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import namedtuple

# ------------------ ENVIRONMENT ------------------
class Env:
    def __init__(self, display=True):
        # display=False for headless/faster training
        self.display = display
        if self.display:
            pygame.init()
        self.width, self.height = 600, 600
        if self.display:
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Lander DQN Training")
            pymunk.pygame_util.positive_y_is_up = False
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        else:
            self.screen = None
            self.clock = None
            self.draw_options = None

        self.space = pymunk.Space()
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

        # Flags
        self.ground_collision = False
        self.landing_pad_collision = False
        self.steps = 0
        self.max_episodes_steps = 1000

        # Collision handlers (uses pymunk's on_collision which works with your pymunk)
        self.space.on_collision(1, 2, begin=self.ship_ground_Collision)
        self.space.on_collision(1, 3, begin=self.ship_pad_Collision)

    def ship_ground_Collision(self, arbiter, space, data):
        self.ground_collision = True
        return True

    def ship_pad_Collision(self, arbiter, space, data):
        self.landing_pad_collision = True
        return True

    def generate_obstacles(self, count=4):
        # remove previous obstacles
        for body, shape in getattr(self, "obstacles", []):
            try:
                self.space.remove(body, shape)
            except Exception:
                pass
        self.obstacles = []
        for _ in range(count):
            width = random.randint(30, 80)
            height = random.randint(30, 120)
            x = random.randint(width // 2, 600 - width // 2)
            y = random.randint(350, 500)
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = (x, y)
            shape = pymunk.Poly.create_box(body, (width, height))
            shape.elasticity = 0.4
            shape.collision_type = 2
            self.space.add(body, shape)
            self.obstacles.append((body, shape))

    def render(self):
        # do nothing if display is disabled
        if not self.display:
            return
        self.screen.fill((255, 255, 255))
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.clock.tick(60)

    # --- UPDATED: Accept seed to reproduce obstacle layout ---
    def reset(self, seed=None):
        # if seed provided, use it to reproduce obstacles
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        start_x = random.randint(100, 500)
        self.ship_body.position = (start_x, 100)
        self.ship_body.velocity = (0, 0)
        self.ship_body.angular_velocity = 0
        self.ship_body.angle = 0
        self.generate_obstacles()
        self.ground_collision = False
        self.landing_pad_collision = False
        self.steps = 0
        return self.getState()

    def step(self, action):
        self.steps += 1
        if action == 1:
            self.ship_body.apply_force_at_local_point((0, -1000), (0, 0))
        elif action == 2:
            self.ship_body.apply_force_at_local_point((0, -1000), (15, 0))
        elif action == 3:
            self.ship_body.apply_force_at_local_point((0, -1000), (-15, 0))

        x, y = self.ship_body.position
        vx, vy = self.ship_body.velocity
        ang = self.ship_body.angle
        angVel = self.ship_body.angular_velocity

        reward = self.calculateReward(x, y, vx, vy, ang, angVel)
        done = self.checkDone(x, y)
        # advance physics
        self.space.step(1 / 60.0)
        return self.getState(), reward, done, {}

    def getState(self):
        pos = self.ship_body.position
        vel = self.ship_body.velocity
        ang = self.ship_body.angle
        angVel = self.ship_body.angular_velocity

        state = [
            pos.x / self.width,
            pos.y / self.height,
            vel.x / 200.0,
            vel.y / 200.0,
            ang / 3.14159,
            angVel / 10.0,
        ]
        min_obstacle_dist = float("inf")
        for body, shape in self.obstacles:
            obstacle_pos = body.position
            dist = ((pos.x - obstacle_pos.x) ** 2 + (pos.y - obstacle_pos.y) ** 2) ** 0.5
            min_obstacle_dist = min(min_obstacle_dist, dist)
        state.append(min_obstacle_dist / 600.0)
        return np.array(state, dtype=np.float32)

    def calculateReward(self, x, y, vx, vy, ang, angVel):
        reward = 1
        landing_pad_center = 300

        # Penalties
        reward -= abs(x - landing_pad_center) * 0.01
        reward -= abs(vx) * 0.01
        reward -= abs(vy) * 0.01
        reward -= abs(ang) * 10
        reward -= abs(angVel) * 5

        # Landing bonus
        if self.landing_pad_collision:
            if abs(vx) < 30 and abs(vy) < 50 and abs(ang) < 0.3:
                reward += 200
            else:
                reward -= 100
        
        # Crash penalty
        elif self.ground_collision:
            reward -= 200

        # Out of bouunds penalty
        if x < 0 or x > self.width or y > self.height:
            reward -= 100
        
        # Small timeout penalty
        if self.steps > 800:
            reward -= 2
        return reward

    def checkDone(self, x, y):
        return (
            self.ground_collision
            or self.landing_pad_collision
            or x < 0
            or x > self.width
            or self.steps >= self.max_episodes_steps
        )


# ------------------ DQN ------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
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


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))


# ------------------ Replay Buffer ------------------
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
        prios = self.priorities[: self.size]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return self.size


# ------------------ AI Training ------------------
class TrainAI:
    def __init__(self):
        # create environment with display disabled for speed
        self.env = Env(display=False)
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
        self.threshold = 200

        # Optimization frequency to reduce overhead (optimize every N env steps)
        self.optimize_every = 4  # speed tradeoff

        self.episode_rewards = []
        # recent_records holds tuples (episode_record, total_reward, seed)
        self.recent_records = []

        # Networks
        self.replay_buffer = PriotizedReplayBuffer(self.buffer_size)
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # internal counter for optimize frequency
        self._env_steps_since_opt = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                # use torch.from_numpy for speed
                if isinstance(state, np.ndarray):
                    state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device).float()
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item()

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        samples, indices, weights = self.replay_buffer.sample(self.batch_size)
        if samples is None:
            return

        # build batch using torch.from_numpy where possible (faster / lower overhead)
        batch = Transition(*zip(*samples))
        state_arr = np.stack(batch.state)
        next_state_arr = np.stack(batch.next_state)
        action_arr = np.array(batch.action, dtype=np.int64)
        reward_arr = np.array(batch.reward, dtype=np.float32)
        done_arr = np.array(batch.done, dtype=np.bool_)

        state_batch = torch.from_numpy(state_arr).to(self.device).float()
        next_state_batch = torch.from_numpy(next_state_arr).to(self.device).float()
        action_batch = torch.from_numpy(action_arr).to(self.device).long().unsqueeze(1)
        reward_batch = torch.from_numpy(reward_arr).to(self.device).float()
        done_batch = torch.from_numpy(done_arr).to(self.device)
        weights_batch = torch.from_numpy(weights).to(self.device).float()

        # Current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()

        # Next Q values (target)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
        target_q_values = reward_batch + self.gamma * next_q_values * (~done_batch)

        td_errors = current_q_values - target_q_values
        loss = (weights_batch * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities
        priorities = torch.abs(td_errors).detach().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, priorities)

    def train(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

        for episode in range(1, self.episodes + 1):
            # choose and save a seed so we can reproduce obstacles for rendering later
            seed = random.randint(0, 2 ** 31 - 1)
            state = self.env.reset(seed=seed)
            done = False
            total_reward = 0
            episode_record = []  # list of (state_list, action)

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                # push transition into replay buffer (store python lists to keep pickling/memory cheaper)
                self.replay_buffer.push(Transition(state.tolist(), int(action), next_state.tolist(), float(reward), bool(done)))
                # record for potential rendering (store list copies)
                episode_record.append((state.tolist(), int(action)))
                state = next_state
                total_reward += reward

                # optimization: only every few steps to save time
                self._env_steps_since_opt += 1
                if self._env_steps_since_opt >= self.optimize_every:
                    self.optimize_model()
                    self._env_steps_since_opt = 0

            # Epsilon decay
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.episode_rewards.append(total_reward)
            # store recent record triple (episode_record, total_reward, seed)
            self.recent_records.append((episode_record, total_reward, seed))

            # keep only last 10 records for averaging
            if len(self.recent_records) > 10:
                # pop the oldest (leftmost)
                self.recent_records.pop(0)

            # Target network update
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Render every 30 episodes (pick episode inside last-10 closest to average)
            if episode % 30 == 0:
                # compute average of the last up to-10 rewards
                last_rewards = [r for (_, r, _) in self.recent_records]
                if len(last_rewards) == 0:
                    avg_reward = 0.0
                else:
                    avg_reward = float(np.mean(last_rewards))
                # pick index within recent_records closest to avg
                diffs = [abs(r - avg_reward) for (_, r, _) in self.recent_records]
                closest_idx = int(np.argmin(diffs))
                episode_record, reward_to_render, seed_to_render = self.recent_records[closest_idx]

                print(f"\nRendering episode block ending at {episode} - picked index {closest_idx} "
                      f"(closest to average reward {avg_reward:.2f})")
                # Enable display temporarily for rendering
                self._render_episode_with_seed(episode_record, seed_to_render)
                print(f"Rendered episode from seed {seed_to_render}\n")

                # delete used data to free memory
                self.recent_records.pop(closest_idx)

            # Print every 10 episodes: show avg reward of the recent_records
            if episode % 10 == 0:
                if len(self.recent_records) > 0:
                    avg_last10 = float(np.mean([r for (_, r, _) in self.recent_records]))
                else:
                    avg_last10 = 0.0
                print(f"Episode {episode}/{self.episodes} - Avg Reward (last 10): {avg_last10:.2f} - Epsilon: {self.epsilon:.3f}")

        print("Training finished!")
        self.plot_training_progress()

    def save_model(self, filename):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.episode_rewards,
        })
    
    def load_model(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loadede from {filename}")

    def plot_training_progress(self):
        # Handle empty data
        if len(self.episode_rewards) == 0:
            print("No episode rewards recorded yet — nothing to plot.")
            return

        plt.figure(figsize=(12, 4))

        # Raw episode rewards
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards, alpha=0.3, label='Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.title('Training Rewards')

        # Moving average (right subplot)
        plt.subplot(1, 2, 2)
        n = len(self.episode_rewards)
        window = min(100, n)

        if window <= 1:
            # Not enough points to smooth — plot raw
            moving_avg = np.array(self.episode_rewards)
            x = np.arange(len(moving_avg))
        else:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window) / window, mode='valid')
            # Align moving average x-axis so each value corresponds to the last index in its window
            x = np.arange(window - 1, n)

        plt.plot(x, moving_avg, label=f'Moving Avg (window={window})', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.title('Moving Average Reward')

        plt.tight_layout()
        plt.show()

        # Summary statistics
        successful_episodes = sum(1 for reward in self.episode_rewards if reward > self.threshold)
        denom = len(self.episode_rewards) if len(self.episode_rewards) > 0 else 1
        success_rate = successful_episodes / denom
        avg_reward = float(np.mean(self.episode_rewards)) if len(self.episode_rewards) > 0 else 0.0

        print(f"Success rate (>{self.threshold}): {success_rate:.3f} ({successful_episodes}/{denom})")
        print(f"Average reward: {avg_reward:.2f}")


    def _render_episode_with_seed(self, episode_record, seed):
        # Ensure display initialized
        if not self.env.display:
            # initialize a display now (user asked window show after training)
            pygame.init()
            self.env.display = True
            self.env.screen = pygame.display.set_mode((self.env.width, self.env.height))
            self.env.clock = pygame.time.Clock()
            pygame.display.set_caption("Lander - Render")
            pymunk.pygame_util.positive_y_is_up = False
            self.env.draw_options = pymunk.pygame_util.DrawOptions(self.env.screen)

        # reset env with same seed to reproduce obstacles
        state = self.env.reset(seed=seed)
        done = False

        # replay actions through the episode_record
        for recorded_state, recorded_action in episode_record:
            # Optionally set ship to recorded_state (we already reset with same seed so obstacles match)
            # It's safer to let env progress naturally from reset; we'll apply the recorded action and step once.
            action = int(recorded_action)
            state, reward, done, _ = self.env.step(action)
            self.env.render()
            # handle pygame events so window remains responsive
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    return
            if done:
                break

        # short pause so user sees result
        if self.env.display:
            pygame.time.delay(600)

        # After rendering we can optionally close the window or keep it.
        # We'll keep it open — caller can call pygame.quit() when finished overall.

if __name__ == "__main__":
    trainer = TrainAI()
    trainer.train()
    # After training you can render more or call pygame.quit() to close display:
    # trainer._render_episode_with_seed(...) or simply:
    pygame.quit()
