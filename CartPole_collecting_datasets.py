import gymnasium as gym
from stable_baselines3 import PPO


class PIDAgent:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0
        self.previous_error = 0

    def get_action(self, observation):
        angle = observation[2]
        error = self.setpoint - angle
        self.integral += error
        derivative = error - self.previous_error
        control = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        action = 1 if control > 0 else 0
        self.previous_error = error
        return action

def test_agent(agent, env, episodes=5):
    """Функция для тестирования агента в окружении"""
    for episode in range(episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        while not (terminated or truncated):
            if agent is pid_agent:
                action = agent.get_action(obs)
            elif agent is None:
                action = env.action_space.sample()
            elif agent is ppo_agent:
                action, _ = agent.predict(obs)
            result = env.step(action)
            obs, reward, terminated, truncated, info = result
            total_reward += reward
            env.render()
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Создание среды
env = gym.make('CartPole-v1')

# # Создание и обучение RL-агента
# ppo_agent = PPO('MlpPolicy', env, verbose=1)
# ppo_agent.learn(total_timesteps=10**5)
# ppo_agent.save("ppo_cartpole")

# Для использования определённого RL-агента
ppo_agent = PPO.load("ppo_cartpole", env)

# Создание и обучение PID-агента
pid_agent = PIDAgent(Kp=0.544, Ki=0, Kd=7)

env = gym.make('CartPole-v1', render_mode='human')

# Демонстрация работы агентов в среде
print("Testing Random Agent")
test_agent(None, env)

print("Testing RL Agent")
test_agent(ppo_agent, env)

print("Testing PID Agent")
test_agent(pid_agent, env)

env.close()