import gymnasium as gym
from stable_baselines3 import PPO
import csv


def test_agent(agent, env, episodes=5):
    """Функция для тестирования агента в окружении"""
    for episode in range(episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        while not(terminated or truncated):
            if agent is None:
                action = env.action_space.sample()
            else:
                action, _ = agent.predict(obs)
            result = env.step(action)
            obs, reward, terminated, truncated, info = result
            total_reward += reward
            env.render()
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")



if __name__=='__main__':
    # Создание среды
    env = gym.make('Ant-v4')

    # Создание и обучение RL-агента
    # ppo_agent = PPO('MlpPolicy', env, verbose=1)
    # ppo_agent.learn(total_timesteps=10**6//4)
    # ppo_agent.save("ppo_ant")

    # Для использования определённого RL-агента
    ppo_agent = PPO.load("better_ppo_ant", env)

    env = gym.make('Ant-v4', render_mode='human')

    # Демонстрация работы агентов в среде
    print("Testing Random Agent")
    test_agent(None, env)

    print("Testing RL Agent")
    test_agent(ppo_agent, env)

    env.close()

    # Сохранение данных о работе RL-агента в CSV файл
    demo_states = []
    demo_actions = []
    for _ in range(5):
        obs, _ = env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action, _ = ppo_agent.predict(obs)
            res = env.step(action)
            next_obs, reward, terminated, truncated, info = res
            demo_states.append(obs)
            demo_actions.append(action)
            obs = next_obs
    env.close()

    with open('Ant_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Observations', 'Actions'])
        for obs, action in zip(demo_states, demo_actions):
            writer.writerow([obs.tolist(), action.tolist()])
    print("Демонстрационные данные сохранены в 'Ant_data.csv' с разделителем ','")