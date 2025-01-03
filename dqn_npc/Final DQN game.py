import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import action_score as score
import re
from classes.game import Person
from classes.magic import Spell
from classes.inventory import Item
from classes.environment import BattleEnv
from classes.agent import DQNAgent
from classes.llm_agent import LLMAgent
from classes.instructor_agent import InstructorAgent
from transformers import AutoTokenizer, T5ForConditionalGeneration


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
tokenizer_instruction = AutoTokenizer.from_pretrained("/Users/macstudio/Desktop/Tesi_magistrale-main/flan-t5-large-ppo")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(device)
model_instruction = T5ForConditionalGeneration.from_pretrained("/Users/macstudio/Desktop/Tesi_magistrale-main/flan-t5-large-ppo").to(device)


def train_dqn(episodes=1000, batch_size=32):
    player_spells = [fire, thunder, blizzard, meteor, cura]
    player_items = [{"item": potion, "quantity": 3}, {"item": grenade, "quantity": 2},
                    {"item": hielixer, "quantity": 1}]
    player1 = Person("Valos", 3260, 132, 300, 34, player_spells, player_items)
    enemy1 = Person("Magus", 5000, 701, 525, 25, [fire, cura], [])

    players = [player1]
    enemies = [enemy1]

    env = BattleEnv(players, enemies)
    agent = DQNAgent(env.state_size, env.action_size)
    llm_agent = LLMAgent(model, tokenizer, device)
    instructor_agent = InstructorAgent(model_instruction, tokenizer_instruction, device)

    rewards_per_episode = []
    agent_wins = []
    enemy_wins = []
    agent_moves_per_episode = []
    success_rate = []
    allucination = 0
    action_scores = []
    total_agent_wins = 0
    agent_wins_after = 0
    agent_win_50 = 0


    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])
        done = False
        total_reward = 0
        moves = 0
        last_enemy_move = "none"
        while not done:
            if e <=500:
                game_description = env.describe_game_state(last_enemy_move)
                print(game_description)

                llm_response = llm_agent.get_best_action(game_description)
                print(f"LLM response: {llm_response}")

                response = instructor_agent.generate_suggestion(game_description, llm_response)
                print(response)

                revised_response = llm_agent.revise_response(game_description, llm_response, response)
                print(f"Final response: {revised_response}")

                action = llm_agent.map_llm_action_to_agent_action(revised_response)

                if action != None:
                    match = re.search(r'\[(.*?)\]', revised_response)
                    total_score = score.calculate_scores(players[0].get_hp(), players[0].get_mp(), enemies[0].get_hp())
                    action_scores.append(round(total_score.get(match.group(1).strip().lower()), 2))
                    score.updage_quantity(match.group(1).strip().lower())
                else:
                    match = re.search(r'\[(.*?)\]', llm_response)
                    total_score = score.calculate_scores(players[0].get_hp(), players[0].get_mp(), enemies[0].get_hp())
                    action_scores.append(round(total_score.get(match.group(1).strip().lower()), 2))
                    score.updage_quantity(match.group(1).strip().lower())
                    allucination += 1
                    action = llm_agent.map_llm_action_to_agent_action(llm_response)
            else:
                action = agent.act(state)

            next_state, reward, done, a_win, e_win, last_enemy_move = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, [1, env.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            moves += 1
            if done:
                print(f"Episode: {e}/{episodes}, Score: {total_reward}, Moves: {moves}, Epsilon: {agent.epsilon}")
                if a_win:
                    agent_wins.append(1)
                    enemy_wins.append(0)
                    total_agent_wins += 1
                    if e > 500:
                        agent_wins_after += 1
                    else:
                        agent_win_50 += 1
                else:
                    agent_wins.append(0)
                    enemy_wins.append(1)

                success_rate.append(total_agent_wins / (e + 1))
                print("Vittorie agente: ", agent_wins.count(1), " Vittorie nemico: ", enemy_wins.count(1))
        rewards_per_episode.append(total_reward)
        agent_moves_per_episode.append(moves)

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    print("Media delle ricompense: ", np.mean(rewards_per_episode))
    print("Media delle mosse: ", np.mean(agent_moves_per_episode))
    print("Media score mosse: ", np.mean(action_scores))
    print("Allucinazioni: ", allucination)
    print("Vittorie prima 500 partite: ", agent_win_50)
    print("Vittorie dopo 500 partite: ", agent_wins_after)

    return rewards_per_episode, agent_wins, enemy_wins, agent_moves_per_episode, success_rate


def plot_training(rewards, agent_wins, enemy_wins, moves, success_rate):
    plt.figure(figsize=(8, 6))
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.savefig("Reward_DQN_final_test.png")
    plt.show()


    plt.figure(figsize=(8, 6))
    cumulative_agent_wins = np.cumsum(agent_wins)
    cumulative_enemy_wins = np.cumsum(enemy_wins)
    plt.plot(cumulative_agent_wins, label="Agent Wins (Cumulative)", color='green')
    plt.plot(cumulative_enemy_wins, label="Enemy Wins (Cumulative)", color='red')

    plt.legend()
    plt.title('Cumulative Wins of Agent vs Enemy per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Wins')
    plt.savefig("Cumulative_Win_DQN_final_test.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(moves)
    plt.title('Number of Moves per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Moves')
    plt.savefig("Moves_DQN_final_test.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(success_rate, label="Success Rate", color='blue')
    plt.title('Success Rate per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.savefig("Success_rate_DQN_final_test.png")
    plt.show()


def export_success_rate(success_rate):
    df = pd.DataFrame({
        "Episode": list(range(1, len(success_rate) + 1)),
        "Success Rate": success_rate
    })
    df.to_csv('success_rate_model_dqn_final_test.csv', index=False)


if __name__ == "__main__":
    # Spells and items setup
    fire = Spell("Fire", 25, 600, "black")
    thunder = Spell("Thunder", 30, 700, "black")
    blizzard = Spell("Blizzard", 35, 800, "black")
    meteor = Spell("Meteor", 40, 1000, "black")
    cura = Spell("Cura", 32, 1500, "white")

    potion = Item("Potion", "potion", "Heals 50 HP", 50)
    hielixer = Item("MegaElixer", "elixer", "Fully restores party's HP/MP", 9999)
    grenade = Item("Grenade", "attack", "Deals 500 damage", 500)

    # Train the agent
    rewards, agent_wins, enemy_wins, moves, success_rate = train_dqn(episodes=1000)
    plot_training(rewards, agent_wins, enemy_wins, moves, success_rate)
    export_success_rate(success_rate)
    print("Vittorie agente: ", agent_wins.count(1), " Vittorie nemico: ", enemy_wins.count(1))