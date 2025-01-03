import csv
import random


actions = {
    'attack': {'damage': 300, 'mp_cost': 0, 'heal': 0},
    'fire spell': {'damage': 600, 'mp_cost': 25, 'heal': 0},
    'thunder spell': {'damage': 700, 'mp_cost': 30, 'heal': 0},
    'blizzard spell': {'damage': 800, 'mp_cost': 35, 'heal': 0},
    'meteor spell': {'damage': 1000, 'mp_cost': 40, 'heal': 0},
    'cura spell': {'damage': 0, 'mp_cost': 32, 'heal': 1500},
    'potion': {'damage': 0, 'mp_cost': 0, 'heal': 50},
    'grenade': {'damage': 500, 'mp_cost': 0, 'heal': 0},
    'elixer': {'damage': 0, 'mp_cost': 0, 'heal': 'full'}
}


def generate_game_scenario():
    player_hp = random.randint(10, 5000)
    player_mp = random.randint(0, 150)
    enemy_hp = random.randint(10, 5000)
    enemy_mp = random.randint(0, 800)
    available_items = {
        'potion': random.randint(0, 3),
        'grenade': random.randint(0, 2),
        'elixer': random.randint(0, 1)
    }
    last_enemy_move = random.choice(['attack', 'fire spell', 'cura spell'])
    available_actions = {action: info for action, info in actions.items() if player_mp >= info['mp_cost']}

    return {
        'player_hp': player_hp,
        'player_mp': player_mp,
        'enemy_hp': enemy_hp,
        'enemy_mp': enemy_mp,
        'available_items': available_items,
        'last_enemy_move': last_enemy_move,
        'available_actions': available_actions
    }


def generate_response(scenario):
    action = random.choice(list(scenario['available_actions'].keys()))
    return f"The next action to take is [{action}]"


def generate_instructions(scenario, response):
    player_hp = scenario['player_hp']
    player_mp = scenario['player_mp']
    enemy_hp = scenario['enemy_hp']

    if player_hp < 1000 and enemy_hp < 1000:
        # If the player has enough mp, we recommend the best attack
        best_attack = None
        for action, info in scenario['available_actions'].items():
            if info['damage'] > 0:
                if best_attack is None or info['damage'] > actions[best_attack]['damage']:
                    best_attack = action
        if best_attack:
            return f"Consider using [{best_attack}] for more damage."
        else:
            return "Use [attack] since you don't have enough MP."
    elif player_hp < 800 and player_mp < 25:
        if scenario['available_items']['elixer'] > 0:
            return "You should use [elixer] to fully restore health and MP."
        elif player_mp >= 32:
            return "You should use [cura spell] to heal more HP."
        elif scenario['available_items']['potion'] > 0:
            return "You should use a [potion] to heal."
    elif player_hp < 800:
        if player_mp >= 32:
            return "You should use [cura spell] to heal more HP."
        elif scenario['available_items']['potion'] > 0:
            return "You should use a [potion] to heal."
        elif scenario['available_items']['elixer'] > 0:
            return "You should use [elixer] to fully restore health and MP."

    best_attack = None
    for action, info in scenario['available_actions'].items():
        if info['damage'] > 0:
            if best_attack is None or info['damage'] > actions[best_attack]['damage']:
                best_attack = action
    if best_attack:
        return f"Consider using [{best_attack}] for more damage."
    else:
        return "Use [attack] since you don't have enough MP."

    return "NaN"


def generate_dataset(n=5000):
    dataset = []
    for _ in range(n):
        scenario = generate_game_scenario()
        response = generate_response(scenario)
        instructions = generate_instructions(scenario, response)
        prompt = (
            f"Player has {scenario['player_hp']} hp and {scenario['player_mp']} mp. "
            f"Enemy has {scenario['enemy_hp']} hp and {scenario['enemy_mp']} mp. "
            f"Available actions: {', '.join(scenario['available_actions'].keys())}. "
            f"Last enemy move was {scenario['last_enemy_move']}."
        )
        dataset.append({
            'prompt': prompt,
            'response': response,
            'instructions': instructions
        })
    return dataset


def save_dataset_to_csv(dataset, filename='game_scenarios_dataset_3.csv'):
    keys = dataset[0].keys()
    with open(filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)


dataset = generate_dataset()

save_dataset_to_csv(dataset)
