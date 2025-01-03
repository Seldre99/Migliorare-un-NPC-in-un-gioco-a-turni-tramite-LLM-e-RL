import random
import pandas as pd

# Definire alcune azioni e i loro effetti
actions = {
    "attack": {"damage": 300, "mp_cost": 0},
    "fire spell": {"damage": 600, "mp_cost": 25},
    "thunder spell": {"damage": 700, "mp_cost": 30},
    "blizzard spell": {"damage": 800, "mp_cost": 35},
    "meteor spell": {"damage": 1000, "mp_cost": 40},
    "cura spell": {"heal": 1500, "mp_cost": 32, "damage": 0},
    "potion": {"heal": 50, "mp_cost": 0, "quantity": 3, "damage": 0},
    "grenade": {"damage": 500, "mp_cost": 0, "quantity": 2},
    "elixir": {"heal": "full", "mp_cost": 0, "mp_restore": "full", "quantity": 1, "damage": 0},
}


# Funzione per generare un prompt casuale
def generate_prompt(player_hp, player_mp):
    enemy_hp = random.randint(1000, 6000)
    last_enemy_move = random.choice(list(actions.keys()))

    # Assicurarsi che ci sia almeno un'azione di cura
    healing_actions = ["cura spell", "potion", "elixir"]
    available_actions = random.sample(list(actions.keys()), k=5)  # Campionare 5 azioni
    available_actions.append(random.choice(healing_actions))  # Aggiungere una cura

    available_actions_str = "; ".join(
        [
            f"{action} deals {actions[action]['damage']} enemy's hp and removes {actions[action]['mp_cost']} player's mp"
            if "damage" in actions[action]
            else f"{action} heals {actions[action]['heal']} player's hp and removes {actions[action]['mp_cost']} player's mp"
            for action in available_actions
        ]
    )

    prompt = (
        f"Player Valos has {player_hp} Health Points (hp) and {player_mp} Magic Points (mp). "
        f"Enemy Magus has {enemy_hp} Health Points (hp). "
        f"Available actions: {available_actions_str}. Last enemy move was {last_enemy_move}."
    )

    return prompt, available_actions, player_hp, player_mp, enemy_hp


# Funzione per generare una risposta meno ottimale
def generate_response(available_actions, player_mp):
    chosen_action = None
    valid_actions = [action for action in available_actions if actions[action].get("mp_cost", 0) <= player_mp]
    if len(valid_actions) == 0:
        chosen_action = "attack"
    else:
        chosen_action = random.choice(valid_actions)
    if chosen_action not in ["cura spell", "potion", "elixir"]:
        return f"[{chosen_action}] deals {actions[chosen_action]['damage']} enemy's hp and removes {actions[chosen_action]['mp_cost']} player's mp. The next action to take is [{chosen_action}]."
    else:
        return f"[{chosen_action}] heals {actions[chosen_action]['heal']} player's hp and removes {actions[chosen_action]['mp_cost']} player's mp. The next action to take is [{chosen_action}]."


# Funzione per generare le istruzioni basate su una scelta migliore
def generate_instructions(chosen_action, available_actions, player_hp, player_mp, instruction_type):
    # Tipo di istruzione: cura o attacco
    if instruction_type == "heal":
        healing_options = [
            action
            for action in available_actions
            if "heal" in actions[action]
               and (
                       actions[action]["heal"] == "full"
                       or actions[action]["heal"] >= 500
               )
               and (actions[action].get("quantity", 1) > 0)
               and (actions[action]["mp_cost"] <= player_mp)
        ]
        if healing_options:
            best_heal = max(
                healing_options,
                key=lambda a: actions[a]["heal"]
                if isinstance(actions[a]["heal"], int)
                else float("inf"),
            )
            if "heal" in actions[chosen_action]:
                return f"The current action [{chosen_action}] is a good choice, as it restores {actions[chosen_action]['heal']} hp."
            else:
                return f"Consider using [{best_heal}] to heal, as it restores more hp ({actions[best_heal]['heal']} player's hp)."
        else:
            return "There are no healing options available."

    # Altrimenti suggerire un attacco
    else:
        better_attack_options = [
            action
            for action in available_actions
            if "damage" in actions[action]
               and actions[action]["damage"] > actions[chosen_action]["damage"]
               and actions[action]["mp_cost"] <= player_mp
               and (actions[action].get("quantity", 1) > 0)
        ]

        if better_attack_options:
            best_attack = max(better_attack_options, key=lambda a: actions[a]["damage"])
            if best_attack == "grenade":
                return f"Consider using [{best_attack}] which deals more damage ({actions[best_attack]['damage']} enemy's hp) and there are {actions[best_attack]['quantity']}."
            else:
                return f"Consider using [{best_attack}] which deals more damage ({actions[best_attack]['damage']} enemy's hp) and removes {actions[best_attack]['mp_cost']} mp."
        else:
            return "The action taken seems reasonable given the current state."


# Funzione per generare il dataset
def generate_dataset(num_examples):
    dataset = []

    # Generare 2500 esempi di attacco
    for _ in range(num_examples // 2):
        # Salute alta per suggerire attacco
        player_hp = random.randint(2000, 5000)
        player_mp = random.randint(0, 200)
        prompt, available_actions, player_hp, player_mp, enemy_hp = generate_prompt(player_hp, player_mp)
        response = generate_response(available_actions, player_mp)
        chosen_action = response.split("[")[1].split("]")[0]
        instructions = generate_instructions(chosen_action, available_actions, player_hp, player_mp,
                                             instruction_type="attack")

        example = {
            "prompt": prompt,
            "response": response,
            "instructions": instructions
        }
        dataset.append(example)

    # Generare 2500 esempi di cura
    for _ in range(num_examples // 2):
        # Salute bassa per suggerire cura
        player_hp = random.randint(1, 1499)
        player_mp = random.randint(0, 200)
        prompt, available_actions, player_hp, player_mp, enemy_hp = generate_prompt(player_hp, player_mp)
        response = generate_response(available_actions, player_mp)
        chosen_action = response.split("[")[1].split("]")[0]
        instructions = generate_instructions(chosen_action, available_actions, player_hp, player_mp,
                                             instruction_type="heal")

        example = {
            "prompt": prompt,
            "response": response,
            "instructions": instructions
        }
        dataset.append(example)

    return dataset


# Generare un dataset di 5000 esempi (2500 attacchi e 2500 cure)
dataset = generate_dataset(5000)

# Convertire il dataset in un DataFrame
df = pd.DataFrame(dataset)

# Mischiare il DataFrame per mescolare le istanze di attacco e cura
df = df.sample(frac=1).reset_index(drop=True)

# Salvare il dataset come file CSV
df.to_csv("game_reasoning_dataset_with_balanced_attacks_and_heals.csv", index=False)

# Visualizzare qualche esempio
for i in range(5):
    print(f"Example {i + 1}")
    print("Prompt:", dataset[i]["prompt"])
    print("Response:", dataset[i]["response"])
    print("Instructions:", dataset[i]["instructions"])
    print("\n")
