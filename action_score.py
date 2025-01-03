actions = {
    'attack': {'damage': 300, 'mp_cost': 0, 'heal': 0, 'quantity': 1},
    'fire spell': {'damage': 600, 'mp_cost': 25, 'heal': 0, 'quantity': 1},
    'thunder spell': {'damage': 700, 'mp_cost': 30, 'heal': 0, 'quantity': 1},
    'blizzard spell': {'damage': 800, 'mp_cost': 35, 'heal': 0, 'quantity': 1},
    'meteor spell': {'damage': 1000, 'mp_cost': 40, 'heal': 0, 'quantity': 1},
    'cura spell': {'damage': 0, 'mp_cost': 32, 'heal': 1500, 'quantity': 1},
    'potion': {'damage': 0, 'mp_cost': 0, 'heal': 50, 'mp_heal': 0, "quantity": 3},
    'grenade': {'damage': 500, 'mp_cost': 0, 'heal': 0, 'quantity': 2},
    'elixer': {'damage': 0, 'mp_cost': 0, 'heal': 3260, 'mp_heal': 132, 'quantity': 1}
}

def updage_quantity(action):
    for v, param in actions.items():
        if v == action:
            actions[v]['quantity'] -= 1

def calculate_scores(hp_player, mp_player, hp_enemy):
    scores = {}
    max_score = 0

    for action, params in actions.items():
        damage = params['damage']
        mp_cost = params['mp_cost']
        heal = params.get('heal', 0)
        quantity = params['quantity']

        if damage > 0:
            score = (0.6 * (damage / hp_enemy)) - (0.4 * (mp_cost / mp_player))
        elif heal > 0 and hp_player <= 1000:
            score = (0.6 * 0) + (0.4 * ((heal + params.get('mp_heal', 0)) / (hp_player + mp_player))) - (0.4 * (mp_cost / mp_player))

        if action != "attack" and quantity == 0 or action != "attack" and mp_cost > mp_player:
            score = 0

        if score < 0:
            score = 0.1

        scores[action] = score
        if score > max_score:
            max_score = score

    normalized_score = {action: score / max_score if max_score > 0 else 0 for action, score in scores.items()}
    return normalized_score

