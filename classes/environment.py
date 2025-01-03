import numpy as np
import random
from classes.magic import Spell
from classes.inventory import Item

# Spells and items setup
fire = Spell("Fire", 25, 600, "black")
thunder = Spell("Thunder", 30, 700, "black")
blizzard = Spell("Blizzard", 35, 800, "black")
meteor = Spell("Meteor", 40, 1000, "black")
cura = Spell("Cura", 32, 1500, "white")

potion = Item("Potion", "potion", "Heals 50 HP", 50)
hielixer = Item("MegaElixer", "elixer", "Fully restores party's HP/MP", 9999)
grenade = Item("Grenade", "attack", "Deals 500 damage", 500)


# Environment setup
class BattleEnv:
    def __init__(self, players, enemies):
        self.players = players
        self.enemies = enemies
        self.state_size = len(self.get_state())
        self.action_size = self.get_action_size()
        self.done = False

    def get_state(self):
        state = []
        # Add player and enemy stats (HP, MP, spells, items)
        for player in self.players:
            state.extend([player.get_hp(), player.get_mp()])
            for spell in player.magic:
                state.append(1 if player.get_mp() >= spell.cost else 0)  # Can cast spell
            for item in player.items:
                state.append(item["quantity"])  # Quantity of items left
        if len(self.enemies) == 0:
            state.extend([0, 0])
        for enemy in self.enemies:
            state.extend([enemy.get_hp(), enemy.get_mp()])
        print(f"State: {state}, Length: {len(state)}")
        return np.array(state)

    def get_action_size(self):
        # Total actions: attack (1), spells (len(player.magic)), items (len(player.items))
        actions = 1  # Attack
        for player in self.players:
            actions += len(player.magic)  # Each spell is a separate action
            actions += len(player.items)  # Each item is a separate action
        return actions

    def reset(self):
        self.done = False
        player_spells = [fire, thunder, blizzard, meteor, cura]
        player_items = [{"item": potion, "quantity": 3}, {"item": grenade, "quantity": 2},
                        {"item": hielixer, "quantity": 1}]
        for player in self.players:
            player.hp = player.maxhp
            player.mp = player.maxmp
            player.magic = player_spells
            player.items = player_items
        for enemy in self.enemies:
            enemy.hp = enemy.maxhp
            enemy.mp = enemy.maxmp

        return self.get_state()

    def step(self, action):
        reward = 0
        agent_win = False
        enemy_win = False


        # Agent choise
        if action == 0:
            for player in self.players:
                dmg = player.generate_damage()
                enemy = self.enemies[0]
                enemy.take_damage(dmg)
                reward += 25

        elif action > 0 and action <= len(self.players[0].magic):
            player = self.players[0]
            spell = player.magic[action - 1]
            if player.get_mp() >= spell.cost:
                magic_dmg = spell.generate_damage()
                player.reduce_mp(spell.cost)
                if spell.type == "white":
                    player.heal(magic_dmg)
                else:
                    enemy = self.enemies[0]
                    enemy.take_damage(magic_dmg)
                reward += 15

        elif action > len(self.players[0].magic):
            item_index = action - len(self.players[0].magic) - 1
            player = self.players[0]
            item = player.items[item_index]["item"]
            if player.items[item_index]["quantity"] > 0:
                player.items[item_index]["quantity"] -= 1
                if item.type == "potion":
                    player.heal(item.prop)
                    reward += 15
                elif item.type == "attack":
                    enemy = self.enemies[0]
                    enemy.take_damage(item.prop)
                    reward += 15
                elif item.type == "elixer":
                    player.hp = player.maxhp
                    player.mp = player.maxmp
                    reward += 50


        # Check for battle ended
        if self.enemies[0].get_hp() <= 0:
            self.done = True
            reward += 100
            agent_win = True
            return self.get_state(), reward, self.done, agent_win, enemy_win, "No action"

        # Enemy choise (Random)
        for enemy in self.enemies:
            enemy_choice = random.choice(['attack', 'magic'])
            if enemy_choice == 'attack':
                target = random.choice(self.players)
                enemy_dmg = enemy.generate_damage()
                target.take_damage(enemy_dmg)
            elif enemy_choice == 'magic':
                spell, magic_dmg = enemy.choose_enemy_spell()
                enemy_choice = spell.name
                if enemy.get_mp() >= spell.cost:
                    enemy.reduce_mp(spell.cost)
                    if spell.type == "white":
                        enemy.heal(magic_dmg)
                    else:
                        target = random.choice(self.players)
                        target.take_damage(magic_dmg)

        #  Check for battle ended
        if all(p.get_hp() <= 0 for p in self.players):
            self.done = True
            reward -= 100  # Penalità per la sconfitta
            enemy_win = True

        return self.get_state(), reward, self.done, agent_win, enemy_win, enemy_choice

    def describe_game_state(self, last_enemy_move):
        state_description = ""

        for player in self.players:
            state_description += f"Player {player.name} has {player.get_hp()} Health Points (hp) and {player.get_mp()} Magic Points (mp). "

        for enemy in self.enemies:
            state_description += f"Enemy {enemy.name} has {enemy.get_hp()} Health Points (hp) and {enemy.get_mp()} Magic Points (mp). "

        actions_description = f"Available actions: attack deals 300 enemy's hp and removes 0 player's mp; "

        player = self.players[0]

        if player.get_mp() >= fire.cost:
            fire_spell = f"fire spell deals 600 enemy's hp and removes 25 player's mp; "
            actions_description += fire_spell
        if player.get_mp() >= thunder.cost:
            thunder_spell = f"thunder spell deals 700 enemy's hp and removes 30 player's mp; "
            actions_description += thunder_spell
        if player.get_mp() >= blizzard.cost:
            blizzard_spell = f"blizzard spell deals 800 enemy's hp and removes 35 player's mp; "
            actions_description += blizzard_spell
        if player.get_mp() >= meteor.cost:
            meteor_spell = f"meteor spell deals 1000 enemy's hp and removes 40 player's mp; "
            actions_description += meteor_spell
        if player.get_mp() >= cura.cost:
            cura_spell = f"cura spell heals 1500 player's hp and removes 32 player's mp; "
            actions_description += cura_spell
        if player.items[0]["quantity"] > 0:
            potion = f"potion heals 50 player's hp and there are {player.items[0]['quantity']}; "
            actions_description += potion
        if player.items[1]["quantity"] > 0:
            grenade = f"grenade deals 500 enemy's hp and there are {player.items[1]['quantity']}; "
            actions_description += grenade
        if player.items[2]["quantity"] > 0:
            elixer = f"elixer fully restores player's hp and mp and there are {player.items[2]['quantity']}. "
            actions_description += elixer

        last_move_description = f"Last enemy move was {last_enemy_move}."

        game_description = state_description + actions_description + last_move_description
        return game_description

