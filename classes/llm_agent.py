import re




class LLMAgent:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device


    def get_best_action(self, game_description):
        input_text = f"Given the game state '{game_description}', what is the next action to take? Please write the " \
                     f"chosen action like this example, [attack], " \
                     f"and explain your reasoning briefly."

        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)

        outputs = self.model.generate(**inputs, max_length=150, num_beams=3, early_stopping=True)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        search = re.search(r'\[(.*?)\]', response)
        if search is None:
            match = re.search(r'\b(\w+)\b(?=\s+deals|\s+heals|\s+fully)', response)
            if match:
                action = match.group(1)
                if not response.startswith(f"[{action}]"):
                    response = response.replace(action, f"[{action}]", 1)

        response = re.sub(r'^\[\[(\w+\s*\w*)\]', r'[\1]', response)
        response = re.sub(r'(\w+)\s+\[spell\]', r'[\1 spell]', response)

        return response


    def revise_response(self, game_description, initial_response, suggestions):
        input_text = f"Given the game state '{game_description}'. Your initial response was '{initial_response}'. Considering " \
                     f"this suggestion '{suggestions}', rephrase your answer"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, max_length=150, num_beams=3, early_stopping=True)
        revised_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        search = re.search(r'\[(.*?)\]', revised_response)
        revised_response = re.sub(r'^\[\[(\w+\s*\w*)\]', r'[\1]', revised_response)
        if search is None:
            match = re.search(r'\b(\w+)\b(?=\s+deals|\s+heals|\s+fully)', revised_response)
            if match:
                action = match.group(1)
                if not revised_response.startswith(f"[{action}]"):
                    revised_response = revised_response.replace(action, f"[{action}]", 1)
        revised_response = re.sub(r'\[meteor heals]', '[meteor spell]', revised_response)
        return revised_response


    def map_llm_action_to_agent_action(self, llm_response):
        match = re.search(r'\[(.*?)\]', llm_response)
        if match:
            action = match.group(1).strip().lower()
            if action == "attack":
                return 0
            elif action == "fire spell" or action == "fire":
                return 1
            elif action == "thunder spell" or action == "thunder":
                return 2
            elif action == "blizzard spell" or action == "blizzard":
                return 3
            elif action == "meteor spell" or action == "meteor":
                return 4
            elif action == "cura spell" or action == "cura":
                return 5
            elif action == "potion":
                return 6
            elif action == "grenade":
                return 7
            elif action == "elixer":
                return 8
        return None
