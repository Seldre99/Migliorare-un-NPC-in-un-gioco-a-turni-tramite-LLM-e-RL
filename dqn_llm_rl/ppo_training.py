import re
import torch
import pandas as pd
from datasets import Dataset
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from transformers import AutoTokenizer
from tqdm import tqdm


def processes_function(examples):
    inputs = [ex for ex in examples['input']]
    targets = [ex for ex in examples['instructions']]

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def calculate_reward(ideal_action, suggested_action):
    match_ideal =  re.search(r'\[(.*?)\]', ideal_action)
    ideal_action = match_ideal.group(1).strip().lower()
    match = re.search(r'\[(.*?)\]', suggested_action)
    if match:
        action = match.group(1).strip().lower()
        if action == "meteor":
            action = "meteor spell"
        elif action == "cura":
            action = "cura spell"
        elif action == "blizzard":
            action = "blizzard spell"
        elif action == "thunder":
            action = "thunder spell"
        elif action == "fire":
            action = "fire spell"

        if ideal_action == action:
            return 5.0  # Positive reward for correct action
        else:
            return -10.0  # Negative reward for non-ideal action or hallucinations
    else:
        return -10


def collators(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("/Users/macstudio/Desktop/Tesi_magistrale-main/flan-t5-large-instruction")
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained("/Users/macstudio/Desktop/Tesi_magistrale-main/flan-t5-large-instruction").to(device)

df = pd.read_csv('/Users/macstudio/Desktop/Tesi_magistrale-main/game_scenarios_dataset_3.csv')

df['input'] = df['prompt'] + " " + df['response']

dataset = Dataset.from_pandas(df[['input', 'instructions']])
tokenizer_dataset = dataset.map(processes_function, batched=True)

ppo_config = PPOConfig(
    learning_rate=5e-7,
    ppo_epochs=1,
    mini_batch_size=1,
    batch_size=1
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
)

generation_kwards = {
    "temperature": 0.4,
    "top_k": 50,
    "top_p": 0.8,
}


def train_ppo(epochs):
    for i in tqdm(range(epochs)):
        for batch in dataset:
            input_tensor = []
            response_tensor = []
            reward_tensor = []
            game_input = batch['input']
            targets = batch['instructions']

            inputs = tokenizer(game_input, return_tensors="pt").to(device).input_ids
            input_tensor.append(inputs[0])

            response = ppo_trainer.generate(inputs[0], **generation_kwards)
            response_tensor.append(response[0])

            response_text = tokenizer.decode(response[0], skip_special_tokens=True)

            reward = calculate_reward(targets, response_text)
            reward_tensor.append((torch.tensor(reward, dtype=torch.float)))

            stats = ppo_trainer.step(input_tensor, response_tensor, reward_tensor)
            print(f"objective/kl: {stats['objective/kl']}")
            print(f"ppo/returns/mean: {stats['ppo/returns/mean']}")
            print(f"ppo/policy/advantages_mean: {stats['ppo/policy/advantages_mean']}")
        print("epoch complete")

    ppo_trainer.save_pretrained("/Users/macstudio/Desktop/Tesi_magistrale-main/flan-t5-large-ppo")

train_ppo(3)

