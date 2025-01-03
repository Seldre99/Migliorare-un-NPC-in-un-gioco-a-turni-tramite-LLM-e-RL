class InstructorAgent:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device


    def generate_suggestion(self, game_description, large_response):
        input_text = f"Game description: '{game_description}'. Initial response: '{large_response}'."
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, max_length=150, num_beams=3, early_stopping=True)
        suggestion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return suggestion
