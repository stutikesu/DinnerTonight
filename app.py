from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import random

app = Flask(__name__)

# Load GPT-2 model and tokenizer
model_name = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_bio', methods=['POST'])
def generate_bio():
    # Get user inputs from the form
    career = request.form.get('career')
    personality = request.form.get('personality')
    interests = request.form.get('interests')
    relationship_goals = request.form.get('relationship_goals')

    # Define prompts for the GPT-2 model
    bio_prompts = [
        f"I am a {career} who is {personality}, loves {interests}, and is {relationship_goals}. Craft a romantic and engaging dating bio.",
        f"Create a charming bio for someone who is a {career}, has a {personality} personality, enjoys {interests}, and is focused on {relationship_goals}.",
        f"Describe a {career} with a {personality} vibe, interested in {interests}, and aiming for {relationship_goals}. The bio should be warm and captivating.",
        f"Imagine a dating bio: I'm a {career} with a {personality} personality, passionate about {interests}, and currently {relationship_goals}. Make it romantic and unique."
    ]

    # Randomly select a prompt
    prompt = random.choice(bio_prompts)

    # Encode the prompt for the model
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text with the model
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_p=0.9,
            top_k=50,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated output
    bio = tokenizer.decode(outputs[0], skip_special_tokens=True)
    bio = bio[len(prompt):].strip()

    # Fallback bio in case of empty or non-ideal output
    if not bio or "list" in bio or "top" in bio:
        bio = "I'm an adventurous, caring soul looking for a meaningful connection with someone who shares my values and interests."

    # Render the result page with the generated bio
    return render_template('result.html', bio=bio)

if __name__ == '__main__':
    app.run(debug=True)
