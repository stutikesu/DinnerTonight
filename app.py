from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import random

app = Flask(__name__)

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_bio', methods=['POST'])
def generate_bio():
    # Retrieve form data
    career = request.form.get('career')
    personality = request.form.get('personality')
    interests = request.form.get('interests')
    relationship_goals = request.form.get('relationship_goals')

    # Add some randomness to the prompt by randomly changing the wording
    bio_prompts = [
        f"Create a romantic bio for someone with the following characteristics:\n- Career: {career}\n- Personality: {personality}\n- Interests: {interests}\n- Relationship goals: {relationship_goals}\nThe bio should reflect these attributes in a warm, engaging, romantic way.",
        f"Write a heartfelt bio about a person who is a {career}. They have a {personality} personality and are interested in {interests}. Their relationship goals are: {relationship_goals}. Make it sound romantic and personal.",
        f"Generate a romantic bio for a {career} who is {personality} and enjoys {interests}. They are looking for {relationship_goals}. Make the bio sound warm and affectionate."
    ]

    # Choose a random prompt for variation
    prompt = random.choice(bio_prompts)

    # Tokenize the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Ensure the input length doesn't exceed the model's max token length
    max_length = 150
    if inputs.shape[1] > max_length:
        inputs = inputs[:, :max_length]  # Trim if necessary

    # Generate bio using the model with higher randomness
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=200,  # Set the max length for the generated text
            num_return_sequences=1,  # Generate a single sequence (use >1 if you want more diversity)
            no_repeat_ngram_size=2,
            top_p=0.85,  # Slightly lower top_p for more variety
            top_k=50,    # Lowering top_k for more diverse outputs
            temperature=1.0,  # Increased randomness
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the output and remove the prompt from the generated bio
    bio = tokenizer.decode(outputs[0], skip_special_tokens=True)
    bio = bio[len(prompt):].strip()  # Remove the prompt from the output

    # If the generated bio is empty or too generic, provide a default bio
    if not bio or "list" in bio or "top" in bio:
        bio = "This person is adventurous, caring, and looking for a meaningful connection with someone who shares their values and interests."

    # Render the result page and pass the generated bio
    return render_template('result.html', bio=bio)

if __name__ == '__main__':
    app.run(debug=True)
