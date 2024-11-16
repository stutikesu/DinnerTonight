from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
import random

app = Flask(__name__)

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_bio', methods=['POST'])
def generate_bio():
    
    career = request.form.get('career')
    personality = request.form.get('personality')
    interests = request.form.get('interests')
    relationship_goals = request.form.get('relationship_goals')


    bio_prompts = [
        f"Create a romantic bio for someone with the following characteristics:\n- Career: {career}\n- Personality: {personality}\n- Interests: {interests}\n- Relationship goals: {relationship_goals}\nThe bio should reflect these attributes in a warm, engaging, romantic way.",
        f"Write a heartfelt bio about a person who is a {career}. They have a {personality} personality and are interested in {interests}. Their relationship goals are: {relationship_goals}. Make it sound romantic and personal.",
        f"Generate a romantic bio for a {career} who is {personality} and enjoys {interests}. They are looking for {relationship_goals}. Make the bio sound warm and affectionate."
    ]

    
    prompt = random.choice(bio_prompts)

    
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    
    max_length = 150
    if inputs.shape[1] > max_length:
        inputs = inputs[:, :max_length]  

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=200,  
            num_return_sequences=1, 
            no_repeat_ngram_size=2,
            top_p=0.85,  
            top_k=50,    
            temperature=1.0,  
            pad_token_id=tokenizer.eos_token_id
        )
        
    bio = tokenizer.decode(outputs[0], skip_special_tokens=True)
    bio = bio[len(prompt):].strip()
    
    if not bio or "list" in bio or "top" in bio:
        bio = "This person is adventurous, caring, and looking for a meaningful connection with someone who shares their values and interests."
        
    return render_template('result.html', bio=bio)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

