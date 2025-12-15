import os
import pandas as pd
import json  # <--- FIX 1: Added missing import
import re    # <--- FIX 2: Added regex for robust parsing
from flask import Flask, request, jsonify, send_from_directory, render_template
from langchain_ollama import ChatOllama
from langchain_experimental.tools import PythonAstREPLTool
from langchain.agents import create_agent

app = Flask(__name__)

# --- CONFIGURATION ---
GLB_FOLDER = os.path.join(os.getcwd(), 'GLB')
CSV_FILE = 'inventory.csv'

# --- 1. SETUP DATA & CONTEXT ---
try:
    df = pd.read_csv(CSV_FILE)
    
    # Preprocess: Ensure categories are actual lists
    if isinstance(df['category'].iloc[0], str):
        # Handle potential whitespace after commas
        df['category'] = df['category'].apply(lambda x: [item.strip() for item in x.split(',')])
        
except (FileNotFoundError, KeyError, IndexError):
    print("Warning: inventory.csv not found or invalid. Using dummy data.")
    df = pd.DataFrame({
        "fullId": ["dummy_chair_id", "dummy_couch_id"], 
        "category": [["Chair", "Office"], ["Couch", "Living"]]
    })

# Prepare Overview for the LLM
all_tags = sorted(list(set(tag for sublist in df['category'] if isinstance(sublist, list) for tag in sublist)))
tag_context = ", ".join(all_tags[:150]) 

# --- 2. INIT LLM ---
llm = ChatOllama(model="llama3.2:3b", temperature=0)

# --- 3. FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model/<fullid>')
def get_model(fullid):
    if '..' in fullid or '/' in fullid:
        return "Invalid ID", 400
    filename = f"{fullid}.glb"
    try:
        return send_from_directory(GLB_FOLDER, filename)
    except Exception:
        return "Model not found", 404

@app.route('/prompt', methods=['POST'])
def handle_prompt():
    user_text = request.json.get('prompt')
    print(f"Received prompt: {user_text}")    
    
    # 1. Ask LLM to output JSON
    extraction_prompt = f"""
    You are an inventory matcher. 
    1. Identify the user's desired item.
    2. Find the SINGLE best matching tag from this list: [{tag_context}]
    3. Determine if they want a random/different one (true/false).
    
    Return ONLY a JSON object with keys "tag" and "random". Do not write any other text.
    
    Example User: "I want a seat"
    Example JSON: {{"tag": "Chair", "random": false}}
    
    User: {user_text}
    """
    
    # Quick inference
    response = llm.invoke(extraction_prompt).content
    print(f"LLM Raw Response: {response}") # Debugging print

    # 2. Robust Parsing and Pandas Logic
    try:
        # FIX 3: Use Regex to find the JSON object (ignores "Here is your JSON..." chatter)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in response")
            
        data = json.loads(json_match.group())
        target_tag = data.get('tag')
        is_random = data.get('random', False)
        
        if not target_tag:
             return jsonify({"message": "I couldn't identify a valid category."})

        # Fast Pandas Filter
        # We allow partial matches (e.g., finding 'Chair' inside ['Chair', 'OfficeChair'])
        matches = df[df['category'].apply(lambda x: target_tag in x)]
        
        if matches.empty:
            return jsonify({"message": f"I understood '{target_tag}', but I do not have any in stock."})
            
        # Select
        if is_random:
            selected = matches.sample(1)
        else:
            selected = matches.head(1)
            
        full_id = selected['fullId'].values[0]
        return jsonify({"fullId": full_id, "message": f"Loading {target_tag}..."})
        
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"message": "I couldn't understand that."})

if __name__ == '__main__':
    os.makedirs(GLB_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)