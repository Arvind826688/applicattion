from flask import Flask, request, jsonify, render_template
import random
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Step 1: Demo Profiles
profiles = [
    {"id": 1, "name": "Alice", "age": 25, "interests": [1, 0, 1, 1, 0], "affiliations": [1, 0, 0]},
    {"id": 2, "name": "Bob", "age": 27, "interests": [0, 1, 1, 0, 1], "affiliations": [0, 1, 0]},
    {"id": 3, "name": "Carol", "age": 23, "interests": [1, 1, 0, 0, 1], "affiliations": [0, 0, 1]},
    {"id": 4, "name": "David", "age": 26, "interests": [1, 0, 1, 1, 1], "affiliations": [1, 0, 0]},
    {"id": 5, "name": "Eve", "age": 22, "interests": [0, 1, 0, 1, 0], "affiliations": [0, 1, 1]},
    {"id": 6, "name": "Frank", "age": 28, "interests": [1, 1, 0, 1, 0], "affiliations": [1, 0, 1]},
    {"id": 7, "name": "Grace", "age": 24, "interests": [0, 0, 1, 1, 1], "affiliations": [0, 1, 0]},
    {"id": 8, "name": "Hank", "age": 25, "interests": [1, 1, 1, 0, 0], "affiliations": [1, 0, 0]},
    {"id": 9, "name": "Ivy", "age": 23, "interests": [0, 1, 1, 1, 0], "affiliations": [0, 1, 1]},
    {"id": 10, "name": "Jack", "age": 27, "interests": [1, 0, 1, 0, 1], "affiliations": [1, 0, 0]},
    {"id": 11, "name": "Liam", "age": 30, "interests": [1, 1, 0, 0, 1], "affiliations": [0, 0, 0]},
    {"id": 12, "name": "Mia", "age": 22, "interests": [0, 1, 1, 1, 0], "affiliations": [0, 1, 1]},
    {"id": 13, "name": "Nathan", "age": 25, "interests": [1, 0, 1, 1, 1], "affiliations": [1, 0, 0]},
    {"id": 14, "name": "Olivia", "age": 27, "interests": [0, 1, 0, 1, 0], "affiliations": [1, 1, 0]},
    {"id": 15, "name": "Paul", "age": 29, "interests": [1, 0, 1, 0, 1], "affiliations": [0, 1, 0]},
    {"id": 16, "name": "Quinn", "age": 23, "interests": [0, 1, 1, 0, 0], "affiliations": [0, 0, 1]},
    {"id": 17, "name": "Rachel", "age": 24, "interests": [1, 0, 1, 1, 0], "affiliations": [1, 1, 0]},
    {"id": 18, "name": "Sam", "age": 26, "interests": [0, 0, 0, 1, 1], "affiliations": [1, 0, 1]},
    {"id": 19, "name": "Tina", "age": 22, "interests": [1, 1, 0, 1, 0], "affiliations": [0, 1, 1]},
    {"id": 20, "name": "Ursula", "age": 28, "interests": [0, 0, 1, 1, 1], "affiliations": [1, 0, 0]},
    {"id": 21, "name": "Vince", "age": 29, "interests": [1, 1, 0, 0, 1], "affiliations": [0, 1, 0]},
    {"id": 22, "name": "Wendy", "age": 26, "interests": [1, 1, 1, 0, 0], "affiliations": [0, 0, 1]},
    {"id": 23, "name": "Xander", "age": 31, "interests": [1, 0, 1, 1, 0], "affiliations": [1, 1, 0]},
    {"id": 24, "name": "Yara", "age": 27, "interests": [0, 1, 1, 0, 0], "affiliations": [0, 1, 1]},
    {"id": 25, "name": "Zane", "age": 23, "interests": [1, 0, 1, 0, 1], "affiliations": [1, 0, 1]},
    {"id": 26, "name": "Aiden", "age": 24, "interests": [0, 1, 0, 1, 0], "affiliations": [0, 1, 1]},
    {"id": 27, "name": "Bella", "age": 28, "interests": [1, 0, 1, 1, 1], "affiliations": [0, 1, 0]},
    {"id": 28, "name": "Caden", "age": 23, "interests": [0, 0, 0, 1, 1], "affiliations": [1, 0, 0]},
    {"id": 29, "name": "Dylan", "age": 26, "interests": [1, 0, 0, 1, 0], "affiliations": [1, 1, 1]},
    {"id": 30, "name": "Ethan", "age": 29, "interests": [0, 1, 1, 1, 0], "affiliations": [1, 0, 0]},
    {"id": 31, "name": "Fiona", "age": 24, "interests": [1, 1, 1, 0, 1], "affiliations": [0, 1, 0]},
    {"id": 32, "name": "Gabe", "age": 27, "interests": [0, 0, 1, 1, 1], "affiliations": [0, 1, 1]},
    {"id": 33, "name": "Holly", "age": 28, "interests": [1, 1, 0, 0, 0], "affiliations": [0, 1, 0]},
    {"id": 34, "name": "Iris", "age": 23, "interests": [1, 0, 1, 1, 0], "affiliations": [1, 0, 1]},
    {"id": 35, "name": "Jake", "age": 25, "interests": [0, 1, 0, 1, 1], "affiliations": [0, 0, 1]},
    {"id": 36, "name": "Kara", "age": 22, "interests": [1, 0, 0, 0, 1], "affiliations": [1, 0, 0]},
    {"id": 37, "name": "Luca", "age": 27, "interests": [0, 1, 1, 1, 0], "affiliations": [0, 1, 0]},
    {"id": 38, "name": "Maya", "age": 24, "interests": [1, 1, 0, 1, 0], "affiliations": [0, 1, 1]},
    {"id": 39, "name": "Noah", "age": 28, "interests": [0, 0, 1, 1, 1], "affiliations": [1, 0, 1]},
    {"id": 40, "name": "Olga", "age": 30, "interests": [1, 1, 0, 0, 0], "affiliations": [1, 0, 0]}
]

# Step 2: Function to calculate similarity scores
def calculate_similarity(profile1, profile2):
    # Cosine similarity of interests and affiliations
    interest_similarity = cosine_similarity([profile1['interests']], [profile2['interests']])[0][0]
    affiliation_similarity = cosine_similarity([profile1['affiliations']], [profile2['affiliations']])[0][0]
    
    # Combine similarities (You can adjust the weight of each if desired)
    total_similarity = (interest_similarity + affiliation_similarity) / 2
    return total_similarity

# Step 3: Function to find the best match for a given user
def find_best_match(user_profile):
    best_match = None
    best_score = -1
    
    for profile in profiles:
        if profile['id'] != user_profile['id']:  # Don't compare with itself
            score = calculate_similarity(user_profile, profile)
            if score > best_score:
                best_score = score
                best_match = profile
    
    return best_match, best_score

# Step 4: API endpoint to match profiles
@app.route('/match', methods=['POST'])
def match_profiles():
    user_data = request.get_json()
    user_profile = {
        "id": random.randint(1000, 9999),  # Random ID for the user
        "name": user_data['name'],
        "age": user_data['age'],
        "interests": user_data['interests'],
        "affiliations": user_data['affiliations']
    }
    
    best_match, best_score = find_best_match(user_profile)
    
    if best_match:
        return jsonify({
            'message': 'Best match found!',
            'user': user_profile,
            'best_match': best_match,
            'similarity_score': best_score
        })
    else:
        return jsonify({'message': 'No match found!'})

# Step 5: Home Route
@app.route('/')
def home():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, port=8080)

