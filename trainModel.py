import pandas as pd
import requests
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# "BL1", "FL1", "SA", "PPL", "PD", "BSA" "CL",

leagues = [ "PL"]
matches = []
match_ids=set()

# Fetch matches for each league
for league in leagues:
    url = f"https://api.football-data.org/v4/competitions/{league}/matches?status=FINISHED"
    headers = {"X-Auth-Token": "5777078bb3ca4a72a3e01a5fdebac8db"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        matches.extend(response.json()["matches"])
        logging.info(f"Data fetched successfully for league {league}")
        print(f"Data fetched successfully for league {league}")
    else:
        logging.error(f"Error fetching data for league {league}")


# Prepare the data for training
data = []
dynamic_head2head_data=None
for match in matches:
    home_team_id = match["homeTeam"]["id"]
    away_team_id = match["awayTeam"]["id"]
    home_goals = match["score"]["fullTime"]["home"]
    away_goals = match["score"]["fullTime"]["away"]
    match_id = match['id']
    match_ids.add(match_id)
    data.append((home_team_id, away_team_id, home_goals, away_goals))
    
for match_id in  match_ids:
    url=f"http://api.football-data.org/v4/matches/{match_id}/head2head?limit=100"
    headers = {"X-Auth-Token": "5777078bb3ca4a72a3e01a5fdebac8db"}
    response = requests.get(url, headers=headers)
    if response.status_code==200:
        head2head_matches=response.json()["head2head"]["matches"]
        head2head_data = []
    
        for match in head2head_matches:
            home_team_id=match["homeTeam"]["id"]
            away_team_id=match["awayTeam"]["id"]
            home_goals=match['score']["fullTime"]["home"]
            away_goals = match["score"]["fullTime"]["away"]
            head2head_data.append((home_team_id, away_team_id, home_goals, away_goals))
            pd.DataFrame(head2head_data, columns=["home_team_id", "away_team_id", "home_goals", "away_goals"])
            dynamic_head2head_data=pd.DataFrame(head2head_data, columns=["home_team_id", "away_team_id", "home_goals", "away_goals"])
    else:
        logging.error("error fetching data")

  
    

# df = pd.DataFrame(data, columns=["home_team_id", "away_team_id", "home_goals", "away_goals"])

# Split the data into training and testing sets
df = pd.DataFrame(data, columns=["home_team_id", "away_team_id", "home_goals", "away_goals"])
# Concatenate dynamic data with existing data
combined_df = pd.concat([df, dynamic_head2head_data], ignore_index=True)

X_combined_df = combined_df[["home_team_id", "away_team_id"]]
X_combined_df.columns = ["home_team_id", "away_team_id"]  # provide the feature names

y = combined_df.apply(lambda x: 0 if x["home_goals"] > x["away_goals"] else 1 if x["home_goals"] < x["away_goals"] else 2, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_combined_df, y, test_size=0.2, random_state=42)

# Train the neural network model
print("Training the model...")
logging.info("Training the model...")

model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Test the model and print the accuracy

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred.argmax(axis=-1).astype(int))
confusion_mat = confusion_matrix(y_test, y_pred.argmax(axis=-1))
print("Confusion Matrix:")
print(confusion_mat)
logging.info(f"Confusion Matrix: {confusion_mat}")
model.save("trained_model.h5") 
logging.info(f"Model trained successfully with accuracy score: {accuracy}")
print(f"Model trained successfully with accuracy score: {accuracy}")
