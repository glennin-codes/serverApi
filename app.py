# import pandas as pd
import pandas as pd
import requests
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Fetch the data from football-data.org API
url = "https://api.football-data.org/v4/competitions/PL/matches?status=FINISHED"
headers = {"X-Auth-Token": "5777078bb3ca4a72a3e01a5fdebac8db"}
response = requests.get(url, headers=headers)
matches = []

if response.status_code == 200:
    matches = response.json()["matches"]
    logging.info("Data fetched successfully")
else:
    logging.error("Error fetching data")

# Prepare the data for training
data = []
for match in matches:
    home_team_id = match["homeTeam"]["id"]
    away_team_id = match["awayTeam"]["id"]
    home_goals = match["score"]["fullTime"]["home"]
    away_goals = match["score"]["fullTime"]["away"]
    data.append((home_team_id, away_team_id, home_goals, away_goals))

df = pd.DataFrame(data, columns=["home_team_id", "away_team_id", "home_goals", "away_goals"])

# Split the data into training and testing sets
df = pd.DataFrame(data, columns=["home_team_id", "away_team_id", "home_goals", "away_goals"])
X = df[["home_team_id", "away_team_id"]]
X.columns = ["home_team_id", "away_team_id"]  # provide the feature names

y = df.apply(lambda x: 0 if x["home_goals"] > x["away_goals"] else 1 if x["home_goals"] < x["away_goals"] else 2, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
accuracy = accuracy_score(y_test, y_pred.argmax(axis=-1))
confusion_mat = confusion_matrix(y_test, y_pred.argmax(axis=-1))

logging.info(f"Model trained successfully with accuracy score: {accuracy}")
print(f"Model trained successfully with accuracy score: {accuracy}")

# Build the REST API with Flask
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    home_team_id = data["home_team_id"]
    away_team_id = data["away_team_id"]
    prediction = model.predict([[home_team_id, away_team_id]])
    logging.info("Prediction made successfully")
    return jsonify({"prediction": int(prediction.argmax(axis=-1))})


# Run the Flask app
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
