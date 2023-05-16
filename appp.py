import joblib
import pandas as pd
import requests
from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import logging

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

# Feature Engineering: Adding features for past match performances
# Add the number of goals scored in the last 3 matches
df['home_last_3_goals'] = df.groupby('home_team_id')['home_goals'].rolling(window=3, min_periods=1).sum().reset_index(0, drop=True)
df['away_last_3_goals'] = df.groupby('away_team_id')['away_goals'].rolling(window=3, min_periods=1).sum().reset_index(0, drop=True)

# Add the home and away win percentages of the teams
df['home_win_percentage'] = df.groupby('home_team_id').apply(lambda x: x['home_goals'].gt(x['away_goals']).rolling(window=len(x)).mean().fillna(method='ffill')).reset_index(0, drop=True)
df['away_win_percentage'] = df.groupby('away_team_id').apply(lambda x: x['away_goals'].gt(x['home_goals']).rolling(window=len(x)).mean().fillna(method='ffill')).reset_index(0, drop=True)

# Split the data into train and test sets
X = df[['home_team_id', 'away_team_id', 'home_last_3_goals', 'away_last_3_goals', 'home_win_percentage', 'away_win_percentage']]
y = df.apply(lambda x: 0 if x['home_goals'] > x['away_goals'] else 1 if x['home_goals'] < x['away_goals'] else 2, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning: Try different regularization parameter values
C_values = [0.01, 0.1, 1, 10, 100]
for C in C_values:
    model = LogisticRegression(C=C, max_iter=10000)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    logging.info(f'Logistic Regression Training accuracy with C={C}: {train_accuracy}')

    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    logging.info(f'Logistic Regression Testing accuracy with C={C}: {test_accuracy}')

    # Print the classification report
    target_names = ['Home Win', 'Away Win', 'Draw']
    report = classification_report(y_test, y_pred_test, target_names=target_names)
    logging.info(f'Classification report with C={C}:\n{report}')

    # Save the model
    filename = f'model_C={C}.joblib'
    joblib.dump(model, filename)
    logging.info(f'Model saved as {filename}')
# Build the REST API with Flask
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    home_team_id = data["homeTeamId"]
    away_team_id = data["awayTeamId"]
    prediction = model.predict([[home_team_id, away_team_id]])
    logging.info("Prediction made successfully")
    return jsonify({"prediction": int(prediction.argmax(axis=-1))})
# Define endpoint for fetching scheduled matches
@app.route('/matches', methods=['GET'])
def get_scheduled_matches():
    # Call the football-data.org API to get the list of scheduled matches
    response = requests.get('https://api.football-data.org/v4/matches?status=SCHEDULED',
                            headers={'X-Auth-Token': '5777078bb3ca4a72a3e01a5fdebac8db'})
    
    # If the API call is successful, parse the response and return the data
    if response.status_code == 200:
        # Extract the list of matches from the response
        matches = response.json()['matches']
        
        # Define an empty list to hold the modified data
        data = []
        
        # Loop through the list of matches and extract the relevant information
        for match in matches:
            # Extract the relevant information from the match object
            match_id=match['match']['id']
            home_team_id = match['homeTeam']['id']
            home_team_name = match['homeTeam']['name']
            home_team_logo = match["homeTeam"]["crest"]
            away_team_id = match['awayTeam']['id']
            away_team_name = match['awayTeam']['name']
            away_team_logo = match["awayTeam"]["crest"]
            match_time = match['utcDate']
           
            # Add the extracted information to the data list
            data.append({
                'matchId' : match_id,
                'homeTeamId': home_team_id,
                'homeTeamName': home_team_name,
                'homeTeamLogo': home_team_logo,
                'awayTeamId': away_team_id,
                'awayTeamName': away_team_name,
                'awayTeamLogo': away_team_logo,
                'matchTime': match_time
            })
        
        # Return the data list as a JSON object
        return jsonify(data)
    
    # If the API call is unsuccessful, return an error message
    else:
        return jsonify({'error': 'Failed to fetch scheduled matches.'}), response.status_code



# Run the Flask app
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)