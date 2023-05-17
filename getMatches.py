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
            home_team_id = match['homeTeam']['id']
            home_team_name = match['homeTeam']['name']
            home_team_logo = match["homeTeam"]["crest"]
            away_team_id = match['awayTeam']['id']
            away_team_name = match['awayTeam']['name']
            away_team_logo = match["awayTeam"]["crest"]
            match_time = match['utcDate']
           
            # Add the extracted information to the data list
            data.append({
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
const leagues = ["CL","PL" "BL1", "FL1", "SA", "PPL", "PD", "BSA"];