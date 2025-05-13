import pymysql
pymysql.install_as_MySQLdb()
import os
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional, List, Tuple
from pymysql.cursors import DictCursor
# Load environment variables
load_dotenv()

def get_connection():
    return pymysql.connect(
        host=os.getenv("DATABASE_HOST"),
        user=os.getenv("DATABASE_USERNAME"),
        password=os.getenv("DATABASE_PASSWORD"),
        database=os.getenv("DATABASE"),
        ssl={"ssl_ca": "/etc/ssl/cert.pem"}  # Path to trusted CA cert on macOS
    )

def get_dict_connection():
    return pymysql.connect(
        host=os.getenv("DATABASE_HOST"),
        user=os.getenv("DATABASE_USERNAME"),
        password=os.getenv("DATABASE_PASSWORD"),
        database=os.getenv("DATABASE"),
        ssl={"ssl_ca": "/etc/ssl/cert.pem"},  # Path to trusted CA cert on macOS
        cursorclass=DictCursor
    )

def add_event(event_id: str, event_date: str, league: str, team_a: str, team_b: str, actual_result: Optional[int] = None):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT IGNORE INTO events (event_id, event_date, league, team_a, team_b, actual_result)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (event_id, event_date, league, team_a, team_b, actual_result))




def insert_nba_player_log(
    player_name: str,
    game_date: str,
    team: Optional[str] = None,
    opponent: Optional[str] = None,
    points: Optional[float] = None,
    rebounds: Optional[float] = None,
    assists: Optional[float] = None,
    personal_fouls: Optional[float] = None,
    threes_made: Optional[float] = None,
    blocks: Optional[float] = None,
    steals: Optional[float] = None,
    turnovers: Optional[float] = None,
    minutes: Optional[float] = None
):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO NBA_Player_Logs (
                player_name, game_date, team, opponent,
                points, rebounds, assists, personal_fouls, threes_made,
                blocks, steals, turnovers, minutes
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            player_name, game_date, team, opponent,
            points, rebounds, assists, personal_fouls, threes_made,
            blocks, steals, turnovers, minutes
        ))

def update_nba_derived_fields():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT entry_id, points, rebounds, assists, blocks, steals, turnovers FROM NBA_Player_Logs")
        rows = cursor.fetchall()

        for row in rows:
            entry_id, points, rebounds, assists, blocks, steals, turnovers = row
            points = points or 0
            rebounds = rebounds or 0
            assists = assists or 0
            blocks = blocks or 0
            steals = steals or 0
            turnovers = turnovers or 0

            pa = points + assists
            pr = points + rebounds
            ra = rebounds + assists
            pra = pa + rebounds
            fantasy_points = (
                points * 1 +
                rebounds * 1.2 +
                assists * 1.5 +
                blocks * 3 +
                steals * 3 +
                turnovers * (-1)
            )

            cursor.execute("""
                UPDATE NBA_Player_Logs
                SET pa = %s, pr = %s, ra = %s, pra = %s, fantasy_points = %s
                WHERE entry_id = %s
            """, (pa, pr, ra, pra, fantasy_points, entry_id))

def add_crowd_prediction(event_id: str, crowd_probability: float, source_name: str = "CrowdConsensus"):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO crowd_predictions (event_id, crowd_probability, source_name)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE crowd_probability = VALUES(crowd_probability)
        """, (event_id, crowd_probability, source_name))

def add_expert_prediction(event_id: str, expert_name: str, prediction: int, confidence: Optional[float] = None):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO expert_predictions (event_id, expert_name, prediction, confidence)
            VALUES (%s, %s, %s, %s)
        """, (event_id, expert_name, prediction, confidence))

def add_multiple_expert_predictions(predictions: List[Tuple[str, str, int, Optional[float]]]):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO expert_predictions (event_id, expert_name, prediction, confidence)
            VALUES (%s, %s, %s, %s)
        """, predictions)

def generate_event_id(pick_name: str, league: str) -> str:
    clean = pick_name.upper().replace(" ", "").replace("/", "-")
    return f"{datetime.today().strftime('%Y-%m-%d')}-{league}-{clean}"

def delete_event(event_id: str):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM expert_predictions WHERE event_id = %s", (event_id,))
        cursor.execute("DELETE FROM crowd_predictions WHERE event_id = %s", (event_id,))
        cursor.execute("DELETE FROM events WHERE event_id = %s", (event_id,))
    print(f"Deleted event {event_id} and all associated predictions.")

def fetch_events_by_date(event_date: str):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT event_id, league, team_a, team_b, actual_result
            FROM events
            WHERE event_date = %s
        """, (event_date,))
        rows = cursor.fetchall()

    print(f"Events on {event_date}:")
    for row in rows:
        event_id, league, team_a, team_b, result = row
        result_text = "Occurred" if result == 1 else "Did Not Occur" if result == 0 else "Unknown"
        print(f" - [{league}] {team_a} vs {team_b} (Event ID: {event_id}) → {result_text}")

def fetch_predictions_by_expert(expert_name: str):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT event_id, prediction, confidence, prediction_time
            FROM expert_predictions
            WHERE expert_name = %s
            ORDER BY prediction_time DESC
        """, (expert_name,))
        rows = cursor.fetchall()

    print(f"Predictions by {expert_name}:")
    for event_id, prediction, confidence, time in rows:
        pred_str = "Too Low" if prediction == 1 else "Too High"
        conf_str = f"{confidence:.2f}" if confidence is not None else "N/A"
        print(f" - [{event_id}] {pred_str} (Confidence: {conf_str}) at {time}")

def submit_event(
    event_id: str,
    event_date: str,
    league: str,
    team_a: str,
    team_b: str,
    crowd_probability: float,
    expert_predictions: List[Tuple[str, int, Optional[float]]],
    actual_result: Optional[int] = None,
    pick_type: str = "MoneyLine",
    player_team: str = "None",
    stat_type: str = "MoneyLine"
) -> Tuple[bool, str]:
    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            # Prevent duplicates
            cursor.execute("SELECT 1 FROM events WHERE event_id = %s", (event_id,))
            if cursor.fetchone():
                return False, f"Event '{event_id}' already exists."

            cursor.execute("""
                INSERT INTO events (event_id, event_date, league, team_a, team_b, actual_result, pick_type, player_team, stat_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (event_id, event_date, league, team_a, team_b, actual_result, pick_type, player_team, stat_type))

            cursor.execute("""
                INSERT INTO crowd_predictions (event_id, crowd_probability)
                VALUES (%s, %s)
            """, (event_id, crowd_probability))

            for expert_name, prediction, confidence in expert_predictions:
                cursor.execute("""
                    INSERT INTO expert_predictions (event_id, expert_name, prediction, confidence)
                    VALUES (%s, %s, %s, %s)
                """, (event_id, expert_name, prediction, confidence))

                cursor.execute("SELECT 1 FROM expert_reliability WHERE expert_name = %s", (expert_name,))
                if not cursor.fetchone():
                    cursor.execute("INSERT INTO expert_reliability (expert_name) VALUES (%s)", (expert_name,))

        return True, "Event submitted successfully."
    except Exception as e:
        return False, f"Database error: {e}"


def fetch_nba_player_log(player_name: str, game_date: Optional[str] = None):
    with get_connection() as conn:
        cursor = conn.cursor()

        if game_date:
            cursor.execute("""
                SELECT * FROM NBA_Player_Logs
                WHERE player_name = %s AND game_date = %s
            """, (player_name, game_date))
        else:
            cursor.execute("""
                SELECT * FROM NBA_Player_Logs
                WHERE player_name = %s
                ORDER BY game_date DESC
                LIMIT 1
            """, (player_name,))
        
        row = cursor.fetchone()

    if row:
        columns = [desc[0] for desc in cursor.description]
        print(f"\n🧾 Statline for {player_name}:")
        for col, val in zip(columns, row):
            print(f"{col}: {val}")
    else:
        print(f"⚠️ No statline found for {player_name}.")

def bulk_insert_sample_data(conn, sample_events, sample_predictions):
    cursor = conn.cursor()

    # Insert sample events
    event_query = """
        INSERT INTO events (event_id, player_name, league, game_date, stat_type, stat_threshold)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE player_name=VALUES(player_name)
    """
    for event in sample_events:
        cursor.execute(event_query, (
            event["event_id"],
            event["player_name"],
            event["league"],
            event["game_date"],
            event["stat_type"],
            event["stat_threshold"]
        ))

    # Insert sample expert predictions
    prediction_query = """
        INSERT INTO expert_predictions (event_id, expert_name, prediction, confidence)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE confidence=VALUES(confidence)
    """
    for pred in sample_predictions:
        cursor.execute(prediction_query, (
            pred["event_id"],
            pred["expert_name"],
            pred["prediction"],
            pred["confidence"]
        ))

    conn.commit()
    print("✅ Sample events and predictions inserted.")

import random
from datetime import datetime, timedelta
import pymysql
import os

def generate_and_insert_sample_data():
    print("Accessing Database")
    conn = get_connection()

    sample_events = []
    sample_predictions = []

    player_pool = ["Tyler Herro", "Bam Adebayo", "Jayson Tatum", "Jaylen Brown", "Davion Mitchell"]
    leagues = ["NBA"]
    stat_types = ["Points", "Assists", "Rebounds", "Steals", "Blocks"]
    expert_pool = ["ExpertA", "ExpertB", "ExpertC", "ExpertD", "ExpertE"]

    base_date = datetime.today() - timedelta(days=30)

    for i in range(500):
        player_name = random.choice(player_pool)
        league = random.choice(leagues)
        stat_type = random.choice(stat_types)
        stat_threshold = round(random.uniform(5, 30), 1)
        event_date = (base_date + timedelta(days=i % 15)).date()
        event_date_str = event_date.isoformat()

        team_a = "MIA"
        team_b = "BOS"

        event_id = f"{event_date_str}-{league}-{player_name.replace(' ', '').upper()}{stat_threshold}{stat_type.upper()}"

        event = {
            "event_id": event_id,
            "player_name": player_name,
            "league": league,
            "event_date": event_date_str,
            "stat_type": stat_type,
            "stat_threshold": stat_threshold,
            "team_a": team_a,
            "team_b": team_b
        }

        sample_events.append(event)

        for expert in expert_pool:
            prediction = random.choice([0, 1])
            confidence = round(random.uniform(0.5, 1.0), 2)


            sample_predictions.append({
                "event_id": event_id,
                "expert_name": expert,
                "prediction": prediction,
                "confidence": confidence
            })

    bulk_insert_sample_data(conn, sample_events, sample_predictions)
    conn.close()

def bulk_insert_sample_data(conn, sample_events, sample_predictions):
    cursor = conn.cursor()

    event_query = """
        INSERT INTO events (
            event_id, player_name, league, event_date,
            stat_type, stat_threshold, team_a, team_b
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            player_name = VALUES(player_name)
    """

    prediction_query = """
        INSERT INTO expert_predictions (
            event_id, expert_name, prediction, confidence
        )
        VALUES (%s, %s, %s, %s)
    """

    for event in sample_events:
        cursor.execute(event_query, (
            event["event_id"],
            event["player_name"],
            event["league"],
            event["event_date"],
            event["stat_type"],
            event["stat_threshold"],
            event["team_a"],
            event["team_b"]
        ))

    for pred in sample_predictions:
        cursor.execute(prediction_query, (
            pred["event_id"],
            pred["expert_name"],
            pred["prediction"],
            pred["confidence"]
        ))

    conn.commit()
    print(f"✅ Inserted {len(sample_events)} events and {len(sample_predictions)} predictions.")

def get_player_team(event_id):
    # Connect to the MySQL database
    conn = get_connection()
    
    cursor = conn.cursor()
    
    # Query to get the team based on event_id
    query = """
        SELECT player_team
        FROM events
        WHERE event_id = %s
    """
    
    # Execute the query with the provided event_id
    cursor.execute(query, (event_id,))
    
    # Fetch the result
    result = cursor.fetchone()

    # Close the database connection
    cursor.close()
    conn.close()
    
    # If result is found, return the team
    if result:
        return result[0]  # team is in the first column
    else:
        return None  # If no result found for the event_id



def update_game_date_league_team_for_predictions():
    # Connect to the MySQL database
    conn = get_connection()
    cursor = conn.cursor()
    
    # Query to fetch all event_ids
    cursor.execute("SELECT event_id FROM expert_predictions")
    event_ids = cursor.fetchall()
    
    # Loop through each event_id and update game_date, league, and team
    for event_id in event_ids:
        # Extract game_date from event_id (first part before the first hyphen)
        event_parts = event_id[0].split('-')
        game_date = '-'.join(event_parts[:3])  # Extract 'YYYY-MM-DD' part
        league = event_parts[3]  # Extract league (4th part)
        team = get_player_team(event_id)
        
        # Update query to set game_date, league, and team based on event_id
        update_query = """
            UPDATE expert_predictions
            SET game_date = %s, league = %s, team = %s
            WHERE event_id = %s
        """
        cursor.execute(update_query, (game_date, league, team, event_id[0]))
    
    # Commit the changes to the database
    conn.commit()
    
    # Close the database connection
    cursor.close()
    conn.close()
    
    print("Game dates, leagues, and teams updated successfully!")


#fetch_nba_player_log("LeBron James", "2025-04-15")
print("Accessing Database")
# update_game_date_league_team_for_predictions()


def insert_lebron_event_prediction():
    conn = get_connection()
    cursor = conn.cursor()

    event_id = "2024-10-22-NBA-KTOWNS14.0PTS"
    event_date = "2024-10-22"
    league = "NBA"
    team_a = "Over"
    team_b = "Under"
    player_name = "K. Towns"
    stat_type = "points"
    stat_threshold = 14.0
    pick_type = "Prop"
    player_team = "New York Knicks"

    # === Insert into events ===
    cursor.execute("""
        INSERT INTO events (
            event_id, event_date, league, team_a, team_b,
            actual_result, context_features, pick_type,
            player_team, stat_type, player_name, stat_threshold
        ) VALUES (
            %s, %s, %s, %s, %s,
            NULL, NULL, %s,
            %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE player_name = VALUES(player_name)
    """, (
        event_id, event_date, league, team_a, team_b,
        pick_type, player_team, stat_type, player_name, stat_threshold
    ))

    # === Insert into crowd_predictions ===
    crowd_prob = round(random.uniform(0.45, 0.7), 4)
    cursor.execute("""
        INSERT INTO crowd_predictions (event_id, crowd_probability)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE crowd_probability = VALUES(crowd_probability)
    """, (event_id, crowd_prob))

    # === Insert expert_predictions ===
    experts = {
        "Expert A": 0.90,
        "Expert B": 0.80,
        "Expert C": 0.75,
        "Expert D": 0.70,
        "Expert E": 0.55,
    }

    for expert, accuracy in experts.items():
        # Simulate prediction based on accuracy
        prediction = int(random.random() < accuracy)
        confidence = round(random.uniform(0.6, 1.0), 3)
        cursor.execute("""
            INSERT INTO expert_predictions (
                event_id, expert_name, prediction, confidence, stat_threshold
            ) VALUES (%s, %s, %s, %s, %s)
        """, (event_id, expert, prediction, confidence, stat_threshold))

    conn.close()
    print(f"✅ Inserted simulated predictions for event: {event_id}")

#insert_lebron_event_prediction()


def get_player_and_stat_type(event_id):
    # Connect to the MySQL database
    conn =get_connection()
    
    cursor = conn.cursor()
    
    # Query to select player_name and stat_type based on event_id
    query = """
        SELECT player_name, stat_type
        FROM events
        WHERE event_id = %s
    """
    
    # Execute the query
    cursor.execute(query, (event_id,))
    
    # Fetch the result
    result = cursor.fetchone()
    
    # Close the database connection
    cursor.close()
    conn.close()
    
    # Return the result (player_name and stat_type)
    if result:
        return result
    else:
        return None  # Return None if no result is found
    

# event_id = '2024-10-22-NBA-JBRUNSON21.5PTS'
# result = get_player_and_stat_type(event_id)
# if result:
#     print(f"Player Name: {result[0]}, Stat Type: {result[1]}")
# else:
#     print("No data found for the given event_id.")