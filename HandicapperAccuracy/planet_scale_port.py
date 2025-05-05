import pymysql
pymysql.install_as_MySQLdb()
import os
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional, List, Tuple

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

fetch_nba_player_log("LeBron James", "2025-04-15")