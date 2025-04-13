import sqlite3
from datetime import datetime
from typing import Optional, List, Tuple

DB_PATH = "/Users/maxhartel/Desktop/Desktop - Max’s MacBook Pro/Project Parlay/Project-Parlay/Pick_Confidence" 


def connect():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def add_event(event_id: str, event_date: str, league: str, team_a: str, team_b: str, actual_result: Optional[int] = None):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO events (event_id, event_date, league, team_a, team_b, actual_result)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (event_id, event_date, league, team_a, team_b, actual_result))
    conn.commit()
    conn.close()


def add_crowd_prediction(event_id: str, crowd_probability: float, source_name: str = "CrowdConsensus"):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO crowd_predictions (event_id, crowd_probability, source_name)
        VALUES (?, ?, ?)
    """, (event_id, crowd_probability, source_name))
    conn.commit()
    conn.close()


def add_expert_prediction(event_id: str, expert_name: str, prediction: int, confidence: Optional[float] = None):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO expert_predictions (event_id, expert_name, prediction, confidence)
        VALUES (?, ?, ?, ?)
    """, (event_id, expert_name, prediction, confidence))
    conn.commit()
    conn.close()


def add_multiple_expert_predictions(predictions: List[Tuple[str, str, int, Optional[float]]]):
    conn = connect()
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT INTO expert_predictions (event_id, expert_name, prediction, confidence)
        VALUES (?, ?, ?, ?)
    """, predictions)
    conn.commit()
    conn.close()

def generate_event_id(pick_name: str, league: str) -> str:
    clean = pick_name.upper().replace(" ", "").replace("/", "-")
    return f"{datetime.today().strftime('%Y-%m-%d')}-{league}-{clean}"


def delete_event(event_id: str):
    conn = connect()
    cursor = conn.cursor()

    # Delete all associated rows first (due to foreign keys)
    cursor.execute("DELETE FROM expert_predictions WHERE event_id = ?", (event_id,))
    cursor.execute("DELETE FROM crowd_predictions WHERE event_id = ?", (event_id,))
    cursor.execute("DELETE FROM events WHERE event_id = ?", (event_id,))

    conn.commit()
    conn.close()
    print(f"Deleted event {event_id} and all associated predictions.")


def fetch_events_by_date(event_date: str):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT event_id, league, team_a, team_b, actual_result
        FROM events
        WHERE event_date = ?
    """, (event_date,))
    rows = cursor.fetchall()
    conn.close()

    print(f"Events on {event_date}:")
    for row in rows:
        event_id, league, team_a, team_b, result = row
        result_text = "Occurred" if result == 1 else "Did Not Occur" if result == 0 else "Unknown"
        print(f" - [{league}] {team_a} vs {team_b} (Event ID: {event_id}) → {result_text}")


def fetch_predictions_by_expert(expert_name: str):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT event_id, prediction, confidence, prediction_time
        FROM expert_predictions
        WHERE expert_name = ?
        ORDER BY prediction_time DESC
    """, (expert_name,))
    rows = cursor.fetchall()
    conn.close()

    print(f"Predictions by {expert_name}:")
    for event_id, prediction, confidence, time in rows:
        pred_str = "Too Low" if prediction == 1 else "Too High"
        conf_str = f"{confidence:.2f}" if confidence is not None else "N/A"
        print(f" - [{event_id}] {pred_str} (Confidence: {conf_str}) at {time}")


# === MAIN INSERT FUNCTION ===
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

    """
    Inserts a full event into the database, checking for duplicates and initializing
    new experts in the reliability table if needed.

    Returns (success: bool, message: str)
    """
    try:
        conn = connect()
        cursor = conn.cursor()

        # Prevent duplicate events
        cursor.execute("SELECT 1 FROM events WHERE event_id = ?", (event_id,))
        if cursor.fetchone():
            conn.close()
            return False, f"Event '{event_id}' already exists."

        # Insert into events
        cursor.execute("""
            INSERT INTO events (
            event_id, event_date, league,
            team_a, team_b, actual_result,
            pick_type, player_team, stat_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (event_id, event_date, league, team_a, team_b, actual_result,pick_type, player_team, stat_type))

        # Insert crowd probability
        cursor.execute("""
            INSERT INTO crowd_predictions (event_id, crowd_probability)
            VALUES (?, ?)
        """, (event_id, crowd_probability))

        # Insert expert predictions
        for expert_name, prediction, confidence in expert_predictions:
            cursor.execute("""
                INSERT INTO expert_predictions (event_id, expert_name, prediction, confidence)
                VALUES (?, ?, ?, ?)
            """, (event_id, expert_name, prediction, confidence))

            # Initialize expert reliability if missing
            cursor.execute("SELECT 1 FROM expert_reliability WHERE expert_name = ?", (expert_name,))
            if not cursor.fetchone():
                cursor.execute("""
                    INSERT INTO expert_reliability (expert_name)
                    VALUES (?)
                """, (expert_name,))

        conn.commit()
        conn.close()
        return True, "Event submitted successfully."

    except Exception as e:
        return False, f"Database error: {e}"


example_event = {
        "event_id": "2025-04-12-NBA-NYK-MIA",
        "event_date": "2025-04-12",
        "league": "NBA",
        "team_a": "NYK",
        "team_b": "MIA",
        "crowd_probability": 0.61,
        "expert_predictions": [
            ("ExpertX", 1, 0.9),
            ("ExpertY", 0, None),
            ("ExpertZ", 1, 0.75)
        ],
        "actual_result": 1
    }

#submit_event(**example_event)
#delete_event("2025-04-12-NBA-NYK-MIA")

