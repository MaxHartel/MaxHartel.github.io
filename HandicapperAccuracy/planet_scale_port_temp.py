import random
from datetime import datetime, timedelta
import pymysql
pymysql.install_as_MySQLdb()
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_connection():
    return pymysql.connect(
        host=os.getenv("DATABASE_HOST"),
        user=os.getenv("DATABASE_USERNAME"),
        password=os.getenv("DATABASE_PASSWORD"),
        database=os.getenv("DATABASE"),
        ssl={"ssl_ca": "/etc/ssl/cert.pem"}
    )

experts = {
    "Expert A": 0.90,
    "Expert B": 0.80,
    "Expert C": 0.75,
    "Expert D": 0.70,
    "Expert E": 0.55
}

def bulk_insert_simulated_data():
    conn = get_connection()
    cursor = conn.cursor()

    base_date = datetime.strptime("2025-01-01", "%Y-%m-%d")
    nba_teams = [
        "Lakers", "Warriors", "Celtics", "Heat", "Bucks",
        "Nuggets", "Suns", "Clippers", "76ers", "Knicks",
        "Mavericks", "Grizzlies", "Kings", "Timberwolves", "Pelicans",
        "Hawks", "Raptors", "Bulls", "Cavaliers", "Pacers",
        "Magic", "Wizards", "Pistons", "Hornets", "Nets",
        "Spurs", "Jazz", "Trail Blazers", "Thunder", "Rockets"
    ]

    for i in range(100):
        date = base_date + timedelta(days=i)
        team_a, team_b = random.sample(nba_teams, 2)
        player_name = f"Player_{i}"
        stat_type = "Points"
        stat_threshold = round(random.uniform(10, 30), 1)
        actual_result = random.randint(0, 1)
        event_id = f"{date.strftime('%Y-%m-%d')}-NBA-{player_name}{stat_threshold}{stat_type.upper()}"

        cursor.execute("""
            INSERT INTO events (event_id, event_date, league, team_a, team_b, actual_result,
                                context_features, stat_type, player_name, stat_threshold)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            event_id, date.strftime("%Y-%m-%d"), "NBA", team_a, team_b, actual_result,
            '{}', stat_type, player_name, stat_threshold
        ))

        for expert, accuracy in experts.items():
            should_predict_correctly = random.random() < accuracy
            prediction = actual_result if should_predict_correctly else 1 - actual_result
            confidence = round(random.uniform(0.6, 1.0), 2)
            cursor.execute("""
                INSERT INTO expert_predictions (event_id, expert_name, prediction, confidence, stat_threshold)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                event_id, expert, prediction, confidence, stat_threshold
            ))

        crowd_probability = round(random.uniform(0.45, 0.55), 4)
        cursor.execute("""
            INSERT INTO crowd_predictions (event_id, crowd_probability)
            VALUES (%s, %s)
        """, (
            event_id, crowd_probability
        ))

    conn.commit()
    conn.close()
    print("✅ Successfully inserted 100 simulated events with expert and crowd predictions.")

def insert_brunson_event_prediction():
    conn = get_connection()
    cursor = conn.cursor()

    # === Event Metadata ===
    event_id = "2024-10-22-NBA-JBRUNSON21.5PTS"
    event_date = "2024-10-22"
    league = "NBA"
    team_a = "Knicks"
    team_b = "Celtics"
    player_name = "J. Brunson"
    stat_type = "Points"
    stat_threshold = 21.5
    pick_type = "Prop"
    player_team = "Knicks"

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
        prediction = int(random.random() < accuracy)
        confidence = round(random.uniform(0.6, 1.0), 3)
        cursor.execute("""
            INSERT INTO expert_predictions (
                event_id, expert_name, prediction, confidence, stat_threshold
            ) VALUES (%s, %s, %s, %s, %s)
        """, (event_id, expert, prediction, confidence, stat_threshold))

    conn.close()
    print(f"✅ Inserted simulated predictions for event: {event_id}")

insert_brunson_event_prediction()