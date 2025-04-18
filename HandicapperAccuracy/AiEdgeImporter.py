import csv
from datetime import datetime
from ConfidenceDB_Port import submit_event, generate_event_id
import time

def import_csv_to_db(csv_path: str):
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                event_date = row["date"]
                expert_name = row["Handicapper"]
                player = row["player_name"]
                league = row["Leauge"]
                projection = row["stat_projection"].replace(" ", "")
                stat_type = row["stat_type"]
                over_under = row["Over/Under"].strip().lower()
                confidence = float(row["confidence_score"]) / 10.0 if row["confidence_score"] else None
                actual_stat = float(row["actual_statistic"]) if row["actual_statistic"] else None

                # === Determine actual_result ===
                if over_under == "over":
                    prediction = 1
                    actual_result = 1 if actual_stat is not None and actual_stat > float(projection) else 0
                else:
                    prediction = 0
                    actual_result = 1 if actual_stat is not None and actual_stat < float(projection) else 0

                # Construct event_id
                clean_player = player.upper().replace(" ", "").replace(".", "")
                clean_proj = str(projection).replace("+", "")
                event_id = f"{event_date}-{league}-{clean_player}{clean_proj}{stat_type.replace(' ', '')}"

                # Use player/team naming convention
                team_a = "Over"
                team_b = "Under"
                pick_type = "Prop"
                player_team = "None"
                crowd_probability = 0.55  # Neutral fallback

                expert_predictions = [(expert_name, prediction, confidence)]

                print(f"Importing: {event_id} → prediction: {prediction}, actual: {actual_result}")

                success, msg = submit_event(
                    event_id=event_id,
                    event_date=event_date,
                    league=league,
                    team_a=team_a,
                    team_b=team_b,
                    crowd_probability=crowd_probability,
                    expert_predictions=expert_predictions,
                    actual_result=actual_result,
                    pick_type=pick_type,
                    player_team=player_team,
                    stat_type=stat_type
                )

                if success:
                    print(f"✅ Inserted: {event_id}")
                else:
                    print(f"❌ Failed: {event_id} → {msg}")

                # After submit_event
                time.sleep(0.05)

            except Exception as e:
                print(f"⚠️ Error importing row: {row}")
                print(f"⛔ {e}")


import_csv_to_db("HandicapperAccuracy/Events.csv")