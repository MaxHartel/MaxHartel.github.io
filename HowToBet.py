from flask import Flask, request, jsonify, render_template
from ppObjects import Pick, BoostPromo, ProtectedPromo
from RROptimizer import analyze_all_splits
import json
import os
import random
from itertools import combinations
from HandicapperAccuracy.ConfidenceDB_Port import submit_event, generate_event_id
from datetime import datetime
import sqlite3

app = Flask(__name__)

# Global object lists
pick_objects = []
boost_promo_objects = []
protected_promo_objects = []
next_id = 1
user_accuracy = 0.0
DB_PATH = "/Users/maxhartel/Desktop/Desktop - Max’s MacBook Pro/Project Parlay/Project-Parlay/Pick_Confidence" 


origin_profiles = {
    "ChalkBoard PI": 0.80,
    "HarryLock": 0.71,
    "DanGamble POD": 0.78,
    "DanGamble AI Edge": 90.0,
    "GameScript": 0.80,
    "Winible": 0.83,
    "DoberMan": 0.76,
    "JoshMiller": 0.60,
    "Me": lambda: user_accuracy
}

def confidence_score(decimal_odds, expert_confidence, expert_accuracy):
    implied_prob = 1 / decimal_odds
    c = expert_confidence / 100
    a = expert_accuracy / 100
    score = 100 * (a * c + (1 - a) * implied_prob)
    return score

def generate_round_robin_subparlays(pick_list, subgroup_size):
    """Returns a list of subparlays (combinations) of picks."""
    return [list(combo) for combo in combinations(pick_list, subgroup_size)]

def get_all_objects():
    return pick_objects, boost_promo_objects, protected_promo_objects

@app.route("/get_picks", methods=["GET"])
def get_picks():
    return jsonify({
    "objects": [p.to_dict() for p in pick_objects]
    })

@app.route("/")
def index():
    return render_template("projectparlay.html", user_accuracy=user_accuracy)

@app.route("/optimize_split", methods=["GET"])
def optimize_split():
    sorted_picks = sorted(pick_objects, key=lambda x: getattr(x, "confidence_score", 0), reverse=True)
    best_score, best_label = analyze_all_splits(sorted_picks)

    if "Full List" in best_label:
        picks_to_use = sorted_picks
    elif "Left" in best_label:
        split_index = int(best_label.split("index")[1].strip())
        picks_to_use = sorted_picks[:split_index]
    elif "Right" in best_label:
        split_index = int(best_label.split("index")[1].strip())
        picks_to_use = sorted_picks[split_index:]
    else:
        picks_to_use = []

    # Extract optimal subgroup size from the label (e.g., "Size 3")
    import re
    match = re.search(r"Size (\d+)", best_label)
    subgroup_size = int(match.group(1)) if match else 2

    # Generate the subparlays (2D array of pick dicts)
    subparlays = generate_round_robin_subparlays(picks_to_use, subgroup_size)

    return jsonify({
        "best_score": best_score,
        "best_config": best_label,
        "sorted_picks": [p.to_dict() for p in sorted_picks],
        "subparlays": [[p.to_dict() for p in sub] for sub in subparlays]
    })



@app.route("/process", methods=["POST"])
def process():
    global next_id
    data = request.get_json()

    try:
        print("🟢 PAYLOAD RECEIVED:", data)

        name = data.get("name", "").strip()
        pick_origins = data.get("pick_origin", [])  # [{ name, confidence }]
        print("odds:" + str(data.get("odds", 0)))
        odds = float(data.get("odds", 0))
        leagues = data.get("league", [])
        reusable = data.get("reusable", True)
        capital_limit = int(data.get("capital_limit", 0))
        mutual_exclusion = int(data.get("mutual_exclusion", -1))
        pick_type = data.get("pick_type", "MoneyLine")
        player_team = data.get("player_team", "None")
        stat_type = data.get("stat_type", "MoneyLine")

        if not name or not odds or not pick_origins or not leagues:
            return jsonify({"response": "Missing required fields", "success": False}), 400

        implied_prob = round(1 / odds, 4)  # crowd probability from odds
        today = datetime.today().strftime("%Y-%m-%d")

        expert_predictions = []
        total_score = 0

        for origin_obj in pick_origins:
            origin = origin_obj.get("name")
            print(origin_obj.get("confidence"))
            origin_conf = origin_obj.get("confidence")

            if not origin:
                continue  # Skip invalid entries

            # 💡 Ensure origin_conf is a usable float
            try:
                used_conf = float(origin_conf)
            except Exception as e:
                used_conf = 75.0  # fallback if None or not a number

            # 🧠 Historical accuracy
            origin_accuracy = origin_profiles[origin]() if callable(origin_profiles[origin]) else origin_profiles[origin]

            norm_conf = used_conf / 100.0
            expert_predictions.append((origin, 1, norm_conf))

            # 🧮 Confidence score calculation
            score = confidence_score(odds, used_conf, origin_accuracy)
            total_score += score

        final_score = round(total_score / len(expert_predictions), 2) if expert_predictions else 0
        print(final_score)

        if pick_type == "MoneyLine":
            team_a = name
            team_b = "Other"
            player_team = "None"
        else:
            team_a = "Over"
            team_b = "Under"

        print("🧠 Parsed expert predictions:", expert_predictions)

        status_messages = []
        success_count = 0

        for league in leagues:
            event_id = generate_event_id(name, league)
            success, message = submit_event(
                event_id=event_id,
                event_date=today,
                league=league,
                team_a=team_a,
                team_b=team_b,
                crowd_probability=implied_prob,
                expert_predictions=expert_predictions,
                actual_result=None,
                pick_type=pick_type,
                player_team=player_team,
                stat_type=stat_type
            )
            status_messages.append(f"[{league}] {message}")

            print("odds:" + str(odds))

            if success:
                new_pick = Pick(
                    name=name,
                    odds=odds,
                    confidence=final_score,
                    mutual_exclusion_group=mutual_exclusion,
                    league=league,
                    event_id=event_id,
                    reusable=reusable,
                    capital_limit=capital_limit
                )
                pick_objects.append(new_pick)
                next_id += 1
                success_count += 1

        return jsonify({
            "response": " | ".join(status_messages),
            "objects": [p.__dict__ for p in pick_objects],
            "success": success_count == len(leagues)
        })

    except Exception as e:
        print("❌ SERVER ERROR:", e)
        return jsonify({"response": f"Server error: {str(e)}", "success": False}), 500



@app.route("/create_boost_promo", methods=["POST"])
def create_boost_promo():
    data = request.get_json()
    boost_percentage = int(data.get("boost_percentage", 0))
    required_picks = int(data.get("required_picks", 0))
    same_sport = data.get("same_sport", False)

    boost = BoostPromo(boost_percentage, required_picks, same_sport)
    boost_promo_objects.append(boost.__dict__)

    return jsonify({"response": f"Created Boost Promo: {boost.name}", "boost_promos": boost_promo_objects})

@app.route("/create_protected_promo", methods=["POST"])
def create_protected_promo():
    data = request.get_json()
    protected_amount = int(data.get("protected_amount", 0))
    eligible_leagues = data.get("eligible_leagues", [])

    protected = ProtectedPromo(protected_amount, eligible_leagues)
    protected_promo_objects.append(protected.__dict__)

    return jsonify({"response": f"Created Protected Play Promo: {protected.name}", "protected_promos": protected_promo_objects})

@app.route("/edit", methods=["POST"])
def edit():
    global pick_objects
    data = request.get_json()
    obj_id = data.get("id")

    for obj in pick_objects:
        if obj.pID == obj_id:
            # Update fields on the Pick object
            obj.name = data.get("name", obj.name)
            obj.decimalOdds = float(data.get("odds", obj.decimalOdds))
            obj.pick_origin = data.get("pick_origin", obj.pick_origin)
            obj.league = data.get("league", obj.league)
            obj.reusable = data.get("reusable", obj.reusable)
            obj.capital_limit = int(data.get("capital_limit", obj.capital_limit))
            obj.gameID = int(data.get("mutual_exclusion", obj.gameID))
            obj.pick_type = data.get("pick_type", obj.pick_type)
            obj.player_team = data.get("player_team", obj.player_team)
            obj.stat_type = data.get("stat_type", obj.stat_type)

            name = obj.name
            odds = obj.decimalOdds
            leagues = obj.league
            pick_origins = obj.pick_origin
            pick_type = obj.pick_type
            player_team = obj.player_team
            stat_type = obj.stat_type

            # Determine team_a and team_b based on pick_type
            if pick_type == "MoneyLine":
                team_a = name
                team_b = "Other"
                player_team = "None"
            else:
                team_a = "Over"
                team_b = "Under"

            implied_prob = round(1 / odds, 4)
            today = datetime.today().strftime("%Y-%m-%d")

            # Recalculate expert prediction score
            expert_predictions = []
            total_score = 0

            for origin_obj in pick_origins:
                origin = origin_obj.get("name")
                origin_conf = origin_obj.get("confidence", None)

                origin_accuracy = origin_profiles[origin]() if callable(origin_profiles[origin]) else origin_profiles[origin]
                norm_conf = origin_conf / 100 if origin_conf is not None else None

                expert_predictions.append((origin, 1, norm_conf))

                used_conf = origin_conf if origin_conf is not None else 75.0
                score = confidence_score(odds, used_conf, origin_accuracy)
                total_score += score

            final_score = round(total_score / len(expert_predictions), 2) if expert_predictions else 0
            obj.confidence = None
            obj.confidence_score = final_score

            # Update the database
            for league in leagues:
                event_id = generate_event_id(name, league)
                obj.event_id = event_id

                try:
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()

                    cursor.execute("""
                        INSERT OR REPLACE INTO events (
                            event_id, event_date, league,
                            team_a, team_b, actual_result,
                            pick_type, player_team, stat_type
                        ) VALUES (?, ?, ?, ?, ?, COALESCE(
                            (SELECT actual_result FROM events WHERE event_id = ?), NULL
                        ), ?, ?, ?)
                    """, (
                        event_id, today, league,
                        team_a, team_b, event_id,
                        pick_type, player_team, stat_type
                    ))

                    cursor.execute("""
                        INSERT OR REPLACE INTO crowd_predictions (event_id, crowd_probability)
                        VALUES (?, ?)
                    """, (event_id, implied_prob))

                    cursor.execute("DELETE FROM expert_predictions WHERE event_id = ?", (event_id,))
                    for origin, prediction, confidence in expert_predictions:
                        cursor.execute("""
                            INSERT INTO expert_predictions (event_id, expert_name, prediction, confidence)
                            VALUES (?, ?, ?, ?)
                        """, (event_id, origin, prediction, confidence))

                        cursor.execute("SELECT 1 FROM expert_reliability WHERE expert_name = ?", (origin,))
                        if not cursor.fetchone():
                            cursor.execute("""
                                INSERT INTO expert_reliability (expert_name)
                                VALUES (?)
                            """, (origin,))

                    conn.commit()
                    conn.close()

                except Exception as e:
                    print(f"❌ DB Error while editing {event_id}: {e}")

            break

    return jsonify({"objects": [p.__dict__ for p in pick_objects]})

##### CURRENT SPOT IS CREATING RELIABLE EVENT ID FROM PICK ID ####
def get_event_id_from_pick_id(pick_id):
    for pick in pick_objects:
        print(pick.pID)
        if pick.pID == pick_id:
            return pick.event_id
    return None



@app.route("/submit_verified", methods=["POST"])
def submit_verified():
    data = request.get_json()
    verified = data.get("verified", [])

    print(pick_objects)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    updated_ids = []
    localID = 0

    for item in verified:
        localID+=1
        pick_id = localID
        actual_result = item["actual_result"]
        event_id = get_event_id_from_pick_id(pick_id)
        print(event_id)

        # ✅ Find pick object by ID (from actual class instances)
        pick = next((p for p in pick_objects if hasattr(p, "pID") and p.pID == pick_id), None)
        if not pick:
            print(f"❌ Pick ID {pick_id} not found in memory.")
            continue

        event_id = getattr(pick, "event_id", None)
        if not event_id:
            print(f"❌ Pick ID {pick_id} has no event_id.")
            continue

        print(f"✅ Updating event_id: {event_id} → actual_result: {actual_result}")
        try:
            cursor.execute("""
                UPDATE events
                SET actual_result = ?
                WHERE event_id = ?
            """, (actual_result, event_id))

            if cursor.rowcount > 0:
                updated_ids.append(event_id)
            else:
                print(f"⚠️ No rows updated for {event_id} (may not exist in DB).")

        except Exception as e:
            print(f"❌ DB error updating {event_id}: {e}")

    conn.commit()
    conn.close()

    return jsonify({"message": f"Updated {len(updated_ids)} events with actual results."})







@app.route("/load_sample_picks", methods=["POST"])
def load_sample_picks():
    global pick_objects
    pick_objects = []

    num_picks = 8
    example_names = ["Lakers ML", "Yankees -1.5", "Chiefs +3", "Over 8.5", "Under 220", "Dodgers ML", "Ravens -2.5", "Heat +6", "Bills ML", "Nets Over 230"]
    origins = ["Model", "Gut", "Trend", "Public", "Sharp", "Me"]
    leagues = ["NBA", "NFL", "MLB", "NHL"]

    for i in range(num_picks):
        name = random.choice(example_names) + f" #{i+1}"
        confidence = random.randint(40, 99)
        odds = round(random.uniform(1.5, 2.5), 2)
        pick_origin = random.sample(origins, k=random.randint(1, 3))
        league = random.sample(leagues, k=random.randint(1, 2))
        reusable = random.choice([True, False])
        capital_limit = random.randint(10, 100)

        # Simple implied confidence score for testing
        implied_prob = 1 / odds
        expert_conf = confidence / 100
        expert_acc = random.uniform(0.5, 0.9)
        score = round(100 * (expert_acc * expert_conf + (1 - expert_acc) * implied_prob), 2)

        pick_objects.append({
            "id": i + 1,
            "name": name,
            "confidence": confidence,
            "odds": odds,
            "pick_origin": pick_origin,
            "league": league,
            "reusable": reusable,
            "capital_limit": capital_limit,
            "confidence_score": score
        })

    return jsonify({"message": f"{num_picks} sample picks loaded.", "objects": pick_objects})


@app.route("/clear_picks", methods=["POST"])
def clear_picks():
    global pick_objects
    pick_objects = []

    # Optional: clear optimizer results too if you're storing those separately
    # e.g., if you eventually save best_score or best_label in global vars

    return jsonify({
        "message": "All picks cleared.",
        "objects": pick_objects
    })


@app.route("/delete", methods=["POST"])
def delete():
    data = request.get_json()
    obj_id = data.get("id")

    global pick_objects
    deleted_pick = None

    # Find the pick
    for obj in pick_objects:
        if obj["id"] == obj_id:
            deleted_pick = obj
            break

    if deleted_pick:
        name = deleted_pick["name"]
        leagues = deleted_pick["league"]
        for league in leagues:
            event_id = generate_event_id(name, league)
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM expert_predictions WHERE event_id = ?", (event_id,))
                cursor.execute("DELETE FROM crowd_predictions WHERE event_id = ?", (event_id,))
                cursor.execute("DELETE FROM events WHERE event_id = ?", (event_id,))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"DB error while deleting event {event_id}: {e}")

        # Remove from in-memory list
        pick_objects = [obj for obj in pick_objects if obj["id"] != obj_id]

    return jsonify({
   "objects": [p.to_dict() for p in pick_objects]
    })


@app.route("/update_accuracy", methods=["POST"])
def update_accuracy():
    global user_accuracy
    data = request.get_json()
    try:
        user_accuracy = float(data.get("accuracy", 0.0))
    except ValueError:
        user_accuracy = 0.0
    return jsonify({"message": f"Accuracy updated to {user_accuracy}", "user_accuracy": user_accuracy})

if __name__ == "__main__":
    app.run(debug=True)
