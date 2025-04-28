import requests
import sqlite3
import time
from datetime import datetime, timedelta

DB_PATH = "/Users/maxhartel/Desktop/Desktop - Max’s MacBook Pro/Project Parlay/Project-Parlay/Pick_Confidence"

SCOREBOARD_URL = "https://stats.nba.com/stats/scoreboardv2"
BOXSCORE_URL = "https://stats.nba.com/stats/boxscoretraditionalv2"

HEADERS = {
    "Host":               "stats.nba.com",
    "Connection":         "keep-alive",
    "Accept":             "application/json, text/plain, */*",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token":  "true",
    "User-Agent":         "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Referer":            "https://www.nba.com/",
    "Accept-Language":    "en-US,en;q=0.9",
}

boxscore_cache = {}
schedule_cache = {}

def connect():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def parse_minutes(min_str):
    """Convert MIN string format (e.g., '32:45') to decimal minutes (e.g., 32.75)."""
    if isinstance(min_str, str) and ":" in min_str:
        parts = min_str.split(":")
        return int(parts[0]) + int(parts[1]) / 60.0
    elif isinstance(min_str, (int, float)):
        return float(min_str)
    return 0.0

def get_games_on(date_obj):
    """Return all NBA game IDs for a given date."""
    key = date_obj.strftime("%Y-%m-%d")
    if key in schedule_cache:
        return schedule_cache[key]

    params = {
        "GameDate": date_obj.strftime("%m/%d/%Y"),
        "LeagueID": "00",
        "DayOffset": 0
    }
    res = requests.get(SCOREBOARD_URL, headers=HEADERS, params=params, timeout=10)
    res.raise_for_status()
    data = res.json()["resultSets"][0]
    hdrs = data["headers"]
    rows = data["rowSet"]
    gid_i = hdrs.index("GAME_ID")
    game_ids = [row[gid_i] for row in rows]
    schedule_cache[key] = game_ids
    return game_ids

def get_boxscore(game_id):
    """Return a dict of player_name_lower → full stat dictionary with team and opponent."""
    if game_id in boxscore_cache:
        return boxscore_cache[game_id]

    params = {"GameID": game_id, "LeagueID": "00"}
    res = requests.get(BOXSCORE_URL, headers=HEADERS, params=params, timeout=10)
    res.raise_for_status()
    data = res.json()["resultSets"]

    player_data = next(d for d in data if d["name"] == "PlayerStats")
    team_data = next(d for d in data if d["name"] == "TeamStats")

    hdrs = player_data["headers"]
    rows = player_data["rowSet"]
    team_hdrs = team_data["headers"]
    team_rows = team_data["rowSet"]

    stat_indices = {
        "PLAYER_NAME": hdrs.index("PLAYER_NAME"),
        "TEAM_ABBREVIATION": hdrs.index("TEAM_ABBREVIATION"),
        "PTS": hdrs.index("PTS"),
        "REB": hdrs.index("REB"),
        "AST": hdrs.index("AST"),
        "STL": hdrs.index("STL"),
        "BLK": hdrs.index("BLK"),
        "TOV": hdrs.index("TO"),
        "MIN": hdrs.index("MIN"),
        "PF": hdrs.index("PF"),
        "FG3M": hdrs.index("FG3M"),
    }

    teams_in_game = [row[team_hdrs.index("TEAM_ABBREVIATION")] for row in team_rows]

    by_player = {}
    for row in rows:
        name = row[stat_indices["PLAYER_NAME"]].strip().lower()
        player_team = row[stat_indices["TEAM_ABBREVIATION"]]

        if len(teams_in_game) == 2:
            opponent_team = next(t for t in teams_in_game if t != player_team)
        else:
            opponent_team = "Unknown"

        by_player[name] = {
            "team": player_team,
            "opponent": opponent_team,
            "points": row[stat_indices["PTS"]],
            "rebounds": row[stat_indices["REB"]],
            "assists": row[stat_indices["AST"]],
            "steals": row[stat_indices["STL"]],
            "blocks": row[stat_indices["BLK"]],
            "turnovers": row[stat_indices["TOV"]],
            "minutes": parse_minutes(row[stat_indices["MIN"]]),
            "personal_fouls": row[stat_indices["PF"]],
            "threes_made": row[stat_indices["FG3M"]],
        }
    boxscore_cache[game_id] = by_player
    return by_player

def insert_player_stat_entry(player_name, game_date, league, team, opponent,
                              points, rebounds, assists, personal_fouls, threes_made,
                              blocks, steals, turnovers, minutes):
    """Insert a single player stat record into player_stats table."""
    try:
        conn = connect()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO player_stats (
                player_name, game_date, league, team, opponent,
                points, rebounds, assists, personal_fouls, threes_made,
                blocks, steals, turnovers, minutes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            player_name, game_date, league, team, opponent,
            points, rebounds, assists, personal_fouls, threes_made,
            blocks, steals, turnovers, minutes
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Insert failed for {player_name} on {game_date}: {e}")

def fetch_and_store_player_season(player_name, start_date, end_date, league="NBA"):
    """Fetch and store full season stats for a single player."""
    current_date = start_date
    player_key = player_name.strip().lower()

    while current_date <= end_date:
        try:
            game_ids = get_games_on(current_date)
            for gid in game_ids:
                boxscores = get_boxscore(gid)
                if player_key in boxscores:
                    stats = boxscores[player_key]
                    insert_player_stat_entry(
                        player_name=player_name,
                        game_date=current_date.strftime("%Y-%m-%d"),
                        league=league,
                        team=stats["team"],
                        opponent=stats["opponent"],
                        points=stats["points"],
                        rebounds=stats["rebounds"],
                        assists=stats["assists"],
                        personal_fouls=stats["personal_fouls"],
                        threes_made=stats["threes_made"],
                        blocks=stats["blocks"],
                        steals=stats["steals"],
                        turnovers=stats["turnovers"],
                        minutes=stats["minutes"],
                    )
                    print(f"✅ Inserted stats for {player_name} on {current_date.strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"⚠️ Error on {current_date.strftime('%Y-%m-%d')}: {e}")

        current_date += timedelta(days=1)
        time.sleep(0.75)  # gentle crawl

# === Usage Example ===
# fetch_and_store_player_season("Tyler Herro", datetime(2023, 10, 24), datetime(2024, 4, 15))