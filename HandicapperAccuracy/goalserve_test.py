import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import pymysql
import os
from planet_scale_port import get_connection

def fetch_goalserve_team_stats(API_KEY, DATE, TEAM_NAME):
    """
    Fetch and print hitter and pitcher stats from Goalserve MLB data for a given team and date.
    
    Parameters:
    - API_KEY (str): Your Goalserve API key
    - DATE (str): Date in format "dd.MM.yyyy"
    - TEAM_NAME (str): Substring of the team name (case-insensitive), e.g., "angels"
    """

    url = f"http://www.goalserve.com/getfeed/{API_KEY}/baseball/usa?date={DATE}&json=1"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"❌ Request failed with status code {response.status_code}")
        return

    data = response.json()
    matches = data.get("scores", {}).get("category", {}).get("match", [])
    found = False

    def to_float(val):
        try:
            return float(val)
        except:
            return None

    for match in matches:
        home = match.get("hometeam", {}).get("name", "").lower()
        away = match.get("awayteam", {}).get("name", "").lower()

        if TEAM_NAME.lower() in home or TEAM_NAME.lower() in away:
            print(f"\n✅ Found {TEAM_NAME.title()} game: {match['awayteam']['name']} at {match['hometeam']['name']}")

            team_side = "hometeam" if TEAM_NAME.lower() in home else "awayteam"
            opponent_side = "awayteam" if team_side == "hometeam" else "hometeam"
            team_name = match[team_side]["name"]
            opponent_name = match[opponent_side]["name"]

            hitters = match.get("stats", {}).get("hitters", {}).get(team_side, {}).get("player", [])
            pitchers = match.get("stats", {}).get("pitchers", {}).get(team_side, {}).get("player", [])

            if hitters:
                print(f"\n📋 Batting Stats for {team_name}:")
                for p in hitters:
                    player_stats = {
                        "player_name": p.get("name"),
                        "game_date": DATE,
                        "isHitter": 1,
                        "isPitcher": 0,
                        "team": team_name,
                        "opponent": opponent_name,
                        "hits": to_float(p.get("hits")),
                        "singles": None,  # Not provided
                        "doubles": to_float(p.get("doubles")),
                        "triples": to_float(p.get("triples")),
                        "runs": to_float(p.get("runs")),
                        "rbi": to_float(p.get("runs_batted_in")),
                        "home_runs": to_float(p.get("home_runs")),
                        "walks_hitter": to_float(p.get("walks")),
                        "hit_by_pitch_hitter": to_float(p.get("hit_by_pitch")),
                        "at_bats_hitter": to_float(p.get("at_bats")),
                        "strikeouts_hitter": to_float(p.get("strikeouts")),
                        "total_bases": to_float(p.get("total_bases")),
                        "stolen_bases": to_float(p.get("stolen_bases")),
                        "fantasy_points_hitter": None
                    }
                    print(player_stats)
            else:
                print("⚠️ No hitter stats found for this team.")

            if pitchers:
                print(f"\n⚾ Pitching Stats for {team_name}:")
                for p in pitchers:
                    ip = p.get("innings_pitched", "0")
                    outs = 0
                    if "." in ip:
                        whole, frac = ip.split(".")
                        outs = int(whole) * 3 + int(frac)
                    else:
                        outs = int(ip) * 3

                    pc_st = p.get("pc-st", "0-0").split("-")[0]
                    player_stats = {
                        "player_name": p.get("name"),
                        "game_date": DATE,
                        "isHitter": 0,
                        "isPitcher": 1,
                        "team": team_name,
                        "opponent": opponent_name,
                        "earned_runs": to_float(p.get("earned_runs")),
                        "total_pitches": int(pc_st),
                        "strikeouts_pitcher": to_float(p.get("strikeouts")),
                        "outs_pitched": outs,
                        "hits_allowed": to_float(p.get("hits")),
                        "hit_by_pitch_pitcher": to_float(p.get("hbp")),
                        "win_credited": 1.0 if p.get("win") else 0.0,
                        "quality_start": 1.0 if outs >= 18 and to_float(p.get("earned_runs")) <= 3 else 0.0
                    }
                    print(player_stats)
            else:
                print("⚠️ No pitcher stats found for this team.")

            found = True
            break

    if not found:
        print(f"❌ No game found for {TEAM_NAME.title()} on {DATE}.")

def parse_minutes(min_str):
    if isinstance(min_str, str) and ":" in min_str:
        parts = min_str.split(":")
        return int(parts[0]) + int(parts[1]) / 60.0
    elif isinstance(min_str, (int, float)):
        return float(min_str)
    return 0.0


# def fetch_nba_box_score_xml(api_key, date_str, team_name):
#     url = f"http://www.goalserve.com/getfeed/{api_key}/bsktbl/nba-scores?date={date_str}"
#     response = requests.get(url)

#     print(f"📡 Status: {response.status_code}")
#     if response.status_code != 200:
#         print("❌ Failed to fetch data.")
#         return 

#     # Parse XML
#     try:
#         root = ET.fromstring(response.content)
#     except ET.ParseError as e:
#         print(f"❌ XML parse error: {e}")
#         return

#     matches = root.findall(".//match")
#     print(f"📆 Found {len(matches)} games on {date_str}")

#     for match in matches:
#         home = match.find("hometeam").attrib.get("name", "")
#         away = match.find("awayteam").attrib.get("name", "")
#         print(f"📝 Game: {away} at {home}")

#         if team_name.lower() in home.lower():
#             side = "hometeam"
#         elif team_name.lower() in away.lower():
#             side = "awayteam"
#         else:
#             continue  # not the team we’re looking for

#         print(f"🎯 Found game involving {team_name}: {away} at {home}")
#         players_path = f".//player_stats/{side}"
#         players = match.find(players_path)

#         if players is None:
#             print("⚠️ No player stats found.")
#             return

#         print("\n📊 Player Stats:")
#         for group in ["starters", "bench"]:
#             for player in players.find(group):
#                 print({
#                     "name": player.attrib.get("name"),
#                     "minutes": player.attrib.get("minutes"),
#                     "points": player.attrib.get("points"),
#                     "rebounds": player.attrib.get("total_rebounds"),
#                     "assists": player.attrib.get("assists"),
#                     "steals": player.attrib.get("steals"),
#                     "blocks": player.attrib.get("blocks"),
#                     "turnovers": player.attrib.get("turnovers"),
#                 })
#         return  # done after first matching game

#     print("❌ No matching team found.")

def fetch_nba_box_score_xml(api_key, date_str, team_name):
    url = f"http://www.goalserve.com/getfeed/{api_key}/bsktbl/nba-scores?date={date_str}"
    response = requests.get(url)
    print(f"📡 Status: {response.status_code}")
    
    if response.status_code != 200 or not response.content.strip():
        print("❌ Failed to fetch data or empty response.")
        return []

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        print(f"❌ XML parse error: {e}")
        return []

    stats = []
    for match in root.findall(".//match"):
        home = match.find("hometeam").attrib.get("name", "")
        away = match.find("awayteam").attrib.get("name", "")
        for side, opponent in [("hometeam", away), ("awayteam", home)]:
            team = match.find(side).attrib.get("name", "")
            if team_name.lower() not in team.lower():
                continue
            players_root = match.find(f".//player_stats/{side}")
            if players_root is None:
                continue
            for group in ["starters", "bench"]:
                for player in players_root.find(group):
                    try:
                        print({
                            "name": player.attrib.get("name")
                        })

                       # Core stats
                        name =  player.attrib.get("name")
                        points = float(player.attrib.get("points", 0) or 0)
                        assists = float(player.attrib.get("assists", 0) or 0)
                        rebounds = float(player.attrib.get("total_rebounds", 0) or 0)
                        off_rebounds = float(player.attrib.get("offence_rebounds", 0) or 0)
                        def_rebounds = float(player.attrib.get("defense_rebounds", 0) or 0)
                        steals = float(player.attrib.get("steals", 0) or 0)
                        blocks = float(player.attrib.get("blocks", 0) or 0)
                        turnovers = float(player.attrib.get("turnovers", 0) or 0)
                        personal_fouls = float(player.attrib.get("personal_fouls", 0) or 0)

                        # Shooting stats
                        fg_made = float(player.attrib.get("field_goals_made", 0) or 0)
                        fg_attempts = float(player.attrib.get("field_goals_attempts", 0) or 0)
                        ft_made = float(player.attrib.get("freethrows_goals_made", 0) or 0)
                        ft_attempts = float(player.attrib.get("freethrows_goals_attempts", 0) or 0)
                        threes_made = float(player.attrib.get("threepoint_goals_made", 0) or 0)
                        threes_attempts = float(player.attrib.get("threepoint_goals_attempts", 0) or 0)

                        # Playtime and misc
                        minutes = float(player.attrib.get("minutes", 0) or 0)
                        plus_minus = player.attrib.get("plus_minus", "0").strip()
                        plus_minus_val = int(plus_minus) if plus_minus.startswith("-") or plus_minus.isdigit() else int(plus_minus.replace("+", ""))

                        pa = points + assists
                        pr = points + rebounds
                        ra = rebounds + assists
                        pra = points + rebounds + assists
                        fantasy_points = (
                            points * 1 +
                            rebounds * 1.2 +
                            assists * 1.5 +
                            blocks * 3 +
                            steals * 3 +
                            turnovers * -1
                        )

                        stats.append({
                            "player_name": name,
                            "points": points,
                            "assists": assists,
                            "rebounds": rebounds,
                            "off_rebounds": off_rebounds,
                            "def_rebounds": def_rebounds,
                            "steals": steals,
                            "blocks": blocks,
                            "turnovers": turnovers,
                            "personal_fouls": personal_fouls,
                            "threes_made": threes_made,
                            "minutes": minutes,
                            "team": team,
                            "opponent": opponent,
                            "pa": pa,
                            "pr": pr,
                            "ra": ra,
                            "pra": pra,
                            "fantasy_points": fantasy_points,
                            "plus_minus": plus_minus_val,
                            "fg_made": fg_made,
                            "fg_attempts": fg_attempts,
                            "ft_made": ft_made,
                            "ft_attempts": ft_attempts,
                            "threes_attempts": threes_attempts,
                            "league": "NBA"
                        })
                    except Exception:
                        continue

    return stats



def bulk_import_nba_logs(api_key, start_date, end_date):
    conn = get_connection()
    cursor = conn.cursor()
    current = start_date

    while current <= end_date:
        date_str = current.strftime("%d.%m.%Y")
        print(f"\n📆 Processing {date_str}")
        all_stats = []

        for team in [
            "Lakers", "Warriors", "Celtics", "Heat", "Bucks",
            "Nuggets", "Suns", "Clippers", "76ers", "Knicks",
            "Mavericks", "Grizzlies", "Kings", "Timberwolves", "Pelicans",
            "Hawks", "Raptors", "Bulls", "Cavaliers", "Pacers",
            "Magic", "Wizards", "Pistons", "Hornets", "Nets",
            "Spurs", "Jazz", "Trail Blazers", "Thunder", "Rockets"
        ]:  # Use full team list for production
            all_stats.extend(fetch_nba_box_score_xml(api_key, date_str, team))

        inserted = 0
        for stat in all_stats:
            try:
                stat["game_date"] = datetime.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")
                cursor.execute("""
                    INSERT INTO NBA_Player_Logs (
                        player_name, game_date, team, opponent,
                        points, rebounds, assists,
                        off_rebounds, def_rebounds,
                        personal_fouls, threes_made, threes_attempts,
                        fg_made, fg_attempts, ft_made, ft_attempts,
                        blocks, steals, turnovers, minutes,
                        pa, pr, ra, pra, fantasy_points,
                        plus_minus, league
                    ) VALUES (
                        %(player_name)s, %(game_date)s, %(team)s, %(opponent)s,
                        %(points)s, %(rebounds)s, %(assists)s,
                        %(off_rebounds)s, %(def_rebounds)s,
                        %(personal_fouls)s, %(threes_made)s, %(threes_attempts)s,
                        %(fg_made)s, %(fg_attempts)s, %(ft_made)s, %(ft_attempts)s,
                        %(blocks)s, %(steals)s, %(turnovers)s, %(minutes)s,
                        %(pa)s, %(pr)s, %(ra)s, %(pra)s, %(fantasy_points)s,
                        %(plus_minus)s, %(league)s
                    );
                """, stat)
                inserted += 1
            except Exception as e:
                print(f"⚠️ Insert error: {e}")

        conn.commit()
        print(f"✅ Inserted {inserted} logs for {date_str}")
        current += timedelta(days=1)

    conn.close()
    print("\n🏁 Done with import.")

# Example usage
#fetch_nba_box_score_xml("b6cccb2cee43489fcb7908dd8718f057", "22.10.2024", "Lakers")
bulk_import_nba_logs("b6cccb2cee43489fcb7908dd8718f057", datetime(2024, 10, 22), datetime(2025, 4, 13))
# fetch_goalserve_team_stats("b6cccb2cee43489fcb7908dd8718f057", "30.04.2025", "angels")
