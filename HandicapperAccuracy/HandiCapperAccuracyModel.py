#import sqlite3
import math
import pandas as pd
import numpy as np
import pymysql
from typing import List, Tuple, Optional, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity
from scipy.interpolate import interp1d
from scipy.special import expit  # sigmoid
from scipy.optimize import curve_fit
import scipy.stats as stats
import mysql.connector
from itertools import combinations
import gc
import psutil, os
from planet_scale_port import get_connection, get_dict_connection, get_player_and_stat_type
conn = get_connection()

DB_PATH = "/Users/maxhartel/Desktop/Desktop - Max’s MacBook Pro/Project Parlay/Project-Parlay/Pick_Confidence"

def connect():
    #return sqlite3.connect(DB_PATH, check_same_thread=False)
    return get_connection()

def crowd_log_odds(p: float) -> float:
    """Convert crowd probability to log-odds."""
    p = max(min(p, 0.999), 0.001)  # avoid division by zero
    return math.log(p / (1 - p))

# ✅ Define which stats to include
STAT_COLUMNS = [
    "points", "rebounds", "assists", "personal_fouls", "threes_made",
    "blocks", "steals", "turnovers", "minutes", "pa", "pr", "ra", "pra",
    "fantasy_points", "off_rebounds", "def_rebounds", "plus_minus",
    "fg_made", "fg_attempts", "ft_made", "ft_attempts", "threes_attempts",
    "entry_id", "team", "opponent", "player_name", "league"
]

STAT_COLUMNS_FOR_COV = [
    "points", "rebounds", "assists", "personal_fouls", "threes_made",
    "blocks", "steals", "turnovers", "minutes", "pa", "pr", "ra", "pra",
    "fantasy_points", "off_rebounds", "def_rebounds",
    "fg_made", "fg_attempts", "ft_made", "ft_attempts", "threes_attempts"
]

# ✅ Rename stat columns to "Player's Stat"
def rename_player_stats(df, player_name, stat_columns):
    renamed_df = df.copy()

    # Only rename the stats you actually care about
    rename_map = {
        col: f"{player_name}'s {col.replace('_', ' ')}"
        for col in stat_columns
        if col in renamed_df.columns  # only rename existing columns
    }

    return renamed_df.rename(columns=rename_map)

def merge_players_stats(players, league, connection):
    all_dfs = []

    for player in players:
        process = psutil.Process(os.getpid())
        print(f"🔍 Memory usage (MB): {process.memory_info().rss / 1024**2:.2f}")

        df = fetch_player_game_logs(player, league, connection)
        # print("Dataframe Preview: ")
        # print(df.head(3))

        if df is None or df.empty:
            print(f"⚠️ No data for player: {player}")
            continue

        try:
            df = rename_player_stats(df, player, STAT_COLUMNS_FOR_COV)
            stat_cols = [col for col in df.columns if any(stat in col.lower() for stat in STAT_COLUMNS_FOR_COV)]
            df = df[['game_date'] + stat_cols]
            # print(f"✅ Columns after renaming for {player}:")
            # print(df.columns.tolist())
        except Exception as e:
            print(f"❌ Rename failed for {player}: {e}")
            continue

        all_dfs.append(df)
        del df
        gc.collect()

    if not all_dfs:
        print("❌ No player data available for merging.")
        return None

    # Concatenate all player DataFrames vertically (stacked format)
    stacked_df = pd.concat(all_dfs, ignore_index=True)
    numeric_cols = stacked_df.select_dtypes(include='number').columns.tolist()
    if "game_date" not in numeric_cols:
        numeric_cols.append("game_date")  # Ensure game_date isn't dropped

    # Pivot to wide format with player-specific stat columns
    try:
         wide_df = stacked_df.pivot_table(
            index="game_date",
            values=[col for col in numeric_cols if col != "game_date"],
            aggfunc="mean"
        ).reset_index()
    except Exception as e:
        print(f"❌ Pivot to wide format failed: {e}")
        return None

    return wide_df

# ✅ Display top N absolute correlations or covariances between stat pairs
# def top_stat_pairs(matrix, top_n=5, metric='Covariance'):
#     pairs = []
#     for i, j in combinations(matrix.columns, 2):
#         val = matrix.loc[i, j]
#         pairs.append(((i, j), val))

#     # Sort by absolute value
#     sorted_pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_n]
#     print(f"\n🔝 Top {top_n} Stat Pairs by {metric}:")
#     for (stat1, stat2), value in sorted_pairs:
#         print(f"  {stat1} & {stat2}: {value:.2f}")

# ✅ Display top N absolute correlations or covariances between stat pairs of 2 different players

def print_matrix_columns(matrix, label="Matrix"):
    print(f"\n📋 Columns in {label}:")
    for col in matrix.columns:
        print(f" - {col}")


def top_stat_pairs(matrix, top_n=5, metric='Covariance'):
    #print_matrix_columns(matrix, label="Covariance Matrix")
    pairs = []
    for i, j in combinations(matrix.columns, 2):
        # Skip pairs from the same player
        player_i = i.split("'s")[0].strip()
        player_j = j.split("'s")[0].strip()
        if player_i == player_j:
            continue

        val = matrix.loc[i, j]
        pairs.append(((i, j), val))

    # Sort by absolute value
    sorted_pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_n]

    print(f"\n🔝 Top {top_n} Stat Pairs by {metric} (Different Players Only):")
    for (stat1, stat2), value in sorted_pairs:
        print(f"  {stat1} & {stat2}: {value:.2f}")

# ✅ Master analysis function
def analyze_multiplayer_stats(players, league, connection):
    df = merge_players_stats(players, league, connection)

    # Drop non-stat columns
    stat_cols = [col for col in df.columns if any(stat in col.lower() for stat in STAT_COLUMNS_FOR_COV)]
    #print(stat_cols)
    df_stats = df[stat_cols].fillna(0)
    # print(df_stats.head(5))

    # Compute covariance and correlation
    cov_matrix = df_stats.cov()
    corr_matrix = df_stats.corr()

    # Top correlated and covariant stat pairs
    top_stat_pairs(cov_matrix, metric='Covariance')
    top_stat_pairs(corr_matrix, metric='Correlation')

    return cov_matrix, corr_matrix, df

#### SQLite Version ####
# def connect():
#     return sqlite3.connect(DB_PATH, check_same_thread=False)

def logistic_cdf(x, mu, s):
    """Logistic CDF used to model expert implied distributions."""
    return expit((x - mu) / s)

def fit_logistic_cdf(thresholds, probs):
    """
    Fit a logistic CDF from expert predictions at different thresholds.
    Returns (mu, s) parameters of the fit.
    """
    thresholds = np.array(thresholds)
    probs = np.clip(probs, 0.01, 0.99)  # avoid edge-case errors

    try:
        popt, _ = curve_fit(logistic_cdf, thresholds, probs, p0=[np.mean(thresholds), 1.0])
        return popt[0], popt[1]  # mu, s
    except:
        return np.mean(thresholds), 1.0  # fallback

def predict_prob_from_cdf(target_threshold, mu, s):
    """Get calibrated expert probability for the actual betting line."""
    return round(1 - logistic_cdf(target_threshold, mu, s), 4)

def bootstrap_posterior_probability(
    crowd_prob: float,
    expert_predictions: List[Tuple[str, int, Optional[float]]],
    expert_reliabilities: Dict[str, Tuple[float, float]],
    n_samples: int = 500
) -> Tuple[float, float, float]:
    """
    Compute posterior probability with uncertainty via bootstrapping.

    Returns:
        mean_prediction, lower_bound, upper_bound
    """
    base_log_odds = log_odds(crowd_prob)
    bootstrap_probs = []

    for _ in range(n_samples):
        total_log_odds = base_log_odds

        for name, prediction, confidence in expert_predictions:
            alpha, beta = expert_reliabilities.get(name, (0.6, 0.6))
            simulated_accuracy = simulate_expert_accuracy(alpha, beta, prediction)

            # fallback confidence
            conf = confidence if confidence is not None else 0.75

            weight = math.log(simulated_accuracy / (1 - simulated_accuracy))
            signal = 1 if prediction == 1 else -1
            total_log_odds += signal * weight * conf

        posterior = sigmoid(total_log_odds)
        bootstrap_probs.append(posterior)

    mean_pred = round(np.mean(bootstrap_probs), 4)
    lower = round(np.percentile(bootstrap_probs, 2.5), 4)
    upper = round(np.percentile(bootstrap_probs, 97.5), 4)

    return mean_pred, lower, upper

def fetch_expert_predictions_for_team_date(team: str, league: str, game_date: str, connection) -> pd.DataFrame:
    """
    Fetches all expert predictions for players on the given team, in the given league, for a specific game date.

    Parameters:
        team (str): The team abbreviation (e.g., "MIA").
        league (str): The league (e.g., "NBA").
        game_date (str): The game date in 'YYYY-MM-DD' format.
        connection: MySQL database connection object.

    Returns:
        pd.DataFrame: A DataFrame containing all matching expert predictions.
    """
    query = """
        SELECT 
            event_id,
            stat_threshold,
            confidence,
            expert_name
        FROM expert_predictions
        WHERE team = %s
          AND league = %s
          AND game_date = %s
    """
    try:
        cursor = connection.cursor()
        cursor.execute(query, (team, league, game_date))
        rows = cursor.fetchall()

        df2 = pd.DataFrame(columns=['event_id', 'stat_threshold', 'confidence', 'expert_name'])

        #Selects First Row
        for i, row in enumerate(rows):
            df2.loc[i] = row

        print("expert predictions for teamates df 2:")
        print(df2.head(5))
        

        df2['player_name'] = [None] * len(rows)
        df2['stat_type'] = [None] * len(rows)

        count = 0
        for event_id in df2['event_id']:
            print(event_id)

            result = get_player_and_stat_type(event_id)

            if result:
                print(f"Player Name: {result[0]}, Stat Type: {result[1]}")
                df2.loc[count, "player_name"] = result[0]
                df2.loc[count, "stat_type"] = result[1]


            else:
                print("No data found for the given event_id.")

            count += 1


        print("expert predictions for teamates df 3:")
        print(df2.head(5))
        return df2
        
    except Exception as e:
        print(f"❌ Error fetching expert predictions: {e}")
        return pd.DataFrame()

def compute_team_cdf_with_kde_shift(
    player_name: str,
    stat_type: str,
    league: str,
    connection,
    target_date: str
    ) -> Optional[interp1d]:
        """
        Creates a KDE-interpolated continuous CDF for a player's target stat
        by modeling the team's multivariate distribution and shifting the mean
        based on weighted expert predictions.

        This function builds a **custom statistical distribution** for a specific player's stat
        (e.g., Tyler Herro's Points) by considering the behavior of all teammates on the same team.
        It adjusts the distribution based on expert predictions for that stat and returns a
        **continuous CDF (cumulative distribution function)** that reflects the likelihood of
        different outcomes.

        Step-by-step explanation:

        1. 🔍 Look up the player's team on the target date (e.g., "MIA" for Miami Heat).
        2. 🧑‍🤝‍🧑 Get a list of all active teammates on that team from a helper function.
        3. 📊 Use `analyze_multiplayer_stats()` to gather past game logs for all those players and
        build a multivariate distribution of their stats. Each stat is labeled like
        "Tyler Herro's Points" to keep them separate.
        4. 🎯 Focus in on the stat of interest (like "Tyler Herro's Points") and extract its
        historical values from the dataset.
        5. 🗣️ Pull all expert predictions from your database related to the same team and date.
        Filter down to predictions about the exact player and stat.
        6. ⚖️ For each matching prediction, use its probability (combined_prob) as a weight
        and compute a weighted average predicted value.
        7. 📈 Shift the mean of the original stat data toward this weighted average — a
        "mean shift" — to reflect what the experts expect to happen.
        8. 🧮 Use kernel density estimation (KDE) to fit a smooth probability curve (PDF)
        to this shifted data, then convert it into a CDF.
        9. 🔁 Return a callable function that interpolates this CDF so you can plug in any
        stat value and get back the cumulative probability (i.e., likelihood that the
        player stays under that value).

        This is useful for building probability curves that are:
        ✅ Data-driven (based on actual player/team stats),
        ✅ Adaptable (influenced by expert insights), and
        ✅ Continuous (supporting fine-grained probability analysis for any stat threshold).
        
        """

        # Step 1: Identify the player's current team
        team_query = f"""
            SELECT * FROM {league}_Player_Logs
            WHERE player_name = %s AND game_date = %s
            LIMIT 10
        """
        cursor = connection.cursor()
        cursor.execute(team_query, (player_name, target_date))
        rows = cursor.fetchall()

        #Selects First Row
        for i, row in enumerate(rows):
            dict_row = dict(row)

        #print("Team: " + dict_row["team"])
        team_name = dict_row["team"]


        if not dict_row:
            print(f"❌ No team info found for {player_name} on {target_date}")
            return None
        

        # Step 2: Get active players on the same team
        players = get_current_team_members(team_name, target_date, league)

        # Step 3: Build the multivariate stat matrix with renamed columns
        cov_matrix, corr_matrix, aligned_df = analyze_multiplayer_stats(players, league, connection)

        
        cov_matrix = cov_matrix.fillna(0)
        corr_matrix = corr_matrix.fillna(0)
        print(cov_matrix.head(5))
        print(corr_matrix.head(5))

        analysis = {
            "aligned_df": aligned_df,
            "cov_matrix": cov_matrix,
            "corr_matrix": corr_matrix
        }

        # Now safe to access:
        data_matrix = analysis["aligned_df"].drop(columns=["game_date"])

        stat_column = player_name + "'s " + stat_type.lower()
        if stat_column not in data_matrix.columns:
            print("⚠️ Stat " + stat_column + " not found for " + player_name)
            return None

        original_data = data_matrix[stat_column].values.reshape(-1, 1)

        #TODO: Find which option is better here, removing nan, interpolating nan, investigating why nan occurs, or other
        #print(shifted_data)
        # Remove NaN values from original_data before shifting
        original_data = original_data[~np.isnan(original_data)]

        # Alternatively, fill NaN values with a default value (e.g., 0)
        #original_data = np.nan_to_num(original_data, nan=0)

        original_mean = np.mean(original_data)

        # Step 4: Collect expert predictions for this player's stat on that date
        expert_df = fetch_expert_predictions_for_team_date(team_name, league, target_date, connection)
        if expert_df.empty:
            print("⚠️ No expert predictions found.")
            return None

        weighted_shifts = []
        total_weight = 0.0

        for _, row in expert_df.iterrows():
            if row["player_name"] != player_name or row["stat_type"] != stat_type:
                continue
            prediction = predict_event_by_id(row["event_id"], connection)
            if prediction and prediction.get("combined_prob") is not None:
                weight = prediction["combined_prob"]
                value = row["stat_threshold"]
                weighted_shifts.append(weight * value)
                total_weight += weight

        if total_weight > 0:
            new_mean = sum(weighted_shifts) / total_weight
            shift_amount = new_mean - original_mean
            shifted_data = original_data + shift_amount
        else:
            shifted_data = original_data

        # Reshape shifted_data to be a 2D array (n_samples, 1)
        shifted_data = shifted_data.reshape(-1, 1)


        # Step 5: KDE fit and build CDF
        kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(shifted_data)
        sample_range = np.linspace(0, max(50, shifted_data.max() + 10), 1000).reshape(-1, 1)
        log_dens = kde.score_samples(sample_range)
        density = np.exp(log_dens)
        cdf = np.cumsum(density)
        cdf /= cdf[-1]

        # Step 6: Return interpolated CDF function
        return interp1d(sample_range.ravel(), cdf, bounds_error=False, fill_value=(0.0, 1.0))


def predict_with_learned_weights(event_row, model, expert_names):
    input_vec = np.zeros(len(expert_names))
    expert_index = {name: i for i, name in enumerate(expert_names)}

    for name, pred, _ in event_row["expert_predictions"]:
        signal = 1 if pred == 1 else -1
        if name in expert_index:
            input_vec[expert_index[name]] = signal

    prob = model.predict_proba(input_vec.reshape(1, -1))[0][1]
    return round(prob, 4)

def simulate_expert_accuracy(alpha: float, beta: float, prediction: int) -> float:
    """
    Simulate expert accuracy using alpha (TPR) and beta (TNR) based on their prediction.
    """
    if prediction == 1:
        return np.random.beta(alpha * 10 + 1, (1 - beta) * 10 + 1)
    else:
        return np.random.beta((1 - alpha) * 10 + 1, beta * 10 + 1)


def prepare_training_matrix(events: List[Dict]):
    expert_set = set()
    for event in events:
        for prediction in event["expert_predictions"]:
            expert_set.add(prediction[0])  # expert_name
    experts = sorted(expert_set)
    expert_index = {name: i for i, name in enumerate(experts)}

    X = []
    y = []

    for event in events:
        row = [0] * len(experts)
        for prediction in event["expert_predictions"]:
            name = prediction[0]
            pred_value = prediction[1]
            confidence = prediction[2]

            signal = (1 if pred_value == 1 else -1) * confidence
            row[expert_index[name]] = signal
        X.append(row)
        y.append(event["actual_result"])

    return np.array(X), np.array(y), experts


    return np.array(X), np.array(y), experts

def train_logistic_weights(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def get_expert_reliability():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT expert_name, 
               true_positive, false_positive,
               true_negative, false_negative
        FROM expert_reliability
    """)
    rows = cursor.fetchall()
    conn.close()

    reliability = {}
    for name, tp, fp, tn, fn in rows:
        alpha = tp / (tp + fn) if (tp + fn) > 0 else 0.5  # P("Higher" | Y=1)
        beta = tn / (tn + fp) if (tn + fp) > 0 else 0.5   # P("Lower" | Y=0)
        reliability[name] = (alpha, beta)
    return reliability

def get_expert_accuracy(expert_name: str) -> float:
    """
    Returns the overall accuracy of an expert based on their correct and total predictions.
    Falls back to a neutral default (e.g., 0.6) if the expert is missing or has no predictions.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT correct_predictions, total_predictions
            FROM expert_reliability
            WHERE expert_name = %s
        """, (expert_name,))
        row = cursor.fetchone()
        conn.close()

        if row:
            correct, total = row
            if total > 0:
                return round(correct / total, 4)
        
        return 0.60  # Default fallback

    except Exception as e:
        print(f"⚠️ Error fetching expert accuracy for {expert_name}: {e}")
        return 0.60


def expert_log_likelihood_ratio(prediction: int, tpr: float, tnr: float) -> float:
    """
    Compute log-likelihood ratio for expert prediction.
    prediction: 1 = "Too Low" (thinks event will happen)
                0 = "Too High" (thinks event won't happen)
    """
    tpr = max(min(tpr, 0.999), 0.001)
    tnr = max(min(tnr, 0.999), 0.001)

    if prediction == 1:
        return math.log(tpr / (1 - tnr))  # likelihood of "too low" given Y=1
    else:
        return math.log((1 - tpr) / tnr)  # likelihood of "too high" given Y=0
    

def bayesian_ensemble_prediction(
    crowd_prob: float,
    expert_predictions: List[Tuple[str, int, Optional[float]]],
) -> float:
    """
    Combines crowd probability and expert predictions to produce a Bayesian posterior.
    - crowd_prob: prior P(Y=1) from the crowd
    - expert_predictions: [(expert_name, prediction, confidence)], prediction is 1 for "Higher" (Y=1), 0 for "Lower" (Y=0)
    Returns posterior probability P(Y=1)
    """
    # Base log-odds from crowd
    total_log_odds = crowd_log_odds(crowd_prob)

    # Load calibration map once
    calibration_map = get_expert_confidence_calibration(DB_PATH)

    for expert_name, prediction, confidence in expert_predictions:
        # Fallback if no confidence provided
        conf = confidence if confidence is not None else 0.75
        bucket = round(conf // 0.05 * 0.05, 2)

        # Try to use calibrated accuracy
        calibrated = calibration_map.get((expert_name, bucket))
        if calibrated is not None:
            accuracy = calibrated
        else:
            accuracy = get_expert_accuracy(expert_name)  # fallback to historical

        # Clamp to avoid math errors
        accuracy = max(min(accuracy, 0.999), 0.001)

        # Signal: +1 for “Higher” prediction, -1 for “Lower”
        signal = 1 if prediction == 1 else -1

        # Weight: log-odds of accuracy
        weight = math.log(accuracy / (1 - accuracy))

        # Add expert contribution to the total log-odds
        total_log_odds += signal * weight

    # Convert log-odds back to probability
    odds = math.exp(total_log_odds)
    posterior_prob = odds / (1 + odds)
    return round(posterior_prob, 4)

def get_all_players():
    """
    Fetches all distinct player names from player_stats table.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT player_name FROM player_stats")
    players = [row[0] for row in cursor.fetchall()]
    conn.close()
    return players




###### SQLite Version #########
# def fetch_player_game_logs(player_name: str):
#     """
#     Fetches all available game logs for a given player into a DataFrame.
#     """
#     conn = connect()
#     query = f"""
#         SELECT {', '.join(STAT_COLUMNS)}
#         FROM player_stats
#         WHERE player_name = ?
#     """
#     df = pd.read_sql_query(query, conn, params=(player_name,))
#     conn.close()
#     return df



#### MySQL Version #######
def fetch_player_game_logs(player_name: str, league: str, connection) -> pd.DataFrame:
    """
    Fetches all available game logs for a given player from the appropriate MySQL league table.
    """
    df = pd.DataFrame()

    table = f"""
        SELECT * FROM {league}_Player_Logs
        WHERE player_name = %s
        ORDER BY game_date ASC
    """
    cursor = connection.cursor()
    cursor.execute(table, (player_name,))
    rows = cursor.fetchall()

    #Selects Rows
    for i, row in enumerate(rows):
        new_row = dict(row)
        new_row_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_row_df], ignore_index=True)

    return df




###### SQLite Version ########
# def compute_player_distribution(player_name: str):
#     """
#     Computes the mean vector and covariance matrix for a single player's stats.
#     Returns a dictionary with mean and covariance.
#     """
#     df = fetch_player_game_logs(player_name)

#     if len(df) < 5:  # minimum games required (you can adjust)
#         print(f"⚠️ Not enough games to build distribution for {player_name}")
#         return None

#     means = df.mean().values
#     covariance = df.cov().values

#     return {
#         "player_name": player_name,
#         "mean_vector": means,
#         "covariance_matrix": covariance

#     }


# Returns all current active team members names that will play on the target date, using MySQL database, for the target team and league
def get_current_team_members(team_name: str, game_date: str, league: str):
    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    print("team: " + team_name)
    print("Date: " + game_date)
    print("leauge: " + league)

    cursor.execute("""
        SELECT DISTINCT player_name
        FROM NBA_Player_Logs
        WHERE team = %s
          AND game_date = %s
          AND league = %s
    """, (team_name, game_date, league))

    rows = cursor.fetchall()
    conn.close()

    team_members = [row["player_name"] for row in rows]
    print("Teammate Names: " + str(team_members))

    return team_members



####### MySQL Version  ########
def compute_player_distribution(player_name: str, game_date: str, league: str, connection) -> Optional[Dict]:
    """
    Computes the mean vector and covariance matrix for all active players on the same team as the target player.
    Uses renamed stat columns for player disambiguation.
    """

    # Step 1: Fetch target player's logs
    player_df = fetch_player_game_logs(player_name, league, connection)
    if len(player_df) < 5:
        print(f"⚠️ Not enough games to build distribution for {player_name}")
        return None

    # Step 2: Extract player's team
    game = player_df.iloc[0]
    team_name = game["team"]

    # Step 3: Get active teammates from team roster
    teammates = get_current_team_members(team_name, game_date, league)
    if player_name not in teammates:
        teammates.append(player_name)  # Ensure target player is included

    # Step 4: Collect game logs for each teammate
    all_logs = []
    for name in teammates:
        logs = fetch_player_game_logs(name, league, connection)
        if not logs.empty:
            all_logs.append(logs)

    if not all_logs:
        print("❌ No data found for any team members.")
        return None

    # Step 5: Analyze using multiplayer stat matrix builder
    full_df = pd.concat(all_logs, ignore_index=True)
    result = analyze_multiplayer_stats(full_df)

    return {
        "player_name": player_name,
        "mean_vector": result["mean_vector"],
        "covariance_matrix": result["covariance_matrix"],
        "renamed_df": result["renamed_df"]
    }






# def build_all_player_distributions():
#     """
#     Builds distributions for all players with enough data.
#     Returns a dictionary { player_name → {mean_vector, covariance_matrix} }
#     """
#     players = get_all_players()
#     player_models = {}

#     for player in players:
#         model = compute_player_distribution(player)
#         if model:
#             player_models[player] = model

#     print(f"✅ Built distributions for {len(player_models)} players.")
#     return player_models


def get_expert_stats(expert_name):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT total_predictions, correct_predictions, incorrect_predictions,
               true_positive, false_positive, true_negative, false_negative
        FROM expert_reliability
        WHERE expert_name = ?
    """, (expert_name,))
    
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "f1_score": 0.0,
            "total_predictions": 0
        }

    total, correct, incorrect, TP, FP, TN, FN = row

    # Avoid division by zero
    accuracy = correct / total if total else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    specificity = TN / (TN + FP) if (TN + FP) else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1_score, 4),
        "total_predictions": total
    }

def fetch_all_events():
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT e.event_id, c.crowd_probability, e.actual_result
        FROM events e
        JOIN crowd_predictions c ON e.event_id = c.event_id
        WHERE e.actual_result IS NOT NULL
    """)
    events = cursor.fetchall()

    event_data = []
    for event_id, crowd_prob, actual in events:
        cursor.execute("""
            SELECT expert_name, prediction, confidence, stat_threshold
            FROM expert_predictions
            WHERE event_id = %s
        """, (event_id,))
        experts = cursor.fetchall()
        event_data.append({
            "event_id": event_id,
            "crowd_prob": crowd_prob,
            "expert_predictions": experts,
            "actual_result": actual
        })

    conn.close()
    return event_data

def log_odds(p):
    return math.log(p / (1 - p))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def get_expert_confidence_calibration(db_path, bucket_size=0.05):
    """
    Returns a dictionary mapping (expert_name, confidence_bin) -> accuracy.
    This supports calibration scaling in Bayesian models.

    Args:
        db_path (str): Path to your SQLite database.
        bucket_size (float): Size of confidence buckets (default = 0.05 for 5%).

    Returns:
        Dict[Tuple[str, float], float]: Mapping from (expert, confidence bin) to empirical accuracy.
    """
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    SELECT ep.expert_name, ep.confidence, ev.actual_result, ep.prediction
    FROM expert_predictions ep
    JOIN events ev ON ep.event_id = ev.event_id
    WHERE ep.confidence IS NOT NULL AND ev.actual_result IS NOT NULL
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Normalize and sanitize
    df["confidence"] = df["confidence"].astype(float) / 100.0
    df = df.dropna(subset=["confidence", "actual_result", "prediction"])
    df["correct"] = (df["prediction"] == df["actual_result"]).astype(int)

    # Bin confidence
    num_buckets = int(1 / bucket_size)
    df["conf_bin"] = (df["confidence"] * num_buckets).astype(int) / num_buckets

    # Group and calculate calibration accuracy
    grouped = (
        df.groupby(["expert_name", "conf_bin"])["correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy_at_conf", "count": "num_samples"})
    )

    # Convert to lookup dictionary
    calibration_dict = {
        (row["expert_name"], round(row["conf_bin"], 2)): round(row["accuracy_at_conf"], 4)
        for _, row in grouped.iterrows()
    }

    return calibration_dict

def compute_bayesian_posterior(crowd_prob, expert_predictions, reliability, correlation_matrix):
    """
    Computes the posterior probability P(Y=1) using:
    - Crowd probability as prior
    - Expert predictions as evidence (with confidence)
    - Expert reliability (TPR, TNR)
    - Correlation matrix to down-weight redundant sources
    """
    logit = log_odds(crowd_prob)

    # Compute correlation-based weight scaling
    correlation_weights = adjust_weights_for_correlation(expert_predictions, correlation_matrix)

    for name, signal, conf in expert_predictions:
        conf = conf if conf is not None else 0.75

        alpha, beta = reliability.get(name, (0.5, 0.5))
        weight = 0

        if signal == 1:  # "Higher"
            weight = math.log((alpha * conf + 1e-6) / (1 - beta * conf + 1e-6))
        else:  # "Lower"
            weight = math.log((1 - alpha * conf + 1e-6) / (beta * conf + 1e-6))

        # Scale weight by redundancy factor
        scale = correlation_weights.get(name, 1.0)
        adjusted_weight = weight * scale

        logit += adjusted_weight

    return sigmoid(logit)

def compute_prediction_quality(
    crowd_prob: float,
    expert_predictions: List[Tuple[str, int, Optional[float]]],
    expert_reliability: Dict[str, float]
) -> float:
    num_experts = len(expert_predictions)
    if num_experts == 0:
        return 0.0

    # 1. Expert count score
    count_score = min(num_experts / 5, 1.0)

    # 2. Confidence spread
    confidences = [conf if conf is not None else 0.75 for _, _, conf in expert_predictions]
    spread = max(confidences) - min(confidences) if num_experts > 1 else 0
    spread_score = 1.0 - spread

    # 3. Agreement with crowd
    expert_avg = np.mean([pred for _, pred, _ in expert_predictions])
    agreement_score = 1 - abs(expert_avg - crowd_prob)

    # 4. Expert reliability score
    avg_reliability = np.mean([
        expert_reliability.get(name, 0.6) for name, _, _ in expert_predictions
    ])
    reliability_score = avg_reliability

    # Combine (weighted average)
    final_score = (
        0.25 * count_score +
        0.25 * spread_score +
        0.25 * agreement_score +
        0.25 * reliability_score
    )
    return round(final_score, 4)

def fetch_event_metadata(event_id: str, connection) -> Optional[Dict]:
    """
    Retrieves metadata for a given event ID from the events table.

    Returns a dictionary with:
        - player_name
        - league
        - game_date (as string YYYY-MM-DD)
        - stat_type
        - stat_threshold

    Returns None if the event is not found.
    """
    cursor = connection.cursor()

    query = """
        SELECT player_name, league, event_date, stat_type, stat_threshold
        FROM events
        WHERE event_id = %s
    """
    cursor.execute(query, (event_id,))
    row = cursor.fetchone()

    if not row:
        print(f"❌ No metadata found for event_id: {event_id}")
        return None

    return {
        "player_name": row[0],
        "league": row[1],
        "game_date": str(row[2]),
        "stat_type": row[3],
        "stat_threshold": row[4]
    }



def compute_expert_correlation_matrix():
    conn = connect()
    cursor = conn.cursor()

    # Get all expert predictions joined on event_id
    query = """
    SELECT event_id, expert_name, prediction
    FROM expert_predictions
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Pivot to event_id as rows, experts as columns
    pivot_df = df.pivot_table(index="event_id", columns="expert_name", values="prediction")
    pivot_df = pivot_df.dropna(thresh=2)  # Drop rows with fewer than 2 expert predictions

    # Compute pairwise correlation (NaN-safe)
    correlation_matrix = pivot_df.corr(method="pearson", min_periods=5).fillna(0)

    return correlation_matrix


def adjust_weights_for_correlation(expert_predictions, correlation_matrix):
    """
    expert_predictions: list of (name, prediction, confidence)
    correlation_matrix: DataFrame of expert pairwise correlations
    Returns: dict of expert_name -> adjusted_weight_factor (0.0 to 1.0)
    """
    names = [e[0] for e in expert_predictions]
    adjusted_weights = {}

    for name in names:
        others = [n for n in names if n != name]
        if not others:
            adjusted_weights[name] = 1.0
            continue

        avg_corr = np.mean([correlation_matrix.loc[name, o] for o in others if name in correlation_matrix and o in correlation_matrix])
        adjusted_weights[name] = max(0.0, 1 - avg_corr)  # scale down weight by average correlation

    return adjusted_weights


def predict_event_by_id(event_id: str, logistic_model=None, logistic_expert_names=None, n_samples=1000):
    """
    Computes a Bayesian ensemble prediction for a single event using expert signals and crowd probability.

    This function:
    - Retrieves the crowd probability (prior) and expert predictions for the given event_id from the database.
    - Applies a Bayesian update rule, treating expert signals as evidence to adjust the crowd's prior probability.
    - Incorporates per-expert reliability (true positive rate, true negative rate) from the database.
    - Enhances accuracy by using expert-specific confidence calibration (how well experts perform at different confidence levels).
    - Includes support for down-weighting correlated sources to avoid double counting overlapping information.
    - Uses Bayesian simulation (Monte Carlo sampling) to estimate a posterior distribution of the event probability.
    - Returns a point estimate (mean posterior probability) along with a 95% credible interval.
    - Predicts binary outcome class (1 = likely to happen, 0 = likely not).

    Returns:
        A dictionary with:
            - event_id (str)
            - predicted_prob (float): Mean posterior probability P(Y=1)
            - predicted_class (int): 1 if predicted_prob ≥ 0.5, else 0
            - lower_bound (float): 2.5th percentile of simulated posterior (confidence interval)
            - upper_bound (float): 97.5th percentile of simulated posterior
        Or None if the event ID or predictions are not found.
    """
    reliability = get_expert_reliability()
    correlation_matrix = compute_expert_correlation_matrix() 

    conn = connect()
    cursor = conn.cursor()

    # Get crowd probability
    cursor.execute("""
        SELECT c.crowd_probability
        FROM crowd_predictions c
        WHERE c.event_id = %s
    """, (event_id,))
    row = cursor.fetchone()

    if not row:
        print(f"❌ Event ID '{event_id}' not found or missing crowd probability.")
        return None

    crowd_prob = row[0]

    # Get expert predictions
    cursor.execute("""
        SELECT expert_name, prediction, confidence
        FROM expert_predictions
        WHERE event_id = %s
    """, (event_id,))
    expert_predictions = cursor.fetchall()
    conn.close()

    if not expert_predictions:
        print(f"⚠️ No expert predictions found for event '{event_id}'.")
        return None
    

    # === Logistic Regression Prediction ===
    logistic_input_vec = np.zeros(len(logistic_expert_names))
    expert_index = {name: i for i, name in enumerate(logistic_expert_names)}
    for name, pred, _ in expert_predictions:
        if name in expert_index:
            logistic_input_vec[expert_index[name]] = 1 if pred == 1 else -1

    logistic_prob = round(logistic_model.predict_proba([logistic_input_vec])[0][1], 4)

    # --- Bootstrapping to estimate uncertainty ---
    from random import choices

    probs = []
    for _ in range(n_samples):
        sampled = choices(expert_predictions, k=len(expert_predictions))
        prob = compute_bayesian_posterior(crowd_prob, sampled, reliability, correlation_matrix=correlation_matrix)
        probs.append(prob)

    mean_prob = round(np.mean(probs), 4)
    std_dev = round(np.std(probs), 4)

    # --- Asessing Quality Of Prediction ---
    quality_score = compute_prediction_quality(
            crowd_prob,
            expert_predictions,
            {k: get_expert_accuracy(k) for k, _, _ in expert_predictions}
        )
    
    # === Combine using quality score as weight ===
    combined_prob = round(
        (quality_score * mean_prob + (1 - quality_score) * logistic_prob), 4
    )

    combined_class = 1 if combined_prob >= 0.5 else 0

    print(f"\nPrediction for Event ID: {event_id}")
    print(f" - Bayesian Model: {int(mean_prob * 100)}% ± {int(std_dev * 100)}%")
    print(" - Bayesian Quality_score: " + str(round(quality_score,2)))
    print(f" - Logistic Model: {int(logistic_prob * 100)}%")
    print(f" - Combined Prob: {int(combined_prob * 100)}% (weighted by quality: {round(quality_score, 2)})")
    print(f" - Final Prediction: {'YES (1)' if combined_class == 1 else 'NO (0)'}")

    return {
        "event_id": event_id,
        "bayesian_prob": mean_prob,
        "logistic_prob": logistic_prob,
        "combined_prob": combined_prob,
        "error_margin": std_dev,
        "quality_score": quality_score,
        "predicted_class": combined_class
    }

#(Old Version)
# def main_model(event_id):
#     training_data = fetch_all_events()
#     X, y, logistic_expert_names = prepare_training_matrix(training_data)
#     logistic_model = train_logistic_weights(X, y)
#     result = predict_event_by_id(event_id, logistic_model, logistic_expert_names)

#     return result

def main_model(event_id):
    print(f"🔍 Starting model pipeline for Event ID: {event_id}")

    # === Step 1: Train Logistic Model ===
    print("📥 Fetching training data...")
    training_data = fetch_all_events()
    print(f"✅ Total training events fetched: {len(training_data)}")

    # Check label diversity
    labels = [e["actual_result"] for e in training_data]
    label_dist = dict(zip(*np.unique(labels, return_counts=True)))
    print("📊 Label distribution:", label_dist)

    print("🔧 Preparing training matrix for logistic regression...")
    X, y, logistic_expert_names = prepare_training_matrix(training_data)

    print("🤖 Training logistic regression model...")
    logistic_model = train_logistic_weights(X, y)
    print("✅ Logistic model trained.")

    # === Step 2: Predict Bayesian + Logistic + Combined ===
    print("🔮 Generating ensemble prediction...")
    result = predict_event_by_id(event_id, logistic_model, logistic_expert_names)
    if result is None:
        print("❌ Prediction failed.")
        return {"error": "Prediction failed. Check logs for more info."}
    print("✅ Prediction complete.")

    # === Step 3: Get Event Metadata for KDE model ===
    print("📦 Fetching event metadata for KDE modeling...")
    connection = connect()
    event_meta = fetch_event_metadata(event_id, connection)
    if not event_meta:
        print("❌ Event metadata not found.")
        return {"error": "Event metadata not found."}

    player_name = event_meta["player_name"]
    league = event_meta["league"]
    game_date = event_meta["game_date"]
    stat_type = event_meta["stat_type"].lower()
    stat_threshold = event_meta["stat_threshold"]

    print(f"📅 Event Date: {game_date}, League: {league}")
    print(f"🎯 Target Player: {player_name}, Stat: {stat_type}, Threshold: {stat_threshold}")

    #compute_player_distribution(player_name, game_date, league, connection)

    # === Step 4: Compute Team KDE CDF with Mean-Shift ===
    print("📈 Computing team KDE CDF with mean-shift adjustment...")
    connection =  get_dict_connection()
    cdf_result = compute_team_cdf_with_kde_shift(
        player_name=player_name,
        stat_type=stat_type,
        target_date=game_date,
        league=league,
        connection=connection
    )
    connection.close()

    if cdf_result:
        result["team_kde_prob"] = cdf_result(stat_threshold)
        result["team_kde_confidence_bounds"] = []

        # Define the x values for which we will evaluate the PDF
        # TODO: make custom reasonable bounds for each stat type
        x_values = np.linspace(0, 70, 140)
        print(x_values)
        top_95_confidence_interval = []

        for x in x_values:

            if cdf_result(x) > 0.70:
                top_95_confidence_interval.append(x)

        result["team_kde_confidence_bounds"].append(round(min(top_95_confidence_interval),1))
        result["team_kde_confidence_bounds"].append(max(top_95_confidence_interval))

        print("✅ KDE probability interpolation successful.")
    else:
        result["team_kde_prob"] = None
        result["team_kde_confidence_bounds"] = None
        print("⚠️ KDE probability interpolation failed.")

    # === Step 5: Display Final Result ===
    print("\n📊 Final Combined Prediction Result:")
    for k, v in result.items():
        print(f" - {k}: {v}")

    return result

# Example run
main_model("2024-10-22-NBA-JBRUNSON21.5PTS")


# if __name__ == "__main__":
#     training_data = fetch_all_events()
#     X, y, logistic_expert_names = prepare_training_matrix(training_data)
#     logistic_model = train_logistic_weights(X, y)
#     result = predict_event_by_id("2025-03-25-NBA-AMENTHOMPSON8Rebounds")