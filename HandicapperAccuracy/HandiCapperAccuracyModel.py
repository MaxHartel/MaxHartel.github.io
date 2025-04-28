import sqlite3
import math
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.linear_model import LogisticRegression
from scipy.special import expit  # sigmoid
from scipy.optimize import curve_fit

DB_PATH = "/Users/maxhartel/Desktop/Desktop - Max’s MacBook Pro/Project Parlay/Project-Parlay/Pick_Confidence"
STAT_COLUMNS = [
    "points", "rebounds", "assists", "personal_fouls",
    "threes_made", "blocks", "steals", "turnovers", "minutes",
    "pa", "pr", "ra", "pra", "fantasy_points"
]


def crowd_log_odds(p: float) -> float:
    """Convert crowd probability to log-odds."""
    p = max(min(p, 0.999), 0.001)  # avoid division by zero
    return math.log(p / (1 - p))

def connect():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

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
        for name, _, _ in event["expert_predictions"]:
            expert_set.add(name)
    experts = sorted(expert_set)
    expert_index = {name: i for i, name in enumerate(experts)}

    X = []
    y = []

    for event in events:
        row = [0] * len(experts)
        for name, prediction, _ in event["expert_predictions"]:
            signal = 1 if prediction == 1 else -1
            row[expert_index[name]] = signal
        X.append(row)
        y.append(event["actual_result"])

    return np.array(X), np.array(y), experts

def train_logistic_weights(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def get_expert_reliability():
    conn = connect()
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
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT correct_predictions, total_predictions
            FROM expert_reliability
            WHERE expert_name = ?
        """, (expert_name,))
        row = cursor.fetchone()
        conn.close()

        if row:
            correct, total = row
            if total > 0:
                return round(correct / total, 4)
        
        # Default fallback if no record or no data
        return 0.60

    except Exception as e:
        print(f"⚠️ Error fetching expert accuracy for {expert_name}: {e}")
        return 0.60  # Conservative fallback

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
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT player_name FROM player_stats")
    players = [row[0] for row in cursor.fetchall()]
    conn.close()
    return players

def fetch_player_game_logs(player_name: str):
    """
    Fetches all available game logs for a given player into a DataFrame.
    """
    conn = connect()
    query = f"""
        SELECT {', '.join(STAT_COLUMNS)}
        FROM player_stats
        WHERE player_name = ?
    """
    df = pd.read_sql_query(query, conn, params=(player_name,))
    conn.close()
    return df

def compute_player_distribution(player_name: str):
    """
    Computes the mean vector and covariance matrix for a single player's stats.
    Returns a dictionary with mean and covariance.
    """
    df = fetch_player_game_logs(player_name)

    if len(df) < 5:  # minimum games required (you can adjust)
        print(f"⚠️ Not enough games to build distribution for {player_name}")
        return None

    means = df.mean().values
    covariance = df.cov().values

    return {
        "player_name": player_name,
        "mean_vector": means,
        "covariance_matrix": covariance
    }

def build_all_player_distributions():
    """
    Builds distributions for all players with enough data.
    Returns a dictionary { player_name → {mean_vector, covariance_matrix} }
    """
    players = get_all_players()
    player_models = {}

    for player in players:
        model = compute_player_distribution(player)
        if model:
            player_models[player] = model

    print(f"✅ Built distributions for {len(player_models)} players.")
    return player_models


def get_expert_stats(expert_name):
    conn = sqlite3.connect(DB_PATH)
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
            WHERE event_id = ?
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
    conn = sqlite3.connect(db_path)
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
        WHERE c.event_id = ?
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
        WHERE event_id = ?
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
    predicted_class = 1 if mean_prob >= 0.5 else 0

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



# if __name__ == "__main__":
#     training_data = fetch_all_events()
#     X, y, logistic_expert_names = prepare_training_matrix(training_data)
#     logistic_model = train_logistic_weights(X, y)
#     result = predict_event_by_id("2025-03-25-NBA-AMENTHOMPSON8Rebounds")


def main_model(event_id):
    training_data = fetch_all_events()
    X, y, logistic_expert_names = prepare_training_matrix(training_data)
    logistic_model = train_logistic_weights(X, y)
    result = predict_event_by_id(event_id, logistic_model, logistic_expert_names)

    return result


#main_model("2025-03-25-NBA-AMENTHOMPSON8Rebounds")

