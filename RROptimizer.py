from math import comb
from scipy.stats import binom
from pprint import pprint

# ------------------------------
# Pick analysis helper function
# ------------------------------

def analyze_pick_list(pick_list):
    """Analyzes a list of pick dictionaries and returns their average stats."""
    if not pick_list:
        return {
            "average_confidence_score": 0.0,
            "average_odds": 0.0,
            "total_picks": 0
        }

    total_confidence_score = sum(getattr(p, "confidence", 0) for p in pick_list)
    total_odds = sum(getattr(p, "odds", 0) for p in pick_list)
    total_picks = len(pick_list)

    return {
        "average_confidence_score": round(total_confidence_score / total_picks, 2),
        "average_odds": round(total_odds / total_picks, 2),
        "total_picks": total_picks
    }

# ------------------------------
# Round Robin Optimizer helpers
# ------------------------------

def get_probabilities(t, p):
    """Returns a list of probabilities of hitting at least k correct picks out of t total."""
    return [sum(binom.pmf(i, t, p) for i in range(k, t + 1)) for k in range(2, t + 1)]

def round_robin_payouts(t, odds, max_size):
    """Generates expected payouts for all subset sizes from 2 to max_size."""
    results = []
    for n in range(2, max_size + 1):
        payouts_for_size = []
        for x in range(2, t + 1):
            if x < n:
                payout = 0.0
            else:
                payout = comb(x, n) * (odds ** n)
            payouts_for_size.append(round(payout, 2))
        results.append(payouts_for_size)
    return results

def calculate_dot_products(odds_list, payouts_lists):
    """Computes dot products of probabilities and payouts to estimate returns."""
    return [sum(o * p for o, p in zip(odds_list, payouts)) for payouts in payouts_lists]

def subtract_binom(values, t, up_to=False):
    """Subtracts expected binomial cost for each combination size."""
    result = []
    for i, value in enumerate(values):
        if up_to:
            binom_sum = sum(comb(t, k) for k in range(2, i + 3))
        else:
            binom_sum = comb(t, i + 2)
        result.append(value - binom_sum)
    return result

def roundRobinOptimizer(prob_success, total_picks, odds):
    """Returns the optimal subgroup size and its value (score) for a round robin strategy."""
    probabilities = get_probabilities(total_picks, prob_success)
    payouts = round_robin_payouts(total_picks, odds, total_picks)
    dot_products = calculate_dot_products(probabilities, payouts)
    dot_products = subtract_binom(dot_products, total_picks)

    highest_dot = max(dot_products)
    optimal_sub_group_size = dot_products.index(highest_dot) + 2  # subgroup sizes start from 2

    return optimal_sub_group_size, highest_dot

# ------------------------------
# Split Analyzer
# ------------------------------

def analyze_all_splits(pick_list):
    print(pick_list)
    if not pick_list or len(pick_list) < 2:
        return None, None

    sorted_picks = sorted(pick_list, key=lambda x: getattr(x, "confidence_score", 0), reverse=True)


    # Consider full list as one group
    full_stats = analyze_pick_list(sorted_picks)
    full_value = 0
    full_label = "N/A"

    if full_stats["total_picks"] >= 2:
        full_size, full_value = roundRobinOptimizer(
            full_stats["average_confidence_score"] / 100,
            full_stats["total_picks"],
            full_stats["average_odds"]
        )
        full_label = f"Size {full_size} only on Full List"

    best_value = full_value
    best_label = full_label

    # Try all splits
    for i in range(1, len(sorted_picks)):
        left = sorted_picks[:i]
        right = sorted_picks[i:]

        left_stats = analyze_pick_list(left)
        right_stats = analyze_pick_list(right)

        left_value, right_value = 0, 0
        left_desc, right_desc = "N/A", "N/A"

        if left_stats["total_picks"] >= 2:
            left_size, left_value = roundRobinOptimizer(
                left_stats["average_confidence_score"] / 100,
                left_stats["total_picks"],
                left_stats["average_odds"]
            )
            left_desc = f"Size {left_size} only on Left"

        if right_stats["total_picks"] >= 2:
            right_size, right_value = roundRobinOptimizer(
                right_stats["average_confidence_score"] / 100,
                right_stats["total_picks"],
                right_stats["average_odds"]
            )
            right_desc = f"Size {right_size} only on Right"

        total_split_value = round(left_value + right_value, 4)

        # 👇 Compare both sides independently
        if left_value > total_split_value and left_value > best_value:
            best_value = round(left_value, 4)
            best_label = f"Only bet on Left side → {left_desc}, split at index {i}"
        elif right_value > total_split_value and right_value > best_value:
            best_value = round(right_value, 4)
            best_label = f"Only bet on Right side → {right_desc}, split at index {i}"
        elif total_split_value > best_value:
            best_value = total_split_value
            best_label = f"{left_desc}, {right_desc}, split at index {i}"

    return best_value, best_label
