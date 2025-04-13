from HandicapperAccuracy.ConfidenceDB_Port import generate_event_id

class Pick:
    pID_counter = 0

    @classmethod
    def pickID(cls):
        cls.pID_counter += 1
        return cls.pID_counter

    def __init__(self, name,odds, confidence, mutual_exclusion_group, league, event_id, reusable=True, capital_limit=0):
        self.pID = Pick.pickID()
        self.gameID = mutual_exclusion_group
        self.name = name
        self.odds = float(odds)
        self.confidence = float(confidence)
        self.league = league
        self.reusable = reusable
        self.capital_limit = int(capital_limit)
        self.event_id = event_id

    def to_dict(self):
        return {
            "pID": self.pID,
            "gameID": self.gameID,
            "name": self.name,
            "odds": self.odds,
            "confidence": self.confidence,
            "league": self.league,
            "reusable": self.reusable,
            "capital_limit": self.capital_limit,
            "event_id": self.event_id,
        }

class BoostPromo:
    boost_counter = 1

    def __init__(self, boost_percentage, required_picks, same_sport=False):
        self.name = f"Boost {BoostPromo.boost_counter}"
        BoostPromo.boost_counter += 1
        self.boost_percentage = boost_percentage
        self.required_picks = required_picks
        self.same_sport = same_sport

class ProtectedPromo:
    protected_counter = 1

    def __init__(self, protected_amount, eligible_leagues):
        self.name = f"Protected {ProtectedPromo.protected_counter}"
        ProtectedPromo.protected_counter += 1
        self.protected_amount = protected_amount
        self.eligible_leagues = eligible_leagues



