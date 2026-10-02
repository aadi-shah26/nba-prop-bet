"""
Odds math shared by props and game lines.

EV here is real expected profit per $1 staked, accounting for pushes:
    EV = P(win) * (decimal_odds - 1) - P(loss)        (a push returns the stake: 0)
The market's "fair" probability removes the bookmaker's margin (vig) from a two-way
market by normalizing the two implied probabilities to sum to 1.
"""


def american_to_decimal(odds):
    odds = float(odds)
    if odds == 0 or -100 < odds < 100:
        raise ValueError(f"Invalid American odds: {odds}")
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))


def implied_prob(odds):
    """Raw implied probability (includes the vig)."""
    return 1 / american_to_decimal(odds)


def no_vig_probs(odds_a, odds_b):
    """Fair (vig-free) probabilities of side A and side B, plus the bookmaker margin."""
    ia, ib = implied_prob(odds_a), implied_prob(odds_b)
    total = ia + ib
    return ia / total, ib / total, total - 1


def expected_value(p_win, p_loss, odds):
    """Expected profit per $1 stake. p_win + p_loss may be < 1 (the rest is push)."""
    return p_win * (american_to_decimal(odds) - 1) - p_loss


def kelly_fraction(p_win, p_loss, odds):
    """Full-Kelly stake fraction with pushes: maximizes p*ln(1+b f) + q*ln(1-f).
    Returns 0 for non-positive edges."""
    b = american_to_decimal(odds) - 1
    if p_win + p_loss <= 0:
        return 0.0
    f = (b * p_win - p_loss) / (b * (p_win + p_loss))
    return max(f, 0.0)


def break_even_prob(odds):
    """Win probability needed to break even (no pushes)."""
    return implied_prob(odds)


def decide(p_a, p_b, odds_a, odds_b, min_ev):
    """
    Compare both sides of a two-way market.
    Returns (pick, ev_a, ev_b) where pick is 'A', 'B' or 'PASS'.
    """
    ev_a = expected_value(p_a, p_b, odds_a)
    ev_b = expected_value(p_b, p_a, odds_b)
    if max(ev_a, ev_b) < min_ev:
        return 'PASS', ev_a, ev_b
    return ('A' if ev_a >= ev_b else 'B'), ev_a, ev_b
