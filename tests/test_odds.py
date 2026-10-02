import pytest

from odds import (american_to_decimal, decide, expected_value, implied_prob, kelly_fraction,
                  no_vig_probs)


def test_decimal_conversion():
    assert american_to_decimal(-110) == pytest.approx(1.909091, rel=1e-6)
    assert american_to_decimal(+150) == pytest.approx(2.5)
    assert american_to_decimal(-200) == pytest.approx(1.5)
    assert american_to_decimal(100) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        american_to_decimal(50)


def test_no_vig():
    a, b, vig = no_vig_probs(-110, -110)
    assert a == pytest.approx(0.5) and b == pytest.approx(0.5)
    assert vig == pytest.approx(2 * 110 / 210 - 1)
    a, b, _ = no_vig_probs(-150, +130)
    assert a + b == pytest.approx(1)
    assert a > implied_prob(+130)


def test_ev_and_push():
    # 55% at -110: 0.55*0.9091 - 0.45 = +5.0%
    assert expected_value(0.55, 0.45, -110) == pytest.approx(0.55 * 100 / 110 - 0.45)
    # break-even at -110 is 52.38%
    assert expected_value(110 / 210, 100 / 210, -110) == pytest.approx(0, abs=1e-12)
    # pushes refund the stake: 50% win, 40% loss, 10% push at +100 -> +10%
    assert expected_value(0.5, 0.4, 100) == pytest.approx(0.10)


def test_kelly():
    # even money, 55/45 -> 10%
    assert kelly_fraction(0.55, 0.45, 100) == pytest.approx(0.10)
    assert kelly_fraction(0.45, 0.55, 100) == 0.0
    # with pushes: maximizer of p ln(1+bf) + q ln(1-f)
    import numpy as np
    p, q, b = 0.5, 0.4, 1.0
    f = np.linspace(0, 0.5, 50001)
    best = f[np.argmax(p * np.log(1 + b * f) + q * np.log(1 - f))]
    assert kelly_fraction(p, q, 100) == pytest.approx(best, abs=1e-4)


def test_decide():
    assert decide(0.6, 0.4, -110, -110, 0.03)[0] == 'A'
    assert decide(0.4, 0.6, -110, -110, 0.03)[0] == 'B'
    assert decide(0.52, 0.48, -110, -110, 0.03)[0] == 'PASS'
