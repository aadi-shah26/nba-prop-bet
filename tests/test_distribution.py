import pytest

from projection_model import count_distribution, fair_line, line_probs


@pytest.mark.parametrize('mean,var', [(25.3, 60.0), (4.2, 6.5), (1.1, 1.0), (0.4, 0.5)])
@pytest.mark.parametrize('line', [0.5, 3.5, 4, 25, 25.5])
def test_probs_sum_to_one(mean, var, line):
    o, u, p = line_probs(mean, var, line)
    assert o + u + p == pytest.approx(1, abs=1e-9)
    assert min(o, u, p) >= 0
    if not float(line).is_integer():
        assert p == 0


def test_moments():
    d = count_distribution(25.0, 60.0)
    assert d.mean() == pytest.approx(25.0)
    assert d.var() == pytest.approx(60.0)
    assert count_distribution(3.0, 2.5).var() == pytest.approx(3.0)  # Poisson floor


def test_over_definition():
    # P(X > 24.5) == P(X >= 25) and integer line push == pmf
    d = count_distribution(25.0, 60.0)
    o, u, p = line_probs(25.0, 60.0, 24.5)
    assert o == pytest.approx(1 - d.cdf(24))
    o, u, p = line_probs(25.0, 60.0, 25)
    assert p == pytest.approx(d.pmf(25))
    assert u == pytest.approx(d.cdf(24))


def test_fair_line_is_near_even():
    for mean, var in [(25.3, 60.0), (6.7, 9.0), (1.4, 1.9)]:
        L = fair_line(mean, var)
        assert L % 1 == 0.5
        o = line_probs(mean, var, L)[0]
        # no x.5 line is closer to 50/50
        for other in (L - 1, L + 1):
            if other > 0:
                assert abs(o - 0.5) <= abs(line_probs(mean, var, other)[0] - 0.5) + 1e-12
