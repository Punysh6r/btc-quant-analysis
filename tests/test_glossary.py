"""Glossary bands must match the logic they document."""

from glossary import (
    drawdown_zone,
    f1_reading,
    momentum_zone,
    regime_label,
    rsi_zone,
    signal_reading,
    trend_zone,
)


def test_rsi_example_from_user_request():
    label, tone, _ = rsi_zone(46.9)
    assert (label, tone) == ("Healthy pullback", "good")


def test_rsi_full_scale():
    assert rsi_zone(20)[1] == "neutral"  # oversold: cheap but unconfirmed
    assert rsi_zone(60)[0] == "Bullish"
    assert rsi_zone(75)[1] == "caution"
    assert rsi_zone(85) == ("Overbought", "bad", rsi_zone(85)[2])


def test_trend_momentum_drawdown_bands():
    assert trend_zone(110, 100)[0] == "Strong uptrend"
    assert trend_zone(90, 100)[1] == "bad"
    assert momentum_zone(100, 100)[0] == "Riding the average"
    assert drawdown_zone(96.5, 100)[0] == "Normal digestion"
    assert drawdown_zone(70, 100)[1] == "bad"


def test_regime_signal_f1_match_app_rules():
    assert regime_label(80)[0] == "Short-term bullrun"
    assert regime_label(50)[0] == "Sideways / undecided"
    assert regime_label(20)[1] == "bad"
    assert signal_reading(1)[1] == "good" and signal_reading(-1)[1] == "bad"
    assert f1_reading(0.2) == ("No signal", "bad")
    assert f1_reading(0.75)[1] == "good"
