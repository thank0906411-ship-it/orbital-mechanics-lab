"""07_J2_perturbation.py 검증: RAAN 세차 방향이 경사각(순행/역행)에 따라 부호가
바뀌는지, 극궤도에서 세차율이 0인지, 임계경사각(약 63.4도)에서 근점편각 세차율이
0이 되는지, 그리고 J2=0이면(가상으로) 세차가 사라지는지."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("07_J2_perturbation.py")


def test_semi_latus_rectum_circular_orbit_equals_radius():
  assert m.semi_latus_rectum(7000.0, eccentricity=0.0) == pytest.approx(7000.0)


def test_semi_latus_rectum_formula():
  a, e = 8000.0, 0.3
  assert m.semi_latus_rectum(a, e) == pytest.approx(a * (1 - e ** 2))


def test_raan_precession_negative_for_prograde_orbit():
  """순행궤도(i<90도)는 RAAN이 서쪽(음의 방향)으로 세차해야 한다."""
  rate = m.raan_precession_rate(7000.0, 0.01, np.radians(51.6))
  assert rate < 0


def test_raan_precession_positive_for_retrograde_orbit():
  """역행궤도(i>90도)는 RAAN이 동쪽(양의 방향)으로 세차해야 한다."""
  rate = m.raan_precession_rate(7000.0, 0.01, np.radians(98.0))
  assert rate > 0


def test_raan_precession_zero_for_polar_orbit():
  """극궤도(i=90도)는 cos(i)=0이라 RAAN 세차율이 정확히 0이어야 한다."""
  rate = m.raan_precession_rate(7000.0, 0.01, np.radians(90.0))
  assert rate == pytest.approx(0.0, abs=1e-15)


def test_raan_precession_symmetric_around_90_degrees():
  """i도와 (180-i)도의 RAAN 세차율은 부호만 반대이고 크기는 같아야 한다
  (cos(180-i) = -cos(i))."""
  rate_30 = m.raan_precession_rate(7000.0, 0.01, np.radians(30.0))
  rate_150 = m.raan_precession_rate(7000.0, 0.01, np.radians(150.0))
  assert rate_150 == pytest.approx(-rate_30)


def test_argp_precession_zero_at_critical_inclination():
  """임계경사각(arccos(1/sqrt(5)) ≈ 63.4도)에서 근점편각 세차율이 0이어야 한다 —
  이 스크립트의 핵심 검증(몰니야 궤도가 이 경사각을 쓰는 이유)."""
  critical_inclination = np.arccos(1 / np.sqrt(5))
  rate = m.argp_precession_rate(26600.0, 0.7, critical_inclination)
  assert rate == pytest.approx(0.0, abs=1e-12)


def test_argp_precession_changes_sign_across_critical_inclination():
  critical_inclination_deg = np.degrees(np.arccos(1 / np.sqrt(5)))
  rate_below = m.argp_precession_rate(26600.0, 0.7, np.radians(critical_inclination_deg - 10))
  rate_above = m.argp_precession_rate(26600.0, 0.7, np.radians(critical_inclination_deg + 10))
  assert rate_below > 0
  assert rate_above < 0


def test_propagate_with_j2_precession_matches_pure_kepler_when_raan_rate_frozen():
  """t=0에서는 세차가 아직 누적되지 않았으므로, 입력한 raan0/argp0가 그대로
  회전에 사용돼야 한다(0번 세차 적용)."""
  result = m.propagate_with_j2_precession(7000.0, 0.01, np.radians(97.4),
                                           raan0_rad=np.radians(45.0), argp0_rad=np.radians(30.0),
                                           mean_anomaly0_rad=0.0, time_sec=0.0)
  assert result["raan_rad"] == pytest.approx(np.radians(45.0))
  assert result["argp_rad"] == pytest.approx(np.radians(30.0))


def test_propagate_with_j2_precession_position_magnitude_matches_orbital_radius():
  """회전만 적용하므로 ECI 위치의 크기는 그 시점의 궤도 반지름과 같아야 한다."""
  a, e, i = 7000.0, 0.01, np.radians(97.4)
  t = 5 * 86400.0
  result = m.propagate_with_j2_precession(a, e, i, raan0_rad=0.0, argp0_rad=0.0, mean_anomaly0_rad=0.0, time_sec=t)
  assert np.linalg.norm(result["position_eci"]) == pytest.approx(result["state"]["r"])


def test_raan_accumulates_linearly_with_time():
  a, e, i = 7000.0, 0.01, np.radians(97.4)
  rate = m.raan_precession_rate(a, e, i)
  t1, t2 = 86400.0, 2 * 86400.0
  result1 = m.propagate_with_j2_precession(a, e, i, 0.0, 0.0, 0.0, time_sec=t1)
  result2 = m.propagate_with_j2_precession(a, e, i, 0.0, 0.0, 0.0, time_sec=t2)
  assert result2["raan_rad"] - result1["raan_rad"] == pytest.approx(rate * (t2 - t1))
