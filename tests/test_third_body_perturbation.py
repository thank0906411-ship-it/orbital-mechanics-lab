"""third_body_perturbation.py 검증: 상대 3체 섭동 공식이 원점에서 정확히
0이 되는지(항등식), 달의 섭동이 태양보다 큰지(교과서값), 태양/달 원궤도
근사의 반지름이 일정한지, 달의 공전 주기가 항성월과 일치하는지, LEO에서는
J2가 압도적이고 고도가 높아질수록 3체 섭동 비율이 커지는지, 데모 함수
자체가 핵심 주장을 재검증하는지 확인."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("perturbations/third_body_perturbation.py")


def test_third_body_acceleration_zero_at_origin():
  """위성이 지구 중심(원점)에 있으면 두 항이 정확히 상쇄돼 가속도가 0이어야
  한다 - 공식의 부호/항 순서를 검증하는 항등식."""
  body_position = np.array([384400.0, 0.0, 0.0])
  accel = m.third_body_acceleration(np.zeros(3), body_position, m.MOON_MU_KM3_S2)
  assert np.allclose(accel, 0.0, atol=1e-20)


def test_lunar_acceleration_exceeds_solar_at_leo():
  """LEO에서 달의 섭동 가속도가 태양보다 커야 한다(교과서값 약 2.2배)."""
  r_km = m.EARTH_RADIUS_KM + 550.0
  position_km = np.array([r_km, 0.0, 0.0])
  _total, sun_accel, moon_accel = m.lunisolar_acceleration(position_km, time_sec=0.0)
  sun_mag = np.linalg.norm(sun_accel)
  moon_mag = np.linalg.norm(moon_accel)
  assert moon_mag > sun_mag
  assert moon_mag / sun_mag == pytest.approx(2.238, rel=0.5)


def test_moon_position_orbit_radius_constant():
  """원궤도 근사이므로 모든 시각에서 |달 위치|가 MOON_ORBIT_RADIUS_KM과
  같아야 한다."""
  for t in [0.0, 1e5, 1e6, 2.36e6]:
    pos = m.moon_position_eci_km(t)
    assert np.linalg.norm(pos) == pytest.approx(m.MOON_ORBIT_RADIUS_KM, rel=1e-10)


def test_sun_position_orbit_radius_constant():
  """원궤도 근사이므로 모든 시각에서 |태양 위치|가 EARTH_ORBIT_RADIUS_KM과
  같아야 한다."""
  for t in [0.0, 1e6, 1e7, 3e7]:
    pos = m.sun_position_eci_km(t)
    assert np.linalg.norm(pos) == pytest.approx(m.EARTH_ORBIT_RADIUS_KM, rel=1e-10)


def test_moon_period_matches_sidereal_month():
  """SIDEREAL_MONTH_DAYS*86400초 후 달 위치가 거의 원위치로 돌아와야 한다
  (한 바퀴)."""
  period_sec = m.SIDEREAL_MONTH_DAYS * 86400
  pos_start = m.moon_position_eci_km(0.0)
  pos_after_period = m.moon_position_eci_km(period_sec)
  assert np.allclose(pos_start, pos_after_period, atol=1e-6)


def test_j2_dominates_at_leo_altitude():
  """LEO(300km)에서는 J2 가속도가 3체 섭동보다 압도적으로 커야 한다(비율
  1% 미만)."""
  result = m.altitude_regime_comparison(300.0)
  assert result["ratio_lunisolar_to_j2"] < 0.01


def test_third_body_ratio_to_j2_increases_with_altitude():
  """altitude_regime_comparison을 LEO와 GEO에서 호출해, GEO에서의 비율이
  LEO보다 훨씬 커야 함을 확인한다(핵심 주장: 고도가 높을수록 3체 섭동의
  상대적 중요성이 커짐)."""
  leo = m.altitude_regime_comparison(300.0)
  geo = m.altitude_regime_comparison(35786.0)
  assert geo["ratio_lunisolar_to_j2"] > leo["ratio_lunisolar_to_j2"] * 100


def test_j2_acceleration_sane_relative_to_central_gravity():
  """7000km에서 J2 가속도 / 중심중력 비율이 J2 계수(~1.08e-3) 자릿수와
  일치해야 한다(공식 자체의 상식적 크기 검증)."""
  r_km = 7000.0
  j2_accel = m.j2_acceleration_magnitude(r_km)
  central_accel = m.EARTH_MU_KM3_S2 / r_km ** 2
  ratio = j2_accel / central_accel
  assert 1e-4 < ratio < 1e-2


def test_demo_lunar_perturbation_exceeds_solar_runs_and_asserts():
  """demo_* 함수를 직접 호출해 재도출 없이 핵심 주장(달 > 태양)이 성립하는지
  확인한다."""
  result = m.demo_lunar_perturbation_exceeds_solar_despite_smaller_mass()
  assert result["moon_accel_km_s2"] > result["sun_accel_km_s2"]
  assert 1.5 < result["ratio"] < 3.0


def test_demo_crossover_rows_are_monotonically_increasing():
  """demo_j2_vs_third_body_crossover_across_altitude가 반환하는 비율이
  고도에 따라 단조증가하는지 재확인한다."""
  rows = m.demo_j2_vs_third_body_crossover_across_altitude()
  ratios = [row["ratio_lunisolar_to_j2"] for row in rows]
  assert all(ratios[i + 1] >= ratios[i] for i in range(len(ratios) - 1))
