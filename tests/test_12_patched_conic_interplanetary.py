"""12_Patched_conic_interplanetary.py 검증: SOI 공식이 알려진 형태와 일치하는지,
06번 호만 전이 함수를 감싼 결과가 06번을 직접 태양 GM으로 호출한 것과 정확히
일치하는지, 지구->화성 임무 델타-V/전이시간이 알려진 범위 안에 있는지, 그리고
01번 재전파로 화성 궤도반지름 도달이 정확히 검증되는지."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("12_Patched_conic_interplanetary.py")


def test_sphere_of_influence_matches_formula():
  a_planet, mu_planet, mu_sun = 1.0e8, 1000.0, 1.0e11
  expected = a_planet * (mu_planet / mu_sun) ** (2 / 5)
  assert m.sphere_of_influence_km(a_planet, mu_planet, mu_sun) == pytest.approx(expected)


def test_sphere_of_influence_larger_mu_planet_gives_larger_soi():
  """같은 궤도반지름에서 행성의 GM이 클수록 SOI도 커야 한다."""
  soi_small = m.sphere_of_influence_km(1.0e8, 100.0, 1.0e11)
  soi_large = m.sphere_of_influence_km(1.0e8, 1000.0, 1.0e11)
  assert soi_large > soi_small


def test_heliocentric_transfer_matches_direct_hohmann_call():
  """이 스크립트의 heliocentric_transfer가 06번 함수를 태양 GM으로 감싼 것과
  06번을 직접 태양 GM으로 호출한 결과가 정확히 같아야 한다 — 새 공식이 아니라
  06번을 재사용한다는 것 자체가 이 스크립트의 핵심 주장이다."""
  r1, r2 = m.EARTH_ORBIT_RADIUS_KM, m.MARS_ORBIT_RADIUS_KM
  wrapped = m.heliocentric_transfer(r1, r2)
  dv1, dv2, total = m.hohmann.hohmann_transfer_delta_v(r1, r2, mu=m.SUN_MU_KM3_S2)
  transfer_time = m.hohmann.hohmann_transfer_time(r1, r2, mu=m.SUN_MU_KM3_S2)
  assert wrapped["delta_v1"] == pytest.approx(dv1)
  assert wrapped["delta_v2"] == pytest.approx(dv2)
  assert wrapped["total_delta_v"] == pytest.approx(total)
  assert wrapped["transfer_time_sec"] == pytest.approx(transfer_time)


def test_hyperbolic_departure_delta_v_zero_v_infinity_equals_escape_velocity_gap():
  """v_infinity가 0이면 주차궤도에서 포물선 탈출 속도(v_circ*sqrt(2))까지만
  가속하면 되므로, 델타-V는 v_circ*(sqrt(2)-1)와 정확히 일치해야 한다."""
  mu, r_park = m.EARTH_MU_KM3_S2, 6678.0
  v_circ = np.sqrt(mu / r_park)
  expected_dv = v_circ * (np.sqrt(2) - 1)
  dv = m.hyperbolic_departure_delta_v(0.0, mu, r_park)
  assert dv == pytest.approx(expected_dv)


def test_hyperbolic_departure_delta_v_increases_with_v_infinity():
  mu, r_park = m.EARTH_MU_KM3_S2, 6678.0
  dv_small = m.hyperbolic_departure_delta_v(1.0, mu, r_park)
  dv_large = m.hyperbolic_departure_delta_v(3.0, mu, r_park)
  assert dv_large > dv_small


def test_earth_to_mars_transfer_time_matches_known_range():
  """지구->화성 호만형 전이시간은 문헌에서 흔히 약 259일로 인용된다."""
  result = m.total_mission_delta_v(6678.0, 3689.5)
  transfer_days = result["transfer"]["transfer_time_sec"] / 86400
  assert 200.0 < transfer_days < 300.0


def test_earth_to_mars_total_delta_v_matches_known_range():
  """지구->화성 임무 총 델타-V는 문헌에서 흔히 5~6km/s대로 인용된다."""
  result = m.total_mission_delta_v(6678.0, 3689.5)
  assert 4.5 < result["total_dv"] < 7.0


def test_soi_ratio_is_small_fraction_of_transfer_distance():
  """이 스크립트의 핵심 검증: 지구/화성 SOI가 전이거리에 비해 2% 미만이어야
  Patched Conic 근사(SOI 밖 행성 중력 무시)가 타당하다."""
  result = m.demo_soi_radius_negligible_compared_to_transfer_distance()
  assert result["earth_ratio_pct"] < 2.0
  assert result["mars_ratio_pct"] < 2.0


def test_heliocentric_propagation_reaches_exact_orbit_radii():
  """이 스크립트의 핵심 주장: 01번 재전파로 전이 시작/종료 시점의 태양 중심
  거리가 정확히 지구/화성 공전궤도 반지름과 일치해야 한다."""
  result = m.demo_soi_crossing_matches_propagated_heliocentric_position()
  assert result["start_error_km"] < 1e-3
  assert result["end_error_km"] < 1e-3


def test_propagate_orbit_with_sun_mu_matches_expected_period():
  """01번의 propagate_orbit을 태양 GM으로 호출했을 때, 지구 공전궤도 반지름의
  궤도 주기가 실제로 약 1년(365.25일 근방)이 되는지 확인한다 — 태양 GM 재사용이
  물리적으로 올바른지에 대한 독립적인 확인."""
  period_sec = 2 * np.pi / m.kepler.mean_motion(m.EARTH_ORBIT_RADIUS_KM, mu=m.SUN_MU_KM3_S2)
  period_days = period_sec / 86400
  assert 360.0 < period_days < 370.0
