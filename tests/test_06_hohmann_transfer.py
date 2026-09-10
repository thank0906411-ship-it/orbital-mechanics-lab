"""06_Hohmann_transfer.py 검증: LEO->GEO 델타-V가 교과서 값과 일치하는지, 전이궤도
공식(반장축/이심률)이 정확한지, 그리고 01번 실제 궤도 전파로 전이궤도를 따라가면
계산된 전이 시간에 정확히 목표 반지름(원지점)에 도달하는지."""

import pytest
from helpers import load_module

m = load_module("missions/06_Hohmann_transfer.py")


def test_leo_to_geo_total_delta_v_matches_textbook_value():
  _dv1, _dv2, total = m.hohmann_transfer_delta_v(6678.0, 42164.0)
  assert 3.8 < total < 4.0


def test_leo_to_geo_transfer_time_matches_textbook_value():
  transfer_time = m.hohmann_transfer_time(6678.0, 42164.0)
  assert 5.0 < transfer_time / 3600 < 5.5


def test_same_radius_transfer_requires_zero_delta_v():
  """출발과 목표 반지름이 같으면 궤도 변경이 필요 없으므로 델타-V는 0이어야 한다."""
  dv1, dv2, total = m.hohmann_transfer_delta_v(7000.0, 7000.0)
  assert dv1 == pytest.approx(0.0, abs=1e-9)
  assert dv2 == pytest.approx(0.0, abs=1e-9)
  assert total == pytest.approx(0.0, abs=1e-9)


def test_transfer_orbit_semi_major_axis_is_average_of_radii():
  a_t, _e_t = m.hohmann_transfer_orbit_elements(7000.0, 15000.0)
  assert a_t == pytest.approx((7000.0 + 15000.0) / 2)


def test_transfer_orbit_eccentricity_in_valid_range():
  _a_t, e_t = m.hohmann_transfer_orbit_elements(7000.0, 15000.0)
  assert 0.0 < e_t < 1.0


def test_transfer_orbit_perigee_and_apogee_match_input_radii():
  """전이궤도의 근지점 r=a(1-e), 원지점 r=a(1+e)가 각각 r1, r2와 일치해야 한다."""
  r1, r2 = 7000.0, 15000.0
  a_t, e_t = m.hohmann_transfer_orbit_elements(r1, r2)
  perigee = a_t * (1 - e_t)
  apogee = a_t * (1 + e_t)
  assert perigee == pytest.approx(r1)
  assert apogee == pytest.approx(r2)


def test_propagated_transfer_orbit_reaches_target_radius_exactly():
  """이 스크립트의 핵심 주장: 전이 시간(pi*sqrt(a_t^3/mu)) 후 01번 궤도 전파로
  계산한 위치가 정확히 목표 반지름(원지점)과 일치해야 한다."""
  result = m.demo_propagated_transfer_reaches_target_radius()
  assert result["start_error_km"] < 1e-6
  assert result["end_error_km"] < 1e-6


def test_delta_v_increases_with_larger_radius_ratio_at_first():
  """비율이 작을 때는 델타-V가 명확히 증가해야 한다(비율이 매우 커지면 증가가
  둔화되므로, 작은 비율 구간에서만 단조 증가를 확인한다)."""
  _dv1_a, _dv2_a, total_a = m.hohmann_transfer_delta_v(7000.0, 7000.0 * 1.5)
  _dv1_b, _dv2_b, total_b = m.hohmann_transfer_delta_v(7000.0, 7000.0 * 4.0)
  assert total_b > total_a


def test_hohmann_delta_v_positive_for_outward_transfer():
  dv1, dv2, total = m.hohmann_transfer_delta_v(7000.0, 15000.0)
  assert dv1 > 0
  assert dv2 > 0
  assert total == pytest.approx(dv1 + dv2)
