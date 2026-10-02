"""solar_radiation_pressure.py 검증: SRP 가속도가 태양 반대방향을 향하는지
(부호 항등식), 그림자 판정이 두 조건을 모두 요구하는지, 그림자 안에서
정확히 0이 되는지, 면적대질량비에 선형 비례하는지, 저고도는 항력이 고고도는
SRP가 지배하는지, 데모 함수 자체가 핵심 주장을 재검증하는지 확인."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("perturbations/solar_radiation_pressure.py")


def test_srp_acceleration_points_away_from_sun():
  """SRP 가속도는 태양->위성 방향을 향해야 한다 - 부호가 반대면 태양 쪽으로
  끌리는 물리적 오류가 됨."""
  sun_position = np.array([m.third_body_module.AU_KM, 0.0, 0.0])
  sat_position = np.array([m.EARTH_RADIUS_KM + 35786.0, 1000.0, 0.0])
  accel = m.srp_acceleration(sat_position, sun_position, 1.5, 0.02)
  direction_sun_to_sat = sat_position - sun_position
  direction_sun_to_sat /= np.linalg.norm(direction_sun_to_sat)
  assert np.dot(accel, direction_sun_to_sat) > 0


def test_srp_magnitude_matches_precomputed_value():
  """Cr=1.5, A/m=0.02m^2/kg 기본값에서 SRP 가속도 크기가 사전 계산값
  (1.362e-10 km/s^2) 근처여야 한다."""
  sun_position = np.array([m.third_body_module.AU_KM, 0.0, 0.0])
  sat_position = np.array([7000.0, 0.0, 0.0])
  accel = m.srp_acceleration(sat_position, sun_position, 1.5, 0.02)
  assert np.linalg.norm(accel) == pytest.approx(1.362e-10, rel=0.01)


def test_shadow_requires_both_conditions():
  """그림자 판정은 '태양 반대쪽'과 '원통 반지름 이내' 두 조건을 모두
  요구해야 한다 - 한쪽만 만족하면 그림자가 아니다."""
  sun_position = np.array([m.third_body_module.AU_KM, 0.0, 0.0])
  behind_but_far = np.array([-50000.0, 50000.0, 0.0])  # 태양 반대쪽이지만 원통 밖
  assert not m.is_in_shadow(behind_but_far, sun_position)
  in_front_and_aligned = np.array([m.EARTH_RADIUS_KM + 500.0, 0.0, 0.0])  # 일광 쪽, 축 정렬
  assert not m.is_in_shadow(in_front_and_aligned, sun_position)
  deep_shadow = np.array([-(m.EARTH_RADIUS_KM + 500.0), 0.0, 0.0])
  assert m.is_in_shadow(deep_shadow, sun_position)


def test_srp_acceleration_with_shadow_is_exactly_zero_in_shadow():
  """그림자 안에서는 srp_acceleration_with_shadow가 정확히 0 벡터를 반환해야
  한다."""
  sun_position = np.array([m.third_body_module.AU_KM, 0.0, 0.0])
  deep_shadow = np.array([-(m.EARTH_RADIUS_KM + 500.0), 0.0, 0.0])
  accel = m.srp_acceleration_with_shadow(deep_shadow, sun_position, 1.5, 0.02)
  assert np.allclose(accel, 0.0, atol=1e-20)


def test_srp_scales_linearly_with_area_to_mass_ratio():
  """SRP 가속도는 면적대질량비에 정확히 선형 비례해야 한다(공식 자체가
  A/m에 선형)."""
  sun_position = np.array([m.third_body_module.AU_KM, 0.0, 0.0])
  sat_position = np.array([7000.0, 0.0, 0.0])
  a1 = np.linalg.norm(m.srp_acceleration(sat_position, sun_position, 1.5, 0.01))
  a2 = np.linalg.norm(m.srp_acceleration(sat_position, sun_position, 1.5, 0.02))
  assert a2 / a1 == pytest.approx(2.0, rel=1e-10)


def test_drag_dominates_at_low_altitude_srp_dominates_at_high_altitude():
  """200km에서는 항력이 SRP보다 훨씬 커야 하고, 3000km에서는 SRP가 항력보다
  훨씬 커야 한다(9번과의 대비 핵심 주장)."""
  low = m.drag_srp_altitude_comparison(200.0, 1.5, 0.02)
  high = m.drag_srp_altitude_comparison(3000.0, 1.5, 0.02)
  assert low["ratio_srp_to_drag"] < 0.01
  assert high["ratio_srp_to_drag"] > 1e6


def test_demo_drag_srp_crossover_runs_and_asserts():
  """demo_* 함수를 직접 호출해 핵심 주장(저고도 항력 지배, 고고도 SRP 지배,
  교차점 존재)이 재확인되는지 확인한다."""
  rows = m.demo_drag_dominates_low_srp_dominates_high_altitude()
  ratios = [row["ratio_srp_to_drag"] for row in rows]
  assert ratios[0] < 0.01
  assert ratios[-1] > 1e6
  assert any(0.1 < r < 10 for r in ratios)


def test_demo_eclipse_produces_physically_sane_fraction():
  """demo_eclipse_zeroes_out_srp_during_shadow_passes를 직접 호출해 식
  비율이 실측 범위(0.30~0.45) 안에 있는지 재확인한다."""
  result = m.demo_eclipse_zeroes_out_srp_during_shadow_passes()
  assert 0.30 < result["eclipse_fraction"] < 0.45
