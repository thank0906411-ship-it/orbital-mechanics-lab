"""low_thrust_transfer.py 검증: 접선 추력 가속도가 속도 방향 단위벡터에 정확한
크기를 곱하는지, 나선 전이 후 반장축이 목표에 도달하고 이심률이 작게 유지되는지,
Edelbaum 근사와 수치적분 델타-V가 일치하는지, 저추력이 호만 전이보다 훨씬 오래
걸리는지, 추력이 커지면 전이 시간이 줄어드는지."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("missions/low_thrust_transfer.py")


def test_tangential_thrust_acceleration_direction_matches_velocity():
  """추력 가속도의 방향은 속도 방향과 정확히 같아야 한다(단위벡터가 같은 방향)."""
  velocity = np.array([3.0, 4.0, 0.0])  # |v| = 5
  accel = m.tangential_thrust_acceleration(velocity, 1e-5)
  velocity_unit = velocity / np.linalg.norm(velocity)
  accel_unit = accel / np.linalg.norm(accel)
  assert accel_unit == pytest.approx(velocity_unit)


def test_tangential_thrust_acceleration_magnitude_matches_input():
  velocity = np.array([3.0, 4.0, 0.0])
  thrust_accel = 2.5e-6
  accel = m.tangential_thrust_acceleration(velocity, thrust_accel)
  assert np.linalg.norm(accel) == pytest.approx(thrust_accel)


def test_circular_orbit_speed_matches_vis_viva_for_circular_case():
  """원궤도 속력은 vis-viva 방정식(2/r - 1/a, a=r)에서 유도된 sqrt(mu/r)와 같아야
  한다."""
  r = 7000.0
  expected = np.sqrt(m.EARTH_MU_KM3_S2 / r)
  assert m.circular_orbit_speed(r) == pytest.approx(expected)


def test_spiral_transfer_reaches_target_radius_with_low_eccentricity():
  """이 스크립트의 핵심 주장: 나선 전이 후 반장축이 목표 반지름 근처에 도달하고,
  이심률은 여전히 작게(원궤도 유지) 남아있어야 한다."""
  result = m.demo_spiral_reaches_target_radius()
  assert result["a_error_pct"] < 1.0
  assert result["achieved_e"] < 0.01


def test_edelbaum_approximation_matches_numerical_integration():
  """이 스크립트의 핵심 주장: 가속도 x 시간으로 계산한 델타-V가 Edelbaum 근사와
  작은 오차 안에서 일치해야 한다."""
  result = m.demo_edelbaum_approximation_matches_numerical_integration()
  assert result["relative_error_pct"] < 10.0


def test_edelbaum_delta_v_zero_for_same_radius():
  """출발과 목표 반지름이 같으면 델타-V가 필요 없어야 한다."""
  assert m.edelbaum_delta_v(7000.0, 7000.0) == pytest.approx(0.0, abs=1e-9)


def test_low_thrust_takes_much_longer_than_hohmann():
  """이 스크립트의 핵심 주장: 저추력 전이는 호만 전이보다 훨씬 오래 걸려야 한다."""
  result = m.demo_low_thrust_vs_hohmann_time_tradeoff()
  assert result["time_ratio"] > 5
  assert result["low_thrust_time_sec"] > result["hohmann_time_sec"]


def test_transfer_time_decreases_with_larger_thrust():
  """추력 가속도가 커질수록 목표 반지름 도달 시간이 줄어들어야 한다."""
  rows = m.demo_transfer_time_vs_thrust_magnitude()
  assert rows[-1]["transfer_time_hr"] < rows[0]["transfer_time_hr"]


def test_spiral_transfer_raises_when_thrust_too_small_for_max_steps():
  """추력이 극단적으로 작으면 max_steps 안에 목표 반지름에 도달하지 못해 명시적
  오류를 내야 한다(조용히 무한루프에 빠지거나 잘못된 결과를 반환하지 않음)."""
  with pytest.raises(RuntimeError):
    m.spiral_transfer(7000.0, 7100.0, 1e-12, 10.0, max_steps=100)
