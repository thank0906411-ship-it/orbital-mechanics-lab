"""two_body_numerical_integration.py 검증: RK4 한 스텝의 정확성, 수치적분이
01/02번의 해석해와 일치하는지, 스텝 크기를 절반으로 줄일 때 오차가 O(h^4)답게
줄어드는지, 그리고 큰 스텝에서 에너지 드리프트가 실제로 커지는지."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("propagation/two_body_numerical_integration.py")


def test_two_body_acceleration_points_toward_origin():
  """중심 인력이므로 가속도는 항상 위치벡터의 반대 방향(원점을 향함)이어야 한다."""
  position = np.array([7000.0, 0.0, 0.0])
  acc = m.two_body_acceleration(position)
  assert acc[0] < 0
  assert acc[1] == pytest.approx(0.0)
  assert acc[2] == pytest.approx(0.0)


def test_two_body_acceleration_magnitude_matches_inverse_square_law():
  r = 8000.0
  position = np.array([r, 0.0, 0.0])
  acc = m.two_body_acceleration(position)
  expected_magnitude = m.EARTH_MU_KM3_S2 / r ** 2
  assert np.linalg.norm(acc) == pytest.approx(expected_magnitude, rel=1e-9)


def test_rk4_step_conserves_energy_over_short_step():
  """짧은 스텝 하나만 밟아도 비에너지가 거의 그대로 유지돼야 한다."""
  position = np.array([7000.0, 0.0, 0.0])
  velocity = np.array([0.0, 7.5, 0.0])
  energy_before = np.linalg.norm(velocity) ** 2 / 2 - m.EARTH_MU_KM3_S2 / np.linalg.norm(position)
  new_pos, new_vel = m.rk4_step(position, velocity, dt_sec=1.0)
  energy_after = np.linalg.norm(new_vel) ** 2 / 2 - m.EARTH_MU_KM3_S2 / np.linalg.norm(new_pos)
  assert energy_after == pytest.approx(energy_before, abs=1e-6)


def test_integrate_two_body_returns_history_covering_full_duration():
  position = np.array([7000.0, 0.0, 0.0])
  velocity = np.array([0.0, 7.5, 0.0])
  history = m.integrate_two_body(position, velocity, total_time_sec=100.0, dt_sec=10.0)
  assert history[0][0] == pytest.approx(0.0)
  assert history[-1][0] == pytest.approx(100.0)


def test_integrate_two_body_handles_non_divisible_duration():
  """total_time이 dt_sec으로 나누어 떨어지지 않아도 정확히 total_time에서 끝나야 한다."""
  position = np.array([7000.0, 0.0, 0.0])
  velocity = np.array([0.0, 7.5, 0.0])
  history = m.integrate_two_body(position, velocity, total_time_sec=95.0, dt_sec=10.0)
  assert history[-1][0] == pytest.approx(95.0)


def test_rk4_matches_analytical_solution_within_tight_tolerance():
  rows = m.demo_rk4_matches_analytical_solution()
  for row in rows:
    assert row["position_error_km"] < 1.0


def test_step_size_halving_reduces_error_by_roughly_sixteen():
  """이 스크립트의 핵심 주장: RK4는 O(h^4)이므로 스텝을 절반으로 줄이면 오차가
  약 16배 줄어야 한다."""
  rows = m.demo_step_size_convergence_order()
  for i in range(1, len(rows)):
    ratio = rows[i - 1]["position_error_km"] / rows[i]["position_error_km"]
    assert 8.0 < ratio < 32.0, f"오차 감소율이 16배 근방이어야 하는데 {ratio:.2f}배가 나옴"


def test_larger_step_size_increases_energy_drift():
  rows = m.demo_energy_drift_with_large_steps()
  drifts = [r["drift"] for r in rows]
  assert drifts == sorted(drifts), "스텝 크기가 커질수록(리스트 순서대로) 드리프트도 커져야 함"


def test_circular_orbit_stays_at_constant_radius():
  """원궤도(e=0)는 수치적분 중에도 거리가 거의 일정하게 유지돼야 한다."""
  a = 7000.0
  velocity_magnitude = np.sqrt(m.EARTH_MU_KM3_S2 / a)  # 원궤도 속도
  position = np.array([a, 0.0, 0.0])
  velocity = np.array([0.0, velocity_magnitude, 0.0])
  history = m.integrate_two_body(position, velocity, total_time_sec=1000.0, dt_sec=10.0)
  radii = [np.linalg.norm(pos) for _t, pos, _v in history]
  assert max(radii) - min(radii) < 1e-3
