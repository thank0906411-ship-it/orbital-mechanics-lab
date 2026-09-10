"""13_Clohessy_wiltshire_rendezvous.py 검증: 상태천이행렬이 알려진 특수해(z축
단순조화진동 주기, 궤도 주기마다 초기값으로 복귀)를 재현하는지, 제로 드리프트
조건이 실제로 발산을 막는지, 랑데부 델타-V 역산이 정확히 원점에 도달하는지,
그리고 CW 근사가 초기 옵셋이 커질수록 실제 비선형 전파와 벗어나는지."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("missions/13_Clohessy_wiltshire_rendezvous.py")


def test_z_axis_returns_to_initial_value_after_one_period():
  """z축은 x,y와 분리된 단순조화진동이므로, 목표 궤도 주기 후 정확히 초기값으로
  복귀해야 한다."""
  a = 6978.0
  n = m.kepler.mean_motion(a)
  period = 2 * np.pi / n
  z0 = 2.0
  state = m.cw_state_transition(0.0, 0.0, z0, 0.0, 0.0, 0.0, n, period)
  assert state["z"] == pytest.approx(z0, abs=1e-6)


def test_z_axis_independent_of_x_and_y():
  """z축 운동은 x0,y0,vx0,vy0에 전혀 영향받지 않아야 한다(완전히 분리됨)."""
  a = 6978.0
  n = m.kepler.mean_motion(a)
  t = 500.0
  state1 = m.cw_state_transition(0.0, 0.0, 1.0, 0.0, 0.0, 0.5, n, t)
  state2 = m.cw_state_transition(10.0, -5.0, 1.0, 0.02, -0.01, 0.5, n, t)
  assert state1["z"] == pytest.approx(state2["z"])
  assert state1["vz"] == pytest.approx(state2["vz"])


def test_zero_drift_velocity_formula():
  n = 1.0e-3
  x0 = 2.0
  assert m.cw_zero_drift_velocity(x0, n) == pytest.approx(-2 * n * x0)


def test_zero_drift_condition_bounds_y_over_multiple_orbits():
  """제로 드리프트 속도를 주면 여러 궤도를 돌아도 y가 발산하지 않아야 한다."""
  a = 6978.0
  n = m.kepler.mean_motion(a)
  period = 2 * np.pi / n
  x0 = 1.0
  vy0 = m.cw_zero_drift_velocity(x0, n)
  state = m.cw_state_transition(x0, 0.0, 0.0, 0.0, vy0, 0.0, n, 10 * period)
  assert abs(state["y"]) < 1.0


def test_no_correction_drifts_linearly_with_orbit_count():
  """제로 드리프트 속도 없이(vy0=0) 두면, y가 궤도 수에 거의 비례해 커져야 한다
  (secular 항 -6*n*t*x0가 선형이므로)."""
  a = 6978.0
  n = m.kepler.mean_motion(a)
  period = 2 * np.pi / n
  x0 = 1.0
  state_3 = m.cw_state_transition(x0, 0.0, 0.0, 0.0, 0.0, 0.0, n, 3 * period)
  state_10 = m.cw_state_transition(x0, 0.0, 0.0, 0.0, 0.0, 0.0, n, 10 * period)
  ratio = state_10["y"] / state_3["y"]
  assert ratio == pytest.approx(10 / 3, rel=0.01)


def test_rendezvous_delta_v_reaches_origin_exactly():
  """이 스크립트의 핵심 주장: 랑데부 델타-V로 역산한 초기속도가 CW 방정식에
  다시 대입했을 때 정확히 원점(상대위치 0,0)에 도달해야 한다."""
  result = m.demo_rendezvous_delta_v_reaches_target()
  assert result["position_error_km"] < 1e-6


def test_cw_rendezvous_delta_v_matches_manual_state_transition():
  a = 6978.0
  n = m.kepler.mean_motion(a)
  x0, y0, z0 = 1.0, -3.0, 0.0
  transfer_time = 900.0
  rendezvous = m.cw_rendezvous_delta_v(x0, y0, z0, n, transfer_time)
  final_state = m.cw_state_transition(x0, y0, z0, rendezvous["vx0"], rendezvous["vy0"], 0.0, n, transfer_time)
  assert final_state["x"] == pytest.approx(0.0, abs=1e-6)
  assert final_state["y"] == pytest.approx(0.0, abs=1e-6)


def test_cw_approximation_error_increases_with_larger_offset():
  """이 스크립트의 핵심 주장: 초기 반경 옵셋이 커질수록 CW 근사가 실제 비선형
  전파와 벗어나는 정도(상대오차)가 커져야 한다."""
  rows = m.demo_cw_valid_only_for_small_separation()
  assert rows[0]["relative_error_pct"] < rows[-1]["relative_error_pct"]
  assert rows[-1]["relative_error_pct"] > 1.0


def test_cw_approximation_small_offset_nearly_exact():
  """초기 옵셋이 매우 작을 때(0.1km, 목표 궤도 반지름의 0.001% 수준)는 CW 근사가
  실제 비선형 전파와 거의 완벽히 일치해야 한다."""
  rows = m.demo_cw_valid_only_for_small_separation()
  assert rows[0]["relative_error_pct"] < 0.1
