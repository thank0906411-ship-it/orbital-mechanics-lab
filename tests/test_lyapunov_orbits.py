"""lyapunov_orbits.py 검증: 고유벡터 선택이 실제 야코비안 고유값과 일치하는
ω를 주는지, 선형 추정 궤도가 예측 주기로 닫히는지, 미분수정이 작은 진폭에서
수렴하고 잔여 vx가 작은지, 수정된 궤도가 실제로 주기적인지(전체 주기 적분 후
시작점 복귀), STM 초기 det가 1에 가까운지, 진폭이 0에 가까울수록 주기가
선형 예측(2π/ω)에 수렴하는지, L1/L2 두 곳에서 모두 방법이 동작하는지."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("missions/lyapunov_orbits.py")


def test_center_eigenvector_omega_matches_jacobian_eigenvalue():
  """planar_center_eigenvector가 반환하는 omega는 실제로 jacobian_at_point의
  순허수 고유값의 허수부와 일치해야 한다(재계산이 아니라 직접 재사용이므로
  항상 성립해야 하지만, 로직 자체가 옳은지 독립적으로 재확인)."""
  mu = m.EARTH_MOON_MASS_RATIO
  x_l1, _, _ = m.find_collinear_lagrange_points(mu)
  omega, direction = m.planar_center_eigenvector(x_l1, 0.0, mu)

  eigenvalues = np.linalg.eigvals(m.jacobian_at_point(x_l1, 0.0, mu))
  matching = [ev for ev in eigenvalues if abs(ev.imag - omega) < 1e-9 and abs(ev.real) < 1e-6]
  assert len(matching) >= 1, "선택된 omega는 야코비안의 순허수 고유값 허수부와 일치해야 함"
  assert np.linalg.norm(direction) > 0, "고유벡터 방향은 0벡터가 아니어야 함"


def test_linear_guess_orbit_closes_near_starting_point():
  """중심 고유벡터 방향의 작은 섭동을 (비선형) CR3BP로 선형 예측 주기(2π/ω)만큼
  적분하면 시작점 근처로 돌아와야 한다(진폭이 작을수록 더 정확)."""
  mu = m.EARTH_MOON_MASS_RATIO
  x_l1, _, _ = m.find_collinear_lagrange_points(mu)
  amplitude = 0.0001
  omega, _ = m.planar_center_eigenvector(x_l1, 0.0, mu)
  period_predicted = 2 * np.pi / omega

  x0, vy0, _ = m.linear_ic_guess(x_l1, 0.0, mu, amplitude)
  dt = 0.0005
  num_steps = int(period_predicted / dt)
  state = np.array([x0, 0.0, 0.0, vy0])
  for _ in range(num_steps):
    state = m._rk4_step_state_only(state, dt, mu)

  distance = np.hypot(state[0] - x0, state[1] - 0.0)
  assert distance < amplitude * 0.05, "선형 추정 궤도는 예측 주기 후 진폭의 5% 이내로 복귀해야 함"


def test_stm_determinant_near_one_at_start():
  """STM은 항등행렬(det=1)에서 출발해야 한다(초기값 검산) — 이후 짧게
  전파해도 심플렉틱 성질(det≈1)이 거의 유지돼야 planar_jacobian_general의
  배선이 올바르다는 신호다."""
  mu = m.EARTH_MOON_MASS_RATIO
  x_l1, _, _ = m.find_collinear_lagrange_points(mu)
  amplitude = 0.01
  x0, vy0, _ = m.linear_ic_guess(x_l1, 0.0, mu, amplitude)

  state = np.array([x0, 0.0, 0.0, vy0])
  stm = np.eye(4)
  assert np.linalg.det(stm) == pytest.approx(1.0, abs=1e-12)

  for _ in range(50):
    state, stm = m.rk4_step_state_and_stm(state, stm, 0.002, mu)
  assert np.linalg.det(stm) == pytest.approx(1.0, abs=1e-3)


def test_differential_correction_converges_for_small_amplitude():
  """작은 진폭(0.01)에서는 미분수정이 반드시 수렴하고, 수렴 후 잔여 vx가
  충분히 작아야 한다."""
  mu = m.EARTH_MOON_MASS_RATIO
  x_l1, _, _ = m.find_collinear_lagrange_points(mu)
  result = m.differential_correct_lyapunov_orbit(x_l1, mu, 0.01)
  assert result["converged"]
  assert result["residual_vx"] < 1e-9


def test_corrected_orbit_is_periodic():
  """미분수정된 초기조건으로 전체 주기를 적분하면 시작점에 거의 정확히
  복귀해야 한다(주기성 검증) — 이 스크립트의 핵심 주장."""
  mu = m.EARTH_MOON_MASS_RATIO
  x_l1, _, _ = m.find_collinear_lagrange_points(mu)
  result = m.differential_correct_lyapunov_orbit(x_l1, mu, 0.01)
  assert result["converged"]

  full_orbit = m.propagate_full_orbit(result["x0"], result["vy0"], mu, dt=0.001,
                                       half_period=result["half_period"])
  final_state = full_orbit["final_state"]
  periodicity_error = np.hypot(final_state[0] - result["x0"], final_state[1] - 0.0)
  assert periodicity_error < 1e-6


def test_period_converges_to_linear_prediction_as_amplitude_shrinks():
  """진폭이 0에 가까워질수록 실제(비선형) 주기가 선형 예측(2π/ω)에 수렴해야
  한다 — 선형 근사가 진폭이 작을 때의 극한이라는 것을 확인한다."""
  mu = m.EARTH_MOON_MASS_RATIO
  x_l1, _, _ = m.find_collinear_lagrange_points(mu)
  omega, _ = m.planar_center_eigenvector(x_l1, 0.0, mu)
  period_linear = 2 * np.pi / omega

  result_small = m.differential_correct_lyapunov_orbit(x_l1, mu, 0.001)
  result_large = m.differential_correct_lyapunov_orbit(x_l1, mu, 0.04)
  assert result_small["converged"] and result_large["converged"]

  deviation_small = abs(2 * result_small["half_period"] - period_linear)
  deviation_large = abs(2 * result_large["half_period"] - period_linear)
  assert deviation_small < deviation_large, "작은 진폭일수록 선형 예측과의 편차가 작아야 함"


def test_method_converges_at_both_l1_and_l2():
  """같은 미분수정 방법이 L1과 L2 양쪽에서 모두 수렴해야 한다 - 평형점에
  종속되지 않는 일반적인 방법임을 확인한다."""
  mu = m.EARTH_MOON_MASS_RATIO
  x_l1, x_l2, _ = m.find_collinear_lagrange_points(mu)
  result_l1 = m.differential_correct_lyapunov_orbit(x_l1, mu, 0.02)
  result_l2 = m.differential_correct_lyapunov_orbit(x_l2, mu, 0.02)
  assert result_l1["converged"]
  assert result_l2["converged"]
  assert result_l1["half_period"] != pytest.approx(result_l2["half_period"], rel=1e-3)


def test_newton_correction_converges_quadratically():
  """보정 분모에 올바른 STM 원소(Φ[y,vy0])를 쓰면 뉴턴법이 2차 수렴해 몇 번
  만에 끝나야 한다 — 잘못된 원소(Φ[x,vy0])를 쓰면 선형 수렴으로 20~40회가
  걸려 반복 한도에 걸리기 직전까지 간다."""
  mu = m.EARTH_MOON_MASS_RATIO
  x_l1, x_l2, _ = m.find_collinear_lagrange_points(mu)
  for x_eq, amplitude in [(x_l1, 0.01), (x_l1, 0.05), (x_l2, 0.02)]:
    result = m.differential_correct_lyapunov_orbit(x_eq, mu, amplitude)
    assert result["converged"]
    assert result["iterations"] <= 8


def test_center_eigenvector_starts_on_x_axis_perpendicular():
  """위상 정렬 후 실수 방향은 x축을 수직으로 지나는 출발점([x, 0, 0, vy])이어야
  하고, 진폭의 대부분이 x·vy 성분에 실려야 한다(고유벡터 위상에 무관)."""
  mu = m.EARTH_MOON_MASS_RATIO
  x_l1, x_l2, _ = m.find_collinear_lagrange_points(mu)
  for x_eq in (x_l1, x_l2):
    _, direction = m.planar_center_eigenvector(x_eq, 0.0, mu)
    assert direction[0] > 0
    assert abs(direction[1]) < 1e-12 and abs(direction[2]) < 1e-12
    assert abs(direction[3]) > 0


def test_non_converged_result_reports_last_evaluated_state():
  """반복 한도에 걸려 수렴하지 못하면, 반환된 vy0로 다시 평가한 잔여 vx가
  residual_vx와 일치해야 한다(갱신만 되고 평가되지 않은 vy0를 돌려주면 안 됨)."""
  mu = m.EARTH_MOON_MASS_RATIO
  x_l1, _, _ = m.find_collinear_lagrange_points(mu)
  result = m.differential_correct_lyapunov_orbit(x_l1, mu, 0.01, max_iterations=2)
  assert not result["converged"]
  crossing = m.find_half_period_crossing(result["x0"], result["vy0"], mu, 0.002)
  assert abs(crossing["state_half"][2]) == pytest.approx(result["residual_vx"], rel=1e-12)
  assert result["half_period"] == pytest.approx(crossing["t_half"], rel=1e-12)
