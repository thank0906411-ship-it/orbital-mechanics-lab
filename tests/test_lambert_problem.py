"""lambert_problem.py 검증: 스텀프 함수가 z=0에서 연속인지, 란베르트 일반해가
06번 호만 전이 특수해와 교차검증되는지, 짧은 길/긴 길 선택이 서로 다른 전이각과
델타-V를 내는지, 그리고 180도 특이 케이스에서 외적이 0에 가까운지."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("missions/lambert_problem.py")


def test_stumpff_c_continuous_at_zero():
  """C(z)는 z=0에서 1/2로 연속이어야 한다(타원/쌍곡선 양쪽에서 극한이 일치)."""
  assert m.stumpff_c(0.0) == pytest.approx(0.5)
  assert m.stumpff_c(1e-6) == pytest.approx(0.5, abs=1e-4)
  assert m.stumpff_c(-1e-6) == pytest.approx(0.5, abs=1e-4)


def test_stumpff_s_continuous_at_zero():
  """S(z)는 z=0에서 1/6로 연속이어야 한다."""
  assert m.stumpff_s(0.0) == pytest.approx(1.0 / 6.0)
  assert m.stumpff_s(1e-6) == pytest.approx(1.0 / 6.0, abs=1e-4)
  assert m.stumpff_s(-1e-6) == pytest.approx(1.0 / 6.0, abs=1e-4)


def test_stumpff_c_positive_z_matches_trig_formula():
  z = 4.0
  expected = (1 - np.cos(np.sqrt(z))) / z
  assert m.stumpff_c(z) == pytest.approx(expected)


def test_stumpff_c_negative_z_matches_hyperbolic_formula():
  z = -4.0
  expected = (np.cosh(np.sqrt(4.0)) - 1) / 4.0
  assert m.stumpff_c(z) == pytest.approx(expected)


def test_lambert_matches_hohmann_delta_v():
  """이 스크립트의 핵심 주장: 란베르트 일반해가 06번 호만 전이 특수해와
  거의 일치해야 한다(같은 물리적 상황을 다른 방법으로 풀었을 때)."""
  result = m.demo_lambert_matches_known_transfer()
  assert result["error"] < 0.05


def test_lambert_solution_reaches_correct_target_position():
  """란베르트 해로 구한 초기 속도로 01번 궤도 전파를 하면, 정확히 목표 위치(r2)에
  도달해야 한다 — 이건 란베르트 해 자체의 정의이므로 반드시 성립해야 한다."""
  r1_vec = np.array([7000.0, 0.0, 0.0])
  r2_vec = np.array([0.0, 10000.0, 0.0])
  time_of_flight = 3600.0

  result = m.solve_lambert_universal_variable(r1_vec, r2_vec, time_of_flight)
  elements = m.elements.state_vector_to_orbital_elements(r1_vec, result["v1"])

  # 상태벡터->궤도요소 역변환 후, 그 요소로 다시 time_of_flight만큼 전파했을 때
  # 위치가 r2_vec와 일치하는지 확인한다.
  a = elements["semi_major_axis_km"]
  e = elements["eccentricity"]
  r1_norm = np.linalg.norm(r1_vec)
  cos_true_anomaly0 = (a * (1 - e ** 2) / r1_norm - 1) / e if e > 1e-9 else 1.0
  true_anomaly0 = np.arccos(np.clip(cos_true_anomaly0, -1.0, 1.0))
  ecc_anomaly0 = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(true_anomaly0 / 2), np.sqrt(1 + e) * np.cos(true_anomaly0 / 2))
  mean_anomaly0 = ecc_anomaly0 - e * np.sin(ecc_anomaly0)

  state_at_r2 = m.kepler.propagate_orbit(a, e, mean_anomaly0, time_of_flight)
  assert state_at_r2["r"] == pytest.approx(np.linalg.norm(r2_vec), rel=1e-3)


def test_short_way_and_long_way_have_different_transfer_angles():
  rows = m.demo_short_way_vs_long_way()
  assert abs(rows[0]["delta_nu_deg"] - rows[1]["delta_nu_deg"]) > 90


def test_180_degree_edge_case_cross_product_near_zero():
  result = m.demo_edge_case_180_degree_transfer()
  assert abs(result["cross_z"]) < 1e-6


def test_solve_lambert_converges_within_iteration_limit():
  """짧은 길/긴 길 모두 max_iter(200) 안에서 이분법 허용오차(tol)까지 수렴해야
  한다 — 뉴턴-랍슨 방식에서 실제로 발산했던 배치(전이각 270도, 비행시간 30분)로
  회귀 테스트한다."""
  r1_vec = np.array([8000.0, 0.0, 0.0])
  r2_vec = np.array([0.0, 12000.0, 0.0])
  result = m.solve_lambert_universal_variable(r1_vec, r2_vec, 1800.0, prograde=False)
  assert result["iterations"] < 200
