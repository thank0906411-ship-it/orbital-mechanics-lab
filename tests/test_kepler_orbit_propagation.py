"""kepler_orbit_propagation.py 검증: 케플러 방정식 뉴턴-랍슨 해의 정확성(잔차),
원궤도 특수해, 진근점 이각 변환의 기하학적 일관성, 케플러 제2법칙(면적속도 보존),
그리고 이심률에 따른 수렴 난이도 변화."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("propagation/kepler_orbit_propagation.py")


def test_solve_kepler_equation_residual_is_near_zero():
  """뉴턴-랍슨 해가 실제로 M = E - e*sin(E)를 만족하는지 직접 확인한다."""
  for mean_anomaly_deg, e in [(30, 0.1), (90, 0.5), (200, 0.7), (10, 0.9)]:
    m_rad = np.radians(mean_anomaly_deg)
    ecc_anomaly, _iterations = m.solve_kepler_equation(m_rad, e)
    residual = ecc_anomaly - e * np.sin(ecc_anomaly) - m_rad
    assert abs(residual) < 1e-9


def test_circular_orbit_solves_in_one_iteration():
  """e=0이면 케플러 방정식이 M=E로 단순화되어 초기값 E0=M 자체가 정답이다."""
  ecc_anomaly, iterations = m.solve_kepler_equation(np.radians(123.0), eccentricity=0.0)
  assert iterations == 1
  assert ecc_anomaly == pytest.approx(np.radians(123.0))


def test_higher_eccentricity_generally_needs_more_iterations():
  _e1, iter_low = m.solve_kepler_equation(np.radians(10.0), eccentricity=0.1)
  _e2, iter_high = m.solve_kepler_equation(np.radians(10.0), eccentricity=0.9)
  assert iter_high >= iter_low


def test_true_anomaly_matches_ecc_anomaly_at_perigee_and_apogee():
  """근지점(E=0)과 원지점(E=pi)에서는 진근점 이각도 정확히 0, pi여야 한다 —
  atan2 기반 변환이 이 경계 케이스에서 안 깨지는지 확인."""
  assert m.eccentric_to_true_anomaly(0.0, eccentricity=0.5) == pytest.approx(0.0)
  assert m.eccentric_to_true_anomaly(np.pi, eccentricity=0.5) == pytest.approx(np.pi)


def test_perifocal_position_distance_at_perigee_and_apogee():
  """근지점 거리 = a(1-e), 원지점 거리 = a(1+e) — 궤도역학의 기본 정의."""
  a, e = 8000.0, 0.3
  r_perigee, _x, _y = m.perifocal_position(a, e, true_anomaly_rad=0.0, ecc_anomaly_rad=0.0)
  r_apogee, _x, _y = m.perifocal_position(a, e, true_anomaly_rad=np.pi, ecc_anomaly_rad=np.pi)
  assert r_perigee == pytest.approx(a * (1 - e))
  assert r_apogee == pytest.approx(a * (1 + e))


def test_mean_motion_matches_period_formula():
  """평균운동 n = 2*pi / T 여야 한다(T는 케플러 제3법칙으로 구한 주기)."""
  a = 7000.0
  n = m.mean_motion(a)
  period = 2 * np.pi / n
  expected_period = 2 * np.pi * np.sqrt(a ** 3 / m.EARTH_MU_KM3_S2)
  assert period == pytest.approx(expected_period)


def test_propagate_orbit_after_one_full_period_returns_to_start():
  """한 궤도 주기가 정확히 지나면 평균 이각이 출발점과 같아야 한다(모듈로 2pi)."""
  a, e = 7000.0, 0.2
  period = 2 * np.pi / m.mean_motion(a)
  state = m.propagate_orbit(a, e, mean_anomaly0_rad=0.5, time_sec=period)
  assert state["mean_anomaly"] == pytest.approx(0.5, abs=1e-6)


def test_kepler_second_law_perigee_faster_than_apogee():
  """이 스크립트의 핵심 주장: 타원궤도는 근지점 근처에서 진근점 이각이 원지점
  근처보다 훨씬 빠르게 변한다(면적속도 일정)."""
  rows = m.demo_circular_vs_elliptical()
  elliptical_rows = [r for r in rows if r["orbit"] == "타원궤도(e=0.7)"]
  near_perigee = elliptical_rows[1]["true_anomaly_deg"] - elliptical_rows[0]["true_anomaly_deg"]
  mid_index = len(elliptical_rows) // 2
  near_apogee = elliptical_rows[mid_index + 1]["true_anomaly_deg"] - elliptical_rows[mid_index]["true_anomaly_deg"]
  assert near_perigee > near_apogee


def test_circular_orbit_true_anomaly_changes_uniformly():
  """원궤도(e=0)는 각속도가 일정해야 한다 — 각 구간 진근점 이각 변화가 전부 같음."""
  rows = m.demo_circular_vs_elliptical()
  circular_rows = [r for r in rows if r["orbit"] == "원궤도(e=0.0)"]
  deltas = [circular_rows[i + 1]["true_anomaly_deg"] - circular_rows[i]["true_anomaly_deg"]
            for i in range(len(circular_rows) - 1)]
  assert all(d == pytest.approx(deltas[0]) for d in deltas)


def test_position_time_series_returns_requested_number_of_points():
  rows = m.demo_position_time_series(semi_major_axis_km=7000.0, eccentricity=0.1,
                                      duration_sec=6000.0, num_steps=50)
  assert len(rows) == 50
  assert rows[0]["t_sec"] == pytest.approx(0.0)
  assert rows[-1]["t_sec"] == pytest.approx(6000.0)


def test_edge_case_near_parabolic_all_converge_within_max_iter():
  rows = m.demo_edge_case_near_parabolic()
  assert all(row["converged"] for row in rows)


def test_kepler_equation_convergence_demo_returns_all_eccentricities():
  rows = m.demo_kepler_equation_convergence()
  assert len(rows) == 8
  assert rows[0]["eccentricity"] == 0.0
  assert rows[0]["iterations"] == 1
