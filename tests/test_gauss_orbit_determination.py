"""gauss_orbit_determination.py 검증: 단위 시선벡터가 알려진 위치 방향과
일치하는지, 물리적으로 유효한 근 선택이 음의 rho2를 내포하는 근을 거르고
유효 후보 중 가장 작은 근을 고르는지, 유효 후보가 없으면 ValueError를 내는지,
반복 개선이 실제로 정확도를 높이는지, 검증된 시나리오에서 5개 비원형 궤도요소
전부가 합리적 오차 안에 있는지, 거의 공면인 관측 기하에서 ValueError가
발생하는지, 거의 원궤도에서 14번의 SVD 방법과 결과가 근접하는지, 데모 함수
자체가 핵심 주장을 재검증하는지 확인."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("missions/gauss_orbit_determination.py")


def test_look_angles_to_eci_unit_vector_matches_known_position_direction():
  """알려진 ECI 위치에서 구한 방위각/고도각으로 복원한 단위벡터가 실제
  (위치-관측자)/거리 방향과 일치해야 한다."""
  site_lat = np.radians(35.0)
  site_lon = np.radians(129.0)
  t = 123.4
  gst = m.frames.gst_at_time(t)
  site_pos = m.frames.site_position_eci(site_lat, site_lon, gst)
  position = site_pos + np.array([1000.0, 2000.0, 3000.0])

  az, el, _r = m.frames.eci_position_to_look_angles(position, site_lat, site_lon, t)
  unit_vec = m.look_angles_to_eci_unit_vector(az, el, site_lat, site_lon, t)

  expected_direction = (position - site_pos) / np.linalg.norm(position - site_pos)
  assert np.allclose(unit_vec, expected_direction, atol=1e-9)
  assert np.linalg.norm(unit_vec) == pytest.approx(1.0)


def test_select_physical_root_rejects_negative_implied_rho2():
  """양의 실수 r2 후보라도 내포하는 rho2가 음수면 제외해야 한다."""
  mu = m.EARTH_MU_KM3_S2
  # r2가 크면 rho2=A+mu*B/r2^3이 A에 수렴 -> A를 음수로 크게 잡아 rho2<0을 유도
  A = -10000.0
  B = 1.0
  earth_r = m.EARTH_RADIUS_KM
  roots = np.array([complex(earth_r + 100.0, 0.0), complex(earth_r + 5000.0, 0.0)])
  # 둘 다 rho2 = A + mu*B/r2^3 이 음수가 되도록(큰 음의 A 대비 mu*B/r2^3이 작음)
  with pytest.raises(ValueError):
    m.select_physical_root(roots, A, B, mu)


def test_select_physical_root_raises_when_no_valid_candidate():
  """유효 후보가 전혀 없으면 ValueError."""
  mu = m.EARTH_MU_KM3_S2
  roots = np.array([complex(1.0, 500.0), complex(-5000.0, 0.0)])  # 허근 하나, 음의 실근 하나
  with pytest.raises(ValueError):
    m.select_physical_root(roots, 100.0, 1.0, mu)


def test_select_physical_root_picks_smallest_valid_candidate():
  """여러 유효 후보가 있으면 가장 작은 r2를 선택해야 한다."""
  mu = m.EARTH_MU_KM3_S2
  earth_r = m.EARTH_RADIUS_KM
  # A,B를 양수로 잡아 r2가 클수록 rho2=A+mu*B/r2^3 도 양수로 유지되도록 함
  A = 100.0
  B = 1.0
  r2_small = earth_r + 500.0
  r2_large = earth_r + 10000.0
  roots = np.array([complex(r2_large, 0.0), complex(r2_small, 0.0)])
  result = m.select_physical_root(roots, A, B, mu)
  assert result["r2"] == pytest.approx(r2_small)
  assert len(result["all_candidates"]) == 2


def test_gauss_iteration_converges_and_improves_accuracy():
  """검증된 기하에서 반복(max_iterations=15)을 쓴 결과가, 반복을 1회로
  제한한 결과보다 반장축 오차가 작아야 한다 - 반복의 효과 자체를 직접 검증."""
  t1, t2, t3 = m.find_gauss_observation_times(
      m.DEFAULT_SEMI_MAJOR_AXIS_KM, m.DEFAULT_ECCENTRICITY, m.DEFAULT_INCLINATION_RAD,
      m.DEFAULT_RAAN_RAD, m.DEFAULT_ARGP_RAD, m.DEFAULT_MEAN_ANOMALY0_RAD,
      m.DEFAULT_SITE_LATITUDE_RAD, m.DEFAULT_SITE_LONGITUDE_RAD)
  observations = m.simulate_gauss_observations(
      m.DEFAULT_SEMI_MAJOR_AXIS_KM, m.DEFAULT_ECCENTRICITY, m.DEFAULT_INCLINATION_RAD,
      m.DEFAULT_RAAN_RAD, m.DEFAULT_ARGP_RAD, m.DEFAULT_MEAN_ANOMALY0_RAD, [t1, t2, t3],
      m.DEFAULT_SITE_LATITUDE_RAD, m.DEFAULT_SITE_LONGITUDE_RAD)

  gst_list = [m.frames.gst_at_time(obs["time_sec"]) for obs in observations]
  R = [m.frames.site_position_eci(m.DEFAULT_SITE_LATITUDE_RAD, m.DEFAULT_SITE_LONGITUDE_RAD, gst)
       for gst in gst_list]
  L = [m.look_angles_to_eci_unit_vector(obs["azimuth_rad"], obs["elevation_rad"],
                                         m.DEFAULT_SITE_LATITUDE_RAD, m.DEFAULT_SITE_LONGITUDE_RAD,
                                         obs["time_sec"]) for obs in observations]
  tau1, tau3 = t1 - t2, t3 - t2

  result_1iter = m.gauss_solve_r2_v2(L[0], L[1], L[2], R[0], R[1], R[2], tau1, tau3, max_iterations=1)
  result_converged = m.gauss_solve_r2_v2(L[0], L[1], L[2], R[0], R[1], R[2], tau1, tau3, max_iterations=15)

  true_r2 = np.linalg.norm(m.true_orbit_position_general(
      m.DEFAULT_SEMI_MAJOR_AXIS_KM, m.DEFAULT_ECCENTRICITY, m.DEFAULT_INCLINATION_RAD,
      m.DEFAULT_RAAN_RAD, m.DEFAULT_ARGP_RAD, m.DEFAULT_MEAN_ANOMALY0_RAD, t2))

  err_1iter = abs(np.linalg.norm(result_1iter["r2_vec"]) - true_r2)
  err_converged = abs(np.linalg.norm(result_converged["r2_vec"]) - true_r2)
  assert err_converged < err_1iter, "반복 개선 후 오차가 1회 패스보다 작아야 함"


def test_gauss_recovers_known_elliptical_orbit_noise_free():
  """검증된 시나리오(a=8000,e=0.1,i=30,raan=40,argp=50)에서 노이즈 없이
  5개 비원형 요소 전부(a,e,i,raan,argp)가 각자의 합리적 허용오차 안에 있는지."""
  t1, t2, t3 = m.find_gauss_observation_times(
      m.DEFAULT_SEMI_MAJOR_AXIS_KM, m.DEFAULT_ECCENTRICITY, m.DEFAULT_INCLINATION_RAD,
      m.DEFAULT_RAAN_RAD, m.DEFAULT_ARGP_RAD, m.DEFAULT_MEAN_ANOMALY0_RAD,
      m.DEFAULT_SITE_LATITUDE_RAD, m.DEFAULT_SITE_LONGITUDE_RAD)
  observations = m.simulate_gauss_observations(
      m.DEFAULT_SEMI_MAJOR_AXIS_KM, m.DEFAULT_ECCENTRICITY, m.DEFAULT_INCLINATION_RAD,
      m.DEFAULT_RAAN_RAD, m.DEFAULT_ARGP_RAD, m.DEFAULT_MEAN_ANOMALY0_RAD, [t1, t2, t3],
      m.DEFAULT_SITE_LATITUDE_RAD, m.DEFAULT_SITE_LONGITUDE_RAD)
  recovered = m.determine_orbit_gauss(observations, m.DEFAULT_SITE_LATITUDE_RAD, m.DEFAULT_SITE_LONGITUDE_RAD)

  a_error_pct = abs(recovered["semi_major_axis_km"] - m.DEFAULT_SEMI_MAJOR_AXIS_KM) / m.DEFAULT_SEMI_MAJOR_AXIS_KM * 100
  e_error_abs = abs(recovered["eccentricity"] - m.DEFAULT_ECCENTRICITY)
  i_error_deg = abs(np.degrees(recovered["inclination_rad"]) - np.degrees(m.DEFAULT_INCLINATION_RAD))
  raan_error_deg = abs(np.degrees(recovered["raan_rad"]) - np.degrees(m.DEFAULT_RAAN_RAD))
  argp_error_deg = abs(np.degrees(recovered["argp_rad"]) - np.degrees(m.DEFAULT_ARGP_RAD))

  assert a_error_pct < 5.0
  assert e_error_abs < 0.05
  assert i_error_deg < 2.0
  assert raan_error_deg < 2.0
  assert argp_error_deg < 5.0


def test_gauss_raises_on_near_coplanar_geometry():
  """D0가 d0_min_threshold 아래가 되도록 구성한(매우 짧은 관측 간격) 관측에서
  ValueError가 발생해야 한다 - 조용히 틀린 결과를 내면 안 됨."""
  t1, t2, t3 = m.find_gauss_observation_times(
      m.DEFAULT_SEMI_MAJOR_AXIS_KM, m.DEFAULT_ECCENTRICITY, m.DEFAULT_INCLINATION_RAD,
      m.DEFAULT_RAAN_RAD, m.DEFAULT_ARGP_RAD, m.DEFAULT_MEAN_ANOMALY0_RAD,
      m.DEFAULT_SITE_LATITUDE_RAD, m.DEFAULT_SITE_LONGITUDE_RAD, spacing_fraction=0.05)
  observations = m.simulate_gauss_observations(
      m.DEFAULT_SEMI_MAJOR_AXIS_KM, m.DEFAULT_ECCENTRICITY, m.DEFAULT_INCLINATION_RAD,
      m.DEFAULT_RAAN_RAD, m.DEFAULT_ARGP_RAD, m.DEFAULT_MEAN_ANOMALY0_RAD, [t1, t2, t3],
      m.DEFAULT_SITE_LATITUDE_RAD, m.DEFAULT_SITE_LONGITUDE_RAD)
  with pytest.raises(ValueError):
    m.determine_orbit_gauss(observations, m.DEFAULT_SITE_LATITUDE_RAD, m.DEFAULT_SITE_LONGITUDE_RAD)


def test_gauss_agrees_with_svd_method_for_near_circular_orbit():
  """데모4: 이심률 거의 0인 궤도에서 14번 SVD 방법과 이 스크립트 가우스법의
  반장축/경사각이 서로 근접해야 한다."""
  result = m.demo_compare_with_circular_svd_method()
  assert result["a_diff_pct"] < 10.0
  assert result["i_diff_deg"] < 5.0


def test_demo_gauss_recovers_elliptical_orbit_returns_consistent_dict():
  """데모1을 직접 재호출해 반환 dict의 핵심 값을 재검증한다."""
  result = m.demo_gauss_recovers_elliptical_orbit()
  assert result["a_error_pct"] < 5.0
  assert result["e_error_abs"] < 0.05
  assert result["recovered"]["iterations_used"] > 0


def test_demo_accuracy_vs_d0_degeneracy_returns_consistent_dict():
  """데모2를 직접 재호출해 D0가 작을 때 ValueError가 트리거됐는지 재검증한다."""
  result = m.demo_accuracy_vs_d0_degeneracy()
  assert result["threshold_triggered"] is True
  assert len(result["rows"]) > 0


def test_demo_small_noise_sensitivity_returns_consistent_dict():
  """데모3을 직접 재호출해 노이즈 하의 평균 오차가 합리적 범위인지 재검증한다."""
  result = m.demo_small_noise_sensitivity(num_trials=5)
  assert result["mean_a_error_pct"] < 10.0
  assert len(result["rows"]) == 5
