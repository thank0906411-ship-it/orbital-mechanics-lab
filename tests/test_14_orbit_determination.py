"""14_Orbit_determination.py 검증: 방위각/고도각/거리 <-> ECI 위치 왕복 변환이
04번의 정확한 역함수인지, 작은 노이즈에서 원궤도 요소가 거의 정확히 복원되는지,
노이즈가 커질수록 오차가 커지는지, 관측 개수가 늘수록 평균 오차와 표준편차가
함께 줄어드는지."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("missions/14_Orbit_determination.py")


def test_look_angles_roundtrip_recovers_original_position():
  """무작위 ECI 위치를 04번 정변환으로 관측값으로 바꾼 뒤, 이 스크립트의
  역변환으로 되돌리면 원래 위치와 부동소수점 정밀도 수준으로 일치해야 한다."""
  site_lat = np.radians(35.0)
  site_lon = np.radians(129.0)
  t = 1200.0
  zenith_direction = m.frames.site_position_eci(site_lat, site_lon, m.frames.gst_at_time(t))
  zenith_direction /= np.linalg.norm(zenith_direction)
  original_position = zenith_direction * 12000.0

  azimuth, elevation, r = m.frames.eci_position_to_look_angles(original_position, site_lat, site_lon, t)
  assert elevation > 0, "테스트 위치는 지평선 위에 있어야 함"
  recovered = m.look_angles_to_eci_position(azimuth, elevation, r, site_lat, site_lon, t)
  assert recovered == pytest.approx(original_position, abs=1e-6)


def test_find_visible_time_window_is_within_horizon():
  """find_visible_time_window가 찾은 구간의 중간 시점은 실제로 고도각이 양수여야
  한다(지평선 위, 관측 가능)."""
  true_a, true_i, true_raan = 7000.0, np.radians(53.0), np.radians(80.0)
  site_lat, site_lon = np.radians(35.0), np.radians(129.0)
  window_start, window_end = m.find_visible_time_window(true_a, true_i, true_raan, site_lat, site_lon)
  assert window_end > window_start

  mid_t = (window_start + window_end) / 2
  position = m.true_orbit_position(true_a, true_i, true_raan, mid_t)
  _az, elevation, _r = m.frames.eci_position_to_look_angles(position, site_lat, site_lon, mid_t)
  assert elevation > 0


def test_circular_orbit_recovered_with_small_noise():
  """이 스크립트의 핵심 주장: 작은 관측 노이즈에서는 복원된 반장축/경사각이
  원래 값과 거의 정확히 일치해야 한다."""
  result = m.demo_circular_orbit_recovered_from_noisy_observations()
  assert result["a_error_pct"] < 1.0
  assert result["i_error_deg"] < 1.0


def test_accuracy_degrades_with_observation_noise():
  """관측 노이즈(각도)가 커질수록 복원된 경사각 오차도 커져야 한다."""
  rows = m.demo_accuracy_degrades_with_observation_noise()
  assert rows[-1]["inclination_error_deg"] > rows[0]["inclination_error_deg"]


def test_accuracy_nearly_exact_with_negligible_noise():
  """각도 노이즈가 0.001도 수준으로 거의 없으면, 경사각 오차도 거의 0에 가까워야
  한다."""
  rows = m.demo_accuracy_degrades_with_observation_noise()
  assert rows[0]["inclination_error_deg"] < 0.01


def test_more_observations_reduce_mean_and_std_error():
  """이 스크립트의 핵심 주장: 관측 개수가 늘수록(2개->30개) 여러 시행에 걸친
  평균 경사각 오차와 표준편차가 함께 줄어들어야 한다 — 단일 시행으로는 노이즈의
  요행 때문에 비단조로 보일 수 있으므로 여러 시행 평균으로 검증한다."""
  rows = m.demo_minimum_observations_needed()
  assert rows[-1]["mean_inclination_error_deg"] < rows[0]["mean_inclination_error_deg"]
  assert rows[-1]["std_inclination_error_deg"] < rows[0]["std_inclination_error_deg"]


def test_orbit_normal_from_positions_matches_known_plane():
  """알려진 궤도면(RAAN=0, 경사각=30도) 위의 위치벡터들로부터 SVD로 추정한 법선이
  실제 각운동량 방향(부호 무관)과 일치해야 한다."""
  inclination = np.radians(30.0)
  rotation = m.rotation_matrix_x(inclination)
  positions = [rotation @ np.array([7000.0 * np.cos(theta), 7000.0 * np.sin(theta), 0.0])
               for theta in np.linspace(0, 2, 6)]
  normal = m.orbit_normal_from_positions(positions)
  expected_normal = rotation @ np.array([0.0, 0.0, 1.0])
  assert abs(abs(np.dot(normal, expected_normal)) - 1.0) < 1e-6
