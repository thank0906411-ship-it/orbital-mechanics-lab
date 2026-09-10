"""intersatellite_link_visibility.py 검증: 기하학적 시선 차단 판정 함수가
정반대편(반드시 가려짐), 같은 위치 근처(가려지지 않음), 선분 바깥에 최근접점이
있는 경우(가려지지 않음)를 올바르게 판정하는지, 그리고 시계열/데모가 정상
동작하는지."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("constellations/intersatellite_link_visibility.py")


def test_opposite_side_of_earth_is_blocked():
  """이 스크립트의 핵심 경계 케이스: 정반대편 위성은 반드시 가려져야 하고
  최근접 거리는 0에 가까워야 한다."""
  a = m.EARTH_RADIUS_KM + 550.0
  pos1 = np.array([a, 0.0, 0.0])
  pos2 = np.array([-a, 0.0, 0.0])
  blocked, closest_distance = m.is_line_of_sight_blocked_by_earth(pos1, pos2)
  assert blocked
  assert closest_distance == pytest.approx(0.0, abs=1e-6)


def test_adjacent_satellites_not_blocked():
  """가까이 있는 두 위성 사이의 시선은 지구에 가려지지 않아야 한다."""
  a = m.EARTH_RADIUS_KM + 550.0
  pos1 = np.array([a, 0.0, 0.0])
  pos2 = np.array([a * np.cos(np.radians(5.0)), a * np.sin(np.radians(5.0)), 0.0])
  blocked, closest_distance = m.is_line_of_sight_blocked_by_earth(pos1, pos2)
  assert not blocked
  assert closest_distance > m.EARTH_RADIUS_KM


def test_closest_point_outside_segment_not_blocked():
  """지구 중심에서 직선(무한 연장)까지의 최근접점이 선분 바깥에 있으면(s*<0),
  clip으로 s=0(pos1 자체)이 최근접점이 되어야 하고, 실제 선분은 지구 중심에서
  더 멀리 있으므로 가려지지 않아야 한다."""
  # 두 위성이 모두 같은 방향(+x)에 멀리 떨어져 있어, 선분 전체가 원점에서 먼 곳에 있음
  a = m.EARTH_RADIUS_KM + 550.0
  pos1 = np.array([a, 1000.0, 0.0])
  pos2 = np.array([a, 2000.0, 0.0])
  blocked, closest_distance = m.is_line_of_sight_blocked_by_earth(pos1, pos2)
  assert not blocked
  assert closest_distance == pytest.approx(np.linalg.norm(pos1))


def test_identical_positions_uses_point_distance():
  """두 위성이 사실상 같은 위치면 선분이 정의되지 않으므로, 그 지점 자체의
  원점까지 거리로 판정해야 한다."""
  a = m.EARTH_RADIUS_KM + 550.0
  pos = np.array([a, 0.0, 0.0])
  blocked, closest_distance = m.is_line_of_sight_blocked_by_earth(pos, pos)
  assert not blocked
  assert closest_distance == pytest.approx(a)


def test_same_position_inside_earth_is_blocked():
  pos = np.array([1000.0, 0.0, 0.0])  # 지구 반지름(6378km)보다 작음
  blocked, closest_distance = m.is_line_of_sight_blocked_by_earth(pos, pos)
  assert blocked
  assert closest_distance == pytest.approx(1000.0)


def test_orbital_position_eci_magnitude_equals_orbital_radius():
  a, e = 7000.0, 0.0
  position = m.orbital_position_eci(a, e, np.radians(53.0), 0.0, 0.0, 0.0, time_sec=0.0)
  state = m.kepler.propagate_orbit(a, e, mean_anomaly0_rad=0.0, time_sec=0.0)
  assert np.linalg.norm(position) == pytest.approx(state["r"])


def test_compute_isl_visibility_time_series_has_expected_length():
  a = m.EARTH_RADIUS_KM + 550.0
  sat1 = {"semi_major_axis_km": a, "eccentricity": 0.0, "inclination_rad": np.radians(53.0),
          "raan_rad": 0.0, "argp_rad": 0.0, "mean_anomaly0_rad": 0.0}
  sat2 = {"semi_major_axis_km": a, "eccentricity": 0.0, "inclination_rad": np.radians(53.0),
          "raan_rad": 0.0, "argp_rad": 0.0, "mean_anomaly0_rad": np.radians(30.0)}
  rows = m.compute_isl_visibility_time_series(sat1, sat2, duration_sec=600.0, num_steps=10)
  assert len(rows) == 10


def test_same_plane_demo_mostly_visible():
  """이 스크립트의 핵심 주장: 위상차가 크지 않은 같은 평면 위성은 가려지는
  시간 비율이 절반 미만이어야 한다."""
  _series, blocked_fraction = m.demo_same_plane_satellites_mostly_visible()
  assert blocked_fraction < 0.5


def test_polar_vs_equatorial_demo_runs():
  series, blocked_fraction = m.demo_polar_vs_equatorial_plane_crossing()
  assert len(series) > 0
  assert 0.0 <= blocked_fraction <= 1.0
