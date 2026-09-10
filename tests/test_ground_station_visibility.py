"""ground_station_visibility.py 검증: 접촉 창 탐지 로직(find_contact_windows)이
연속 구간을 정확히 찾는지, 경계 조건(관찰 끝날 때도 접촉 중인 경우)을 올바르게
처리하는지, 그리고 실측 궤도 전파로 접촉 창이 실제로 발견되는지."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("missions/ground_station_visibility.py")


def _series(elevations, dt=10.0):
  return [{"t_sec": i * dt, "elevation_deg": el} for i, el in enumerate(elevations)]


def test_find_contact_windows_detects_single_window():
  series = _series([-5, -2, 15, 30, 20, -1, -10])
  windows = m.find_contact_windows(series, min_elevation_deg=10.0)
  assert len(windows) == 1
  assert windows[0]["max_elevation_deg"] == 30


def test_find_contact_windows_detects_multiple_separate_windows():
  series = _series([-5, 15, 20, -5, -5, 12, 18, -5])
  windows = m.find_contact_windows(series, min_elevation_deg=10.0)
  assert len(windows) == 2


def test_find_contact_windows_no_contact_returns_empty():
  series = _series([-5, -2, -1, -8])
  windows = m.find_contact_windows(series, min_elevation_deg=10.0)
  assert windows == []


def test_find_contact_windows_always_visible_returns_one_window():
  series = _series([15, 20, 25, 30, 22])
  windows = m.find_contact_windows(series, min_elevation_deg=10.0)
  assert len(windows) == 1
  assert windows[0]["aos_sec"] == series[0]["t_sec"]
  assert windows[0]["los_sec"] == series[-1]["t_sec"]


def test_find_contact_windows_still_in_contact_at_series_end():
  """관찰이 끝나는 시점까지 접촉 중이면(LOS를 못 봤어도) 그 구간을 접촉 창으로
  마감해야 한다 — 관찰 윈도우 경계에서 접촉을 놓치면 안 된다."""
  series = _series([-5, -2, 15, 25, 30])
  windows = m.find_contact_windows(series, min_elevation_deg=10.0)
  assert len(windows) == 1
  assert windows[0]["los_sec"] == series[-1]["t_sec"]


def test_contact_window_duration_matches_aos_los_difference():
  series = _series([-5, 15, 20, 15, -5])
  windows = m.find_contact_windows(series, min_elevation_deg=10.0)
  w = windows[0]
  assert w["duration_sec"] == pytest.approx(w["los_sec"] - w["aos_sec"])


def test_orbital_position_eci_magnitude_equals_orbital_radius():
  """궤도면 위치를 회전만 시킨 것이므로, ECI 위치의 크기는 궤도 반지름 r과 같아야 한다."""
  a, e = 7000.0, 0.1
  i, raan, argp = np.radians(45.0), np.radians(30.0), np.radians(60.0)
  position = m.orbital_position_eci(a, e, i, raan, argp, mean_anomaly0_rad=0.0, time_sec=0.0)
  state = m.kepler.propagate_orbit(a, e, mean_anomaly0_rad=0.0, time_sec=0.0)
  assert np.linalg.norm(position) == pytest.approx(state["r"])


def test_single_pass_demo_finds_at_least_one_contact_window():
  """51.6도 경사궤도, 위도 35도 지상국이면 3주기 관찰 시 접촉 창이 최소 1개는
  있어야 한다 — 이 프로젝트의 핵심 주장(실측 접촉 창이 실제로 계산된다)."""
  _elevation_series, windows = m.demo_single_pass_contact_window()
  assert len(windows) >= 1
  for w in windows:
    assert w["max_elevation_deg"] > 10.0
    assert w["duration_sec"] > 0


def test_elevation_time_series_has_expected_length():
  a, e, i, raan, argp = 6978.0, 0.001, np.radians(51.6), 0.0, 0.0
  rows = m.compute_elevation_time_series(a, e, i, raan, argp,
                                          site_latitude_rad=np.radians(35.0), site_longitude_rad=np.radians(129.0),
                                          duration_sec=600.0, num_steps=10)
  assert len(rows) == 10
