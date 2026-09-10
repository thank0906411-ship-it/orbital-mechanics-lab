"""10_Constellation_coverage.py 검증: 워커 델타 성좌 생성이 요청한 위성 수/평면
수대로 정확히 만들어지는지(RAAN이 평면마다 균등 분배되는지), 커버리지 공백 탐지
로직이 05번의 접촉 창 탐지와 대칭적으로 동작하는지, 그리고 실제로 위성 수가
늘어나면 공백이 줄어드는(또는 최소한 늘어나지 않는) 경향이 있는지."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("constellations/10_Constellation_coverage.py")


def test_walker_delta_creates_correct_number_of_satellites():
  satellites = m.walker_delta_constellation(12, 3, phasing_factor=1,
                                             inclination_rad=np.radians(53.0), altitude_km=550.0)
  assert len(satellites) == 12


def test_walker_delta_raises_when_not_divisible():
  with pytest.raises(ValueError):
    m.walker_delta_constellation(10, 3, phasing_factor=1, inclination_rad=np.radians(53.0), altitude_km=550.0)


def test_walker_delta_raan_evenly_distributed_across_planes():
  satellites = m.walker_delta_constellation(9, 3, phasing_factor=0,
                                             inclination_rad=np.radians(53.0), altitude_km=550.0)
  raans_deg = sorted({round(np.degrees(s["raan_rad"]), 3) for s in satellites})
  assert raans_deg == pytest.approx([0.0, 120.0, 240.0])


def test_walker_delta_all_satellites_share_inclination_and_altitude():
  inclination = np.radians(53.0)
  satellites = m.walker_delta_constellation(6, 2, phasing_factor=1, inclination_rad=inclination, altitude_km=550.0)
  for sat in satellites:
    assert sat["inclination_rad"] == pytest.approx(inclination)
    assert sat["semi_major_axis_km"] == pytest.approx(m.frames.EARTH_RADIUS_KM + 550.0)
    assert sat["eccentricity"] == 0.0


def test_find_coverage_gaps_detects_gap_between_visible_periods():
  series = [{"t_sec": i * 10.0, "max_elevation_deg": el} for i, el in enumerate([15, 12, -5, -5, -5, 20, 18])]
  gaps = m.find_coverage_gaps(series, min_elevation_deg=10.0)
  assert len(gaps) == 1
  assert gaps[0]["duration_sec"] == pytest.approx(30.0)


def test_find_coverage_gaps_no_gap_when_always_visible():
  series = [{"t_sec": i * 10.0, "max_elevation_deg": el} for i, el in enumerate([15, 20, 25, 18])]
  assert m.find_coverage_gaps(series, min_elevation_deg=10.0) == []


def test_find_coverage_gaps_gap_at_series_end_is_closed():
  """관찰이 끝나는 시점까지 공백이면(다시 보이는 걸 못 봤어도) 그 구간을 공백으로
  마감해야 한다 — 05번의 접촉 창 탐지와 대칭적인 동작이다."""
  series = [{"t_sec": i * 10.0, "max_elevation_deg": el} for i, el in enumerate([15, 12, -5, -5])]
  gaps = m.find_coverage_gaps(series, min_elevation_deg=10.0)
  assert len(gaps) == 1
  assert gaps[0]["end_sec"] == series[-1]["t_sec"]


def test_more_satellites_do_not_increase_max_gap():
  """이 스크립트의 핵심 주장: 위성 수를 늘렸을 때 최대 공백 시간이 늘어나면 안 된다
  (반드시 줄어들 필요는 없지만, 절대 나빠지면 안 됨)."""
  rows = m.demo_more_satellites_reduce_max_gap()
  assert rows[-1]["max_gap_min"] <= rows[0]["max_gap_min"]


def test_single_vs_multi_plane_demo_runs_and_returns_two_rows():
  """이 데모는 '평면을 나누면 항상 좋다'는 가정이 뒤집히는 실측 결과를 그대로
  보여주는 것이 핵심이므로, 값의 대소를 강제하지 않고 데모가 정상 실행되는지와
  두 시나리오 모두 유효한 공백 데이터를 반환하는지만 확인한다."""
  rows = m.demo_single_plane_vs_multi_plane()
  assert len(rows) == 2
  for row in rows:
    assert row["max_gap_min"] >= 0.0


def test_coverage_degrades_above_inclination_latitude_demo_runs():
  rows = m.demo_coverage_degrades_above_inclination_latitude()
  assert len(rows) == 3
  for row in rows:
    assert row["max_gap_min"] >= 0.0
