"""station_keeping.py 검증: 평면 변경 델타-V 공식이 알려진 경계 케이스(각도 0,
180도)와 일치하는지, 보정 없는 잔여 드리프트가 선형인지, 주기적 보정이 RAAN
오차를 허용 범위 이내로 묶는지, 허용 오차가 좁을수록 기동이 잦아지는지, 임무
기간 총 델타-V가 허용 오차 선택과 거의 무관한지."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("missions/station_keeping.py")


def test_plane_change_delta_v_zero_for_zero_angle():
  """평면 사이 각도가 0이면 기동이 필요 없으므로 델타-V는 0이어야 한다."""
  assert m.plane_change_delta_v(7.5, 0.0) == pytest.approx(0.0)


def test_plane_change_delta_v_doubles_speed_for_180_degrees():
  """180도 평면 변경은 속도 방향이 완전히 뒤집히므로 델타-V = 2v여야 한다."""
  v = 7.5
  assert m.plane_change_delta_v(v, np.pi) == pytest.approx(2 * v)


def test_plane_change_delta_v_small_angle_approximation():
  """각도가 작을 때 Δv ≈ v*angle로 근사되어야 한다(sin(x/2)≈x/2)."""
  v = 7.5
  small_angle = np.radians(0.01)
  exact = m.plane_change_delta_v(v, small_angle)
  approx = v * small_angle
  assert exact == pytest.approx(approx, rel=1e-4)


def test_circular_orbit_speed_matches_vis_viva():
  r = 7000.0
  expected = np.sqrt(m.EARTH_MU_KM3_S2 / r)
  assert m.circular_orbit_speed(r) == pytest.approx(expected)


def test_uncorrected_drift_grows_linearly_with_time():
  """이 스크립트의 핵심 주장: 보정 없이 방치하면 잔여 RAAN 오차가 시간에
  선형으로 비례해 커져야 한다."""
  rows = m.demo_uncorrected_drift_grows_linearly()
  assert rows[-1]["raan_error_deg"] > rows[0]["raan_error_deg"] * 5


def test_periodic_correction_keeps_error_within_tolerance():
  """이 스크립트의 핵심 주장: 주기적 보정을 넣으면 RAAN 오차가 항상 허용
  오차 이내로 유지되어야 한다(톱니파 패턴)."""
  result = m.demo_periodic_correction_bounds_the_drift()
  max_error_deg = max(np.degrees(err) for _t, err in result["history"])
  assert max_error_deg <= 0.1 * 1.001
  assert result["num_burns"] > 0


def test_tighter_tolerance_increases_burn_count():
  """허용 오차가 좁을수록 기동 횟수가 늘어나야 한다."""
  rows = m.demo_tighter_tolerance_needs_more_frequent_burns()
  assert rows[0]["num_burns"] >= 1
  assert rows[-1]["num_burns"] > rows[0]["num_burns"]


def test_total_delta_v_budget_roughly_independent_of_tolerance():
  """선형 근사 영역에서는 허용 오차를 바꿔도 임무 기간 총 델타-V가 크게
  달라지지 않아야 한다."""
  rows = m.demo_total_delta_v_budget_over_mission_lifetime()
  dv_values = [row["total_delta_v_ms"] for row in rows]
  spread_pct = (max(dv_values) - min(dv_values)) / min(dv_values) * 100
  assert spread_pct < 20.0


def test_residual_rate_is_small_fraction_of_predicted_rate():
  """잔여 오차율은 07번 예측 세차율보다 훨씬 작아야 한다(보정 대상은 예측
  자체가 아니라 그 예측에서 벗어나는 작은 차이여야 함)."""
  a, e, i = 7000.0, 0.01, np.radians(97.4)
  predicted = m.raan_precession_rate(a, e, i)
  residual = m.residual_raan_rate(a, e, i, residual_fraction=0.001)
  assert abs(residual) < abs(predicted) * 0.01
