"""attitude_dependent_srp.py 검증: 쿼터니언 벡터 회전이 항등/90도 경계
케이스에서 올바른지, 코사인 법칙 유효 단면적이 0/60/90/180도에서 정확한지,
패널이 정면을 향할 때 24번의 캐논볼 모델과 일치하는지(극한 케이스), 텀블링
위성의 SRP가 캐논볼과 달리 진동하는지, 태양지향 제어가 캐논볼 최댓값 근처를
유지하는지, 궤도 경로 편차가 0보다 크지만 반장축 대비 작은지 확인."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("perturbations/attitude_dependent_srp.py")


def test_rotate_vector_by_quaternion_identity_returns_same_vector():
  """항등 쿼터니언으로 회전시키면 벡터가 그대로 나와야 한다."""
  v = np.array([1.0, 2.0, 3.0])
  rotated = m.rotate_vector_by_quaternion(v, m.pid.IDENTITY_QUATERNION)
  assert np.allclose(rotated, v)


def test_rotate_vector_by_quaternion_90deg_matches_known_rotation():
  """몸체 +z를 x축 기준 90도 회전시키면 -y가 나와야 한다(오른손 법칙:
  +x축 양의 회전은 +z를 -y로 보낸다)."""
  q = m.pid.axis_angle_to_quaternion([1, 0, 0], 90.0)
  rotated = m.rotate_vector_by_quaternion(np.array([0.0, 0.0, 1.0]), q)
  assert np.allclose(rotated, [0.0, -1.0, 0.0], atol=1e-9)


def test_effective_area_is_zero_when_panel_faces_away_from_sun():
  """패널이 태양 반대쪽(180도)을 향하면 유효 A/m이 정확히 0이어야 한다."""
  q = m.pid.axis_angle_to_quaternion([1, 0, 0], 180.0)
  eff = m.effective_area_to_mass_ratio(q, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), 0.02)
  assert eff == 0.0


def test_effective_area_matches_full_value_when_panel_faces_sun_directly():
  """패널이 태양을 정면으로 향하면(0도) 유효 A/m이 max_A/m과 같아야 한다."""
  eff = m.effective_area_to_mass_ratio(m.pid.IDENTITY_QUATERNION, np.array([0.0, 0.0, 1.0]),
                                        np.array([0.0, 0.0, 1.0]), 0.02)
  assert eff == pytest.approx(0.02)


def test_effective_area_scales_with_cosine_at_intermediate_angle():
  """60도에서 유효 A/m이 max_A/m*cos(60)=max_A/m*0.5와 일치해야 한다."""
  q = m.pid.axis_angle_to_quaternion([1, 0, 0], 60.0)
  eff = m.effective_area_to_mass_ratio(q, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), 0.02)
  assert eff == pytest.approx(0.01, rel=1e-6)


def test_effective_area_is_zero_at_exactly_90_degrees():
  """90도(패널 가장자리로 태양을 봄)에서는 유효 A/m이 0이어야 한다."""
  q = m.pid.axis_angle_to_quaternion([1, 0, 0], 90.0)
  eff = m.effective_area_to_mass_ratio(q, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), 0.02)
  assert abs(eff) < 1e-9


def test_attitude_dependent_srp_falls_back_to_cannonball_when_panel_faces_sun():
  """패널이 태양을 정면으로 향하는 자세에서는 이 스크립트의 SRP 가속도가
  24번의 캐논볼 srp_acceleration과 (같은 max A/m이라면) 일치해야 한다 -
  캐논볼 모델이 "항상 정면"이라는 특수 케이스의 극한임을 확인."""
  position_km = np.array([7000.0, 0.0, 0.0])
  sun_position_km = m.sun_position_eci_km(0.0)
  sun_direction = m.sun_direction_from_satellite(position_km, sun_position_km)

  # 패널 법선(몸체 +z)이 태양 방향과 일치하도록 자세를 역산
  axis = np.cross([0.0, 0.0, 1.0], sun_direction)
  axis_norm = np.linalg.norm(axis)
  angle_deg = np.degrees(np.arccos(np.clip(np.dot([0.0, 0.0, 1.0], sun_direction), -1.0, 1.0)))
  q = m.pid.axis_angle_to_quaternion(axis / axis_norm, angle_deg) if axis_norm > 1e-9 \
      else m.pid.IDENTITY_QUATERNION.copy()

  attitude_accel, eff_ratio = m.attitude_dependent_srp_acceleration(
      position_km, sun_position_km, q, np.array([0.0, 0.0, 1.0]), 1.5, 0.02)
  cannonball_accel = m.srp.srp_acceleration(position_km, sun_position_km, 1.5, 0.02)

  assert eff_ratio == pytest.approx(0.02, rel=1e-6)
  assert np.allclose(attitude_accel, cannonball_accel, rtol=1e-6)


def test_tumbling_satellite_produces_varying_srp_unlike_cannonball():
  """데모2: 유효 SRP 가속도의 표준편차가 캐논볼 값 대비 뚜렷이 커야 한다."""
  result = m.demo_tumbling_satellite_srp_oscillates_while_cannonball_stays_constant()
  assert result["std"] > result["cannonball_accel"] * 0.05
  assert abs(result["max_val"] - result["cannonball_accel"]) / result["cannonball_accel"] < 0.05


def test_sun_pointing_control_keeps_srp_near_cannonball_max():
  """데모3: 정착 후 유효 SRP가 캐논볼 최댓값의 95% 이상이어야 한다."""
  result = m.demo_sun_pointing_control_keeps_srp_near_cannonball_maximum()
  assert result["min_ratio"] > 0.95


def test_orbit_paths_diverge_but_remain_close():
  """데모4: 위치 차이가 0보다 크지만 반장축 대비 1% 미만이어야 한다."""
  result = m.demo_attitude_dependent_srp_diverges_orbit_path_from_cannonball_over_time()
  assert result["final_diff_km"] > 0.0
  assert result["final_diff_km"] < m.DEFAULT_SEMI_MAJOR_AXIS_KM * 0.01


def test_cosine_law_demo_matches_known_angles():
  """데모1: 0/60/90/180도 네 각도 전부 코사인 법칙과 일치해야 한다."""
  result = m.demo_effective_area_matches_cosine_law_at_known_angles()
  rows = {row["angle_deg"]: row["effective_area_to_mass_ratio_m2_kg"] for row in result["rows"]}
  assert rows[0.0] == pytest.approx(0.02)
  assert rows[60.0] == pytest.approx(0.01, rel=1e-6)
  assert abs(rows[90.0]) < 1e-9
  assert rows[180.0] == 0.0
