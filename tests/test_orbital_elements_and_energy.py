"""orbital_elements_and_energy.py 검증: 궤도요소<->상태벡터 왕복 변환의 정확성,
비에너지/비각운동량 보존, vis-viva 방정식, 그리고 원궤도에서 근점 편각이 정의되지
않는 특이 케이스가 안전하게 처리되는지."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("propagation/orbital_elements_and_energy.py")


def test_rotation_matrix_z_is_identity_at_zero_angle():
  result = m.rotation_matrix_z(0.0)
  assert np.allclose(result, np.eye(3))


def test_rotation_matrix_x_is_identity_at_zero_angle():
  result = m.rotation_matrix_x(0.0)
  assert np.allclose(result, np.eye(3))


def test_perifocal_velocity_matches_angular_momentum_definition():
  """h = sqrt(mu*p)로 정의된 속도가, 실제로 그 각운동량을 내는지 위치와 외적해 확인."""
  mu, p, e, nu = m.EARTH_MU_KM3_S2, 7000.0, 0.2, np.radians(45.0)
  vx_p, vy_p = m.perifocal_velocity(mu, p, e, nu)
  r = p / (1 + e * np.cos(nu))
  x_p, y_p = r * np.cos(nu), r * np.sin(nu)
  h_computed = x_p * vy_p - y_p * vx_p  # 2D 외적
  h_expected = np.sqrt(mu * p)
  assert h_computed == pytest.approx(h_expected, rel=1e-9)


def test_orbital_elements_roundtrip_recovers_original_values():
  a, e, i, raan, argp, nu = 7000.0, 0.15, np.radians(50.0), np.radians(80.0), np.radians(30.0), np.radians(100.0)
  ecc_anomaly = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))
  pos, vel = m.orbital_elements_to_state_vector(a, e, i, raan, argp, nu, ecc_anomaly)
  recovered = m.state_vector_to_orbital_elements(pos, vel)
  assert recovered["semi_major_axis_km"] == pytest.approx(a, rel=1e-6)
  assert recovered["eccentricity"] == pytest.approx(e, rel=1e-6)
  assert recovered["inclination_rad"] == pytest.approx(i, rel=1e-6)
  assert recovered["raan_rad"] == pytest.approx(raan, rel=1e-6)


def test_specific_energy_matches_vis_viva_formula():
  """비에너지가 -mu/(2a) 공식과 일치해야 한다."""
  a, e = 8000.0, 0.3
  pos, vel = m.orbital_elements_to_state_vector(a, e, 0.0, 0.0, 0.0, true_anomaly_rad=0.0, ecc_anomaly_rad=0.0)
  recovered = m.state_vector_to_orbital_elements(pos, vel)
  expected_energy = -m.EARTH_MU_KM3_S2 / (2 * a)
  assert recovered["specific_energy"] == pytest.approx(expected_energy, rel=1e-9)


def test_specific_angular_momentum_matches_formula():
  """비각운동량이 sqrt(mu*a*(1-e^2)) 공식과 일치해야 한다."""
  a, e = 7500.0, 0.4
  pos, vel = m.orbital_elements_to_state_vector(a, e, np.radians(20.0), 0.0, 0.0,
                                                 true_anomaly_rad=0.0, ecc_anomaly_rad=0.0)
  recovered = m.state_vector_to_orbital_elements(pos, vel)
  expected_h = np.sqrt(m.EARTH_MU_KM3_S2 * a * (1 - e ** 2))
  assert recovered["angular_momentum"] == pytest.approx(expected_h, rel=1e-9)


def test_conserved_quantities_are_constant_along_orbit():
  rows = m.demo_conserved_quantities_along_orbit()
  energies = [r["specific_energy"] for r in rows]
  h_values = [r["angular_momentum"] for r in rows]
  assert max(energies) - min(energies) < 1e-6
  assert max(h_values) - min(h_values) < 1e-6


def test_vis_viva_predicts_actual_speed_exactly():
  rows = m.demo_vis_viva_speed_prediction()
  for row in rows:
    assert row["v_predicted"] == pytest.approx(row["v_actual"], rel=1e-6)


def test_perigee_speed_exceeds_apogee_speed():
  rows = m.demo_vis_viva_speed_prediction()
  perigee = next(r for r in rows if r["location"] == "근지점")
  apogee = next(r for r in rows if r["location"] == "원지점")
  assert perigee["v_actual"] > apogee["v_actual"]


def test_circular_orbit_eccentricity_recovers_near_zero():
  """이 스크립트의 핵심 정직성 지점: 원궤도 입력을 역변환하면 이심률이 0에
  가깝게 나와야 하고, 근점 편각은 임의값이 아니라 명시적으로 0이어야 한다."""
  result = m.demo_singular_case_circular_orbit_argp_undefined()
  assert result["recovered_eccentricity"] < 1e-6
  assert result["recovered_argp_deg"] == pytest.approx(0.0)


def test_state_vector_to_orbital_elements_handles_equatorial_orbit():
  """적도궤도(i=0)에서는 승교점이 정의되지 않으므로 RAAN이 예외 없이 0으로
  처리돼야 한다."""
  pos, vel = m.orbital_elements_to_state_vector(
      7000.0, 0.1, inclination_rad=0.0, raan_rad=0.0, argp_rad=np.radians(45.0),
      true_anomaly_rad=0.0, ecc_anomaly_rad=0.0)
  recovered = m.state_vector_to_orbital_elements(pos, vel)
  assert recovered["raan_rad"] == pytest.approx(0.0)
