"""04_Coordinate_frame_transforms.py 검증: ECI<->ECEF 왕복 변환의 정확성, 천정에서
고도각이 정확히 90도가 되는지, 지구 반대편 위성의 고도각이 음수인지, 그리고 방위각/
고도각 계산이 알려진 특수 케이스에서 기대한 값을 내는지."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("frames/04_Coordinate_frame_transforms.py")


def test_eci_to_ecef_zero_gst_is_identity():
  position = np.array([7000.0, 1000.0, 2000.0])
  ecef = m.eci_to_ecef(position, gst_rad=0.0)
  assert ecef == pytest.approx(position)


def test_eci_ecef_roundtrip_recovers_original_position():
  position = np.array([4000.0, 5000.0, 2000.0])
  gst = np.radians(73.0)
  ecef = m.eci_to_ecef(position, gst)
  recovered = m.ecef_to_eci(ecef, gst)
  assert recovered == pytest.approx(position, abs=1e-9)


def test_eci_to_ecef_preserves_vector_magnitude():
  """회전은 벡터 길이를 바꾸지 않아야 한다(회전행렬은 직교행렬)."""
  position = np.array([3000.0, -4000.0, 5000.0])
  ecef = m.eci_to_ecef(position, gst_rad=np.radians(150.0))
  assert np.linalg.norm(ecef) == pytest.approx(np.linalg.norm(position))


def test_gst_at_time_increases_linearly():
  gst_early = m.gst_at_time(0.0)
  gst_later = m.gst_at_time(3600.0)
  expected = (m.EARTH_ROTATION_RATE_RAD_S * 3600.0) % (2 * np.pi)
  assert gst_later == pytest.approx(expected)
  assert gst_early == pytest.approx(0.0)


def test_site_position_eci_at_equator_prime_meridian_lies_on_x_axis():
  """적도(위도 0), 본초자오선(경도 0), GST=0이면 관측자는 x축 위에 있어야 한다."""
  site = m.site_position_eci(latitude_rad=0.0, longitude_rad=0.0, gst_rad=0.0)
  assert site[0] == pytest.approx(m.EARTH_RADIUS_KM)
  assert site[1] == pytest.approx(0.0, abs=1e-9)
  assert site[2] == pytest.approx(0.0, abs=1e-9)


def test_site_position_eci_at_north_pole_lies_on_z_axis():
  site = m.site_position_eci(latitude_rad=np.pi / 2, longitude_rad=0.0, gst_rad=0.0)
  assert site[2] == pytest.approx(m.EARTH_RADIUS_KM)
  assert np.linalg.norm(site[:2]) == pytest.approx(0.0, abs=1e-9)


def test_zenith_satellite_has_elevation_90_degrees():
  """관측자 바로 위에 있는 위성은 고도각이 정확히 90도여야 한다 — 이 스크립트의
  핵심 검증(SEZ 회전 부호를 실제로 겪은 버그를 통해 고정했다)."""
  lat = np.radians(35.0)
  lon = np.radians(129.0)
  gst = 0.0
  site = m.site_position_eci(lat, lon, gst)
  site_unit = site / np.linalg.norm(site)
  satellite = site + 500.0 * site_unit
  _az, el, dist = m.eci_position_to_look_angles(satellite, lat, lon, time_sec=0.0)
  assert np.degrees(el) == pytest.approx(90.0, abs=1e-4)
  assert dist == pytest.approx(500.0, abs=1e-6)


def test_satellite_on_opposite_side_of_earth_has_negative_elevation():
  lat = np.radians(35.0)
  lon = np.radians(129.0)
  site = m.site_position_eci(lat, lon, gst_rad=0.0)
  opposite_direction_satellite = -site / np.linalg.norm(site) * (m.EARTH_RADIUS_KM + 1000.0)
  _az, el, _dist = m.eci_position_to_look_angles(opposite_direction_satellite, lat, lon, time_sec=0.0)
  assert el < 0


def test_sez_to_azimuth_elevation_due_south_on_horizon():
  """SEZ에서 S축 방향(고도각 0, 정남쪽)이면 방위각은 180도여야 한다(atan2(0,-S), S>0)."""
  sez = np.array([100.0, 0.0, 0.0])
  az, el, r = m.sez_to_azimuth_elevation(sez)
  assert np.degrees(az) == pytest.approx(180.0)
  assert np.degrees(el) == pytest.approx(0.0)
  assert r == pytest.approx(100.0)


def test_sez_to_azimuth_elevation_due_north_on_horizon():
  """SEZ에서 -S축 방향(정북쪽)이면 방위각은 0(=360)도여야 한다."""
  sez = np.array([-100.0, 0.0, 0.0])
  az, _el, _r = m.sez_to_azimuth_elevation(sez)
  assert np.degrees(az) % 360 == pytest.approx(0.0, abs=1e-6)


def test_sez_to_azimuth_elevation_due_east_on_horizon():
  sez = np.array([0.0, 100.0, 0.0])
  az, _el, _r = m.sez_to_azimuth_elevation(sez)
  assert np.degrees(az) == pytest.approx(90.0)
