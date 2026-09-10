"""
좌표계 변환 - ECI, ECEF, 지평좌표계(SEZ)와 방위각/고도각

01~03번은 위성의 위치를 지구 중심 관성좌표계(ECI) 기준으로 계산했다. 하지만 지상
관측자(예: 지상국)가 "저 위성이 지금 하늘 어디에 보이는가"를 알려면 ECI 좌표를
관측자 기준의 지평좌표계(방위각 Az, 고도각 El)로 바꿔야 한다. 이 변환은 지구가
자전한다는 사실(ECI -> ECEF)과 관측자의 위치(위도, 경도)를 모두 반영해야 하는,
"궤도가 아니라 관측 기하"의 문제다. 05번(지상국 가시성)은 이 스크립트의 변환을
01번의 실제 궤도 위치에 적용해 실측 접촉 창을 계산한다.

핵심 개념 1: ECI와 ECEF는 "지구 자전"만큼 차이난다
  ECI(Earth-Centered Inertial)는 별에 대해 고정된 관성좌표계이고, ECEF(Earth-Centered
  Earth-Fixed)는 지구와 함께 자전하는 좌표계다. 두 좌표계는 원점(지구 중심)과 z축
  (지구 자전축)은 같지만, x축이 지구 자전각(GST, Greenwich Sidereal Time)만큼 벌어져
  있다. 따라서 ECI -> ECEF 변환은 z축 기준 회전 하나로 충분하다:
      r_ECEF = Rz(theta_GST) * r_ECI
  (부호에 주의: ECEF가 ECI보다 자전각만큼 "뒤처져" 보이므로 +theta 회전이다.)

핵심 개념 2: 관측자 기준 지평좌표계(SEZ)는 남(South)-동(East)-천정(Zenith) 축이다
  관측자의 위치(위도 phi, 경도 lambda)를 알면, ECEF에서 관측자를 원점으로 삼는
  국소 좌표계로 다시 회전할 수 있다. SEZ 좌표는
      [S; E; Z] = Ry(phi - 90 deg) * Rz(-theta_LST) * (r_ECI - r_site_ECI)
  로 얻는다(theta_LST는 관측자의 지방항성시 = GST + 관측자 경도). 부호를 직접
  검증했다: S,E,Z 기저벡터를 위경도로 직접 쓴 표준식과 비교했을 때 흔히 인용되는
  "Ry(90-phi)*Rz(+theta)" 형태는 이 프로젝트의 회전행렬 정의(rotation_matrix_y/z가
  능동회전, "좌표계를 회전"이 아니라 "벡터를 회전"하는 관례)와 부호가 맞지
  않았다 — 회전 방향 관례 차이로 부호가 뒤집힐 수 있다는 것을 실제로 겪었다.
  이 스크립트는 검증된 부호(phi-90, -theta_LST)를 사용한다. 단순화를 위해
  관측자 위치를 이미 ECI로 표현한 뒤 위 식을 그대로 적용한다.

핵심 개념 3: 방위각과 고도각은 SEZ 벡터에서 바로 나온다
  고도각(elevation) El은 지평선 위로 얼마나 떠 있는지: El = asin(Z / |r|).
  방위각(azimuth) Az는 진북(North)에서 시계방향으로 잰 각도인데, SEZ가 남쪽 기준이라
      Az = atan2(E, -S)
  로 계산한다(atan2로 사분면을 자동 처리). 천정(머리 바로 위, El=90도)에서는 S=E=0이
  되어 atan2(0,0)이 정의되지 않으므로 이 특이 케이스를 별도로 확인한다.

핵심 개념 4: 지방항성시(LST)는 "그 시각 그 경도에서 본 하늘의 기준각"이다
  단순화 모델로 GST가 시간에 비례해 증가한다고 가정한다(GST = GST0 + omega_earth * t,
  omega_earth = 지구 자전각속도 ≈ 7.2921159e-5 rad/s). 지방항성시는
  theta_LST = GST + 관측자 경도(rad)다. 실제 정밀 계산은 세차/장동까지 고려하지만,
  이 프로젝트는 "관측 기하가 어떻게 성립하는지"를 보여주는 것이 목적이라 이
  단순화 모델로 충분하다 — 이 근사를 감추지 않고 명시한다.
"""

import argparse
import csv
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")
  sys.stderr.reconfigure(encoding="utf-8")

from orbit_math import EARTH_RADIUS_KM, rotation_matrix_y, rotation_matrix_z

EARTH_ROTATION_RATE_RAD_S = 7.2921159e-5  # 지구 자전각속도 (rad/s), 항성일 기준


def eci_to_ecef(position_eci_km, gst_rad):
  """ECI -> ECEF: z축 기준 +theta_GST 회전."""
  return rotation_matrix_z(gst_rad) @ np.array(position_eci_km, dtype=float)


def ecef_to_eci(position_ecef_km, gst_rad):
  """ECEF -> ECI: eci_to_ecef의 역변환(회전행렬은 직교행렬이라 전치=역행렬)."""
  return rotation_matrix_z(gst_rad).T @ np.array(position_ecef_km, dtype=float)


def gst_at_time(time_sec, gst0_rad=0.0):
  """단순화 모델: GST가 지구 자전각속도로 선형 증가한다고 가정."""
  return (gst0_rad + EARTH_ROTATION_RATE_RAD_S * time_sec) % (2 * np.pi)


def site_position_eci(latitude_rad, longitude_rad, gst_rad, altitude_km=0.0):
  """관측자(지상국)의 위경도로부터 ECI 위치를 구한다(구형 지구 근사).
  경도는 ECEF 기준이므로, ECI로 바꾸려면 GST만큼 회전해야 한다."""
  r = EARTH_RADIUS_KM + altitude_km
  site_ecef = r * np.array([
      np.cos(latitude_rad) * np.cos(longitude_rad),
      np.cos(latitude_rad) * np.sin(longitude_rad),
      np.sin(latitude_rad),
  ])
  return ecef_to_eci(site_ecef, gst_rad)


def eci_to_topocentric_sez(position_eci_km, site_position_eci_km, latitude_rad, lst_rad):
  """ECI 위치를 관측자 기준 SEZ(남-동-천정) 좌표로 변환.
  [S;E;Z] = Ry(phi - 90deg) * Rz(-theta_LST) * (r_ECI - r_site)"""
  relative = np.array(position_eci_km, dtype=float) - np.array(site_position_eci_km, dtype=float)
  rotation = rotation_matrix_y(latitude_rad - np.pi / 2) @ rotation_matrix_z(-lst_rad)
  return rotation @ relative


def sez_to_azimuth_elevation(sez_vector_km):
  """SEZ 벡터 -> (방위각 rad, 고도각 rad, 거리 km).
  Az = atan2(E, -S), El = asin(Z/|r|)."""
  s, e, z = sez_vector_km
  r = np.linalg.norm(sez_vector_km)
  elevation = np.arcsin(np.clip(z / r, -1.0, 1.0))
  azimuth = np.arctan2(e, -s) % (2 * np.pi)
  return azimuth, elevation, r


def eci_position_to_look_angles(position_eci_km, site_latitude_rad, site_longitude_rad, time_sec, gst0_rad=0.0):
  """위성 ECI 위치 + 관측자 위경도 + 시각 -> (방위각, 고도각, 거리). 05번이 재사용할 핵심 함수."""
  gst = gst_at_time(time_sec, gst0_rad)
  lst = (gst + site_longitude_rad) % (2 * np.pi)
  site_pos = site_position_eci(site_latitude_rad, site_longitude_rad, gst)
  sez = eci_to_topocentric_sez(position_eci_km, site_pos, site_latitude_rad, lst)
  return sez_to_azimuth_elevation(sez)


def demo_eci_ecef_roundtrip():
  """ECI -> ECEF -> ECI 왕복 변환이 원래 위치를 정확히 복원하는지 확인한다."""
  print("=" * 70)
  print("[1] ECI <-> ECEF 왕복 변환 검증 (회전행렬의 직교성)")
  print("=" * 70)
  test_positions = [
      np.array([7000.0, 0.0, 0.0]),
      np.array([4000.0, 5000.0, 2000.0]),
      np.array([-3000.0, 6000.0, -1500.0]),
  ]
  gst_values = [0.0, np.radians(45.0), np.radians(200.0)]

  rows = []
  print(f"  {'GST(도)':>10}{'원본 위치':>28}{'왕복 후 위치':>28}{'오차(km)':>12}")
  for pos, gst in zip(test_positions, gst_values):
    ecef = eci_to_ecef(pos, gst)
    recovered = ecef_to_eci(ecef, gst)
    error = np.linalg.norm(recovered - pos)
    print(f"  {np.degrees(gst):>10.1f}{np.round(pos, 1)!s:>28}{np.round(recovered, 4)!s:>28}{error:>12.2e}")
    rows.append({"gst_deg": np.degrees(gst), "position_error_km": error})
    assert error < 1e-8, "ECI<->ECEF 왕복 변환은 원래 위치를 정확히 복원해야 함(회전행렬은 직교행렬)"

  print("\n(회전행렬은 직교행렬이라 전치행렬이 곧 역행렬이다 — ECEF->ECI가 ECI->ECEF의")
  print(" 정확한 역변환이라는 것을 수치로 확인했다.)")
  return rows


def demo_zenith_case_elevation_90():
  """위성이 관측자 바로 머리 위(천정)에 있을 때 고도각이 정확히 90도가 되는지,
  그리고 이때 방위각이 수치적으로 불안정해지는 특이 케이스를 확인한다."""
  print("\n" + "=" * 70)
  print("[2] 특이 케이스: 천정(고도각 90도)에서는 방위각이 정의되지 않는다")
  print("=" * 70)
  site_lat = np.radians(35.0)
  site_lon = np.radians(129.0)
  gst = 0.0
  lst = (gst + site_lon) % (2 * np.pi)
  site_pos = site_position_eci(site_lat, site_lon, gst)

  # 관측자 바로 위 500km 지점에 위성을 둔다 (관측자 위치벡터 방향으로 연장)
  site_unit = site_pos / np.linalg.norm(site_pos)
  satellite_pos = site_pos + 500.0 * site_unit

  sez = eci_to_topocentric_sez(satellite_pos, site_pos, site_lat, lst)
  azimuth, elevation, distance = sez_to_azimuth_elevation(sez)
  print(f"관측자 위도={np.degrees(site_lat):.1f}도, 경도={np.degrees(site_lon):.1f}도")
  print(f"천정 위성까지 거리: {distance:.2f}km (예상: 500km)")
  print(f"고도각: {np.degrees(elevation):.4f}도 (예상: 90도)")
  print(f"SEZ 수평성분 (S,E): ({sez[0]:.2e}, {sez[1]:.2e}) — 0에 가까움, 방위각은 수치잡음에 좌우됨")
  print(f"계산된 방위각: {np.degrees(azimuth):.1f}도 (물리적으로 무의미 — 천정에서는 '방향'이 없음)")

  print("\n(천정에서는 지평선 위 어느 방향이랄 것도 없이 정확히 위에 있으므로, atan2(E,-S)의")
  print(" 두 입력이 모두 부동소수점 잡음 수준으로 0에 가까워져 방위각이 이론적으로 정의되지")
  print(" 않는다 — 이 스크립트는 이 값을 감추지 않고 그대로 노출한다.)")

  assert abs(distance - 500.0) < 1e-6, "천정 위성까지의 거리는 500km여야 함"
  assert abs(np.degrees(elevation) - 90.0) < 1e-4, "천정에서는 고도각이 90도에 가까워야 함"
  return {"distance_km": distance, "elevation_deg": np.degrees(elevation)}


def demo_horizon_and_below_horizon():
  """지평선 위(고도각>0)와 지평선 아래(고도각<0)를 구분해, 지구 반대편에 있는
  위성은 고도각이 음수로 나와야 한다는 것을 확인한다."""
  print("\n" + "=" * 70)
  print("[3] 지평선 위 vs 지평선 아래: 고도각의 부호로 가시성 판단")
  print("=" * 70)
  site_lat = np.radians(35.0)
  site_lon = np.radians(129.0)
  gst = 0.0
  site_pos = site_position_eci(site_lat, site_lon, gst)
  site_unit = site_pos / np.linalg.norm(site_pos)

  # 관측자 방향(천정 근처)으로 살짝 벗어난 지점 -> 지평선 위(고도각 양수)여야 함
  above_horizon_pos = site_pos + 1000.0 * site_unit + np.array([200.0, 100.0, 50.0])
  below_horizon_pos = -np.array([EARTH_RADIUS_KM + 1000.0, 500.0, 2000.0])  # 지구 반대편 방향

  rows = []
  for label, pos in [("천정 근처(관측자 방향)", above_horizon_pos), ("지구 반대편", below_horizon_pos)]:
    az, el, dist = eci_position_to_look_angles(pos, site_lat, site_lon, time_sec=0.0)
    print(f"[{label}] 방위각={np.degrees(az):.1f}도, 고도각={np.degrees(el):.1f}도, 거리={dist:.1f}km")
    rows.append({"label": label, "azimuth_deg": np.degrees(az), "elevation_deg": np.degrees(el), "distance_km": dist})

  assert rows[0]["elevation_deg"] > 0, "천정 근처 위성은 고도각이 양수(지평선 위)여야 함"
  assert rows[1]["elevation_deg"] < 0, "지구 반대편 위성은 고도각이 음수(지평선 아래)여야 함"
  print("\n(천정 근처 위성은 고도각이 양수, 지구 반대편 위성은 음수로 나온다 — 05번은 이")
  print(" 고도각이 최소 기준(예: 10도)을 넘는 구간만 실측 접촉 창으로 센다.)")
  return rows


def parse_args():
  parser = argparse.ArgumentParser(description="ECI/ECEF/지평좌표계(SEZ) 변환과 방위각/고도각 계산")
  parser.add_argument("--latitude-deg", type=float, default=35.0, help="관측자 위도(도), 기본값: 35.0")
  parser.add_argument("--longitude-deg", type=float, default=129.0, help="관측자 경도(도), 기본값: 129.0")
  return parser.parse_args()


def main():
  args = parse_args()

  roundtrip_rows = demo_eci_ecef_roundtrip()
  zenith_case = demo_zenith_case_elevation_90()
  horizon_rows = demo_horizon_and_below_horizon()

  results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
  os.makedirs(results_dir, exist_ok=True)

  roundtrip_csv = os.path.join(results_dir, "eci_ecef_roundtrip.csv")
  with open(roundtrip_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["gst_deg", "position_error_km"])
    for row in roundtrip_rows:
      writer.writerow([f"{row['gst_deg']:.2f}", f"{row['position_error_km']:.2e}"])
  print(f"\n[기록] ECI<->ECEF 왕복 변환 결과 저장됨 → {roundtrip_csv}")

  zenith_csv = os.path.join(results_dir, "zenith_case.csv")
  with open(zenith_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["distance_km", "elevation_deg"])
    writer.writerow([f"{zenith_case['distance_km']:.4f}", f"{zenith_case['elevation_deg']:.4f}"])
  print(f"[기록] 천정 특이 케이스 결과 저장됨 → {zenith_csv}")

  horizon_csv = os.path.join(results_dir, "horizon_cases.csv")
  with open(horizon_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "azimuth_deg", "elevation_deg", "distance_km"])
    for row in horizon_rows:
      writer.writerow([row["label"], f"{row['azimuth_deg']:.2f}", f"{row['elevation_deg']:.2f}", f"{row['distance_km']:.1f}"])
  print(f"[기록] 지평선 위/아래 비교 결과 저장됨 → {horizon_csv}")

  print(f"\n(참고: --latitude-deg={args.latitude_deg}, --longitude-deg={args.longitude_deg}는 "
        f"05번 지상국 가시성 스크립트에서 실제로 사용된다.)")


if __name__ == "__main__":
  main()
