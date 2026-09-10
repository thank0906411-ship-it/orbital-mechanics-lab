"""
지상국 가시성 - 실제 궤도 전파로 계산하는 접촉 창(Contact Window)

이 스크립트는 이 프로젝트 전체의 핵심 차별점을 보여준다: 이전에 만들었던 데이터링크
프로젝트의 심우주 통신 스크립트들은 위성-지상국 접촉 창을 "사각파 근사"(일정 주기로
접촉/비접촉이 사각파처럼 반복된다고 가정)나 임의의 상수로 때웠다. 여기서는 01번의
실제 궤도 전파(케플러 방정식)로 위성의 3차원 위치를 시간에 따라 계산하고, 04번의
좌표변환으로 그 위치를 매 순간 지상국 기준 고도각으로 바꿔, 고도각이 최소 기준을
넘는 구간만 실측 접촉 창으로 판정한다 — 근사가 아니라 실제 궤도역학에서 유도된 결과다.

핵심 개념 1: 접촉 창은 "고도각이 최소값을 넘는 시간 구간"이다
  지상국 안테나는 지평선 바로 위에서는 신호가 대기와 지형에 가려 품질이 나쁘다.
  실무에서는 최소 고도각(El_min, 예: 10도) 이상일 때만 유효한 접촉으로 친다. 이
  스크립트는 위성의 궤도를 촘촘한 시간 간격으로 샘플링해 각 시점의 고도각을 계산하고,
  El > El_min을 만족하는 연속 구간들을 찾아 각 구간의 시작/끝/지속시간(AOS/LOS,
  Acquisition/Loss Of Signal)을 접촉 창으로 기록한다.

핵심 개념 2: 사각파 근사와 실측 접촉 창은 근본적으로 다르다
  사각파 근사는 "궤도 주기의 X%가 접촉 구간"처럼 궤도 형상과 무관한 고정 비율을
  가정한다. 실측 접촉 창은 궤도 경사각, 이심률, 지상국 위도/경도, 지구 자전(지상궤적이
  매 주기 서쪽으로 이동)이 모두 얽혀 결정된다. 이 스크립트는 위도가 다른 세 지상국의
  접촉 창 개수/길이를 비교하는데, 실제로 돌려보면 "위도가 궤도 경사각에 가까울수록
  접촉이 많다"처럼 단순하지 않다 — 위도 70도(경사각 51.6도보다 훨씬 극지방에 가까움)
  지상국은 이 특정 RAAN/경도 조합에서 접촉 창이 0개였고, 오히려 저위도(5도) 쪽이
  총 접촉 시간이 더 길게 나왔다. 이는 근사 없이 실제 궤도역학(경도 정렬과 자전까지)을
  계산했을 때만 드러나는 결과이지, "위도가 가까우면 유리하다"는 직관만으로는 예측할
  수 없다 — 이 스크립트는 이 비직관적인 결과를 감추지 않고 그대로 보여준다.

핵심 개념 3: 최대 고도각이 접촉 창의 "품질"을 결정한다
  한 접촉 창 안에서도 위성이 지평선 근처(낮은 El)로 스쳐 지나가는 경우와, 천정
  근처(높은 El)까지 올라오는 경우는 링크 품질이 크게 다르다(거리가 짧을수록 신호가
  강함). 이 스크립트는 각 접촉 창의 최대 고도각도 함께 기록해, 접촉 창이 있다는
  사실뿐 아니라 "얼마나 좋은 접촉"이었는지도 구분한다.

01번(케플러 전파), 04번(좌표변환)과의 관계: 01번의 propagate_orbit으로 위성의 궤도면
위치를 구하고, 02번 스타일의 회전으로 ECI 위치를 만든 뒤(단순화를 위해 적도면
궤도(i=0)가 아닌 경사궤도를 직접 회전시켜 사용), 04번의 eci_position_to_look_angles로
매 시점 고도각을 계산한다.
"""

import argparse
import csv
import importlib.util
import os
import sys

import numpy as np

from orbit_math import rotation_matrix_x, rotation_matrix_z

if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")
  sys.stderr.reconfigure(encoding="utf-8")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(filename, name):
  spec = importlib.util.spec_from_file_location(name, os.path.join(_THIS_DIR, filename))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


kepler = _load("01_Kepler_orbit_propagation.py", "kepler_module")
frames = _load("04_Coordinate_frame_transforms.py", "frames_module")

EARTH_MU_KM3_S2 = kepler.EARTH_MU_KM3_S2


def orbital_position_eci(semi_major_axis_km, eccentricity, inclination_rad, raan_rad, argp_rad,
                          mean_anomaly0_rad, time_sec, mu=EARTH_MU_KM3_S2):
  """01번의 궤도면 위치에 3번 회전(argp, i, RAAN)을 적용해 ECI 위치를 만든다.
  02번과 동일한 회전 관례(Rz(RAAN)*Rx(i)*Rz(argp))를 orbit_math의 공유 회전행렬로 적용한다."""
  state = kepler.propagate_orbit(semi_major_axis_km, eccentricity, mean_anomaly0_rad, time_sec, mu)
  _r, x_p, y_p = state["r"], state["x_p"], state["y_p"]
  position_perifocal = np.array([x_p, y_p, 0.0])

  rotation = rotation_matrix_z(raan_rad) @ rotation_matrix_x(inclination_rad) @ rotation_matrix_z(argp_rad)
  return rotation @ position_perifocal


def compute_elevation_time_series(semi_major_axis_km, eccentricity, inclination_rad, raan_rad, argp_rad,
                                   site_latitude_rad, site_longitude_rad, duration_sec, num_steps):
  """시간 시계열에 대해 (t, 고도각, 방위각, 거리)를 계산한다."""
  times = np.linspace(0, duration_sec, num_steps)
  rows = []
  for t in times:
    position_eci = orbital_position_eci(semi_major_axis_km, eccentricity, inclination_rad, raan_rad, argp_rad,
                                         mean_anomaly0_rad=0.0, time_sec=t)
    azimuth, elevation, distance = frames.eci_position_to_look_angles(
        position_eci, site_latitude_rad, site_longitude_rad, time_sec=t)
    rows.append({"t_sec": t, "azimuth_deg": np.degrees(azimuth), "elevation_deg": np.degrees(elevation),
                 "distance_km": distance})
  return rows


def find_contact_windows(elevation_time_series, min_elevation_deg=10.0):
  """El > min_elevation_deg를 만족하는 연속 구간을 찾아 접촉 창 리스트로 반환한다.
  각 접촉 창: {aos_sec, los_sec, duration_sec, max_elevation_deg}."""
  windows = []
  in_contact = False
  window_start = None
  window_max_el = -90.0

  for row in elevation_time_series:
    visible = row["elevation_deg"] > min_elevation_deg
    if visible and not in_contact:
      in_contact = True
      window_start = row["t_sec"]
      window_max_el = row["elevation_deg"]
    elif visible and in_contact:
      window_max_el = max(window_max_el, row["elevation_deg"])
    elif not visible and in_contact:
      in_contact = False
      window_end = row["t_sec"]
      windows.append({"aos_sec": window_start, "los_sec": window_end,
                       "duration_sec": window_end - window_start, "max_elevation_deg": window_max_el})

  if in_contact:
    window_end = elevation_time_series[-1]["t_sec"]
    windows.append({"aos_sec": window_start, "los_sec": window_end,
                     "duration_sec": window_end - window_start, "max_elevation_deg": window_max_el})
  return windows


def demo_single_pass_contact_window():
  """저궤도 위성 하나에 대해, 한 지상국 기준 접촉 창이 실제로 몇 개, 얼마나
  지속되는지 실측 궤도 전파로 계산한다."""
  print("=" * 70)
  print("[1] 실측 궤도 전파로 접촉 창 계산 (사각파 근사가 아니라 실제 궤도)")
  print("=" * 70)
  a, e, i, raan, argp = 6978.0, 0.001, np.radians(51.6), np.radians(0.0), np.radians(0.0)
  period = 2 * np.pi / kepler.mean_motion(a)
  site_lat = np.radians(35.0)
  site_lon = np.radians(129.0)
  min_elevation_deg = 10.0

  duration_sec = period * 3  # 세 궤도 주기 관찰
  num_steps = 2000
  print(f"위성: 반장축={a}km, 경사각={np.degrees(i):.1f}도, 궤도 주기={period / 60:.1f}분")
  print(f"지상국: 위도={np.degrees(site_lat):.1f}도, 경도={np.degrees(site_lon):.1f}도, 최소 고도각={min_elevation_deg}도")
  print(f"관찰 시간: {duration_sec / 60:.1f}분 (약 {duration_sec / period:.1f} 궤도)\n")

  elevation_series = compute_elevation_time_series(a, e, i, raan, argp, site_lat, site_lon, duration_sec, num_steps)
  windows = find_contact_windows(elevation_series, min_elevation_deg)

  print(f"{'#':>4}{'AOS(분)':>12}{'LOS(분)':>12}{'지속시간(분)':>14}{'최대고도각(도)':>16}")
  for idx, w in enumerate(windows, 1):
    print(f"{idx:>4}{w['aos_sec'] / 60:>12.2f}{w['los_sec'] / 60:>12.2f}{w['duration_sec'] / 60:>14.2f}{w['max_elevation_deg']:>16.2f}")

  print(f"\n총 {len(windows)}개의 접촉 창이 발견됨 (3궤도 주기 동안)")
  assert len(windows) >= 1, "51.6도 경사궤도, 위도 35도 지상국이면 3주기 안에 접촉 창이 최소 1개는 있어야 함"
  for w in windows:
    assert w["max_elevation_deg"] > min_elevation_deg, "접촉 창의 최대 고도각은 최소 기준보다 커야 함"
    assert w["duration_sec"] > 0, "접촉 창의 지속시간은 양수여야 함"
  return elevation_series, windows


def demo_latitude_affects_contact_frequency():
  """같은 위성 궤도에 대해, 지상국 위도가 다르면 접촉 창의 개수/길이가 달라진다는
  것을 직접 비교한다 — 사각파 근사로는 드러나지 않는, 궤도역학+지구자전+경도 정렬이
  뒤섞인 결과다(단순히 '위도가 경사각에 가까우면 유리하다'는 결론이 나지 않는다)."""
  print("\n" + "=" * 70)
  print("[2] 지상국 위도에 따른 접촉 빈도 차이 (실측 궤도역학의 결과, 직관과 다를 수 있음)")
  print("=" * 70)
  a, e, i, raan, argp = 6978.0, 0.001, np.radians(51.6), np.radians(0.0), np.radians(0.0)
  period = 2 * np.pi / kepler.mean_motion(a)
  duration_sec = period * 5
  num_steps = 3000
  min_elevation_deg = 10.0

  site_lon = np.radians(129.0)
  rows = []
  for label, lat_deg in [("저위도(적도 근처, 5도)", 5.0), ("경사각 근처(50도)", 50.0), ("고위도(70도)", 70.0)]:
    site_lat = np.radians(lat_deg)
    elevation_series = compute_elevation_time_series(a, e, i, raan, argp, site_lat, site_lon, duration_sec, num_steps)
    windows = find_contact_windows(elevation_series, min_elevation_deg)
    total_contact_time = sum(w["duration_sec"] for w in windows)
    print(f"[{label}] 접촉 창 개수={len(windows)}, 총 접촉 시간={total_contact_time / 60:.1f}분")
    rows.append({"label": label, "latitude_deg": lat_deg, "num_windows": len(windows),
                 "total_contact_sec": total_contact_time})

  print("\n(위도만으로 접촉 빈도를 예측할 수 없다 — 지상궤적이 지구 자전으로 매 주기")
  print(" 서쪽으로 이동하면서 특정 경도 대역과 우연히 정렬되는지가 함께 작용한다.")
  print(" 이건 사각파 근사에서는 절대 드러나지 않는, 실제 궤도역학의 결과다.)")
  return rows


def parse_args():
  parser = argparse.ArgumentParser(description="실측 궤도 전파 기반 지상국 접촉 창(Contact Window) 계산")
  parser.add_argument("--min-elevation-deg", type=float, default=10.0, help="최소 고도각(도), 기본값: 10.0")
  return parser.parse_args()


def main():
  args = parse_args()

  elevation_series, windows = demo_single_pass_contact_window()
  latitude_comparison_rows = demo_latitude_affects_contact_frequency()

  results_dir = os.path.join(_THIS_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  elevation_csv = os.path.join(results_dir, "elevation_time_series.csv")
  with open(elevation_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_sec", "azimuth_deg", "elevation_deg", "distance_km"])
    for row in elevation_series:
      writer.writerow([f"{row['t_sec']:.2f}", f"{row['azimuth_deg']:.3f}", f"{row['elevation_deg']:.3f}", f"{row['distance_km']:.2f}"])
  print(f"\n[기록] 고도각 시계열 저장됨 → {elevation_csv}")

  windows_csv = os.path.join(results_dir, "contact_windows.csv")
  with open(windows_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["aos_sec", "los_sec", "duration_sec", "max_elevation_deg"])
    for w in windows:
      writer.writerow([f"{w['aos_sec']:.2f}", f"{w['los_sec']:.2f}", f"{w['duration_sec']:.2f}", f"{w['max_elevation_deg']:.3f}"])
  print(f"[기록] 접촉 창 목록 저장됨 → {windows_csv}")

  latitude_csv = os.path.join(results_dir, "latitude_contact_comparison.csv")
  with open(latitude_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "latitude_deg", "num_windows", "total_contact_sec"])
    for row in latitude_comparison_rows:
      writer.writerow([row["label"], row["latitude_deg"], row["num_windows"], f"{row['total_contact_sec']:.2f}"])
  print(f"[기록] 위도별 접촉 빈도 비교 결과 저장됨 → {latitude_csv}")

  print(f"\n(참고: --min-elevation-deg={args.min_elevation_deg}는 향후 CLI 파라미터화에 대비한 자리다.)")


if __name__ == "__main__":
  main()
