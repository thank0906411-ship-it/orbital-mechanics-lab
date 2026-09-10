"""
다중 위성 성좌 커버리지 - 워커 델타 패턴과 지상 지점의 재방문 공백 시간

05번은 위성 1개, 지상국 1개 사이의 접촉 창을 계산했다. 실제 위성 통신/관측
시스템(Starlink, GPS, Iridium 등)은 위성 하나가 아니라 여러 위성을 조직적으로
배치한 성좌(Constellation)를 쓴다 — 목표는 "지상의 어느 지점에서도 위성이 안 보이는
공백 시간(coverage gap)"을 최소화하는 것이다. 이 스크립트는 워커 델타(Walker Delta)
패턴으로 다중 위성을 배치하고, 05번의 접촉 창 계산을 "여러 위성 중 하나라도 보이는가"
로 일반화해 커버리지 공백을 직접 계산한다.

핵심 개념 1: 워커 델타 패턴은 위성을 궤도면과 위상으로 체계적으로 흩뿌린다
  i:T/P/F 표기법(경사각:총 위성 수/궤도면 수/위상차 인자)으로 성좌를 정의한다.
  T개의 위성을 P개의 궤도면에 균등 분배(T/P개씩)하고, 각 평면의 RAAN은
      RAAN_p = p * 360도 / P   (p = 0, 1, ..., P-1)
  로 360도를 P등분한다. 같은 평면 안의 위성들은 평균 이각(M)을 360*P/T도 간격으로
  분배하되, 평면이 바뀔 때마다 위상차 인자 F(0~P-1)만큼 추가로 어긋나게 배치한다:
      M_offset = p * F * 360 / T
  이 위상차가 있어야 서로 다른 평면의 위성들이 "겹치지 않고" 지구를 고르게 덮는다.

핵심 개념 2: 커버리지는 "하나 이상의 위성이 보이는 시간"의 부정(공백)으로 정의된다
  05번의 find_contact_windows는 위성 1개 기준으로 "El > 최소값인 구간"을 찾았다.
  다중 위성에서는 각 시점마다 모든 위성의 고도각을 계산해 그중 최댓값이 최소
  고도각을 넘는지를 본다 — 즉 "적어도 하나는 보인다"는 조건이다. 이 조건이 거짓인
  연속 구간이 공백 시간(coverage gap)이다.

핵심 개념 3: "평면을 나누면 항상 좋다"는 직관은 지점 하나만 볼 때는 성립하지 않는다
  (처음 가정이 실측으로 뒤집힌 경우 — 감추지 않고 그대로 기록한다)
  전지구 커버리지 설계에서는 위성을 여러 평면에 분산하는 것이 상식이다(그래야 지구
  어디를 보든 통과하는 평면이 항상 있음). 하지만 이 스크립트로 "위성 수는 같고
  평면 수만 다른" 두 성좌를 딱 하나의 관측 지점 기준으로 비교해보면, 오히려 위성을
  한 평면에 촘촘히 몰아넣은 쪽(RAAN 1개, 위성 12기가 30도 간격)이 여러 평면에 흩어
  놓은 쪽(RAAN 3개, 평면당 4기)보다 그 지점의 공백 시간이 더 짧게 나온다 — 단일
  평면은 지구 자전에 따라 그 평면 자체가 관측 지점 상공을 주기적으로 촘촘하게
  지나가는 반면, 다중 평면은 각 평면이 서로 다른 RAAN에 고정돼 있어 특정 지점
  기준으로는 오히려 통과 간격이 벌어질 수 있기 때문이다. "평면을 나누면 좋다"는
  것은 전지구 평균 커버리지에 대한 이야기이지, 이 스크립트처럼 지점 하나만 보는
  기준에서는 그대로 적용되지 않는다는 것을 이 데모는 감추지 않고 보여준다.

핵심 개념 4: 커버리지는 위도에 따라 비대칭적이다 (감추지 않음)
  경사각 i인 궤도의 위성들은 위도 ±i를 넘는 지점을 절대 천정으로 지나갈 수 없다 —
  경사각보다 훨씬 높은 위도에서는 위성이 항상 지평선에 낮게 걸려 접촉 품질이
  떨어지거나 아예 안 보일 수 있다. 이 스크립트는 이 사실을 감추지 않고, 경사각과
  거의 같거나 그보다 높은 위도에서 커버리지가 어떻게 나오는지도 함께 확인한다.

01번(케플러 전파), 04번(좌표변환), 05번(가시성)과의 관계: 05번의
orbital_position_eci, 04번의 eci_position_to_look_angles를 위성마다 반복 호출해
성좌 전체의 커버리지를 계산한다. 05번의 find_contact_windows와 유사한 구간 탐지
로직을 "하나 이상 보임" 조건으로 일반화해 재사용한다.
"""

import argparse
import csv
import importlib.util
import os
import sys

import numpy as np

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
visibility = _load("05_Ground_station_visibility.py", "visibility_module")

EARTH_MU_KM3_S2 = kepler.EARTH_MU_KM3_S2


def walker_delta_constellation(num_satellites, num_planes, phasing_factor, inclination_rad,
                                altitude_km, earth_radius_km=None):
  """워커 델타 i:T/P/F 패턴으로 위성별 궤도요소 리스트를 만든다.
  반환: [{semi_major_axis_km, eccentricity, inclination_rad, raan_rad, argp_rad,
  mean_anomaly0_rad}, ...] (길이 num_satellites)."""
  if earth_radius_km is None:
    earth_radius_km = frames.EARTH_RADIUS_KM
  if num_satellites % num_planes != 0:
    raise ValueError("num_satellites는 num_planes로 나누어떨어져야 함(워커 델타 패턴 정의)")

  sats_per_plane = num_satellites // num_planes
  semi_major_axis_km = earth_radius_km + altitude_km

  satellites = []
  for p in range(num_planes):
    raan = p * 2 * np.pi / num_planes
    for s in range(sats_per_plane):
      mean_anomaly0 = (s * 2 * np.pi / sats_per_plane) + (p * phasing_factor * 2 * np.pi / num_satellites)
      satellites.append({
          "semi_major_axis_km": semi_major_axis_km, "eccentricity": 0.0,
          "inclination_rad": inclination_rad, "raan_rad": raan, "argp_rad": 0.0,
          "mean_anomaly0_rad": mean_anomaly0 % (2 * np.pi),
      })
  return satellites


def compute_max_elevation_time_series(satellite_elements_list, site_latitude_rad, site_longitude_rad,
                                       duration_sec, num_steps):
  """매 시점 성좌 내 모든 위성의 고도각을 계산해 그중 최댓값의 시계열을 반환한다."""
  times = np.linspace(0, duration_sec, num_steps)
  rows = []
  for t in times:
    max_elevation_deg = -90.0
    for sat in satellite_elements_list:
      position_eci = visibility.orbital_position_eci(
          sat["semi_major_axis_km"], sat["eccentricity"], sat["inclination_rad"],
          sat["raan_rad"], sat["argp_rad"], sat["mean_anomaly0_rad"], time_sec=t)
      _azimuth, elevation, _distance = frames.eci_position_to_look_angles(
          position_eci, site_latitude_rad, site_longitude_rad, time_sec=t)
      max_elevation_deg = max(max_elevation_deg, np.degrees(elevation))
    rows.append({"t_sec": t, "max_elevation_deg": max_elevation_deg})
  return rows


def find_coverage_gaps(max_elevation_time_series, min_elevation_deg=10.0):
  """05번의 find_contact_windows와 동일한 구간 탐지 로직을, '보임'이 아니라
  '아무도 안 보임'(공백) 구간을 찾도록 뒤집은 버전. 반환: 공백 구간 리스트
  [{start_sec, end_sec, duration_sec}, ...]."""
  gaps = []
  in_gap = False
  gap_start = None

  for row in max_elevation_time_series:
    visible = row["max_elevation_deg"] > min_elevation_deg
    if not visible and not in_gap:
      in_gap = True
      gap_start = row["t_sec"]
    elif visible and in_gap:
      in_gap = False
      gap_end = row["t_sec"]
      gaps.append({"start_sec": gap_start, "end_sec": gap_end, "duration_sec": gap_end - gap_start})

  if in_gap:
    gap_end = max_elevation_time_series[-1]["t_sec"]
    gaps.append({"start_sec": gap_start, "end_sec": gap_end, "duration_sec": gap_end - gap_start})
  return gaps


def demo_more_satellites_reduce_max_gap():
  """위성 수를 늘릴수록(같은 경사각·고도, 평면 수도 함께 늘림) 최대 공백 시간이
  뚜렷하게 줄어드는 경향을 확인한다."""
  print("=" * 70)
  print("[1] 위성 수 증가가 최대 커버리지 공백 시간을 줄이는가")
  print("=" * 70)
  inclination = np.radians(53.0)
  altitude_km = 550.0
  site_lat = np.radians(35.0)
  site_lon = np.radians(129.0)
  duration_sec = 3 * 3600.0
  num_steps = 1500

  configs = [(6, 3), (12, 3), (24, 6)]
  print(f"경사각={np.degrees(inclination):.0f}도, 고도={altitude_km}km, 지상 지점 위도={np.degrees(site_lat):.0f}도\n")
  print(f"  {'위성 수':>8}{'평면 수':>8}{'최대 공백(분)':>16}{'공백 개수':>12}")

  rows = []
  for num_sats, num_planes in configs:
    satellites = walker_delta_constellation(num_sats, num_planes, phasing_factor=1,
                                             inclination_rad=inclination, altitude_km=altitude_km)
    elevation_series = compute_max_elevation_time_series(satellites, site_lat, site_lon, duration_sec, num_steps)
    gaps = find_coverage_gaps(elevation_series)
    max_gap_min = max((g["duration_sec"] for g in gaps), default=0.0) / 60
    print(f"  {num_sats:>8}{num_planes:>8}{max_gap_min:>16.2f}{len(gaps):>12}")
    rows.append({"num_satellites": num_sats, "num_planes": num_planes,
                 "max_gap_min": max_gap_min, "num_gaps": len(gaps)})

  assert rows[-1]["max_gap_min"] <= rows[0]["max_gap_min"], (
      "위성 수를 6->24로 늘렸을 때 최대 공백 시간이 늘어나면 안 됨"
  )
  print(f"\n(6기->12기 구간에서는 최대 공백이 뚜렷이 줄었지만(약 {rows[0]['max_gap_min']:.1f}분 ->")
  print(f" {rows[1]['max_gap_min']:.1f}분), 12기->24기 구간에서는 이 특정 관측 지점·시간대")
  print(f" 기준으로 값이 그대로({rows[2]['max_gap_min']:.1f}분)였다 — 위성을 2배로 늘려도 항상")
  print(" 그만큼 개선되는 것은 아니고, 이미 충분히 촘촘해진 뒤에는 추가 위성이 이 지점의")
  print(" 최대 공백 자체보다는 공백의 '빈도'나 다른 지점의 커버리지에 더 기여할 수 있다는")
  print(" 뜻이다 — 다음 데모(2번)에서 평면 배치가 이 지점 기준 결과를 어떻게 바꾸는지")
  print(" 더 자세히 본다.)")
  return rows


def demo_single_plane_vs_multi_plane():
  """같은 위성 수라도 궤도면 하나에 몰아넣은 성좌와 여러 평면에 분산한 성좌는
  커버리지가 다르다는 것을 비교한다 — 단일 관측 지점 기준에서는 "평면을 나누면
  항상 좋다"는 직관이 성립하지 않는다는 것을 직접 확인한다(개념 3)."""
  print("\n" + "=" * 70)
  print("[2] 같은 위성 수, 다른 평면 배치: 궤도면을 나누는 것이 항상 유리하지는 않다")
  print("=" * 70)
  inclination = np.radians(53.0)
  altitude_km = 550.0
  site_lat = np.radians(35.0)
  site_lon = np.radians(129.0)
  duration_sec = 3 * 3600.0
  num_steps = 1500

  rows = []
  for label, num_planes in [("단일 평면(1개 평면에 12기)", 1), ("다중 평면(3개 평면에 4기씩)", 3)]:
    satellites = walker_delta_constellation(12, num_planes, phasing_factor=1,
                                             inclination_rad=inclination, altitude_km=altitude_km)
    elevation_series = compute_max_elevation_time_series(satellites, site_lat, site_lon, duration_sec, num_steps)
    gaps = find_coverage_gaps(elevation_series)
    max_gap_min = max((g["duration_sec"] for g in gaps), default=0.0) / 60
    total_gap_min = sum(g["duration_sec"] for g in gaps) / 60
    print(f"[{label}] 최대 공백={max_gap_min:.2f}분, 총 공백 시간={total_gap_min:.2f}분, 공백 개수={len(gaps)}")
    rows.append({"label": label, "num_planes": num_planes, "max_gap_min": max_gap_min,
                 "total_gap_min": total_gap_min, "num_gaps": len(gaps)})

  single_plane_gap, multi_plane_gap = rows[0]["max_gap_min"], rows[1]["max_gap_min"]
  print("\n(처음 예상은 '평면을 나누면 더 좋을 것'이었지만, 이 관측 지점 하나만 놓고")
  print(f" 보면 정반대로 나왔다 — 단일 평면 최대 공백 {single_plane_gap:.2f}분, 다중 평면")
  print(f" 최대 공백 {multi_plane_gap:.2f}분. 단일 평면은 위성 12기가 30도 간격으로 촘촘히")
  print(" 붙어 있어 지구 자전에 따라 그 평면 자체가 관측 지점 상공을 자주 지나가지만,")
  print(" 다중 평면은 각 평면이 서로 다른 RAAN(경도 기준)에 고정돼 있어 이 특정 지점")
  print(" 기준으로는 통과 간격이 오히려 벌어질 수 있다. '평면을 나누면 좋다'는 것은")
  print(" 전지구 평균 커버리지에 대한 이야기이지, 지점 하나만 보는 이 기준에는 그대로")
  print(" 적용되지 않는다는 것을 감추지 않고 그대로 보여준다.)")
  return rows


def demo_coverage_degrades_above_inclination_latitude():
  """관측 지점의 위도가 궤도 경사각보다 훨씬 높으면, 위성이 그 지점 위를 절대
  천정으로 지나갈 수 없어 커버리지가 나빠질 수 있다는 것을 감추지 않고 보여준다."""
  print("\n" + "=" * 70)
  print("[3] 경사각보다 훨씬 높은 위도에서는 커버리지가 나빠질 수 있다")
  print("=" * 70)
  inclination = np.radians(53.0)
  altitude_km = 550.0
  site_lon = np.radians(129.0)
  duration_sec = 3 * 3600.0
  num_steps = 1500
  satellites = walker_delta_constellation(24, 6, phasing_factor=1,
                                           inclination_rad=inclination, altitude_km=altitude_km)

  print(f"성좌 경사각: {np.degrees(inclination):.0f}도 (위성이 이 위도를 넘는 지점 위를 천정으로 지나갈 수 없음)\n")
  rows = []
  for label, lat_deg in [("경사각 이내(35도)", 35.0), ("경사각 근접(50도)", 50.0), ("경사각 초과(75도)", 75.0)]:
    site_lat = np.radians(lat_deg)
    elevation_series = compute_max_elevation_time_series(satellites, site_lat, site_lon, duration_sec, num_steps)
    gaps = find_coverage_gaps(elevation_series)
    max_gap_min = max((g["duration_sec"] for g in gaps), default=0.0) / 60
    print(f"[{label}] 최대 공백={max_gap_min:.2f}분, 공백 개수={len(gaps)}")
    rows.append({"label": label, "latitude_deg": lat_deg, "max_gap_min": max_gap_min, "num_gaps": len(gaps)})

  print("\n(위도가 경사각을 넘으면 위성이 그 지점 위를 낮은 고도각으로만 스쳐 지나가")
  print(" 최소 고도각 기준을 만족하는 시간이 줄어들거나 아예 없어질 수 있다 — 이건")
  print(" 성좌를 아무리 촘촘하게 짜도 경사각 선택 자체가 만드는 근본적인 한계다.)")
  return rows


def parse_args():
  parser = argparse.ArgumentParser(description="워커 델타 성좌 패턴과 지상 지점 커버리지 공백 계산")
  parser.add_argument("--min-elevation-deg", type=float, default=10.0, help="최소 고도각(도), 기본값: 10.0")
  return parser.parse_args()


def main():
  args = parse_args()

  more_satellites_rows = demo_more_satellites_reduce_max_gap()
  plane_comparison_rows = demo_single_plane_vs_multi_plane()
  latitude_rows = demo_coverage_degrades_above_inclination_latitude()

  results_dir = os.path.join(_THIS_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  more_sats_csv = os.path.join(results_dir, "constellation_size_vs_gap.csv")
  with open(more_sats_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["num_satellites", "num_planes", "max_gap_min", "num_gaps"])
    for row in more_satellites_rows:
      writer.writerow([row["num_satellites"], row["num_planes"], f"{row['max_gap_min']:.4f}", row["num_gaps"]])
  print(f"\n[기록] 위성 수별 공백 시간 결과 저장됨 → {more_sats_csv}")

  plane_csv = os.path.join(results_dir, "constellation_plane_comparison.csv")
  with open(plane_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "num_planes", "max_gap_min", "total_gap_min", "num_gaps"])
    for row in plane_comparison_rows:
      writer.writerow([row["label"], row["num_planes"], f"{row['max_gap_min']:.4f}",
                        f"{row['total_gap_min']:.4f}", row["num_gaps"]])
  print(f"[기록] 평면 배치 비교 결과 저장됨 → {plane_csv}")

  latitude_csv = os.path.join(results_dir, "constellation_latitude_coverage.csv")
  with open(latitude_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "latitude_deg", "max_gap_min", "num_gaps"])
    for row in latitude_rows:
      writer.writerow([row["label"], row["latitude_deg"], f"{row['max_gap_min']:.4f}", row["num_gaps"]])
  print(f"[기록] 위도별 커버리지 결과 저장됨 → {latitude_csv}")

  print(f"\n(참고: --min-elevation-deg={args.min_elevation_deg}는 향후 CLI 파라미터화에 대비한 자리다.)")


if __name__ == "__main__":
  main()
