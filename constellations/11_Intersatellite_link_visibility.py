"""
위성간 링크(ISL) 가시선 - 지구가 두 위성 사이의 시선을 가리는지 기하학적으로 판정

05번은 위성-지상국 사이의 가시성(고도각이 지평선 위인지)을 다뤘다. 위성간 링크
(Inter-Satellite Link, ISL)는 지상국이 아니라 위성과 위성 사이의 직접 통신인데,
이때는 "지평선"이라는 개념 자체가 없다 — 대신 두 위성을 잇는 시선(line of sight)이
지구라는 구형 장애물에 막히는지를 순수 기하학으로 판정해야 한다. Starlink 같은
저궤도 성좌가 지상 중계 없이 위성끼리 직접 데이터를 주고받을 수 있는 것은 이
가시선 조건이 성립하는 동안만이다.

핵심 개념 1: 시선이 가려지는지는 "선분과 구의 최근접 거리" 문제로 환원된다
  두 위성의 위치벡터를 P1, P2라 하면, 이 둘을 잇는 선분 위의 점은
      P(s) = P1 + s*(P2 - P1),   s ∈ [0, 1]
  로 매개변수화된다. 지구 중심(원점)에서 이 직선까지의 최근접 거리를 최소화하는 s는
      s* = -P1 . (P2-P1) / |P2-P1|^2
  이고, 이 s*가 [0,1] 범위 안에 있을 때만 그 최근접점이 두 위성 "사이"에 있다(범위
  밖이면 최근접점은 선분 바깥에 있으므로 지구가 시선을 가릴 수 없다). s*를 [0,1]로
  clip한 뒤 그 지점까지의 거리를 지구 반지름과 비교한다.

핵심 개념 2: 최근접 거리가 지구 반지름보다 작으면 시선이 막힌다
  clip된 s에서의 최근접점 P(s)의 원점까지 거리가 지구 반지름보다 작으면, 두 위성을
  잇는 직선이 지구 내부를 통과한다는 뜻이므로 시선이 가려진다. 크면(또는 s*가
  [0,1] 밖으로 나가 clip됐다면) 가려지지 않는다.

핵심 개념 3: 같은 궤도면 위 위성들은 서로 가려지는 구간이 짧거나 없다
  같은 궤도면(같은 반지름, 같은 경사각/RAAN)을 도는 두 위성은 위상차만 다르므로,
  둘 사이의 거리가 궤도 반지름에 비해 상대적으로 가까울 때가 많아 지구에 가려지는
  구간이 짧거나 아예 없을 수 있다 — 위상차가 클수록(거의 반대편) 가려지는 구간이
  나타난다.

핵심 개념 4: 지구 정반대편에 있는 두 위성은 확실히 가려진다 (경계 케이스 직접 검증)
  두 위성이 지구를 사이에 두고 정확히 반대편에 있으면, 그 둘을 잇는 직선은 반드시
  지구 중심 근처를 지나가므로 최근접 거리가 지구 반지름보다 훨씬 작다 — 이 스크립트는
  이 자명해 보이는 경우를 실제로 계산해 기하 판정 함수 자체가 올바른지 검증한다.

01번(케플러 전파), 05번(orbital_position_eci), orbit_math(회전행렬)과의 관계: 01번의
propagate_orbit/mean_motion을 직접 재사용하고, 궤도면 위치를 ECI로 바꾸는 회전은
05번의 orbital_position_eci를 그대로 import해서 쓴다(중복 재구현하지 않음).
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
_ROOT_DIR = os.path.dirname(_THIS_DIR)

_KEPLER_PATH = os.path.join(_ROOT_DIR, "propagation", "01_Kepler_orbit_propagation.py")
_spec = importlib.util.spec_from_file_location("kepler_module", _KEPLER_PATH)
kepler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kepler)

_VISIBILITY_PATH = os.path.join(_ROOT_DIR, "missions", "05_Ground_station_visibility.py")
_visibility_spec = importlib.util.spec_from_file_location("visibility_module", _VISIBILITY_PATH)
visibility = importlib.util.module_from_spec(_visibility_spec)
_visibility_spec.loader.exec_module(visibility)

_orbit_math_spec = importlib.util.spec_from_file_location("orbit_math", os.path.join(_ROOT_DIR, "orbit_math.py"))
orbit_math = importlib.util.module_from_spec(_orbit_math_spec)
_orbit_math_spec.loader.exec_module(orbit_math)
EARTH_RADIUS_KM = orbit_math.EARTH_RADIUS_KM

EARTH_MU_KM3_S2 = kepler.EARTH_MU_KM3_S2
orbital_position_eci = visibility.orbital_position_eci


def is_line_of_sight_blocked_by_earth(position1_eci_km, position2_eci_km, earth_radius_km=EARTH_RADIUS_KM):
  """두 위성을 잇는 선분과 지구 중심 사이의 최근접 거리로 시선 차단 여부를 판정한다.
  반환: (blocked: bool, closest_distance_km: float)."""
  p1 = np.array(position1_eci_km, dtype=float)
  p2 = np.array(position2_eci_km, dtype=float)
  segment = p2 - p1
  segment_length_sq = np.dot(segment, segment)

  if segment_length_sq < 1e-9:
    # 두 위성이 사실상 같은 위치 - 선분 자체가 정의되지 않으므로 그 지점의 거리로 판정
    distance = np.linalg.norm(p1)
    return distance < earth_radius_km, distance

  s = -np.dot(p1, segment) / segment_length_sq
  s_clipped = np.clip(s, 0.0, 1.0)
  closest_point = p1 + s_clipped * segment
  closest_distance = np.linalg.norm(closest_point)
  return closest_distance < earth_radius_km, closest_distance


def compute_isl_visibility_time_series(sat1_elements, sat2_elements, duration_sec, num_steps):
  """두 위성의 궤도요소로 매 시점 ISL 가시 여부 시계열을 만든다.
  각 위성 요소는 dict(semi_major_axis_km, eccentricity, inclination_rad, raan_rad,
  argp_rad, mean_anomaly0_rad)."""
  times = np.linspace(0, duration_sec, num_steps)
  rows = []
  for t in times:
    pos1 = orbital_position_eci(sat1_elements["semi_major_axis_km"], sat1_elements["eccentricity"],
                                 sat1_elements["inclination_rad"], sat1_elements["raan_rad"],
                                 sat1_elements["argp_rad"], sat1_elements["mean_anomaly0_rad"], time_sec=t)
    pos2 = orbital_position_eci(sat2_elements["semi_major_axis_km"], sat2_elements["eccentricity"],
                                 sat2_elements["inclination_rad"], sat2_elements["raan_rad"],
                                 sat2_elements["argp_rad"], sat2_elements["mean_anomaly0_rad"], time_sec=t)
    blocked, closest_distance = is_line_of_sight_blocked_by_earth(pos1, pos2)
    separation_km = np.linalg.norm(pos2 - pos1)
    rows.append({"t_sec": t, "blocked": blocked, "closest_distance_km": closest_distance,
                 "separation_km": separation_km})
  return rows


def demo_same_plane_satellites_mostly_visible(altitude_km=550.0):
  """같은 궤도면 위 두 위성(위상차만 다름)은 지구에 가려지는 구간이 짧거나
  없다는 것을 확인한다."""
  print("=" * 70)
  print("[1] 같은 궤도면 위 두 위성: ISL 가시 구간이 대부분이다")
  print("=" * 70)
  a = EARTH_RADIUS_KM + altitude_km
  inclination = np.radians(53.0)
  raan = 0.0
  duration_sec = 2 * np.pi / kepler.mean_motion(a)  # 한 궤도 주기
  num_steps = 500

  sat1 = {"semi_major_axis_km": a, "eccentricity": 0.0, "inclination_rad": inclination,
          "raan_rad": raan, "argp_rad": 0.0, "mean_anomaly0_rad": 0.0}
  sat2 = {"semi_major_axis_km": a, "eccentricity": 0.0, "inclination_rad": inclination,
          "raan_rad": raan, "argp_rad": 0.0, "mean_anomaly0_rad": np.radians(30.0)}  # 30도 위상차

  print(f"반장축={a:.1f}km, 경사각={np.degrees(inclination):.0f}도, 위상차=30도\n")
  series = compute_isl_visibility_time_series(sat1, sat2, duration_sec, num_steps)
  blocked_fraction = sum(1 for row in series if row["blocked"]) / len(series)
  print(f"가려진 시간 비율: {blocked_fraction * 100:.2f}%")

  assert blocked_fraction < 0.5, "30도 위상차의 같은 평면 위성은 가려지는 시간이 절반 미만이어야 함"
  print("\n(위상차가 30도로 크지 않아, 두 위성 사이의 거리가 궤도 반지름에 비해 상대적으로")
  print(" 가까운 시간이 대부분이다 — 지구가 시선을 가로막을 만큼 두 위성이 서로 반대편에")
  print(" 있는 시간이 짧다.)")
  return series, blocked_fraction


def demo_opposite_side_of_earth_blocked():
  """지구 정반대편에 있는 두 위성 사이의 시선은 확실히 가려진다는 것을 직접
  계산해 기하 판정 함수 자체를 검증한다."""
  print("\n" + "=" * 70)
  print("[2] 경계 케이스: 지구 정반대편의 두 위성은 시선이 반드시 가려진다")
  print("=" * 70)
  a = EARTH_RADIUS_KM + 550.0
  pos1 = np.array([a, 0.0, 0.0])
  pos2 = np.array([-a, 0.0, 0.0])  # 정확히 반대편

  blocked, closest_distance = is_line_of_sight_blocked_by_earth(pos1, pos2)
  print(f"위성 1 위치: {pos1}, 위성 2 위치: {pos2}")
  print(f"시선-지구중심 최근접 거리: {closest_distance:.2f}km (지구 반지름: {EARTH_RADIUS_KM:.2f}km)")
  print(f"시선 차단 여부: {'예(가려짐)' if blocked else '아니오'}")

  assert blocked, "지구 정반대편의 두 위성은 시선이 반드시 가려져야 함"
  assert closest_distance < 1e-6, "정반대편 위성을 잇는 직선은 정확히 지구 중심을 지나야 함(최근접 거리 0)"
  print("\n(두 위성을 잇는 직선이 정확히 지구 중심을 통과하므로, 최근접 거리가 0에 가깝다 —")
  print(" 기하 판정 함수가 이 자명한 경계 케이스에서 올바르게 동작함을 확인했다.)")
  return {"closest_distance_km": closest_distance, "blocked": blocked}


def demo_polar_vs_equatorial_plane_crossing():
  """서로 다른 궤도면(극궤도 vs 적도궤도) 위 두 위성의 ISL 가시성이 시간에 따라
  어떻게 변하는지 시계열로 관찰한다."""
  print("\n" + "=" * 70)
  print("[3] 서로 다른 궤도면(극궤도 vs 적도궤도)의 ISL 가시성 시계열")
  print("=" * 70)
  a = EARTH_RADIUS_KM + 1000.0
  duration_sec = 2 * np.pi / kepler.mean_motion(a)
  num_steps = 500

  # 두 위성이 mean_anomaly0=0으로 같은 시점에 승교점에서 출발하면, 반장축(따라서
  # 평균운동)이 같으므로 둘의 상대 기하가 궤도 내내 고정돼버린다(자전만 다른 평면으로
  # 돌 뿐 서로에 대한 위상차는 절대 안 바뀜) — 실제로 이 버그를 겪었다: 500개 시점
  # 전부에서 거리/차단 여부가 완전히 똑같은 상수로 나왔다. 위상차(45도)를 줘야
  # 시간에 따라 상대 거리가 실제로 변하는 시계열이 나온다.
  sat_polar = {"semi_major_axis_km": a, "eccentricity": 0.0, "inclination_rad": np.radians(90.0),
               "raan_rad": 0.0, "argp_rad": 0.0, "mean_anomaly0_rad": 0.0}
  sat_equatorial = {"semi_major_axis_km": a, "eccentricity": 0.0, "inclination_rad": 0.0,
                    "raan_rad": 0.0, "argp_rad": 0.0, "mean_anomaly0_rad": np.radians(45.0)}

  print(f"반장축={a:.1f}km, 위성1: 극궤도(90도), 위성2: 적도궤도(0도), 위상차=45도\n")
  series = compute_isl_visibility_time_series(sat_polar, sat_equatorial, duration_sec, num_steps)
  blocked_fraction = sum(1 for row in series if row["blocked"]) / len(series)
  print(f"가려진 시간 비율: {blocked_fraction * 100:.2f}%")
  print(f"최소 시선-지구중심 거리: {min(row['closest_distance_km'] for row in series):.1f}km")
  print(f"최대 위성간 거리: {max(row['separation_km'] for row in series):.1f}km")

  print("\n(경사각이 크게 다른 두 궤도면은 위성끼리의 상대 위치가 궤도를 도는 동안 계속")
  print(" 크게 바뀌어, 같은 평면 위성 쌍(데모 1번)보다 가려지는 시간 비율이 훨씬 크게")
  print(" 나올 수 있다 — 성좌 설계에서 서로 다른 평면끼리 ISL을 놓을 때 이 점을 고려해야")
  print(" 한다.)")
  return series, blocked_fraction


def parse_args():
  parser = argparse.ArgumentParser(description="위성간 링크(ISL) 가시선의 지구 차단 여부 기하학적 판정")
  parser.add_argument("--altitude-km", type=float, default=550.0, help="위성 고도(km), 기본값: 550km")
  return parser.parse_args()


def main():
  args = parse_args()

  same_plane_series, same_plane_fraction = demo_same_plane_satellites_mostly_visible(args.altitude_km)
  opposite_result = demo_opposite_side_of_earth_blocked()
  cross_plane_series, cross_plane_fraction = demo_polar_vs_equatorial_plane_crossing()

  results_dir = os.path.join(_ROOT_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  same_plane_csv = os.path.join(results_dir, "isl_same_plane_visibility.csv")
  with open(same_plane_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_sec", "blocked", "closest_distance_km", "separation_km"])
    for row in same_plane_series:
      writer.writerow([f"{row['t_sec']:.2f}", row["blocked"], f"{row['closest_distance_km']:.2f}", f"{row['separation_km']:.2f}"])
  print(f"\n[기록] 같은 평면 ISL 가시성 시계열 저장됨 → {same_plane_csv}")

  opposite_csv = os.path.join(results_dir, "isl_opposite_side_edge_case.csv")
  with open(opposite_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["closest_distance_km", "blocked"])
    writer.writerow([f"{opposite_result['closest_distance_km']:.4f}", opposite_result["blocked"]])
  print(f"[기록] 정반대편 경계 케이스 결과 저장됨 → {opposite_csv}")

  cross_plane_csv = os.path.join(results_dir, "isl_polar_vs_equatorial.csv")
  with open(cross_plane_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_sec", "blocked", "closest_distance_km", "separation_km"])
    for row in cross_plane_series:
      writer.writerow([f"{row['t_sec']:.2f}", row["blocked"], f"{row['closest_distance_km']:.2f}", f"{row['separation_km']:.2f}"])
  print(f"[기록] 극궤도-적도궤도 ISL 가시성 시계열 저장됨 → {cross_plane_csv}")

  print(f"\n(같은 평면 가려짐 비율: {same_plane_fraction * 100:.2f}%, "
        f"극-적도 평면 가려짐 비율: {cross_plane_fraction * 100:.2f}%)")


if __name__ == "__main__":
  main()
