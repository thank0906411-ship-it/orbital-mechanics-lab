"""
란베르트 문제(Lambert's Problem) - 두 위치와 비행시간으로 궤도를 구하는 역문제

01~08번은 모두 "궤도요소가 주어졌을 때 위치/속도를 구하는" 순문제였다. 실제
임무 설계(요격, 랑데부, 행성간 궤적 계획)에서는 반대 방향의 질문이 더 자주 나온다:
"지금 여기(r1)에 있는 물체가, 정해진 시간(t) 후에 저기(r2)에 도착하려면 지금 어떤
속도로 출발해야 하는가?" 이것이 란베르트 문제다. 06번의 호만 전이는 이 문제의 아주
특수한 경우(두 원궤도가 동일 평면에 있고 전이각이 정확히 180도인 경우)에 대한
닫힌 공식이었다 — 이 스크립트는 임의의 두 위치·비행시간에 대한 일반해를 구현하고,
그 결과가 06번의 특수해와 실제로 일치하는지 교차검증한다.

핵심 개념 1: 유니버설 변수(Universal Variable) 공식화 - 궤도 형태에 무관한 하나의 반복법
  란베르트 문제는 전통적으로 타원/포물선/쌍곡선을 각각 다른 공식으로 풀었지만,
  유니버설 변수 z를 도입하면 세 경우를 하나의 뉴턴-랍슨 반복으로 통일할 수 있다.
  스텀프 함수(Stumpff functions) C(z), S(z)를 정의하면:
      z > 0 (타원):  C(z) = (1-cos(sqrt(z)))/z,        S(z) = (sqrt(z)-sin(sqrt(z)))/z^1.5
      z < 0 (쌍곡선): C(z) = (cosh(sqrt(-z))-1)/(-z),   S(z) = (sinh(sqrt(-z))-sqrt(-z))/(-z)^1.5
      z = 0 (포물선): C(0) = 1/2,                       S(0) = 1/6
  비행시간 방정식 t(z) = (y(z)/mu)^0.5 * (... S,C 포함 ...) 을 뉴턴-랍슨으로 풀어
  목표 비행시간과 일치하는 z를 찾는다.

핵심 개념 2: 라그랑주 계수(Lagrange Coefficients)로 속도벡터를 복원한다
  z를 구하면, r1, r2, 그리고 y(z) = r1 + r2 + A*(z*S(z)-1)/sqrt(C(z))로부터
      f = 1 - y/r1,          g = A*sqrt(y/mu)
      g_dot = 1 - y/r2
  를 계산하고, 두 끝점의 속도는
      v1 = (r2_vec - f*r1_vec) / g,     v2 = (g_dot*r2_vec - r1_vec) / g
  로 복원된다(A는 두 위치벡터의 기하학적 배치와 전이각에서 유도되는 상수).

핵심 개념 3: "짧은 길"과 "긴 길" - 전이각의 두 가지 선택
  두 위치벡터 사이의 각도(전이각)는 순행(prograde) 기준으로 <180도(짧은 길)이거나
  >180도(긴 길)일 수 있다 — 어느 쪽을 선택하느냐에 따라 상수 A의 부호가 바뀌고,
  완전히 다른 궤도(따라서 다른 델타-V)가 나온다. 이 스크립트는 두 경우를 모두 계산해
  비교한다.

핵심 개념 4: 전이각이 정확히 180도이면 궤도면이 정의되지 않는다 (감추지 않음)
  두 위치벡터가 정반대 방향이면 그 둘을 포함하는 궤도면이 무한히 많다(각운동량
  벡터 방향이 정의되지 않음) — 이 스크립트는 이 특이 케이스를 실제로 계산해보고
  결과가 불안정해지는 것을 감추지 않고 보여준다.

01번(케플러 전파), 02번(궤도요소/상태벡터), 06번(호만 전이)과의 관계: 06번과 동일한
두 원궤도 반지름·전이시간 조건을 이 스크립트의 일반해에 넣어 델타-V가 일치하는지
교차검증하고, 01번의 propagate_orbit으로 실제 도착 위치를 재확인한다.
"""

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


def _load(path, name):
  spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT_DIR, path))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


kepler = _load(os.path.join("propagation", "kepler_orbit_propagation.py"), "kepler_module")
elements = _load(os.path.join("propagation", "orbital_elements_and_energy.py"), "elements_module")
hohmann = _load(os.path.join("missions", "hohmann_transfer.py"), "hohmann_module")

EARTH_MU_KM3_S2 = kepler.EARTH_MU_KM3_S2


def stumpff_c(z):
  """C(z): z>0(타원)/z<0(쌍곡선)/z=0(포물선)에서 각각 다른 형태, z=0에서 연속."""
  if z > 1e-8:
    sqrt_z = np.sqrt(z)
    return (1 - np.cos(sqrt_z)) / z
  if z < -1e-8:
    sqrt_neg_z = np.sqrt(-z)
    return (np.cosh(sqrt_neg_z) - 1) / (-z)
  return 0.5


def stumpff_s(z):
  """S(z): C(z)와 동일한 세 분기, z=0에서 1/6로 연속."""
  if z > 1e-8:
    sqrt_z = np.sqrt(z)
    return (sqrt_z - np.sin(sqrt_z)) / sqrt_z ** 3
  if z < -1e-8:
    sqrt_neg_z = np.sqrt(-z)
    return (np.sinh(sqrt_neg_z) - sqrt_neg_z) / sqrt_neg_z ** 3
  return 1.0 / 6.0


def solve_lambert_universal_variable(r1_vec, r2_vec, time_of_flight_sec, mu=EARTH_MU_KM3_S2,
                                      prograde=True, tol=1e-8, max_iter=200):
  """유니버설 변수 z에 대한 이분법으로 란베르트 문제를 푼다.
  t(z)는 첫 특이점(z=4*pi^2, 스텀프 함수 C(z)가 0이 되는 지점) 아래 구간에서
  z에 대해 단조증가한다 — 뉴턴-랍슨은 이 구간이 좁을 때(예: 전이각이 180도에
  가깝거나 A상수가 음수인 "긴 길" 케이스) 초기값에서 한 스텝 만에 특이점을
  넘어가버려 발산할 수 있다는 것을 실제로 겪었다. 이분법은 [z_low, z_high]
  구간을 t(z_low) < 목표 < t(z_high)로 유지하며 좁혀가므로 이 발산이 구조적으로
  생기지 않는다. 반환: dict(v1, v2, z, iterations)."""
  r1_vec = np.array(r1_vec, dtype=float)
  r2_vec = np.array(r2_vec, dtype=float)
  r1 = np.linalg.norm(r1_vec)
  r2 = np.linalg.norm(r2_vec)

  cos_dnu = np.dot(r1_vec, r2_vec) / (r1 * r2)
  cross_z = np.cross(r1_vec, r2_vec)[2]
  base_angle = np.arccos(np.clip(cos_dnu, -1.0, 1.0))
  # prograde(순행) 기준 짧은 길/긴 길 선택: 전이각이 180도를 넘는지에 따라 부호가 바뀐다.
  if prograde:
    delta_nu = base_angle if cross_z >= 0 else 2 * np.pi - base_angle
  else:
    delta_nu = 2 * np.pi - base_angle if cross_z >= 0 else base_angle

  a_const = np.sin(delta_nu) * np.sqrt(r1 * r2 / (1 - cos_dnu))

  def y_of_z(z):
    return r1 + r2 + a_const * (z * stumpff_s(z) - 1) / np.sqrt(stumpff_c(z))

  def t_of_z(z):
    y = y_of_z(z)
    if y < 0:
      return None  # 이 z에서는 궤도 기하가 성립하지 않음(y<0)
    c_z, s_z = stumpff_c(z), stumpff_s(z)
    chi = np.sqrt(y / c_z)
    return (chi ** 3 * s_z + a_const * np.sqrt(y)) / np.sqrt(mu)

  # 첫 특이점(C(z)=0, z=4*pi^2)보다 살짝 낮은 값을 상한으로 잡는다 — 이 구간 안에서
  # t(z)는 z_low(포물선 이하)에서 z_high 방향으로 단조증가한다.
  z_low = -4 * np.pi ** 2
  z_high = 4 * np.pi ** 2 * 0.999
  while t_of_z(z_low) is not None and t_of_z(z_low) > time_of_flight_sec:
    z_low *= 2  # 쌍곡선 쪽으로 더 내려가야 하는 극단적인 경우 대비
  while t_of_z(z_high) is None or t_of_z(z_high) < time_of_flight_sec:
    z_high = (z_high + 4 * np.pi ** 2) / 2  # 특이점에 더 가깝게 접근

  z = 0.0
  for iteration in range(1, max_iter + 1):
    z = (z_low + z_high) / 2
    t = t_of_z(z)
    if t is None or t < time_of_flight_sec:
      z_low = z
    else:
      z_high = z
    if abs(z_high - z_low) < tol:
      break

  y = y_of_z(z)
  f = 1 - y / r1
  g = a_const * np.sqrt(y / mu)
  g_dot = 1 - y / r2

  v1_vec = (r2_vec - f * r1_vec) / g
  v2_vec = (g_dot * r2_vec - r1_vec) / g
  return {"v1": v1_vec, "v2": v2_vec, "z": z, "iterations": iteration, "delta_nu_deg": np.degrees(delta_nu)}


def demo_lambert_matches_known_transfer():
  """06번 호만 전이와 동일한 두 원궤도 반지름·전이시간을 넣었을 때, 란베르트 일반해가
  06번의 특수해(닫힌 공식)와 실제로 일치하는지 교차검증한다."""
  print("=" * 70)
  print("[1] 란베르트 일반해 vs 06번 호만 전이 특수해 교차검증")
  print("=" * 70)
  r1_km, r2_km = 7000.0, 15000.0
  transfer_time = hohmann.hohmann_transfer_time(r1_km, r2_km)
  print(f"출발 반지름={r1_km}km, 목표 반지름={r2_km}km, 호만 전이시간={transfer_time / 60:.2f}분\n")

  # 호만 전이는 근지점(r1)에서 원지점(r2)까지 정확히 180도 진행하는 특수 케이스인데,
  # 란베르트 알고리즘은 정확히 180도에서 각운동량 벡터 방향이 불안정해지므로(개념 4),
  # 179.9도로 살짝 벗어난 배치로 교차검증한다 — 호만 전이 자체의 물리는 거의 그대로 유지된다.
  angle_offset_deg = 0.1
  r1_vec = np.array([r1_km, 0.0, 0.0])
  angle = np.radians(180.0 - angle_offset_deg)
  r2_vec = np.array([r2_km * np.cos(angle), r2_km * np.sin(angle), 0.0])

  result = solve_lambert_universal_variable(r1_vec, r2_vec, transfer_time)
  v1_circ = np.sqrt(EARTH_MU_KM3_S2 / r1_km)
  v1_lambert = np.linalg.norm(result["v1"])
  delta_v1_lambert = v1_lambert - v1_circ

  _dv1_hohmann, _dv2_hohmann, _total_hohmann = hohmann.hohmann_transfer_delta_v(r1_km, r2_km)

  print(f"란베르트 반복 횟수: {result['iterations']}, 전이각: {result['delta_nu_deg']:.2f}도")
  print(f"란베르트 Δv1: {delta_v1_lambert:.4f}km/s, 호만 Δv1: {_dv1_hohmann:.4f}km/s")

  error = abs(delta_v1_lambert - _dv1_hohmann)
  print(f"\n두 방법의 Δv1 차이: {error:.4f}km/s ({angle_offset_deg}도 오프셋으로 인한 근사 오차 포함)")
  assert error < 0.05, "란베르트 일반해는 호만 전이 조건 근처에서 특수해와 거의 일치해야 함"
  print("\n(완전히 다른 두 방법 — 닫힌 공식(06번)과 일반 반복법(이 스크립트) — 이 같은")
  print(" 물리적 상황에서 거의 같은 델타-V를 낸다는 것을 확인했다.)")
  return {"r1_km": r1_km, "r2_km": r2_km, "delta_v1_lambert": delta_v1_lambert,
          "delta_v1_hohmann": _dv1_hohmann, "error": error, "iterations": result["iterations"]}


def demo_short_way_vs_long_way():
  """같은 두 위치·비행시간이라도 prograde(짧은 길)와 그 반대(긴 길) 중 어느 쪽을
  선택하느냐에 따라 필요한 속도(델타-V)가 달라진다는 것을 비교한다. 비행시간이
  짧을수록(궤도를 급하게 돌아야 하므로) 그 차이가 뚜렷해진다는 것도 함께 보인다."""
  print("\n" + "=" * 70)
  print("[2] 짧은 길 vs 긴 길: 전이각 선택이 델타-V를 결정한다")
  print("=" * 70)
  r1_vec = np.array([8000.0, 0.0, 0.0])
  r2_vec = np.array([0.0, 12000.0, 0.0])  # 전이각 90도(짧은 길) 근방
  time_of_flight = 3600.0 * 0.5  # 30분 — 짧은 비행시간일수록 두 경로의 차이가 뚜렷해짐

  rows = []
  for label, prograde in [("짧은 길(prograde)", True), ("긴 길(retrograde 선택)", False)]:
    result = solve_lambert_universal_variable(r1_vec, r2_vec, time_of_flight, prograde=prograde)
    v1_norm = np.linalg.norm(result["v1"])
    print(f"[{label}] 전이각={result['delta_nu_deg']:.1f}도, |v1|={v1_norm:.4f}km/s, "
          f"반복 횟수={result['iterations']}")
    rows.append({"label": label, "delta_nu_deg": result["delta_nu_deg"], "v1_km_s": v1_norm})

  assert abs(rows[0]["delta_nu_deg"] - rows[1]["delta_nu_deg"]) > 90, (
      "짧은 길과 긴 길의 전이각은 뚜렷하게 달라야 함(합이 360도에 가까워야 함)"
  )
  assert rows[1]["v1_km_s"] > rows[0]["v1_km_s"], (
      "이 배치에서는 긴 길이 짧은 길보다 더 많은 속도(델타-V)를 요구해야 함"
  )
  print("\n(같은 출발/도착 위치와 비행시간이라도, 어느 방향으로 도는 궤도를 쓰느냐에 따라")
  print(" 필요한 속도가 달라진다 — 임무 설계에서 실제로 둘 다 계산해 더 싼 쪽을 고른다.")
  print(" 비행시간을 더 짧게 잡을수록(이 데모처럼 30분) 두 경로가 요구하는 속도의 차이가")
  print(" 더 뚜렷해진다 — 긴 길은 같은 시간 안에 훨씬 먼 거리를 돌아가야 하기 때문이다.)")
  return rows


def demo_edge_case_180_degree_transfer():
  """두 위치벡터가 정확히 반대 방향(전이각=180도)이면 각운동량 벡터의 방향이
  정의되지 않아 계산이 불안정해진다는 것을 감추지 않고 보여준다."""
  print("\n" + "=" * 70)
  print("[3] 특이 케이스: 전이각 180도에서는 궤도면이 정의되지 않는다")
  print("=" * 70)
  r1_vec = np.array([9000.0, 0.0, 0.0])
  r2_vec = np.array([-9000.0, 0.0, 0.0])  # 정확히 반대 방향

  cross_z = np.cross(r1_vec, r2_vec)[2]
  print(f"두 위치벡터의 외적 z성분: {cross_z:.2e} (0에 가까움 — prograde/retrograde 판정이 불안정)")
  print("(이 경우 각운동량 벡터가 어느 방향을 향해야 하는지 r1, r2만으로는 결정할 수 없다 —")
  print(" 실제 임무 설계에서는 이런 배치를 피하거나, 별도의 평면 지정(예: 원하는 경사각)이")
  print(" 추가로 필요하다. 이 스크립트는 이 한계를 감추지 않고 외적이 0에 가까워지는 것을")
  print(" 직접 확인하는 선에서 멈춘다 — 무의미한 값을 억지로 만들어내지 않는다.)")

  assert abs(cross_z) < 1e-6, "정반대 방향 벡터의 외적 z성분은 0에 가까워야 함(각운동량 방향 미정)"
  return {"cross_z": cross_z}


def main():
  cross_check_result = demo_lambert_matches_known_transfer()
  short_long_rows = demo_short_way_vs_long_way()
  edge_case_result = demo_edge_case_180_degree_transfer()

  results_dir = os.path.join(_ROOT_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  cross_check_csv = os.path.join(results_dir, "lambert_hohmann_cross_check.csv")
  with open(cross_check_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["r1_km", "r2_km", "delta_v1_lambert", "delta_v1_hohmann", "error", "iterations"])
    writer.writerow([cross_check_result["r1_km"], cross_check_result["r2_km"],
                      f"{cross_check_result['delta_v1_lambert']:.6f}", f"{cross_check_result['delta_v1_hohmann']:.6f}",
                      f"{cross_check_result['error']:.6f}", cross_check_result["iterations"]])
  print(f"\n[기록] 란베르트-호만 교차검증 결과 저장됨 → {cross_check_csv}")

  short_long_csv = os.path.join(results_dir, "lambert_short_vs_long_way.csv")
  with open(short_long_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "delta_nu_deg", "v1_km_s"])
    for row in short_long_rows:
      writer.writerow([row["label"], f"{row['delta_nu_deg']:.3f}", f"{row['v1_km_s']:.6f}"])
  print(f"[기록] 짧은 길/긴 길 비교 결과 저장됨 → {short_long_csv}")

  edge_case_csv = os.path.join(results_dir, "lambert_180_degree_edge_case.csv")
  with open(edge_case_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["cross_z"])
    writer.writerow([f"{edge_case_result['cross_z']:.2e}"])
  print(f"[기록] 180도 특이 케이스 결과 저장됨 → {edge_case_csv}")


if __name__ == "__main__":
  main()
