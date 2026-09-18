"""
저추력 전기추진 궤도 전이(Low-Thrust Orbit Transfer) - 연속 추력으로 서서히 올라가는 나선 궤적

06번(호만 전이)은 "순간적인 임펄스 두 번"으로 궤도를 바꾸는 화학 추진 방식을
다뤘다. 실제 현대 통신위성/스타링크 같은 저궤도 성좌는 이온엔진 같은 전기추진으로
훨씬 작은 추력을 몇 주~몇 달에 걸쳐 연속으로 가해 궤도를 서서히 올린다 — 06번과
정반대 축(순간 vs 연속)의 궤도 변경 방법이다. 이 스크립트는 03번의 RK4 적분기
구조를 그대로 빌려, 중심력에 접선방향(진행방향) 추력 가속도를 더해 원궤도가
나선형으로 상승하는 과정을 직접 적분한다.

핵심 개념 1: 접선 추력은 에너지를 계속 주입해 반지름을 서서히 키운다
  중심력만 있으면 에너지가 보존되지만(02번에서 검증), 속도 방향으로 작은 가속도
  a_T를 더하면 매 순간 dE/dt = a_T * |v|만큼 비에너지가 늘어난다. 추력이 항상
  속도 방향(반지름 방향이 아님)이므로 이심률은 거의 늘리지 않고 반지름만 서서히
  키운다 — 원궤도를 거의 유지한 채로 상승하는 나선이 나오는 이유다.

핵심 개념 2: 이 근사의 닫힌 해(Edelbaum 근사)가 검증 기준이다
  경사각 변화가 없는 평면 내 원궤도 간 전이라면, 필요한 총 델타-V는 출발/도착
  원궤도 속력의 차이로 근사된다:
      Δv_Edelbaum = |v_circ(r1) - v_circ(r2)|
  이는 06번의 vis-viva 기반 두 임펄스 합과는 다른 공식이다(에너지가 아니라 속력
  차이를 직접 쓴다). RK4로 실제 나선을 적분한 뒤 "추력 가속도 크기 x 걸린 시간"
  으로 계산한 총 델타-V가 이 근사와 일치하는지가 이 스크립트의 핵심 검증이다.

핵심 개념 3: 델타-V는 비슷해도 걸리는 시간은 자릿수가 다르다
  06번 hohmann_transfer_delta_v로 같은 반지름 쌍의 호만 전이 델타-V/시간을
  계산해 직접 비교한다. 저추력 델타-V는 호만과 비슷한 수준이지만(원궤도를
  유지하는 효율 덕분), 전이 시간은 몇 시간(호만) 대 며칠(저추력)로 자릿수가
  다르다 — "델타-V가 비슷해도 무조건 유리한 건 아니다"라는, 실제 임무 설계에서
  잘 알려진 트레이드오프를 이 프로젝트 방식대로 직접 계산해서 재현한다.

핵심 개념 4: 추력이 클수록 나선이 가팔라지고 전이 시간이 짧아진다
  추력 가속도 크기 a_T를 바꿔가며 목표 반지름 도달 시간이 어떻게 변하는지
  확인한다. 총 임펄스(시간 x 가속도)가 대략 일정하므로, a_T가 커지면 전이
  시간은 대략 반비례로 줄어드는 경향을 보인다.

단순화: 이 스크립트는 평면 내(경사각 변화 없음) 원궤도 간 전이만 다룬다. 실제
저추력 임무는 경사각 변경, 중력 손실(gravity loss), 추력기 On/Off 스케줄까지
고려하지만, 이 프로젝트는 "접선 추력 하나로 반지름만 바꾸는" 가장 단순한 경우로
범위를 제한한다.

01번(케플러 전파)의 EARTH_MU_KM3_S2, 03번(RK4 수치적분)의 두 물체 중심력
가속도(two_body_acceleration, 수정 없이 재사용), 04번의 orbital elements
변환과 동일한 상태벡터-궤도요소 역변환(propagation/orbital_elements_and_energy.py
의 state_vector_to_orbital_elements), missions/hohmann_transfer.py의
hohmann_transfer_delta_v와의 관계: RK4 루프 자체는 03번 rk4_step의 derivative
클로저가 중심력만 하드코딩되어 있어 그대로 재사용할 수 없으므로(위치만 받아
가속도를 계산하는 구조), 이 스크립트는 그 구조를 복제하되 접선 추력 항을 더한
자체 RK4 스텝 함수를 새로 정의한다.
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

_KEPLER_PATH = os.path.join(_ROOT_DIR, "propagation", "kepler_orbit_propagation.py")
_kepler_spec = importlib.util.spec_from_file_location("kepler_module", _KEPLER_PATH)
kepler = importlib.util.module_from_spec(_kepler_spec)
_kepler_spec.loader.exec_module(kepler)

_ELEMENTS_PATH = os.path.join(_ROOT_DIR, "propagation", "orbital_elements_and_energy.py")
_elements_spec = importlib.util.spec_from_file_location("elements_module", _ELEMENTS_PATH)
elements_module = importlib.util.module_from_spec(_elements_spec)
_elements_spec.loader.exec_module(elements_module)

_RK4_PATH = os.path.join(_ROOT_DIR, "propagation", "two_body_numerical_integration.py")
_rk4_spec = importlib.util.spec_from_file_location("rk4_module", _RK4_PATH)
rk4_module = importlib.util.module_from_spec(_rk4_spec)
_rk4_spec.loader.exec_module(rk4_module)

_HOHMANN_PATH = os.path.join(_ROOT_DIR, "missions", "hohmann_transfer.py")
_hohmann_spec = importlib.util.spec_from_file_location("hohmann_module", _HOHMANN_PATH)
hohmann_module = importlib.util.module_from_spec(_hohmann_spec)
_hohmann_spec.loader.exec_module(hohmann_module)

EARTH_MU_KM3_S2 = kepler.EARTH_MU_KM3_S2
EARTH_RADIUS_KM = 6378.137
two_body_acceleration = rk4_module.two_body_acceleration
state_vector_to_orbital_elements = elements_module.state_vector_to_orbital_elements
hohmann_transfer_delta_v = hohmann_module.hohmann_transfer_delta_v
hohmann_transfer_time = hohmann_module.hohmann_transfer_time


def circular_orbit_speed(radius_km, mu=EARTH_MU_KM3_S2):
  """반지름 r인 원궤도의 속력: v = sqrt(mu/r)."""
  return np.sqrt(mu / radius_km)


def tangential_thrust_acceleration(velocity_km_s, thrust_accel_km_s2):
  """속도 방향 단위벡터 x 추력 가속도 크기. 방향은 항상 진행방향(접선방향)."""
  velocity = np.array(velocity_km_s, dtype=float)
  speed = np.linalg.norm(velocity)
  return velocity / speed * thrust_accel_km_s2


def rk4_step_with_thrust(position_km, velocity_km_s, dt_sec, thrust_accel_km_s2, mu=EARTH_MU_KM3_S2):
  """03번 rk4_step과 동일한 구조지만, derivative 클로저에 접선 추력 가속도 항을
  더한다. 03번의 rk4_step은 중심력만 계산하도록 하드코딩되어 있어 그대로 재사용할
  수 없으므로(위치만 받는 구조), 이 함수가 03번의 구조를 복제해 확장한다."""
  def derivative(pos, vel):
    accel = two_body_acceleration(pos, mu) + tangential_thrust_acceleration(vel, thrust_accel_km_s2)
    return vel, accel

  k1_r, k1_v = derivative(position_km, velocity_km_s)
  k2_r, k2_v = derivative(position_km + dt_sec / 2 * k1_r, velocity_km_s + dt_sec / 2 * k1_v)
  k3_r, k3_v = derivative(position_km + dt_sec / 2 * k2_r, velocity_km_s + dt_sec / 2 * k2_v)
  k4_r, k4_v = derivative(position_km + dt_sec * k3_r, velocity_km_s + dt_sec * k3_v)

  new_position = position_km + dt_sec / 6 * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
  new_velocity = velocity_km_s + dt_sec / 6 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
  return new_position, new_velocity


def spiral_transfer(initial_radius_km, target_radius_km, thrust_accel_km_s2, dt_sec,
                     mu=EARTH_MU_KM3_S2, record_every_n_steps=5, max_steps=2_000_000):
  """초기 원궤도(initial_radius_km)에서 접선 추력을 계속 가해, 반지름이
  target_radius_km 이상이 될 때까지 RK4로 적분한다. 매 스텝을 다 기록하면 나선
  전이가 수천~수만 스텝이 될 수 있어 record_every_n_steps마다 한 번만 기록한다.
  반환: {"history": [(t, position, velocity), ...], "final_position", "final_velocity",
  "final_time_sec", "num_steps"}."""
  position = np.array([initial_radius_km, 0.0, 0.0])
  velocity = np.array([0.0, circular_orbit_speed(initial_radius_km, mu), 0.0])
  t = 0.0
  step_count = 0
  history = [(t, position.copy(), velocity.copy())]

  while np.linalg.norm(position) < target_radius_km:
    position, velocity = rk4_step_with_thrust(position, velocity, dt_sec, thrust_accel_km_s2, mu)
    t += dt_sec
    step_count += 1
    if step_count % record_every_n_steps == 0:
      history.append((t, position.copy(), velocity.copy()))
    if step_count >= max_steps:
      raise RuntimeError(
          f"목표 반지름({target_radius_km}km)에 {max_steps}스텝 안에 도달하지 못함 — "
          "추력이 너무 작거나 dt_sec이 부적절할 수 있음")

  history.append((t, position.copy(), velocity.copy()))
  return {"history": history, "final_position": position, "final_velocity": velocity,
          "final_time_sec": t, "num_steps": step_count}


def edelbaum_delta_v(r1_km, r2_km, mu=EARTH_MU_KM3_S2):
  """평면 내 원궤도 간 저추력 전이의 Edelbaum 근사: Δv = |v_circ(r1) - v_circ(r2)|."""
  return abs(circular_orbit_speed(r1_km, mu) - circular_orbit_speed(r2_km, mu))


def demo_spiral_reaches_target_radius(thrust_accel_km_s2=5e-6):
  """LEO 고도(500km)에서 800km까지 접선 추력으로 나선 전이시켜, 도달한 궤도의
  반장축이 목표 반지름과 일치하고 이심률이 여전히 작게(원궤도 유지) 남아있는지
  확인한다."""
  print("=" * 70)
  print("[1] 저추력 나선 전이: 목표 반지름 도달과 원궤도 유지 검증")
  print("=" * 70)
  r1 = EARTH_RADIUS_KM + 500.0
  r2 = EARTH_RADIUS_KM + 800.0
  dt_sec = 10.0

  print(f"출발 반지름={r1:.1f}km(고도 500km), 목표 반지름={r2:.1f}km(고도 800km)")
  print(f"추력 가속도={thrust_accel_km_s2:.2e}km/s², 스텝={dt_sec}s\n")

  result = spiral_transfer(r1, r2, thrust_accel_km_s2, dt_sec)
  elements = state_vector_to_orbital_elements(result["final_position"], result["final_velocity"])

  achieved_a = elements["semi_major_axis_km"]
  achieved_e = elements["eccentricity"]
  transfer_time_hr = result["final_time_sec"] / 3600

  print(f"도달 시점: {transfer_time_hr:.2f}시간 ({result['num_steps']}스텝)")
  print(f"도달 궤도: 반장축={achieved_a:.2f}km(목표 {r2:.1f}km), 이심률={achieved_e:.6f}")

  a_error_pct = abs(achieved_a - r2) / r2 * 100
  assert a_error_pct < 1.0, "도달한 반장축은 목표 반지름의 1% 이내여야 함"
  assert achieved_e < 0.01, "접선 추력은 이심률을 거의 늘리지 않아야 함(원궤도 유지)"
  print(f"\n반장축 오차: {a_error_pct:.4f}%")
  print("\n(접선 추력이 궤도를 거의 원형으로 유지하면서 반지름만 서서히 키운다는 것을")
  print(" 실제 RK4 적분으로 확인했다.)")
  return {"r1_km": r1, "r2_km": r2, "thrust_accel_km_s2": thrust_accel_km_s2,
          "transfer_time_sec": result["final_time_sec"], "num_steps": result["num_steps"],
          "achieved_a_km": achieved_a, "achieved_e": achieved_e, "a_error_pct": a_error_pct,
          "history": result["history"]}


def demo_edelbaum_approximation_matches_numerical_integration(thrust_accel_km_s2=5e-6):
  """이 스크립트의 핵심 주장: 나선 전이를 "추력 가속도 x 걸린 시간"으로 계산한
  델타-V가, 출발/도착 원궤도 속력 차이로 얻는 Edelbaum 근사와 거의 일치해야
  한다."""
  print("\n" + "=" * 70)
  print("[2] Edelbaum 근사 vs 수치적분 델타-V 비교")
  print("=" * 70)
  r1 = EARTH_RADIUS_KM + 500.0
  r2 = EARTH_RADIUS_KM + 800.0
  dt_sec = 10.0

  result = spiral_transfer(r1, r2, thrust_accel_km_s2, dt_sec)
  numerical_delta_v = thrust_accel_km_s2 * result["final_time_sec"]
  approx_delta_v = edelbaum_delta_v(r1, r2)

  print(f"수치적분 델타-V(가속도 x 시간): {numerical_delta_v * 1000:.4f} m/s")
  print(f"Edelbaum 근사 델타-V(|v1-v2|): {approx_delta_v * 1000:.4f} m/s")

  relative_error_pct = abs(numerical_delta_v - approx_delta_v) / approx_delta_v * 100
  print(f"상대 오차: {relative_error_pct:.2f}%")

  assert relative_error_pct < 10.0, "수치적분 델타-V는 Edelbaum 근사의 10% 이내여야 함"
  print("\n(가속도 x 시간으로 계산한 실제 소요 델타-V가, 원궤도 속력 차이만으로 구한")
  print(" 닫힌 형태 근사와 거의 일치한다 — 접선 추력 모델의 표준 근사식이 이 RK4")
  print(" 시뮬레이션으로 재현된다.)")
  return {"numerical_delta_v_km_s": numerical_delta_v, "edelbaum_delta_v_km_s": approx_delta_v,
          "relative_error_pct": relative_error_pct}


def demo_low_thrust_vs_hohmann_time_tradeoff(thrust_accel_km_s2=5e-6):
  """같은 반지름 쌍에 대해 호만 전이(임펄스)와 저추력 전이(연속 추력)의 델타-V와
  소요 시간을 나란히 비교한다 — 델타-V는 비슷한 수준이어도 시간이 자릿수로
  다르다는 트레이드오프를 직접 보여준다."""
  print("\n" + "=" * 70)
  print("[3] 저추력 vs 호만 전이: 델타-V는 비슷해도 시간은 자릿수가 다르다")
  print("=" * 70)
  r1 = EARTH_RADIUS_KM + 500.0
  r2 = EARTH_RADIUS_KM + 800.0

  _dv1, _dv2, hohmann_total_dv = hohmann_transfer_delta_v(r1, r2)
  hohmann_time_sec = hohmann_transfer_time(r1, r2)

  result = spiral_transfer(r1, r2, thrust_accel_km_s2, 10.0)
  low_thrust_dv = thrust_accel_km_s2 * result["final_time_sec"]

  print(f"{'':>12}{'델타-V(m/s)':>16}{'소요 시간':>16}")
  print(f"{'호만 전이':>12}{hohmann_total_dv * 1000:>16.2f}{hohmann_time_sec / 3600:>13.2f}시간")
  print(f"{'저추력 전이':>12}{low_thrust_dv * 1000:>16.2f}{result['final_time_sec'] / 3600:>13.2f}시간")

  time_ratio = result["final_time_sec"] / hohmann_time_sec
  print(f"\n저추력이 호만 전이보다 {time_ratio:.0f}배 더 오래 걸린다")

  assert time_ratio > 5, "저추력 전이는 호만 전이보다 최소 5배 이상 오래 걸려야 함"
  print("\n(델타-V 예산만 보면 저추력이 불리하지 않지만, 실제로 목표 궤도에 도달하는")
  print(" 데 걸리는 시간은 호만 전이가 압도적으로 짧다 — 시급한 임무는 화학 추진,")
  print(" 시간 여유가 있는 임무는 전기추진을 쓰는 실제 이유다.)")
  return {"r1_km": r1, "r2_km": r2, "hohmann_delta_v_km_s": hohmann_total_dv,
          "hohmann_time_sec": hohmann_time_sec, "low_thrust_delta_v_km_s": low_thrust_dv,
          "low_thrust_time_sec": result["final_time_sec"], "time_ratio": time_ratio}


def demo_transfer_time_vs_thrust_magnitude():
  """추력 가속도 크기를 바꿔가며 목표 반지름 도달 시간이 어떻게 변하는지
  확인한다 — 추력이 커질수록 전이 시간이 대략 반비례로 줄어드는 경향을 보인다."""
  print("\n" + "=" * 70)
  print("[4] 추력 크기에 따른 전이 시간 변화")
  print("=" * 70)
  r1 = EARTH_RADIUS_KM + 500.0
  r2 = EARTH_RADIUS_KM + 800.0
  thrust_values = [2e-6, 5e-6, 1e-5, 2e-5]

  rows = []
  print(f"  {'추력(km/s²)':>16}{'전이시간(시간)':>18}")
  for thrust_accel in thrust_values:
    result = spiral_transfer(r1, r2, thrust_accel, 10.0)
    time_hr = result["final_time_sec"] / 3600
    print(f"  {thrust_accel:>16.2e}{time_hr:>18.2f}")
    rows.append({"thrust_accel_km_s2": thrust_accel, "transfer_time_hr": time_hr})

  assert rows[-1]["transfer_time_hr"] < rows[0]["transfer_time_hr"], (
      "가장 큰 추력의 전이 시간이 가장 작은 추력보다 짧아야 함"
  )
  print("\n(추력 가속도가 커질수록 같은 반지름 변화에 필요한 시간이 줄어든다 — 총")
  print(" 임펄스(시간 x 가속도)가 거의 일정하게 유지되는 경향을 보여준다.)")
  return rows


def parse_args():
  parser = argparse.ArgumentParser(description="저추력 전기추진 접선 연속 추력에 의한 원궤도 나선 전이 시뮬레이션")
  parser.add_argument("--thrust-accel-km-s2", type=float, default=5e-6,
                       help="접선 추력 가속도 크기(km/s²), 기본값: 5e-6 (전형적인 이온엔진 수준)")
  parser.add_argument("--initial-altitude-km", type=float, default=500.0, help="출발 궤도 고도(km), 기본값: 500km")
  parser.add_argument("--target-altitude-km", type=float, default=800.0, help="목표 궤도 고도(km), 기본값: 800km")
  return parser.parse_args()


def main():
  args = parse_args()

  spiral_result = demo_spiral_reaches_target_radius(args.thrust_accel_km_s2)
  edelbaum_result = demo_edelbaum_approximation_matches_numerical_integration(args.thrust_accel_km_s2)
  tradeoff_result = demo_low_thrust_vs_hohmann_time_tradeoff(args.thrust_accel_km_s2)
  thrust_rows = demo_transfer_time_vs_thrust_magnitude()

  r1 = EARTH_RADIUS_KM + args.initial_altitude_km
  r2 = EARTH_RADIUS_KM + args.target_altitude_km
  print("\n" + "=" * 70)
  print(f"[사용자 지정] 고도 {args.initial_altitude_km}km -> {args.target_altitude_km}km, "
        f"추력={args.thrust_accel_km_s2:.2e}km/s²")
  print("=" * 70)
  custom_result = spiral_transfer(r1, r2, args.thrust_accel_km_s2, 10.0)
  custom_dv = args.thrust_accel_km_s2 * custom_result["final_time_sec"]
  print(f"소요 시간: {custom_result['final_time_sec'] / 3600:.2f}시간, "
        f"델타-V: {custom_dv * 1000:.2f}m/s")

  results_dir = os.path.join(_ROOT_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  spiral_csv = os.path.join(results_dir, "low_thrust_spiral_trajectory.csv")
  with open(spiral_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_sec", "x_km", "y_km", "z_km", "radius_km"])
    for t, position, _velocity in spiral_result["history"]:
      writer.writerow([f"{t:.2f}", f"{position[0]:.4f}", f"{position[1]:.4f}", f"{position[2]:.4f}",
                        f"{np.linalg.norm(position):.4f}"])
  print(f"\n[기록] 나선 전이 궤적 저장됨 → {spiral_csv}")

  summary_csv = os.path.join(results_dir, "low_thrust_transfer_summary.csv")
  with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["r1_km", "r2_km", "thrust_accel_km_s2", "transfer_time_sec", "achieved_a_km",
                      "achieved_e", "a_error_pct", "numerical_delta_v_km_s", "edelbaum_delta_v_km_s",
                      "edelbaum_relative_error_pct", "hohmann_delta_v_km_s", "hohmann_time_sec", "time_ratio"])
    writer.writerow([spiral_result["r1_km"], spiral_result["r2_km"], spiral_result["thrust_accel_km_s2"],
                      f"{spiral_result['transfer_time_sec']:.2f}", f"{spiral_result['achieved_a_km']:.4f}",
                      f"{spiral_result['achieved_e']:.6f}", f"{spiral_result['a_error_pct']:.4f}",
                      f"{edelbaum_result['numerical_delta_v_km_s']:.6f}",
                      f"{edelbaum_result['edelbaum_delta_v_km_s']:.6f}",
                      f"{edelbaum_result['relative_error_pct']:.4f}",
                      f"{tradeoff_result['hohmann_delta_v_km_s']:.6f}",
                      f"{tradeoff_result['hohmann_time_sec']:.2f}", f"{tradeoff_result['time_ratio']:.2f}"])
  print(f"[기록] 저추력 전이 요약 결과 저장됨 → {summary_csv}")

  thrust_csv = os.path.join(results_dir, "low_thrust_transfer_time_vs_thrust.csv")
  with open(thrust_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["thrust_accel_km_s2", "transfer_time_hr"])
    for row in thrust_rows:
      writer.writerow([row["thrust_accel_km_s2"], f"{row['transfer_time_hr']:.4f}"])
  print(f"[기록] 추력 크기별 전이시간 결과 저장됨 → {thrust_csv}")


if __name__ == "__main__":
  main()
