"""
궤도 재진입/대기항력 감쇠(Orbital Decay) - 지수함수 대기밀도 모델로 계산하는 위성 수명

15번(저추력 전기추진)은 접선방향 연속 추력이 원궤도를 서서히 나선형으로 "밀어
올리는" 과정을 다뤘다. 이 스크립트는 정확히 거울상이다 — 저궤도 위성은 완전한
진공이 아니라 극도로 희박하지만 0이 아닌 대기 속을 지나가므로, 속도 반대방향으로
작용하는 미세한 항력이 계속 에너지를 빼앗아 고도가 서서히 떨어지다 결국
재진입한다. 15번의 "접선 추력" 항을 "접선 감속(항력)" 항으로 부호만 반전한
구조로 재사용해, 초기 고도와 위성의 물리적 특성(탄도계수)에 따라 궤도 수명이
어떻게 달라지는지 계산한다.

핵심 개념 1: 항력은 에너지를 계속 빼앗아 반지름을 서서히 줄인다
  15번이 "추력이 에너지를 주입해 반지름을 키운다"를 보여줬다면, 이 스크립트는
  정반대로 항력이 에너지를 빼앗아 반지름을 줄이는 것을 보여준다. 대기밀도를
  스케일고도 H로 지수적으로 감소하는 근사식
      rho(h) = rho0 * exp(-(h-h0)/H)
  로 두고, 항력 가속도는 속도 반대방향으로
      a_drag = -0.5 * rho(h) * v^2 / BC
  (BC = 탄도계수[kg/m^2], 클수록 항력의 영향을 덜 받음)로 계산한다. dE/dt =
  -0.5*rho(h)*v^3/BC < 0이므로 고도는 항상 감소하는 방향으로만 움직인다.

핵심 개념 2: 대기밀도가 지수함수이므로 하강이 갈수록 가속화된다
  고도가 낮아질수록 밀도가 지수적으로 커지므로 항력도 지수적으로 강해진다.
  초기에는 매우 느리게 고도가 떨어지다가, 재진입 고도에 가까워질수록 하강
  속도가 급격히 빨라지는 비선형("runaway") 패턴이 나타난다 — 15번의 거의
  균일한 상승 나선과 뚜렷이 대비되는 지점이다.

핵심 개념 3: 탄도계수가 클수록(무겁고 작은 위성) 수명이 길다
  같은 초기 고도에서 탄도계수 BC를 바꿔가며 재진입까지 걸리는 시간(궤도 수명)을
  비교하면, BC가 클수록(단위 면적당 질량이 커서 항력의 상대적 영향이 작음)
  수명이 길어지는 것을 직접 확인한다.

핵심 개념 4: 초기 고도가 조금만 높아져도 수명이 기하급수적으로 늘어난다
  지수함수 밀도 모델의 직접적 결과로, 초기 고도를 선형으로 올리면 궤도 수명은
  기하급수적으로 늘어난다. 이는 "폐기위성을 완전히 처리하지 않고 낮은 고도로만
  유도해도 자연스럽게 몇 년 안에 재진입한다"는 실제 우주 쓰레기 처리 정책의
  물리적 근거를 수치로 보여준다.

단순화: 이 스크립트는 지수함수 대기밀도라는 단순화된 모델을 쓴다(실제 대기는
태양활동에 따라 크게 요동치는 훨씬 복잡한 모델을 쓴다). 항력만 다루고(J2 등
다른 섭동과의 결합은 다루지 않음), 탄도계수가 궤도 내내 일정하다고 가정한다
(실제로는 자세에 따라 단면적이 달라져 BC가 바뀔 수 있다).

01번(케플러 전파)의 EARTH_MU_KM3_S2, 03번(RK4 수치적분)의 두 물체 중심력
가속도(two_body_acceleration, 수정 없이 재사용)와의 관계는 15번과 동일하다 —
03번 rk4_step의 derivative 클로저가 중심력만 계산하도록 하드코딩되어 있어
그대로 재사용할 수 없으므로, 이 스크립트도 그 구조를 복제하되 항력 감속 항을
더한 자체 RK4 스텝을 새로 정의한다. orbit_math.py의 EARTH_RADIUS_KM은 15번이
로컬에 하드코딩했던 것과 달리 이번에는 정식으로 import한다.
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

_RK4_PATH = os.path.join(_ROOT_DIR, "propagation", "two_body_numerical_integration.py")
_rk4_spec = importlib.util.spec_from_file_location("rk4_module", _RK4_PATH)
rk4_module = importlib.util.module_from_spec(_rk4_spec)
_rk4_spec.loader.exec_module(rk4_module)

_ORBIT_MATH_PATH = os.path.join(_ROOT_DIR, "orbit_math.py")
_orbit_math_spec = importlib.util.spec_from_file_location("orbit_math", _ORBIT_MATH_PATH)
orbit_math = importlib.util.module_from_spec(_orbit_math_spec)
_orbit_math_spec.loader.exec_module(orbit_math)

EARTH_MU_KM3_S2 = kepler.EARTH_MU_KM3_S2
EARTH_RADIUS_KM = orbit_math.EARTH_RADIUS_KM
two_body_acceleration = rk4_module.two_body_acceleration

# 400km 고도(ISS 근방) 기준 대기밀도와 스케일고도의 전형적인 문헌값.
REFERENCE_ALTITUDE_KM = 400.0
REFERENCE_DENSITY_KG_M3 = 5e-12
SCALE_HEIGHT_KM = 60.0


def atmospheric_density(altitude_km, reference_altitude_km=REFERENCE_ALTITUDE_KM,
                         reference_density_kg_m3=REFERENCE_DENSITY_KG_M3, scale_height_km=SCALE_HEIGHT_KM):
  """지수함수 대기밀도 근사: rho(h) = rho0 * exp(-(h-h0)/H)."""
  return reference_density_kg_m3 * np.exp(-(altitude_km - reference_altitude_km) / scale_height_km)


def drag_deceleration(velocity_km_s, altitude_km, ballistic_coefficient_kg_m2):
  """속도 반대방향 항력 감속: a = -0.5*rho(h)*v^2/BC. rho는 kg/m^3, v는 km/s이므로
  m/s로 환산해 가속도를 구한 뒤 다시 km/s^2로 되돌린다(단위 정합성을 명시적으로
  다룸)."""
  velocity = np.array(velocity_km_s, dtype=float)
  speed_km_s = np.linalg.norm(velocity)
  speed_m_s = speed_km_s * 1000.0
  rho = atmospheric_density(altitude_km)
  drag_accel_m_s2 = 0.5 * rho * speed_m_s ** 2 / ballistic_coefficient_kg_m2
  drag_accel_km_s2 = drag_accel_m_s2 / 1000.0
  if speed_km_s < 1e-12:
    return np.zeros(3)
  return -velocity / speed_km_s * drag_accel_km_s2


def rk4_step_with_drag(position_km, velocity_km_s, dt_sec, ballistic_coefficient_kg_m2, mu=EARTH_MU_KM3_S2):
  """15번 rk4_step_with_thrust와 동일한 구조지만, derivative 클로저에 항력
  감속 항을 더한다(부호만 반전된 거울상). 03번 rk4_step의 derivative는 중심력만
  계산하도록 하드코딩되어 있어 그대로 재사용할 수 없으므로 이 함수가 그 구조를
  복제해 확장한다."""
  def derivative(pos, vel):
    altitude = np.linalg.norm(pos) - EARTH_RADIUS_KM
    accel = two_body_acceleration(pos, mu) + drag_deceleration(vel, altitude, ballistic_coefficient_kg_m2)
    return vel, accel

  k1_r, k1_v = derivative(position_km, velocity_km_s)
  k2_r, k2_v = derivative(position_km + dt_sec / 2 * k1_r, velocity_km_s + dt_sec / 2 * k1_v)
  k3_r, k3_v = derivative(position_km + dt_sec / 2 * k2_r, velocity_km_s + dt_sec / 2 * k2_v)
  k4_r, k4_v = derivative(position_km + dt_sec * k3_r, velocity_km_s + dt_sec * k3_v)

  new_position = position_km + dt_sec / 6 * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
  new_velocity = velocity_km_s + dt_sec / 6 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
  return new_position, new_velocity


def circular_orbit_speed(semi_major_axis_km, mu=EARTH_MU_KM3_S2):
  """원궤도 속력: v = sqrt(mu/a)."""
  return np.sqrt(mu / semi_major_axis_km)


def decay_transfer(initial_altitude_km, reentry_altitude_km, ballistic_coefficient_kg_m2, dt_sec,
                    mu=EARTH_MU_KM3_S2, record_every_n_steps=5, max_steps=5_000_000):
  """초기 고도(initial_altitude_km)의 원궤도에서 항력을 계속 받아, 고도가
  reentry_altitude_km 이하가 될 때까지 RK4로 적분한다(15번 spiral_transfer의
  거울상 — 반지름이 줄어들 때까지 반복). 반환: {"history": [(t, position,
  velocity), ...], "final_position", "final_velocity", "final_time_sec",
  "num_steps"}."""
  initial_radius = EARTH_RADIUS_KM + initial_altitude_km
  position = np.array([initial_radius, 0.0, 0.0])
  velocity = np.array([0.0, circular_orbit_speed(initial_radius, mu), 0.0])
  t = 0.0
  step_count = 0
  history = [(t, position.copy(), velocity.copy())]

  altitude = np.linalg.norm(position) - EARTH_RADIUS_KM
  while altitude > reentry_altitude_km:
    position, velocity = rk4_step_with_drag(position, velocity, dt_sec, ballistic_coefficient_kg_m2, mu)
    t += dt_sec
    step_count += 1
    altitude = np.linalg.norm(position) - EARTH_RADIUS_KM
    if step_count % record_every_n_steps == 0:
      history.append((t, position.copy(), velocity.copy()))
    if step_count >= max_steps:
      raise RuntimeError(
          f"재진입 고도({reentry_altitude_km}km)에 {max_steps}스텝 안에 도달하지 못함 — "
          "탄도계수가 너무 크거나 dt_sec이 부적절할 수 있음")

  history.append((t, position.copy(), velocity.copy()))
  return {"history": history, "final_position": position, "final_velocity": velocity,
          "final_time_sec": t, "num_steps": step_count}


def demo_decay_reaches_reentry_altitude(ballistic_coefficient_kg_m2=50.0):
  """고도 200km에서 시작해 항력만으로 재진입 고도(100km, 카르만선 근처)까지
  하강하는지 확인하고, 걸린 시간(궤도 수명)을 계산한다."""
  print("=" * 70)
  print("[1] 대기항력에 의한 궤도 감쇠: 재진입 고도 도달 검증")
  print("=" * 70)
  initial_altitude = 200.0
  reentry_altitude = 100.0
  dt_sec = 30.0

  print(f"초기 고도={initial_altitude}km, 재진입 고도={reentry_altitude}km")
  print(f"탄도계수={ballistic_coefficient_kg_m2}kg/m², 스텝={dt_sec}s\n")

  result = decay_transfer(initial_altitude, reentry_altitude, ballistic_coefficient_kg_m2, dt_sec)
  lifetime_days = result["final_time_sec"] / 86400
  final_altitude = np.linalg.norm(result["final_position"]) - EARTH_RADIUS_KM

  print(f"궤도 수명: {lifetime_days:.2f}일 ({result['num_steps']}스텝)")
  print(f"최종 고도: {final_altitude:.2f}km (목표 {reentry_altitude}km)")

  assert final_altitude <= reentry_altitude, "최종 고도는 재진입 고도 이하여야 함"
  assert lifetime_days > 0, "궤도 수명은 양수여야 함"
  print("\n(대기항력이 계속 에너지를 빼앗아 원궤도가 서서히 나선형으로 하강하다")
  print(" 결국 재진입 고도에 도달한다 — 15번의 상승 나선과 정확히 거울상이다.)")
  return {"initial_altitude_km": initial_altitude, "reentry_altitude_km": reentry_altitude,
          "ballistic_coefficient_kg_m2": ballistic_coefficient_kg_m2, "lifetime_sec": result["final_time_sec"],
          "num_steps": result["num_steps"], "history": result["history"]}


def demo_descent_accelerates_near_reentry(ballistic_coefficient_kg_m2=50.0):
  """이 스크립트의 핵심 주장: 대기밀도가 지수함수이므로 궤적 후반부(재진입
  근처)의 고도 감소율이 전반부보다 훨씬 커야 한다(비선형 가속 하강)."""
  print("\n" + "=" * 70)
  print("[2] 재진입 근처에서 하강이 가속화됨 (지수함수 밀도의 직접적 결과)")
  print("=" * 70)
  result = decay_transfer(200.0, 100.0, ballistic_coefficient_kg_m2, 30.0)
  history = result["history"]

  midpoint_idx = len(history) // 2
  t0, pos0, _v0 = history[0]
  t_mid, pos_mid, _v_mid = history[midpoint_idx]
  t_end, pos_end, _v_end = history[-1]

  alt0 = np.linalg.norm(pos0) - EARTH_RADIUS_KM
  alt_mid = np.linalg.norm(pos_mid) - EARTH_RADIUS_KM
  alt_end = np.linalg.norm(pos_end) - EARTH_RADIUS_KM

  first_half_rate = (alt0 - alt_mid) / (t_mid - t0) if t_mid > t0 else 0.0
  second_half_rate = (alt_mid - alt_end) / (t_end - t_mid) if t_end > t_mid else 0.0

  print(f"전반부: 고도 {alt0:.2f}km → {alt_mid:.2f}km, 감소율 {first_half_rate * 86400:.4f}km/일")
  print(f"후반부: 고도 {alt_mid:.2f}km → {alt_end:.2f}km, 감소율 {second_half_rate * 86400:.4f}km/일")

  rate_ratio = second_half_rate / first_half_rate if first_half_rate > 0 else float("inf")
  print(f"\n후반부/전반부 감소율 배율: {rate_ratio:.2f}배")

  assert second_half_rate > first_half_rate * 2, "후반부 고도 감소율은 전반부의 2배 이상이어야 함(가속 하강)"
  print("\n(고도가 낮아질수록 대기밀도가 지수적으로 커지므로 항력도 강해져,")
  print(" 궤적 후반부로 갈수록 하강 속도가 훨씬 빨라진다 — 15번의 거의 균일한")
  print(" 상승 나선과 달리, 이 하강은 뚜렷하게 비선형적이다.)")
  return {"first_half_rate_km_day": first_half_rate * 86400, "second_half_rate_km_day": second_half_rate * 86400,
          "rate_ratio": rate_ratio}


def demo_larger_ballistic_coefficient_extends_lifetime():
  """같은 초기 고도에서 탄도계수를 바꿔가며 궤도 수명이 늘어나는지(단조
  증가) 확인한다."""
  print("\n" + "=" * 70)
  print("[3] 탄도계수가 클수록 궤도 수명이 길어짐")
  print("=" * 70)
  ballistic_coefficients = [20.0, 50.0, 100.0, 200.0]

  rows = []
  print(f"  {'탄도계수(kg/m²)':>18}{'궤도 수명(일)':>16}")
  for bc in ballistic_coefficients:
    result = decay_transfer(200.0, 100.0, bc, 30.0)
    lifetime_days = result["final_time_sec"] / 86400
    print(f"  {bc:>18.1f}{lifetime_days:>16.2f}")
    rows.append({"ballistic_coefficient_kg_m2": bc, "lifetime_days": lifetime_days})

  assert rows[-1]["lifetime_days"] > rows[0]["lifetime_days"], "가장 큰 탄도계수의 수명이 가장 작은 탄도계수보다 길어야 함"
  print("\n(탄도계수가 클수록(무겁고 단면적이 작은 위성) 항력의 상대적 영향이 작아")
  print(" 궤도 수명이 길어진다 — 가볍고 넓적한 물체(데브리 조각 등)일수록 더 빨리")
  print(" 재진입한다는 실제 관측과 일치하는 경향이다.)")
  return rows


def demo_lifetime_grows_exponentially_with_initial_altitude():
  """이 스크립트의 핵심 주장: 초기 고도를 선형으로 올리면 궤도 수명은
  기하급수적으로 늘어나야 한다(지수함수 밀도 모델의 직접적 결과)."""
  print("\n" + "=" * 70)
  print("[4] 초기 고도가 조금만 높아져도 수명이 기하급수적으로 늘어남")
  print("=" * 70)
  altitudes = [150.0, 180.0, 210.0, 240.0]
  bc = 50.0

  rows = []
  print(f"  {'초기 고도(km)':>16}{'궤도 수명(일)':>16}")
  for alt in altitudes:
    result = decay_transfer(alt, 100.0, bc, 30.0)
    lifetime_days = result["final_time_sec"] / 86400
    print(f"  {alt:>16.1f}{lifetime_days:>16.2f}")
    rows.append({"initial_altitude_km": alt, "lifetime_days": lifetime_days})

  ratio_last = rows[-1]["lifetime_days"] / rows[-2]["lifetime_days"]
  ratio_first = rows[1]["lifetime_days"] / rows[0]["lifetime_days"]
  print(f"\n마지막 30km 구간 수명 증가 배율: {ratio_last:.2f}배 (첫 30km 구간: {ratio_first:.2f}배)")

  assert rows[-1]["lifetime_days"] > rows[0]["lifetime_days"] * 5, (
      "90km 고도 차이만으로 수명이 최소 5배 이상 늘어나야 함(기하급수적 증가)"
  )
  print("\n(초기 고도가 선형으로(150→240km) 올라가는데도 궤도 수명은 훨씬 가파르게")
  print(" 늘어난다 — 이는 '폐기위성을 완전히 처리하지 않고 낮은 고도로만 유도해도")
  print(" 자연스럽게 몇 년 안에 재진입한다'는 실제 우주 쓰레기 정책의 물리적")
  print(" 근거를 수치로 보여준다.)")
  return rows


def parse_args():
  parser = argparse.ArgumentParser(description="지수함수 대기밀도 모델로 계산하는 저궤도 위성의 항력 감쇠 및 궤도 수명")
  parser.add_argument("--initial-altitude-km", type=float, default=200.0, help="초기 궤도 고도(km), 기본값: 200km")
  parser.add_argument("--ballistic-coefficient", type=float, default=50.0,
                       help="탄도계수(kg/m²), 기본값: 50 (전형적인 소형위성 수준)")
  parser.add_argument("--reentry-altitude-km", type=float, default=100.0, help="재진입 판정 고도(km), 기본값: 100km(카르만선 근처)")
  return parser.parse_args()


def main():
  args = parse_args()

  decay_result = demo_decay_reaches_reentry_altitude(args.ballistic_coefficient)
  acceleration_result = demo_descent_accelerates_near_reentry(args.ballistic_coefficient)
  bc_rows = demo_larger_ballistic_coefficient_extends_lifetime()
  altitude_rows = demo_lifetime_grows_exponentially_with_initial_altitude()

  print("\n" + "=" * 70)
  print(f"[사용자 지정] 초기 고도={args.initial_altitude_km}km, "
        f"탄도계수={args.ballistic_coefficient}kg/m², 재진입 고도={args.reentry_altitude_km}km")
  print("=" * 70)
  custom_result = decay_transfer(args.initial_altitude_km, args.reentry_altitude_km,
                                  args.ballistic_coefficient, 30.0)
  custom_lifetime_days = custom_result["final_time_sec"] / 86400
  print(f"궤도 수명: {custom_lifetime_days:.2f}일")

  results_dir = os.path.join(_ROOT_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  trajectory_csv = os.path.join(results_dir, "orbital_decay_trajectory.csv")
  with open(trajectory_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_sec", "x_km", "y_km", "z_km", "altitude_km"])
    for t, position, _velocity in decay_result["history"]:
      altitude = np.linalg.norm(position) - EARTH_RADIUS_KM
      writer.writerow([f"{t:.2f}", f"{position[0]:.4f}", f"{position[1]:.4f}", f"{position[2]:.4f}",
                        f"{altitude:.4f}"])
  print(f"\n[기록] 궤도 감쇠 궤적 저장됨 → {trajectory_csv}")

  summary_csv = os.path.join(results_dir, "orbital_decay_summary.csv")
  with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["initial_altitude_km", "reentry_altitude_km", "ballistic_coefficient_kg_m2",
                      "lifetime_days", "num_steps", "first_half_rate_km_day", "second_half_rate_km_day",
                      "rate_ratio"])
    writer.writerow([decay_result["initial_altitude_km"], decay_result["reentry_altitude_km"],
                      decay_result["ballistic_coefficient_kg_m2"],
                      f"{decay_result['lifetime_sec'] / 86400:.4f}", decay_result["num_steps"],
                      f"{acceleration_result['first_half_rate_km_day']:.6f}",
                      f"{acceleration_result['second_half_rate_km_day']:.6f}",
                      f"{acceleration_result['rate_ratio']:.4f}"])
  print(f"[기록] 궤도 감쇠 요약 결과 저장됨 → {summary_csv}")

  bc_csv = os.path.join(results_dir, "orbital_decay_ballistic_coefficient_vs_lifetime.csv")
  with open(bc_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["ballistic_coefficient_kg_m2", "lifetime_days"])
    for row in bc_rows:
      writer.writerow([row["ballistic_coefficient_kg_m2"], f"{row['lifetime_days']:.4f}"])
  print(f"[기록] 탄도계수별 궤도 수명 결과 저장됨 → {bc_csv}")

  altitude_csv = os.path.join(results_dir, "orbital_decay_altitude_vs_lifetime.csv")
  with open(altitude_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["initial_altitude_km", "lifetime_days"])
    for row in altitude_rows:
      writer.writerow([row["initial_altitude_km"], f"{row['lifetime_days']:.4f}"])
  print(f"[기록] 초기 고도별 궤도 수명 결과 저장됨 → {altitude_csv}")


if __name__ == "__main__":
  main()
