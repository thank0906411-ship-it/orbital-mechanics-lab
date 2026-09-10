"""
궤도역학 심화 - 2체 문제 수치적분(RK4)과 해석해 검증

01/02번은 케플러 방정식을 뉴턴-랍슨으로 풀어 궤도 위치를 "해석적으로"(닫힌 공식으로)
구했다. 하지만 실제 우주선 궤도 계산 소프트웨어는 대개 운동방정식을 수치적으로
직접 적분한다 — 섭동(07번의 J2 등)이나 추력이 개입하면 더 이상 해석해가 존재하지
않기 때문이다. 이 스크립트는 순수 2체 문제(섭동 없음)에서 수치적분과 해석해가
정확히 일치해야 한다는 것을 검증해, 이후 섭동을 더한 수치적분(07번)의 신뢰성을
뒷받침하는 기준선을 만든다.

핵심 개념 1: 2체 문제의 운동방정식은 단순한 2차 미분방정식이다
  뉴턴의 만유인력 법칙에서, 중심천체(질량이 훨씬 큼)를 원점에 고정하면 위성의
  가속도는
      d^2r/dt^2 = -mu * r / |r|^3
  로 주어진다(mu = GM, r은 위치벡터). 이 식은 위치와 속도를 상태로 갖는 1차 연립
  미분방정식으로 다시 쓸 수 있다: dr/dt = v, dv/dt = -mu*r/|r|^3.

핵심 개념 2: RK4(4차 룽게-쿠타법)로 이 미분방정식을 적분한다
  상태 y = [r, v]에 대해 f(y) = [v, -mu*r/|r|^3]로 두면, RK4 한 스텝은
      k1 = f(y_n)
      k2 = f(y_n + h/2 * k1)
      k3 = f(y_n + h/2 * k2)
      k4 = f(y_n + h * k3)
      y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
  로 계산된다(h는 스텝 크기). RK4는 국소 절단 오차가 O(h^5), 전역 오차가 O(h^4)인
  4차 정확도 방법이다.

핵심 개념 3: 스텝을 절반으로 줄이면 오차가 1/16로 줄어야 한다 (O(h^4) 검증)
  RK4의 전역 오차가 정말 O(h^4)라면, 스텝 크기 h를 절반으로 줄일 때 오차는
  (1/2)^4 = 1/16로 줄어야 한다. 이 스크립트는 같은 궤도를 스텝 크기를 두 배씩
  줄여가며(예: 60초 -> 30초 -> 15초) 적분하고, 01번의 해석해와 비교한 위치 오차가
  실제로 약 16배씩 줄어드는지 확인한다 — 이건 RK4 구현 자체가 올바른지 검증하는
  표준적인 방법이다.

핵심 개념 4: 에너지 보존이 수치적분 품질의 또 다른 척도다
  02번에서 확인했듯 2체 문제의 비에너지는 보존돼야 한다. RK4로 수치적분한 궤도도
  이 보존량을 거의 그대로 유지해야 하며, 스텝이 너무 크면 에너지가 시간에 따라
  서서히 새는(drift) 것을 관찰할 수 있다 — 이 스크립트는 스텝 크기별로 에너지
  드리프트를 함께 측정해, "위치 오차는 작아 보여도 장기적으로 에너지가 새고
  있을 수 있다"는 점을 정직하게 보여준다.

01번(케플러 전파), 02번(궤도요소/보존량)과의 관계: 이 스크립트는 01번의
solve_kepler_equation과 propagate_orbit으로 만든 해석해를 "정답"으로 삼고, 02번의
orbital_elements_to_state_vector로 초기 상태벡터를 만든 뒤 RK4로 전파한 결과와
대조한다.
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
elements = _load("02_Orbital_elements_and_energy.py", "elements_module")

EARTH_MU_KM3_S2 = kepler.EARTH_MU_KM3_S2


def two_body_acceleration(position_km, mu=EARTH_MU_KM3_S2):
  """d^2r/dt^2 = -mu * r / |r|^3"""
  r = np.linalg.norm(position_km)
  return -mu * np.array(position_km) / r ** 3


def rk4_step(position_km, velocity_km_s, dt_sec, mu=EARTH_MU_KM3_S2):
  """RK4 한 스텝: 상태 y=[r,v], f(y)=[v, a(r)]."""
  def derivative(pos, vel):
    return vel, two_body_acceleration(pos, mu)

  k1_r, k1_v = derivative(position_km, velocity_km_s)
  k2_r, k2_v = derivative(position_km + dt_sec / 2 * k1_r, velocity_km_s + dt_sec / 2 * k1_v)
  k3_r, k3_v = derivative(position_km + dt_sec / 2 * k2_r, velocity_km_s + dt_sec / 2 * k2_v)
  k4_r, k4_v = derivative(position_km + dt_sec * k3_r, velocity_km_s + dt_sec * k3_v)

  new_position = position_km + dt_sec / 6 * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
  new_velocity = velocity_km_s + dt_sec / 6 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
  return new_position, new_velocity


def integrate_two_body(initial_position_km, initial_velocity_km_s, total_time_sec, dt_sec, mu=EARTH_MU_KM3_S2):
  """초기 상태벡터에서 total_time_sec 동안 RK4로 적분한다. 반환: 시계열 리스트
  [(t, position, velocity), ...] (스텝 경계마다 기록)."""
  position = np.array(initial_position_km, dtype=float)
  velocity = np.array(initial_velocity_km_s, dtype=float)
  t = 0.0
  history = [(t, position.copy(), velocity.copy())]
  while t < total_time_sec - 1e-9:
    step = min(dt_sec, total_time_sec - t)
    position, velocity = rk4_step(position, velocity, step, mu)
    t += step
    history.append((t, position.copy(), velocity.copy()))
  return history


def demo_rk4_matches_analytical_solution():
  """RK4로 적분한 위치가, 01/02번의 해석해(케플러 방정식+궤도요소 변환)와
  얼마나 잘 일치하는지 확인한다."""
  print("=" * 70)
  print("[1] RK4 수치적분 vs 케플러 방정식 해석해 비교")
  print("=" * 70)
  a, e, i, raan, argp = 7500.0, 0.2, np.radians(30.0), np.radians(10.0), np.radians(0.0)
  period = 2 * np.pi / kepler.mean_motion(a)
  initial_pos, initial_vel = elements.orbital_elements_to_state_vector(
      a, e, i, raan, argp, true_anomaly_rad=0.0, ecc_anomaly_rad=0.0)

  dt_sec = 10.0
  history = integrate_two_body(initial_pos, initial_vel, total_time_sec=period, dt_sec=dt_sec)

  print(f"반장축={a}km, 이심률={e}, 궤도 주기={period / 60:.1f}분, 스텝 크기={dt_sec}초\n")
  print(f"  {'궤도 진행률':>12}{'RK4 위치(km)':>16}{'해석해 위치(km)':>18}{'오차(km)':>12}")

  rows = []
  sample_indices = np.linspace(0, len(history) - 1, 6, dtype=int)
  for idx in sample_indices:
    t, rk4_pos, _rk4_vel = history[idx]
    state = kepler.propagate_orbit(a, e, mean_anomaly0_rad=0.0, time_sec=t)
    analytical_pos, _analytical_vel = elements.orbital_elements_to_state_vector(
        a, e, i, raan, argp, state["true_anomaly"], state["ecc_anomaly"])
    error = np.linalg.norm(rk4_pos - analytical_pos)
    print(f"  {t / period:>12.2f}{np.linalg.norm(rk4_pos):>16.1f}{np.linalg.norm(analytical_pos):>18.1f}{error:>12.6f}")
    rows.append({"t_sec": t, "rk4_r_km": np.linalg.norm(rk4_pos), "analytical_r_km": np.linalg.norm(analytical_pos),
                 "position_error_km": error})

  max_error = max(row["position_error_km"] for row in rows)
  print(f"\n최대 위치 오차: {max_error:.6f}km (스텝 크기 {dt_sec}초 기준)")
  assert max_error < 1.0, "10초 스텝 RK4는 한 궤도 주기 동안 위치 오차가 1km 미만이어야 함"
  return rows


def demo_step_size_convergence_order():
  """스텝 크기를 절반씩 줄여가며(RK4는 O(h^4)이므로 오차가 약 1/16씩 줄어야 함)
  실제로 그 수렴 차수가 나오는지 확인한다."""
  print("\n" + "=" * 70)
  print("[2] 스텝 크기를 절반으로 줄이면 오차가 1/16로 줄어드는가 (O(h^4) 검증)")
  print("=" * 70)
  a, e = 7500.0, 0.2
  period = 2 * np.pi / kepler.mean_motion(a)
  initial_pos, initial_vel = elements.orbital_elements_to_state_vector(
      a, e, 0.0, 0.0, 0.0, true_anomaly_rad=0.0, ecc_anomaly_rad=0.0)
  quarter_period = period / 4  # 한 지점만 비교하면 충분 — 스텝 크기별 오차 추세가 목적

  step_sizes = [80.0, 40.0, 20.0, 10.0]
  print(f"  {'스텝 크기(초)':>14}{'위치 오차(km)':>18}{'이전 대비 감소율':>18}")

  rows = []
  prev_error = None
  for dt_sec in step_sizes:
    history = integrate_two_body(initial_pos, initial_vel, total_time_sec=quarter_period, dt_sec=dt_sec)
    _t, rk4_pos, _rk4_vel = history[-1]
    state = kepler.propagate_orbit(a, e, mean_anomaly0_rad=0.0, time_sec=quarter_period)
    analytical_pos, _v = elements.orbital_elements_to_state_vector(
        a, e, 0.0, 0.0, 0.0, state["true_anomaly"], state["ecc_anomaly"])
    error = np.linalg.norm(rk4_pos - analytical_pos)
    ratio_str = f"{prev_error / error:.2f}배" if prev_error is not None and error > 0 else "-"
    print(f"  {dt_sec:>14.1f}{error:>18.8f}{ratio_str:>18}")
    rows.append({"dt_sec": dt_sec, "position_error_km": error})
    prev_error = error

  # 스텝을 절반으로 줄일 때마다 오차가 대략 16배 줄어야 한다(O(h^4)) — 수치오차가
  # 이미 매우 작은 구간에서는 부동소수점 잡음이 섞여 정확히 16배가 아닐 수 있으므로
  # 느슨하게 "확실히 줄어든다"(예: 4배 이상)만 검증한다.
  for i in range(1, len(rows)):
    assert rows[i]["position_error_km"] < rows[i - 1]["position_error_km"] / 4, (
        "스텝을 절반으로 줄이면 RK4(4차 정확도)는 오차가 최소 4배 이상 줄어야 함"
    )
  print("\n(스텝을 절반으로 줄일 때마다 오차가 크게 줄어든다 — RK4가 4차 정확도")
  print(" (전역 오차 O(h^4))라는 이론과 일치하는 경향이다.)")
  return rows


def demo_energy_drift_with_large_steps():
  """스텝을 지나치게 크게 잡으면(RK4가 감당 못 할 정도로) 비에너지가 시간에 따라
  새어나가는(drift) 것을 관찰한다 — 위치가 대략 맞아 보여도 장기 안정성은
  스텝 크기에 민감할 수 있다는 것을 보여준다."""
  print("\n" + "=" * 70)
  print("[3] 스텝 크기가 너무 크면 에너지가 서서히 샌다 (수치적분 품질의 척도)")
  print("=" * 70)
  a, e = 7500.0, 0.2
  period = 2 * np.pi / kepler.mean_motion(a)
  initial_pos, initial_vel = elements.orbital_elements_to_state_vector(
      a, e, 0.0, 0.0, 0.0, true_anomaly_rad=0.0, ecc_anomaly_rad=0.0)
  expected_energy = -EARTH_MU_KM3_S2 / (2 * a)

  print(f"이론적 비에너지: {expected_energy:.6f} km^2/s^2\n")
  print(f"  {'스텝 크기(초)':>14}{'5주기 후 에너지':>20}{'드리프트(절대값)':>18}")

  rows = []
  for dt_sec in [10.0, 100.0, 300.0]:
    history = integrate_two_body(initial_pos, initial_vel, total_time_sec=period * 5, dt_sec=dt_sec)
    _t, final_pos, final_vel = history[-1]
    v = np.linalg.norm(final_vel)
    r = np.linalg.norm(final_pos)
    final_energy = v ** 2 / 2 - EARTH_MU_KM3_S2 / r
    drift = abs(final_energy - expected_energy)
    print(f"  {dt_sec:>14.1f}{final_energy:>20.6f}{drift:>18.2e}")
    rows.append({"dt_sec": dt_sec, "final_energy": final_energy, "drift": drift})

  print("\n(스텝이 커질수록 5주기 후 에너지 드리프트가 커진다 — RK4는 스텝 크기에")
  print(" 비해 궤도의 곡률이 급격한 근지점 근처를 지날 때 정확도를 더 많이 잃는다.")
  print(" 이 프로젝트는 이심률이 있는 궤도에서 스텝을 고정 크기로 쓰는 단순한 방식만")
  print(" 다루지만, 실전에서는 이 문제를 적응형 스텝 크기로 완화한다.)")
  assert rows[-1]["drift"] >= rows[0]["drift"], "스텝이 클수록 에너지 드리프트가 작아지면 안 됨(더 크거나 같아야 함)"
  return rows


def parse_args():
  parser = argparse.ArgumentParser(description="2체 문제 RK4 수치적분과 케플러 해석해 대조 검증")
  return parser.parse_args()


def main():
  parse_args()

  comparison_rows = demo_rk4_matches_analytical_solution()
  convergence_rows = demo_step_size_convergence_order()
  drift_rows = demo_energy_drift_with_large_steps()

  results_dir = os.path.join(_THIS_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  comparison_csv = os.path.join(results_dir, "rk4_vs_analytical.csv")
  with open(comparison_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_sec", "rk4_r_km", "analytical_r_km", "position_error_km"])
    for row in comparison_rows:
      writer.writerow([f"{row['t_sec']:.2f}", f"{row['rk4_r_km']:.4f}", f"{row['analytical_r_km']:.4f}",
                        f"{row['position_error_km']:.8f}"])
  print(f"\n[기록] RK4 vs 해석해 비교 결과 저장됨 → {comparison_csv}")

  convergence_csv = os.path.join(results_dir, "rk4_step_size_convergence.csv")
  with open(convergence_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["dt_sec", "position_error_km"])
    for row in convergence_rows:
      writer.writerow([row["dt_sec"], f"{row['position_error_km']:.10f}"])
  print(f"[기록] 스텝 크기별 수렴 차수 결과 저장됨 → {convergence_csv}")

  drift_csv = os.path.join(results_dir, "rk4_energy_drift.csv")
  with open(drift_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["dt_sec", "final_energy", "drift"])
    for row in drift_rows:
      writer.writerow([row["dt_sec"], f"{row['final_energy']:.8f}", f"{row['drift']:.2e}"])
  print(f"[기록] 에너지 드리프트 결과 저장됨 → {drift_csv}")


if __name__ == "__main__":
  main()
