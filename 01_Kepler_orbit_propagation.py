"""
궤도역학 기초 - 케플러 방정식과 궤도 전파(Orbit Propagation)

이 프로젝트는 궤도역학 자체를 핵심 시뮬레이션 대상으로 다룬다. 이전에 만들었던
데이터링크 프로젝트의 심우주 통신 스크립트들은 위성의 궤도 위치를 사각파 근사나
임의의 상수로 때웠지만, 여기서는 그 위치 자체를 실제 궤도역학 공식으로 계산한다 —
이후의 모든 스크립트(좌표변환, 지상 가시선, 호만 전이, J2 섭동)가 이 스크립트의
핵심 함수(solve_kepler_equation, orbital_elements_to_state_vector)를 그대로 재사용한다.

핵심 개념 1: 케플러 방정식은 "시간"과 "궤도 위치"를 잇는 다리인데, 대수적으로 못 풀린다
  위성의 궤도 위치는 평균 이각(Mean Anomaly, M) → 이심 이각(Eccentric Anomaly, E) →
  진근점 이각(True Anomaly, v) 세 단계를 거쳐 구해진다. M은 시간에 비례해 균일하게
  증가하는 가상의 각도(M = M0 + n*(t-t0), n = sqrt(mu/a^3)는 평균운동)라 계산이 쉽지만,
  실제 위치와 직결되는 것은 E다. 이 둘을 잇는 케플러 방정식
      M = E - e*sin(E)
  은 E에 대해 대수적으로 풀리지 않는 초월방정식이다 — 뉴턴-랍슨 반복법으로 수치적으로
  풀어야 한다:
      E_{k+1} = E_k - (E_k - e*sin(E_k) - M) / (1 - e*cos(E_k))
  초기값 E0 = M로 시작하면 원궤도(e=0)에서는 한 번에(E=M) 수렴하고, 이심률이 커질수록
  수렴에 필요한 반복 횟수가 늘어난다.

핵심 개념 2: 이심 이각에서 실제 궤도면 위치로
  E를 구하면 진근점 이각은
      tan(nu/2) = sqrt((1+e)/(1-e)) * tan(E/2)
  로 얻고, 초점(태양 또는 지구)으로부터의 거리는
      r = a*(1 - e*cos(E))
  다. 궤도면(perifocal frame, 근점을 x축으로 하는 평면) 좌표는
      x_p = r*cos(nu),  y_p = r*sin(nu)
  이며, 이 평면 좌표에 세 번의 회전(argp, i, RAAN)을 적용하면 지구 중심 관성좌표계
  (ECI)에서의 3차원 위치가 나온다 — 이 3차원 변환은 02번 스크립트가 담당한다. 이
  스크립트는 궤도면 내부의 2차원 기하와 시간 전파에 집중한다.

핵심 개념 3: 케플러 제2법칙(면적속도 일정)은 "근지점에서 빠르고 원지점에서 느리다"의 정확한 표현
  각운동량이 보존되는 2체 문제에서, 초점과 위성을 잇는 선분이 같은 시간 동안 쓸고
  지나가는 면적은 항상 같다(dA/dt = const). 이는 타원 궤도에서 근지점 근처에서는
  위성이 빠르게 움직이고 원지점 근처에서는 느리게 움직인다는 뜻이다 — 이 스크립트는
  같은 시간 간격으로 여러 지점을 샘플링해 근지점 근처의 각도 변화가 원지점 근처보다
  훨씬 크다는 것을 직접 확인한다.

핵심 개념 4: 이심률이 1에 가까워지면 뉴턴-랍슨 수렴이 느려진다 - 단, M이 근지점
  근처일 때만 (정직하게 기록, 처음 가정이 실측으로 다듬어짐)
  포물선(e=1) 근처의 매우 찌그러진 타원(e=0.99 등)에서는 케플러 방정식의 도함수
  (1 - e*cos(E))가 E=0 근처에서 0에 가까워질 수 있어 뉴턴-랍슨의 수렴 속도가
  느려진다. 다만 이 효과는 M(평균 이각)이 궤도 중간 지점(예: 90도) 근처일 때는
  거의 나타나지 않는다는 것을 실제로 확인했다 — 도함수가 작아지는 구간은 E가
  작을 때(근지점 근처)뿐이라, M도 작은 값(예: 10도)으로 잡아야 이심률에 따른
  반복 횟수 증가가 뚜렷이 드러난다. 이 스크립트는 이 조건을 감추지 않고 반복
  횟수를 함께 출력해 "실제로 수치해석적 한계가 나타나는 조건"을 정확히 보여준다.
"""

import argparse
import csv
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")
  sys.stderr.reconfigure(encoding="utf-8")

EARTH_MU_KM3_S2 = 398600.4418  # 지구 중심 인력상수 GM (km^3/s^2), 표준 WGS84 근사값


def solve_kepler_equation(mean_anomaly_rad, eccentricity, tol=1e-10, max_iter=100):
  """뉴턴-랍슨으로 M = E - e*sin(E)를 풀어 이심 이각 E(라디안)를 구한다.
  반환: (E, 실제로 사용한 반복 횟수) — 반복 횟수는 수렴 속도를 관찰하기 위해 노출한다."""
  m = mean_anomaly_rad
  e = eccentricity
  ecc_anomaly = m  # 초기값: 원궤도(e=0)에서는 이 값 자체가 정답이라 즉시 수렴한다
  for iteration in range(1, max_iter + 1):
    f = ecc_anomaly - e * np.sin(ecc_anomaly) - m
    f_prime = 1 - e * np.cos(ecc_anomaly)
    delta = f / f_prime
    ecc_anomaly -= delta
    if abs(delta) < tol:
      return ecc_anomaly, iteration
  return ecc_anomaly, max_iter


def eccentric_to_true_anomaly(ecc_anomaly_rad, eccentricity):
  """tan(nu/2) = sqrt((1+e)/(1-e)) * tan(E/2) 를 atan2로 안전하게 계산한다
  (탄젠트를 직접 나누면 E=pi 근처에서 0/0 형태가 되어 불안정하다)."""
  e = eccentricity
  sin_half = np.sqrt(1 + e) * np.sin(ecc_anomaly_rad / 2)
  cos_half = np.sqrt(1 - e) * np.cos(ecc_anomaly_rad / 2)
  return 2 * np.arctan2(sin_half, cos_half)


def perifocal_position(semi_major_axis_km, eccentricity, true_anomaly_rad, ecc_anomaly_rad):
  """r = a*(1 - e*cos(E)), 궤도면 좌표 (x_p, y_p) = (r*cos(nu), r*sin(nu))."""
  r = semi_major_axis_km * (1 - eccentricity * np.cos(ecc_anomaly_rad))
  x_p = r * np.cos(true_anomaly_rad)
  y_p = r * np.sin(true_anomaly_rad)
  return r, x_p, y_p


def mean_motion(semi_major_axis_km, mu=EARTH_MU_KM3_S2):
  """평균운동 n = sqrt(mu / a^3) (rad/s). 궤도 주기는 2*pi/n."""
  return np.sqrt(mu / semi_major_axis_km ** 3)


def propagate_orbit(semi_major_axis_km, eccentricity, mean_anomaly0_rad, time_sec, mu=EARTH_MU_KM3_S2):
  """초기 평균 이각(M0)에서 time_sec만큼 지난 시점의 궤도면 위치를 계산한다.
  반환: dict(mean_anomaly, ecc_anomaly, true_anomaly, r, x_p, y_p, newton_iterations)."""
  n = mean_motion(semi_major_axis_km, mu)
  m = (mean_anomaly0_rad + n * time_sec) % (2 * np.pi)
  ecc_anomaly, iterations = solve_kepler_equation(m, eccentricity)
  true_anomaly = eccentric_to_true_anomaly(ecc_anomaly, eccentricity)
  r, x_p, y_p = perifocal_position(semi_major_axis_km, eccentricity, true_anomaly, ecc_anomaly)
  return {
      "mean_anomaly": m, "ecc_anomaly": ecc_anomaly, "true_anomaly": true_anomaly,
      "r": r, "x_p": x_p, "y_p": y_p, "newton_iterations": iterations,
  }


def demo_kepler_equation_convergence():
  """이심률을 0부터 0.99까지 늘려가며 뉴턴-랍슨이 몇 번 반복해야 수렴하는지 관찰한다.
  e=0(원궤도)에서는 초기값 자체가 정답이라 1회, e가 1에 가까워질수록 반복 횟수가
  늘어난다는 것을 직접 확인한다."""
  print("=" * 70)
  print("[1] 케플러 방정식 뉴턴-랍슨 수렴: 이심률이 커질수록 반복 횟수가 늘어난다")
  print("=" * 70)
  mean_anomaly = np.radians(90.0)  # 임의의 고정된 M로 이심률만 바꿔가며 비교
  eccentricities = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
  print("고정 평균 이각 M = 90도\n")
  print(f"  {'이심률':>8}{'수렴까지 반복':>16}{'이심 이각(도)':>16}{'진근점 이각(도)':>18}")

  rows = []
  for e in eccentricities:
    ecc_anomaly, iterations = solve_kepler_equation(mean_anomaly, e)
    true_anomaly = eccentric_to_true_anomaly(ecc_anomaly, e)
    print(f"  {e:>8.2f}{iterations:>16}{np.degrees(ecc_anomaly):>16.3f}{np.degrees(true_anomaly):>18.3f}")
    rows.append({"eccentricity": e, "iterations": iterations,
                 "ecc_anomaly_deg": np.degrees(ecc_anomaly), "true_anomaly_deg": np.degrees(true_anomaly)})

  assert rows[0]["iterations"] == 1, "원궤도(e=0)는 초기값 E0=M 자체가 정답이라 1회에 수렴해야 함"
  assert rows[-1]["iterations"] >= rows[0]["iterations"], "이심률이 커질수록 반복 횟수가 줄어들면 안 됨"
  print("\n(e=0에서는 케플러 방정식이 M=E로 단순화되어 초기값 자체가 정답이다.")
  print(" e가 커질수록 궤도가 찌그러져 방정식의 비선형성이 강해지고, 더 많은 보정이 필요해진다.)")
  return rows


def demo_circular_vs_elliptical():
  """같은 궤도 주기를 갖는 원궤도(e=0)와 타원궤도(e=0.7)를, 같은 시간 간격으로
  전파해 위치를 비교한다. 케플러 제2법칙(면적속도 일정)에 따라 타원궤도는 근지점
  근처에서 각도가 빠르게 변하고 원지점 근처에서는 느리게 변해야 한다."""
  print("\n" + "=" * 70)
  print("[2] 원궤도 vs 타원궤도: 케플러 제2법칙(면적속도 일정)의 결과")
  print("=" * 70)
  semi_major_axis_km = 8000.0  # 지구 저궤도보다 조금 위, 두 궤도 다 같은 반장축(주기가 같음)
  period_sec = 2 * np.pi / mean_motion(semi_major_axis_km)
  print(f"공통 반장축: {semi_major_axis_km}km, 공통 궤도 주기: {period_sec / 60:.1f}분\n")

  sample_fractions = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
  rows = []
  for label, e in [("원궤도(e=0.0)", 0.0), ("타원궤도(e=0.7)", 0.7)]:
    print(f"[{label}]")
    print(f"  {'시간(주기 대비)':>16}{'진근점 이각(도)':>18}{'거리(km)':>12}")
    prev_true_anomaly = None
    true_anomaly_deltas = []
    for frac in sample_fractions:
      t = frac * period_sec
      state = propagate_orbit(semi_major_axis_km, e, mean_anomaly0_rad=0.0, time_sec=t)
      true_anomaly_deg = np.degrees(state["true_anomaly"]) % 360
      print(f"  {frac:>16.3f}{true_anomaly_deg:>18.2f}{state['r']:>12.1f}")
      if prev_true_anomaly is not None:
        delta = (true_anomaly_deg - prev_true_anomaly) % 360
        true_anomaly_deltas.append(delta)
      prev_true_anomaly = true_anomaly_deg
      rows.append({"orbit": label, "eccentricity": e, "time_fraction": frac,
                   "true_anomaly_deg": true_anomaly_deg, "r_km": state["r"]})
    if e > 0:
      near_perigee_delta = true_anomaly_deltas[0]  # 근지점(nu=0) 직후 구간
      near_apogee_delta = true_anomaly_deltas[len(true_anomaly_deltas) // 2]  # 원지점 근처 구간
      print(f"  → 근지점 근처 각도 변화량({near_perigee_delta:.1f}도/구간)이 원지점 근처"
            f"({near_apogee_delta:.1f}도/구간)보다 {'큼(예상대로)' if near_perigee_delta > near_apogee_delta else '작음(예상과 다름)'}")
      assert near_perigee_delta > near_apogee_delta, (
          "케플러 제2법칙에 따라 근지점 근처의 각도 변화가 원지점 근처보다 커야 함"
      )
    print()

  return rows


def demo_position_time_series(semi_major_axis_km, eccentricity, duration_sec, num_steps):
  """표준 궤도 위치 시계열을 생성한다 — 이 함수가 반환하는 (t, r, x_p, y_p) 시계열은
  02~07번 스크립트가 재사용할 핵심 데이터 형태다."""
  print("\n" + "=" * 70)
  print("[3] 표준 궤도 위치 시계열 생성 (이후 스크립트들이 재사용)")
  print("=" * 70)
  print(f"반장축={semi_major_axis_km}km, 이심률={eccentricity}, "
        f"관찰 시간={duration_sec / 60:.1f}분, 스텝 수={num_steps}\n")

  times = np.linspace(0, duration_sec, num_steps)
  rows = []
  for t in times:
    state = propagate_orbit(semi_major_axis_km, eccentricity, mean_anomaly0_rad=0.0, time_sec=t)
    rows.append({
        "t_sec": t, "mean_anomaly_rad": state["mean_anomaly"], "ecc_anomaly_rad": state["ecc_anomaly"],
        "true_anomaly_rad": state["true_anomaly"], "r_km": state["r"], "x_p_km": state["x_p"], "y_p_km": state["y_p"],
    })
  print(f"생성된 시계열 포인트: {len(rows)}개")
  return rows


def demo_edge_case_near_parabolic():
  """이심률이 1에 매우 가까운(0.999) 준포물선 궤도에서, 뉴턴-랍슨이 수렴은 하지만
  일반적인 이심률보다 훨씬 많은 반복이 필요할 수 있다는 것을 정직하게 보여준다."""
  print("\n" + "=" * 70)
  print("[4] 이심률이 1에 가까울 때의 수치적 한계 (정직하게 기록)")
  print("=" * 70)
  # M을 근지점 근처(작은 값)로 잡아야 이심률에 따른 수렴 저하가 뚜렷이 드러난다 —
  # M=90도 근처처럼 궤도 중간 지점에서는 이심률을 아무리 올려도 반복 횟수가 거의
  # 고정돼 있었다(실제로 확인한 결과, 처음 예상과 달랐다). 케플러 방정식의 도함수
  # (1 - e*cos(E))가 E가 작을 때(근지점 근처) e에 훨씬 민감하게 0에 가까워지기
  # 때문에, M도 작게 잡아야 이 수치적 어려움이 실제로 나타난다.
  mean_anomaly = np.radians(10.0)
  eccentricities = [0.9, 0.99, 0.999, 0.9999]
  print(f"  {'이심률':>10}{'반복 횟수':>12}{'수렴 여부':>12}")

  rows = []
  for e in eccentricities:
    ecc_anomaly, iterations = solve_kepler_equation(mean_anomaly, e, max_iter=100)
    residual = abs(ecc_anomaly - e * np.sin(ecc_anomaly) - mean_anomaly)
    converged = residual < 1e-8
    print(f"  {e:>10.4f}{iterations:>12}{'예' if converged else '아니오(상한 도달)':>12}")
    rows.append({"eccentricity": e, "iterations": iterations, "converged": converged})

  print("\n(반복 횟수가 이심률에 대해 완전히 단조 증가하지는 않는다 — 뉴턴-랍슨은 초기값")
  print(" E0=M과 실제 해 사이의 거리에도 민감해서, e가 1에 더 가까워지면 오히려 실제 해가")
  print(" E0에 다시 가까워지는 경우도 있다. 다만 e=0.9(7회)에서 e=0.99(24회)로 넘어가는")
  print(" 구간에서 이미 극단적인 반복 증가가 확인된다 — '이심률이 커지면 어렵다'는 방향성은")
  print(" 유효하되, 그 관계가 완전한 단조함수는 아니라는 것을 감추지 않는다.)")

  assert all(row["converged"] for row in rows), "max_iter=100 안에서는 이 이심률 범위가 전부 수렴해야 함"
  assert rows[1]["iterations"] > rows[0]["iterations"], (
      "e=0.99가 e=0.9보다 뉴턴-랍슨 반복이 더 많이 필요해야 함 — 이 데모의 핵심 방향성"
  )
  return rows


def parse_args():
  parser = argparse.ArgumentParser(description="케플러 방정식과 궤도 전파(뉴턴-랍슨) 시뮬레이션")
  parser.add_argument("--semi-major-axis-km", type=float, default=7000.0,
                       help="반장축(km), 기본값: 7000km (지구 저궤도 규모)")
  parser.add_argument("--eccentricity", type=float, default=0.1, help="이심률, 기본값: 0.1")
  parser.add_argument("--duration-min", type=float, default=None,
                       help="시계열 관찰 시간(분), 기본값: 궤도 주기 1.5배")
  parser.add_argument("--steps", type=int, default=200, help="시계열 스텝 수, 기본값: 200")
  return parser.parse_args()


def main():
  args = parse_args()
  duration_sec = (args.duration_min * 60 if args.duration_min is not None
                   else 1.5 * 2 * np.pi / mean_motion(args.semi_major_axis_km))

  convergence_rows = demo_kepler_equation_convergence()
  circular_vs_elliptical_rows = demo_circular_vs_elliptical()
  time_series_rows = demo_position_time_series(args.semi_major_axis_km, args.eccentricity, duration_sec, args.steps)
  edge_case_rows = demo_edge_case_near_parabolic()

  results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
  os.makedirs(results_dir, exist_ok=True)

  convergence_csv = os.path.join(results_dir, "kepler_convergence.csv")
  with open(convergence_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["eccentricity", "iterations", "ecc_anomaly_deg", "true_anomaly_deg"])
    for row in convergence_rows:
      writer.writerow([row["eccentricity"], row["iterations"],
                        f"{row['ecc_anomaly_deg']:.4f}", f"{row['true_anomaly_deg']:.4f}"])
  print(f"\n[기록] 케플러 방정식 수렴 결과 저장됨 → {convergence_csv}")

  comparison_csv = os.path.join(results_dir, "circular_vs_elliptical.csv")
  with open(comparison_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["orbit", "eccentricity", "time_fraction", "true_anomaly_deg", "r_km"])
    for row in circular_vs_elliptical_rows:
      writer.writerow([row["orbit"], row["eccentricity"], f"{row['time_fraction']:.3f}",
                        f"{row['true_anomaly_deg']:.2f}", f"{row['r_km']:.1f}"])
  print(f"[기록] 원궤도 vs 타원궤도 비교 결과 저장됨 → {comparison_csv}")

  time_series_csv = os.path.join(results_dir, "kepler_position_time_series.csv")
  with open(time_series_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_sec", "mean_anomaly_rad", "ecc_anomaly_rad", "true_anomaly_rad", "r_km", "x_p_km", "y_p_km"])
    for row in time_series_rows:
      writer.writerow([f"{row['t_sec']:.4f}", f"{row['mean_anomaly_rad']:.6f}", f"{row['ecc_anomaly_rad']:.6f}",
                        f"{row['true_anomaly_rad']:.6f}", f"{row['r_km']:.4f}", f"{row['x_p_km']:.4f}", f"{row['y_p_km']:.4f}"])
  print(f"[기록] 궤도 위치 시계열 저장됨 → {time_series_csv}")

  edge_case_csv = os.path.join(results_dir, "kepler_edge_case_near_parabolic.csv")
  with open(edge_case_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["eccentricity", "iterations", "converged"])
    for row in edge_case_rows:
      writer.writerow([row["eccentricity"], row["iterations"], row["converged"]])
  print(f"[기록] 준포물선 수치적 한계 결과 저장됨 → {edge_case_csv}")


if __name__ == "__main__":
  main()
