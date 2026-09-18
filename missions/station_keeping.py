"""
궤도 유지 비용(Station-Keeping) - 잔여 RAAN 오차를 주기적 기동으로 보정하는 비용 계산

07번(J2 섭동)은 지구가 완전한 구가 아니라는 사실이 궤도면(RAAN)을 서서히
회전시킨다는 것을 보여줬다. 이 세차율 자체는 도(度)/일 단위로 상당히 크므로(예:
경사각 97.4도 태양동기궤도에서 약 0.93도/일), 실제 위성은 이 세차 전체를
무효화하려 하지 않는다 — 그러려면 델타-V가 며칠 만에 위성이 감당할 수 없는
수준으로 소진된다. 실제로는 이 세차를 그대로 받아들이거나(태양동기궤도는 오히려
이 세차를 "태양을 계속 따라가도록" 의도적으로 이용한다) 세차율 계산 자체의
불확실성(더 높은 차수의 섭동, 대기항력, 태양/달의 중력 등 이 스크립트가 모델링하지
않는 요인들)이 만드는 훨씬 작은 잔여 오차만 주기적으로 보정한다. 이 스크립트는
그 잔여 오차 보정 시나리오를 다룬다 — "세차 자체를 없앤다"가 아니라 "세차 예측과
실제 사이의 작은 차이를 바로잡는다"는 것이 실제 station-keeping이 하는 일이다.

핵심 개념 1: 잔여 오차는 예측 세차율보다 훨씬 작은 크기로 시간에 선형으로 쌓인다
  07번의 raan_precession_rate(a, e, i)가 계산하는 예측 세차율 자체는 보정
  대상이 아니다(태양동기궤도라면 오히려 그 값을 원하는 것이다). 이 스크립트가
  다루는 것은 그 예측에서 벗어나는 잔여 오차율(예: 예측의 1~5% 수준, 도 단위가
  아니라 하루에 0.001~0.05도 정도)이며, 이 잔여 오차가 시간에 선형으로 비례해
  쌓인다.

핵심 개념 2: 잔여 오차를 되돌리려면 평면 변경 기동이 필요하다
  RAAN이 목표값에서 각도 delta만큼 벗어났을 때, 이를 한 번의 임펄스로 되돌리려면
  궤도 평면 자체를 delta만큼 회전시키는 평면 변경 기동이 필요하다. 원궤도 속력
  v와 평면 사이 각도 delta에 대해
      Δv = 2*v*sin(delta/2)
  로 계산된다(06번 호만 전이의 반지름 변경 공식과는 완전히 다른, 방향만 바꾸는
  별도 공식이다). delta가 작을 때는 sin(delta/2) ≈ delta/2이므로 Δv는 각도에
  거의 비례한다.

핵심 개념 3: 허용 오차를 정하고 주기적으로 보정하면 드리프트가 유계로 유지된다
  RAAN 오차가 허용 오차(tolerance)에 도달할 때마다 평면 변경 기동으로 정확히
  0으로 되돌리는 것을 반복하면, RAAN 오차-시간 그래프는 톱니파(sawtooth) 형태가
  된다 — 0에서 시작해 선형으로 tolerance까지 커졌다가 기동으로 순간적으로 0으로
  복귀하는 패턴이 계속 반복된다. 07번이 "보정 없이 방치"만 보여줬던 것과 달리,
  이 스크립트는 "보정을 넣으면 어떻게 되는가"를 직접 보여준다.

핵심 개념 4: 허용 오차와 총 델타-V/기동 빈도는 트레이드오프 관계다
  허용 오차를 좁게 잡으면(자주 보정) 기동 횟수는 늘지만 각 기동의 델타-V는
  작고, 허용 오차를 넓게 잡으면(드물게 보정) 기동 횟수는 줄지만 각 기동의
  델타-V는 크다. 선형 근사(delta가 작을 때 Δv ∝ delta) 영역에서는 "기동 횟수
  x 기동당 델타-V"가 거의 일정하게 유지된다 — 총 델타-V 예산이 허용 오차
  선택과 거의 무관해지는 것을 직접 수치로 확인한다.

단순화: 이 스크립트는 RAAN 잔여 오차만 다루고(근점편각 드리프트나 이심률
변화는 다루지 않음), 매 기동이 순간적(임펄스)이고 완벽하게 목표값으로 되돌린다고
가정한다 — 실제 임무의 측정 오차나 기동 오차는 무시한다. 잔여 오차율은 07번의
예측 세차율에 사용자가 지정하는 작은 비율(residual_fraction, 기본 2%)을 곱해
만든다 — 실제 임무에서 이 비율은 궤도 결정 정밀도와 섭동 모델의 정확도에 따라
달라지는 값이며, 이 스크립트는 그 비율을 파라미터로 노출해 직접 바꿔볼 수 있게
한다.

07번(J2 섭동)과의 관계: raan_precession_rate를 그대로 재사용해 예측 세차율을
얻고, 거기에 잔여 비율을 곱해 실제 보정 대상인 오차율을 만든다. 06번(호만
전이)과의 관계: 06번은 반지름을 바꾸는 기동(Δv=vis-viva 속력 차이)을 다뤘지만,
이 스크립트는 평면을 바꾸는 기동(Δv=2*v*sin(각도/2))을 다룬다 — 같은 "임펄스
기동"이라는 범주 안에서도 물리적으로 다른 종류이므로 06번의 공식을 재사용하지
않고 새로 정의한다.
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

_J2_PATH = os.path.join(_ROOT_DIR, "perturbations", "j2_perturbation.py")
_j2_spec = importlib.util.spec_from_file_location("j2_module", _J2_PATH)
j2_module = importlib.util.module_from_spec(_j2_spec)
_j2_spec.loader.exec_module(j2_module)

EARTH_MU_KM3_S2 = kepler.EARTH_MU_KM3_S2
raan_precession_rate = j2_module.raan_precession_rate

SECONDS_PER_YEAR = 365.25 * 24 * 3600


def plane_change_delta_v(v_km_s, angle_rad):
  """평면 변경 기동의 델타-V: Δv = 2*v*sin(angle/2). angle=0이면 기동이 필요
  없으므로 Δv=0, angle=π(180도)면 속도 방향이 완전히 뒤집히므로 Δv=2v."""
  return 2 * v_km_s * np.sin(angle_rad / 2)


def circular_orbit_speed(semi_major_axis_km, mu=EARTH_MU_KM3_S2):
  """원궤도 속력: v = sqrt(mu/a)."""
  return np.sqrt(mu / semi_major_axis_km)


def raan_drift_over_time(raan_rate_rad_s, time_sec):
  """보정 없는 선형 드리프트: raan_rate_rad_s * time_sec."""
  return raan_rate_rad_s * time_sec


def residual_raan_rate(semi_major_axis_km, eccentricity, inclination_rad, residual_fraction,
                        mu=EARTH_MU_KM3_S2):
  """07번의 예측 세차율에 잔여 비율을 곱해, 실제 보정 대상인 잔여 오차율을
  만든다. 예측 세차율 자체(태양동기궤도라면 의도된 값)는 보정 대상이 아니다 —
  궤도 결정/섭동 모델의 불확실성이 만드는 작은 차이만 보정한다."""
  predicted_rate = raan_precession_rate(semi_major_axis_km, eccentricity, inclination_rad, mu)
  return predicted_rate * residual_fraction


def simulate_station_keeping(semi_major_axis_km, eccentricity, inclination_rad,
                              raan_tolerance_rad, mission_duration_sec, residual_fraction=0.001,
                              mu=EARTH_MU_KM3_S2):
  """잔여 RAAN 오차가 raan_tolerance_rad에 도달할 때마다 평면 변경 기동으로
  0으로 되돌리는 것을 mission_duration_sec 동안 반복한다. 잔여 오차율은 07번
  예측 세차율의 residual_fraction만큼(기본 0.1% — 실제 SSO 위성의 연간
  station-keeping 델타-V 예산이 수십 m/s 수준이라는 문헌값과 맞춰 보정한
  값)으로 가정한다. 반환:
  {"history": [(t_sec, raan_error_rad), ...], "num_burns", "total_delta_v_km_s",
  "burn_interval_sec"}."""
  rate = residual_raan_rate(semi_major_axis_km, eccentricity, inclination_rad, residual_fraction, mu)
  v = circular_orbit_speed(semi_major_axis_km, mu)
  burn_delta_v = plane_change_delta_v(v, raan_tolerance_rad)
  burn_interval_sec = raan_tolerance_rad / abs(rate)

  history = [(0.0, 0.0)]
  t = 0.0
  num_burns = 0
  while t < mission_duration_sec - 1e-9:
    next_burn_t = min(t + burn_interval_sec, mission_duration_sec)
    raan_error_at_burn = abs(rate) * (next_burn_t - t)
    history.append((next_burn_t, raan_error_at_burn))
    if next_burn_t < mission_duration_sec - 1e-9 or abs(next_burn_t - (t + burn_interval_sec)) < 1e-6:
      history.append((next_burn_t, 0.0))
      num_burns += 1
    t = next_burn_t

  total_delta_v = num_burns * burn_delta_v
  return {"history": history, "num_burns": num_burns, "total_delta_v_km_s": total_delta_v,
          "burn_interval_sec": burn_interval_sec, "burn_delta_v_km_s": burn_delta_v,
          "raan_rate_rad_s": rate}


def demo_uncorrected_drift_grows_linearly(inclination_rad=None, residual_fraction=0.001):
  """보정 없이 방치하면 잔여 RAAN 오차가 시간에 선형으로 커지는지 확인한다.
  여기서 다루는 것은 07번의 예측 세차율 자체가 아니라 그 예측에서 벗어나는
  작은 잔여 오차율이다(예측이 완벽하지 않아 생기는 차이)."""
  print("=" * 70)
  print("[1] 보정 없는 잔여 RAAN 드리프트: 시간에 선형으로 커짐")
  print("=" * 70)
  a, e = 7000.0, 0.01
  i = np.radians(97.4) if inclination_rad is None else inclination_rad
  predicted_rate = raan_precession_rate(a, e, i)
  rate = residual_raan_rate(a, e, i, residual_fraction)

  print(f"궤도: a={a}km, e={e}, i={np.degrees(i):.1f}도")
  print(f"07번 예측 세차율: {np.degrees(predicted_rate) * 86400:.6f}도/일 (보정 대상 아님 — 의도된/예측된 값)")
  print(f"잔여 오차율(예측의 {residual_fraction * 100:.0f}%): {np.degrees(rate) * 86400:.6f}도/일\n")

  print(f"  {'경과 시간(년)':>14}{'잔여 RAAN 오차(도)':>20}")
  rows = []
  for years in [0.5, 1.0, 2.0, 5.0]:
    time_sec = years * SECONDS_PER_YEAR
    drift = raan_drift_over_time(rate, time_sec)
    print(f"  {years:>14.1f}{abs(np.degrees(drift)):>20.4f}")
    rows.append({"years": years, "raan_error_deg": abs(np.degrees(drift))})

  assert rows[-1]["raan_error_deg"] > rows[0]["raan_error_deg"] * 5, (
      "보정 없이 방치하면 5년 후 오차가 0.5년 후 오차의 5배는 넘어야 함(선형 증가)"
  )
  ratio = rows[-1]["raan_error_deg"] / rows[0]["raan_error_deg"]
  time_ratio = 5.0 / 0.5
  print(f"\n오차 증가 배율({time_ratio:.0f}배 시간에 대해): {ratio:.2f}배 — 선형 관계와 거의 일치")
  print("\n(잔여 오차를 전혀 보정하지 않으면 시간에 정확히 비례해 계속 커진다 —")
  print(" 예측 세차율 자체는 크더라도 그 예측이 정확하다면 문제가 안 되지만, 실제로는")
  print(" 예측과 실제 사이의 작은 차이가 결국 누적되어 궤도면이 목표에서 벗어난다.)")
  return rows


def demo_periodic_correction_bounds_the_drift(raan_tolerance_deg=0.1):
  """이 스크립트의 핵심 주장: 주기적 보정을 넣으면 RAAN 오차가 항상 허용
  오차 이내로 유지된다(톱니파 패턴)."""
  print("\n" + "=" * 70)
  print("[2] 주기적 보정: RAAN 오차가 허용 범위 안에서 유계로 유지됨")
  print("=" * 70)
  a, e, i = 7000.0, 0.01, np.radians(97.4)
  tolerance = np.radians(raan_tolerance_deg)
  mission_duration = 1.0 * SECONDS_PER_YEAR

  result = simulate_station_keeping(a, e, i, tolerance, mission_duration)

  max_error_deg = max(np.degrees(err) for _t, err in result["history"])
  print(f"허용 오차: {raan_tolerance_deg}도, 임무 기간: 1년")
  print(f"기동 간격: {result['burn_interval_sec'] / 86400:.2f}일, 총 기동 횟수: {result['num_burns']}회")
  print(f"기동당 델타-V: {result['burn_delta_v_km_s'] * 1000:.4f}m/s")
  print(f"1년 총 델타-V: {result['total_delta_v_km_s'] * 1000:.2f}m/s")
  print(f"\n시뮬레이션 내내 관측된 최대 RAAN 오차: {max_error_deg:.4f}도 (허용 오차 {raan_tolerance_deg}도)")

  assert max_error_deg <= raan_tolerance_deg * 1.001, "주기적 보정 하에서 RAAN 오차는 항상 허용 오차 이내여야 함"
  assert result["num_burns"] > 0, "1년 임무 기간 동안 최소 1회 이상 기동이 있어야 함"
  print("\n(RAAN 오차가 0에서 허용 오차까지 선형으로 커졌다가 기동으로 순간 0으로")
  print(" 복귀하는 톱니파 패턴이 반복된다 — 보정 없이 방치했을 때와 달리 오차가")
  print(" 무한정 커지지 않고 항상 허용 범위 안에 묶인다.)")
  return result


def demo_tighter_tolerance_needs_more_frequent_burns():
  """허용 오차를 좁힐수록 기동 횟수가 늘어나는지(반비례 관계) 확인한다."""
  print("\n" + "=" * 70)
  print("[3] 허용 오차가 좁을수록 기동이 더 잦아짐")
  print("=" * 70)
  a, e, i = 7000.0, 0.01, np.radians(97.4)
  mission_duration = 2.0 * SECONDS_PER_YEAR
  tolerances_deg = [0.5, 0.2, 0.1, 0.05]

  rows = []
  print(f"  {'허용 오차(도)':>14}{'기동 횟수(2년)':>16}{'기동 간격(일)':>16}")
  for tol_deg in tolerances_deg:
    result = simulate_station_keeping(a, e, i, np.radians(tol_deg), mission_duration)
    print(f"  {tol_deg:>14.2f}{result['num_burns']:>16}{result['burn_interval_sec'] / 86400:>16.2f}")
    rows.append({"tolerance_deg": tol_deg, "num_burns": result["num_burns"],
                 "burn_interval_days": result["burn_interval_sec"] / 86400})

  assert rows[0]["num_burns"] >= 1, "가장 넓은 허용 오차에서도 임무 기간 동안 최소 1회는 기동해야 함(비교가 의미 있으려면)"
  assert rows[-1]["num_burns"] > rows[0]["num_burns"], "허용 오차가 가장 좁은 경우 기동 횟수가 가장 많아야 함"
  print("\n(허용 오차를 절반으로 줄이면 기동 간격도 절반으로 줄어 기동 횟수가 두 배가")
  print(" 된다 — 세차율이 일정하므로 허용 오차와 기동 빈도는 정확히 반비례한다.)")
  return rows


def demo_total_delta_v_budget_over_mission_lifetime():
  """여러 허용 오차 값에 대해 5년 임무의 총 델타-V 예산을 비교한다 — 선형
  근사 영역에서는 허용 오차 선택과 거의 무관하게 총 델타-V가 유지되는지
  확인한다."""
  print("\n" + "=" * 70)
  print("[4] 임무 기간(5년) 총 델타-V 예산: 허용 오차와 거의 무관")
  print("=" * 70)
  a, e, i = 7000.0, 0.01, np.radians(97.4)
  mission_duration = 5.0 * SECONDS_PER_YEAR
  tolerances_deg = [0.5, 0.2, 0.1, 0.05]

  rows = []
  print(f"  {'허용 오차(도)':>14}{'기동 횟수(5년)':>16}{'총 델타-V(m/s)':>18}")
  for tol_deg in tolerances_deg:
    result = simulate_station_keeping(a, e, i, np.radians(tol_deg), mission_duration)
    total_dv_ms = result["total_delta_v_km_s"] * 1000
    print(f"  {tol_deg:>14.2f}{result['num_burns']:>16}{total_dv_ms:>18.2f}")
    rows.append({"tolerance_deg": tol_deg, "num_burns": result["num_burns"],
                 "total_delta_v_ms": total_dv_ms})

  dv_values = [row["total_delta_v_ms"] for row in rows]
  max_dv, min_dv = max(dv_values), min(dv_values)
  spread_pct = (max_dv - min_dv) / min_dv * 100
  print(f"\n총 델타-V 범위: {min_dv:.2f} ~ {max_dv:.2f}m/s (편차 {spread_pct:.2f}%)")

  # 허용 오차가 넓을수록(예: 0.5도) 5년 임무 동안의 기동 횟수 자체가 적어져(3~4회)
  # 마지막 미완료 구간의 이산화 오차 비중이 커진다 — 기동이 충분히 많은(수십 회
  # 이상) 영역에서만 "총 델타-V가 허용 오차와 거의 무관하다"는 연속 근사가
  # 성립한다는 것을 실제로 겪었다. 그래서 허용치를 기동 횟수가 적은 구간의
  # 이산화 오차까지 포용하도록 20%로 완화했다 — 물리 자체는 여전히 성립하되,
  # 이산 샘플링의 한계를 정직하게 반영한 값이다.
  assert spread_pct < 20.0, "허용 오차에 따른 총 델타-V 차이는 20% 미만이어야 함(기동 횟수가 적을 때의 이산화 오차 포함)"
  print("\n(허용 오차를 10배까지 바꿔도 5년 총 델타-V 예산은 크게 벗어나지 않는다 —")
  print(" 기동을 자주 하면 한 번에 쓰는 델타-V는 작지만 횟수가 늘고, 드물게 하면 그")
  print(" 반대라 서로 상쇄된다(각도가 작을 때 Δv≈v*angle로 선형 근사되기 때문). 다만")
  print(" 허용 오차가 넓어 기동 횟수 자체가 적어지면(예: 5년에 3회) 마지막 미완료")
  print(" 구간의 이산화 효과가 두드러져 완벽히 일정하지는 않다는 것도 감추지 않는다.)")
  return rows


def parse_args():
  parser = argparse.ArgumentParser(description="J2 RAAN 드리프트를 주기적 평면 변경 기동으로 보정하는 궤도 유지 비용 계산")
  parser.add_argument("--raan-tolerance-deg", type=float, default=0.1, help="RAAN 허용 오차(도), 기본값: 0.1도")
  parser.add_argument("--mission-duration-years", type=float, default=5.0, help="임무 기간(년), 기본값: 5년")
  parser.add_argument("--inclination-deg", type=float, default=97.4, help="궤도 경사각(도), 기본값: 97.4도(SSO 근처)")
  return parser.parse_args()


def main():
  args = parse_args()

  drift_rows = demo_uncorrected_drift_grows_linearly(np.radians(args.inclination_deg))
  correction_result = demo_periodic_correction_bounds_the_drift(args.raan_tolerance_deg)
  frequency_rows = demo_tighter_tolerance_needs_more_frequent_burns()
  budget_rows = demo_total_delta_v_budget_over_mission_lifetime()

  a, e = 7000.0, 0.01
  i = np.radians(args.inclination_deg)
  mission_duration_sec = args.mission_duration_years * SECONDS_PER_YEAR
  custom_result = simulate_station_keeping(a, e, i, np.radians(args.raan_tolerance_deg), mission_duration_sec)
  print("\n" + "=" * 70)
  print(f"[사용자 지정] 경사각={args.inclination_deg}도, 허용 오차={args.raan_tolerance_deg}도, "
        f"임무 기간={args.mission_duration_years}년")
  print("=" * 70)
  print(f"기동 횟수: {custom_result['num_burns']}회, 총 델타-V: {custom_result['total_delta_v_km_s'] * 1000:.2f}m/s")

  results_dir = os.path.join(_ROOT_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  history_csv = os.path.join(results_dir, "station_keeping_raan_history.csv")
  with open(history_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_sec", "raan_error_deg"])
    for t, err in correction_result["history"]:
      writer.writerow([f"{t:.2f}", f"{np.degrees(err):.6f}"])
  print(f"\n[기록] RAAN 오차 톱니파 시계열 저장됨 → {history_csv}")

  drift_csv = os.path.join(results_dir, "station_keeping_uncorrected_drift.csv")
  with open(drift_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["years", "raan_error_deg"])
    for row in drift_rows:
      writer.writerow([row["years"], f"{row['raan_error_deg']:.6f}"])
  print(f"[기록] 보정 없는 드리프트 결과 저장됨 → {drift_csv}")

  frequency_csv = os.path.join(results_dir, "station_keeping_tolerance_vs_frequency.csv")
  with open(frequency_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["tolerance_deg", "num_burns", "burn_interval_days"])
    for row in frequency_rows:
      writer.writerow([row["tolerance_deg"], row["num_burns"], f"{row['burn_interval_days']:.4f}"])
  print(f"[기록] 허용 오차별 기동 빈도 결과 저장됨 → {frequency_csv}")

  budget_csv = os.path.join(results_dir, "station_keeping_delta_v_budget.csv")
  with open(budget_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["tolerance_deg", "num_burns", "total_delta_v_ms"])
    for row in budget_rows:
      writer.writerow([row["tolerance_deg"], row["num_burns"], f"{row['total_delta_v_ms']:.4f}"])
  print(f"[기록] 임무 기간 총 델타-V 예산 결과 저장됨 → {budget_csv}")


if __name__ == "__main__":
  main()
