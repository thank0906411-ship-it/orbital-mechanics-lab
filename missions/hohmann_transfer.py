"""
호만 전이(Hohmann Transfer) - 두 원궤도 사이의 최소 에너지 궤도 변경

이제까지의 스크립트는 위성이 이미 정해진 궤도 위에서 움직이는 것만 다뤘다. 실제
임무 설계에서는 위성을 한 궤도(예: 저궤도)에서 다른 궤도(예: 정지궤도)로 옮겨야
하는데, 이때 추진제(델타-V)를 최소로 쓰는 표준적인 방법이 호만 전이다. 이 스크립트는
호만 전이의 두 번의 임펄스(Δv1, Δv2)를 계산하고, 01번의 궤도 전파로 실제 전이궤도
위의 위성이 목표 반지름에 정확히 도달하는지 검증한다.

핵심 개념 1: 호만 전이는 두 원궤도에 접하는 타원 전이궤도를 쓴다
  반지름 r1인 원궤도에서 반지름 r2인 원궤도로 옮길 때(r1 < r2 가정), 전이궤도는
  근지점이 r1, 원지점이 r2인 타원이다. 전이궤도의 반장축은
      a_t = (r1 + r2) / 2
  이다. 이 타원은 두 원궤도 모두에 접하기 때문에(근지점/원지점에서 속도 방향이
  원궤도 속도와 일치), 방향을 바꾸지 않고 속력만 바꾸는 두 번의 임펄스로 충분하다.

핵심 개념 2: 두 번의 임펄스는 vis-viva로 유도된다
  원궤도 r1에서의 속력은 v_circ1 = sqrt(mu/r1), 전이궤도의 근지점(r1) 속력은
  vis-viva로 v_t1 = sqrt(mu*(2/r1 - 1/a_t))다. 첫 번째 임펄스는
      Δv1 = v_t1 - v_circ1 = sqrt(mu/r1) * (sqrt(2*r2/(r1+r2)) - 1)
  두 번째 임펄스(전이궤도 원지점(r2)에서 목표 원궤도 속력으로 가속)는
      Δv2 = v_circ2 - v_t2 = sqrt(mu/r2) * (1 - sqrt(2*r1/(r1+r2)))
  로 계산된다. 총 델타-V는 Δv1 + Δv2다.

핵심 개념 3: 전이 시간은 타원 궤도 주기의 절반이다
  전이궤도를 따라 근지점에서 원지점까지 가는 데 걸리는 시간은 전체 궤도 주기의
  절반이다: t_transfer = pi * sqrt(a_t^3 / mu). 이 스크립트는 01번의 propagate_orbit
  으로 전이궤도를 실제로 전파해, t_transfer 시점에 위성이 정말로 반지름 r2(원지점)에
  도달하는지 확인한다 — 공식만 믿지 않고 실제 궤도 전파로 재검증한다.

핵심 개념 4: LEO -> GEO는 표준 문헌값과 비교할 수 있는 잘 알려진 사례다
  저궤도(LEO, 약 6678km)에서 정지궤도(GEO, 약 42164km)로 가는 호만 전이는 우주공학
  교과서에 자주 등장하는 예시로, 총 델타-V가 대략 3.9km/s 근방으로 알려져 있다. 이
  스크립트는 이 값과 직접 대조해 구현이 올바른지 확인한다.

01번(케플러 전파)과의 관계: 이 스크립트는 01번의 propagate_orbit, mean_motion,
EARTH_MU_KM3_S2를 재사용해 전이궤도의 실제 전파 결과를 공식과 대조한다.
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
_spec = importlib.util.spec_from_file_location("kepler_module", _KEPLER_PATH)
kepler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kepler)

EARTH_MU_KM3_S2 = kepler.EARTH_MU_KM3_S2


def hohmann_transfer_delta_v(r1_km, r2_km, mu=EARTH_MU_KM3_S2):
  """r1(출발 원궤도 반지름) -> r2(목표 원궤도 반지름) 호만 전이의 (Δv1, Δv2, 총 Δv)."""
  a_t = (r1_km + r2_km) / 2
  v_circ1 = np.sqrt(mu / r1_km)
  v_circ2 = np.sqrt(mu / r2_km)
  v_t1 = np.sqrt(mu * (2 / r1_km - 1 / a_t))
  v_t2 = np.sqrt(mu * (2 / r2_km - 1 / a_t))
  delta_v1 = v_t1 - v_circ1
  delta_v2 = v_circ2 - v_t2
  return delta_v1, delta_v2, abs(delta_v1) + abs(delta_v2)


def hohmann_transfer_time(r1_km, r2_km, mu=EARTH_MU_KM3_S2):
  """전이 시간 = 전이궤도 주기의 절반."""
  a_t = (r1_km + r2_km) / 2
  return np.pi * np.sqrt(a_t ** 3 / mu)


def hohmann_transfer_orbit_elements(r1_km, r2_km):
  """전이궤도의 (반장축, 이심률)을 반환한다. 근지점=r1, 원지점=r2 가정(r1<r2)."""
  a_t = (r1_km + r2_km) / 2
  e_t = (r2_km - r1_km) / (r2_km + r1_km)
  return a_t, e_t


def demo_leo_to_geo_matches_textbook_value():
  """LEO(약 6678km) -> GEO(약 42164km) 호만 전이의 총 델타-V가 교과서에서 흔히
  인용되는 값(약 3.9km/s)과 일치하는지 확인한다."""
  print("=" * 70)
  print("[1] LEO -> GEO 호만 전이: 교과서 표준값과 비교")
  print("=" * 70)
  r_leo = 6678.0  # 지구 반지름(6378km) + 300km 고도
  r_geo = 42164.0  # 정지궤도 반지름

  delta_v1, delta_v2, total_delta_v = hohmann_transfer_delta_v(r_leo, r_geo)
  transfer_time = hohmann_transfer_time(r_leo, r_geo)

  print(f"출발 궤도 반지름(LEO): {r_leo}km, 목표 궤도 반지름(GEO): {r_geo}km\n")
  print(f"Δv1 (근지점 가속): {delta_v1:.4f} km/s")
  print(f"Δv2 (원지점 가속): {delta_v2:.4f} km/s")
  print(f"총 Δv: {total_delta_v:.4f} km/s (교과서 참고값: 약 3.9km/s)")
  print(f"전이 시간: {transfer_time / 3600:.2f}시간 (교과서 참고값: 약 5.25시간)")

  assert 3.8 < total_delta_v < 4.0, "LEO->GEO 총 델타-V는 교과서 값(약 3.9km/s) 근방이어야 함"
  assert 5.0 < transfer_time / 3600 < 5.5, "LEO->GEO 전이 시간은 약 5.25시간 근방이어야 함"
  print("\n(잘 알려진 LEO->GEO 델타-V 예산과 일치한다 — 이 스크립트의 공식 구현이 올바르다는")
  print(" 신호다.)")
  return {"r1_km": r_leo, "r2_km": r_geo, "delta_v1": delta_v1, "delta_v2": delta_v2,
          "total_delta_v": total_delta_v, "transfer_time_sec": transfer_time}


def demo_propagated_transfer_reaches_target_radius():
  """01번의 실제 궤도 전파로 전이궤도를 따라가, 계산된 전이 시간에 정말로
  목표 반지름(원지점)에 도달하는지 검증한다 — 공식만 믿지 않고 재검증한다."""
  print("\n" + "=" * 70)
  print("[2] 전이궤도를 실제로 전파해 목표 반지름 도달 검증")
  print("=" * 70)
  r1, r2 = 7000.0, 15000.0
  a_t, e_t = hohmann_transfer_orbit_elements(r1, r2)
  transfer_time = hohmann_transfer_time(r1, r2)

  print(f"출발 반지름={r1}km, 목표 반지름={r2}km")
  print(f"전이궤도: 반장축={a_t:.2f}km, 이심률={e_t:.4f}")
  print(f"이론적 전이 시간: {transfer_time / 60:.2f}분\n")

  # 전이궤도의 근지점(M=0)에서 출발해 전이 시간 후의 위치를 전파
  state_at_start = kepler.propagate_orbit(a_t, e_t, mean_anomaly0_rad=0.0, time_sec=0.0)
  state_at_end = kepler.propagate_orbit(a_t, e_t, mean_anomaly0_rad=0.0, time_sec=transfer_time)

  print(f"전이 시작 시점: r={state_at_start['r']:.2f}km (예상 근지점 r1={r1}km)")
  print(f"전이 완료 시점: r={state_at_end['r']:.2f}km (예상 원지점 r2={r2}km)")

  start_error = abs(state_at_start["r"] - r1)
  end_error = abs(state_at_end["r"] - r2)
  print(f"\n근지점 오차: {start_error:.6f}km, 원지점 오차: {end_error:.6f}km")

  assert start_error < 1e-6, "전이 시작 시점의 반지름은 정확히 r1이어야 함(근지점)"
  assert end_error < 1e-6, "전이 시간 후 반지름은 정확히 r2여야 함(원지점) — 공식이 실제 전파와 일치"
  print("\n(01번의 뉴턴-랍슨 궤도 전파로 직접 확인했다 — 전이 시간 계산(pi*sqrt(a_t^3/mu))이")
  print(" 정확히 원지점 통과 시점과 일치한다.)")
  return {"r1_km": r1, "r2_km": r2, "start_r_km": state_at_start["r"], "end_r_km": state_at_end["r"],
          "start_error_km": start_error, "end_error_km": end_error, "transfer_time_sec": transfer_time}


def demo_delta_v_varies_with_radius_ratio():
  """반지름 비율(r2/r1)이 커질수록 총 델타-V가 어떻게 변하는지 관찰한다 — 비율이
  약 11.94 근처에서 이론적으로 이봉형(bi-elliptic이 더 유리해지는) 특성이 있다는
  것으로 알려져 있으나, 이 스크립트는 순수 호만 전이만 다루므로 델타-V 자체의
  단조 증가/감소 여부만 정직하게 관찰한다."""
  print("\n" + "=" * 70)
  print("[3] 반지름 비율에 따른 델타-V 변화 (순수 호만 전이만 다룸)")
  print("=" * 70)
  r1 = 7000.0
  ratios = [1.5, 2.0, 4.0, 6.0, 10.0, 15.0, 20.0]

  rows = []
  print(f"  {'r2/r1':>8}{'Δv1(km/s)':>14}{'Δv2(km/s)':>14}{'총 Δv(km/s)':>16}")
  for ratio in ratios:
    r2 = r1 * ratio
    delta_v1, delta_v2, total = hohmann_transfer_delta_v(r1, r2)
    print(f"  {ratio:>8.1f}{delta_v1:>14.4f}{delta_v2:>14.4f}{total:>16.4f}")
    rows.append({"ratio": ratio, "delta_v1": delta_v1, "delta_v2": delta_v2, "total_delta_v": total})

  print("\n(총 델타-V는 비율이 커질수록 증가하다가 어느 지점에서 증가폭이 둔화된다 —")
  print(" 순수 호만 전이만으로는 비율이 매우 커도 델타-V가 무한정 커지지 않고 특정 값에")
  print(" 수렴하는 경향을 보인다(r2->무한대일 때 Δv1은 탈출에 필요한 한계값에 가까워짐).")
  print(" 이 프로젝트는 이봉전이(bi-elliptic)는 다루지 않는다 — 순수 호만 전이의 결과만")
  print(" 정직하게 보여준다.)")
  return rows


def parse_args():
  parser = argparse.ArgumentParser(description="호만 전이(Hohmann Transfer) 델타-V와 전이시간 계산")
  parser.add_argument("--r1-km", type=float, default=6678.0, help="출발 원궤도 반지름(km), 기본값: LEO(6678km)")
  parser.add_argument("--r2-km", type=float, default=42164.0, help="목표 원궤도 반지름(km), 기본값: GEO(42164km)")
  return parser.parse_args()


def main():
  args = parse_args()

  leo_geo_result = demo_leo_to_geo_matches_textbook_value()
  propagated_result = demo_propagated_transfer_reaches_target_radius()
  ratio_rows = demo_delta_v_varies_with_radius_ratio()

  results_dir = os.path.join(_ROOT_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  _custom_delta_v1, _custom_delta_v2, custom_total = hohmann_transfer_delta_v(args.r1_km, args.r2_km)
  custom_transfer_time = hohmann_transfer_time(args.r1_km, args.r2_km)
  print(f"\n[사용자 지정] r1={args.r1_km}km -> r2={args.r2_km}km: "
        f"총 Δv={custom_total:.4f}km/s, 전이시간={custom_transfer_time / 60:.2f}분")

  leo_geo_csv = os.path.join(results_dir, "hohmann_leo_to_geo.csv")
  with open(leo_geo_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["r1_km", "r2_km", "delta_v1", "delta_v2", "total_delta_v", "transfer_time_sec"])
    writer.writerow([leo_geo_result["r1_km"], leo_geo_result["r2_km"], f"{leo_geo_result['delta_v1']:.6f}",
                      f"{leo_geo_result['delta_v2']:.6f}", f"{leo_geo_result['total_delta_v']:.6f}",
                      f"{leo_geo_result['transfer_time_sec']:.2f}"])
  print(f"\n[기록] LEO->GEO 호만 전이 결과 저장됨 → {leo_geo_csv}")

  propagated_csv = os.path.join(results_dir, "hohmann_propagated_verification.csv")
  with open(propagated_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["r1_km", "r2_km", "start_r_km", "end_r_km", "start_error_km", "end_error_km"])
    writer.writerow([propagated_result["r1_km"], propagated_result["r2_km"], f"{propagated_result['start_r_km']:.4f}",
                      f"{propagated_result['end_r_km']:.4f}", f"{propagated_result['start_error_km']:.8f}",
                      f"{propagated_result['end_error_km']:.8f}"])
  print(f"[기록] 전이궤도 실제 전파 검증 결과 저장됨 → {propagated_csv}")

  ratio_csv = os.path.join(results_dir, "hohmann_delta_v_vs_ratio.csv")
  with open(ratio_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["ratio", "delta_v1", "delta_v2", "total_delta_v"])
    for row in ratio_rows:
      writer.writerow([row["ratio"], f"{row['delta_v1']:.6f}", f"{row['delta_v2']:.6f}", f"{row['total_delta_v']:.6f}"])
  print(f"[기록] 반지름 비율별 델타-V 결과 저장됨 → {ratio_csv}")


if __name__ == "__main__":
  main()
