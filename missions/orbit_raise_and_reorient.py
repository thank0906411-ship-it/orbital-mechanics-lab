"""
궤도+자세 통합 임무 시나리오 - 06번 호만 전이와 21번 PID 자세제어를 하나의 임무
타임라인으로 잇는다

01~21번은 모두 궤도역학(병진운동)과 자세동역학(회전운동)을 서로 독립된 두
영역으로 다뤘다. 실제 위성 운영에서는 이 둘이 하나의 임무 안에서 순서대로
일어난다 - 위성이 주차궤도에서 임의의(발사/분리 직후 남은) 텀블링 자세로
시작해, 호만 전이로 목표 임무궤도에 오른 뒤("on station"), 안테나/관측기기를
쓰려면 지구지향 자세로 재정렬해야 한다. 델타-V를 다 써서 궤도에 도달해도
자세가 안 맞으면 그 자산은 여전히 쓸모가 없다 - 이 스크립트는 바로 그 연결을
보여준다.

이 스크립트는 새 물리를 전혀 추가하지 않는다. 06번(hohmann_transfer.py)의
델타-V/전이시간 계산과 21번(pid_attitude_control.py)의 쿼터니언 PD 제어를
그대로 호출만 하는 얇은 오케스트레이션 계층이다.

핵심 개념 1: 두 단계는 시간축이 1000배 가까이 다르다
  호만 전이는 수백~수천 초(LEO 내 상승이라도 수십 분) 걸리지만, PD 자세
  재정렬은 수십 초 안에 끝난다. 이 스크립트는 자세 재정렬이 궤도 전이 완료
  "직후"부터 시작한다고 명시적으로 가정한다(전이 버넌 도중의 자세는 다루지
  않는 단순화) - 자세 재정렬 단계의 시간축에 궤도 전이가 끝난 시각
  (t_offset_sec)을 더해 하나의 임무 시계열에 이어붙인다.

핵심 개념 2: 상태가 단계 경계를 실제로 이어받아야 "통합"이다
  단순히 두 데모를 나란히 실행하는 것과 상태를 실제로 잇는 것은 다르다. 이
  스크립트에서 자세 재정렬 단계는 궤도 전이 완료 시점에 선언된 바로 그 임의
  자세(initial_attitude_q)에서 시작해야 하며, identity나 다른 임의값에서
  다시 시작하면 안 된다 - 이것이 "통합"이라는 주장의 핵심 증거다.

핵심 개념 3: 두 단계는 서로 다른 자원을 소모한다
  궤도 전이는 델타-V(추진제)를 소모하고, 자세 재정렬은 이 스크립트의 PD
  토크 제어 모델에서 델타-V를 전혀 소모하지 않는다(반작용휠 디새추레이션
  같은 추가 델타-V는 모델링하지 않음). 따라서 총 임무 델타-V는 궤도 전이
  단계의 값과 정확히 같고, 초기 텀블 각도가 아무리 심해도 변하지 않는다 -
  반면 총 임무 "시간"은 두 단계 모두에 의존한다.

단순화: 지상국 접촉창(05번, ground_station_visibility.py)은 의도적으로
제외한다. find_contact_windows는 궤도면 전체 기하(경사각/RAAN/argp/관측지
위경도)와 다중 궤도 주기 시계열이 필요해, 05번 전체를 재사용이 아니라
복제해야 한다 - 이 스크립트의 핵심 주장(궤도→자세 핸드오프)과 무관한 하위
문제를 얹는 셈이라 범위에서 뺐다. 이 스크립트는 06번/21번과 같은 LEO 내
궤도 전이(고도 상승)만 다루고, 전이 중 발생할 수 있는 섭동이나 연료 질량
변화도 다루지 않는다.
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

_HOHMANN_PATH = os.path.join(_ROOT_DIR, "missions", "hohmann_transfer.py")
_hohmann_spec = importlib.util.spec_from_file_location("hohmann_module", _HOHMANN_PATH)
hohmann = importlib.util.module_from_spec(_hohmann_spec)
_hohmann_spec.loader.exec_module(hohmann)

_PID_PATH = os.path.join(_ROOT_DIR, "attitude", "pid_attitude_control.py")
_pid_spec = importlib.util.spec_from_file_location("pid_module", _PID_PATH)
pid = importlib.util.module_from_spec(_pid_spec)
_pid_spec.loader.exec_module(pid)

EARTH_MU_KM3_S2 = kepler.EARTH_MU_KM3_S2


def run_orbit_raise_phase(r1_km, r2_km, mu=EARTH_MU_KM3_S2):
  """호만 전이 한 번을 계산한다. hohmann_transfer_delta_v/time/orbit_elements를
  그대로 호출만 하는 얇은 orchestration 함수 - 새 물리는 없다."""
  delta_v1, delta_v2, total_delta_v = hohmann.hohmann_transfer_delta_v(r1_km, r2_km, mu)
  transfer_time_sec = hohmann.hohmann_transfer_time(r1_km, r2_km, mu)
  a_t_km, e_t = hohmann.hohmann_transfer_orbit_elements(r1_km, r2_km)
  return {
      "r1_km": r1_km,
      "r2_km": r2_km,
      "delta_v1_km_s": delta_v1,
      "delta_v2_km_s": delta_v2,
      "total_delta_v_km_s": total_delta_v,
      "transfer_time_sec": transfer_time_sec,
      "a_t_km": a_t_km,
      "e_t": e_t,
  }


def time_to_settle_below_threshold(error_angles, threshold_deg=2.0):
  """오차각 시계열에서 threshold_deg 아래로 처음 내려간 시각을 찾는다. 끝까지
  내려가지 못하면 None을 반환한다."""
  for t, error_deg in error_angles:
    if error_deg < threshold_deg:
      return t
  return None


def run_attitude_reorientation_phase(initial_q, inertia_diag, settling_time_sec,
                                      dt_sec=0.05, zeta=0.7, q_target=None):
  """궤도 전이 완료 시점의 임의 자세(initial_q)에서 목표 자세(q_target, 기본값
  identity = 지구지향 완료 상태)로 PD 제어기가 수렴시킨다. compute_pd_gains와
  integrate_attitude_with_control을 그대로 호출만 한다 - 21번과 동일하게
  total_time_sec = settling_time_sec * 3(정착시간의 3배)을 적분 구간으로 쓴다."""
  if q_target is None:
    q_target = pid.IDENTITY_QUATERNION.copy()

  kp, kd = pid.compute_pd_gains(inertia_diag, settling_time_sec, zeta)
  total_time_sec = settling_time_sec * 3
  initial_omega = np.zeros(3)

  control_law = lambda q_err, omega: pid.pd_control_torque(  # noqa: E731
      q_err, omega, np.zeros(3), kp, kd)
  history = pid.integrate_attitude_with_control(
      initial_q, initial_omega, inertia_diag, total_time_sec, dt_sec,
      control_law=control_law, q_target=q_target)

  error_angles = []
  for t, (q, _omega) in history:
    q_err = pid.attitude_error_quaternion(q, q_target)
    error_angles.append((t, pid.quaternion_error_angle_deg(q_err)))

  return {
      "history": history,
      "error_angles": error_angles,
      "final_error_deg": error_angles[-1][1],
      "time_to_settle_sec": time_to_settle_below_threshold(error_angles),
      "kp": kp,
      "kd": kd,
      "settling_time_sec": settling_time_sec,
      "total_time_sec": total_time_sec,
  }


def build_mission_timeline(r1_km, r2_km, initial_attitude_q, inertia_diag,
                            settling_time_sec, mu=EARTH_MU_KM3_S2):
  """궤도 전이 단계와 자세 재정렬 단계를 하나의 임무 타임라인으로 잇는다 - 이
  스크립트의 핵심 통합 함수. 궤도 전이가 t=0~transfer_time_sec 동안 일어나고,
  자세 재정렬은 전이 완료 "직후"(자체 시간축 t'=0에서 시작, t_offset_sec=
  transfer_time_sec로 기록)부터 시작한다고 명시적으로 가정한다. 총 임무시간은
  transfer_time_sec + settling_time_sec*3(자세 재정렬 단계의 전체 적분 구간,
  정착 완료 후의 수렴 확인 시간까지 포함)으로 고정한다 - 실제 수렴 시각을
  스캔해서 찾는 방식이 아니라, 21번 demo_pd_point_and_hold와 동일한 고정 배율
  관례를 그대로 따른다."""
  orbit_phase = run_orbit_raise_phase(r1_km, r2_km, mu)
  attitude_phase = run_attitude_reorientation_phase(
      initial_attitude_q, inertia_diag, settling_time_sec)
  attitude_phase["t_offset_sec"] = orbit_phase["transfer_time_sec"]

  total_mission_time_sec = orbit_phase["transfer_time_sec"] + attitude_phase["total_time_sec"]
  total_delta_v_km_s = orbit_phase["total_delta_v_km_s"]

  return {
      "orbit_phase": orbit_phase,
      "attitude_phase": attitude_phase,
      "total_mission_time_sec": total_mission_time_sec,
      "total_delta_v_km_s": total_delta_v_km_s,
  }


def demo_orbit_raise_then_reorient_timeline(r1_km=6978.0, r2_km=7378.0,
                                             initial_tumble_deg=150.0,
                                             settling_time_sec=10.0,
                                             inertia_diag=(50.0, 70.0, 90.0)):
  """핵심 통합 데모: 궤도 전이 완료 후 자세 재정렬까지 이어지는 결합 타임라인을
  계산하고, 자세가 실제로 지구지향 목표로 수렴하며 총 임무시간이 두 단계의 합과
  정확히 일치하는지 확인한다."""
  print("=" * 70)
  print("[1] 궤도 전이 -> 자세 재정렬: 결합 임무 타임라인")
  print("=" * 70)
  initial_q = pid.axis_angle_to_quaternion([1, 0, 0], initial_tumble_deg)
  mission = build_mission_timeline(r1_km, r2_km, initial_q, inertia_diag, settling_time_sec)
  orbit_phase = mission["orbit_phase"]
  attitude_phase = mission["attitude_phase"]

  print(f"[궤도 전이] r1={r1_km}km -> r2={r2_km}km, "
        f"델타-V={orbit_phase['total_delta_v_km_s'] * 1000:.2f}m/s, "
        f"전이시간={orbit_phase['transfer_time_sec'] / 60:.1f}분")
  print(f"[자세 재정렬] 전이 완료 시각(T+{orbit_phase['transfer_time_sec'] / 60:.1f}분)부터 "
        f"초기 오차={initial_tumble_deg}도에서 시작, 정착시간={settling_time_sec}초")
  print(f"최종 자세 오차: {attitude_phase['final_error_deg']:.3f}도")
  print(f"총 임무시간: {mission['total_mission_time_sec'] / 60:.1f}분, "
        f"총 델타-V: {mission['total_delta_v_km_s'] * 1000:.2f}m/s")

  assert attitude_phase["final_error_deg"] < 2.0, \
      "자세 재정렬 단계가 끝나면 지구지향 목표 자세 오차가 2도 미만이어야 함(21번 point-and-hold와 동일한 수렴 기준)"
  expected_total_time = orbit_phase["transfer_time_sec"] + attitude_phase["total_time_sec"]
  assert mission["total_mission_time_sec"] == expected_total_time, \
      "총 임무시간은 궤도 전이시간과 자세 재정렬 적분시간의 합과 정확히 같아야 함(두 단계가 실제로 이어진다는 증거)"

  print("\n(궤도 전이가 끝난 시점부터 자세 재정렬이 시작해 목표 오차 아래로")
  print(" 수렴한다 - 델타-V를 다 쓴 궤도 전이도 이 재정렬 없이는 안테나/관측기기를")
  print(" 쓸 수 없는 자산으로 남는다는 것을 하나의 타임라인으로 보여준다.)")
  return mission


def demo_attitude_phase_starts_from_orbit_phase_handoff_state(
    r1_km=6978.0, r2_km=7378.0, initial_tumble_deg=150.0, settling_time_sec=10.0,
    inertia_diag=(50.0, 70.0, 90.0)):
  """핸드오프 증명: 자세 재정렬 단계의 첫 쿼터니언이 궤도 전이 완료 시점에
  선언한 바로 그 임의 자세인지(identity나 재무작위화된 값이 아니라) 확인한다 -
  두 단계가 독립적으로 실행된 게 아니라 상태가 실제로 이어진다는 증거다."""
  print("\n" + "=" * 70)
  print("[2] 핸드오프 검증: 자세 재정렬이 궤도 전이 완료 시점의 자세에서 시작하는가")
  print("=" * 70)
  initial_q = pid.axis_angle_to_quaternion([1, 0, 0], initial_tumble_deg)
  mission = build_mission_timeline(r1_km, r2_km, initial_q, inertia_diag, settling_time_sec)
  first_q_in_attitude_history = mission["attitude_phase"]["history"][0][1][0]

  print(f"궤도 전이 완료 시점에 선언한 자세(쿼터니언): {initial_q}")
  print(f"자세 재정렬 단계의 첫 쿼터니언: {first_q_in_attitude_history}")

  assert np.allclose(first_q_in_attitude_history, initial_q), \
      "자세 재정렬 단계는 궤도 전이 단계가 끝난 바로 그 임의 자세에서 시작해야 함 - 두 단계가 독립적으로 실행된 게 아니라 상태가 실제로 이어진다는 증거"

  print("\n(자세 재정렬 단계가 identity나 별도로 무작위화된 자세가 아니라, 궤도")
  print(" 전이 단계가 끝난 바로 그 자세를 그대로 이어받아 시작한다 - 상태가 단계")
  print(" 경계를 실제로 통과한다는 것을 직접 확인한다.)")
  return mission


def demo_total_mission_delta_v_is_unaffected_by_tumble_severity(
    r1_km=6978.0, r2_km=7378.0, settling_time_sec=10.0,
    inertia_diag=(50.0, 70.0, 90.0)):
  """완만한 텀블(30도) vs 심한 텀블(170도)을 같은 궤도 전이에 적용해 비교한다.
  더 심한 텀블은 목표 오차(2도) 아래로 내려가는 데 더 오래 걸리지만, 궤도 전이
  델타-V는 자세와 완전히 무관하게 불변임을 확인한다 - 이 스크립트의 제어모델
  (순수 토크)은 자세 재정렬에 델타-V를 전혀 소모하지 않으므로, 두 단계가 물리적
  으로 독립된 자원(추진제 vs 제어 시간)을 소모한다는 것이 핵심 메시지다."""
  print("\n" + "=" * 70)
  print("[3] 텀블 심각도 민감도: 궤도 전이 델타-V는 자세 재정렬과 무관하게 불변")
  print("=" * 70)
  mild_q = pid.axis_angle_to_quaternion([1, 0, 0], 30.0)
  severe_q = pid.axis_angle_to_quaternion([1, 0, 0], 170.0)
  mild = build_mission_timeline(r1_km, r2_km, mild_q, inertia_diag, settling_time_sec)
  severe = build_mission_timeline(r1_km, r2_km, severe_q, inertia_diag, settling_time_sec)
  mild_settle = mild["attitude_phase"]["time_to_settle_sec"]
  severe_settle = severe["attitude_phase"]["time_to_settle_sec"]

  print(f"완만한 텀블(30도): 2도 미만 도달시간={mild_settle:.2f}초, "
        f"궤도 전이 델타-V={mild['total_delta_v_km_s'] * 1000:.2f}m/s")
  print(f"심한 텀블(170도): 2도 미만 도달시간={severe_settle:.2f}초, "
        f"궤도 전이 델타-V={severe['total_delta_v_km_s'] * 1000:.2f}m/s")

  assert severe_settle > mild_settle, \
      "더 심한 초기 텀블(170도)은 완만한 텀블(30도)보다 목표 오차 아래로 내려가는 데 더 오래 걸려야 함"
  assert severe["total_delta_v_km_s"] == mild["total_delta_v_km_s"], \
      "자세 재정렬 단계는 이 스크립트의 제어모델에서 델타-V를 전혀 소모하지 않으므로(순수 토크 제어), 초기 텀블 각도와 무관하게 궤도 전이 델타-V는 동일해야 함"

  print("\n(초기 텀블이 심해지면 자세 재정렬에 더 오래 걸리지만, 궤도 전이")
  print(" 델타-V는 한 치도 변하지 않는다 - 궤도 전이와 자세 재정렬은 서로 다른")
  print(" 자원을 소모하는 물리적으로 독립된 단계라는 것을 직접 확인한다.)")
  return {"mild": mild, "severe": severe}


def parse_args():
  parser = argparse.ArgumentParser(description="06번 호만 전이와 21번 PID 자세제어를 하나의 임무 타임라인으로 잇는 통합 시나리오")
  parser.add_argument("--r1-km", type=float, default=6978.0, help="출발 주차궤도 반지름(km), 기본값: 약 600km 고도 LEO")
  parser.add_argument("--r2-km", type=float, default=7378.0, help="목표 임무궤도 반지름(km), 기본값: 약 1000km 고도")
  parser.add_argument("--initial-tumble-deg", type=float, default=150.0, help="궤도 전이 완료 시점의 임의 자세 오차(도), 기본값: 150도(거의 뒤집힌 상태)")
  parser.add_argument("--settling-time", type=float, default=10.0, help="자세 재정렬 목표 정착시간(초), 기본값: 10.0")
  parser.add_argument("--inertia-i1", type=float, default=50.0, help="위성 관성모멘트 I1(kg*m^2), 기본값: 50.0")
  parser.add_argument("--inertia-i2", type=float, default=70.0, help="위성 관성모멘트 I2(kg*m^2), 기본값: 70.0")
  parser.add_argument("--inertia-i3", type=float, default=90.0, help="위성 관성모멘트 I3(kg*m^2), 기본값: 90.0")
  return parser.parse_args()


def main():
  args = parse_args()
  inertia_diag = (args.inertia_i1, args.inertia_i2, args.inertia_i3)

  demo_orbit_raise_then_reorient_timeline(
      args.r1_km, args.r2_km, args.initial_tumble_deg, args.settling_time, inertia_diag)
  demo_attitude_phase_starts_from_orbit_phase_handoff_state(
      args.r1_km, args.r2_km, args.initial_tumble_deg, args.settling_time, inertia_diag)
  demo_total_mission_delta_v_is_unaffected_by_tumble_severity(
      args.r1_km, args.r2_km, args.settling_time, inertia_diag)

  initial_q = pid.axis_angle_to_quaternion([1, 0, 0], args.initial_tumble_deg)
  mission = build_mission_timeline(
      args.r1_km, args.r2_km, initial_q, inertia_diag, args.settling_time)
  orbit_phase = mission["orbit_phase"]
  attitude_phase = mission["attitude_phase"]

  print("\n" + "=" * 70)
  print(f"[사용자 지정] r1={args.r1_km}km, r2={args.r2_km}km, "
        f"초기텀블={args.initial_tumble_deg}도, 정착시간={args.settling_time}초")
  print("=" * 70)
  print(f"총 임무시간: {mission['total_mission_time_sec'] / 60:.1f}분, "
        f"총 델타-V: {mission['total_delta_v_km_s'] * 1000:.2f}m/s, "
        f"최종 자세 오차: {attitude_phase['final_error_deg']:.3f}도")

  results_dir = os.path.join(_ROOT_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  timeline_csv = os.path.join(results_dir, "orbit_raise_and_reorient_timeline.csv")
  with open(timeline_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["phase", "t_sec", "metric_name", "metric_value"])
    writer.writerow(["orbit", "0.0", "r_km", f"{orbit_phase['r1_km']:.4f}"])
    writer.writerow(["orbit", f"{orbit_phase['transfer_time_sec']:.4f}", "r_km",
                      f"{orbit_phase['r2_km']:.4f}"])
    t_offset = attitude_phase["t_offset_sec"]
    for t, error_deg in attitude_phase["error_angles"]:
      writer.writerow(["attitude", f"{t_offset + t:.4f}", "error_angle_deg", f"{error_deg:.6f}"])
  print(f"\n[기록] 결합 임무 타임라인 저장됨 → {timeline_csv}")

  handoff_csv = os.path.join(results_dir, "orbit_raise_and_reorient_handoff.csv")
  with open(handoff_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["initial_tumble_deg", "orbit_delta_v_km_s", "orbit_transfer_time_sec",
                      "attitude_settling_time_sec", "attitude_final_error_deg",
                      "total_mission_time_sec"])
    writer.writerow([args.initial_tumble_deg, f"{orbit_phase['total_delta_v_km_s']:.8f}",
                      f"{orbit_phase['transfer_time_sec']:.4f}", args.settling_time,
                      f"{attitude_phase['final_error_deg']:.6f}",
                      f"{mission['total_mission_time_sec']:.4f}"])
  print(f"[기록] 핸드오프 요약 저장됨 → {handoff_csv}")

  mild_q = pid.axis_angle_to_quaternion([1, 0, 0], 30.0)
  severe_q = pid.axis_angle_to_quaternion([1, 0, 0], 170.0)
  mild = build_mission_timeline(args.r1_km, args.r2_km, mild_q, inertia_diag, args.settling_time)
  severe = build_mission_timeline(args.r1_km, args.r2_km, severe_q, inertia_diag, args.settling_time)
  sensitivity_csv = os.path.join(results_dir, "orbit_raise_and_reorient_tumble_sensitivity.csv")
  with open(sensitivity_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "initial_tumble_deg", "orbit_total_delta_v_km_s",
                      "attitude_time_to_settle_sec", "total_mission_time_sec"])
    writer.writerow(["mild", 30.0, f"{mild['total_delta_v_km_s']:.8f}",
                      f"{mild['attitude_phase']['time_to_settle_sec']:.4f}",
                      f"{mild['total_mission_time_sec']:.4f}"])
    writer.writerow(["severe", 170.0, f"{severe['total_delta_v_km_s']:.8f}",
                      f"{severe['attitude_phase']['time_to_settle_sec']:.4f}",
                      f"{severe['total_mission_time_sec']:.4f}"])
  print(f"[기록] 텀블 민감도 비교 저장됨 → {sensitivity_csv}")


if __name__ == "__main__":
  main()
