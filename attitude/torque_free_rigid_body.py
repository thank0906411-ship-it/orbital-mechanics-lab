"""
자세동역학(Attitude Dynamics) - 토크 없는 강체 자유회전과 중간축 정리(테니스 라켓 정리)

이제까지의 모든 스크립트는 위성을 점 질량으로만 다뤘다 — 궤도 위의 위치와 속도만
계산했지, 위성 자체가 어느 방향을 향하고 있는지(자세)는 전혀 다루지 않았다. 이
스크립트는 그 방향으로 새로 열리는 독립적인 물리 영역이다: 외부 토크가 전혀 없는
강체가 자유롭게 회전할 때, 관성모멘트가 가장 크거나 가장 작은 축 주위 회전은
안정적이지만 중간 크기 축 주위 회전은 아무리 작은 섭동에도 발산한다는 "중간축
정리"(intermediate axis theorem, 테니스 라켓을 던지면 중간축으로만 돌리려 해도
저절로 뒤집히는 현상으로 잘 알려져 있다)를 오일러 회전방정식의 직접 수치적분으로
재현한다. 이 프로젝트는 이전에 "자세제어(PID 피드백)"는 범위 밖으로 정해뒀지만,
이 스크립트는 제어가 아니라 순수한 동역학(외부 입력 없는 자유회전)만 다루므로 그
범위 제한과 충돌하지 않는다.

핵심 개념 1: 오일러 회전방정식 - 각속도 벡터 하나만으로 강체 회전을 기술한다
  관성주축(principal axes)이 몸체좌표계와 일치하는 강체(관성텐서가 대각행렬
  diag(I1,I2,I3))가 외부 토크 없이 회전할 때, 몸체좌표계 기준 각속도
  ω=(ω1,ω2,ω3)는
      I1*dω1/dt = (I2-I3)*ω2*ω3
      I2*dω2/dt = (I3-I1)*ω3*ω1
      I3*dω3/dt = (I1-I2)*ω1*ω2
  를 만족한다. 이 프로젝트에는 상태벡터에 맞춘 RK4를 매번 복제해 쓰는 확립된
  관례가 있다(03번의 rk4_step, 15번의 rk4_step_with_thrust) — 이 스크립트도
  같은 4단계 RK4 구조를 각속도 3성분 상태에 맞게 새로 정의한다.

핵심 개념 2: 회전 에너지와 각운동량 크기는 항상 보존된다
  토크가 없으므로 회전 운동에너지 T=(I1*ω1^2+I2*ω2^2+I3*ω3^2)/2와 각운동량
  크기 |L|^2=(I1*ω1)^2+(I2*ω2)^2+(I3*ω3)^2는 시간에 무관하게 일정해야 한다.
  이는 02/03번의 비에너지/비각운동량 보존 검증과 같은 성격이다 — 적분 자체가
  올바른지 감시하는 용도로도 쓴다.

핵심 개념 3: 최대축/최소축 회전은 안정, 중간축 회전은 불안정하다
  각속도가 거의 완전히 한 주축에 정렬된 상태에서 시작해 다른 두 축에 작은
  섭동을 주면, 최대 관성모멘트(I3) 또는 최소 관성모멘트(I1) 축 정렬 회전은
  섭동 성분이 작은 진폭으로 진동만 할 뿐 커지지 않는다("안정"). 반면 중간
  관성모멘트(I2) 축 정렬 회전은 똑같은 크기의 섭동이 기하급수적으로 커지다가
  축 자체가 거의 뒤집히는 텀블링으로 이어진다("불안정") — 우주정거장/위성이
  중간축 회전을 피하는 실제 이유다.

핵심 개념 4: 섭동 성장률로 안정성을 정량 비교한다
  "섭동된 각속도 성분의 최대 크기 / 초기 섭동 크기" 비율을 세 축 각각에 대해
  계산하면, 중간축의 성장 배율이 다른 두 축보다 훨씬 크다는 것을 숫자로 직접
  비교할 수 있다.

단순화: 이 스크립트는 관성주축이 이미 몸체좌표계와 일치한다고 가정하고(관성텐서가
대각행렬), 자세 자체(쿼터니언이나 오일러각)의 시간 변화는 적분하지 않는다 —
순수하게 각속도 벡터의 동역학(오일러 방정식)만 다룬다. 쿼터니언 기반 자세 전파나
피드백 제어(PID 등)는 이 프로젝트에서 계속 범위 밖으로 남겨둔다.

이전 스크립트들과의 관계: 이 스크립트는 궤도 위치/속도(병진운동)를 다루는 01~15번과
물리적으로 독립적이다 — 궤도역학 인프라(01/02/04번 등)를 억지로 재사용하지 않고,
새 물리 영역(회전운동)을 그 자체로 다룬다. 03번 RK4 적분기, 15번 rk4_step_with_thrust
와 구조적으로 같은 "상태벡터에 맞춘 RK4 복제" 패턴만 이어받는다.
"""

import argparse
import csv
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")
  sys.stderr.reconfigure(encoding="utf-8")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)


def euler_equations_derivative(omega, inertia_diag):
  """오일러 회전방정식의 우변: dω/dt = (I2-I3)/I1*ω2*ω3, ... (각 성분별)."""
  omega1, omega2, omega3 = omega
  i1, i2, i3 = inertia_diag
  domega1 = (i2 - i3) / i1 * omega2 * omega3
  domega2 = (i3 - i1) / i2 * omega3 * omega1
  domega3 = (i1 - i2) / i3 * omega1 * omega2
  return np.array([domega1, domega2, domega3])


def rk4_step(omega, dt_sec, inertia_diag):
  """03/15번과 동일한 RK4 4단계 구조를 각속도 3성분 상태에 맞게 정의한다."""
  k1 = euler_equations_derivative(omega, inertia_diag)
  k2 = euler_equations_derivative(omega + dt_sec / 2 * k1, inertia_diag)
  k3 = euler_equations_derivative(omega + dt_sec / 2 * k2, inertia_diag)
  k4 = euler_equations_derivative(omega + dt_sec * k3, inertia_diag)
  return omega + dt_sec / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_attitude(initial_omega, inertia_diag, total_time_sec, dt_sec):
  """초기 각속도에서 total_time_sec 동안 RK4로 적분한다. 반환: [(t, omega), ...]
  (03번 integrate_two_body와 동일한 시계열 반환 패턴)."""
  omega = np.array(initial_omega, dtype=float)
  t = 0.0
  history = [(t, omega.copy())]
  while t < total_time_sec - 1e-9:
    step = min(dt_sec, total_time_sec - t)
    omega = rk4_step(omega, step, inertia_diag)
    t += step
    history.append((t, omega.copy()))
  return history


def rotational_energy(omega, inertia_diag):
  """T = (I1*ω1^2 + I2*ω2^2 + I3*ω3^2) / 2."""
  return np.sum(np.array(inertia_diag) * np.array(omega) ** 2) / 2


def angular_momentum_magnitude(omega, inertia_diag):
  """|L| = sqrt((I1*ω1)^2 + (I2*ω2)^2 + (I3*ω3)^2)."""
  l_vec = np.array(inertia_diag) * np.array(omega)
  return np.linalg.norm(l_vec)


def perturbation_growth_ratio(history, perturbed_axes):
  """섭동을 준 두 축 성분의 (시계열 최대 크기) / (초기 크기) 비율 중 더 큰 쪽을
  반환한다 — 그 축 정렬 회전이 섭동에 얼마나 민감한지를 하나의 숫자로 요약."""
  initial_omega = history[0][1]
  max_ratio = 0.0
  for axis in perturbed_axes:
    initial_magnitude = abs(initial_omega[axis])
    max_magnitude = max(abs(omega[axis]) for _t, omega in history)
    ratio = max_magnitude / initial_magnitude
    max_ratio = max(max_ratio, ratio)
  return max_ratio


def demo_conservation_of_energy_and_momentum(inertia_diag=(1.0, 2.0, 3.0)):
  """토크가 없으므로 회전 에너지와 각운동량 크기가 적분 내내 거의 일정하게
  유지되는지 확인한다 — 적분 자체가 올바른지 감시하는 용도이기도 하다."""
  print("=" * 70)
  print("[1] 토크 없는 자유회전: 회전 에너지와 각운동량 보존 검증")
  print("=" * 70)
  initial_omega = np.array([0.5, 0.8, 0.3])
  dt_sec = 0.01
  total_time_sec = 20.0

  print(f"관성모멘트(I1,I2,I3)={inertia_diag}, 초기 각속도={initial_omega} rad/s\n")
  history = integrate_attitude(initial_omega, inertia_diag, total_time_sec, dt_sec)

  energies = [rotational_energy(omega, inertia_diag) for _t, omega in history]
  momenta = [angular_momentum_magnitude(omega, inertia_diag) for _t, omega in history]

  energy_drift = max(energies) - min(energies)
  momentum_drift = max(momenta) - min(momenta)
  print(f"회전 에너지: 시작={energies[0]:.8f}, 끝={energies[-1]:.8f}, 최대편차={energy_drift:.2e}")
  print(f"각운동량 크기: 시작={momenta[0]:.8f}, 끝={momenta[-1]:.8f}, 최대편차={momentum_drift:.2e}")

  assert energy_drift < 1e-6, "토크가 없으므로 회전 에너지는 수치오차 수준 안에서 보존되어야 함"
  assert momentum_drift < 1e-6, "토크가 없으므로 각운동량 크기는 수치오차 수준 안에서 보존되어야 함"
  print("\n(외부 토크가 없는 자유회전에서 회전 에너지와 각운동량 크기가 RK4 적분 내내")
  print(" 거의 완벽하게 보존된다 — 오일러 방정식 구현과 적분기가 올바르다는 신호다.)")
  return {"energy_drift": energy_drift, "momentum_drift": momentum_drift, "history": history}


def demo_major_and_minor_axis_spins_are_stable(inertia_diag=(1.0, 2.0, 3.0), perturbation_fraction=0.01):
  """최대축(I3)과 최소축(I1) 정렬 회전에 작은 섭동을 주면, 섭동 성분이 초기
  크기의 작은 배수 이내로만 진동해야 한다(안정)."""
  print("\n" + "=" * 70)
  print("[2] 최대축/최소축 정렬 회전: 작은 섭동에도 안정적")
  print("=" * 70)
  spin_rate = 1.0
  perturbation = spin_rate * perturbation_fraction
  dt_sec = 0.01
  total_time_sec = 50.0

  # 최소축(I1, 인덱스 0) 정렬 회전, 나머지 두 축에 섭동
  initial_omega_minor = np.array([spin_rate, perturbation, perturbation])
  history_minor = integrate_attitude(initial_omega_minor, inertia_diag, total_time_sec, dt_sec)
  ratio_minor = perturbation_growth_ratio(history_minor, perturbed_axes=[1, 2])

  # 최대축(I3, 인덱스 2) 정렬 회전, 나머지 두 축에 섭동
  initial_omega_major = np.array([perturbation, perturbation, spin_rate])
  history_major = integrate_attitude(initial_omega_major, inertia_diag, total_time_sec, dt_sec)
  ratio_major = perturbation_growth_ratio(history_major, perturbed_axes=[0, 1])

  print(f"섭동 크기={perturbation:.4f} rad/s (주회전율의 {perturbation_fraction * 100:.1f}%)\n")
  print(f"최소축(I1) 정렬 회전: 섭동 성장 배율={ratio_minor:.2f}배")
  print(f"최대축(I3) 정렬 회전: 섭동 성장 배율={ratio_major:.2f}배")

  assert ratio_minor < 10.0, "최소축 정렬 회전의 섭동은 10배 미만으로만 커져야 함(안정)"
  assert ratio_major < 10.0, "최대축 정렬 회전의 섭동은 10배 미만으로만 커져야 함(안정)"
  print("\n(관성모멘트가 가장 작거나 가장 큰 축 주위 회전은 작은 섭동을 줘도 그 섭동이")
  print(" 커지지 않고 진동만 한다 — 안정적인 회전축이다.)")
  return {"ratio_minor": ratio_minor, "ratio_major": ratio_major,
          "history_minor": history_minor, "history_major": history_major}


def demo_intermediate_axis_spin_is_unstable(inertia_diag=(1.0, 2.0, 3.0), perturbation_fraction=0.01):
  """이 스크립트의 핵심 주장(중간축 정리): 중간축(I2) 정렬 회전에 같은 크기
  섭동을 주면 섭동 성분이 크게 발산해야 한다(불안정)."""
  print("\n" + "=" * 70)
  print("[3] 중간축 정렬 회전: 작은 섭동에도 크게 발산 (중간축 정리)")
  print("=" * 70)
  spin_rate = 1.0
  perturbation = spin_rate * perturbation_fraction
  dt_sec = 0.01
  total_time_sec = 50.0

  # 중간축(I2, 인덱스 1) 정렬 회전, 나머지 두 축에 섭동
  initial_omega = np.array([perturbation, spin_rate, perturbation])
  history = integrate_attitude(initial_omega, inertia_diag, total_time_sec, dt_sec)
  ratio = perturbation_growth_ratio(history, perturbed_axes=[0, 2])

  print(f"섭동 크기={perturbation:.4f} rad/s (주회전율의 {perturbation_fraction * 100:.1f}%)")
  print(f"중간축(I2) 정렬 회전: 섭동 성장 배율={ratio:.2f}배")

  assert ratio > 20.0, "중간축 정렬 회전의 섭동은 20배 이상으로 크게 커져야 함(불안정, 중간축 정리)"
  print("\n(관성모멘트가 중간 크기인 축 주위 회전은 아무리 작은 섭동이라도 기하급수적으로")
  print(" 커진다 — 테니스 라켓을 중간축으로 던지면 저절로 뒤집히는 것과 같은 현상을")
  print(" 오일러 방정식의 직접 수치적분으로 재현했다.)")
  return {"perturbation_growth_ratio": ratio, "history": history}


def demo_perturbation_growth_comparison(inertia_diag=(1.0, 2.0, 3.0), perturbation_fraction=0.01):
  """세 축(최소/중간/최대) 정렬 회전의 섭동 성장 배율을 나란히 비교해, 중간축이
  다른 두 축보다 압도적으로 큰 배율을 보이는지 확인한다."""
  print("\n" + "=" * 70)
  print("[4] 세 축의 섭동 성장 배율 비교")
  print("=" * 70)
  spin_rate = 1.0
  perturbation = spin_rate * perturbation_fraction
  dt_sec = 0.01
  total_time_sec = 50.0

  axis_configs = [
      ("최소축(I1)", np.array([spin_rate, perturbation, perturbation]), [1, 2]),
      ("중간축(I2)", np.array([perturbation, spin_rate, perturbation]), [0, 2]),
      ("최대축(I3)", np.array([perturbation, perturbation, spin_rate]), [0, 1]),
  ]

  rows = []
  print(f"  {'정렬 축':>12}{'섭동 성장 배율':>18}")
  for label, initial_omega, perturbed_axes in axis_configs:
    history = integrate_attitude(initial_omega, inertia_diag, total_time_sec, dt_sec)
    ratio = perturbation_growth_ratio(history, perturbed_axes)
    print(f"  {label:>12}{ratio:>18.2f}")
    rows.append({"axis_label": label, "growth_ratio": ratio})

  intermediate_ratio = rows[1]["growth_ratio"]
  minor_ratio = rows[0]["growth_ratio"]
  major_ratio = rows[2]["growth_ratio"]
  assert intermediate_ratio > minor_ratio * 5, "중간축 성장 배율은 최소축보다 최소 5배 이상 커야 함"
  assert intermediate_ratio > major_ratio * 5, "중간축 성장 배율은 최대축보다 최소 5배 이상 커야 함"
  print("\n(중간축의 섭동 성장 배율이 최소축/최대축보다 압도적으로 크다 — 세 축 중")
  print(" 정확히 중간축 하나만 불안정하다는 것을 숫자로 직접 비교했다.)")
  return rows


def parse_args():
  parser = argparse.ArgumentParser(description="토크 없는 강체 자유회전의 오일러 방정식 수치적분 - 중간축 정리 재현")
  parser.add_argument("--inertia-i1", type=float, default=1.0, help="최소 주관성모멘트 I1, 기본값: 1.0")
  parser.add_argument("--inertia-i2", type=float, default=2.0, help="중간 주관성모멘트 I2, 기본값: 2.0")
  parser.add_argument("--inertia-i3", type=float, default=3.0, help="최대 주관성모멘트 I3, 기본값: 3.0")
  parser.add_argument("--perturbation-fraction", type=float, default=0.01,
                       help="주회전율 대비 섭동 크기 비율, 기본값: 0.01(1%%)")
  return parser.parse_args()


def main():
  args = parse_args()
  inertia_diag = (args.inertia_i1, args.inertia_i2, args.inertia_i3)

  conservation_result = demo_conservation_of_energy_and_momentum(inertia_diag)
  stable_result = demo_major_and_minor_axis_spins_are_stable(inertia_diag, args.perturbation_fraction)
  unstable_result = demo_intermediate_axis_spin_is_unstable(inertia_diag, args.perturbation_fraction)
  comparison_rows = demo_perturbation_growth_comparison(inertia_diag, args.perturbation_fraction)

  print("\n" + "=" * 70)
  print(f"[사용자 지정] I=({args.inertia_i1}, {args.inertia_i2}, {args.inertia_i3}), "
        f"섭동 비율={args.perturbation_fraction * 100:.1f}%")
  print("=" * 70)
  print(f"최소/최대축 안정성: {stable_result['ratio_minor']:.2f}배 / {stable_result['ratio_major']:.2f}배")
  print(f"중간축 불안정성: {unstable_result['perturbation_growth_ratio']:.2f}배")

  results_dir = os.path.join(_ROOT_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  stable_csv = os.path.join(results_dir, "attitude_stable_axis_omega_history.csv")
  with open(stable_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_sec", "omega1", "omega2", "omega3", "axis_label"])
    for t, omega in stable_result["history_minor"]:
      writer.writerow([f"{t:.4f}", f"{omega[0]:.8f}", f"{omega[1]:.8f}", f"{omega[2]:.8f}", "minor"])
    for t, omega in stable_result["history_major"]:
      writer.writerow([f"{t:.4f}", f"{omega[0]:.8f}", f"{omega[1]:.8f}", f"{omega[2]:.8f}", "major"])
  print(f"\n[기록] 안정축 각속도 시계열 저장됨 → {stable_csv}")

  unstable_csv = os.path.join(results_dir, "attitude_unstable_axis_omega_history.csv")
  with open(unstable_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_sec", "omega1", "omega2", "omega3"])
    for t, omega in unstable_result["history"]:
      writer.writerow([f"{t:.4f}", f"{omega[0]:.8f}", f"{omega[1]:.8f}", f"{omega[2]:.8f}"])
  print(f"[기록] 중간축(불안정) 각속도 시계열 저장됨 → {unstable_csv}")

  comparison_csv = os.path.join(results_dir, "attitude_perturbation_growth_comparison.csv")
  with open(comparison_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["axis_label", "growth_ratio"])
    for row in comparison_rows:
      writer.writerow([row["axis_label"], f"{row['growth_ratio']:.4f}"])
  print(f"[기록] 축별 섭동 성장 배율 비교 저장됨 → {comparison_csv}")

  summary_csv = os.path.join(results_dir, "attitude_summary.csv")
  with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["inertia_i1", "inertia_i2", "inertia_i3", "energy_drift", "momentum_drift",
                      "minor_axis_ratio", "major_axis_ratio", "intermediate_axis_ratio"])
    writer.writerow([args.inertia_i1, args.inertia_i2, args.inertia_i3,
                      f"{conservation_result['energy_drift']:.2e}", f"{conservation_result['momentum_drift']:.2e}",
                      f"{stable_result['ratio_minor']:.4f}", f"{stable_result['ratio_major']:.4f}",
                      f"{unstable_result['perturbation_growth_ratio']:.4f}"])
  print(f"[기록] 요약 결과 저장됨 → {summary_csv}")


if __name__ == "__main__":
  main()
