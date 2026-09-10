"""
Clohessy-Wiltshire(Hill) 방정식 - 도킹/랑데부를 위한 근접 상대운동

09번 란베르트 문제는 "지금 여기서 저기로 정해진 시간에 도달하는" 큰 스케일의
전이를 다뤘다. 실제 도킹/랑데부는 그 전이가 끝난 뒤, 목표 위성 바로 근처(수백m~
수km)에서 벌어지는 정밀 접근 문제다 — 이 스케일에서는 절대좌표(ECI)보다 목표
위성을 원점으로 하는 상대좌표계에서 생각하는 게 훨씬 직관적이다. Clohessy-Wiltshire
(CW, 또는 Hill) 방정식은 목표 위성이 원궤도를 돈다고 가정하고 상대운동방정식을
선형화해, 09번의 반복(iterative) solver와는 완전히 다른 방식 — 닫힌 형태(closed-form)
해석해 — 로 상대 궤적을 구한다.

핵심 개념 1: 상대좌표계(LVLH/Hill 프레임)에서 상대운동을 선형화한다
  목표 위성을 원점으로 하고, x=반경 방향(지구 반대쪽), y=진행 방향(along-track),
  z=궤도면 법선 방향(cross-track)인 좌표계를 쓴다. 목표가 반지름 a, 평균운동
  n=sqrt(mu/a^3)인 원궤도를 돈다고 가정하면, 추격 위성의 상대운동방정식은
      x'' - 2*n*y' - 3*n^2*x = 0
      y'' + 2*n*x' = 0
      z'' + n^2*z = 0
  로 선형화된다(비선형 중력장을 목표 위치 근처에서 1차 테일러 전개). z축(궤도면
  법선)은 x, y와 완전히 분리(decoupled)되어 있다는 점이 눈에 띈다 — 궤도면을
  벗어나는 방향의 운동은 반경/진행 방향과 아무 상호작용 없이 그 자체로 독립적인
  단순조화진동이다.

핵심 개념 2: 이 선형계는 상태천이행렬로 닫힌 해를 갖는다
  초기 상대위치/속도 (x0,y0,z0,vx0,vy0,vz0)가 주어지면, 시간 t 후의 상태는
      x(t) = (4-3cos(nt))*x0 + (sin(nt)/n)*vx0 + (2/n)*(1-cos(nt))*vy0
      y(t) = 6*(sin(nt)-nt)*x0 + y0 - (2/n)*(1-cos(nt))*vx0 + (1/n)*(4*sin(nt)-3*nt)*vy0
      z(t) = z0*cos(nt) + (vz0/n)*sin(nt)
  로 정확히 닫힌 형태로 주어진다(속도도 이 식을 미분해 닫힌 형태로 구함). 09번의
  반복 solver와 근본적으로 다른 종류의 풀이법 — 선형 시스템이라 상태천이행렬을
  한 번 계산하면 어떤 초기조건에도 즉시 적용할 수 있다.

핵심 개념 3: "제로 드리프트" 조건을 벗어나면 진행 방향으로 무한히 멀어진다
  y(t) 식의 6*(sin(nt)-nt)*x0 항을 보면, sin(nt)는 유계(-1~1)이지만 -nt는 시간에
  비례해 무한히 커진다 — 즉 x0가 0이 아니면서 이 항을 상쇄할 초기 속도가 없으면,
  상대위치는 진동하며 멀어지는 게 아니라 선형으로(secular) 계속 멀어진다. 이
  발산을 막는 조건이 vy0 = -2*n*x0(vx0=vz0=0일 때) — 이 조건을 만족하는 궤적은
  반경 방향으로 벗어난 채 목표와 일정한 거리를 유지하며 자유표류(free drift)한다.
  이 스크립트는 이 조건을 만족하는 경우와 안 만족하는 경우를 직접 비교해 발산이
  실제로 일어나는 것을 보여준다.

핵심 개념 4: CW 근사는 상대거리가 작을 때만 유효하다 (감추지 않음)
  CW 방정식은 목표 위치 근처에서 중력장을 1차 테일러 전개한 선형화 근사다 —
  상대거리가 목표 궤도 반지름에 비해 무시할 수 있을 만큼 작아야 정확하다. 이
  스크립트는 CW 닫힌 해와, 01번의 실제(비선형) 궤도 전파로 두 위성을 독립적으로
  전파해 계산한 "진짜" 상대위치를 비교해, 상대거리가 커질수록 CW 근사가 실제
  결과에서 벗어난다는 것을 직접 보여준다 — 이게 CW 방정식이 근접 운용(수km 이내)
  에만 쓰이는 이유다.

01번(케플러 전파)과의 관계: 목표 위성의 평균운동 n을 01번의 mean_motion으로
계산하고, 근사 검증 데모에서는 01번의 propagate_orbit으로 목표/추격 위성을 각각
독립적으로 전파해 CW 선형해와 대조한다.
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
_KEPLER_PATH = os.path.join(_THIS_DIR, "01_Kepler_orbit_propagation.py")
_spec = importlib.util.spec_from_file_location("kepler_module", _KEPLER_PATH)
kepler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kepler)

_ELEMENTS_PATH = os.path.join(_THIS_DIR, "02_Orbital_elements_and_energy.py")
_elements_spec = importlib.util.spec_from_file_location("elements_module", _ELEMENTS_PATH)
elements_module = importlib.util.module_from_spec(_elements_spec)
_elements_spec.loader.exec_module(elements_module)

EARTH_MU_KM3_S2 = kepler.EARTH_MU_KM3_S2


def cw_state_transition(x0, y0, z0, vx0, vy0, vz0, n, t):
  """CW 방정식의 닫힌 해로 시간 t 후 상대 상태(위치+속도)를 계산한다.
  반환: dict(x, y, z, vx, vy, vz)."""
  cos_nt = np.cos(n * t)
  sin_nt = np.sin(n * t)

  x = (4 - 3 * cos_nt) * x0 + (sin_nt / n) * vx0 + (2 / n) * (1 - cos_nt) * vy0
  y = 6 * (sin_nt - n * t) * x0 + y0 - (2 / n) * (1 - cos_nt) * vx0 + (1 / n) * (4 * sin_nt - 3 * n * t) * vy0
  z = z0 * cos_nt + (vz0 / n) * sin_nt

  vx = 3 * n * sin_nt * x0 + cos_nt * vx0 + 2 * sin_nt * vy0
  vy = 6 * n * (cos_nt - 1) * x0 - 2 * sin_nt * vx0 + (4 * cos_nt - 3) * vy0
  vz = -z0 * n * sin_nt + cos_nt * vz0

  return {"x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz}


def cw_zero_drift_velocity(x0, n):
  """자유표류가 선형(secular)으로 발산하지 않는 초기 진행방향 속도(vx0=vz0=0 가정).
  vy0 = -2*n*x0."""
  return -2 * n * x0


def cw_rendezvous_delta_v(x0, y0, z0, n, transfer_time_sec):
  """상태천이행렬을 역산해, transfer_time_sec 후 정확히 원점(상대위치 0,0,0)에
  도달하는 데 필요한 초기 상대속도를 구한다. 반환: dict(vx0, vy0, vz0,
  natural_vy0, delta_v_km_s) — natural_vy0는 제로 드리프트 속도(비교 기준),
  delta_v_km_s는 필요 속도와 제로 드리프트 속도의 차이(첫 임펄스 크기, x-y
  평면만 고려)."""
  nt = n * transfer_time_sec
  cos_nt = np.cos(nt)
  sin_nt = np.sin(nt)

  # x(t)=0, y(t)=0 두 식을 (vx0, vy0)에 대한 2x2 선형계로 풀어 해석적으로 역산한다.
  # x(t) = (4-3cos)*x0 + (sin/n)*vx0 + (2/n)*(1-cos)*vy0 = 0
  # y(t) = 6*(sin-nt)*x0 + y0 - (2/n)*(1-cos)*vx0 + (1/n)*(4*sin-3*nt)*vy0 = 0
  a11, a12 = sin_nt / n, (2 / n) * (1 - cos_nt)
  a21, a22 = -(2 / n) * (1 - cos_nt), (1 / n) * (4 * sin_nt - 3 * nt)
  b1 = -(4 - 3 * cos_nt) * x0
  b2 = -(6 * (sin_nt - nt) * x0 + y0)

  det = a11 * a22 - a12 * a21
  vx0 = (b1 * a22 - a12 * b2) / det
  vy0 = (a11 * b2 - b1 * a21) / det

  natural_vy0 = cw_zero_drift_velocity(x0, n)
  delta_v_km_s = np.sqrt(vx0 ** 2 + (vy0 - natural_vy0) ** 2)

  return {"vx0": vx0, "vy0": vy0, "vz0": 0.0, "natural_vy0": natural_vy0, "delta_v_km_s": delta_v_km_s}


def demo_z_axis_is_simple_harmonic_motion(target_semi_major_axis_km=6978.0):
  """cross-track(z) 운동이 x,y와 완전히 분리된 단순조화진동이며, 그 주기가
  목표 위성의 궤도 주기와 정확히 같다는 것을 확인한다."""
  print("=" * 70)
  print("[1] Cross-track(z) 운동: x,y와 분리된 단순조화진동")
  print("=" * 70)
  a = target_semi_major_axis_km
  n = kepler.mean_motion(a)
  orbital_period = 2 * np.pi / n
  z0, vz0 = 2.0, 0.0  # 2km 옵셋으로 시작, 초기 속도 0

  print(f"목표 궤도 반장축={a}km, 평균운동 n={n:.6e}rad/s, 궤도 주기={orbital_period / 60:.2f}분\n")
  rows = []
  sample_fractions = np.linspace(0, 1.0, 9)
  print(f"  {'주기 대비 시간':>14}{'z(km)':>12}")
  for frac in sample_fractions:
    t = frac * orbital_period
    state = cw_state_transition(0.0, 0.0, z0, 0.0, 0.0, vz0, n, t)
    print(f"  {frac:>14.3f}{state['z']:>12.4f}")
    rows.append({"time_fraction": frac, "t_sec": t, "z_km": state["z"]})

  z_at_full_period = rows[-1]["z_km"]
  print(f"\n한 궤도 주기 후 z: {z_at_full_period:.6f}km (초기값 {z0}km과 일치해야 함 — 한 주기 후 원점으로 복귀)")
  assert abs(z_at_full_period - z0) < 1e-6, "z축 단순조화진동은 궤도 주기마다 초기값으로 정확히 복귀해야 함"
  print("\n(z축 운동은 x,y와 아무런 상호작용 없이, 목표 위성의 궤도 주기와 정확히 같은")
  print(" 주기로 단순조화진동한다 — 궤도면을 벗어난 옵셋은 궤도역학적으로 궤도면 안의")
  print(" 운동과 완전히 독립적이라는 것을 보여준다.)")
  return rows


def demo_zero_drift_condition_prevents_divergence(target_semi_major_axis_km=6978.0):
  """제로 드리프트 조건(vy0=-2*n*x0)을 만족하는 궤적과 안 만족하는 궤적(vy0=0)을
  비교해, 조건을 벗어나면 진행 방향으로 선형 발산한다는 것을 직접 보인다."""
  print("\n" + "=" * 70)
  print("[2] 제로 드리프트 조건: 벗어나면 진행 방향으로 선형 발산한다")
  print("=" * 70)
  a = target_semi_major_axis_km
  n = kepler.mean_motion(a)
  orbital_period = 2 * np.pi / n
  x0 = 1.0  # 반경 방향 1km 옵셋

  zero_drift_vy0 = cw_zero_drift_velocity(x0, n)
  print(f"반경 방향 초기 옵셋 x0={x0}km, 제로 드리프트 속도 vy0={zero_drift_vy0:.6e}km/s\n")

  rows = []
  print(f"  {'궤도 수':>10}{'제로드리프트 y(km)':>20}{'vy0=0 y(km)':>16}")
  for num_orbits in [1, 3, 5, 10]:
    t = num_orbits * orbital_period
    zero_drift_state = cw_state_transition(x0, 0.0, 0.0, 0.0, zero_drift_vy0, 0.0, n, t)
    no_correction_state = cw_state_transition(x0, 0.0, 0.0, 0.0, 0.0, 0.0, n, t)
    print(f"  {num_orbits:>10}{zero_drift_state['y']:>20.4f}{no_correction_state['y']:>16.4f}")
    rows.append({"num_orbits": num_orbits, "t_sec": t, "y_zero_drift_km": zero_drift_state["y"],
                 "y_no_correction_km": no_correction_state["y"]})

  assert abs(rows[-1]["y_zero_drift_km"]) < 0.1, "제로 드리프트 조건을 만족하면 y는 발산하지 않고 유계여야 함"
  assert abs(rows[-1]["y_no_correction_km"]) > abs(rows[0]["y_no_correction_km"]) * 3, (
      "제로 드리프트 조건 없이는 궤도가 반복될수록 y가 계속 멀어져야 함"
  )
  print("\n(제로 드리프트 속도를 준 경우 y는 궤도를 아무리 돌아도 거의 그대로 유지되지만,")
  print(" 보정 없이 두면 궤도를 돌 때마다 진행 방향으로 계속 멀어진다 — y(t)식의")
  print(" '-6*n*t*x0' 항이 시간에 비례해 무한히 커지는 선형(secular) 드리프트이기 때문이다.)")
  return rows


def demo_rendezvous_delta_v_reaches_target(target_semi_major_axis_km=6978.0):
  """특정 초기 상대위치에서 랑데부 델타-V로 계산한 초기속도를 CW 방정식에 다시
  대입했을 때, 정확히 t_transfer 후 원점(상대위치 0,0,0)에 도달하는지 검증한다."""
  print("\n" + "=" * 70)
  print("[3] 랑데부 델타-V로 계산한 궤적이 실제로 목표에 도달하는가")
  print("=" * 70)
  a = target_semi_major_axis_km
  n = kepler.mean_motion(a)
  x0, y0, z0 = 0.5, -2.0, 0.1  # 임의의 초기 상대위치(km)
  transfer_time = 1800.0  # 30분 후 도킹 목표

  rendezvous = cw_rendezvous_delta_v(x0, y0, z0, n, transfer_time)
  print(f"초기 상대위치: x0={x0}km, y0={y0}km, z0={z0}km, 목표 도달 시간={transfer_time / 60:.1f}분")
  print(f"필요 초기속도: vx0={rendezvous['vx0']:.6f}km/s, vy0={rendezvous['vy0']:.6f}km/s")
  print(f"제로 드리프트 속도(비교기준): vy0_natural={rendezvous['natural_vy0']:.6f}km/s")
  print(f"첫 임펄스 델타-V: {rendezvous['delta_v_km_s'] * 1000:.4f}m/s")

  final_state = cw_state_transition(x0, y0, z0, rendezvous["vx0"], rendezvous["vy0"], 0.0, n, transfer_time)
  print(f"\n{transfer_time / 60:.1f}분 후 실제 도달 위치: x={final_state['x']:.2e}km, y={final_state['y']:.2e}km")

  position_error_km = np.sqrt(final_state["x"] ** 2 + final_state["y"] ** 2)
  assert position_error_km < 1e-6, "역산한 초기속도로 전파했을 때 정확히 원점(상대위치 0)에 도달해야 함"
  print("\n(상태천이행렬을 역산해 구한 초기속도가 CW 방정식에 다시 대입했을 때 정확히")
  print(" 목표 지점(원점)에 도달한다 — 행렬 역산 자체가 정확하다는 뜻이다. z축은 이미")
  print(" 별도로 독립적이므로 vz0=0으로 두면 z0가 남아있는 채 도킹이 완료되지 않는다는")
  print(" 점도 실무에서는 별도로 처리해야 한다.)")
  return {"x0": x0, "y0": y0, "z0": z0, "transfer_time_sec": transfer_time,
          "delta_v_km_s": rendezvous["delta_v_km_s"], "position_error_km": position_error_km}


def demo_cw_valid_only_for_small_separation(target_semi_major_axis_km=6978.0):
  """CW 선형화 근사와 01/02번을 재사용한 실제(비선형) 궤도 전파 결과를 비교해,
  초기 반경 옵셋이 커질수록 CW 근사가 실제 결과에서 벗어난다는 것을 감추지 않고
  보여준다. CW의 상대 초기조건(x0, vy0)을 회전좌표계-관성좌표계 변환 공식으로
  정확한 절대 상태벡터로 바꿔, 02번의 state_vector_to_orbital_elements로 추격
  위성의 실제 궤도요소를 구하고 01번으로 전파한다 — "같은 초기조건"을 두 방법
  (CW 선형해 vs 완전한 비선형 케플러 궤도)으로 각각 풀어 정확히 비교한다."""
  print("\n" + "=" * 70)
  print("[4] CW 근사의 한계: 초기 반경 옵셋이 커지면 오차가 커진다")
  print("=" * 70)
  a = target_semi_major_axis_km
  mu = EARTH_MU_KM3_S2
  n = kepler.mean_motion(a, mu)
  v_circ_target = np.sqrt(mu / a)
  orbital_period = 2 * np.pi / n
  # 제로 드리프트 조건(vy0=-2*n*x0)을 대입하면 CW의 x(t)=x0*cos(nt)로 단순화된다 —
  # 궤도의 1/4(nt=pi/2)에서는 cos(pi/2)=0이라 x0 크기와 무관하게 항상 정확히 0이
  # 되어 비교 자체가 무의미해진다(실제로 이 값으로 처음 시도했다가 알아챘다).
  # cos(nt)가 0이 아닌 궤도의 1/8 지점으로 비교 시점을 옮겨 이 특이점을 피한다.
  t = orbital_period * 0.125

  # 목표 위성: ECI에서 x축 위 (a,0,0), 속도는 y축 방향(0,v_circ,0) — 이 두 벡터가
  # 그대로 CW 좌표계의 x_hat(반경 방향), y_hat(진행 방향) 기저가 된다.
  target_position = np.array([a, 0.0, 0.0])
  target_velocity = np.array([0.0, v_circ_target, 0.0])
  x_hat = np.array([1.0, 0.0, 0.0])
  y_hat = np.array([0.0, 1.0, 0.0])
  z_hat = np.array([0.0, 0.0, 1.0])  # 궤도면 법선(cross-track), CW의 z축
  omega_vec = n * z_hat  # 목표 궤도의 각속도 벡터(회전좌표계 보정에 필요)

  print(f"목표 궤도 반장축={a}km, 비교 시점={t / 60:.2f}분(궤도의 1/8)\n")
  rows = []
  print(f"  {'초기 반경옵셋(km)':>18}{'CW 예측 x(km)':>16}{'실제(비선형) x(km)':>20}{'상대오차(%)':>14}")
  for x0_km in [0.1, 1.0, 10.0, 100.0, 500.0]:
    vy0 = cw_zero_drift_velocity(x0_km, n)
    cw_state = cw_state_transition(x0_km, 0.0, 0.0, 0.0, vy0, 0.0, n, t)

    # CW 초기조건(x0, vy0, 나머지 0)을 관성좌표계 상태벡터로 정확히 변환한다.
    # 위치: r_chaser = r_target + x0*x_hat (CW의 y0=z0=0이므로 x축 옵셋뿐)
    # 속도: v_chaser = v_target + vy0*y_hat + omega x (x0*x_hat)
    #   (회전좌표계에서 본 상대속도 vy0*y_hat에, 좌표계 자체가 각속도 omega로
    #    회전하는 데서 오는 보정항 omega x r_rel을 더해야 관성좌표계 속도가 된다)
    relative_position = x0_km * x_hat
    chaser_position = target_position + relative_position
    chaser_velocity = target_velocity + vy0 * y_hat + np.cross(omega_vec, relative_position)

    elements = elements_module.state_vector_to_orbital_elements(chaser_position, chaser_velocity, mu)
    a_chaser, e_chaser = elements["semi_major_axis_km"], elements["eccentricity"]
    # 초기 위치가 x축 위(진근점 이각 0, 근지점 또는 원지점 근처는 아님 — 일반적인
    # 위치)이므로, 02번과 동일하게 상태벡터에서 이심 이각을 직접 역산해야 한다.
    r0_chaser = np.linalg.norm(chaser_position)
    cos_ecc_anomaly0 = np.clip((1 - r0_chaser / a_chaser) / e_chaser, -1.0, 1.0) if e_chaser > 1e-9 else 1.0
    # 반경방향 속도의 부호로 근점 통과 전/후(이심 이각 부호)를 판별한다. 이 데모의
    # 초기 위치는 항상 반경방향 속도가 정확히 0(np.sign이 0을 반환)인 근/원지점이므로,
    # 그 경우 +1로 처리한다 — arccos(cos_ecc_anomaly0) 자체가 이미 0(근지점) 또는
    # pi(원지점)를 정확히 담고 있어 부호가 결과에 영향을 주지 않는다.
    radial_velocity_sign = np.sign(np.dot(chaser_position, chaser_velocity)) or 1.0
    ecc_anomaly0 = radial_velocity_sign * np.arccos(cos_ecc_anomaly0) if e_chaser > 1e-9 else 0.0
    mean_anomaly0_chaser = ecc_anomaly0 - e_chaser * np.sin(ecc_anomaly0)

    chaser_state = kepler.propagate_orbit(a_chaser, e_chaser, mean_anomaly0_rad=mean_anomaly0_chaser, time_sec=t, mu=mu)
    target_state = kepler.propagate_orbit(a, 0.0, mean_anomaly0_rad=0.0, time_sec=t, mu=mu)
    actual_radial_offset = chaser_state["r"] - target_state["r"]

    relative_error_pct = (abs(cw_state["x"] - actual_radial_offset) / abs(actual_radial_offset) * 100
                           if actual_radial_offset != 0 else 0.0)
    print(f"  {x0_km:>18.1f}{cw_state['x']:>16.4f}{actual_radial_offset:>20.4f}{relative_error_pct:>14.2f}")
    rows.append({"x0_km": x0_km, "cw_x_km": cw_state["x"], "actual_x_km": actual_radial_offset,
                 "relative_error_pct": relative_error_pct})

  assert rows[0]["relative_error_pct"] < rows[-1]["relative_error_pct"], (
      "초기 옵셋이 작을 때(0.1km)의 상대오차가 클 때(500km)보다 작아야 함(CW는 근접 근사)"
  )
  print("\n(초기 반경 옵셋이 1km 안팎일 때는 CW 근사와 실제 비선형 전파가 거의 일치하지만,")
  print(" 옵셋이 수백km로 커지면 오차가 뚜렷해진다 — CW 방정식이 선형화 근사이기 때문에")
  print(" 목표 궤도 반지름에 비해 상대거리가 무시할 만큼 작을 때만 정확하다. 실제 도킹")
  print(" 운용이 CW를 근접 단계(수km 이내)에서만 쓰는 이유다.)")
  return rows


def parse_args():
  parser = argparse.ArgumentParser(description="Clohessy-Wiltshire(Hill) 방정식으로 도킹/랑데부 상대운동 계산")
  parser.add_argument("--target-altitude-km", type=float, default=600.0, help="목표 위성 고도(km), 기본값: 600km")
  return parser.parse_args()


def main():
  args = parse_args()
  a = 6378.137 + args.target_altitude_km

  # CLI로 지정한 목표 고도를 각 데모의 반장축 인자로 그대로 전달한다 — 09/11/12번에서
  # CLI 인자가 파싱만 되고 실제 계산에 반영되지 않던 문제를 반복하지 않기 위함이다.
  z_axis_rows = demo_z_axis_is_simple_harmonic_motion(a)
  zero_drift_rows = demo_zero_drift_condition_prevents_divergence(a)
  rendezvous_result = demo_rendezvous_delta_v_reaches_target(a)
  approximation_rows = demo_cw_valid_only_for_small_separation(a)

  results_dir = os.path.join(_THIS_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  z_axis_csv = os.path.join(results_dir, "cw_z_axis_shm.csv")
  with open(z_axis_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["time_fraction", "t_sec", "z_km"])
    for row in z_axis_rows:
      writer.writerow([f"{row['time_fraction']:.4f}", f"{row['t_sec']:.2f}", f"{row['z_km']:.6f}"])
  print(f"\n[기록] Cross-track 단순조화진동 결과 저장됨 → {z_axis_csv}")

  zero_drift_csv = os.path.join(results_dir, "cw_zero_drift_comparison.csv")
  with open(zero_drift_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["num_orbits", "t_sec", "y_zero_drift_km", "y_no_correction_km"])
    for row in zero_drift_rows:
      writer.writerow([row["num_orbits"], f"{row['t_sec']:.2f}", f"{row['y_zero_drift_km']:.6f}",
                        f"{row['y_no_correction_km']:.6f}"])
  print(f"[기록] 제로 드리프트 비교 결과 저장됨 → {zero_drift_csv}")

  rendezvous_csv = os.path.join(results_dir, "cw_rendezvous_delta_v.csv")
  with open(rendezvous_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["x0_km", "y0_km", "z0_km", "transfer_time_sec", "delta_v_km_s", "position_error_km"])
    writer.writerow([rendezvous_result["x0"], rendezvous_result["y0"], rendezvous_result["z0"],
                      f"{rendezvous_result['transfer_time_sec']:.1f}", f"{rendezvous_result['delta_v_km_s']:.8f}",
                      f"{rendezvous_result['position_error_km']:.2e}"])
  print(f"[기록] 랑데부 델타-V 검증 결과 저장됨 → {rendezvous_csv}")

  approximation_csv = os.path.join(results_dir, "cw_approximation_validity.csv")
  with open(approximation_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["x0_km", "cw_x_km", "actual_x_km", "relative_error_pct"])
    for row in approximation_rows:
      writer.writerow([row["x0_km"], f"{row['cw_x_km']:.6f}", f"{row['actual_x_km']:.6f}",
                        f"{row['relative_error_pct']:.4f}"])
  print(f"[기록] CW 근사 타당성 검증 결과 저장됨 → {approximation_csv}")


if __name__ == "__main__":
  main()
