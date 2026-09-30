"""
평면 리아푸노프 궤도 - 라그랑주점 주위를 도는 주기궤도 가족을 미분수정으로 완성

19번(lagrange_points.py)은 라그랑주점의 "위치"와 "선형 안정성"까지만 다뤘다 -
그 스크립트 docstring이 명시적으로 "헤일로 궤도, 저에너지 전이 같은 CR3BP의
실전 응용은 다루지 않는다"고 범위를 그었다. 이 스크립트는 그 다음 단계로,
라그랑주점 자체가 아니라 그 주위를 도는 주기궤도 가족이 존재한다는 것을
보여준다 - 실제 L1/L2 임무(SOHO, JWST 등)가 정확히 라그랑주점에 위성을 두지
않고 그 주위의 작은 주기궤도에 두는 이유의 수학적 근거다.

조사 결과 3차원 헤일로 궤도는 Richardson 3차 근사 계수(15~20개, 각각
c2/c3/c4의 유리함수) 전사 실수 위험이 크고 150~250줄 규모라, 범위를
평면(z=0) 리아푸노프 궤도로 좁혔다. 리아푸노프 궤도는 19번이 이미 만든 4x4
평면 야코비안(jacobian_at_point)의 순허수 고유벡터를 그대로 초기 추정치로
쓸 수 있어 헤일로보다 훨씬 안전하고, "라그랑주점 외에도 주기궤도 가족이
존재한다"는 핵심 메시지는 동일하게 전달한다.

핵심 개념 1: 야코비안의 순허수 고유벡터가 선형 리아푸노프 궤도를 근사한다
  19번에서 확인했듯 L1/L2의 4x4 평면 야코비안은 고유값 두 쌍을 갖는다 -
  실수 쌍(±λ, 안장점 방향, 불안정)과 순허수 쌍(±iω, 중심 방향, 진동). 순허수
  고유값의 고유벡터 방향으로 라그랑주점을 작은 진폭만큼 섭동시키면, 선형화된
  계에서는 주기 T=2π/ω로 닫히는 타원 궤도가 나온다. 이것이 진짜 비선형
  주기궤도의 1차 근사(초기 추정치)다 - 그 자체로는 한 바퀴 돈 뒤 정확히
  출발점으로 돌아오지 않는다.

핵심 개념 2: 상태천이행렬(STM)이 미분수정의 나침반이다
  초기조건을 조금 바꿨을 때 반주기 후 상태가 어떻게 바뀌는지 아는 게
  미분수정의 핵심이다. 이를 위해 상태 4차원과 STM 16차원을 합친 20차원 계를
  RK4로 함께 적분한다(Φ̇=A(t)Φ, Φ(0)=단위행렬). A(t)는 평형점에서의 고정
  야코비안이 아니라, 궤적 위의 매 순간에서 계산하는 일반 야코비안이다 -
  19번의 jacobian_at_point(평형점 전용)와는 다른 새 함수가 필요하다.

핵심 개념 3: x축 대칭을 이용한 단일 슈팅 미분수정
  리아푸노프 궤도는 x축을 수직으로 지난다(y=0에서 vx=0). 이 대칭성 덕분에
  자유변수가 x0, vy0 두 개뿐이고(y0=0, vx0=0은 항상 고정), 반주기 후 y=0으로
  돌아오는 순간 vx=0이 되도록 vy0 하나만 뉴턴법으로 보정하면 충분하다(1차원
  뉴턴 스텝). 반주기 지점은 y=0을 지나는 순간을 선형보간으로 정밀하게
  찾아야 한다(고정 스텝 RK4가 정확히 그 지점에 멈추지 않으므로).

핵심 개념 4: 진폭을 늘려가면 궤도 가족이 나오고, 주기가 선형 예측에서 벗어난다
  진폭을 0에 가깝게 하면 주기는 정확히 2π/ω(선형 예측)에 수렴하지만, 진폭이
  커질수록(그래도 여전히 작은, 무차원 0.05 안팎) 비선형 효과로 주기가 그
  값에서 벗어난다. 이 편차 자체가 "선형 근사는 출발점일 뿐, 진짜 주기궤도는
  미분수정으로 완성해야 한다"는 핵심 메시지를 수치로 보여준다.

단순화: 평면(z=0) 리아푸노프 궤도만 다룬다 - 3차원 헤일로 궤도, 수직 궤도는
범위 밖이다. 진폭 범위도 작게(무차원 0.001~0.05, L1이 원점에서 약
0.83~0.84 떨어져 있으므로 약 6% 이내) 제한해 미분수정이 안정적으로 수렴하는
구간만 다룬다. 궤도의 안정성(모노드로미 행렬 고유값)은 다루지 않는다 -
궤도를 찾고 완성하는 것까지가 범위다.
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

_LAGRANGE_PATH = os.path.join(_THIS_DIR, "lagrange_points.py")
_lagrange_spec = importlib.util.spec_from_file_location("lagrange_points_module", _LAGRANGE_PATH)
lagrange_points = importlib.util.module_from_spec(_lagrange_spec)
_lagrange_spec.loader.exec_module(lagrange_points)

EARTH_MOON_MASS_RATIO = lagrange_points.EARTH_MOON_MASS_RATIO
find_collinear_lagrange_points = lagrange_points.find_collinear_lagrange_points
jacobian_at_point = lagrange_points.jacobian_at_point

DEFAULT_AMPLITUDE = 0.01
CORRECTION_TOLERANCE = 1e-10
MAX_CORRECTION_ITERATIONS = 40


def planar_center_eigenvector(x_eq, y_eq, mu):
  """jacobian_at_point의 4개 고유값 중 순허수 쌍(±iω)에서 +iω 하나를 골라
  중심(center) 고유벡터의 실수 방향을 반환한다. 이 방향으로 섭동시키면
  선형화된 계에서 주기 T=2π/ω로 닫히는 타원 궤도가 나온다(핵심 개념 1).
  반환: (omega, direction) — direction은 [x,y,vx,vy] 4차원 실수 벡터."""
  eigenvalues, eigenvectors = np.linalg.eig(jacobian_at_point(x_eq, y_eq, mu))
  center_mask = (np.abs(eigenvalues.real) < 1e-6 * np.abs(eigenvalues.imag)) & (eigenvalues.imag > 0)
  center_indices = np.where(center_mask)[0]
  assert len(center_indices) == 1, (
      f"순허수 고유값(+iω)이 정확히 하나여야 하는데 {len(center_indices)}개 발견됨 - "
      "평형점 좌표나 질량비가 잘못됐을 수 있음")
  idx = center_indices[0]
  omega = eigenvalues[idx].imag
  v_complex = eigenvectors[:, idx]
  direction = v_complex.real
  if np.linalg.norm(direction) < 1e-8:
    direction = v_complex.imag
  return omega, direction


def linear_ic_guess(x_eq, y_eq, mu, amplitude):
  """중심 고유벡터를 진폭만큼 스케일해 비선형 리아푸노프 궤도의 초기
  추정치를 만든다. 그 자체로는 정확한 주기궤도가 아니라 미분수정의
  출발점일 뿐이다. 반환: (x0, vy0, omega) — y0=0, vx0=0은 x축 대칭성으로
  항상 0(리아푸노프 궤도는 x축을 수직으로 지남)."""
  omega, direction = planar_center_eigenvector(x_eq, y_eq, mu)
  direction = direction / np.linalg.norm(direction)
  scaled = direction * amplitude
  x0 = x_eq + scaled[0]
  vy0 = scaled[3]
  return x0, vy0, omega


def planar_jacobian_general(x, y, vx, mu):
  """평형점이 아닌 궤적 위 임의 상태에서 평면 CR3BP 우변의 4x4 야코비 행렬
  (STM 전파용 A(t)). jacobian_at_point는 평형점(속도 0)에서만 유효하므로
  궤적을 따라가며 매 순간 다시 계산해야 하는 이 함수가 별도로 필요하다."""
  r1 = np.sqrt((x + mu) ** 2 + y ** 2)
  r2 = np.sqrt((x - (1 - mu)) ** 2 + y ** 2)

  uxx = (1 - (1 - mu) / r1 ** 3 - mu / r2 ** 3
         + 3 * (1 - mu) * (x + mu) ** 2 / r1 ** 5
         + 3 * mu * (x - (1 - mu)) ** 2 / r2 ** 5)
  uyy = (1 - (1 - mu) / r1 ** 3 - mu / r2 ** 3
         + 3 * (1 - mu) * y ** 2 / r1 ** 5
         + 3 * mu * y ** 2 / r2 ** 5)
  uxy = (3 * (1 - mu) * (x + mu) * y / r1 ** 5
         + 3 * mu * (x - (1 - mu)) * y / r2 ** 5)

  return np.array([
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0],
      [uxx, uxy, 0.0, 2.0],
      [uxy, uyy, -2.0, 0.0],
  ])


def _planar_derivative(state4, mu):
  """평면(z=0) CR3BP 상태 [x,y,vx,vy]의 시간미분. 19번 cr3bp_acceleration의
  평면 특수화(z=vz=0 고정)와 수치적으로 동일하지만, STM 결합 적분에서
  4차원만 다루는 게 더 간단해 이 스크립트 전용으로 다시 정의한다."""
  x, y, vx, vy = state4
  r1 = np.sqrt((x + mu) ** 2 + y ** 2)
  r2 = np.sqrt((x - (1 - mu)) ** 2 + y ** 2)
  ax = x + 2 * vy - (1 - mu) * (x + mu) / r1 ** 3 - mu * (x - (1 - mu)) / r2 ** 3
  ay = y - 2 * vx - (1 - mu) * y / r1 ** 3 - mu * y / r2 ** 3
  return np.array([vx, vy, ax, ay])


def rk4_step_state_and_stm(state4, stm, dt, mu):
  """4차원 평면 상태와 4x4 STM을 묶은 20차원 계를 한 스텝 RK4로 전파한다.
  Φ̇=A(t)Φ를 상태 전파와 동시에 적분해 미분수정에 필요한 민감도 행렬을
  얻는다(핵심 개념 2)."""
  def deriv(s, phi):
    a = planar_jacobian_general(s[0], s[1], s[2], mu)
    return _planar_derivative(s, mu), a @ phi

  k1s, k1p = deriv(state4, stm)
  k2s, k2p = deriv(state4 + dt / 2 * k1s, stm + dt / 2 * k1p)
  k3s, k3p = deriv(state4 + dt / 2 * k2s, stm + dt / 2 * k2p)
  k4s, k4p = deriv(state4 + dt * k3s, stm + dt * k3p)

  new_state = state4 + dt / 6 * (k1s + 2 * k2s + 2 * k3s + k4s)
  new_stm = stm + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
  return new_state, new_stm


def find_half_period_crossing(x0, vy0, mu, dt, max_steps=200_000):
  """y=0을 벗어난 뒤 다시 y=0으로 돌아오는 half-period 이벤트를
  선형보간으로 정밀하게 찾는다(핵심 개념 3). 반환: t_half, state_half(4,),
  stm_half(4,4)."""
  state = np.array([x0, 0.0, 0.0, vy0])
  stm = np.eye(4)
  t = 0.0
  prev_state, prev_stm, prev_t = state.copy(), stm.copy(), t

  for step in range(max_steps):
    state, stm = rk4_step_state_and_stm(state, stm, dt, mu)
    t += dt
    if step > 0 and prev_state[1] * state[1] < 0:
      frac = -prev_state[1] / (state[1] - prev_state[1])
      t_half = prev_t + frac * dt
      state_half = prev_state + frac * (state - prev_state)
      stm_half = prev_stm + frac * (stm - prev_stm)
      return {"t_half": t_half, "state_half": state_half, "stm_half": stm_half}
    prev_state, prev_stm, prev_t = state.copy(), stm.copy(), t

  raise RuntimeError(f"{max_steps}스텝 안에 y=0 재교차를 찾지 못함 - 마지막 상태: {state}")


def differential_correct_lyapunov_orbit(x_eq, mu, amplitude, dt=0.002,
                                         tol=CORRECTION_TOLERANCE,
                                         max_iterations=MAX_CORRECTION_ITERATIONS):
  """x축 대칭 단일 슈팅 미분수정. vy0를 뉴턴 스텝으로 갱신하며 half-period
  교차에서 vx=0이 될 때까지 반복한다(핵심 개념 3). 반환: x0, vy0,
  half_period, converged, iterations, residual_vx, residual_history."""
  x0, vy0, _ = linear_ic_guess(x_eq, 0.0, mu, amplitude)
  residual_history = []

  for iteration in range(max_iterations):
    crossing = find_half_period_crossing(x0, vy0, mu, dt)
    state_half = crossing["state_half"]
    vx_half = state_half[2]
    residual_history.append(abs(vx_half))

    if abs(vx_half) < tol:
      return {"x0": x0, "vy0": vy0, "half_period": crossing["t_half"],
              "converged": True, "iterations": iteration + 1,
              "residual_vx": abs(vx_half), "residual_history": residual_history}

    stm_half = crossing["stm_half"]
    ax_half = _planar_derivative(state_half, mu)[2]  # vx의 시간미분(가속도)
    vy_half = state_half[3]
    # 교차 조건 y=0을 유지하려면 δt = -Φ[y,vy0]·δvy0/vy 이므로 d(vx)/d(vy0) = Φ[vx,vy0] - Φ[y,vy0]·ax/vy
    denom = stm_half[2, 3] - stm_half[1, 3] * ax_half / vy_half if abs(vy_half) > 1e-14 else stm_half[2, 3]
    if abs(denom) < 1e-12:
      return {"x0": x0, "vy0": vy0, "half_period": crossing["t_half"],
              "converged": False, "iterations": iteration + 1,
              "residual_vx": abs(vx_half), "residual_history": residual_history}
    vy0 = vy0 - vx_half / denom

  # 마지막 반복에서 vy0를 갱신했으므로 residual_vx는 갱신 전 vy0 기준이다
  return {"x0": x0, "vy0": vy0, "half_period": None, "converged": False,
          "iterations": max_iterations, "residual_vx": residual_history[-1],
          "residual_history": residual_history}


def propagate_full_orbit(x0, vy0, mu, dt, num_periods=1.0, half_period=None):
  """수정된 초기조건으로 전체 주기(또는 num_periods배) 동안 궤적을 적분해
  플로팅용 (t,x,y) 이력을 만든다. half_period*2를 온전한 주기 T로 사용.
  목표 시간을 dt로 나눈 몫만큼 스텝 수를 잡고 실제 스텝 크기를 살짝 조정해
  마지막 스텝이 정확히 주기 경계에서 끝나게 한다 - 그렇지 않으면 이산화로
  남는 자투리 시간(최대 dt 미만)만큼 주기성 오차가 인위적으로 섞인다."""
  if half_period is None:
    half_period = find_half_period_crossing(x0, vy0, mu, dt)["t_half"]
  period = 2 * half_period
  target_time = period * num_periods
  num_steps = max(1, round(target_time / dt))
  actual_dt = target_time / num_steps

  state = np.array([x0, 0.0, 0.0, vy0])
  history = [(0.0, state[0], state[1])]
  for step in range(num_steps):
    state = _rk4_step_state_only(state, actual_dt, mu)
    history.append(((step + 1) * actual_dt, state[0], state[1]))
  return {"history": history, "period": period, "final_state": state}


def _rk4_step_state_only(state4, dt, mu):
  """STM 없이 상태 4차원만 RK4로 한 스텝 전파(전체 궤적 플로팅용, STM 계산
  비용을 아끼기 위해 미분수정과 별도 함수로 둔다)."""
  k1 = _planar_derivative(state4, mu)
  k2 = _planar_derivative(state4 + dt / 2 * k1, mu)
  k3 = _planar_derivative(state4 + dt / 2 * k2, mu)
  k4 = _planar_derivative(state4 + dt * k3, mu)
  return state4 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def lyapunov_family(x_eq, mu, amplitudes, dt=0.002):
  """진폭을 늘려가며 순차적으로 미분수정을 반복 실행하는 궤도 가족 계산
  (핵심 개념 4). 각 진폭의 수렴 결과를 리스트로 반환한다."""
  results = []
  for amplitude in amplitudes:
    result = differential_correct_lyapunov_orbit(x_eq, mu, amplitude, dt=dt)
    result["amplitude"] = amplitude
    results.append(result)
  return results


def demo_linear_guess_matches_predicted_period():
  """작은 진폭에서 선형화된 계를 1주기(T=2π/ω) 적분해 타원이 닫히고 주기가
  예측과 일치하는지 확인한다 - 고유벡터 선택과 ω 자체의 정합성 검증,
  비선형 미분수정 이전 단계."""
  print("=" * 70)
  print("[1] 선형 추정: 중심 고유벡터 방향 섭동이 예측 주기로 닫히는가")
  print("=" * 70)
  mu = EARTH_MOON_MASS_RATIO
  x_l1, _, _ = find_collinear_lagrange_points(mu)
  amplitude = 0.0001
  omega, _ = planar_center_eigenvector(x_l1, 0.0, mu)
  period_predicted = 2 * np.pi / omega
  print(f"L1 x좌표={x_l1:.6f}, 중심 고유값 각속도 ω={omega:.6f}, 예측 주기={period_predicted:.4f} (무차원)")

  x0, vy0, _ = linear_ic_guess(x_l1, 0.0, mu, amplitude)
  dt = 0.001
  num_steps = int(period_predicted / dt)
  state = np.array([x0, 0.0, 0.0, vy0])
  for _ in range(num_steps):
    state = _rk4_step_state_only(state, dt, mu)

  distance_from_start = np.hypot(state[0] - x0, state[1] - 0.0)
  print(f"1주기 적분 후 시작점과의 거리: {distance_from_start:.6e} (진폭 {amplitude}의 {distance_from_start / amplitude * 100:.2f}%)")

  assert distance_from_start < amplitude * 0.05, "선형 추정 궤도는 1주기 후 진폭의 5% 이내로 시작점에 복귀해야 함"
  print("\n(중심 고유벡터 방향으로 섭동시킨 선형화 궤적이 예측 주기 2π/ω 후")
  print(" 거의 정확히 시작점으로 돌아온다 - 고유벡터 선택과 ω 계산이 올바르다는 신호다.")
  print(" 완전히 0이 아닌 이유는 비선형 항 때문이며, 이걸 없애는 게 다음 단계인 미분수정이다.)")
  return {"mu": mu, "x_l1": x_l1, "omega": omega, "period_predicted": period_predicted,
          "distance_from_start": distance_from_start}


def demo_differential_correction_converges():
  """진폭 0.01에서 미분수정을 실행해 반복별 잔여 vx가 줄어들며 수렴하는지,
  수정된 궤도가 실제로 주기적인지(전체 주기 적분 후 시작점 복귀) 확인한다
  - 이 스크립트의 핵심 검증."""
  print("\n" + "=" * 70)
  print("[2] 미분수정: 반주기 교차에서 vx=0이 될 때까지 vy0를 뉴턴법으로 보정")
  print("=" * 70)
  mu = EARTH_MOON_MASS_RATIO
  x_l1, _, _ = find_collinear_lagrange_points(mu)
  amplitude = 0.01

  result = differential_correct_lyapunov_orbit(x_l1, mu, amplitude)
  print(f"진폭={amplitude}, 반복 횟수={result['iterations']}")
  for i, residual in enumerate(result["residual_history"]):
    print(f"  반복 {i + 1}: |vx at half-period| = {residual:.3e}")

  assert result["converged"], "작은 진폭(0.01)에서는 미분수정이 반드시 수렴해야 함"
  assert result["residual_vx"] < 1e-9, "수렴 후 잔여 vx는 1e-9 미만이어야 함"

  full_orbit = propagate_full_orbit(result["x0"], result["vy0"], mu, dt=0.001,
                                     half_period=result["half_period"])
  final_state = full_orbit["final_state"]
  periodicity_error = np.hypot(final_state[0] - result["x0"], final_state[1] - 0.0)
  print(f"\n전체 주기(T={full_orbit['period']:.4f}) 적분 후 시작점과의 거리: {periodicity_error:.3e}")

  assert periodicity_error < 1e-6, "미분수정된 궤도는 전체 주기 후 시작점에 거의 정확히 복귀해야 함(주기성)"
  print("\n(뉴턴법이 몇 번 반복만에 잔여 vx를 1e-12 수준까지 줄였고, 그렇게 수정된")
  print(" 초기조건으로 전체 주기를 적분하면 시작점으로 거의 완벽하게 돌아온다 -")
  print(" 선형 추정과 달리 이번엔 진짜 비선형 주기궤도를 찾은 것이다.)")
  return {"mu": mu, "x_l1": x_l1, "amplitude": amplitude, "result": result,
          "full_orbit": full_orbit, "periodicity_error": periodicity_error}


def demo_lyapunov_family_amplitude_vs_period():
  """진폭 0.001~0.05를 스윕해 전부 수렴하는지, 주기가 선형 예측(평평한 값)
  에서 벗어나는지 확인한다 - 진짜 주기궤도는 선형 근사와 다르다는 핵심
  메시지."""
  print("\n" + "=" * 70)
  print("[3] 리아푸노프 궤도 가족: 진폭이 커질수록 주기가 선형 예측에서 벗어남")
  print("=" * 70)
  mu = EARTH_MOON_MASS_RATIO
  x_l1, _, _ = find_collinear_lagrange_points(mu)
  omega, _ = planar_center_eigenvector(x_l1, 0.0, mu)
  period_linear = 2 * np.pi / omega

  amplitudes = np.linspace(0.001, 0.05, 8)
  results = lyapunov_family(x_l1, mu, amplitudes)

  print(f"선형 예측 주기(진폭 무관): {period_linear:.6f}\n")
  print(f"{'진폭':>8s}  {'수렴':>6s}  {'주기':>10s}  {'선형예측 대비 편차(%)':>18s}")
  for r in results:
    period = 2 * r["half_period"] if r["converged"] else float("nan")
    deviation_pct = abs(period - period_linear) / period_linear * 100 if r["converged"] else float("nan")
    print(f"{r['amplitude']:8.4f}  {r['converged']!s:>6s}  {period:10.6f}  {deviation_pct:18.4f}")

  assert all(r["converged"] for r in results), "안전 진폭 범위(0.001~0.05) 안에서는 모두 수렴해야 함"
  smallest_deviation = abs(2 * results[0]["half_period"] - period_linear) / period_linear
  largest_deviation = abs(2 * results[-1]["half_period"] - period_linear) / period_linear
  assert largest_deviation > smallest_deviation, "진폭이 커질수록 선형 예측과의 편차도 커져야 함(비선형 효과)"
  print("\n(진폭이 커질수록 실제 주기가 선형 예측 2π/ω에서 점점 벗어난다 - 이게 바로")
  print(" 미분수정이 필요한 이유다: 선형 근사는 진폭이 작을 때만 좋은 추정치를 준다.)")
  return {"mu": mu, "x_l1": x_l1, "period_linear": period_linear, "results": results}


def demo_compare_l1_l2_families():
  """L1과 L2에서 각각 궤도 가족을 계산해 같은 진폭에서 주기가 서로 다르지만
  같은 자릿수임을 확인한다 - x_eq 매개변수화 하나로 메서드가 일반화됨을
  보여준다."""
  print("\n" + "=" * 70)
  print("[4] L1 vs L2: 같은 방법이 두 평형점 모두에서 동작하는가")
  print("=" * 70)
  mu = EARTH_MOON_MASS_RATIO
  x_l1, x_l2, _ = find_collinear_lagrange_points(mu)
  amplitude = 0.02

  result_l1 = differential_correct_lyapunov_orbit(x_l1, mu, amplitude)
  result_l2 = differential_correct_lyapunov_orbit(x_l2, mu, amplitude)

  period_l1 = 2 * result_l1["half_period"] if result_l1["converged"] else float("nan")
  period_l2 = 2 * result_l2["half_period"] if result_l2["converged"] else float("nan")
  print(f"L1: x={x_l1:.6f}, 수렴={result_l1['converged']}, 주기={period_l1:.6f}")
  print(f"L2: x={x_l2:.6f}, 수렴={result_l2['converged']}, 주기={period_l2:.6f}")

  assert result_l1["converged"] and result_l2["converged"], "L1, L2 모두 같은 진폭에서 수렴해야 함"
  ratio = period_l2 / period_l1
  assert 0.5 < ratio < 2.0, "L1/L2 주기는 물리적으로 같은 자릿수여야 함(둘 다 지구-달 공전 스케일)"
  print(f"\n주기 비율(L2/L1): {ratio:.4f}")
  print("\n(같은 코드가 x_eq 하나만 바꿔서 L1, L2 양쪽에서 모두 수렴하는 진짜 주기궤도를 찾는다 -")
  print(" 미분수정 방법 자체가 특정 라그랑주점에 종속되지 않는다는 것을 보여준다.)")
  return {"mu": mu, "x_l1": x_l1, "x_l2": x_l2, "amplitude": amplitude,
          "result_l1": result_l1, "result_l2": result_l2, "ratio": ratio}


def parse_args():
  parser = argparse.ArgumentParser(description="원형 제한 3체 문제(CR3BP)에서 평면 리아푸노프 궤도를 미분수정으로 찾기")
  parser.add_argument("--amplitude", type=float, default=DEFAULT_AMPLITUDE,
                       help=f"리아푸노프 궤도 진폭(무차원), 기본값: {DEFAULT_AMPLITUDE}")
  parser.add_argument("--libration-point", choices=["L1", "L2"], default="L1",
                       help="리아푸노프 궤도를 계산할 평형점, 기본값: L1")
  parser.add_argument("--mass-ratio", type=float, default=EARTH_MOON_MASS_RATIO,
                       help=f"질량비 mu=M2/(M1+M2), 기본값: 지구-달계 약 {EARTH_MOON_MASS_RATIO:.6f}")
  return parser.parse_args()


def main():
  args = parse_args()

  demo_linear_guess_matches_predicted_period()
  demo_differential_correction_converges()
  family_result = demo_lyapunov_family_amplitude_vs_period()
  comparison_result = demo_compare_l1_l2_families()

  mu = args.mass_ratio
  x_l1, x_l2, _ = find_collinear_lagrange_points(mu)
  x_eq = x_l1 if args.libration_point == "L1" else x_l2
  print("\n" + "=" * 70)
  print(f"[사용자 지정] 평형점={args.libration_point}, 진폭={args.amplitude}, 질량비={mu:.6f}")
  print("=" * 70)
  custom_result = differential_correct_lyapunov_orbit(x_eq, mu, args.amplitude)
  if custom_result["converged"]:
    custom_orbit = propagate_full_orbit(custom_result["x0"], custom_result["vy0"], mu, dt=0.001,
                                         half_period=custom_result["half_period"])
    print(f"수렴={custom_result['converged']}, x0={custom_result['x0']:.6f}, "
          f"vy0={custom_result['vy0']:.6f}, 주기={custom_orbit['period']:.6f}")
  else:
    print(f"수렴 실패 (반복 {custom_result['iterations']}회, 잔여 vx={custom_result['residual_vx']:.3e})")
    custom_orbit = propagate_full_orbit(custom_result["x0"], custom_result["vy0"], mu, dt=0.001, num_periods=1.0)

  results_dir = os.path.join(_ROOT_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  trajectory_csv = os.path.join(results_dir, "lyapunov_orbits_trajectory.csv")
  with open(trajectory_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_dimensionless", "x", "y"])
    for t, x, y in custom_orbit["history"]:
      writer.writerow([f"{t:.6f}", f"{x:.6f}", f"{y:.6f}"])
  print(f"\n[기록] 사용자 지정 궤도 저장됨 → {trajectory_csv}")

  family_csv = os.path.join(results_dir, "lyapunov_orbits_family.csv")
  with open(family_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["libration_point", "amplitude", "converged", "period", "period_linear_prediction"])
    for r in family_result["results"]:
      period = f"{2 * r['half_period']:.6f}" if r["converged"] else ""
      writer.writerow(["L1", f"{r['amplitude']:.6f}", str(r["converged"]), period,
                        f"{family_result['period_linear']:.6f}"])
    comparison_amplitude = comparison_result["amplitude"]
    for name, r in [("L1", comparison_result["result_l1"]), ("L2", comparison_result["result_l2"])]:
      period = f"{2 * r['half_period']:.6f}" if r["converged"] else ""
      writer.writerow([name, f"{comparison_amplitude:.6f}", str(r["converged"]), period, ""])
  print(f"[기록] 궤도 가족 결과 저장됨 → {family_csv}")

  points_csv = os.path.join(results_dir, "lyapunov_orbits_lagrange_reference.csv")
  with open(points_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["point", "x_dimensionless"])
    writer.writerow(["L1", f"{x_l1:.9f}"])
    writer.writerow(["L2", f"{x_l2:.9f}"])
    writer.writerow(["M2_Moon", f"{1 - mu:.9f}"])
  print(f"[기록] 라그랑주점 참조 좌표 저장됨 → {points_csv}")


if __name__ == "__main__":
  main()
