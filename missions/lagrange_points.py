"""
3체 문제와 라그랑주점 - 원형 제한 3체 문제(CR3BP)로 지구-달계의 평형점 찾기

12번(Patched Conic)은 태양-지구-화성이라는 진짜 3체 문제를 명시적으로 피하고
세 개의 독립된 2체 문제로 이어붙였다. 이 스크립트는 12번이 피했던 문제를 정면으로
연다 - 원형 제한 3체 문제(Circular Restricted Three-Body Problem, CR3BP)로
두 개의 큰 천체(지구-달)가 만드는 회전 중력장 안에서, 질량이 무시할 만큼 작은
세 번째 물체(우주선)가 정지해 있을 수 있는 5개의 평형점(라그랑주점 L1~L5)을
찾고, 그중 L4/L5가 안정적이고 L1~L3는 불안정하다는 것을 수치로 확인한다.

01~18번은 모두 "하나의 중심천체 + 작은 섭동/추력/항력"이라는 공통 골격이라
two_body_acceleration을 그대로 또는 사소하게 확장해 재사용할 수 있었다. CR3BP는
그렇지 않다 - 두 개의 중심천체가 공통 질량중심을 공전하고, 그 공전과 함께
회전하는 좌표계에서 운동방정식을 쓴다. 이 회전좌표계에는 원심력과 코리올리
힘이라는, 이 프로젝트에 처음 등장하는 항이 들어간다. 또한 무차원화(정규화)를
쓴다 - 두 천체 사이 거리를 1, 공통 각속도를 1, 전체 질량을 1로 두고, 질량비
mu = M2/(M1+M2) 하나로 계의 기하가 전부 결정된다. km, km/s 같은 이 프로젝트의
기존 단위계를 그대로 쓰지 않는다. 그래서 이 스크립트는 03번 RK4 적분기 구조를
복제한 15/17/18번과 달리, CR3BP 고유의 함수 세트를 새로 만든다 - 재사용 가능한
이전 스크립트가 거의 없고, orbit_math.py의 회전행렬도 3차원 관성좌표계 변환용
이라 이 스크립트가 쓰는 "질량중심 공전과 함께 도는 좌표계" 개념과는 다르므로
재사용하지 않는다. 01번의 뉴턴-랍슨과 09번(란베르트)이 특이점 근처에서 뉴턴-랍슨
대신 이분법으로 바꿔 발산을 피한 교훈만 구조적으로 재사용한다.

핵심 개념 1: 회전좌표계 운동방정식 - 원심력과 코리올리 힘이 처음 등장한다
  두 주천체(질량 M1 > M2)가 공통 질량중심 주위를 원궤도로 공전한다고 가정하고,
  거리 단위를 두 천체 사이 거리로, 시간 단위를 1/공전각속도로, 질량 단위를
  M1+M2로 정규화하면 mu = M2/(M1+M2) 하나로 계 전체가 결정된다. 회전좌표계에서
  M1은 (-mu, 0, 0)에, M2는 (1-mu, 0, 0)에 고정된다. 질량이 무시할 만한 세 번째
  물체의 무차원 위치를 (x,y,z), 두 주천체까지의 거리를 r1(M1까지), r2(M2까지)라
  하면
      x'' - 2y' = x - (1-mu)(x+mu)/r1^3 - mu(x-1+mu)/r2^3
      y'' + 2x' = y - (1-mu)y/r1^3 - mu*y/r2^3
      z'' = -(1-mu)z/r1^3 - mu*z/r2^3
  로 주어진다. 우변의 x, y 항이 원심력(회전좌표계라서 생김), -2y'/+2x' 항이
  코리올리 힘이다 - 01~18번의 어떤 관성좌표계 운동방정식에도 없던 새 항이다.

핵심 개념 2: 라그랑주점은 이 방정식이 정지해 있을 수 있는 5개의 평형점이다
  가속도와 속도가 모두 0인 점을 찾는 문제로 귀착된다. L1/L2/L3는 y=z=0인 x축
  위에 있고, 유효 퍼텐셜의 x축 단면을 미분한 5차 다항식의 근이라 각 구간
  (M1-M2 사이, M2 바깥쪽, M1 바깥쪽)에서 이분법으로 풀어야 한다. L4/L5는 두
  주천체와 정삼각형을 이루는 점으로, 닫힌 공식 (x,y) = (0.5-mu, +-sqrt(3)/2)로
  바로 나온다 - 질량비와 무관하게 항상 정삼각형이라는 것 자체가 놀라운 결과다.

핵심 개념 3: L4/L5는 안정, L1/L2/L3는 불안정 - 선형화 고유값으로 확인
  각 라그랑주점 근처에서 운동방정식을 선형화하면 평면 내(x,y,x',y') 운동에
  대해 4x4 야코비 행렬이 나온다(z축 운동은 분리되어 항상 단순조화진동이므로
  평면 내 안정성 판정에서 제외). 이 행렬의 고유값이 순허수(진동, 안정)인지
  양의 실수부를 갖는지(발산, 불안정)로 안정성을 판정한다. 콜린스(L1/L2/L3)는
  모든 질량비에서 불안정하고, 트로이(L4/L5)는 질량비가 임계값(약 0.0385, 루스
  판별)보다 작을 때만 안정하다는 것을 지구-달 질량비(약 0.0121)로 직접 확인한다.

핵심 개념 4: 안정성은 직접 궤적을 쏘아봐서도 확인할 수 있다
  라그랑주점 근처에 작은 섭동을 준 초기조건으로 무차원 운동방정식을 RK4로 직접
  적분해, L4는 섭동을 줘도 근처에서 맴돌고(경계 유지) L1은 섭동을 주면 금방
  멀리 벗어나는 것을 궤적으로도 보여준다 - 16번(중간축 정리)에서 "고유값/에너지
  분석"과 "직접 수치적분"을 둘 다 보여줬던 것과 같은 이중 검증 패턴을 반복한다.

단순화: 원형 제한 3체 문제(CR3BP)만 다룬다 - 두 주천체의 궤도가 완전한 원이라고
가정하고(실제 지구-달 궤도는 이심률 약 0.055의 타원), 세 번째 물체의 질량이
계에 영향을 주지 않는다고 가정한다. 헤일로 궤도, 저에너지 전이 같은 CR3BP의
실전 응용은 다루지 않는다 - 라그랑주점의 위치와 선형 안정성까지만 다룬다.
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

EARTH_MOON_MASS_RATIO = 0.012150585  # mu = M_moon / (M_earth + M_moon)
EARTH_MOON_DISTANCE_KM = 384400.0  # 무차원 결과를 실감 나는 값으로 환산할 때만 사용
EARTH_MOON_PERIOD_DAYS = 27.321661  # 무차원 시간 1단위 = 이 값 / (2*pi) 일


def effective_potential_x_derivative(x, mu):
  """콜린스 라그랑주점(y=z=0, x축 위)이 만족하는 방정식. 유효 퍼텐셜의 x축
  단면을 미분한 식으로, x=-mu(M1)와 x=1-mu(M2)에서 특이점(분모 0)을 갖는다."""
  r1 = abs(x + mu)
  r2 = abs(x - (1 - mu))
  return x - (1 - mu) * (x + mu) / r1 ** 3 - mu * (x - (1 - mu)) / r2 ** 3


def _bisect_root(f, lo, hi, tol=1e-12, max_iter=200):
  """09번 란베르트에서 뉴턴-랍슨이 특이점 근처에서 발산했던 교훈을 반영해,
  콜린스 라그랑주점 근 찾기는 이분법을 기본으로 쓴다(각 구간에서 f가 부호를
  바꾼다는 것만 보장되면 반드시 수렴함)."""
  f_lo, f_hi = f(lo), f(hi)
  if f_lo == 0.0:
    return lo
  if f_hi == 0.0:
    return hi
  if f_lo * f_hi > 0:
    raise ValueError("구간 [lo, hi]에서 부호가 바뀌지 않아 이분법을 적용할 수 없음")
  for _ in range(max_iter):
    mid = (lo + hi) / 2
    f_mid = f(mid)
    if abs(f_mid) < tol:
      return mid
    if f_lo * f_mid < 0:
      hi, f_hi = mid, f_mid
    else:
      lo, f_lo = mid, f_mid
  return (lo + hi) / 2


def find_collinear_lagrange_points(mu):
  """L1(M1-M2 사이), L2(M2 바깥쪽), L3(M1 바깥쪽) x좌표를 각각 다른 구간에서
  이분법으로 찾는다. 세 점 모두 M1=(-mu,0,0), M2=(1-mu,0,0) 근처의 특이점을
  피해 안전한 초기 구간을 잡는다."""
  eps = 1e-6

  def f(x):
    return effective_potential_x_derivative(x, mu)

  x_l1 = _bisect_root(f, -mu + eps, 1 - mu - eps)
  x_l2 = _bisect_root(f, 1 - mu + eps, 2.0)
  x_l3 = _bisect_root(f, -2.0, -mu - eps)
  return x_l1, x_l2, x_l3


def triangular_lagrange_points(mu):
  """L4, L5는 두 주천체와 정삼각형을 이루는 닫힌 공식 - 질량비와 무관하게
  항상 (0.5-mu, +-sqrt(3)/2)다. 반환: (x_l4, y_l4), (x_l5, y_l5)."""
  x = 0.5 - mu
  y = np.sqrt(3) / 2
  return (x, y), (x, -y)


def jacobian_at_point(x, y, mu):
  """라그랑주점 (x,y) 근처에서 선형화한 평면 내(x,y,x',y') 4x4 야코비 행렬.
  z축 운동은 항상 단순조화진동으로 분리되므로 평면 내 안정성 판정에서
  제외한다(핵심 개념 3)."""
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


def stability_eigenvalues(x, y, mu):
  """야코비 행렬의 고유값. 실수부 최대값이 0보다 뚜렷이 크면 불안정,
  모든 고유값의 실수부가 0에 가까우면(순허수) 안정으로 판정한다."""
  jacobian = jacobian_at_point(x, y, mu)
  return np.linalg.eigvals(jacobian)


def cr3bp_acceleration(state, mu):
  """회전좌표계 CR3BP 운동방정식(핵심 개념 1). state=[x,y,z,vx,vy,vz] 무차원
  상태벡터를 받아 [vx,vy,vz,ax,ay,az]를 반환한다."""
  x, y, z, vx, vy, vz = state
  r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
  r2 = np.sqrt((x - (1 - mu)) ** 2 + y ** 2 + z ** 2)

  ax = x + 2 * vy - (1 - mu) * (x + mu) / r1 ** 3 - mu * (x - (1 - mu)) / r2 ** 3
  ay = y - 2 * vx - (1 - mu) * y / r1 ** 3 - mu * y / r2 ** 3
  az = -(1 - mu) * z / r1 ** 3 - mu * z / r2 ** 3
  return np.array([vx, vy, vz, ax, ay, az])


def rk4_step_cr3bp(state, dt, mu):
  """표준 RK4 한 스텝(01~18번과 같은 4단계 구조), 이번엔 무차원 시간/위치
  단위로 cr3bp_acceleration을 적분한다."""
  k1 = cr3bp_acceleration(state, mu)
  k2 = cr3bp_acceleration(state + dt / 2 * k1, mu)
  k3 = cr3bp_acceleration(state + dt / 2 * k2, mu)
  k4 = cr3bp_acceleration(state + dt * k3, mu)
  return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def propagate_near_lagrange_point(lagrange_xy, perturbation, num_steps, dt, mu):
  """라그랑주점 근처에 작은 섭동을 준 궤적을 RK4로 적분해 궤적 이력을 반환한다.
  반환: {"history": [(t, x, y), ...], "max_distance_from_point": float}."""
  x0, y0 = lagrange_xy
  state = np.array([x0 + perturbation[0], y0 + perturbation[1], 0.0,
                     perturbation[2], perturbation[3], 0.0])
  history = [(0.0, state[0], state[1])]
  max_distance = 0.0
  t = 0.0
  for _ in range(num_steps):
    state = rk4_step_cr3bp(state, dt, mu)
    t += dt
    distance = np.sqrt((state[0] - x0) ** 2 + (state[1] - y0) ** 2)
    max_distance = max(max_distance, distance)
    history.append((t, state[0], state[1]))
  return {"history": history, "max_distance_from_point": max_distance}


def demo_collinear_points_match_known_earth_moon_values():
  """지구-달 질량비로 L1/L2/L3를 계산해 문헌에서 흔히 인용되는 값(L1은
  지구-달 사이, L2/L3는 각각 달 바깥쪽/지구 반대쪽)과 비교한다."""
  print("=" * 70)
  print("[1] 콜린스 라그랑주점(L1/L2/L3): 지구-달계 문헌값과 비교")
  print("=" * 70)
  mu = EARTH_MOON_MASS_RATIO
  x_l1, x_l2, x_l3 = find_collinear_lagrange_points(mu)

  l1_km = x_l1 * EARTH_MOON_DISTANCE_KM
  l2_km = x_l2 * EARTH_MOON_DISTANCE_KM
  l3_km = x_l3 * EARTH_MOON_DISTANCE_KM
  print(f"L1: x={x_l1:.6f} (무차원) = {l1_km:,.0f}km (지구 중심 기준)")
  print(f"L2: x={x_l2:.6f} (무차원) = {l2_km:,.0f}km (지구 중심 기준)")
  print(f"L3: x={x_l3:.6f} (무차원) = {l3_km:,.0f}km (지구 중심 기준)")

  moon_from_earth_km = (1 - mu) * EARTH_MOON_DISTANCE_KM
  l1_from_moon_km = moon_from_earth_km - l1_km
  l2_from_moon_km = l2_km - moon_from_earth_km
  print(f"\n(참고: 달은 지구에서 {moon_from_earth_km:,.0f}km 지점)")
  print(f"L1은 달에서 지구 쪽으로 {l1_from_moon_km:,.0f}km, L2는 달 바깥쪽으로 {l2_from_moon_km:,.0f}km")

  assert 320_000.0 < l1_km < 330_000.0, "L1은 지구에서 약 320,000~330,000km 사이여야 함(문헌값 약 326,000km)"
  assert 440_000.0 < l2_km < 450_000.0, "L2는 지구에서 약 440,000~450,000km 사이여야 함(문헌값 약 449,000km)"
  assert -390_000.0 < l3_km < -380_000.0, "L3는 지구 반대편, 약 -381,000~-390,000km 사이여야 함(문헌값 약 -381,000km)"
  print("\n(L1/L2가 달 근방 6만km 안팎, L3가 지구 반대편 거의 달 거리만큼 떨어진 지점이라는")
  print(" 잘 알려진 문헌값과 일치한다 - 이분법으로 푼 5차 방정식 근이 올바르다는 신호다.)")
  return {"mu": mu, "x_l1": x_l1, "x_l2": x_l2, "x_l3": x_l3,
          "l1_km": l1_km, "l2_km": l2_km, "l3_km": l3_km}


def demo_triangular_points_form_equilateral_triangle():
  """L4/L5와 두 주천체 사이 거리가 세 변 모두 정확히 1(무차원)인 정삼각형을
  이루는지, 질량비를 바꿔도 항상 성립하는지 확인한다."""
  print("\n" + "=" * 70)
  print("[2] 삼각 라그랑주점(L4/L5): 질량비와 무관하게 항상 정삼각형")
  print("=" * 70)
  mass_ratios = [EARTH_MOON_MASS_RATIO, 0.001, 0.1, 0.3]
  rows = []
  for mu in mass_ratios:
    (x4, y4), (x5, y5) = triangular_lagrange_points(mu)
    m1 = np.array([-mu, 0.0])
    m2 = np.array([1 - mu, 0.0])
    p4 = np.array([x4, y4])
    d_m1_m2 = np.linalg.norm(m2 - m1)
    d_m1_l4 = np.linalg.norm(p4 - m1)
    d_m2_l4 = np.linalg.norm(p4 - m2)
    print(f"mu={mu:.6f}: |M1-M2|={d_m1_m2:.9f}, |M1-L4|={d_m1_l4:.9f}, |M2-L4|={d_m2_l4:.9f}")
    rows.append({"mu": mu, "x_l4": x4, "y_l4": y4, "x_l5": x5, "y_l5": y5,
                 "d_m1_m2": d_m1_m2, "d_m1_l4": d_m1_l4, "d_m2_l4": d_m2_l4})
    assert abs(d_m1_m2 - 1.0) < 1e-9, "무차원화에서 두 주천체 사이 거리는 정확히 1이어야 함"
    assert abs(d_m1_l4 - 1.0) < 1e-9, "L4는 M1로부터 거리 1(정삼각형 변)이어야 함"
    assert abs(d_m2_l4 - 1.0) < 1e-9, "L4는 M2로부터 거리 1(정삼각형 변)이어야 함"

  print("\n(질량비를 0.0012부터 0.3까지 250배 넘게 바꿔도 세 변의 길이가 항상 정확히 1이다 -")
  print(" L4/L5가 두 주천체와 이루는 정삼각형은 질량비에 전혀 의존하지 않는다는 것을 확인했다.)")
  return rows


def demo_collinear_points_are_unstable():
  """L1/L2/L3에서 야코비 행렬 고유값의 실수부에 뚜렷한 양수가 존재하는지
  (불안정) 확인한다."""
  print("\n" + "=" * 70)
  print("[3] 콜린스 라그랑주점의 선형 안정성: 모두 불안정해야 한다")
  print("=" * 70)
  mu = EARTH_MOON_MASS_RATIO
  x_l1, x_l2, x_l3 = find_collinear_lagrange_points(mu)

  rows = []
  for name, x in [("L1", x_l1), ("L2", x_l2), ("L3", x_l3)]:
    eigenvalues = stability_eigenvalues(x, 0.0, mu)
    max_real = np.max(eigenvalues.real)
    print(f"{name}: 고유값={np.round(eigenvalues, 4)}")
    print(f"     최대 실수부={max_real:.6f}")
    rows.append({"point": name, "x": x, "y": 0.0, "max_real_eigenvalue": max_real})
    assert max_real > 0.1, f"{name}은 최대 고유값 실수부가 뚜렷한 양수여야 함(불안정)"

  print("\n(세 콜린스 라그랑주점 모두 실수부가 양수인 고유값을 갖는다 - 아무리 작은 섭동도")
  print(" 시간이 지나면 지수적으로 커진다는 뜻이다. 이래서 L1/L2에 실제로 위성을 두려면")
  print(" 헤일로 궤도 + 주기적인 궤도 유지 기동이 필요하다.)")
  return rows


def demo_triangular_points_stable_for_earth_moon_mass_ratio():
  """L4/L5에서 지구-달 질량비 기준 고유값이 모두 순허수에 가까운지(안정)
  확인하고, 라그랑주점 근처 섭동 궤적을 RK4로 직접 적분해 L4는 유계,
  L1은 크게 발산하는 정성적 차이를 궤적으로도 확인한다."""
  print("\n" + "=" * 70)
  print("[4] 삼각 라그랑주점의 선형 안정성 + 직접 궤적 전파로 재검증")
  print("=" * 70)
  mu = EARTH_MOON_MASS_RATIO
  (x_l4, y_l4), _ = triangular_lagrange_points(mu)
  eigenvalues_l4 = stability_eigenvalues(x_l4, y_l4, mu)
  max_real_l4 = np.max(eigenvalues_l4.real)
  print(f"L4: 고유값={np.round(eigenvalues_l4, 4)}")
  print(f"    최대 실수부={max_real_l4:.6f}")
  assert abs(max_real_l4) < 1e-6, "지구-달 질량비에서 L4는 모든 고유값의 실수부가 0에 가까워야 함(안정)"
  print("\n(지구-달 질량비(약 0.0122)는 안정성 임계값(약 0.0385)보다 작아 L4/L5가 안정하다 -")
  print(" 실제로 지구-달계의 L4/L5 근처에 소행성/먼지가 트로이군처럼 모여 있는 이유다.)")

  dt = 0.001
  num_steps = 20_000
  total_dimless_time = num_steps * dt
  total_days = total_dimless_time * EARTH_MOON_PERIOD_DAYS / (2 * np.pi)
  print(f"\n적분 시간: 무차원 {total_dimless_time:.2f} = 실제 약 {total_days:.1f}일 (지구-달 공전 기준)")

  perturbation = (0.001, 0.001, 0.0, 0.0)
  l4_result = propagate_near_lagrange_point((x_l4, y_l4), perturbation, num_steps, dt, mu)

  x_l1, _, _ = find_collinear_lagrange_points(mu)
  l1_result = propagate_near_lagrange_point((x_l1, 0.0), perturbation, num_steps, dt, mu)

  print(f"\nL4 근처 섭동(크기={np.hypot(*perturbation[:2]):.4f}) 궤적: 최대 이탈거리={l4_result['max_distance_from_point']:.4f}")
  print(f"L1 근처 섭동(크기={np.hypot(*perturbation[:2]):.4f}) 궤적: 최대 이탈거리={l1_result['max_distance_from_point']:.4f}")

  ratio = l1_result["max_distance_from_point"] / l4_result["max_distance_from_point"]
  print(f"\nL1/L4 최대 이탈거리 비율: {ratio:.1f}배")
  assert l1_result["max_distance_from_point"] > l4_result["max_distance_from_point"] * 5, \
      "L1 근처 궤적이 L4 근처 궤적보다 훨씬 크게 발산해야 함(불안정 vs 안정)"
  print("\n(같은 크기의 초기 섭동을 줘도, L1 근처 궤적은 라그랑주점에서 멀리 벗어나지만 L4 근처")
  print(" 궤적은 근처에 머문다 - 고유값 분석(선형 이론)과 RK4 직접 적분(비선형 실제 궤적)이")
  print(" 같은 결론을 준다는 것을 확인했다.)")
  return {"mu": mu, "max_real_l4": max_real_l4, "l4_history": l4_result["history"],
          "l1_history": l1_result["history"],
          "l4_max_distance": l4_result["max_distance_from_point"],
          "l1_max_distance": l1_result["max_distance_from_point"], "ratio": ratio}


def parse_args():
  parser = argparse.ArgumentParser(description="원형 제한 3체 문제(CR3BP)로 라그랑주점 위치/안정성 계산")
  parser.add_argument("--mass-ratio", type=float, default=EARTH_MOON_MASS_RATIO,
                       help=f"질량비 mu=M2/(M1+M2), 기본값: 지구-달계 약 {EARTH_MOON_MASS_RATIO:.6f}")
  parser.add_argument("--perturbation-km", type=float, default=1000.0,
                       help="라그랑주점 근처 궤적 전파에 쓸 섭동 크기(km, 지구-달 거리 기준 무차원 변환), 기본값: 1000km")
  return parser.parse_args()


def main():
  args = parse_args()

  collinear_result = demo_collinear_points_match_known_earth_moon_values()
  triangular_result = demo_triangular_points_form_equilateral_triangle()
  instability_result = demo_collinear_points_are_unstable()
  stability_result = demo_triangular_points_stable_for_earth_moon_mass_ratio()

  mu = args.mass_ratio
  perturbation_dimless = args.perturbation_km / EARTH_MOON_DISTANCE_KM
  x_l1, x_l2, x_l3 = find_collinear_lagrange_points(mu)
  (x_l4, y_l4), (x_l5, y_l5) = triangular_lagrange_points(mu)
  print("\n" + "=" * 70)
  print(f"[사용자 지정] 질량비={mu:.6f}, 섭동={args.perturbation_km:.0f}km")
  print("=" * 70)
  print(f"L1={x_l1:.6f}, L2={x_l2:.6f}, L3={x_l3:.6f} (무차원 x좌표)")
  print(f"L4=({x_l4:.6f}, {y_l4:.6f}), L5=({x_l5:.6f}, {y_l5:.6f})")
  custom_l4_result = propagate_near_lagrange_point(
      (x_l4, y_l4), (perturbation_dimless, 0.0, 0.0, 0.0), 5000, 0.001, mu)
  print(f"L4 근처 {args.perturbation_km:.0f}km 섭동 궤적 최대 이탈거리: "
        f"{custom_l4_result['max_distance_from_point'] * EARTH_MOON_DISTANCE_KM:,.0f}km")

  results_dir = os.path.join(_ROOT_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  points_csv = os.path.join(results_dir, "lagrange_points_positions.csv")
  with open(points_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["point", "x_dimensionless", "y_dimensionless", "x_km"])
    writer.writerow(["L1", f"{collinear_result['x_l1']:.9f}", "0.0", f"{collinear_result['l1_km']:.1f}"])
    writer.writerow(["L2", f"{collinear_result['x_l2']:.9f}", "0.0", f"{collinear_result['l2_km']:.1f}"])
    writer.writerow(["L3", f"{collinear_result['x_l3']:.9f}", "0.0", f"{collinear_result['l3_km']:.1f}"])
    writer.writerow(["L4", f"{x_l4:.9f}", f"{y_l4:.9f}", f"{x_l4 * EARTH_MOON_DISTANCE_KM:.1f}"])
    writer.writerow(["L5", f"{x_l5:.9f}", f"{y_l5:.9f}", f"{x_l5 * EARTH_MOON_DISTANCE_KM:.1f}"])
  print(f"\n[기록] 라그랑주점 위치 저장됨 → {points_csv}")

  stability_csv = os.path.join(results_dir, "lagrange_points_stability.csv")
  with open(stability_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["point", "max_real_eigenvalue", "stable"])
    for row in instability_result:
      writer.writerow([row["point"], f"{row['max_real_eigenvalue']:.6f}", "False"])
    writer.writerow(["L4", f"{stability_result['max_real_l4']:.9f}", "True"])
  print(f"[기록] 안정성 결과 저장됨 → {stability_csv}")

  l4_trajectory_csv = os.path.join(results_dir, "lagrange_points_l4_trajectory.csv")
  with open(l4_trajectory_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_dimensionless", "x", "y"])
    for t, x, y in stability_result["l4_history"]:
      writer.writerow([f"{t:.4f}", f"{x:.6f}", f"{y:.6f}"])
  print(f"[기록] L4 근처 섭동 궤적 저장됨 → {l4_trajectory_csv}")

  l1_trajectory_csv = os.path.join(results_dir, "lagrange_points_l1_trajectory.csv")
  with open(l1_trajectory_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_dimensionless", "x", "y"])
    for t, x, y in stability_result["l1_history"]:
      writer.writerow([f"{t:.4f}", f"{x:.6f}", f"{y:.6f}"])
  print(f"[기록] L1 근처 섭동 궤적 저장됨 → {l1_trajectory_csv}")

  triangle_csv = os.path.join(results_dir, "lagrange_points_triangle_check.csv")
  with open(triangle_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["mass_ratio", "d_m1_m2", "d_m1_l4", "d_m2_l4"])
    for row in triangular_result:
      writer.writerow([f"{row['mu']:.6f}", f"{row['d_m1_m2']:.9f}",
                        f"{row['d_m1_l4']:.9f}", f"{row['d_m2_l4']:.9f}"])
  print(f"[기록] 정삼각형 검증 결과 저장됨 → {triangle_csv}")


if __name__ == "__main__":
  main()
