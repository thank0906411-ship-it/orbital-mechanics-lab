"""orbit_raise_and_reorient.py 검증: 06번 호만 전이와 21번 PID 자세제어를 재검증
하지 않고, 두 단계를 잇는 이음매(seam)만 검증한다 — 궤도 전이 델타-V/시간이
hohmann_transfer.py를 직접 호출한 값과 정확히 같은지, 자세 재정렬 게인이
compute_pd_gains를 직접 호출한 값과 같은지, 자세 재정렬 단계의 첫 쿼터니언이
궤도 전이 완료 시점에 선언한 바로 그 임의 자세인지(핸드오프), 총 임무시간이
두 단계의 합과 정확히 같은지, 자세 재정렬이 델타-V에 전혀 기여하지 않는지."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("missions/orbit_raise_and_reorient.py")
hohmann = load_module("missions/hohmann_transfer.py")
pid = load_module("attitude/pid_attitude_control.py")

INERTIA = (50.0, 70.0, 90.0)


def test_orbit_phase_delta_v_matches_hohmann_module_directly():
  """run_orbit_raise_phase의 델타-V가 hohmann_transfer.py를 직접 호출한 값과
  정확히 같아야 한다(새 공식이 아니라 순수 재사용임을 증명)."""
  result = m.run_orbit_raise_phase(7000.0, 8000.0)
  _dv1, _dv2, expected_total = hohmann.hohmann_transfer_delta_v(7000.0, 8000.0)
  expected_time = hohmann.hohmann_transfer_time(7000.0, 8000.0)
  assert result["total_delta_v_km_s"] == pytest.approx(expected_total)
  assert result["transfer_time_sec"] == pytest.approx(expected_time)


def test_attitude_phase_uses_pid_module_gains_directly():
  """자세 재정렬 단계의 게인이 pid_attitude_control.compute_pd_gains를 직접
  호출한 값과 같아야 한다."""
  initial_q = pid.axis_angle_to_quaternion([1, 0, 0], 150.0)
  result = m.run_attitude_reorientation_phase(initial_q, INERTIA, settling_time_sec=10.0)
  expected_kp, expected_kd = pid.compute_pd_gains(INERTIA, 10.0)
  assert np.allclose(result["kp"], expected_kp)
  assert np.allclose(result["kd"], expected_kd)


def test_attitude_phase_starts_from_orbit_phase_specified_tumble():
  """자세 재정렬 단계의 첫 쿼터니언이 궤도 전이 완료 시점에 지정한 바로 그
  임의 자세여야 한다 - 상태가 실제로 단계 사이를 이어받는지 확인(통합의
  핵심 주장)."""
  initial_q = pid.axis_angle_to_quaternion([1, 0, 0], 150.0)
  mission = m.build_mission_timeline(7000.0, 8000.0, initial_q, INERTIA, 10.0)
  first_q = mission["attitude_phase"]["history"][0][1][0]
  assert np.allclose(first_q, initial_q)


def test_total_mission_time_equals_sum_of_both_phases():
  """총 임무시간 = 궤도 전이시간 + 자세 재정렬 시뮬레이션 시간(핵심 통합
  불변식)."""
  initial_q = pid.axis_angle_to_quaternion([1, 0, 0], 150.0)
  mission = m.build_mission_timeline(7000.0, 8000.0, initial_q, INERTIA, 10.0)
  expected = (mission["orbit_phase"]["transfer_time_sec"]
              + mission["attitude_phase"]["total_time_sec"])
  assert mission["total_mission_time_sec"] == pytest.approx(expected)


def test_attitude_phase_delta_v_contribution_is_zero():
  """이 스크립트의 제어모델(순수 토크, 반작용휠 디새추레이션 델타-V 없음)에서는
  자세 재정렬이 궤도 전이 델타-V에 전혀 기여하지 않아야 한다(두 단계가 물리적
  으로 독립적인 자원을 소모한다는 명시적 확인 - 단위 혼동 방지)."""
  initial_q = pid.axis_angle_to_quaternion([1, 0, 0], 150.0)
  mission = m.build_mission_timeline(7000.0, 8000.0, initial_q, INERTIA, 10.0)
  assert mission["total_delta_v_km_s"] == pytest.approx(mission["orbit_phase"]["total_delta_v_km_s"])


def test_time_to_settle_below_threshold_finds_crossing_point():
  """time_to_settle_below_threshold가 오차가 실제로 임계값 아래로 내려가는
  시각을 정확히 찾아야 한다."""
  error_angles = [(0.0, 10.0), (1.0, 5.0), (2.0, 1.5), (3.0, 0.5)]
  assert m.time_to_settle_below_threshold(error_angles, threshold_deg=2.0) == 2.0


def test_time_to_settle_below_threshold_returns_none_if_never_settles():
  """오차가 끝까지 임계값 아래로 내려가지 못하면 None을 반환해야 한다."""
  error_angles = [(0.0, 10.0), (1.0, 8.0), (2.0, 6.0)]
  assert m.time_to_settle_below_threshold(error_angles, threshold_deg=2.0) is None


def test_severe_tumble_takes_longer_to_settle_than_mild_tumble():
  """같은 정착시간 설정에서, 더 심한 초기 텀블(170도)은 완만한 텀블(30도)보다
  목표 오차 아래로 내려가는 데 더 오래 걸려야 한다(21번 point-and-hold 제어
  법칙의 선형성에서 기대되는 물리적 결과)."""
  mild_q = pid.axis_angle_to_quaternion([1, 0, 0], 30.0)
  severe_q = pid.axis_angle_to_quaternion([1, 0, 0], 170.0)
  mild = m.build_mission_timeline(7000.0, 8000.0, mild_q, INERTIA, 10.0)
  severe = m.build_mission_timeline(7000.0, 8000.0, severe_q, INERTIA, 10.0)
  assert severe["attitude_phase"]["time_to_settle_sec"] > mild["attitude_phase"]["time_to_settle_sec"]
