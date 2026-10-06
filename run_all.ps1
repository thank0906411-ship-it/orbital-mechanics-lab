# 시뮬레이션 스크립트(propagation/frames/perturbations/constellations/missions/
# attitude/visualization 하위 폴더)를 순서대로 한 번에 실행하고, 마지막에 report.html까지 재생성하는 스크립트
# 실행: powershell -ExecutionPolicy Bypass -File .\run_all.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Write-Section($title) {
    Write-Output ""
    Write-Output ("=" * 70)
    Write-Output $title
    Write-Output ("=" * 70)
}

# $ErrorActionPreference = "Stop"은 PowerShell 종료 오류만 잡을 뿐, python.exe 같은 네이티브
# 실행 파일의 비정상 종료 코드는 잡지 못한다. 그래서 매 단계마다 $LASTEXITCODE를 직접 확인해
# 실패 시 즉시 중단시킨다 — 안 그러면 중간 단계가 죽어도 스크립트가 끝까지 돌고 "완료" 메시지가
# 뜨는 조용한 실패(silent failure)가 발생한다.
function Invoke-Step($description, $scriptBlock) {
    & $scriptBlock
    if ($LASTEXITCODE -ne 0) {
        Write-Output "`n[중단] $description 단계가 종료 코드 $LASTEXITCODE 로 실패했습니다. 파이프라인을 멈춥니다."
        exit $LASTEXITCODE
    }
}

# results/를 매 실행 전에 비운다 — 안 그러면 예전에 스크립트를 리팩터링/삭제하면서
# 더 이상 어떤 스크립트도 만들지 않는 CSV/PNG가 폴더에 계속 남아 최신 결과와 뒤섞인다.
# 번호 있는 단계(1/29~)에 넣지 않은 이유: 이건 "시뮬레이션을 실행"하는 게 아니라 그
# 전에 실행 환경을 정리하는 준비 작업이라, 파이프라인 진행률 표시에 포함시키지 않는다.
Write-Output "[정리] results/의 이전 실행 잔재물을 비웁니다."
if (Test-Path "results") { Remove-Item "results" -Recurse -Force }

Write-Section "1/29 자동 테스트 (pytest) - 코드가 깨진 상태로 시뮬레이션을 돌리지 않기 위한 사전 검증"
Invoke-Step "pytest" { python -m pytest tests/ -q }

Write-Section "2/29 케플러 방정식과 궤도 전파"
Invoke-Step "propagation/kepler_orbit_propagation.py" { python propagation/kepler_orbit_propagation.py }

Write-Section "3/29 궤도요소-상태벡터 변환과 보존량(에너지/각운동량)"
Invoke-Step "propagation/orbital_elements_and_energy.py" { python propagation/orbital_elements_and_energy.py }

Write-Section "4/29 2체 문제 수치적분(RK4)과 해석해 검증"
Invoke-Step "propagation/two_body_numerical_integration.py" { python propagation/two_body_numerical_integration.py }

Write-Section "5/29 좌표계 변환 (ECI/ECEF/SEZ)과 방위각/고도각"
Invoke-Step "frames/coordinate_frame_transforms.py" { python frames/coordinate_frame_transforms.py }

Write-Section "6/29 지상국 가시성: 실측 궤도 전파로 계산하는 접촉 창"
Invoke-Step "missions/ground_station_visibility.py" { python missions/ground_station_visibility.py }

Write-Section "7/29 호만 전이(Hohmann Transfer) 델타-V와 전이시간"
Invoke-Step "missions/hohmann_transfer.py" { python missions/hohmann_transfer.py }

Write-Section "8/29 저추력 전기추진 궤도 전이: 연속 접선 추력에 의한 나선 상승"
Invoke-Step "missions/low_thrust_transfer.py" { python missions/low_thrust_transfer.py }

Write-Section "9/29 궤도 재진입/대기항력 감쇠: 15번의 거울상, 나선 하강"
Invoke-Step "missions/orbital_decay.py" { python missions/orbital_decay.py }

Write-Section "10/29 J2 섭동: RAAN/근점편각 세차"
Invoke-Step "perturbations/j2_perturbation.py" { python perturbations/j2_perturbation.py }

Write-Section "11/29 궤도 유지 비용: 잔여 RAAN 오차를 주기적 기동으로 보정"
Invoke-Step "missions/station_keeping.py" { python missions/station_keeping.py }

Write-Section "12/29 란베르트 문제: 두 위치-비행시간으로 궤도 속도 계산"
Invoke-Step "missions/lambert_problem.py" { python missions/lambert_problem.py }

Write-Section "13/29 다중 위성 성좌 커버리지: 워커 델타 패턴과 재방문 공백"
Invoke-Step "constellations/constellation_coverage.py" { python constellations/constellation_coverage.py }

Write-Section "14/29 위성간 링크(ISL) 가시선: 지구 차단 기하 판정"
Invoke-Step "constellations/intersatellite_link_visibility.py" { python constellations/intersatellite_link_visibility.py }

Write-Section "15/29 행성간 궤적: Patched Conic 근사로 지구-화성 임무 델타-V"
Invoke-Step "missions/patched_conic_interplanetary.py" { python missions/patched_conic_interplanetary.py }

Write-Section "16/29 3체 문제와 라그랑주점: 원형 제한 3체 문제(CR3BP)로 L1~L5 찾기"
Invoke-Step "missions/lagrange_points.py" { python missions/lagrange_points.py }

Write-Section "17/29 평면 리아푸노프 궤도: 선형 추정을 미분수정으로 완성한 주기궤도"
Invoke-Step "missions/lyapunov_orbits.py" { python missions/lyapunov_orbits.py }

Write-Section "18/29 도킹/랑데부: Clohessy-Wiltshire 근접 상대운동"
Invoke-Step "missions/clohessy_wiltshire_rendezvous.py" { python missions/clohessy_wiltshire_rendezvous.py }

Write-Section "19/29 궤도 결정: 노이즈 낀 레이더 관측값에서 궤도요소 역산"
Invoke-Step "missions/orbit_determination.py" { python missions/orbit_determination.py }

Write-Section "20/29 자세동역학: 토크 없는 강체 자유회전, 중간축 정리"
Invoke-Step "attitude/torque_free_rigid_body.py" { python attitude/torque_free_rigid_body.py }

Write-Section "21/29 PID 자세제어: 쿼터니언 오차 기반 PD 피드백으로 목표 자세 수렴"
Invoke-Step "attitude/pid_attitude_control.py" { python attitude/pid_attitude_control.py }

Write-Section "22/29 궤도+자세 통합 임무: 06번 호만 전이 + 21번 PID 자세제어를 하나의 타임라인으로"
Invoke-Step "missions/orbit_raise_and_reorient.py" { python missions/orbit_raise_and_reorient.py }

Write-Section "23/29 3체(태양/달) 섭동: 17번이 모델링하지 않는다고 밝힌 요인을 실제로 계산"
Invoke-Step "perturbations/third_body_perturbation.py" { python perturbations/third_body_perturbation.py }

Write-Section "24/29 태양복사압(SRP) 섭동: 9번(항력)과 거울상, 고고도에서 남는 비중력 섭동"
Invoke-Step "perturbations/solar_radiation_pressure.py" { python perturbations/solar_radiation_pressure.py }

Write-Section "25/29 반작용휠 모멘텀 저장과 디새추레이션: 22번이 모델링하지 않는다고 밝힌 델타-V를 실제로 계산"
Invoke-Step "attitude/reaction_wheel_desaturation.py" { python attitude/reaction_wheel_desaturation.py }

Write-Section "26/29 자세 의존 태양복사압: 24번의 캐논볼 모델을 넘어 21번의 자세를 결합"
Invoke-Step "perturbations/attitude_dependent_srp.py" { python perturbations/attitude_dependent_srp.py }

Write-Section "27/29 가우스법 궤도결정: 14번이 범위 밖에 둔 일반 타원궤도 결정"
Invoke-Step "missions/gauss_orbit_determination.py" { python missions/gauss_orbit_determination.py }

Write-Section "28/29 결과 시각화 (results/*.png 생성)"
Invoke-Step "visualization/visualize_orbits.py" { python visualization/visualize_orbits.py }

Write-Section "29/29 프로젝트 요약 리포트 생성 (report.html)"
Invoke-Step "generate_report.py" { python generate_report.py }

Write-Section "전체 시뮬레이션 실행 완료"
