# 01~11번 시뮬레이션을 순서대로 한 번에 실행하고, 마지막에 report.html까지 재생성하는 스크립트
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
# 번호 있는 단계(1/13~)에 넣지 않은 이유: 이건 "시뮬레이션을 실행"하는 게 아니라 그
# 전에 실행 환경을 정리하는 준비 작업이라, 파이프라인 진행률 표시에 포함시키지 않는다.
Write-Output "[정리] results/의 이전 실행 잔재물을 비웁니다."
if (Test-Path "results") { Remove-Item "results" -Recurse -Force }

Write-Section "1/13 자동 테스트 (pytest) - 코드가 깨진 상태로 시뮬레이션을 돌리지 않기 위한 사전 검증"
Invoke-Step "pytest" { python -m pytest tests/ -q }

Write-Section "2/13 케플러 방정식과 궤도 전파"
Invoke-Step "01_Kepler_orbit_propagation.py" { python 01_Kepler_orbit_propagation.py }

Write-Section "3/13 궤도요소-상태벡터 변환과 보존량(에너지/각운동량)"
Invoke-Step "02_Orbital_elements_and_energy.py" { python 02_Orbital_elements_and_energy.py }

Write-Section "4/13 2체 문제 수치적분(RK4)과 해석해 검증"
Invoke-Step "03_Two_body_numerical_integration.py" { python 03_Two_body_numerical_integration.py }

Write-Section "5/13 좌표계 변환 (ECI/ECEF/SEZ)과 방위각/고도각"
Invoke-Step "04_Coordinate_frame_transforms.py" { python 04_Coordinate_frame_transforms.py }

Write-Section "6/13 지상국 가시성: 실측 궤도 전파로 계산하는 접촉 창"
Invoke-Step "05_Ground_station_visibility.py" { python 05_Ground_station_visibility.py }

Write-Section "7/13 호만 전이(Hohmann Transfer) 델타-V와 전이시간"
Invoke-Step "06_Hohmann_transfer.py" { python 06_Hohmann_transfer.py }

Write-Section "8/13 J2 섭동: RAAN/근점편각 세차"
Invoke-Step "07_J2_perturbation.py" { python 07_J2_perturbation.py }

Write-Section "9/13 란베르트 문제: 두 위치-비행시간으로 궤도 속도 계산"
Invoke-Step "09_Lambert_problem.py" { python 09_Lambert_problem.py }

Write-Section "10/13 다중 위성 성좌 커버리지: 워커 델타 패턴과 재방문 공백"
Invoke-Step "10_Constellation_coverage.py" { python 10_Constellation_coverage.py }

Write-Section "11/13 위성간 링크(ISL) 가시선: 지구 차단 기하 판정"
Invoke-Step "11_Intersatellite_link_visibility.py" { python 11_Intersatellite_link_visibility.py }

Write-Section "12/13 결과 시각화 (results/*.png 생성)"
Invoke-Step "08_visualize_orbits.py" { python 08_visualize_orbits.py }

Write-Section "13/13 프로젝트 요약 리포트 생성 (report.html)"
Invoke-Step "generate_report.py" { python generate_report.py }

Write-Section "전체 시뮬레이션 실행 완료"
