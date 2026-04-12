@echo off
:: 1. Scripts 폴더로 이동 (경로에 공백이 있으므로 따옴표 필수)
cd /d "E:\Downloads\hitomi downloader\BallonsTranslator-latest\Scripts"

:: 2. 가상 환경 활성화 (call 명령어를 써야 가상환경 실행 후 다음 줄로 넘어갑니다)
call activate.bat

:: 3. 상위 폴더(BallonsTranslator-latest)로 이동
cd ..

:: 4. launch.py 실행
python launch.py

:: 5. 프로그램이 종료되거나 에러가 났을 때 창이 바로 꺼지지 않게 대기
pause