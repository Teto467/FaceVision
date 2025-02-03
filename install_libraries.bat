whoami /priv | find "SeDebugPrivilege" > nul
if %errorlevel% neq 0 (
 @powershell start-process %~0 -verb runas
 exit
)
chcp 65001
@echo off
echo ライブラリのインストールを開始します...

REM Pythonがインストールされていることを確認
python --version 2>NUL
if errorlevel 1 (
    echo Pythonがインストールされていません。Pythonをインストールしてから再度実行してください。
    pause
    exit
)

REM 必要なライブラリをインストール
pip install opencv-python
pip install numpy
pip install deepface
pip install tensorflow
pip install tf-keras

echo インストールが完了しました。
pause
