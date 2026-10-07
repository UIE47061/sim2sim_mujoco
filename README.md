# Little White v3：standalone MuJoCo sim2sim

在 MuJoCo 中播放小白 v3 的 **WTW、DreamWaQ 與 MJLab** TorchScript 策略，提供鍵盤控制與固定命令，場景由各設定的 XML 載入。推論只需本專案、模型與匯出權重，無需 mjlab、Isaac Gym 或相鄰訓練專案。

## 安裝

本機驗證環境：Python 3.12、MuJoCo 3.2.7、CPU PyTorch 2.14.1。版本列於 `requirements.txt`；以下使用 uv 建立環境：

```bash
cd sim2sim_mujoco
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt --torch-backend cpu
```

已有本機 `.venv` 可直接使用。Linux GUI 需要可用桌面；macOS native viewer 請將播放指令的 `.venv/bin/python` 換成 `.venv/bin/mjpython`。無視窗執行使用 `--headless`。

## 權重與 Git

Git 保留程式、設定、XML／mesh、metadata 及上游授權；`.venv/`、權重、測試、`docs/`、影片與模擬輸出由 `.gitignore` 排除。**新 clone 必須另外取得 policy.pt**，放入下列路徑；WTW／DreamWaQ 權重必須與相同 export 的 `metadata.json` 配對。

| 設定檔（皆在 `config/`） | 權重路徑 | 用途 |
| --- | --- | --- |
| `little_white_v3_wtw.yaml` | `pre_train_mjlab/wtw_seed0_best/policy.pt` | WTW 預設最佳策略，update 5500 |
| 同上，使用 `--policy` 替換 | `pre_train_mjlab/wtw_seed1_best/policy.pt` | WTW seed 1，update 4000 |
| 同上，使用 `--policy` 替換 | `pre_train_mjlab/wtw_seed2_best/policy.pt` | WTW seed 2，update 2000 |
| `little_white_v3_dreamwaq.yaml` | `pre_train_mjlab/dreamwaq_rough_best/policy.pt` | DreamWaQ 混合地形最佳策略 |
| `little_white_v3_mjlab.yaml` | `pre_train_mjlab/blind_stairs_19998/policy.pt` | 原有 MJLab 策略 |

本機上述五個權重均保留原位置，可直接播放。先前整理時的權重、舊輸出與測試已備份至 `../sim2sim_mujoco_artifacts/20261007_cleanup/`；備份的 SHA-256 已核對。

在同一工作站還原權重的範例：

```bash
cp ../sim2sim_mujoco_artifacts/20261007_cleanup/pre_train_mjlab/wtw_seed0_best/policy.pt \
  pre_train_mjlab/wtw_seed0_best/policy.pt
```

此歸檔路徑僅適用本機，不是執行依賴。其他機器請另外傳輸完整 export。原先已提交的權重已從 Git 索引移除，既有提交歷史仍保留它們。

## WTW 播放

預設 seed 0，播放 300 秒，顯示棋盤格地面與鍵盤視窗：

```bash
MUJOCO_GL=glfw .venv/bin/python mujoco_mjlab.py config/little_white_v3_wtw.yaml
```

點選 **WTW Keyboard Control** 視窗操作。放開移動鍵後速度逐漸歸零，步態及身體參數保持最後設定。

| 按鍵 | 命令／範圍 |
| --- | --- |
| W / S | 前進／後退，±0.3 m/s |
| A / D | 左／右側移，±0.15 m/s |
| Q / E | 左／右轉，±0.5 rad/s |
| 1 / 2 / 3 / 4 | trot（對角）／pace（同側）／bound（前後腳對）／pronk（四腳同步） |
| R / F | 增加／減少步頻，1.8–3.2 Hz |
| T / G | 增加／減少身高偏移，±0.04 m |
| Y / H | 增加／減少抬腳高度，0.03–0.09 m |
| U / J | 增加／減少 pitch，±0.15 rad |
| I / K | 增加／減少 roll，±0.15 rad |
| O / L | 增加／減少站距寬，nominal ±0.025 m |
| P / ; | 增加／減少站距長，nominal ±0.025 m |
| ] / [ | 增加／減少 duty factor，0.4–0.6 |
| 0 | 恢復設定檔的步態與身體命令 |
| Space | 移動命令歸零，依 smoothing 逐漸停下 |
| Esc | 關閉鍵盤視窗，速度逐漸歸零 |

切換步態或重設命令保留連續 phase 與 history。關閉 MuJoCo 視窗或按 Ctrl+C 結束模擬。

固定命令、四步態切換、替換 seed：

```bash
.venv/bin/python mujoco_mjlab.py config/little_white_v3_wtw.yaml \
  --gait pace --frequency 2.5 --command 0.3 0 0 --duration 60

.venv/bin/python mujoco_mjlab.py config/little_white_v3_wtw.yaml \
  --headless --switches --command 0.3 0 0 --duration 20 \
  --report output/wtw_switches.json

.venv/bin/python mujoco_mjlab.py config/little_white_v3_wtw.yaml \
  --policy pre_train_mjlab/wtw_seed1_best/policy.pt
```

另支援 `--height`、`--swing-height`、`--pitch`、`--roll`、`--width`、`--length`、`--duty`。`--command VX VY YAW_RATE` 使用機身 yaw 座標，單位 m/s、m/s、rad/s；正 vy 向左、正 yaw-rate 左轉。固定命令與自動切換會停用鍵盤。報告預設 `output/wtw_report.json`。

WTW student 輸入 `[1,2100]`：70 維觀測、30 幀 history，由舊到新排列，初始零 padding 加當前觀測。history → adaptation → actor 已內嵌，推論不讀 privileged 值。關節順序 FR／FL／RL／RR，每腿 hip／thigh／calf。

控制：physics dt 0.005 s、decimation 4（50 Hz），nominal `[0,0.95,-1.7]`、Kp 20、Kd 0.5、action scale 0.25。action 裁切 ±4，每個 physics step 重算 PD 並裁切扭矩 ±27 Nm。啟動時核對策略 SHA-256、觀測／控制／步態版本、資產轉換版本及模型 hash。

WTW 固定模型位於 `assets/little_white_v3_wtw/`，來源與 hash 見 `manifest.json`，授權見 `third_party/wtw/`。`show_grid: true` 在模型驗證後於記憶體加入棋盤格材質，只改視覺。模型原檔與接觸參數保持相同；`false` 使用訓練場景的純色地面。

三個 WTW seed 各訓練至 10,000 updates，使用最佳 checkpoint 匯出。訓練專案已完成平地四步態及切換驗收；抬腳命令的實際響應偏弱，±5° 坡度速度追蹤未達標，adaptation 尚未證明有整體收益。本專案 MuJoCo 3.2.7 的單一初始狀態播放檢查，不等同訓練專案 MuJoCo 3.11.0 的完整 20-seed 評估。

## DreamWaQ 與 MJLab

```bash
# DreamWaQ 混合地形策略，鍵盤控制。
.venv/bin/python mujoco_mjlab.py config/little_white_v3_dreamwaq.yaml

# 原有 MJLab 策略。
.venv/bin/python mujoco_mjlab.py config/little_white_v3_mjlab.yaml

# 無視窗平地檢查。
.venv/bin/python mujoco_mjlab.py config/little_white_v3_dreamwaq.yaml \
  --headless --command 0.3 0 0 --duration 20
```

兩者使用 W/S 前後、A/D 側移、Q/E 轉向；DreamWaQ rough 的速度範圍為 ±0.6／±0.4 m/s、yaw ±0.8 rad/s。DreamWaQ 使用六幀 45 維觀測，MJLab 使用單幀 48 維觀測。策略格式及控制設定不可混用。

DreamWaQ 使用 `assets/little_white_v3/scene.xml`，載入時套用訓練接觸設定；WTW 使用自己的固定平地模型。報告記錄模擬時長、停止原因、位移、最低基座高度與最大扭矩。

## 檔案結構

```text
mujoco_mjlab.py         模擬、PD、策略推論與報告
keyboard_controller.py 鍵盤視窗
wtw_commands.py        WTW 命令與步態
wtw_runtime.py         WTW 觀測、history、模型／版本驗證
config/                三個播放設定
assets/                兩套模型與相對路徑 mesh
pre_train_mjlab/        metadata、來源說明與本機權重
third_party/           上游授權與來源
requirements.txt       已驗證的直接依賴版本
```

精簡 repo 不包含測試與舊輸出。先前的回歸測試已歸檔；目前保留的三種策略會以載入、推論與物理步進檢查驗證。
