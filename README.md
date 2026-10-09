# Little White v3 · MuJoCo sim2sim

在 standalone MuJoCo 中播放小白 v3 的 WTW、DreamWaQ 或 MJLab 策略，支援鍵盤控制、固定命令與無視窗模擬。

模型、推論程式與控制設定都在本專案，不需要 mjlab、Isaac Gym 或相鄰訓練專案。策略權重須另外取得。

## 安裝

使用 Python 3.12 與 `uv`，在專案根目錄執行：

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt --torch-backend cpu
```

互動式播放需要圖形桌面。macOS 的 native viewer 使用 `.venv/bin/mjpython`；無視窗模擬使用 `--headless`。

## 策略權重

Git 不包含策略 export。將自行匯出的權重放到 `policies/`，或用 `--policy` 指定檔案。WTW 與 DreamWaQ 的 `policy.pt` 必須搭配同一次 export 的 `metadata.json`，放在同一資料夾。

| 設定檔 | 預設策略路徑 | 用途 |
| --- | --- | --- |
| `config/little_white_v3_wtw_original.yaml` | `policies/wtw/policy.pt` | WTW，原始小白模型 |
| `config/little_white_v3_wtw.yaml` | `policies/wtw/policy.pt` | WTW，訓練轉換模型 |
| `config/little_white_v3_wtw_extended.yaml` | `policies/wtw_extended/policy.pt` | WTW，擴展速度與身高控制 |
| `config/little_white_v3_dreamwaq.yaml` | `policies/dreamwaq/policy.pt` | DreamWaQ |
| `config/little_white_v3_mjlab.yaml` | `policies/mjlab/policy.pt` | MJLab |

不同策略的觀測與控制契約不同，請搭配對應設定及 metadata。啟動時會檢查 WTW 策略／模型 hash 與控制參數。

## WTW 播放

使用原始小白模型與鍵盤控制：

```bash
MUJOCO_GL=glfw .venv/bin/python mujoco_mjlab.py \
  config/little_white_v3_wtw_original.yaml

# 指定自己的 export
MUJOCO_GL=glfw .venv/bin/python mujoco_mjlab.py \
  config/little_white_v3_wtw_original.yaml --policy policies/my_policy/policy.pt
```

點選鍵盤控制視窗後操作：

| 按鍵 | 控制 |
| --- | --- |
| W / S | 前進／後退 |
| A / D | 左／右側移 |
| Q / E | 左／右轉 |
| 1 / 2 / 3 / 4 | trot／pace／bound／pronk |
| R / F | 增加／降低步頻 |
| T / G | 增加／降低身高偏移 |
| Y / H | 增加／降低抬腳高度 |
| U / J、I / K | pitch、roll |
| O / L、P / ; | 站距寬、長 |
| ] / [ | stance duty factor |
| 0 | 恢復設定檔的步態與身體命令 |
| Space | 移動命令歸零 |

移動鍵放開後速度逐漸歸零，步態與身體命令保留；關閉 MuJoCo 視窗或按 Ctrl+C 結束。

固定命令或無視窗執行：

```bash
.venv/bin/python mujoco_mjlab.py config/little_white_v3_wtw_original.yaml \
  --gait trot --command 0.3 0 0 --height 0.02 --swing-height 0.06 --duration 60

.venv/bin/python mujoco_mjlab.py config/little_white_v3_wtw_original.yaml \
  --headless --command 0.3 0 0 --duration 20 --report output/result.json
```

`--command VX VY YAW` 的速度單位為 m/s、yaw rate 為 rad/s；身高與抬腳高度為 m。固定命令及 `--switches` 自動步態切換會停用鍵盤。其他選項可用 `--help` 查看。

## 模型與控制範圍

WTW 可使用 `--model-profile original` 或 `training`。`original` 保留固定版本原始小白模型的物理設定；`training` 使用 WTW 訓練轉換模型。兩者沿用相同策略控制契約，physics dt 為 0.005 s、控制頻率為 50 Hz。

`show_grid` 控制訓練模型的地面格線顯示，不更改物理參數。

擴展設定提供前進 0.6、後退 0.3、側移 0.25 m/s、yaw 0.75 rad/s，以及身高偏移 −0.08 到 +0.04 m，需搭配已訓練及驗證該範圍的策略：

```bash
MUJOCO_GL=glfw .venv/bin/python mujoco_mjlab.py \
  config/little_white_v3_wtw_extended.yaml --policy policies/my_policy/policy.pt
```

鍵盤速度在 `keyboard` 設定；倒退速度可用 `backward_scale` 個別指定，身高範圍可用 `wtw.control_limits.height: [MIN, MAX]` 指定。

## DreamWaQ 與 MJLab

```bash
.venv/bin/python mujoco_mjlab.py config/little_white_v3_dreamwaq.yaml
.venv/bin/python mujoco_mjlab.py config/little_white_v3_mjlab.yaml
```

使用 W/S、A/D、Q/E 控制速度，也支援 `--policy`、`--command` 與 `--headless`。DreamWaQ 使用六幀觀測歷史，MJLab 使用單幀觀測；兩者的策略格式不可混用。

## 檔案與版本管理

```text
mujoco_mjlab.py         模擬、策略推論與 PD 控制
keyboard_controller.py 鍵盤控制
wtw_commands.py        WTW 命令與步態
wtw_runtime.py         WTW 觀測、history 與契約驗證
config/                播放設定
assets/                原始／訓練模型、mesh 與來源 hash
policies/              本機策略 export（不納入 Git）
third_party/           上游授權與來源說明
requirements.txt       推論依賴版本
```

Git 保留程式、設定、模型與上游授權；權重及 metadata、虛擬環境、模擬輸出、影片與快取由 `.gitignore` 排除。換機時另外傳送完整策略 export。

模型與 WTW 移植來源及授權見 [third_party/wtw/NOTICE.md](third_party/wtw/NOTICE.md)。
