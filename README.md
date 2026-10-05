# Little White V3：MJLab → MuJoCo sim2sim

支援 **MJLab / DreamWaQ → MuJoCo**：載入匯出的 TorchScript 策略，透過 PD 控制器驅動 Little White V3 的 12 個關節。兩種策略共用執行入口 `mujoco_mjlab.py`，使用各自的設定與機器人 XML。

## DreamWaQ 混合地形最佳模型

已放入 `dreamwaq_mjlab/logs/2026-10-04_22-52-48_rough_round/best.pt` 匯出的策略，來自第 9,000 次更新。

```bash
cd /home/uie/U1_ws/sim2sim_mujoco
.venv/bin/python mujoco_mjlab.py config/little_white_v3_dreamwaq.yaml
```

會開啟 MuJoCo 視窗及 Pygame 鍵盤控制視窗；點選鍵盤控制視窗後使用 W/S 前後移動（±0.6 m/s）、A/D 左右側移（±0.4 m/s）、Q/E 左右旋轉（±0.8 rad/s）。放開按鍵會逐漸歸零。預設 60 秒，可加 `--duration 300`。

固定指令觀看與無視窗檢查：

```bash
.venv/bin/python mujoco_mjlab.py config/little_white_v3_dreamwaq.yaml \
  --duration 60 --command 0.4 0 0

.venv/bin/python mujoco_mjlab.py config/little_white_v3_dreamwaq.yaml \
  --headless --duration 20 --command 0 0.4 0
```

`--command VX VY YAW_RATE` 使用機身 yaw 座標（m/s、m/s、rad/s），指定後停用鍵盤；正 vy 向左、正 yaw-rate 左轉。預設報告為 `output/dreamwaq_report.json`。

DreamWaQ 使用六幀 45 維觀測 `[batch, 6, 45] → [batch, 12]`。CENet 估測速度、actor 與正規化均已內嵌；不將模擬器量到的線速度送進策略。`pre_train_mjlab/dreamwaq_rough_best/policy.pt` 與 `metadata.json` 需放在同一資料夾，啟動時會核對指令座標、觀測、關節順序及控制參數。

兩種策略統一使用 `assets/little_white_v3/scene.xml`。DreamWaQ 模式會在載入的模型上套用訓練接觸參數：零碰撞 margin、腳底三維接觸／摩擦係數 1，以及機器人碰撞遮罩。直接使用舊 XML 的腳底摩擦與接觸設定，可能出現有前進指令卻在原地踏步；不需另建 scene 或更換機器人 XML。

更新模型時，可重新匯出至本專案：

```bash
cd /home/uie/U1_ws/dreamwaq_mjlab
uv run --extra train dreamwaq-export \
  logs/2026-10-04_22-52-48_rough_round/best.pt \
  --output /home/uie/U1_ws/sim2sim_mujoco/pre_train_mjlab/dreamwaq_rough_best
```

這個 standalone 依照 `scene.xml` 載入地形，可使用其中的箱體與樓梯並透過鍵盤控制。訓練的九種混合地形及難度測試使用 `dreamwaq-play`；不會自動在此場景生成訓練地圖。現有模型在本 scene 啟用的樓梯前仍可能停滯：第一階高 17 cm，後續升高 15 cm；箱體重疊後，中間階面的有效深度約 19 cm，而訓練樓梯使用 28～34 cm。這次 0.4／0.6 m/s 固定指令均出現停滯，0.8 m/s 則跌倒，因此平地驗證通過不代表已能通過這組樓梯。

## 原有 MJLab 策略：安裝與執行

已有 `.venv` 可直接執行：

```bash
cd /home/uie/U1_ws/sim2sim_mujoco
.venv/bin/python mujoco_mjlab.py config/little_white_v3_mjlab.yaml
```

新環境需先安裝依賴：

```bash
python3 -m venv .venv
.venv/bin/pip install mujoco torch numpy pygame pyyaml
```

## 操作

點選 Pygame 鍵盤控制視窗後操作：

| 按鍵 | 動作 |
| --- | --- |
| W / S | 前進 / 後退 |
| A / D | 左右平移 |
| Q / E | 左右旋轉 |

預設執行 60 秒，可用 `--duration 300` 調整。關閉 MuJoCo 視窗或按 Ctrl+C 結束；傾斜超過 70 度時停止。

無視窗測試：

```bash
.venv/bin/python mujoco_mjlab.py config/little_white_v3_mjlab.yaml \
  --headless --duration 10 --command 0.3 0 0
```

`--command` 依序是前後速度、左右速度（m/s）及旋轉速度（rad/s），指定後停用鍵盤控制。

## 主要檔案

- `mujoco_mjlab.py`：策略推論與模擬控制。
- `keyboard_controller.py`：鍵盤控制視窗。
- `config/little_white_v3_mjlab.yaml`：路徑、站姿、PD 與關節順序等參數。
- `assets/little_white_v3/`：機器人模型、場景與 mesh。
- `pre_train_mjlab/flat_9999/policy.pt`：匯出的策略，已包含觀測正規化；部署只需這個權重檔。

原有 MJLab 模式使用單幀 48 維觀測、12 維動作；DreamWaQ 模式使用六幀 45 維觀測。兩者策略更新皆為 50 Hz、物理模擬 200 Hz。更換策略時需確認觀測格式與關節順序符合訓練設定。

執行報告預設儲存在 `output/mjlab_report.json`。
