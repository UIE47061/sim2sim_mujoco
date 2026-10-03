# Little White V3：MJLab → MuJoCo sim2sim

目前的執行流程為 **MJLab → MuJoCo**：載入 MJLab 匯出的 TorchScript 平地行走策略，透過 PD 控制器驅動 Little White V3 的 12 個關節。執行入口為 `mujoco_mjlab.py`。

舊版 DWAQ 程式已移除，以下指令與設定皆適用於目前的 MJLab 流程。

## 安裝與執行

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

目前使用 48 維觀測、12 維動作，策略更新 50 Hz、物理模擬 200 Hz。更換策略時需確認觀測格式與關節順序符合訓練設定。

執行報告預設儲存在 `output/mjlab_report.json`。
