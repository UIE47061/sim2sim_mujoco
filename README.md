# MuJoCo Sim2Sim 四足機器人 (小白)

此儲存庫包含使用 MuJoCo 在模擬四足機器人上執行已訓練的深度強化學習 (DRL) 策略的部署程式碼。它被設計作為一個「Sim2Sim」的驗證橋樑——在部署到實際硬體之前，於 MuJoCo 這種高精度的物理模擬器中測試通常在 Isaac Gym 或 Isaac Lab 訓練好的策略。

## 功能特色
- **支援 DWAQ 策略**：整合了 `ActorCritic_DWAQ`，透過觀察歷史緩衝區來評估上下文編碼器 (V-VAE)。
- **多進程鍵盤控制器**：透過在獨立的衍生進程上執行 `pygame` 介面來發布使用者移動指令 (`W/A/S/D/Q/E`)，以避開 macOS 的 UI 阻塞問題，同時保持 MuJoCo 渲染執行緒暢通。
- **自動對應 (Auto-Mapping)**：開箱即用，將 Isaac Gym (策略定義) 輸出的關節序列與任意 URDF/XML 馬達描述進行無縫對應，解決順序不一致的問題。

## 專案架構
```text
.
├── config/                  # YAML 設定檔 (例如 little_white.yaml)
├── meshes/                  # 視覺與碰撞網格 (Mesh) 幾何檔
├── pre_train/               # PyTorch 模型權重檔目錄 (例如 policy.pt)
├── urdf/                    # 原始 URDF 描述檔
├── xml/                     # 編譯後的 MuJoCo .xml 場景與機器人描述檔
├── keyboard_controller.py   # 用於監聽 WASD/QE 輸入的多進程控制器
└── mujoco_dwaq.py           # 核心的主評估迴圈
```

## 依賴套件
- `mujoco`
- `torch`
- `numpy`
- `pygame`
- `pyyaml`
- `matplotlib`

*注意：在 macOS 上，執行 `mujoco.viewer` 需要使用特定的別名執行檔 `mjpython`，而不是標準的 `python`。*

## 執行方式
透過 YAML 設定檔執行策略，例如 `little_white.yaml` 的配置：

```bash
# macOS 使用者「必須」使用 mjpython 而不是標準的 python
mjpython mujoco_dwaq.py config/little_white.yaml
```

**控制方式**：
- `W / S`：前進 / 後退 (vx)
- `A / D`：向左移動 / 向右移動 (vy)
- `Q / E`：向左旋轉 / 向右旋轉 (yaw)

請確保目前啟用的終端機視窗沒有阻擋 Pygame 接收輸入；在移動機器人時，只需點選彈出的 Pygame 視窗即可操作。
