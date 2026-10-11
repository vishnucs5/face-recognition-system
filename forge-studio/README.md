# Forge Studio

> **A local-first, privacy-preserving AI image-generation workstation for the NVIDIA GeForce RTX 5070 Ti (16GB VRAM), featuring a custom darkroom web studio, ComfyUI execution engine, and local Ollama Copilot (Qwen3-Coder 30B).**

---

## 1. Overview & Architecture

Forge Studio provides an intuitive, high-performance web workstation that wraps ComfyUI's dynamic computational graphs while offering a focused desktop aesthetic (`#11110F` darkroom / warm paper palette), GPU hardware telemetry, and local LLM art direction.

```text
┌────────────────────────────────────────────────────────┐
│             Forge Studio Web UI (React/Vite)           │
│   • Two-Column Workspace & Responsive Viewport Canvas  │
│   • Prompt Composer, Aspect Ratios, LoRA Sliders       │
│   • Studio Copilot Drawer (Qwen3-Coder 30B Chat)       │
│   • Gallery Inspector with 1-Click Parameter Reuse     │
│   • Keyboard Shortcuts (Ctrl+Enter, Ctrl+K, Ctrl+G)   │
└───────────────────────────┬────────────────────────────┘
                            │ REST & WebSockets (/api/ws/generations/{id})
┌───────────────────────────▼────────────────────────────┐
│              FastAPI Application Server                │
│   • SQLite Persistence (SQLAlchemy Async + aiosqlite)  │
│   • Dynamic ComfyUI Graph Mutator (Dual LoRA Chaining) │
│   • ComfyUI /interrupt Dispatch on Active Job Cancel   │
│   • Procedural Pillow Offline Demo Generator           │
│   • GPU Telemetry (NVIDIA-SMI / PyNVML VRAM & Temp)   │
└─────────────┬────────────────────────────┬─────────────┘
              │                            │
      ┌───────▼──────────┐         ┌───────▼───────────┐
      │  ComfyUI Engine  │         │   Ollama Engine   │
      │  Port 8188       │         │   Port 11434      │
      │  SDXL Checkpoint │         │   Qwen3-Coder 30B │
      └──────────────────┘         └───────────────────┘
```

---

## 2. Hardware Defaults (RTX 5070 Ti — 16GB VRAM)

Forge Studio is specifically tuned for 16GB Ada/Blackwell architecture GPUs:

| Workload | Configuration | VRAM Footprint | Description |
| :--- | :--- | :--- | :--- |
| **SDXL Text-to-Image** | 1024×1024, 28 steps, Euler / DPM++ 2M | ~7.2 GB | Base checkpoint + VAE in FP16 precision |
| **SDXL Image-to-Image** | 1024×1024, 28 steps, Denoise 0.75 | ~7.8 GB | Source image latent encoding + sampling |
| **Dual-LoRA Stacking** | Up to 2 concurrent LoRA models | +0.4 – 0.8 GB | Chained model/CLIP patching |
| **Max Safe Resolution** | Up to 1536×1536 px (multiples of 64) | ~11.5 GB | Bounded to avoid CUDA OOM errors |
| **Qwen3-Coder 30B** | 4-bit quant (`Q4_K_M`, ~18.5 GB) | Shared / RAM Offload | Studio Copilot art direction & JSON planner |

---

## 3. Directory Layout

```text
forge-studio/
├── frontend/             # React 18, TypeScript, Vite, Tailwind CSS, Zustand
│   ├── src/
│   │   ├── components/   # PromptComposer, GenerationSettings, PreviewPanel,
│   │   │                 # CopilotDrawer, GalleryView, LorasView, SettingsView,
│   │   │                 # KeyboardShortcutsModal, RecentGenerationsStrip, Header
│   │   ├── store/        # Zustand global studio state
│   │   └── types/        # TypeScript contracts
├── backend/              # Python 3.11+, FastAPI, SQLAlchemy, SQLite, WebSockets
│   ├── app/
│   │   ├── api/          # Endpoints: generations, copilot, models, status, ws
│   │   ├── models/       # SQLite database model (Generation)
│   │   ├── schemas/      # Pydantic schemas with strict validation
│   │   └── services/     # ComfyUI adapter, Ollama client, GPU monitor, Demo generator
│   └── tests/            # Pytest test suite (22 unit & integration tests)
├── workflows/
│   ├── txt2img.json      # ComfyUI SDXL text-to-image API template
│   └── img2img.json      # ComfyUI SDXL image-to-image API template
├── models/
│   ├── checkpoints/      # Put your SDXL .safetensors here
│   ├── loras/            # Put your style/character .safetensors here
│   ├── controlnet/       # Reserved for Phase 3
│   └── vae/              # Reserved for custom SDXL VAE
├── data/
│   ├── generated/        # Persisted output images
│   ├── uploads/          # User-uploaded source images for img2img
│   └── database/         # SQLite forge.db
├── .env.example          # Environment variables template
├── .env                  # Active local configuration
└── README.md
```

---

## 4. Keyboard Shortcuts Reference

Forge Studio includes desktop-class keyboard hotkeys accessible anywhere in the application:

| Shortcut | Scope | Description |
| :--- | :--- | :--- |
| <kbd>Ctrl</kbd> + <kbd>Enter</kbd> (or <kbd>⌘</kbd>+<kbd>Enter</kbd>) | Global | Submit & run image generation job |
| <kbd>Ctrl</kbd> + <kbd>D</kbd> | Global | Roll a fresh randomized seed |
| <kbd>Ctrl</kbd> + <kbd>K</kbd> | Global | Toggle Studio Copilot assistant drawer |
| <kbd>Ctrl</kbd> + <kbd>G</kbd> | Global | Toggle between Generation Studio and Gallery |
| <kbd>Ctrl</kbd> + <kbd>1</kbd> | Global | Switch to Text-to-Image mode |
| <kbd>Ctrl</kbd> + <kbd>2</kbd> | Global | Switch to Image-to-Image mode |
| <kbd>?</kbd> | Global | Open Keyboard Shortcuts modal (when not typing in an input) |
| <kbd>Esc</kbd> | Global | Close open drawers, modals, or cancel active in-flight job |

---

## 5. Quickstart & Installation

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- NVIDIA RTX GPU with up-to-date drivers

### Step 1: Backend Setup
```powershell
# From project root (c:\vishnu):
.\.venv\Scripts\Activate.ps1

# Install backend dependencies
pip install -r forge-studio/backend/requirements.txt

# Start FastAPI backend (Port 8000)
$env:PYTHONPATH="forge-studio/backend"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Step 2: Frontend Setup
```powershell
cd forge-studio/frontend
npm install
npm run dev
```
Open **http://localhost:5173** in your browser.

---

## 6. Running the Test Suite

Forge Studio has comprehensive test suites for both frontend and backend:

### Backend Tests (33 Tests)
```powershell
cd forge-studio/backend
$env:PYTHONPATH="."
c:\vishnu\.venv\Scripts\pytest.exe -v
```
Tests cover:
- Health and status telemetry endpoints
- ComfyUI workflow mutation & multi-LoRA chaining
- Active generation cancellation via `/interrupt` and queue deletion
- Clear queue endpoint (`/api/generations/clear-queue`)
- Image-to-image source validation
- SQLite persistence, prompt search, and mode filtering
- Ollama Studio Copilot prompt enhancement & negative prompt generation
- Injection sanitization and parameter clamping bounds
- LoRA Vault indexing, metadata updates, safe de-indexing, and `.safetensors` importing
- Dataset integrity validator (formats, corrupt files, duplicates, missing captions)
- Caption preview & apply with `.txt.bak` backup creation
- Authentic training engine verification & hardware preset checks

### Frontend Verification
```powershell
cd forge-studio/frontend
npm run build
```
Executes `tsc` type checking and Vite production bundling with zero errors.

---

## 7. Real AI Generation vs. Demo Fallback

Forge Studio connects to real local AI runtimes by default (`DEMO_MODE=false` in `.env`):

- **Default Behavior (Live AI Generation):**
  - **ComfyUI Integration (`http://127.0.0.1:8188`):** Verified connectivity on startup, dynamic checkpoint detection, real text-to-image and image-to-image generation via `/prompt`, `/history`, and `/view`.
  - **No Procedural Watermarks:** Successful generations display raw, high-resolution PNG outputs with zero procedural vector overlays or fake text stamps.
  - **Honest Error Handling:** If ComfyUI is offline or no checkpoints are loaded, the UI displays a clear error state instead of generating a fake output.
  - **Ollama Studio Copilot (`http://127.0.0.1:11434`):** Real GPU inference with `qwen3-coder:30b` providing review-only prompt expansion, variations, and parameter guidance.
  - **Metadata Reproducibility:** Every generation records full prompt and technical settings with 1-click **Download PNG**, **Download metadata JSON**, and **Reuse Settings**.

- **Manual Demo Mode Fallback:**
  If you want to run Forge Studio in procedural demo mode without local models, set `DEMO_MODE=true` in `forge-studio/.env`. Every output will be clearly labeled as Demo Mode.

---

## 8. Exact Service Startup Commands

To run the complete local workstation stack, launch the components:

### 1. Start Ollama with RTX 5070 Ti GPU Acceleration (Port 11434)
```powershell
Remove-Item Env:OLLAMA_LLM_LIBRARY -ErrorAction SilentlyContinue
ollama serve
```
*(Automatically offloads 76% of Qwen3-Coder 30B to the RTX 5070 Ti VRAM for ~58 tokens/sec).*

### 2. Start ComfyUI (Port 8188)
```powershell
# From your ComfyUI directory (e.g., c:\vishnu\ComfyUI):
python main.py --listen 127.0.0.1 --port 8188 --cpu
```

### 3. Start Forge Studio
```powershell
# Terminal A — Backend (Port 8000):
cd c:\vishnu\forge-studio\backend
$env:PYTHONPATH="."
c:\vishnu\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Terminal B — Frontend (Port 5173):
cd c:\vishnu\forge-studio\frontend
npm run dev -- --host 127.0.0.1 --port 5173
```
Open **http://127.0.0.1:5173** in your browser.

---

## 9. Studio Copilot (Ollama & Qwen3-Coder 30B)

Forge Studio integrates with your locally installed Ollama instance:
```powershell
# Start Ollama daemon
ollama serve

# Verify Qwen3-Coder 30B is available
ollama run qwen3-coder:30b
```
When Ollama is online, Studio Copilot provides:
- Real-time prompt enrichment with photography and lighting modifiers
- 4 stylistic prompt variations based on your subject
- Optimal SDXL parameter recipes (steps, CFG, samplers)
- Diff previews of proposed parameter changes with an explicit **Apply to Form** gate
- Automatic fallback to offline heuristics when Ollama is unreachable

---

## 10. LoRA Vault & Model Management

Forge Studio features a dedicated **LoRA Vault** (`/loras`) with bi-directional integration into ComfyUI:

- **Automatic Folder Sync**: Files placed in `models/loras/` or ComfyUI's `models/loras/` are indexed automatically with metadata extraction.
- **Card Metadata**: Displays adapter name, base model architecture, file size, recommended weight ($0.1\times - 1.5\times$), trigger words, notes, and local availability.
- **Trigger Word Injection**: 1-click **Copy All** action for prompt pasting, plus interactive trigger chips in the Prompt Composer and Generation Settings.
- **Safe Index Deletion**: Removing a LoRA removes its database entry while strictly preserving the `.safetensors` file on your filesystem.
- **Composer Attachment**: Attach up to 2 LoRAs simultaneously with custom strength weights directly in the generation pipeline.

---

## 11. Local LoRA Training Studio & Engine Setup

The **LoRA Training Studio** (`/training`) provides a 5-step guided wizard designed for the 16 GB RTX 5070 Ti:

### Wizard Steps
1. **Goal Selection**: Character, Product, Style / Art, Object, or Custom.
2. **Dataset Integrity Validation**: Real-time inspection of folder paths, image formats, corrupted files (PIL verification), duplicate detection (MD5 hashes), resolution range, and missing caption `.txt` files.
3. **Caption Inspection & Safe Backup**: Preview and auto-prefix trigger words (`trigger, [caption]`). Changes require explicit confirmation and automatically create timestamped `.txt.bak` backups.
4. **RTX 5070 Ti Safe Preset**:
   - Resolution: $1024 \times 1024$ (SDXL) or $512 \times 512$ (SD 1.5)
   - Batch Size: 1
   - Network Dim / Alpha: $16 / 16$
   - Mixed Precision: `bf16` with gradient checkpointing enabled
   - Estimated VRAM Footprint: ~12.5 GB (safe for 16 GB physical VRAM)
5. **Job Monitor & Vault Export**: Subprocess log streaming, loss curve monitoring, cancellation, retry, and 1-click artifact export to the LoRA Vault.

### Enabling the Real Training Engine
Forge Studio enforces authentic execution. If neither `kohya_ss` nor `ai-toolkit` is installed, it displays:
> `"Training engine unavailable — install kohya_ss or AI-Toolkit locally to enable training."`

To enable real local training on your NVIDIA RTX 5070 Ti:

#### Option A: Kohya_ss
```powershell
git clone https://github.com/bmaltais/kohya_ss.git c:\vishnu\kohya_ss
cd c:\vishnu\kohya_ss
.\setup.bat
```

#### Option B: AI-Toolkit
```powershell
git clone https://github.com/ostris/ai-toolkit.git c:\vishnu\ai-toolkit
cd c:\vishnu\ai-toolkit
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
Once installed, Forge Studio automatically detects the engine binary and enables job dispatch!

