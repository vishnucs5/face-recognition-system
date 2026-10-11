import { create } from 'zustand';
import {
  AppMode,
  AppNav,
  BenchmarkResult,
  ControlNetModelInfo,
  ControlNetTypeInfo,
  DiagnosticsReport,
  ErrorClassification,
  GenerationProgressUpdate,
  GenerationResponse,
  LoraInfo,
  LoraInput,
  LoraVaultItem,
  ModelInfo,
  PresetDiffReview,
  PresetModel,
  SystemStatusResponse,
  TelemetryResponse,
  TrainingHealthResponse,
  TrainingJobResponse,
} from '../types';

interface StudioState {
  // Navigation & Mode
  activeNav: AppNav;
  mode: AppMode;
  setActiveNav: (nav: AppNav) => void;
  setMode: (mode: AppMode) => void;

  // LoRA Vault & Training State
  loraVault: LoraVaultItem[];
  setLoraVault: (items: LoraVaultItem[]) => void;
  sendLoraToComposer: (lora: LoraVaultItem, weight?: number) => void;
  trainingHealth: TrainingHealthResponse | null;
  setTrainingHealth: (health: TrainingHealthResponse | null) => void;
  trainingJobs: TrainingJobResponse[];
  setTrainingJobs: (jobs: TrainingJobResponse[]) => void;
  activeTrainingJob: TrainingJobResponse | null;
  setActiveTrainingJob: (job: TrainingJobResponse | null) => void;

  // Form State
  prompt: string;
  negative_prompt: string;
  model: string;
  width: number;
  height: number;
  steps: number;
  cfg_scale: number;
  sampler: string;
  seed: number | null;
  denoise: number;
  loras: LoraInput[];
  source_image_id: string | null;
  source_image_preview: string | null;
  mask_image_id: string | null;
  mask_image_preview: string | null;
  brush_size: number;
  brush_mode: 'draw' | 'erase';

  // ControlNet Settings
  controlnet_type: string;
  controlnet_model: string;
  controlnet_strength: number;
  controlnet_start_percent: number;
  controlnet_end_percent: number;
  controlnet_preprocessor: string;
  availableControlNetModels: ControlNetModelInfo[];
  availableControlNetTypes: ControlNetTypeInfo[];

  setPrompt: (prompt: string) => void;
  setNegativePrompt: (negative_prompt: string) => void;
  setModel: (model: string) => void;
  setDimensions: (width: number, height: number) => void;
  setSteps: (steps: number) => void;
  setCfgScale: (cfg_scale: number) => void;
  setSampler: (sampler: string) => void;
  setSeed: (seed: number | null) => void;
  setDenoise: (denoise: number) => void;
  addLora: (lora: LoraInput) => void;
  removeLora: (name: string) => void;
  updateLoraStrength: (name: string, strength: number) => void;
  setSourceImage: (id: string | null, previewUrl: string | null) => void;
  setMaskImage: (id: string | null, previewUrl: string | null) => void;
  setBrushSize: (size: number) => void;
  setBrushMode: (mode: 'draw' | 'erase') => void;
  clearMask: () => void;

  setControlNetType: (type: string) => void;
  setControlNetModel: (model: string) => void;
  setControlNetStrength: (strength: number) => void;
  setControlNetStartPercent: (start: number) => void;
  setControlNetEndPercent: (end: number) => void;
  setControlNetPreprocessor: (prep: string) => void;
  setAvailableControlNetModels: (models: ControlNetModelInfo[]) => void;
  setAvailableControlNetTypes: (types: ControlNetTypeInfo[]) => void;

  // System & Telemetry
  systemStatus: SystemStatusResponse | null;
  availableModels: ModelInfo[];
  availableLoras: LoraInfo[];
  setSystemStatus: (status: SystemStatusResponse) => void;
  setAvailableModels: (models: ModelInfo[]) => void;
  setAvailableLoras: (loras: LoraInfo[]) => void;

  // Generation Lifecycle
  isGenerating: boolean;
  isSubmitting: boolean;
  setIsSubmitting: (submitting: boolean) => void;
  currentJobId: string | null;
  progress: GenerationProgressUpdate | null;
  activeGeneration: GenerationResponse | null;
  recentGenerations: GenerationResponse[];
  startGeneration: (job: GenerationResponse) => void;
  setProgress: (progress: GenerationProgressUpdate) => void;
  finishGeneration: (result: GenerationResponse) => void;
  cancelActiveGeneration: (jobId?: string) => void;
  selectGeneration: (gen: GenerationResponse) => void;
  setRecentGenerations: (gens: GenerationResponse[]) => void;
  emptyQueue: () => Promise<void>;

  // Copilot drawer toggle
  copilotOpen: boolean;
  setCopilotOpen: (open: boolean) => void;

  // Error Banner & Notifications
  errorMessage: string | null;
  setErrorMessage: (msg: string | null) => void;

  // Responsive Workspace View (on compact screens)
  mobileView: 'controls' | 'preview';
  setMobileView: (view: 'controls' | 'preview') => void;

  // Shortcuts Dialog
  shortcutsModalOpen: boolean;
  setShortcutsModalOpen: (open: boolean) => void;
  randomizeSeed: () => void;

  // Actions
  reuseSettings: (gen: GenerationResponse) => void;
  sendToImg2Img: (imageUrl: string, sourceId?: string) => void;
  sendToInpaint: (imageUrl: string, sourceId?: string) => void;
  sendToControlNet: (imageUrl: string, sourceId?: string) => void;

  // VRAM Policy & Telemetry
  telemetry: TelemetryResponse | null;
  setTelemetry: (status: TelemetryResponse | null) => void;
  vramPolicy: 'balanced' | 'max_quality' | 'max_speed' | 'conservative';
  setVramPolicy: (policy: 'balanced' | 'max_quality' | 'max_speed' | 'conservative') => void;

  // Queue Scheduling
  priority: number;
  setPriority: (priority: number) => void;
  isQueuePaused: boolean;
  setIsQueuePaused: (paused: boolean) => void;
  pauseQueue: () => Promise<void>;
  resumeQueue: () => Promise<void>;
  reorderQueue: (jobId: string, newPriority: number) => Promise<void>;
  cancelQueuedJobsOnly: () => Promise<void>;
  interruptActiveJob: () => Promise<void>;

  // Presets
  presets: PresetModel[];
  setPresets: (presets: PresetModel[]) => void;
  selectedPreset: PresetModel | null;
  setSelectedPreset: (preset: PresetModel | null) => void;
  diffModalPreset: PresetModel | null;
  setDiffModalPreset: (preset: PresetModel | null) => void;
  loadPresets: () => Promise<void>;
  applyPresetToComposer: (preset: PresetModel) => void;

  // Diagnostics & Hardware Benchmark
  diagnosticsReport: DiagnosticsReport | null;
  setDiagnosticsReport: (report: DiagnosticsReport | null) => void;
  benchmarkResults: BenchmarkResult[];
  setBenchmarkResults: (results: BenchmarkResult[]) => void;
  isBenchmarking: boolean;
  setIsBenchmarking: (benchmarking: boolean) => void;
  benchmarkWarmup: boolean;
  setBenchmarkWarmup: (warmup: boolean) => void;

  // Sanitized Error Classification
  errorClassification: ErrorClassification | null;
  setErrorClassification: (classification: ErrorClassification | null) => void;
}

const STORAGE_KEY = 'forge_recent_generations';

const loadCachedGenerations = (): GenerationResponse[] => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: GenerationResponse[] = JSON.parse(raw);
    return parsed.map((g) => {
      // Reconcile stale in-flight generations after refresh
      if (g.status === 'queued' || g.status === 'running') {
        return {
          ...g,
          status: 'cancelled',
          error_message: 'Session interrupted by reload',
        };
      }
      return g;
    });
  } catch (e) {
    console.warn('Failed to parse cached generations:', e);
    return [];
  }
};

const saveCachedGenerations = (gens: GenerationResponse[]) => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(gens.slice(0, 50)));
  } catch (e) {
    console.warn('Failed to save cached generations:', e);
  }
};

const initialCachedGenerations = loadCachedGenerations();

export const useStudioStore = create<StudioState>((set) => ({
  activeNav: 'generate',
  mode: 'txt2img',
  setActiveNav: (activeNav) => set({ activeNav }),
  setMode: (mode) => set({ mode }),

  // Default Form values (SDXL baseline)
  prompt: 'A cinematic high-detail photograph of a futuristic research laboratory in the mountains at golden hour',
  negative_prompt: 'blurry, low quality, distorted, deformed, watermark, oversaturated',
  model: 'sd_xl_base_1.0.safetensors',
  width: 1024,
  height: 1024,
  steps: 28,
  cfg_scale: 6.5,
  sampler: 'euler',
  seed: null,
  denoise: 0.75,
  loras: [],
  source_image_id: null,
  source_image_preview: null,
  mask_image_id: null,
  mask_image_preview: null,
  brush_size: 32,
  brush_mode: 'draw',

  controlnet_type: 'canny',
  controlnet_model: '',
  controlnet_strength: 1.0,
  controlnet_start_percent: 0.0,
  controlnet_end_percent: 1.0,
  controlnet_preprocessor: 'canny',
  availableControlNetModels: [],
  availableControlNetTypes: [],

  setPrompt: (prompt) => set({ prompt }),
  setNegativePrompt: (negative_prompt) => set({ negative_prompt }),
  setModel: (model) =>
    set((state) => {
      const isSDXL =
        model.toLowerCase().includes('sd_xl') ||
        model.toLowerCase().includes('sdxl') ||
        model.toLowerCase().includes('pony') ||
        model.toLowerCase().includes('juggernaut');
      // If switching architecture and user was at previous default, adapt dimensions
      let width = state.width;
      let height = state.height;
      if (!isSDXL && state.width === 1024 && state.height === 1024) {
        width = 512;
        height = 512;
      } else if (isSDXL && state.width === 512 && state.height === 512) {
        width = 1024;
        height = 1024;
      }
      return { model, width, height };
    }),
  setDimensions: (width, height) => set({ width, height }),
  setSteps: (steps) => set({ steps }),
  setCfgScale: (cfg_scale) => set({ cfg_scale }),
  setSampler: (sampler) => set({ sampler }),
  setSeed: (seed) => set({ seed }),
  setDenoise: (denoise) => set({ denoise }),

  addLora: (lora) =>
    set((state) => {
      if (state.loras.length >= 2 || state.loras.some((l) => l.name === lora.name)) {
        return state;
      }
      return { loras: [...state.loras, lora] };
    }),

  removeLora: (name) =>
    set((state) => ({
      loras: state.loras.filter((l) => l.name !== name),
    })),

  updateLoraStrength: (name, strength) =>
    set((state) => ({
      loras: state.loras.map((l) => (l.name === name ? { ...l, strength } : l)),
    })),

  setSourceImage: (source_image_id, source_image_preview) =>
    set({ source_image_id, source_image_preview }),

  setMaskImage: (mask_image_id, mask_image_preview) =>
    set({ mask_image_id, mask_image_preview }),

  setBrushSize: (brush_size) => set({ brush_size }),
  setBrushMode: (brush_mode) => set({ brush_mode }),
  clearMask: () => set({ mask_image_id: null, mask_image_preview: null }),

  setControlNetType: (controlnet_type) => set({ controlnet_type }),
  setControlNetModel: (controlnet_model) => set({ controlnet_model }),
  setControlNetStrength: (controlnet_strength) => set({ controlnet_strength }),
  setControlNetStartPercent: (controlnet_start_percent) => set({ controlnet_start_percent }),
  setControlNetEndPercent: (controlnet_end_percent) => set({ controlnet_end_percent }),
  setControlNetPreprocessor: (controlnet_preprocessor) => set({ controlnet_preprocessor }),
  setAvailableControlNetModels: (availableControlNetModels) =>
    set((state) => {
      const selectedExists = availableControlNetModels.some((m) => m.filename === state.controlnet_model);
      const first = availableControlNetModels.length > 0 ? availableControlNetModels[0].filename : '';
      return {
        availableControlNetModels,
        controlnet_model: selectedExists ? state.controlnet_model : first,
      };
    }),
  setAvailableControlNetTypes: (availableControlNetTypes) => set({ availableControlNetTypes }),

  // LoRA Vault & Training State
  loraVault: [],
  setLoraVault: (loraVault) => set({ loraVault }),
  sendLoraToComposer: (lora, weight = 0.8) =>
    set((state) => {
      // If already attached, update strength
      if (state.loras.some((l) => l.name === lora.filename)) {
        return {
          loras: state.loras.map((l) => (l.name === lora.filename ? { ...l, strength: weight } : l)),
          activeNav: 'generate',
        };
      }
      // If at capacity (2), replace the second
      if (state.loras.length >= 2) {
        return {
          loras: [state.loras[0], { name: lora.filename, strength: weight }],
          activeNav: 'generate',
        };
      }
      return {
        loras: [...state.loras, { name: lora.filename, strength: weight }],
        activeNav: 'generate',
      };
    }),
  trainingHealth: null,
  setTrainingHealth: (trainingHealth) => set({ trainingHealth }),
  trainingJobs: [],
  setTrainingJobs: (trainingJobs) => set({ trainingJobs }),
  activeTrainingJob: null,
  setActiveTrainingJob: (activeTrainingJob) => set({ activeTrainingJob }),

  systemStatus: null,
  availableModels: [],
  availableLoras: [],
  setSystemStatus: (systemStatus) => set({ systemStatus }),
  setAvailableModels: (availableModels) =>
    set((state) => {
      const modelExists = availableModels.some(
        (m) => m.filename === state.model || m.name === state.model
      );
      if (!modelExists && availableModels.length > 0) {
        const first = availableModels[0].filename;
        const isFirstSDXL =
          first.toLowerCase().includes('sd_xl') ||
          first.toLowerCase().includes('sdxl') ||
          first.toLowerCase().includes('pony') ||
          first.toLowerCase().includes('juggernaut');
        return {
          availableModels,
          model: first,
          width: isFirstSDXL ? 1024 : 512,
          height: isFirstSDXL ? 1024 : 512,
        };
      }
      return { availableModels };
    }),
  setAvailableLoras: (availableLoras) => set({ availableLoras }),

  isGenerating: false,
  isSubmitting: false,
  setIsSubmitting: (isSubmitting) => set({ isSubmitting }),
  currentJobId: null,
  progress: null,
  activeGeneration: initialCachedGenerations.length > 0 ? initialCachedGenerations[0] : null,
  recentGenerations: initialCachedGenerations,

  startGeneration: (job) =>
    set((state) => {
      const nextRecent = [job, ...state.recentGenerations.filter((g) => g.id !== job.id)];
      saveCachedGenerations(nextRecent);
      return {
        isGenerating: true,
        isSubmitting: false,
        currentJobId: job.id,
        activeGeneration: job,
        progress: {
          job_id: job.id,
          status: 'queued',
          step: 0,
          total_steps: job.steps || 28,
          progress_percentage: 0,
          message: 'Queued for generation...',
        },
        recentGenerations: nextRecent,
      };
    }),

  setProgress: (progress) =>
    set((state) => {
      const updatedActive =
        state.activeGeneration && state.activeGeneration.id === progress.job_id
          ? {
              ...state.activeGeneration,
              status: progress.status,
              image_url: progress.image_url || state.activeGeneration.image_url,
            }
          : state.activeGeneration;

      const updatedRecent = state.recentGenerations.map((g) =>
        g.id === progress.job_id
          ? {
              ...g,
              status: progress.status,
              image_url: progress.image_url || g.image_url,
            }
          : g
      );
      saveCachedGenerations(updatedRecent);

      return { progress, activeGeneration: updatedActive, recentGenerations: updatedRecent };
    }),

  finishGeneration: (result) =>
    set((state) => {
      const nextRecent = [result, ...state.recentGenerations.filter((g) => g.id !== result.id)];
      saveCachedGenerations(nextRecent);
      return {
        isGenerating: false,
        isSubmitting: false,
        currentJobId: null,
        activeGeneration: result,
        recentGenerations: nextRecent,
        progress: null,
      };
    }),

  cancelActiveGeneration: (jobId) =>
    set((state) => {
      const targetId = jobId || state.currentJobId;
      const updatedRecent = state.recentGenerations.map((g) =>
        g.id === targetId ? { ...g, status: 'cancelled' as const, error_message: 'Locally canceled' } : g
      );
      saveCachedGenerations(updatedRecent);
      const updatedActive =
        state.activeGeneration?.id === targetId
          ? { ...state.activeGeneration, status: 'cancelled' as const, error_message: 'Locally canceled' }
          : state.activeGeneration;
      return {
        isGenerating: false,
        isSubmitting: false,
        currentJobId: null,
        progress: null,
        recentGenerations: updatedRecent,
        activeGeneration: updatedActive,
      };
    }),

  selectGeneration: (gen) => set({ activeGeneration: gen }),

  setRecentGenerations: (recentGenerations) =>
    set((state) => {
      saveCachedGenerations(recentGenerations);
      const active = state.activeGeneration
        ? recentGenerations.find((g) => g.id === state.activeGeneration?.id) || state.activeGeneration
        : recentGenerations.length > 0
        ? recentGenerations[0]
        : null;
      return { recentGenerations, activeGeneration: active };
    }),

  emptyQueue: async () => {
    try {
      await fetch('/api/generations/clear-queue', { method: 'POST' });
      set((state) => {
        const filtered = state.recentGenerations.filter(
          (g) => g.status !== 'queued' && g.status !== 'cancelled'
        );
        saveCachedGenerations(filtered);
        return { recentGenerations: filtered };
      });
    } catch (err) {
      console.error('Failed to clear queue:', err);
    }
  },

  copilotOpen: false,
  setCopilotOpen: (copilotOpen) => set({ copilotOpen }),

  errorMessage: null,
  setErrorMessage: (errorMessage) => set({ errorMessage }),

  mobileView: 'controls',
  setMobileView: (mobileView) => set({ mobileView }),

  shortcutsModalOpen: false,
  setShortcutsModalOpen: (shortcutsModalOpen) => set({ shortcutsModalOpen }),
  randomizeSeed: () => set({ seed: Math.floor(Math.random() * 1000000000) }),

  reuseSettings: (gen) =>
    set({
      prompt: gen.prompt,
      negative_prompt: gen.negative_prompt,
      model: gen.model,
      width: gen.width,
      height: gen.height,
      steps: gen.steps,
      cfg_scale: gen.cfg_scale,
      sampler: gen.sampler,
      seed: gen.seed,
      denoise: gen.denoise,
      loras: gen.loras || [],
      controlnet_type: gen.controlnet_type || 'canny',
      controlnet_model: gen.controlnet_model || '',
      controlnet_strength: gen.controlnet_strength ?? 1.0,
      controlnet_start_percent: gen.controlnet_start_percent ?? 0.0,
      controlnet_end_percent: gen.controlnet_end_percent ?? 1.0,
      controlnet_preprocessor: gen.controlnet_preprocessor || 'canny',
      mode: (['txt2img', 'img2img', 'inpaint', 'controlnet'].includes(gen.mode) ? gen.mode : 'txt2img') as AppMode,
      activeNav: 'generate',
    }),

  sendToImg2Img: (imageUrl, sourceId) =>
    set({
      mode: 'img2img',
      activeNav: 'generate',
      source_image_preview: imageUrl,
      source_image_id: sourceId || null,
    }),

  sendToInpaint: (imageUrl, sourceId) =>
    set({
      mode: 'inpaint',
      activeNav: 'generate',
      source_image_preview: imageUrl,
      source_image_id: sourceId || null,
      mask_image_id: null,
      mask_image_preview: null,
    }),

  sendToControlNet: (imageUrl, sourceId) =>
    set({
      mode: 'controlnet',
      activeNav: 'generate',
      source_image_preview: imageUrl,
      source_image_id: sourceId || null,
    }),

  // VRAM Policy & Telemetry Implementation
  telemetry: null,
  setTelemetry: (telemetry) => set({ telemetry }),
  vramPolicy: 'balanced',
  setVramPolicy: (vramPolicy) => set({ vramPolicy }),

  // Queue Scheduling Implementation
  priority: 2,
  setPriority: (priority) => set({ priority }),
  isQueuePaused: false,
  setIsQueuePaused: (isQueuePaused) => set({ isQueuePaused }),

  pauseQueue: async () => {
    try {
      const res = await fetch('/api/queue/pause', { method: 'POST' });
      if (res.ok) set({ isQueuePaused: true });
    } catch (e) {
      console.error('Failed to pause queue:', e);
    }
  },

  resumeQueue: async () => {
    try {
      const res = await fetch('/api/queue/resume', { method: 'POST' });
      if (res.ok) set({ isQueuePaused: false });
    } catch (e) {
      console.error('Failed to resume queue:', e);
    }
  },

  reorderQueue: async (jobId: string, newPriority: number) => {
    try {
      const res = await fetch('/api/queue/reorder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: jobId, new_priority: newPriority }),
      });
      if (res.ok) {
        const gensRes = await fetch('/api/generations?limit=12');
        if (gensRes.ok) set({ recentGenerations: await gensRes.json() });
      }
    } catch (e) {
      console.error('Failed to reorder queue:', e);
    }
  },

  cancelQueuedJobsOnly: async () => {
    try {
      const res = await fetch('/api/queue/cancel-all-queued', { method: 'POST' });
      if (res.ok) {
        const gensRes = await fetch('/api/generations?limit=12');
        if (gensRes.ok) set({ recentGenerations: await gensRes.json() });
      }
    } catch (e) {
      console.error('Failed to cancel queued jobs:', e);
    }
  },

  interruptActiveJob: async () => {
    try {
      const res = await fetch('/api/queue/interrupt-active', { method: 'POST' });
      if (res.ok) {
        set({ isGenerating: false, currentJobId: null });
        const gensRes = await fetch('/api/generations?limit=12');
        if (gensRes.ok) set({ recentGenerations: await gensRes.json() });
      }
    } catch (e) {
      console.error('Failed to interrupt active job:', e);
    }
  },

  // Presets Implementation
  presets: [],
  setPresets: (presets) => set({ presets }),
  selectedPreset: null,
  setSelectedPreset: (selectedPreset) => set({ selectedPreset }),
  diffModalPreset: null,
  setDiffModalPreset: (diffModalPreset) => set({ diffModalPreset }),

  loadPresets: async () => {
    try {
      const res = await fetch('/api/presets');
      if (res.ok) {
        const data = await res.json();
        set({ presets: data });
      }
    } catch (e) {
      console.error('Failed to load presets:', e);
    }
  },

  applyPresetToComposer: (preset: PresetModel) => {
    set({
      mode: preset.mode,
      prompt: preset.prompt_template,
      negative_prompt: preset.negative_prompt,
      model: preset.checkpoint_name || 'sd_xl_base_1.0.safetensors',
      width: preset.width,
      height: preset.height,
      steps: preset.steps,
      cfg_scale: preset.cfg_scale,
      sampler: preset.sampler,
      seed: preset.seed_policy === 'fixed' ? preset.fixed_seed : null,
      denoise: preset.denoise,
      loras: preset.loras || [],
      vramPolicy: preset.vram_profile || 'balanced',
      activeNav: 'generate',
      selectedPreset: preset,
    });
  },

  // Diagnostics & Hardware Benchmark Implementation
  diagnosticsReport: null,
  setDiagnosticsReport: (diagnosticsReport) => set({ diagnosticsReport }),
  benchmarkResults: [],
  setBenchmarkResults: (benchmarkResults) => set({ benchmarkResults }),
  isBenchmarking: false,
  setIsBenchmarking: (isBenchmarking) => set({ isBenchmarking }),
  benchmarkWarmup: false,
  setBenchmarkWarmup: (benchmarkWarmup) => set({ benchmarkWarmup }),

  // Sanitized Error Classification Implementation
  errorClassification: null,
  setErrorClassification: (errorClassification) => set({ errorClassification }),
}));

