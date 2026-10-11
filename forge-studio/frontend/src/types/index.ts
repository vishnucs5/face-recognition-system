export type AppMode = 'txt2img' | 'img2img' | 'inpaint' | 'controlnet';
export type AppNav = 'generate' | 'gallery' | 'loras' | 'training' | 'presets' | 'settings';

export interface LoraInput {
  name: string;
  strength: number;
}

export interface GenerationRequest {
  mode: 'txt2img' | 'img2img' | 'inpaint' | 'controlnet';
  prompt: string;
  negative_prompt?: string;
  model: string;
  width: number;
  height: number;
  steps: number;
  cfg_scale: number;
  sampler: string;
  seed?: number | null;
  denoise: number;
  loras: LoraInput[];
  source_image_id?: string | null;
  mask_image_id?: string | null;
  priority?: number;
  client_request_id?: string | null;
  preset_name?: string | null;
  preset_version?: number | null;
  controlnet_type?: string | null;
  controlnet_model?: string | null;
  controlnet_strength?: number;
  controlnet_start_percent?: number;
  controlnet_end_percent?: number;
  controlnet_preprocessor?: string | null;
}

export interface GenerationResponse {
  id: string;
  mode: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'waiting_for_gpu';
  priority?: number;
  estimated_vram_mb?: number | null;
  estimated_duration_sec?: number | null;
  client_request_id?: string | null;
  generation_hash?: string | null;
  retry_count?: number;
  image_path?: string | null;
  image_url?: string | null;
  source_image_path?: string | null;
  source_image_url?: string | null;
  mask_image_path?: string | null;
  mask_image_url?: string | null;
  prompt: string;
  negative_prompt: string;
  model: string;
  loras: LoraInput[];
  width: number;
  height: number;
  steps: number;
  cfg_scale: number;
  sampler: string;
  seed?: number | null;
  denoise: number;
  controlnet_type?: string | null;
  controlnet_model?: string | null;
  controlnet_strength?: number | null;
  controlnet_start_percent?: number | null;
  controlnet_end_percent?: number | null;
  controlnet_preprocessor?: string | null;
  preset_name?: string | null;
  preset_version?: number | null;
  settings_snapshot_json?: string | null;
  gpu_telemetry_snapshot_json?: string | null;
  workflow_graph_hash?: string | null;
  asset_hashes_json?: string | null;
  created_at?: string | null;
  completed_at?: string | null;
  duration_ms?: number | null;
  error_message?: string | null;
  comfy_prompt_id?: string | null;
}

export interface ControlNetModelInfo {
  name: string;
  filename: string;
  type: string;
  size_mb: number;
}

export interface ControlNetTypeInfo {
  id: string;
  name: string;
  description: string;
  preprocessor: string;
  default_strength: number;
}

export interface GenerationProgressUpdate {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  step: number;
  total_steps: number;
  progress_percentage: number;
  preview_url?: string | null;
  message?: string | null;
  image_url?: string | null;
}

export interface GpuStatus {
  available: boolean;
  name: string;
  vram_used_mb: number;
  vram_total_mb: number;
  vram_free_mb: number;
  utilization_pct: number;
  temperature_c?: number | null;
}

export interface SystemStatusResponse {
  comfyui_online: boolean;
  comfyui_url: string;
  ollama_online: boolean;
  ollama_url: string;
  ollama_model: string;
  demo_mode: boolean;
  gpu: GpuStatus;
}

export interface ModelInfo {
  name: string;
  filename: string;
  size_mb: number;
  type: string;
}

export interface LoraInfo {
  name: string;
  filename: string;
  size_mb: number;
}

export interface CopilotSuggestion {
  prompt?: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
  steps?: number;
  cfg_scale?: number;
  sampler?: string;
  denoise?: number;
  controlnet_type?: string;
  controlnet_strength?: number;
  suggested_loras?: string[];
}

export interface CopilotResponse {
  reply: string;
  suggested_settings?: CopilotSuggestion | null;
  variations: string[];
}

export interface LoraVaultItem {
  id: string;
  name: string;
  filename: string;
  file_path: string;
  file_size_mb: number;
  base_model: string;
  trigger_words: string[];
  preview_image_url?: string | null;
  recommended_weight: number;
  training_status: string;
  notes?: string | null;
  is_available: boolean;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface DatasetIssue {
  type: 'corrupt' | 'duplicate' | 'unsupported_format' | 'missing_caption' | 'low_resolution' | 'high_aspect_ratio';
  file: string;
  message: string;
}

export interface DatasetValidationResponse {
  is_valid: boolean;
  dataset_path: string;
  total_images: number;
  supported_formats: string[];
  missing_captions_count: number;
  corrupt_images_count: number;
  duplicates_count: number;
  resolution_range: string;
  suggested_resolution: number;
  issues: DatasetIssue[];
  sample_images: string[];
}

export interface CaptionPreviewItem {
  image_file: string;
  caption_file: string;
  existing_caption: string;
  proposed_caption: string;
  changed: boolean;
}

export interface CaptionPreviewResponse {
  items: CaptionPreviewItem[];
  total_files: number;
  changed_files: number;
}

export interface TrainingHealthResponse {
  engine_available: boolean;
  engine_name: string | null;
  message: string;
  supported_engines: string[];
  gpu_vram_gb: number;
  recommended_preset: {
    resolution: number;
    batch_size: number;
    network_dim: number;
    network_alpha: number;
    learning_rate: number;
    text_encoder_lr: number;
    mixed_precision: string;
    gradient_checkpointing: boolean;
    max_train_steps: number;
    estimated_vram_gb: number;
    notes?: string;
  };
}

export interface TrainingJobResponse {
  id: string;
  status: 'queued' | 'validating' | 'training' | 'saving' | 'completed' | 'cancelled' | 'failed';
  goal: 'character' | 'product' | 'style' | 'object' | 'custom';
  training_engine: string;
  dataset_path: string;
  dataset_image_count: number;
  base_model: string;
  lora_name: string;
  trigger_word: string;
  config: Record<string, any>;
  current_epoch: number;
  total_epochs: number;
  current_step: number;
  total_steps: number;
  current_loss?: number | null;
  vram_usage_mb?: number | null;
  logs: string[];
  error_message?: string | null;
  artifact_path?: string | null;
  preview_images: Array<{
    epoch: number;
    step: number;
    url: string;
    prompt?: string;
    weight?: number;
  }>;
  created_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface ObservedProcess {
  pid: number;
  name: string;
  vram_used_mb: number;
}

export interface ActiveLeaseInfo {
  lease_type: string;
  job_id: string;
  holder: string;
  acquired_at: string;
  expires_at: string;
}

export interface TelemetryResponse {
  gpu_name: string;
  total_vram_mb: number;
  used_vram_mb: number;
  free_vram_mb: number;
  utilization_pct: number;
  temperature_c: number | null;
  cpu_ram_used_mb: number;
  cpu_ram_total_mb: number;
  observed_processes: ObservedProcess[];
  active_lease: ActiveLeaseInfo | null;
  state: 'gpu_ready' | 'gpu_memory_pressure' | 'gpu_unavailable' | 'cuda_unavailable' | 'ollama_active' | 'comfyui_active' | 'training_active' | 'waiting_for_gpu';
  pressure_threshold_pct: number;
  critical_threshold_pct: number;
  freshness_ms: number;
  is_stale: boolean;
  backend_latency_ms: number;
  sampled_at: string;
}

export interface PresetModel {
  id: string;
  name: string;
  description: string;
  category: 'built-in' | 'user';
  version: number;
  is_built_in: boolean;
  mode: 'txt2img' | 'img2img' | 'inpaint' | 'controlnet';
  base_model_family: string;
  checkpoint_name?: string | null;
  prompt_template: string;
  negative_prompt: string;
  width: number;
  height: number;
  steps: number;
  cfg_scale: number;
  sampler: string;
  scheduler: string;
  seed_policy: 'random' | 'fixed' | 'increment';
  fixed_seed?: number | null;
  denoise: number;
  loras: LoraInput[];
  controlnet_settings: Record<string, any>;
  vram_profile: 'balanced' | 'max_quality' | 'max_speed' | 'conservative';
  created_at?: string;
  updated_at?: string;
}

export interface PresetDiffReview {
  preset_name: string;
  preset_version: number;
  changes: Record<string, { current: any; preset: any }>;
  dependencies: Record<string, boolean>;
  estimated_vram_mb: number;
}

export interface DiagnosticCheck {
  id: string;
  name: string;
  category: string;
  status: 'ok' | 'warning' | 'error';
  message: string;
  details?: string | null;
  remediation_cmd?: string | null;
  checked_at: string;
}

export interface DiagnosticsReport {
  checks: DiagnosticCheck[];
  sampled_at: string;
  cache_ttl_sec: number;
}

export interface BenchmarkResult {
  id: string;
  timestamp: string;
  gpu_name: string;
  total_time_ms: number;
  prompt_to_first_img_ms: number;
  steps_per_sec: number;
  peak_vram_mb: number;
  avg_vram_mb: number;
  gpu_util_pct: number;
  ollama_first_token_ms: number | null;
  ollama_tokens_per_sec: number | null;
  env_metadata: Record<string, any>;
  options: Record<string, any>;
}

export interface CopilotActionResponse {
  action_type: string;
  explanation: string;
  suggested_changes: Record<string, any>;
  preset_id?: string | null;
  preset_name?: string | null;
  metrics: Record<string, any>;
}

export interface ErrorClassification {
  subsystem: string;
  category: string;
  can_retry: boolean;
  sanitized_message: string;
  remediation_guidance?: string | null;
  error_code?: string | null;
}
