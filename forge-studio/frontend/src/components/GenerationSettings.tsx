import React, { useEffect, useRef, useState } from 'react';
import {
  Dice5,
  Sliders,
  Layers,
  Upload,
  X,
  Play,
  Square,
  Sparkles,
  AlertCircle,
  Copy,
  Check,
  Lock,
  Unlock,
  Settings2,
  Cpu,
  Target,
  Zap,
  Clock,
  ShieldAlert,
  ArrowUpCircle,
} from 'lucide-react';
import { useStudioStore } from '../store/useStudioStore';
import { SettingsSummary } from './SettingsSummary';
import { InpaintCanvas } from './InpaintCanvas';

interface GenerationSettingsProps {
  onGenerate: () => void;
  onCancel: () => void;
}

export const GenerationSettings: React.FC<GenerationSettingsProps> = ({
  onGenerate,
  onCancel,
}) => {
  const {
    mode,
    model,
    setModel,
    priority,
    setPriority,
    vramPolicy,
    telemetry,
    availableModels,
    availableLoras,
    loras,
    addLora,
    removeLora,
    updateLoraStrength,
    width,
    height,
    setDimensions,
    steps,
    setSteps,
    cfg_scale,
    setCfgScale,
    sampler,
    setSampler,
    seed,
    setSeed,
    denoise,
    setDenoise,
    source_image_id,
    source_image_preview,
    setSourceImage,
    mask_image_id,
    controlnet_type,
    controlnet_model,
    controlnet_strength,
    controlnet_start_percent,
    controlnet_end_percent,
    controlnet_preprocessor,
    availableControlNetModels,
    availableControlNetTypes,
    setControlNetType,
    setControlNetModel,
    setControlNetStrength,
    setControlNetStartPercent,
    setControlNetEndPercent,
    setControlNetPreprocessor,
    setAvailableControlNetModels,
    setAvailableControlNetTypes,
    isGenerating,
    isSubmitting,
    progress,
    prompt,
    setPrompt,
    loraVault,
    setErrorMessage,
  } = useStudioStore();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [customDimOpen, setCustomDimOpen] = useState(false);
  const [copiedSeed, setCopiedSeed] = useState(false);

  useEffect(() => {
    fetch('/api/controlnet/types')
      .then((res) => res.json())
      .then((types) => setAvailableControlNetTypes(types))
      .catch((err) => console.error('Failed to load ControlNet types', err));

    fetch('/api/controlnet/models')
      .then((res) => res.json())
      .then((models) => setAvailableControlNetModels(models))
      .catch((err) => console.error('Failed to load ControlNet models', err));
  }, [setAvailableControlNetTypes, setAvailableControlNetModels]);

  const samplers = [
    { id: 'euler', name: 'Euler (Fast baseline)' },
    { id: 'euler_ancestral', name: 'Euler Ancestral (Artistic)' },
    { id: 'dpmpp_2m', name: 'DPM++ 2M (Standard high detail)' },
    { id: 'dpmpp_sde', name: 'DPM++ SDE (Photorealistic)' },
    { id: 'dpmpp_2m_sde', name: 'DPM++ 2M SDE (Balanced)' },
    { id: 'heun', name: 'Heun (Crisp contrast)' },
    { id: 'ddim', name: 'DDIM (Deterministic)' },
  ];

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setErrorMessage('Please upload a valid image file (PNG, JPG, or WEBP).');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/generations/upload', {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        const data = await res.json();
        setSourceImage(data.source_image_id, data.url);
      } else {
        const err = await res.json();
        setErrorMessage(err.detail || 'Failed to upload image.');
      }
    } catch (err) {
      console.error('Failed to upload image:', err);
      setErrorMessage('Network error while uploading image.');
    }
  };

  const validateAndGenerate = () => {
    if (isGenerating || isSubmitting) return;
    if (!prompt.trim()) {
      setErrorMessage('Prompt cannot be empty. Please enter a description.');
      return;
    }
    if (prompt.length > 4000) {
      setErrorMessage('Prompt exceeds maximum length of 4000 characters.');
      return;
    }
    if (mode === 'img2img' && !source_image_id) {
      setErrorMessage('Image-to-Image mode requires an uploaded source image.');
      return;
    }
    if (mode === 'inpaint') {
      if (!source_image_id || !mask_image_id) {
        setErrorMessage('Inpainting requires both a source image and a painted mask.');
        return;
      }
    }
    if (mode === 'controlnet') {
      if (!source_image_id) {
        setErrorMessage('ControlNet mode requires an input guide image.');
        return;
      }
      if (!controlnet_model || availableControlNetModels.length === 0) {
        setErrorMessage(
          'ControlNet model unavailable — place a controlnet model (e.g. control_v11p_sd15_canny.safetensors) in ComfyUI/models/controlnet/ to enable ControlNet generation.'
        );
        return;
      }
    }
    if (width < 256 || width > 1536 || height < 256 || height > 1536) {
      setErrorMessage('Dimensions must stay within 256 and 1536 pixels for stable VRAM execution.');
      return;
    }
    if (loras.length > 2) {
      setErrorMessage('Forge Studio MVP permits a maximum of 2 active LoRAs to ensure memory stability.');
      return;
    }

    onGenerate();
  };

  const handleCopySeed = () => {
    if (seed === null) return;
    navigator.clipboard.writeText(String(seed));
    setCopiedSeed(true);
    setTimeout(() => setCopiedSeed(false), 2000);
  };

  const isSDXL =
    model.toLowerCase().includes('sd_xl') ||
    model.toLowerCase().includes('sdxl') ||
    model.toLowerCase().includes('pony') ||
    model.toLowerCase().includes('juggernaut');

  const dimensionPresets = isSDXL
    ? [
        { label: '1:1 Square', w: 1024, h: 1024 },
        { label: '3:4 Portrait', w: 896, h: 1152 },
        { label: '4:3 Landscape', w: 1152, h: 896 },
        { label: '9:16 Story', w: 768, h: 1344 },
        { label: '16:9 Cinema', w: 1344, h: 768 },
      ]
    : [
        { label: '1:1 Square', w: 512, h: 512 },
        { label: '2:3 Portrait', w: 512, h: 768 },
        { label: '3:2 Landscape', w: 768, h: 512 },
        { label: '9:16 Story', w: 512, h: 896 },
        { label: '16:9 Cinema', w: 896, h: 512 },
      ];
  const baseVram = isSDXL
    ? 6400 + 3.2e-6 * width * height + 500 * (loras?.length || 0)
    : 2600 + 1.8e-6 * width * height + 200 * (loras?.length || 0);
  const cnetVram = controlnet_model && mode === 'controlnet' ? 1600 : 0;
  const estimatedVramMb = Math.round((baseVram + cnetVram + 600) * 1.20);
  const estimatedDurationSec = (steps * (isSDXL ? 0.12 : 0.05)).toFixed(1);
  const totalPhysicalVramMb = telemetry?.total_vram_mb || 16303;
  const policyLimitPct = vramPolicy === 'max_quality' ? 92 : vramPolicy === 'max_speed' ? 78 : vramPolicy === 'conservative' ? 65 : 85;
  const policyLimitMb = Math.round(totalPhysicalVramMb * (policyLimitPct / 100));
  const isOverPolicy = estimatedVramMb > policyLimitMb;

  return (
    <div className="space-y-6">
      {/* Img2Img Source Upload (Conditional) */}
      {mode === 'img2img' && (
        <div className="p-4 rounded-xl bg-surface-elevated border border-border-subtle space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-text-muted flex items-center gap-1.5">
              <Upload className="w-3.5 h-3.5 text-accent" />
              Source Image Ingestion
            </span>
            {source_image_preview && (
              <button
                onClick={() => setSourceImage(null, null)}
                className="text-xs text-text-dim hover:text-status-error flex items-center gap-1 transition-colors"
              >
                <X className="w-3 h-3" /> Remove
              </button>
            )}
          </div>

          {source_image_preview ? (
            <div className="relative rounded-lg overflow-hidden border border-border-subtle max-h-52 bg-background flex items-center justify-center p-1">
              <img
                src={source_image_preview}
                alt="Source preview"
                className="max-h-48 object-contain rounded"
              />
            </div>
          ) : (
            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-border-strong rounded-lg p-6 text-center cursor-pointer hover:border-accent/60 transition-colors bg-surface/40 hover:bg-surface-hover/30"
            >
              <Upload className="w-6 h-6 mx-auto text-accent mb-2" />
              <p className="text-xs text-text-primary font-medium">Click to select source image</p>
              <p className="text-[11px] text-text-dim">Supports PNG, JPG, or WEBP up to 20MB</p>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            className="hidden"
          />

          {/* Denoise slider */}
          <div className="space-y-1.5 pt-2 border-t border-border-subtle">
            <div className="flex justify-between text-xs">
              <span className="text-text-muted">Denoising Strength</span>
              <span className="font-mono text-text-primary font-medium">{denoise.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={denoise}
              onChange={(e) => setDenoise(parseFloat(e.target.value))}
              className="w-full accent-accent cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-text-dim">
              <span>Subtle variation (0.3)</span>
              <span>Reconstruction (0.75)</span>
              <span>Full redesign (0.95)</span>
            </div>
          </div>
        </div>
      )}

      {/* Inpaint Mode Section */}
      {mode === 'inpaint' && (
        <div className="space-y-4">
          <InpaintCanvas />

          <div className="p-4 rounded-xl bg-surface-elevated border border-border-subtle space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-text-muted font-medium">Inpaint Denoising Strength</span>
              <span className="font-mono text-accent font-semibold">{denoise.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={denoise}
              onChange={(e) => setDenoise(parseFloat(e.target.value))}
              className="w-full accent-accent cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-text-dim">
              <span>Subtle texture edit (0.45)</span>
              <span>Object replacement (0.85)</span>
              <span>Total redraw (1.00)</span>
            </div>
          </div>
        </div>
      )}

      {/* ControlNet Mode Section */}
      {mode === 'controlnet' && (
        <div className="p-4 rounded-xl bg-surface-elevated border border-border-subtle space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-text-muted flex items-center gap-1.5">
              <Target className="w-3.5 h-3.5 text-accent" />
              ControlNet Guide Image & Conditioning
            </span>
            {source_image_preview && (
              <button
                onClick={() => setSourceImage(null, null)}
                className="text-xs text-text-dim hover:text-status-error flex items-center gap-1 transition-colors"
              >
                <X className="w-3 h-3" /> Remove
              </button>
            )}
          </div>

          {source_image_preview ? (
            <div className="relative rounded-lg overflow-hidden border border-border-subtle max-h-52 bg-background flex items-center justify-center p-1">
              <img
                src={source_image_preview}
                alt="ControlNet guide preview"
                className="max-h-48 object-contain rounded"
              />
            </div>
          ) : (
            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-border-strong rounded-lg p-6 text-center cursor-pointer hover:border-accent/60 transition-colors bg-surface/40 hover:bg-surface-hover/30"
            >
              <Upload className="w-6 h-6 mx-auto text-accent mb-2" />
              <p className="text-xs text-text-primary font-medium">Click to select guide image</p>
              <p className="text-[11px] text-text-dim">Edges or poses will be extracted from this image</p>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            className="hidden"
          />

          {/* ControlNet Type Selector */}
          <div className="space-y-1.5">
            <label className="text-xs text-text-muted font-medium">ControlNet Type</label>
            <div className="grid grid-cols-3 sm:grid-cols-5 gap-1.5">
              {[
                { id: 'canny', label: 'Canny' },
                { id: 'depth', label: 'Depth' },
                { id: 'openpose', label: 'OpenPose' },
                { id: 'lineart', label: 'Line Art' },
                { id: 'soft_edge', label: 'Soft Edge' },
              ].map((t) => (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => {
                    setControlNetType(t.id);
                    setControlNetPreprocessor(t.id);
                  }}
                  className={`py-1.5 px-2 rounded-lg text-xs font-medium transition-all ${
                    controlnet_type === t.id
                      ? 'bg-accent text-background font-semibold shadow-sm'
                      : 'bg-surface text-text-muted hover:text-text-primary border border-border-subtle'
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>

          {/* ControlNet Model Dropdown */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-text-muted font-medium">ControlNet Model</span>
              <span className="text-[10px] text-text-dim">
                {availableControlNetModels.length} detected
              </span>
            </div>

            {availableControlNetModels.length === 0 ? (
              <div className="p-3 rounded-lg bg-status-error/10 border border-status-error/30 text-xs text-status-error space-y-1">
                <div className="flex items-center gap-1.5 font-semibold">
                  <AlertCircle className="w-3.5 h-3.5" />
                  ControlNet model unavailable
                </div>
                <p className="text-[11px] opacity-90">
                  Place a controlnet model (e.g. control_v11p_sd15_canny.safetensors) in ComfyUI/models/controlnet/ to enable ControlNet generation.
                </p>
              </div>
            ) : (
              <select
                value={controlnet_model}
                onChange={(e) => setControlNetModel(e.target.value)}
                className="w-full bg-surface text-text-primary text-xs font-mono rounded-lg p-2.5 border border-border-subtle focus:border-accent focus:outline-none"
              >
                {availableControlNetModels.map((m) => (
                  <option key={m.filename} value={m.filename}>
                    {m.name} ({m.type.toUpperCase()})
                  </option>
                ))}
              </select>
            )}
          </div>

          {/* Strength & Percentage Sliders */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 border-t border-border-subtle">
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-text-muted">Conditioning Strength</span>
                <span className="font-mono text-accent font-medium">{controlnet_strength.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="2.0"
                step="0.05"
                value={controlnet_strength}
                onChange={(e) => setControlNetStrength(parseFloat(e.target.value))}
                className="w-full accent-accent cursor-pointer"
              />
            </div>

            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-text-muted">Step Range</span>
                <span className="font-mono text-text-primary text-[11px]">
                  {Math.round(controlnet_start_percent * 100)}% – {Math.round(controlnet_end_percent * 100)}%
                </span>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={controlnet_start_percent}
                  onChange={(e) => setControlNetStartPercent(parseFloat(e.target.value))}
                  className="w-full accent-accent cursor-pointer"
                  title="Start step percentage"
                />
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={controlnet_end_percent}
                  onChange={(e) => setControlNetEndPercent(parseFloat(e.target.value))}
                  className="w-full accent-accent cursor-pointer"
                  title="End step percentage"
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Model & Checkpoint */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <label className="text-xs font-semibold uppercase tracking-wider text-text-muted">
            Base Model Checkpoint
          </label>
          <span className="text-[10px] font-mono text-text-dim">
            {isSDXL ? 'SDXL Architecture (Native 1024px)' : 'SD 1.5 Architecture (Native 512px)'}
          </span>
        </div>
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="w-full bg-surface-elevated text-text-primary text-xs font-mono rounded-lg p-2.5 border border-border-subtle focus:border-accent focus:outline-none"
        >
          {availableModels.length > 0 ? (
            availableModels.map((m) => (
              <option key={m.filename} value={m.filename}>
                {m.name} ({m.filename}) {m.size_mb > 0 ? `[${Math.round(m.size_mb / 1024)}GB]` : ''}
              </option>
            ))
          ) : (
            <option value="" disabled>No checkpoints detected in ComfyUI</option>
          )}
        </select>
        {availableModels.length === 0 && (
          <div className="p-2.5 rounded-lg bg-status-warning/15 border border-status-warning/30 text-status-warning text-xs flex items-center gap-2 mt-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>No checkpoint found. Place a model file in ComfyUI/models/checkpoints/ to generate.</span>
          </div>
        )}
      </div>

      {/* LoRA Stack (Up to 2 for MVP) */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-xs font-semibold uppercase tracking-wider text-text-muted flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-accent" />
            LoRA Stacking ({loras.length}/2)
          </label>
          {availableLoras.length > 0 && loras.length < 2 && (
            <select
              onChange={(e) => {
                if (e.target.value) {
                  addLora({ name: e.target.value, strength: 0.8 });
                  e.target.value = '';
                }
              }}
              className="text-xs bg-surface-elevated text-accent border border-border-subtle rounded px-2 py-1 focus:outline-none font-medium hover:border-accent/40"
            >
              <option value="">+ Attach LoRA</option>
              {availableLoras
                .filter((l) => !loras.some((active) => active.name === l.filename))
                .map((l) => (
                  <option key={l.filename} value={l.filename}>
                    {l.name}
                  </option>
                ))}
            </select>
          )}
        </div>

        {loras.length === 0 ? (
          <p className="text-xs text-text-dim italic">
            No LoRAs attached. Synthesizing directly with base checkpoint weights.
          </p>
        ) : (
          <div className="space-y-2.5">
            {loras.map((l) => {
              const vaultItem = loraVault.find(
                (v) => v.filename === l.name || v.name === l.name
              );
              const displayName = vaultItem?.name || l.name;
              const triggerWords = vaultItem?.trigger_words || [];

              return (
                <div
                  key={l.name}
                  className="p-3 rounded-xl bg-surface-elevated border border-border-subtle space-y-2"
                >
                  <div className="flex items-center justify-between text-xs">
                    <div className="min-w-0 pr-2">
                      <span className="font-semibold text-text-primary truncate block" title={displayName}>
                        {displayName}
                      </span>
                      {vaultItem && vaultItem.name !== l.name && (
                        <span className="font-mono text-[10px] text-text-dim truncate block" title={l.name}>
                          {l.name}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center space-x-2 shrink-0">
                      <span className="font-mono text-accent text-[11px] font-semibold">
                        {l.strength.toFixed(2)}x
                      </span>
                      <button
                        onClick={() => removeLora(l.name)}
                        className="text-text-dim hover:text-status-error p-0.5 transition-colors"
                        title="Detach LoRA"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  <input
                    type="range"
                    min="0.0"
                    max="2.0"
                    step="0.05"
                    value={l.strength}
                    onChange={(e) => updateLoraStrength(l.name, parseFloat(e.target.value))}
                    className="w-full accent-accent cursor-pointer"
                  />

                  {/* Quick preset buttons */}
                  <div className="flex items-center justify-between text-[10px] font-mono text-text-dim pt-0.5">
                    <span className="font-sans">Presets:</span>
                    <div className="flex gap-1.5">
                      {[0.5, 0.75, 1.0, 1.25].map((val) => (
                        <button
                          key={val}
                          type="button"
                          onClick={() => updateLoraStrength(l.name, val)}
                          className={`px-1.5 py-0.5 rounded border ${
                            Math.abs(l.strength - val) < 0.01
                              ? 'bg-accent/20 border-accent/40 text-accent font-bold'
                              : 'bg-surface border-border-subtle text-text-dim hover:text-text-primary'
                          }`}
                        >
                          {val}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Trigger Words Quick Insertion */}
                  {triggerWords.length > 0 && (
                    <div className="pt-1.5 border-t border-border-subtle/50 flex items-center justify-between text-[10px]">
                      <span className="text-text-dim truncate max-w-[170px]" title={triggerWords.join(', ')}>
                        Triggers: {triggerWords.join(', ')}
                      </span>
                      <button
                        type="button"
                        onClick={() => {
                          const triggersToAdd = triggerWords.filter(
                            (tw) => !prompt.toLowerCase().includes(tw.toLowerCase())
                          );
                          if (triggersToAdd.length > 0) {
                            const addition = triggersToAdd.join(', ');
                            setPrompt(prompt ? `${prompt.trim()}, ${addition}` : addition);
                          }
                        }}
                        className="text-[10px] text-accent hover:underline flex items-center gap-1 font-medium shrink-0 ml-1"
                      >
                        + Add to Prompt
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Resolution & Aspect Ratio Presets */}
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <label className="text-xs font-semibold uppercase tracking-wider text-text-muted">
            Dimensions
          </label>
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs text-accent font-medium">
              {width} × {height}
            </span>
            <button
              type="button"
              onClick={() => setCustomDimOpen(!customDimOpen)}
              className="text-[11px] text-text-dim hover:text-accent flex items-center gap-1"
            >
              <Settings2 className="w-3 h-3" />
              {customDimOpen ? 'Presets' : 'Custom'}
            </button>
          </div>
        </div>

        {customDimOpen ? (
          <div className="p-3 rounded-xl bg-surface-elevated border border-border-subtle grid grid-cols-2 gap-3 text-xs">
            <div>
              <span className="text-[10px] text-text-dim uppercase font-mono block mb-1">Width (px)</span>
              <input
                type="number"
                min={256}
                max={1536}
                step={64}
                value={width}
                onChange={(e) => setDimensions(parseInt(e.target.value) || 1024, height)}
                className="w-full bg-surface text-text-primary font-mono p-2 rounded border border-border-subtle focus:border-accent focus:outline-none"
              />
            </div>
            <div>
              <span className="text-[10px] text-text-dim uppercase font-mono block mb-1">Height (px)</span>
              <input
                type="number"
                min={256}
                max={1536}
                step={64}
                value={height}
                onChange={(e) => setDimensions(width, parseInt(e.target.value) || 1024)}
                className="w-full bg-surface text-text-primary font-mono p-2 rounded border border-border-subtle focus:border-accent focus:outline-none"
              />
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-3 gap-2">
            {dimensionPresets.map((preset) => {
              const isMatch = width === preset.w && height === preset.h;
              return (
                <button
                  key={preset.label}
                  type="button"
                  onClick={() => setDimensions(preset.w, preset.h)}
                  className={`py-2 px-2 rounded-lg text-xs font-mono transition-all border text-center ${
                    isMatch
                      ? 'bg-accent/15 border-accent text-accent font-semibold'
                      : 'bg-surface-elevated border-border-subtle text-text-muted hover:text-text-primary'
                  }`}
                >
                  <div className="text-[11px] font-sans font-medium">{preset.label}</div>
                  <div className="text-[10px] text-text-dim mt-0.5">{preset.w}×{preset.h}</div>
                </button>
              );
            })}
          </div>
        )}

        {/* Warning if SD 1.5 model is used above safe resolution */}
        {!isSDXL && (width > 768 || height > 768) && (
          <div className="p-2.5 rounded-lg bg-status-warning/15 border border-status-warning/30 text-status-warning text-xs flex items-center justify-between gap-2 mt-2">
            <div className="flex items-center gap-1.5 min-w-0">
              <AlertCircle className="w-4 h-4 shrink-0 text-status-warning" />
              <span>
                SD 1.5 native resolution is 512px. Generating at {width}×{height} often creates duplicate subjects or floating limbs.
              </span>
            </div>
            <button
              type="button"
              onClick={() => setDimensions(512, 512)}
              className="px-2 py-1 bg-status-warning/25 hover:bg-status-warning/35 text-status-warning font-semibold rounded text-[11px] whitespace-nowrap transition-colors border border-status-warning/40 shrink-0"
            >
              Fix to 512×512
            </button>
          </div>
        )}
      </div>

      {/* Sampling Parameters: Steps & CFG */}
      <div className="grid grid-cols-2 gap-4">
        {/* Steps */}
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs">
            <span className="text-text-muted">Steps</span>
            <span className="font-mono text-text-primary font-medium">{steps}</span>
          </div>
          <input
            type="range"
            min="10"
            max="60"
            step="1"
            value={steps}
            onChange={(e) => setSteps(parseInt(e.target.value))}
            className="w-full accent-accent cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-text-dim">
            <span>Fast (20)</span>
            <span>Balanced (28)</span>
            <span>Fine (40)</span>
          </div>
        </div>

        {/* CFG Scale */}
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs">
            <span className="text-text-muted">CFG Scale</span>
            <span className="font-mono text-text-primary font-medium">{cfg_scale.toFixed(1)}</span>
          </div>
          <input
            type="range"
            min="1.0"
            max="15.0"
            step="0.5"
            value={cfg_scale}
            onChange={(e) => setCfgScale(parseFloat(e.target.value))}
            className="w-full accent-accent cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-text-dim">
            <span>Creative (4.5)</span>
            <span>Default (6.5)</span>
            <span>Strict (9.0)</span>
          </div>
        </div>
      </div>

      {/* Sampler & Seed */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-1.5">
          <label className="text-xs text-text-muted font-medium">Sampling Method</label>
          <select
            value={sampler}
            onChange={(e) => setSampler(e.target.value)}
            className="w-full bg-surface-elevated text-text-primary text-xs font-mono rounded-lg p-2.5 border border-border-subtle focus:border-accent focus:outline-none"
          >
            {samplers.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>
        </div>

        {/* Seed Controls */}
        <div className="space-y-1.5">
          <div className="flex justify-between items-center text-xs">
            <span className="text-text-muted font-medium">Generation Seed</span>
            <div className="flex items-center gap-1.5">
              {seed !== null && (
                <button
                  type="button"
                  onClick={handleCopySeed}
                  className="text-[10px] text-text-dim hover:text-accent flex items-center gap-0.5"
                  title="Copy seed to clipboard"
                >
                  {copiedSeed ? <Check className="w-3 h-3 text-status-success" /> : <Copy className="w-3 h-3" />}
                </button>
              )}
              <button
                type="button"
                onClick={() => setSeed(null)}
                className={`text-[10px] uppercase font-mono px-1.5 py-0.5 rounded ${
                  seed === null
                    ? 'bg-accent/20 text-accent font-semibold'
                    : 'bg-surface border border-border-subtle text-text-dim hover:text-text-primary'
                }`}
              >
                Randomize
              </button>
            </div>
          </div>
          <div className="relative">
            <input
              type="number"
              placeholder="Randomized each run"
              value={seed === null ? '' : seed}
              onChange={(e) => {
                const val = e.target.value === '' ? null : Math.max(0, parseInt(e.target.value) || 0);
                setSeed(val);
              }}
              className="w-full bg-surface-elevated text-text-primary text-xs font-mono rounded-lg p-2.5 pr-8 border border-border-subtle focus:border-accent focus:outline-none placeholder:text-text-dim"
            />
            <button
              type="button"
              onClick={() => setSeed(Math.floor(Math.random() * 4294967295))}
              className="absolute right-2.5 top-2.5 text-text-dim hover:text-accent transition-colors"
              title="Roll fresh random seed"
            >
              <Dice5 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Queue Priority & VRAM Safety Estimation */}
      <div className="p-3.5 rounded-xl bg-surface-elevated border border-border-subtle space-y-3">
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold uppercase tracking-wider text-text-muted flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5 text-accent" />
            Queue Priority & VRAM Estimate
          </span>
          <span className="font-mono text-[11px] text-text-dim">
            Policy: <strong className="text-text-primary capitalize">{vramPolicy.replace('_', ' ')}</strong>
          </span>
        </div>

        <div className="grid grid-cols-2 gap-3 items-center">
          {/* Priority Selector */}
          <div className="flex items-center space-x-1 bg-surface p-1 rounded-lg border border-border-subtle">
            {[
              { id: 1, label: 'High' },
              { id: 2, label: 'Normal' },
              { id: 3, label: 'Low' },
            ].map((p) => (
              <button
                key={p.id}
                type="button"
                onClick={() => setPriority(p.id)}
                className={`flex-1 py-1.5 px-2 rounded-md text-[11px] font-mono font-medium transition-all ${
                  priority === p.id
                    ? 'bg-surface-elevated text-accent font-semibold shadow-sm border border-border-subtle'
                    : 'text-text-muted hover:text-text-primary'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>

          {/* VRAM & Duration Estimates */}
          <div className="flex items-center justify-end space-x-2 font-mono text-[11px]">
            <div className="text-right">
              <div className="flex items-center justify-end gap-1">
                <Cpu className="w-3 h-3 text-accent" />
                <span className={isOverPolicy ? 'text-status-warning font-semibold' : 'text-text-primary'}>
                  {(estimatedVramMb / 1024).toFixed(1)} GB
                </span>
                <span className="text-[10px] text-text-dim">est.</span>
              </div>
              <div className="flex items-center justify-end gap-1 text-[10px] text-text-muted">
                <Clock className="w-2.5 h-2.5 text-text-dim" />
                <span>~{estimatedDurationSec}s</span>
              </div>
            </div>
          </div>
        </div>

        {isOverPolicy && (
          <div className="p-2 rounded-lg bg-status-warning/15 border border-status-warning/30 text-[11px] text-status-warning flex items-center gap-1.5">
            <ShieldAlert className="w-3.5 h-3.5 shrink-0" />
            <span>Estimated VRAM ({(estimatedVramMb / 1024).toFixed(1)} GB) exceeds {vramPolicy} limit ({policyLimitPct}%). Generation may experience paging or swap.</span>
          </div>
        )}
      </div>

      {/* Settings Summary Widget */}
      <SettingsSummary />

      {/* Action Button: Generate or Cancel */}
      <div className="pt-2">
        {isGenerating ? (
          <button
            onClick={onCancel}
            type="button"
            aria-label="Cancel active image generation"
            className="w-full py-3.5 px-4 rounded-xl font-medium text-sm transition-all bg-status-error/20 text-status-error hover:bg-status-error/30 border border-status-error/40 flex items-center justify-center space-x-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-status-error"
          >
            <Square className="w-4 h-4 fill-current" />
            <span>Cancel Active Generation ({progress?.progress_percentage ?? 0}%)</span>
          </button>
        ) : (
          <button
            onClick={validateAndGenerate}
            disabled={
              isSubmitting ||
              (mode === 'img2img' && !source_image_id) ||
              (mode === 'inpaint' && (!source_image_id || !mask_image_id)) ||
              (mode === 'controlnet' && (!source_image_id || !controlnet_model || availableControlNetModels.length === 0))
            }
            type="button"
            aria-label="Submit and generate image (Ctrl+Enter)"
            className={`w-full py-3.5 px-4 rounded-xl font-semibold text-sm transition-all shadow-lg flex items-center justify-center space-x-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
              isSubmitting ||
              (mode === 'img2img' && !source_image_id) ||
              (mode === 'inpaint' && (!source_image_id || !mask_image_id)) ||
              (mode === 'controlnet' && (!source_image_id || !controlnet_model || availableControlNetModels.length === 0))
                ? 'bg-surface-elevated text-text-dim border border-border-subtle cursor-not-allowed opacity-70'
                : 'bg-accent text-background hover:bg-accent-hover shadow-accent/10'
            }`}
          >
            {isSubmitting ? (
              <>
                <div className="w-4 h-4 border-2 border-accent border-t-transparent rounded-full animate-spin" />
                <span>Submitting Generation...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>
                  {mode === 'img2img' && !source_image_id
                    ? 'Select Source Image to Generate'
                    : mode === 'inpaint' && !source_image_id
                    ? 'Upload Image to Inpaint'
                    : mode === 'inpaint' && !mask_image_id
                    ? 'Paint Mask to Inpaint'
                    : mode === 'controlnet' && !source_image_id
                    ? 'Select Guide Image for ControlNet'
                    : mode === 'controlnet' && (!controlnet_model || availableControlNetModels.length === 0)
                    ? 'ControlNet Model Unavailable'
                    : 'Generate Image'}
                </span>
                <span className="text-[11px] opacity-70 font-mono ml-2">(Ctrl+Enter)</span>
              </>
            )}
          </button>
        )}
      </div>
    </div>
  );
};
