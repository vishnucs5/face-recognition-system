import React, { useEffect, useState } from 'react';
import {
  RotateCcw,
  ArrowRightCircle,
  Download,
  Calendar,
  Layers,
  Sparkles,
  Search,
  X,
  Trash2,
  Copy,
  Check,
  Clock,
  Sliders,
  ExternalLink,
  FileJson,
  AlertTriangle,
  RefreshCw,
  Ban,
  Paintbrush,
  Target,
  Bookmark,
} from 'lucide-react';
import { useStudioStore } from '../store/useStudioStore';
import { GenerationResponse } from '../types';

export const GalleryView: React.FC = () => {
  const { reuseSettings, sendToImg2Img, sendToInpaint, sendToControlNet, setActiveNav } = useStudioStore();
  const [generations, setGenerations] = useState<GenerationResponse[]>([]);
  const [selected, setSelected] = useState<GenerationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [modeFilter, setModeFilter] = useState<'all' | 'txt2img' | 'img2img' | 'inpaint' | 'controlnet'>('all');
  const [copiedPrompt, setCopiedPrompt] = useState(false);
  const [copiedNegative, setCopiedNegative] = useState(false);
  const [copyPresetOpen, setCopyPresetOpen] = useState(false);
  const [copyPresetName, setCopyPresetName] = useState('');
  const [copyPresetSuccess, setCopyPresetSuccess] = useState(false);

  useEffect(() => {
    fetchGenerations();
  }, [searchQuery, modeFilter]);

  const fetchGenerations = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ limit: '60' });
      if (searchQuery.trim()) params.append('search', searchQuery.trim());
      if (modeFilter !== 'all') params.append('mode', modeFilter);

      const res = await fetch(`/api/generations?${params.toString()}`);
      if (res.ok) {
        const data: GenerationResponse[] = await res.json();
        setGenerations(data);
        if (data.length > 0 && (!selected || !data.some((d) => d.id === selected.id))) {
          setSelected(data[0]);
        }
      }
    } catch (err) {
      console.error('Failed to fetch gallery items:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = (gen: GenerationResponse) => {
    if (!gen.image_url) return;
    const a = document.createElement('a');
    a.href = gen.image_url;
    a.download = `forge_${gen.id.slice(0, 8)}_${gen.seed ?? 'rnd'}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleDownloadMetadata = (gen: GenerationResponse) => {
    const metadata = {
      prompt: gen.prompt,
      negative_prompt: gen.negative_prompt,
      checkpoint: gen.model,
      loras: gen.loras,
      width: gen.width,
      height: gen.height,
      steps: gen.steps,
      cfg_scale: gen.cfg_scale,
      sampler: gen.sampler,
      seed: gen.seed,
      denoise: gen.denoise,
      source_mode: gen.mode,
      mask_image_path: gen.mask_image_path,
      controlnet_type: gen.controlnet_type,
      controlnet_model: gen.controlnet_model,
      controlnet_strength: gen.controlnet_strength,
      controlnet_start_percent: gen.controlnet_start_percent,
      controlnet_end_percent: gen.controlnet_end_percent,
      controlnet_preprocessor: gen.controlnet_preprocessor,
      duration_ms: gen.duration_ms,
      timestamp: gen.created_at,
      completed_at: gen.completed_at,
      comfy_prompt_id: gen.comfy_prompt_id,
    };
    const blob = new Blob([JSON.stringify(metadata, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `forge_${gen.id.slice(0, 8)}_metadata.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleDelete = async (gen: GenerationResponse) => {
    if (!confirm('Are you sure you want to delete this generation from your local history?')) {
      return;
    }
    try {
      const res = await fetch(`/api/generations/${gen.id}`, { method: 'DELETE' });
      if (res.ok) {
        setGenerations((prev) => prev.filter((g) => g.id !== gen.id));
        if (selected?.id === gen.id) {
          const remaining = generations.filter((g) => g.id !== gen.id);
          setSelected(remaining.length > 0 ? remaining[0] : null);
        }
      }
    } catch (err) {
      console.error('Failed to delete generation:', err);
    }
  };

  const copyToClipboard = (text: string, type: 'prompt' | 'negative') => {
    navigator.clipboard.writeText(text);
    if (type === 'prompt') {
      setCopiedPrompt(true);
      setTimeout(() => setCopiedPrompt(false), 2000);
    } else {
      setCopiedNegative(true);
      setTimeout(() => setCopiedNegative(false), 2000);
    }
  };

  return (
    <div className="flex-1 flex overflow-hidden select-none bg-background">
      {/* 1. Main Gallery Grid Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Gallery Toolbar: Title, Search, and Mode Filter */}
        <div className="p-6 border-b border-border-subtle bg-surface flex flex-col md:flex-row md:items-center justify-between gap-4 shrink-0">
          <div>
            <h2 className="text-lg font-semibold text-text-primary tracking-tight flex items-center gap-2">
              Studio Gallery & History
              <span className="text-[11px] font-mono font-normal px-2 py-0.5 rounded bg-surface-elevated text-accent border border-border-subtle">
                {generations.length} records
              </span>
            </h2>
            <p className="text-xs text-text-muted mt-0.5">
              Reproducible SQLite archives with embedded prompt metadata
            </p>
          </div>

          <div className="flex items-center gap-3">
            {/* Search Input */}
            <div className="relative w-56 md:w-64">
              <Search className="w-3.5 h-3.5 absolute left-3 top-3 text-text-dim" />
              <input
                type="text"
                placeholder="Filter by prompt keyword..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-surface-elevated text-text-primary text-xs rounded-lg pl-8 pr-7 py-2 border border-border-subtle focus:border-accent focus:outline-none placeholder:text-text-dim"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  className="absolute right-2.5 top-2.5 text-text-dim hover:text-text-primary"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            {/* Mode Filter Selector */}
            <div className="flex items-center bg-surface-elevated p-1 rounded-lg border border-border-subtle text-xs">
              {(['all', 'txt2img', 'img2img', 'inpaint', 'controlnet'] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setModeFilter(m)}
                  className={`px-2 py-1 rounded capitalize font-medium transition-colors ${
                    modeFilter === m
                      ? 'bg-surface text-accent font-semibold shadow-sm'
                      : 'text-text-muted hover:text-text-primary'
                  }`}
                >
                  {m === 'all'
                    ? 'All'
                    : m === 'txt2img'
                    ? 'Txt2Img'
                    : m === 'img2img'
                    ? 'Img2Img'
                    : m === 'inpaint'
                    ? 'Inpaint'
                    : 'ControlNet'}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Image Grid Container */}
        <div className="flex-1 p-6 overflow-y-auto">
          {loading ? (
            <div className="text-center py-24 text-text-dim text-xs flex flex-col items-center gap-2">
              <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
              <span>Querying local generation archives...</span>
            </div>
          ) : generations.length === 0 ? (
            <div className="text-center py-24 max-w-sm mx-auto space-y-4">
              <div className="w-16 h-16 rounded-2xl bg-surface-elevated border border-border-subtle mx-auto flex items-center justify-center text-accent">
                <Sparkles className="w-8 h-8 opacity-60" />
              </div>
              <div className="space-y-1">
                <h3 className="text-sm font-semibold text-text-primary">
                  {searchQuery ? 'No Matching Generations Found' : 'No Generations in Vault'}
                </h3>
                <p className="text-xs text-text-muted leading-relaxed">
                  {searchQuery
                    ? `No prompt matches "${searchQuery}". Try clearing search filters.`
                    : 'Execute a generation in Forge Studio to persist outputs and reproduction recipes to SQLite.'}
                </p>
              </div>
              {searchQuery ? (
                <button
                  onClick={() => {
                    setSearchQuery('');
                    setModeFilter('all');
                  }}
                  className="px-4 py-2 rounded-lg bg-surface-elevated hover:bg-surface-hover text-accent border border-border-subtle text-xs font-medium"
                >
                  Clear Filters
                </button>
              ) : (
                <button
                  onClick={() => setActiveNav('generate')}
                  className="px-4 py-2 rounded-lg bg-accent text-background text-xs font-semibold hover:bg-accent-hover transition-colors shadow"
                >
                  Create Your First Generation
                </button>
              )}
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
              {generations.map((gen) => {
                const isSelected = selected?.id === gen.id;
                return (
                  <div
                    key={gen.id}
                    onClick={() => setSelected(gen)}
                    className={`group relative rounded-xl overflow-hidden border cursor-pointer transition-all aspect-square bg-surface-elevated shadow-sm ${
                      isSelected
                        ? 'border-accent shadow-lg shadow-accent/15 ring-2 ring-accent'
                        : 'border-border-subtle hover:border-border-strong hover:shadow-md'
                    }`}
                  >
                    {gen.image_url ? (
                      <img
                        src={gen.image_url}
                        alt={gen.prompt}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                        loading="lazy"
                      />
                    ) : gen.status === 'failed' ? (
                      <div className="w-full h-full flex flex-col items-center justify-center text-status-error text-xs font-mono p-2 text-center bg-status-error/5">
                        <AlertTriangle className="w-6 h-6 text-status-error mb-1.5" />
                        <span className="font-semibold text-[11px]">Failed</span>
                      </div>
                    ) : gen.status === 'cancelled' ? (
                      <div className="w-full h-full flex flex-col items-center justify-center text-text-dim text-xs font-mono p-2 text-center">
                        <Ban className="w-6 h-6 text-text-dim mb-1.5" />
                        <span className="text-[11px]">Canceled</span>
                      </div>
                    ) : (
                      <div className="w-full h-full flex flex-col items-center justify-center text-accent text-xs font-mono p-2 text-center">
                        <Sparkles className="w-6 h-6 text-accent mb-1.5 animate-spin" />
                        <span className="font-semibold text-[11px]">Running</span>
                      </div>
                    )}

                    {/* Status / Mode tag pill */}
                    <div
                      className={`absolute top-2 left-2 px-1.5 py-0.5 rounded text-[9px] font-mono backdrop-blur-sm border uppercase ${
                        gen.status === 'completed'
                          ? 'bg-background/80 text-text-muted border-border-subtle/60'
                          : gen.status === 'failed'
                          ? 'bg-status-error/90 text-background border-status-error font-bold'
                          : gen.status === 'cancelled'
                          ? 'bg-background/80 text-text-dim border-border-subtle'
                          : 'bg-accent text-background border-accent font-bold animate-pulse'
                      }`}
                    >
                      {gen.status === 'completed' ? gen.mode : gen.status}
                    </div>

                    {/* Hover detail overlay */}
                    <div className="absolute inset-0 bg-gradient-to-t from-background/90 via-background/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity p-3 flex flex-col justify-end">
                      <p className="text-[11px] text-text-primary line-clamp-2 font-medium">
                        {gen.prompt}
                      </p>
                      <div className="flex items-center justify-between mt-1 text-[10px] font-mono text-accent">
                        <span>{gen.width}×{gen.height}</span>
                        <span>{gen.steps} steps</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* 2. Selected Item Inspector Drawer (Persistent on right) */}
      {selected && (
        <aside className="w-96 bg-surface border-l border-border-subtle flex flex-col shrink-0 overflow-y-auto">
          {/* Top Inspector Bar */}
          <div className="p-4 border-b border-border-subtle flex items-center justify-between bg-surface-elevated/30">
            <span className="text-xs font-semibold uppercase tracking-wider text-text-muted">
              Metadata & Parameters
            </span>
            <div className="flex items-center space-x-1.5">
              <button
                onClick={() => reuseSettings(selected)}
                className="p-1.5 rounded-lg bg-surface-elevated border border-border-subtle text-text-muted hover:text-text-primary transition-colors"
                title="Load these settings into generation form"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
              {selected.image_url && (
                <button
                  onClick={() => sendToImg2Img(selected.image_url!)}
                  className="p-1.5 rounded-lg bg-surface-elevated border border-border-subtle text-text-muted hover:text-accent transition-colors"
                  title="Send to Image-to-Image"
                >
                  <ArrowRightCircle className="w-3.5 h-3.5" />
                </button>
              )}
              {selected.image_url && (
                <button
                  onClick={() => sendToInpaint(selected.image_url!)}
                  className="p-1.5 rounded-lg bg-surface-elevated border border-border-subtle text-text-muted hover:text-accent transition-colors"
                  title="Send to Inpainting"
                >
                  <Paintbrush className="w-3.5 h-3.5" />
                </button>
              )}
              {selected.image_url && (
                <button
                  onClick={() => sendToControlNet(selected.image_url!)}
                  className="p-1.5 rounded-lg bg-surface-elevated border border-border-subtle text-text-muted hover:text-accent transition-colors"
                  title="Send to ControlNet"
                >
                  <Target className="w-3.5 h-3.5" />
                </button>
              )}
              {selected.image_url && (
                <button
                  onClick={() => handleDownloadMetadata(selected)}
                  className="p-1.5 rounded-lg bg-surface-elevated border border-border-subtle text-text-muted hover:text-text-primary transition-colors"
                  title="Export generation metadata JSON"
                >
                  <FileJson className="w-3.5 h-3.5" />
                </button>
              )}
              {selected.image_url && (
                <button
                  onClick={() => handleDownload(selected)}
                  className="p-1.5 rounded-lg bg-surface-elevated border border-border-subtle text-text-muted hover:text-text-primary transition-colors"
                  title="Export PNG"
                >
                  <Download className="w-3.5 h-3.5" />
                </button>
              )}
              <button
                onClick={() => handleDelete(selected)}
                className="p-1.5 rounded-lg bg-surface-elevated border border-border-subtle text-text-dim hover:text-status-error transition-colors"
                title="Delete record"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          <div className="p-5 space-y-5 text-xs">
            {/* Full-view Image Canvas */}
            {selected.image_url && (
              <div className="rounded-xl overflow-hidden border border-border-subtle bg-background flex items-center justify-center shadow-inner">
                <img
                  src={selected.image_url}
                  alt={selected.prompt}
                  className="w-full max-h-72 object-contain"
                />
              </div>
            )}

            {/* Prompt with Copy Button */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-[10px] uppercase font-semibold text-text-dim">
                <span>Prompt</span>
                <button
                  onClick={() => copyToClipboard(selected.prompt, 'prompt')}
                  className="text-accent hover:underline flex items-center gap-1 font-sans"
                >
                  {copiedPrompt ? <Check className="w-3 h-3 text-status-success" /> : <Copy className="w-3 h-3" />}
                  <span>{copiedPrompt ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
              <p className="text-xs text-text-primary leading-relaxed bg-surface-elevated p-3 rounded-xl border border-border-subtle select-text">
                {selected.prompt}
              </p>
            </div>

            {/* Negative Prompt (if any) */}
            {selected.negative_prompt && (
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-[10px] uppercase font-semibold text-text-dim">
                  <span>Negative Prompt</span>
                  <button
                    onClick={() => copyToClipboard(selected.negative_prompt, 'negative')}
                    className="text-accent hover:underline flex items-center gap-1 font-sans"
                  >
                    {copiedNegative ? <Check className="w-3 h-3 text-status-success" /> : <Copy className="w-3 h-3" />}
                    <span>{copiedNegative ? 'Copied' : 'Copy'}</span>
                  </button>
                </div>
                <p className="text-xs text-text-muted leading-relaxed bg-surface-elevated p-3 rounded-xl border border-border-subtle select-text">
                  {selected.negative_prompt}
                </p>
              </div>
            )}

            {/* Failed Generation Callout Banner with Retry */}
            {selected.status === 'failed' && (
              <div className="p-3.5 rounded-xl bg-status-error/10 border border-status-error/30 space-y-2">
                <div className="flex items-center gap-2 text-status-error font-semibold text-xs">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span>Generation Failed</span>
                </div>
                <p className="text-[11px] text-text-muted leading-relaxed font-mono">
                  {selected.error_message || 'An error occurred during workflow execution in ComfyUI.'}
                </p>
                <button
                  onClick={() => {
                    reuseSettings(selected);
                    setActiveNav('generate');
                  }}
                  className="w-full py-2 px-3 rounded-lg font-semibold text-xs bg-status-error/20 text-status-error hover:bg-status-error/30 border border-status-error/40 transition-colors flex items-center justify-center space-x-1.5"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Load Parameters & Retry</span>
                </button>
              </div>
            )}

            {/* Cancelled Generation Notice */}
            {selected.status === 'cancelled' && (
              <div className="p-3 rounded-xl bg-surface-elevated border border-border-subtle flex items-center justify-between text-xs text-text-dim">
                <span className="flex items-center gap-1.5 font-mono">
                  <Ban className="w-3.5 h-3.5" />
                  <span>Generation was cancelled</span>
                </span>
                <button
                  onClick={() => {
                    reuseSettings(selected);
                    setActiveNav('generate');
                  }}
                  className="text-accent text-[11px] hover:underline"
                >
                  Retry in Studio
                </button>
              </div>
            )}

            {/* Reproducibility Grid */}
            <div className="space-y-2">
              <span className="text-[10px] font-semibold uppercase text-text-dim block">
                Technical Blueprint
              </span>
              <div className="grid grid-cols-2 gap-2 font-mono text-[11px]">
                <div className="p-2.5 rounded-lg bg-surface-elevated border border-border-subtle">
                  <span className="text-text-dim block text-[9px] uppercase font-sans">Mode & Status</span>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <span className="text-accent uppercase font-semibold text-[10px]">{selected.mode}</span>
                    <span className="text-text-dim">•</span>
                    <span className={`capitalize text-[10px] ${
                      selected.status === 'completed' ? 'text-status-success' :
                      selected.status === 'failed' ? 'text-status-error' :
                      selected.status === 'cancelled' ? 'text-text-dim' : 'text-accent'
                    }`}>{selected.status}</span>
                  </div>
                </div>

                <div className="p-2.5 rounded-lg bg-surface-elevated border border-border-subtle">
                  <span className="text-text-dim block text-[9px] uppercase font-sans">Model Checkpoint</span>
                  <span className="text-text-primary truncate block font-medium" title={selected.model}>{selected.model}</span>
                </div>

                <div className="p-2.5 rounded-lg bg-surface-elevated border border-border-subtle">
                  <span className="text-text-dim block text-[9px] uppercase font-sans">Resolution</span>
                  <span className="text-text-primary font-medium">{selected.width} × {selected.height}</span>
                </div>

                <div className="p-2.5 rounded-lg bg-surface-elevated border border-border-subtle">
                  <span className="text-text-dim block text-[9px] uppercase font-sans">Sampler & Steps</span>
                  <span className="text-text-primary font-medium">{selected.sampler} ({selected.steps})</span>
                </div>

                <div className="p-2.5 rounded-lg bg-surface-elevated border border-border-subtle">
                  <span className="text-text-dim block text-[9px] uppercase font-sans">Seed</span>
                  <span className="text-text-primary font-medium">{selected.seed ?? 'Random'}</span>
                </div>

                <div className="p-2.5 rounded-lg bg-surface-elevated border border-border-subtle">
                  <span className="text-text-dim block text-[9px] uppercase font-sans">CFG & Denoise</span>
                  <span className="text-text-primary font-medium">
                    CFG {selected.cfg_scale}{selected.mode === 'img2img' ? ` / Den ${selected.denoise}` : ''}
                  </span>
                </div>

                <div className="p-2.5 rounded-lg bg-surface-elevated border border-border-subtle">
                  <span className="text-text-dim block text-[9px] uppercase font-sans">Inference Time</span>
                  <span className="text-text-primary font-medium">
                    {selected.duration_ms ? `${(selected.duration_ms / 1000).toFixed(1)}s` : 'N/A'}
                  </span>
                </div>

                <div className="p-2.5 rounded-lg bg-surface-elevated border border-border-subtle">
                  <span className="text-text-dim block text-[9px] uppercase font-sans">ComfyUI Prompt ID</span>
                  <span className="text-accent truncate block font-medium" title={selected.comfy_prompt_id || 'Direct Job'}>
                    {selected.comfy_prompt_id ? selected.comfy_prompt_id.slice(0, 10) + '...' : 'Direct Job'}
                  </span>
                </div>

                {selected.mode === 'controlnet' && (
                  <div className="col-span-2 p-2.5 rounded-lg bg-surface-elevated border border-border-subtle space-y-1">
                    <span className="text-text-dim block text-[9px] uppercase font-sans">ControlNet Conditioning</span>
                    <div className="flex items-center justify-between text-accent font-medium">
                      <span>Type: {selected.controlnet_type?.toUpperCase() || 'CANNY'}</span>
                      <span>Str: {selected.controlnet_strength ?? 1.0}</span>
                      <span>Range: {Math.round((selected.controlnet_start_percent ?? 0) * 100)}%–{Math.round((selected.controlnet_end_percent ?? 1) * 100)}%</span>
                    </div>
                    {selected.controlnet_model && (
                      <span className="text-[10px] text-text-muted truncate block">
                        Model: {selected.controlnet_model}
                      </span>
                    )}
                  </div>
                )}

                {selected.mode === 'inpaint' && (
                  <div className="col-span-2 p-2.5 rounded-lg bg-surface-elevated border border-border-subtle space-y-1">
                    <span className="text-text-dim block text-[9px] uppercase font-sans">Inpaint Configuration</span>
                    <div className="flex items-center justify-between text-accent font-medium">
                      <span>Mask: Applied</span>
                      <span>Denoise: {selected.denoise}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Inpaint Mask Thumbnail if available */}
            {selected.mask_image_url && (
              <div className="p-3 rounded-xl bg-surface-elevated border border-border-subtle space-y-1.5">
                <span className="text-text-dim block text-[10px] uppercase font-sans font-medium">
                  Inpaint Mask Overlay
                </span>
                <div className="rounded-lg overflow-hidden border border-border-subtle bg-black max-h-36 flex items-center justify-center p-1">
                  <img
                    src={selected.mask_image_url}
                    alt="Mask preview"
                    className="max-h-32 object-contain"
                  />
                </div>
              </div>
            )}

            {/* LoRAs applied chip strip */}
            {selected.loras && selected.loras.length > 0 && (
              <div className="p-3 rounded-xl bg-surface-elevated border border-border-subtle space-y-1.5">
                <span className="text-text-dim block text-[10px] uppercase font-sans font-medium">
                  LoRA Weights Applied
                </span>
                <div className="flex flex-wrap gap-1.5 font-mono text-[11px]">
                  {selected.loras.map((l) => (
                    <span
                      key={l.name}
                      className="px-2 py-0.5 rounded bg-surface text-accent border border-border-subtle"
                    >
                      {l.name} ({l.strength}x)
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Primary Action Buttons */}
            <div className="space-y-2 pt-2">
              <button
                onClick={() => reuseSettings(selected)}
                aria-label="Load generation parameters into studio form"
                className="w-full py-2.5 px-4 rounded-xl font-semibold text-xs bg-accent text-background hover:bg-accent-hover transition-colors flex items-center justify-center space-x-2 shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Load Settings into Form</span>
              </button>

              {/* Reproduce Snapshot Button */}
              <button
                onClick={async () => {
                  try {
                    const res = await fetch(`/api/generations/${selected.id}/reproduce`);
                    if (res.ok) {
                      const data = await res.json();
                      const cfg = data.reproduction_config;
                      useStudioStore.setState({
                        prompt: cfg.prompt,
                        negative_prompt: cfg.negative_prompt,
                        model: cfg.model,
                        width: cfg.width,
                        height: cfg.height,
                        steps: cfg.steps,
                        cfg_scale: cfg.cfg_scale,
                        sampler: cfg.sampler,
                        seed: cfg.seed,
                        denoise: cfg.denoise,
                        loras: cfg.loras || [],
                        mode: cfg.mode,
                        activeNav: 'generate',
                      });
                    }
                  } catch (e) {
                    console.error('Failed to reproduce generation:', e);
                  }
                }}
                className="w-full py-2 px-4 rounded-xl font-medium text-xs bg-surface-elevated hover:bg-surface text-text-primary border border-border-subtle transition-colors flex items-center justify-center space-x-2"
                title="Reproduce exact parameters from stored settings snapshot"
              >
                <RefreshCw className="w-3.5 h-3.5 text-accent" />
                <span>Reproduce Exact Generation</span>
              </button>

              {/* Copy As Preset Button */}
              <button
                onClick={() => {
                  setCopyPresetName(selected.preset_name || `Preset from ${selected.mode}`);
                  setCopyPresetOpen(true);
                }}
                className="w-full py-2 px-4 rounded-xl font-medium text-xs bg-surface-elevated hover:bg-surface text-text-primary border border-border-subtle transition-colors flex items-center justify-center space-x-2"
                title="Save this generation's exact snapshot as a user preset"
              >
                <Bookmark className="w-3.5 h-3.5 text-accent" />
                <span>Copy as Preset</span>
              </button>

              {selected.image_url && (
                <button
                  onClick={() => sendToImg2Img(selected.image_url!)}
                  aria-label="Send image to image-to-image canvas"
                  className="w-full py-2.5 px-4 rounded-xl font-medium text-xs bg-surface-elevated hover:bg-surface-hover text-accent border border-border-subtle transition-colors flex items-center justify-center space-x-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                >
                  <ArrowRightCircle className="w-3.5 h-3.5" />
                  <span>Send to Image-to-Image Canvas</span>
                </button>
              )}
            </div>

            {/* Copy As Preset Dialog Modal */}
            {copyPresetOpen && (
              <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm">
                <div className="w-full max-w-sm bg-surface border border-border-subtle rounded-2xl p-5 space-y-3 shadow-2xl">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-text-primary">Copy as Workflow Preset</h3>
                    <button
                      onClick={() => setCopyPresetOpen(false)}
                      className="text-text-dim hover:text-text-primary"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                  <p className="text-xs text-text-muted">
                    Save the parameters from generation <code className="font-mono text-accent">{selected.id.slice(0, 8)}</code> as a preset.
                  </p>
                  <div>
                    <input
                      type="text"
                      value={copyPresetName}
                      onChange={(e) => setCopyPresetName(e.target.value)}
                      placeholder="Preset Name"
                      className="w-full bg-surface-elevated text-text-primary text-xs rounded-lg p-2.5 border border-border-subtle focus:border-accent focus:outline-none font-mono"
                    />
                  </div>
                  {copyPresetSuccess && (
                    <div className="text-xs text-status-success font-mono flex items-center gap-1">
                      <Check className="w-3.5 h-3.5" />
                      <span>Preset saved successfully!</span>
                    </div>
                  )}
                  <div className="flex justify-end space-x-2 pt-2">
                    <button
                      type="button"
                      onClick={() => setCopyPresetOpen(false)}
                      className="px-3 py-1.5 rounded-lg text-xs text-text-muted hover:bg-surface-elevated"
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      onClick={async () => {
                        if (!copyPresetName.trim()) return;
                        try {
                          const res = await fetch(`/api/generations/${selected.id}/copy-as-preset`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                              name: copyPresetName.trim(),
                              description: `Preset copied from generation ${selected.id.slice(0, 8)}`,
                            }),
                          });
                          if (res.ok) {
                            setCopyPresetSuccess(true);
                            useStudioStore.getState().loadPresets();
                            setTimeout(() => {
                              setCopyPresetSuccess(false);
                              setCopyPresetOpen(false);
                            }, 1200);
                          }
                        } catch (e) {
                          console.error('Failed to copy preset:', e);
                        }
                      }}
                      className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-accent text-background hover:bg-accent-hover"
                    >
                      Save Preset
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </aside>
      )}
    </div>
  );
};
