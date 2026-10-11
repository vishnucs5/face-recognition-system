import React, { useState } from 'react';
import {
  Download,
  RotateCcw,
  Sparkles,
  ArrowRightCircle,
  Info,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  RefreshCw,
  Maximize2,
  Copy,
  Check,
  Crosshair,
  FileJson,
  Paintbrush,
} from 'lucide-react';
import { useStudioStore } from '../store/useStudioStore';

interface PreviewPanelProps {
  onRetry?: () => void;
}

export const PreviewPanel: React.FC<PreviewPanelProps> = ({ onRetry }) => {
  const {
    activeGeneration,
    isGenerating,
    progress,
    reuseSettings,
    sendToImg2Img,
    sendToInpaint,
    cancelActiveGeneration,
    systemStatus,
    mode,
    source_image_preview,
  } = useStudioStore();

  const [metadataOpen, setMetadataOpen] = useState(true);
  const [copiedPrompt, setCopiedPrompt] = useState(false);
  const [zoomMode, setZoomMode] = useState<'contain' | 'cover'>('contain');

  const handleDownload = () => {
    if (!activeGeneration?.image_url) return;
    const a = document.createElement('a');
    a.href = activeGeneration.image_url;
    a.download = `forge_${activeGeneration.id.slice(0, 8)}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleDownloadMetadata = () => {
    if (!activeGeneration) return;
    const metadata = {
      prompt: activeGeneration.prompt,
      negative_prompt: activeGeneration.negative_prompt,
      checkpoint: activeGeneration.model,
      loras: activeGeneration.loras,
      width: activeGeneration.width,
      height: activeGeneration.height,
      steps: activeGeneration.steps,
      cfg_scale: activeGeneration.cfg_scale,
      sampler: activeGeneration.sampler,
      seed: activeGeneration.seed,
      denoise: activeGeneration.denoise,
      source_mode: activeGeneration.mode,
      duration_ms: activeGeneration.duration_ms,
      timestamp: activeGeneration.created_at,
      completed_at: activeGeneration.completed_at,
      comfy_prompt_id: activeGeneration.comfy_prompt_id,
    };
    const blob = new Blob([JSON.stringify(metadata, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `forge_${activeGeneration.id.slice(0, 8)}_metadata.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleCopyPrompt = () => {
    if (!activeGeneration?.prompt) return;
    navigator.clipboard.writeText(activeGeneration.prompt);
    setCopiedPrompt(true);
    setTimeout(() => setCopiedPrompt(false), 2000);
  };

  const isFailed = activeGeneration?.status === 'failed';

  return (
    <div className="h-full flex flex-col bg-surface border-l border-border-subtle overflow-hidden select-none">
      {/* Top action bar */}
      <div className="px-5 py-4 border-b border-border-subtle flex items-center justify-between shrink-0 bg-surface-elevated/30">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-text-muted flex items-center gap-1.5">
            <Crosshair className="w-3.5 h-3.5 text-accent" />
            Viewport Canvas
          </span>
          {activeGeneration && !isFailed && (
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-surface-elevated text-accent border border-border-subtle">
              {activeGeneration.width} × {activeGeneration.height}
            </span>
          )}
        </div>

        {/* Quick action buttons */}
        {activeGeneration?.image_url && !isGenerating && !isFailed && (
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setZoomMode(zoomMode === 'contain' ? 'cover' : 'contain')}
              aria-label={zoomMode === 'contain' ? 'Zoom to fill canvas' : 'Fit image to screen'}
              className={`p-1.5 rounded-lg border text-xs transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                zoomMode === 'cover'
                  ? 'bg-accent/20 border-accent/40 text-accent'
                  : 'bg-surface-elevated border-border-subtle text-text-muted hover:text-text-primary'
              }`}
              title={zoomMode === 'contain' ? 'Zoom to fill' : 'Fit to screen'}
            >
              <Maximize2 className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => reuseSettings(activeGeneration)}
              aria-label="Reuse generation parameters in studio form"
              className="p-1.5 rounded-lg bg-surface-elevated border border-border-subtle text-text-muted hover:text-text-primary transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              title="Reuse these generation parameters"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => sendToImg2Img(activeGeneration.image_url!)}
              aria-label="Send image into Image-to-Image canvas mode"
              className="p-1.5 rounded-lg bg-surface-elevated border border-border-subtle text-text-muted hover:text-accent transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              title="Send to Image-to-Image"
            >
              <ArrowRightCircle className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => sendToInpaint(activeGeneration.image_url!)}
              aria-label="Send image into Inpaint canvas mode"
              className="p-1.5 rounded-lg bg-surface-elevated border border-border-subtle text-text-muted hover:text-accent transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              title="Send to Inpainting"
            >
              <Paintbrush className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={handleDownloadMetadata}
              aria-label="Export metadata JSON"
              className="p-1.5 rounded-lg bg-surface-elevated border border-border-subtle text-text-muted hover:text-text-primary transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              title="Export generation metadata JSON"
            >
              <FileJson className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={handleDownload}
              aria-label="Export generated image as high-resolution PNG"
              className="px-2.5 py-1.5 rounded-lg bg-accent text-background hover:bg-accent-hover text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              title="Export high-resolution PNG"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export</span>
            </button>
          </div>
        )}
      </div>

      {/* Main Viewport */}
      <div className="flex-1 relative flex items-center justify-center p-6 bg-background/60 overflow-hidden">
        {/* State A: In-flight Sampling Progress */}
        {isGenerating ? (
          <div className="w-full max-w-md p-8 rounded-2xl bg-surface border border-border-subtle text-center space-y-5 shadow-2xl relative overflow-hidden animate-in fade-in zoom-in-95 duration-200">
            {/* Ambient amber glow behind sampling HUD */}
            <div className="absolute -top-12 -left-12 w-40 h-40 bg-accent/10 rounded-full blur-3xl pointer-events-none" />
            <div className="absolute -bottom-12 -right-12 w-40 h-40 bg-accent/10 rounded-full blur-3xl pointer-events-none" />

            {/* If in img2img mode, display original source image while pending */}
            {mode === 'img2img' && source_image_preview && (
              <div className="relative rounded-xl overflow-hidden border border-border-subtle max-h-48 mx-auto bg-background flex items-center justify-center p-1">
                <img
                  src={source_image_preview}
                  alt="Original Source"
                  className="max-h-44 object-contain opacity-75 rounded"
                />
                <span className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-background/80 text-[10px] font-mono text-accent border border-border-subtle">
                  Transforming Source Image...
                </span>
              </div>
            )}

            <div className="w-14 h-14 rounded-2xl bg-accent/15 border border-accent/40 mx-auto flex items-center justify-center text-accent shadow-inner">
              <Sparkles className="w-7 h-7 animate-spin duration-1000" />
            </div>

            <div className="space-y-1">
              <h3 className="text-sm font-semibold text-text-primary tracking-tight">
                {progress?.status === 'queued' ? 'Queued in Compute Pipeline' : 'Sampling Latent Space...'}
              </h3>
              <p className="text-xs text-text-muted font-mono">
                {progress?.message || 'Executing SDXL workflow graph...'}
              </p>
            </div>

            {/* Shimmer progress bar */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-text-muted">
                  Step {progress?.step || 0} / {progress?.total_steps || 28}
                </span>
                <span className="text-accent font-semibold">
                  {Math.round(progress?.progress_percentage || 0)}%
                </span>
              </div>
              <div className="w-full bg-surface-elevated rounded-full h-2.5 overflow-hidden border border-border-subtle p-0.5">
                <div
                  className="bg-accent h-full transition-all duration-300 rounded-full shadow-sm"
                  style={{ width: `${Math.max(2, progress?.progress_percentage || 0)}%` }}
                />
              </div>
            </div>

            {/* Cancel Generation button */}
            <button
              onClick={() => cancelActiveGeneration(activeGeneration?.id)}
              className="text-xs text-text-dim hover:text-status-error font-medium transition-colors"
            >
              Cancel this job
            </button>
          </div>
        ) : isFailed ? (
          /* State B: Failed Generation State */
          <div className="w-full max-w-md p-8 rounded-2xl bg-surface border border-status-error/40 text-center space-y-4 shadow-xl animate-in fade-in duration-200">
            <div className="w-12 h-12 rounded-xl bg-status-error/15 border border-status-error/30 mx-auto flex items-center justify-center text-status-error">
              <AlertTriangle className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-status-error">Generation Unsuccessful</h3>
              <p className="text-xs text-text-muted mt-1 leading-relaxed">
                {activeGeneration?.error_message || 'The workflow engine encountered an unhandled condition.'}
              </p>
            </div>
            {onRetry && (
              <button
                onClick={() => {
                  if (activeGeneration) reuseSettings(activeGeneration);
                  onRetry();
                }}
                className="px-4 py-2 rounded-lg bg-surface-elevated hover:bg-surface-hover text-accent border border-border-subtle text-xs font-medium inline-flex items-center gap-1.5 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Retry Generation</span>
              </button>
            )}
          </div>
        ) : activeGeneration?.image_url ? (
          /* State C: Active Image Display */
          <div className="relative max-h-full max-w-full flex items-center justify-center group">
            <img
              src={activeGeneration.image_url}
              alt={activeGeneration.prompt}
              className={`max-h-[calc(100vh-230px)] max-w-full rounded-xl shadow-2xl border border-border-subtle transition-all ${
                zoomMode === 'cover' ? 'w-full h-full object-cover' : 'object-contain'
              }`}
            />
            {systemStatus?.demo_mode && (
              <div className="absolute bottom-3 left-1/2 -translate-x-1/2 px-3 py-1 rounded-full bg-background/90 backdrop-blur-md border border-accent/40 text-[11px] font-mono text-accent flex items-center gap-1.5 shadow-lg pointer-events-none">
                <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
                <span>Demo Synthesis • Start ComfyUI (:8188) for Real SDXL</span>
              </div>
            )}
          </div>
        ) : (
          /* State D: Darkroom Editorial Empty State */
          <div className="text-center p-8 max-w-sm space-y-4 text-text-dim">
            {/* Bespoke Darkroom SVG Canvas Graphic */}
            <div className="relative w-28 h-28 mx-auto flex items-center justify-center">
              <svg
                viewBox="0 0 100 100"
                className="w-full h-full text-border-strong"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
              >
                {/* Outer frame */}
                <rect x="10" y="10" width="80" height="80" rx="8" strokeDasharray="4 4" />
                {/* Optical crosshairs */}
                <line x1="50" y1="20" x2="50" y2="35" stroke="#D7A85C" strokeWidth="1.5" />
                <line x1="50" y1="65" x2="50" y2="80" stroke="#D7A85C" strokeWidth="1.5" />
                <line x1="20" y1="50" x2="35" y2="50" stroke="#D7A85C" strokeWidth="1.5" />
                <line x1="65" y1="50" x2="80" y2="50" stroke="#D7A85C" strokeWidth="1.5" />
                {/* Center aperture ring */}
                <circle cx="50" cy="50" r="14" stroke="#D7A85C" strokeWidth="1.5" opacity="0.6" />
                <circle cx="50" cy="50" r="4" fill="#D7A85C" />
              </svg>
            </div>

            <div className="space-y-1">
              <h3 className="text-sm font-semibold text-text-muted tracking-tight">Darkroom Ready</h3>
              <p className="text-xs leading-relaxed text-text-dim">
                Enter your prompt and click <span className="text-accent font-medium">Generate</span> to synthesize on your RTX 5070 Ti.
              </p>
            </div>

            <div className="pt-2 flex items-center justify-center gap-2 text-[11px] font-mono text-text-dim">
              <span className="px-2 py-0.5 rounded bg-surface-elevated border border-border-subtle">
                Ctrl + Enter
              </span>
              <span>Quick trigger</span>
            </div>
          </div>
        )}
      </div>

      {/* Metadata Collapsible Drawer */}
      {activeGeneration && !isFailed && (
        <div className="border-t border-border-subtle bg-surface shrink-0">
          <button
            onClick={() => setMetadataOpen(!metadataOpen)}
            className="w-full px-5 py-2.5 flex items-center justify-between text-xs text-text-muted hover:text-text-primary transition-colors bg-surface-elevated/20"
          >
            <span className="flex items-center gap-1.5 font-medium">
              <Info className="w-3.5 h-3.5 text-accent" />
              Generation Parameters & Reproducibility
            </span>
            {metadataOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
          </button>

          {metadataOpen && (
            <div className="px-5 pb-4 pt-2 space-y-2.5 text-xs border-t border-border-subtle/50 font-mono">
              {/* Technical parameter grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
                <div className="p-2.5 rounded-lg bg-surface-elevated border border-border-subtle">
                  <span className="text-text-dim block text-[10px] uppercase font-sans">Mode & Status</span>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <span className="text-accent uppercase font-semibold text-[10px]">{activeGeneration.mode}</span>
                    <span className="text-text-dim">•</span>
                    <span className={`capitalize text-[10px] ${
                      activeGeneration.status === 'completed' ? 'text-status-success' :
                      activeGeneration.status === 'failed' ? 'text-status-error' :
                      activeGeneration.status === 'cancelled' ? 'text-text-dim' : 'text-accent'
                    }`}>{activeGeneration.status}</span>
                  </div>
                </div>
                <div className="p-2.5 rounded-lg bg-surface-elevated border border-border-subtle">
                  <span className="text-text-dim block text-[10px] uppercase font-sans">Model Checkpoint</span>
                  <span className="text-text-primary truncate block font-medium" title={activeGeneration.model}>{activeGeneration.model}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-surface-elevated border border-border-subtle">
                  <span className="text-text-dim block text-[10px] uppercase font-sans">Resolution</span>
                  <span className="text-text-primary font-medium">{activeGeneration.width} × {activeGeneration.height}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-surface-elevated border border-border-subtle">
                  <span className="text-text-dim block text-[10px] uppercase font-sans">Sampler & Steps</span>
                  <span className="text-text-primary font-medium">{activeGeneration.sampler} ({activeGeneration.steps})</span>
                </div>
                <div className="p-2.5 rounded-lg bg-surface-elevated border border-border-subtle">
                  <span className="text-text-dim block text-[10px] uppercase font-sans">Synthesis Seed</span>
                  <span className="text-text-primary font-medium">{activeGeneration.seed ?? 'Random'}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-surface-elevated border border-border-subtle">
                  <span className="text-text-dim block text-[10px] uppercase font-sans">CFG & Denoise</span>
                  <span className="text-text-primary font-medium">
                    CFG {activeGeneration.cfg_scale}{activeGeneration.mode === 'img2img' ? ` / Den ${activeGeneration.denoise}` : ''}
                  </span>
                </div>
                <div className="p-2.5 rounded-lg bg-surface-elevated border border-border-subtle">
                  <span className="text-text-dim block text-[10px] uppercase font-sans">Inference Time</span>
                  <span className="text-text-primary font-medium">
                    {activeGeneration.duration_ms ? `${(activeGeneration.duration_ms / 1000).toFixed(1)}s` : 'N/A'}
                  </span>
                </div>
                <div className="p-2.5 rounded-lg bg-surface-elevated border border-border-subtle">
                  <span className="text-text-dim block text-[10px] uppercase font-sans">ComfyUI Prompt ID</span>
                  <span className="text-accent truncate block font-medium" title={activeGeneration.comfy_prompt_id || 'Direct Job'}>
                    {activeGeneration.comfy_prompt_id ? activeGeneration.comfy_prompt_id.slice(0, 10) + '...' : 'Direct Job'}
                  </span>
                </div>
              </div>

              {/* LoRAs applied chip strip */}
              {activeGeneration.loras && activeGeneration.loras.length > 0 && (
                <div className="p-2.5 rounded-lg bg-surface-elevated border border-border-subtle text-[11px]">
                  <span className="text-text-dim block text-[10px] uppercase font-sans mb-1.5">LoRAs Applied</span>
                  <div className="flex flex-wrap gap-1.5">
                    {activeGeneration.loras.map((l) => (
                      <span key={l.name} className="px-2 py-0.5 rounded bg-surface text-accent border border-border-subtle">
                        {l.name} ({l.strength})
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Quick copy prompt */}
              <div className="space-y-1 pt-1">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-text-dim font-sans truncate max-w-[280px]">
                    "{activeGeneration.prompt}"
                  </span>
                  <button
                    onClick={handleCopyPrompt}
                    className="text-accent hover:underline flex items-center gap-1 font-sans shrink-0 ml-2"
                  >
                    {copiedPrompt ? <Check className="w-3 h-3 text-status-success" /> : <Copy className="w-3 h-3" />}
                    <span>{copiedPrompt ? 'Copied' : 'Copy Prompt'}</span>
                  </button>
                </div>
                {activeGeneration.negative_prompt && (
                  <div className="text-[10px] text-text-dim font-sans italic truncate">
                    Neg: "{activeGeneration.negative_prompt}"
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
