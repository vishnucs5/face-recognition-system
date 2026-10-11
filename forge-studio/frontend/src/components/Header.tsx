import React, { useState } from 'react';
import { Bot, Cpu, Sliders, Image as ImageIcon, Keyboard, Radio, Wrench, Shield, Activity, ChevronDown } from 'lucide-react';
import { useStudioStore } from '../store/useStudioStore';
import { AppMode } from '../types';

export const Header: React.FC = () => {
  const {
    mode,
    setMode,
    systemStatus,
    telemetry,
    vramPolicy,
    setVramPolicy,
    copilotOpen,
    setCopilotOpen,
    mobileView,
    setMobileView,
    setShortcutsModalOpen,
    isGenerating,
    progress,
    activeGeneration,
  } = useStudioStore();

  const [policyDropdownOpen, setPolicyDropdownOpen] = useState(false);

  const modes: { id: AppMode; label: string; comingSoon?: boolean }[] = [
    { id: 'txt2img', label: 'Text to Image' },
    { id: 'img2img', label: 'Image to Image' },
    { id: 'inpaint', label: 'Inpainting' },
    { id: 'controlnet', label: 'ControlNet' },
  ];

  const policies = [
    { id: 'balanced', label: 'Balanced', limit: '85%' },
    { id: 'max_quality', label: 'Max Quality', limit: '92%' },
    { id: 'max_speed', label: 'Max Speed', limit: '78%' },
    { id: 'conservative', label: 'Conservative', limit: '65%' },
  ] as const;

  // Determine state label and color from real telemetry
  const getStateBadge = () => {
    if (!telemetry) {
      return (
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-surface-elevated text-text-dim border border-border-subtle">
          Telemetry unavailable
        </span>
      );
    }

    switch (telemetry.state) {
      case 'gpu_ready':
        return (
          <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-status-success/15 text-status-success border border-status-success/30 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-status-success" />
            GPU ready
          </span>
        );
      case 'gpu_memory_pressure':
        return (
          <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-status-warning/20 text-status-warning border border-status-warning/40 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-status-warning animate-pulse" />
            GPU memory pressure
          </span>
        );
      case 'waiting_for_gpu':
        return (
          <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-status-warning/20 text-status-warning border border-status-warning/40 flex items-center gap-1 animate-pulse">
            <span className="w-1.5 h-1.5 rounded-full bg-status-warning" />
            Wait for GPU
          </span>
        );
      case 'comfyui_active':
        return (
          <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-accent/20 text-accent border border-accent/40 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-accent animate-ping" />
            ComfyUI active
          </span>
        );
      case 'ollama_active':
        return (
          <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/40 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
            Ollama active
          </span>
        );
      case 'training_active':
        return (
          <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-purple-500/20 text-purple-400 border border-purple-500/40 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-purple-400" />
            Training active
          </span>
        );
      case 'cuda_unavailable':
        return (
          <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-status-error/20 text-status-error border border-status-error/40 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-status-error" />
            CUDA unavailable
          </span>
        );
      case 'gpu_unavailable':
      default:
        return (
          <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-status-error/20 text-status-error border border-status-error/40 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-status-error" />
            GPU unavailable
          </span>
        );
    }
  };

  const vramPercent = telemetry && telemetry.total_vram_mb > 0
    ? Math.min(100, Math.round((telemetry.used_vram_mb / telemetry.total_vram_mb) * 100))
    : 0;

  return (
    <header className="h-16 px-4 md:px-6 bg-surface border-b border-border-subtle flex items-center justify-between shrink-0">
      <div className="flex items-center space-x-3">
        {/* Mode Navigation Tabs */}
        <div className="flex items-center space-x-1 bg-surface-elevated p-1 rounded-xl border border-border-subtle">
          {modes.map((m) => {
            const isActive = mode === m.id;
            return (
              <button
                key={m.id}
                onClick={() => !m.comingSoon && setMode(m.id)}
                disabled={m.comingSoon}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
                  isActive
                    ? 'bg-surface text-accent shadow-sm border border-border-subtle'
                    : m.comingSoon
                    ? 'text-text-dim cursor-not-allowed opacity-60'
                    : 'text-text-muted hover:text-text-primary'
                }`}
              >
                <span>{m.label}</span>
                {m.comingSoon && (
                  <span className="hidden sm:inline-block text-[9px] font-mono uppercase px-1 py-0.2 rounded bg-surface border border-border-subtle text-text-dim">
                    Coming next
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {/* Compact screen View Switcher (Controls vs Preview) */}
        <div className="lg:hidden flex items-center bg-surface-elevated p-0.5 rounded-lg border border-border-subtle">
          <button
            onClick={() => setMobileView('controls')}
            className={`px-2.5 py-1 rounded text-xs font-medium flex items-center gap-1 ${
              mobileView === 'controls'
                ? 'bg-surface text-accent font-semibold'
                : 'text-text-muted hover:text-text-primary'
            }`}
          >
            <Sliders className="w-3 h-3" />
            <span>Form</span>
          </button>
          <button
            onClick={() => setMobileView('preview')}
            className={`px-2.5 py-1 rounded text-xs font-medium flex items-center gap-1 ${
              mobileView === 'preview'
                ? 'bg-surface text-accent font-semibold'
                : 'text-text-muted hover:text-text-primary'
            }`}
          >
            <ImageIcon className="w-3 h-3" />
            <span>Canvas</span>
          </button>
        </div>
      </div>

      {/* Right controls: Telemetry, Policy, Indicators, Copilot */}
      <div className="flex items-center space-x-2">
        {/* State Badge */}
        <div className="hidden sm:flex items-center">
          {getStateBadge()}
        </div>

        {/* Real GPU Telemetry Gauge */}
        {telemetry ? (
          <div className="hidden xl:flex items-center space-x-2.5 px-3 py-1.5 rounded-lg bg-surface-elevated border border-border-subtle text-xs">
            <Cpu className="w-3.5 h-3.5 text-accent shrink-0" />
            <div className="flex items-center space-x-2 font-mono text-[11px]">
              <span className="text-text-primary font-medium truncate max-w-[140px]" title={telemetry.gpu_name}>
                {telemetry.gpu_name}
              </span>
              <span className="text-text-dim">|</span>
              <div className="flex items-center space-x-1.5">
                <span className={vramPercent > 85 ? 'text-status-warning font-semibold' : 'text-text-muted'}>
                  {(telemetry.used_vram_mb / 1024).toFixed(1)} / {(telemetry.total_vram_mb / 1024).toFixed(1)} GB
                </span>
                <span className="text-text-dim text-[10px]">
                  ({(telemetry.free_vram_mb / 1024).toFixed(1)} GB free)
                </span>
              </div>
              <span className="text-text-dim">|</span>
              <span className="text-text-muted">
                {telemetry.temperature_c !== null ? `${telemetry.temperature_c}°C` : `${telemetry.utilization_pct}% load`}
              </span>
              <span className="text-text-dim">|</span>
              <span className="text-text-dim text-[10px]" title="Backend telemetry latency">
                {telemetry.backend_latency_ms}ms
              </span>
            </div>
          </div>
        ) : (
          <div className="hidden xl:flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg bg-surface-elevated border border-border-subtle text-xs text-text-dim font-mono text-[11px]">
            <Cpu className="w-3.5 h-3.5 text-text-dim" />
            <span>Telemetry unavailable</span>
          </div>
        )}

        {/* VRAM Safety Policy Selector */}
        <div className="relative">
          <button
            onClick={() => setPolicyDropdownOpen(!policyDropdownOpen)}
            className="flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg bg-surface-elevated hover:bg-surface border border-border-subtle text-xs transition-colors"
            title="VRAM Safety Allocation Policy"
          >
            <Shield className="w-3.5 h-3.5 text-accent shrink-0" />
            <span className="font-mono text-[11px] font-semibold text-text-primary capitalize">
              {vramPolicy.replace('_', ' ')}
            </span>
            <ChevronDown className="w-3 h-3 text-text-muted" />
          </button>

          {policyDropdownOpen && (
            <div className="absolute right-0 mt-1 w-48 bg-surface-elevated border border-border-subtle rounded-xl shadow-xl z-50 p-1 space-y-0.5">
              <div className="px-2 py-1 text-[10px] font-mono uppercase text-text-dim border-b border-border-subtle">
                VRAM Safety Policy
              </div>
              {policies.map((p) => (
                <button
                  key={p.id}
                  onClick={() => {
                    setVramPolicy(p.id);
                    setPolicyDropdownOpen(false);
                  }}
                  className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs font-mono transition-colors ${
                    vramPolicy === p.id
                      ? 'bg-accent/20 text-accent font-semibold'
                      : 'text-text-muted hover:text-text-primary hover:bg-surface'
                  }`}
                >
                  <span>{p.label}</span>
                  <span className="text-[10px] text-text-dim">{p.limit} limit</span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Keyboard Shortcuts Reference Trigger */}
        <button
          onClick={() => setShortcutsModalOpen(true)}
          className="p-2 rounded-lg bg-surface-elevated hover:bg-surface border border-border-subtle text-text-muted hover:text-text-primary transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          aria-label="View keyboard shortcuts reference dialog"
          title="Keyboard shortcuts (?)"
        >
          <Keyboard className="w-4 h-4" />
        </button>

        {/* Copilot Drawer Toggle */}
        <button
          onClick={() => setCopilotOpen(!copilotOpen)}
          aria-label={copilotOpen ? "Close Studio Copilot drawer" : "Open Studio Copilot drawer"}
          aria-expanded={copilotOpen}
          className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
            copilotOpen
              ? 'bg-accent text-background border-accent font-semibold shadow'
              : 'bg-surface-elevated text-accent border-accent/40 hover:bg-accent/10'
          }`}
        >
          <Bot className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">Studio Copilot</span>
        </button>
      </div>
    </header>
  );
};
