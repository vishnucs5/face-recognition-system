import React from 'react';
import { Sparkles, Image as ImageIcon, Layers, Cpu, Settings as SettingsIcon, Radio, Wrench, Bookmark } from 'lucide-react';
import { useStudioStore } from '../store/useStudioStore';
import { AppNav } from '../types';

export const LeftRail: React.FC = () => {
  const { activeNav, setActiveNav, systemStatus, telemetry } = useStudioStore();

  const navItems: { id: AppNav; label: string; icon: React.ReactNode; comingSoon?: boolean }[] = [
    { id: 'generate', label: 'Generation', icon: <Sparkles className="w-4 h-4" /> },
    { id: 'presets', label: 'Presets', icon: <Bookmark className="w-4 h-4" /> },
    { id: 'gallery', label: 'Gallery', icon: <ImageIcon className="w-4 h-4" /> },
    { id: 'loras', label: 'LoRA Vault', icon: <Layers className="w-4 h-4" /> },
    { id: 'training', label: 'LoRA Training', icon: <Cpu className="w-4 h-4" /> },
    { id: 'settings', label: 'Settings', icon: <SettingsIcon className="w-4 h-4" /> },
  ];

  const comfyOnline = systemStatus?.comfyui_online ?? false;
  const isDemo = systemStatus?.demo_mode ?? true;
  const ollamaOnline = systemStatus?.ollama_online ?? false;

  return (
    <aside className="w-64 bg-surface border-r border-border-subtle flex flex-col justify-between shrink-0 select-none">
      {/* Brand & Wordmark */}
      <div>
        <div className="px-5 py-6 border-b border-border-subtle flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-accent/20 border border-accent/40 flex items-center justify-center text-accent">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h1 className="font-semibold text-text-primary text-base tracking-tight flex items-center gap-1.5">
              Forge Studio
              <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-accent/15 text-accent font-medium">
                {isDemo ? 'Demo' : 'Live'}
              </span>
            </h1>
            <p className="text-xs text-text-muted">SDXL Workstation</p>
          </div>
        </div>

        {/* Navigation items */}
        <nav className="p-3 space-y-1">
          {navItems.map((item) => {
            const isActive = activeNav === item.id;
            return (
              <button
                key={item.id}
                onClick={() => !item.comingSoon && setActiveNav(item.id)}
                disabled={item.comingSoon}
                aria-label={item.comingSoon ? `${item.label} (Coming next)` : `Navigate to ${item.label}`}
                aria-current={isActive ? 'page' : undefined}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                  isActive
                    ? 'bg-surface-elevated text-accent border border-accent/30 shadow-sm'
                    : item.comingSoon
                    ? 'text-text-dim cursor-not-allowed opacity-60'
                    : 'text-text-muted hover:text-text-primary hover:bg-surface-hover'
                }`}
              >
                <div className="flex items-center space-x-3">
                  <span className={isActive ? 'text-accent' : 'text-text-muted'}>
                    {item.icon}
                  </span>
                  <span>{item.label}</span>
                </div>
                {item.comingSoon && (
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-surface-elevated text-text-dim border border-border-subtle">
                    Next
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Connection & Telemetry Footer */}
      <div className="p-4 border-t border-border-subtle space-y-3 bg-surface/50 text-xs">
        <div className="space-y-2">
          {/* ComfyUI Indicator */}
          <div className="flex items-center justify-between">
            <span className="text-text-muted flex items-center gap-1.5">
              <Radio className="w-3.5 h-3.5" />
              ComfyUI Engine
            </span>
            <span
              className={`font-mono text-[11px] px-1.5 py-0.5 rounded ${
                comfyOnline
                  ? 'bg-status-success/20 text-status-success'
                  : isDemo
                  ? 'bg-accent/20 text-accent'
                  : 'bg-status-error/20 text-status-error'
              }`}
            >
              {comfyOnline ? 'Online' : isDemo ? 'Demo Mode' : 'Offline'}
            </span>
          </div>

          {/* Ollama Copilot Indicator */}
          <div className="flex items-center justify-between">
            <span className="text-text-muted flex items-center gap-1.5">
              <Wrench className="w-3.5 h-3.5" />
              Ollama Copilot
            </span>
            <span
              className={`font-mono text-[11px] px-1.5 py-0.5 rounded ${
                ollamaOnline
                  ? 'bg-status-success/20 text-status-success'
                  : 'bg-text-dim/20 text-text-muted'
              }`}
            >
              {ollamaOnline ? 'Connected' : 'Offline'}
            </span>
          </div>
        </div>

        {/* Real GPU VRAM Telemetry */}
        {telemetry ? (
          <div className="pt-2 border-t border-border-subtle space-y-1">
            <div className="flex justify-between items-center text-[11px]">
              <span className="text-text-muted truncate max-w-[110px]" title={telemetry.gpu_name}>
                {telemetry.gpu_name}
              </span>
              <span className="font-mono text-text-primary text-[10px]">
                {(telemetry.used_vram_mb / 1024).toFixed(1)} / {(telemetry.total_vram_mb / 1024).toFixed(1)} GB
              </span>
            </div>
            <div className="w-full bg-surface-elevated rounded-full h-1.5 overflow-hidden">
              <div
                className={`h-full transition-all duration-500 rounded-full ${
                  (telemetry.used_vram_mb / (telemetry.total_vram_mb || 1)) > 0.85
                    ? 'bg-status-warning'
                    : 'bg-accent'
                }`}
                style={{
                  width: `${Math.min(100, Math.round((telemetry.used_vram_mb / (telemetry.total_vram_mb || 1)) * 100))}%`,
                }}
              />
            </div>
            <div className="flex justify-between items-center text-[10px] text-text-dim font-mono">
              <span>{telemetry.utilization_pct}% load</span>
              {telemetry.temperature_c !== null && <span>{telemetry.temperature_c}°C</span>}
            </div>
          </div>
        ) : (
          <div className="pt-2 border-t border-border-subtle text-[11px] text-text-dim font-mono">
            Telemetry unavailable
          </div>
        )}
      </div>
    </aside>
  );
};
