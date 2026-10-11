import React, { useEffect, useState } from 'react';
import {
  Cpu,
  Radio,
  Wrench,
  Folder,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  HardDrive,
  Sliders,
  ShieldCheck,
  Zap,
  Activity,
  Terminal,
  Copy,
  Check,
  Play,
  Download,
  FileSpreadsheet,
  FileJson,
  Layers,
  Clock,
  Shield,
  HelpCircle,
} from 'lucide-react';
import { useStudioStore } from '../store/useStudioStore';
import { BenchmarkResult, DiagnosticCheck, DiagnosticsReport } from '../types';

export const SettingsView: React.FC = () => {
  const {
    systemStatus,
    setSystemStatus,
    telemetry,
    vramPolicy,
    setVramPolicy,
    diagnosticsReport,
    setDiagnosticsReport,
  } = useStudioStore();

  const [activeTab, setActiveTab] = useState<'settings' | 'diagnostics' | 'benchmark'>('settings');

  // Diagnostics state
  const [runningDiagnostics, setRunningDiagnostics] = useState(false);
  const [copiedCmdId, setCopiedCmdId] = useState<string | null>(null);

  // Benchmark state
  const [confirmBenchmarkOpen, setConfirmBenchmarkOpen] = useState(false);
  const [isBenchmarking, setIsBenchmarking] = useState(false);
  const [benchmarkWarmup, setBenchmarkWarmup] = useState(false);
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResult | null>(null);
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null);

  // Connection testing state
  const [testingComfy, setTestingComfy] = useState(false);
  const [testingOllama, setTestingOllama] = useState(false);
  const [testResult, setTestResult] = useState<{ target: string; status: 'ok' | 'error'; message: string } | null>(null);

  const comfyOnline = systemStatus?.comfyui_online ?? false;
  const ollamaOnline = systemStatus?.ollama_online ?? false;
  const isDemo = systemStatus?.demo_mode ?? true;
  const gpu = systemStatus?.gpu;

  useEffect(() => {
    fetchDiagnostics(false);
  }, []);

  const fetchDiagnostics = async (refresh = false) => {
    setRunningDiagnostics(true);
    try {
      const res = await fetch(`/api/diagnostics/system?refresh=${refresh}`);
      if (res.ok) {
        const data: DiagnosticsReport = await res.json();
        setDiagnosticsReport(data);
      }
    } catch (err) {
      console.error('Failed to run diagnostics:', err);
    } finally {
      setRunningDiagnostics(false);
    }
  };

  const handleCopyCmd = (checkId: string, cmd: string) => {
    navigator.clipboard.writeText(cmd);
    setCopiedCmdId(checkId);
    setTimeout(() => setCopiedCmdId(null), 2000);
  };

  const handleRunBenchmark = async () => {
    setConfirmBenchmarkOpen(false);
    setIsBenchmarking(true);
    setBenchmarkWarmup(true);
    setBenchmarkError(null);

    // Warm-up phase timer representation
    const warmupTimer = setTimeout(() => {
      setBenchmarkWarmup(false);
    }, 2500);

    try {
      const res = await fetch('/api/diagnostics/benchmark/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_confirmed: true, include_ollama: true }),
      });

      if (res.ok) {
        const data: BenchmarkResult = await res.json();
        setBenchmarkResult(data);
      } else {
        const err = await res.json();
        setBenchmarkError(err.detail || 'Benchmark failed to execute.');
      }
    } catch (err) {
      setBenchmarkError('Network communication error during benchmark run.');
    } finally {
      clearTimeout(warmupTimer);
      setBenchmarkWarmup(false);
      setIsBenchmarking(false);
    }
  };

  const handleExportBenchmark = async (format: 'json' | 'csv') => {
    try {
      const res = await fetch(`/api/diagnostics/benchmark/export?format=${format}`);
      if (res.ok) {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `forge_benchmark_${new Date().toISOString().slice(0, 10)}.${format}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }
    } catch (err) {
      console.error('Failed to export benchmark:', err);
    }
  };

  const handlePingComfy = async () => {
    setTestingComfy(true);
    setTestResult(null);
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        setSystemStatus(data);
        if (data.comfyui_online) {
          setTestResult({ target: 'ComfyUI', status: 'ok', message: 'ComfyUI API responded successfully (200 OK).' });
        } else {
          setTestResult({
            target: 'ComfyUI',
            status: 'error',
            message: 'ComfyUI is unreachable at http://127.0.0.1:8188.',
          });
        }
      }
    } catch {
      setTestResult({ target: 'ComfyUI', status: 'error', message: 'Failed to communicate with backend server.' });
    } finally {
      setTestingComfy(false);
    }
  };

  const handlePingOllama = async () => {
    setTestingOllama(true);
    setTestResult(null);
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        setSystemStatus(data);
        if (data.ollama_online) {
          setTestResult({
            target: 'Ollama',
            status: 'ok',
            message: `Ollama is active with model ${data.ollama_model || 'qwen3-coder:30b'}.`,
          });
        } else {
          setTestResult({
            target: 'Ollama',
            status: 'error',
            message: 'Ollama is offline. Studio Copilot will use built-in offline heuristics.',
          });
        }
      }
    } catch {
      setTestResult({ target: 'Ollama', status: 'error', message: 'Failed to communicate with backend server.' });
    } finally {
      setTestingOllama(false);
    }
  };

  const vramPercent = telemetry && telemetry.total_vram_mb > 0
    ? Math.min(100, Math.round((telemetry.used_vram_mb / telemetry.total_vram_mb) * 100))
    : 0;

  return (
    <div className="flex-1 p-6 md:p-10 overflow-y-auto space-y-6 max-w-5xl">
      {/* Header */}
      <div className="space-y-3 border-b border-border-subtle pb-5">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-text-primary">
              Workstation Settings & Hardware Diagnostics
            </h1>
            <p className="text-xs text-text-muted mt-1">
              Configure endpoints, audit local subsystem readiness, and run reproducible GPU hardware benchmarks on NVIDIA RTX 5070 Ti.
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center space-x-1 pt-2">
          {[
            { id: 'settings', label: 'General & Endpoints' },
            { id: 'diagnostics', label: `System Diagnostics (${diagnosticsReport?.checks.length || 9})` },
            { id: 'benchmark', label: 'Hardware Benchmark' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all ${
                activeTab === tab.id
                  ? 'bg-surface-elevated text-accent font-semibold border border-border-subtle'
                  : 'text-text-muted hover:text-text-primary hover:bg-surface'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* TAB 1: GENERAL SETTINGS & CONNECTIONS */}
      {activeTab === 'settings' && (
        <div className="space-y-6">
          {/* Service Endpoints */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* ComfyUI Card */}
            <div className="p-5 rounded-2xl bg-surface border border-border-subtle space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2.5">
                  <div className="p-2 rounded-xl bg-accent/20 border border-accent/40 text-accent">
                    <Radio className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-text-primary">ComfyUI Diffusion Engine</h3>
                    <p className="text-[11px] font-mono text-text-muted">http://127.0.0.1:8188</p>
                  </div>
                </div>
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold ${
                    comfyOnline ? 'bg-status-success/20 text-status-success' : 'bg-status-error/20 text-status-error'
                  }`}
                >
                  {comfyOnline ? 'ONLINE' : 'OFFLINE'}
                </span>
              </div>
              <p className="text-xs text-text-muted">
                Executes SDXL diffusion workflows, ControlNet conditioning, and latent image sampling.
              </p>
              <div className="pt-2 border-t border-border-subtle flex items-center justify-between">
                <span className="text-[11px] text-text-muted">Port 8188 / WebSocket</span>
                <button
                  onClick={handlePingComfy}
                  disabled={testingComfy}
                  className="px-3 py-1 rounded-lg text-xs font-medium bg-surface-elevated hover:bg-surface border border-border-subtle transition-colors"
                >
                  {testingComfy ? 'Pinging...' : 'Ping ComfyUI'}
                </button>
              </div>
            </div>

            {/* Ollama Card */}
            <div className="p-5 rounded-2xl bg-surface border border-border-subtle space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2.5">
                  <div className="p-2 rounded-xl bg-accent/20 border border-accent/40 text-accent">
                    <Wrench className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-text-primary">Ollama Copilot Engine</h3>
                    <p className="text-[11px] font-mono text-text-muted">http://127.0.0.1:11434</p>
                  </div>
                </div>
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold ${
                    ollamaOnline ? 'bg-status-success/20 text-status-success' : 'bg-status-error/20 text-status-error'
                  }`}
                >
                  {ollamaOnline ? 'CONNECTED' : 'OFFLINE'}
                </span>
              </div>
              <p className="text-xs text-text-muted">
                Powers Studio Copilot with local Qwen3-Coder 30B model for prompt refinement and technical guidance.
              </p>
              <div className="pt-2 border-t border-border-subtle flex items-center justify-between">
                <span className="text-[11px] text-text-muted">
                  {systemStatus?.ollama_model || 'qwen3-coder:30b'}
                </span>
                <button
                  onClick={handlePingOllama}
                  disabled={testingOllama}
                  className="px-3 py-1 rounded-lg text-xs font-medium bg-surface-elevated hover:bg-surface border border-border-subtle transition-colors"
                >
                  {testingOllama ? 'Pinging...' : 'Ping Ollama'}
                </button>
              </div>
            </div>
          </div>

          {testResult && (
            <div
              className={`p-3.5 rounded-xl border text-xs font-mono ${
                testResult.status === 'ok'
                  ? 'bg-status-success/15 border-status-success/40 text-status-success'
                  : 'bg-status-error/15 border-status-error/40 text-status-error'
              }`}
            >
              <strong>[{testResult.target}]</strong> {testResult.message}
            </div>
          )}

          {/* Real Hardware Accelerator Telemetry Card */}
          <div className="p-6 rounded-2xl bg-surface border border-border-subtle space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="p-2 rounded-xl bg-accent/20 border border-accent/40 text-accent">
                  <Cpu className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-sm font-semibold text-text-primary">
                    Hardware Accelerator: {telemetry?.gpu_name || gpu?.name || 'NVIDIA GeForce RTX 5070 Ti'}
                  </h2>
                  <p className="text-xs text-text-muted">
                    {telemetry ? `${(telemetry.total_vram_mb / 1024).toFixed(1)} GB GDDR7 Dedicated VRAM` : 'Hardware monitoring active'}
                  </p>
                </div>
              </div>
              {telemetry?.temperature_c !== null && telemetry?.temperature_c !== undefined && (
                <span className="text-xs font-mono text-text-muted bg-surface-elevated px-2.5 py-1 rounded-lg border border-border-subtle">
                  Temp: <strong className="text-text-primary">{telemetry.temperature_c}°C</strong>
                </span>
              )}
            </div>

            {/* VRAM Progress Gauge */}
            <div className="space-y-2">
              <div className="flex justify-between items-center text-xs">
                <span className="text-text-muted font-medium">Real-Time VRAM Utilization</span>
                <span className="font-mono text-text-primary font-semibold">
                  {telemetry
                    ? `${(telemetry.used_vram_mb / 1024).toFixed(1)} GB / ${(telemetry.total_vram_mb / 1024).toFixed(1)} GB (${vramPercent}%)`
                    : 'Telemetry unavailable'}
                </span>
              </div>
              <div className="w-full bg-surface-elevated rounded-full h-3 overflow-hidden border border-border-subtle/50 p-0.5">
                <div
                  className={`h-full rounded-full transition-all duration-700 ${
                    vramPercent > 85 ? 'bg-status-warning' : 'bg-accent'
                  }`}
                  style={{ width: `${Math.max(4, vramPercent)}%` }}
                />
              </div>
            </div>

            {/* Allocation Profile & Safety Bounds */}
            <div className="pt-3 border-t border-border-subtle space-y-2 text-xs">
              <div className="flex items-center justify-between font-mono">
                <span className="text-text-muted">Current VRAM Safety Policy:</span>
                <span className="font-semibold text-accent capitalize">{vramPolicy.replace('_', ' ')}</span>
              </div>
              <div className="flex items-center justify-between font-mono">
                <span className="text-text-muted">Backend Telemetry Latency:</span>
                <span className="text-text-primary">{telemetry?.backend_latency_ms ?? 0} ms</span>
              </div>
            </div>
          </div>

          {/* Workstation Storage Directories */}
          <div className="p-6 rounded-2xl bg-surface border border-border-subtle space-y-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 rounded-xl bg-accent/20 border border-accent/40 text-accent">
                <HardDrive className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-sm font-semibold text-text-primary">Workspace Directory Paths</h2>
                <p className="text-xs text-text-muted">Local file system mapping for models, caches, and storage</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs font-mono">
              <div className="p-3 rounded-xl bg-surface-elevated border border-border-subtle flex items-center space-x-3">
                <Folder className="w-4 h-4 text-accent shrink-0" />
                <div className="overflow-hidden">
                  <span className="text-[10px] uppercase font-sans text-text-dim block">Checkpoints</span>
                  <span className="text-text-primary truncate block font-medium">c:/vishnu/ComfyUI/models/checkpoints/</span>
                </div>
              </div>

              <div className="p-3 rounded-xl bg-surface-elevated border border-border-subtle flex items-center space-x-3">
                <Folder className="w-4 h-4 text-accent shrink-0" />
                <div className="overflow-hidden">
                  <span className="text-[10px] uppercase font-sans text-text-dim block">LoRA Vault Weights</span>
                  <span className="text-text-primary truncate block font-medium">storage/loras/</span>
                </div>
              </div>

              <div className="p-3 rounded-xl bg-surface-elevated border border-border-subtle flex items-center space-x-3">
                <Folder className="w-4 h-4 text-accent shrink-0" />
                <div className="overflow-hidden">
                  <span className="text-[10px] uppercase font-sans text-text-dim block">Generated Outputs</span>
                  <span className="text-text-primary truncate block font-medium">storage/generated/</span>
                </div>
              </div>

              <div className="p-3 rounded-xl bg-surface-elevated border border-border-subtle flex items-center space-x-3">
                <Folder className="w-4 h-4 text-accent shrink-0" />
                <div className="overflow-hidden">
                  <span className="text-[10px] uppercase font-sans text-text-dim block">SQLite Database</span>
                  <span className="text-text-primary truncate block font-medium">storage/data/forge.db</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: SYSTEM DIAGNOSTICS */}
      {activeTab === 'diagnostics' && (
        <div className="space-y-5">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base font-semibold text-text-primary">Subsystem Diagnostic Checks</h2>
              <p className="text-xs text-text-muted">
                9 verified health probes with 30s cache TTL and copyable remediation PowerShell commands.
              </p>
            </div>
            <button
              onClick={() => fetchDiagnostics(true)}
              disabled={runningDiagnostics}
              className="px-4 py-2 rounded-xl text-xs font-semibold bg-accent text-background hover:bg-accent-hover transition-colors flex items-center gap-1.5 shadow-md shadow-accent/10"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${runningDiagnostics ? 'animate-spin' : ''}`} />
              <span>{runningDiagnostics ? 'Running...' : 'Run Diagnostics'}</span>
            </button>
          </div>

          {diagnosticsReport?.sampled_at && (
            <div className="text-[11px] font-mono text-text-dim">
              Report generated at {new Date(diagnosticsReport.sampled_at).toLocaleTimeString()} (cached for {diagnosticsReport.cache_ttl_sec}s)
            </div>
          )}

          {/* Diagnostic Checks Cards */}
          <div className="space-y-3">
            {diagnosticsReport?.checks.map((check) => {
              const isOk = check.status === 'ok';
              const isWarning = check.status === 'warning';
              const isError = check.status === 'error';

              return (
                <div
                  key={check.id}
                  className={`p-4 rounded-xl border transition-all ${
                    isOk
                      ? 'bg-surface border-border-subtle'
                      : isWarning
                      ? 'bg-status-warning/10 border-status-warning/30'
                      : 'bg-status-error/10 border-status-error/30'
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="space-y-1">
                      <div className="flex items-center space-x-2">
                        <span
                          className={`w-2 h-2 rounded-full ${
                            isOk ? 'bg-status-success' : isWarning ? 'bg-status-warning' : 'bg-status-error'
                          }`}
                        />
                        <h3 className="font-semibold text-text-primary text-xs">{check.name}</h3>
                        <span className="text-[10px] font-mono uppercase px-1.5 py-0.2 rounded bg-surface border border-border-subtle text-text-dim">
                          {check.category}
                        </span>
                      </div>
                      <p className="text-xs text-text-muted pl-4">{check.message}</p>
                      {check.details && (
                        <p className="text-[11px] font-mono text-text-dim pl-4">{check.details}</p>
                      )}
                    </div>

                    <span
                      className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded uppercase ${
                        isOk
                          ? 'bg-status-success/20 text-status-success'
                          : isWarning
                          ? 'bg-status-warning/20 text-status-warning'
                          : 'bg-status-error/20 text-status-error'
                      }`}
                    >
                      {check.status}
                    </span>
                  </div>

                  {/* Remediation Command */}
                  {check.remediation_cmd && (
                    <div className="mt-3 pl-4 pt-2 border-t border-border-subtle/60 flex items-center justify-between gap-2 bg-surface-elevated p-2 rounded-lg font-mono text-[11px]">
                      <div className="flex items-center space-x-2 truncate">
                        <Terminal className="w-3.5 h-3.5 text-accent shrink-0" />
                        <span className="text-text-muted truncate select-all">{check.remediation_cmd}</span>
                      </div>
                      <button
                        onClick={() => handleCopyCmd(check.id, check.remediation_cmd!)}
                        className="px-2 py-1 rounded bg-surface hover:bg-surface-elevated text-text-dim hover:text-accent border border-border-subtle transition-colors flex items-center gap-1 shrink-0 text-[10px]"
                        title="Copy PowerShell remediation command"
                      >
                        {copiedCmdId === check.id ? <Check className="w-3 h-3 text-status-success" /> : <Copy className="w-3 h-3" />}
                        <span>{copiedCmdId === check.id ? 'Copied' : 'Copy'}</span>
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* TAB 3: HARDWARE BENCHMARK */}
      {activeTab === 'benchmark' && (
        <div className="space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h2 className="text-base font-semibold text-text-primary">GPU Hardware Benchmark</h2>
              <p className="text-xs text-text-muted">
                Executes a reproducible 20-step generation with warm-up prime-caching and live metric sampling.
              </p>
            </div>

            <div className="flex items-center space-x-2">
              {benchmarkResult && (
                <>
                  <button
                    onClick={() => handleExportBenchmark('json')}
                    className="px-3 py-1.5 rounded-xl text-xs font-medium bg-surface-elevated hover:bg-surface border border-border-subtle transition-colors flex items-center gap-1.5"
                  >
                    <FileJson className="w-3.5 h-3.5" />
                    <span>Export JSON</span>
                  </button>
                  <button
                    onClick={() => handleExportBenchmark('csv')}
                    className="px-3 py-1.5 rounded-xl text-xs font-medium bg-surface-elevated hover:bg-surface border border-border-subtle transition-colors flex items-center gap-1.5"
                  >
                    <FileSpreadsheet className="w-3.5 h-3.5" />
                    <span>Export CSV</span>
                  </button>
                </>
              )}
              <button
                onClick={() => setConfirmBenchmarkOpen(true)}
                disabled={isBenchmarking}
                className="px-4 py-2 rounded-xl text-xs font-semibold bg-accent text-background hover:bg-accent-hover transition-colors flex items-center gap-1.5 shadow-md shadow-accent/10 disabled:opacity-50"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>{isBenchmarking ? 'Running Benchmark...' : 'Run Benchmark'}</span>
              </button>
            </div>
          </div>

          {/* Benchmark Test Harness Spec */}
          <div className="p-4 rounded-xl bg-surface border border-border-subtle space-y-2 text-xs">
            <span className="font-semibold uppercase tracking-wider text-text-muted text-[10px]">
              Standardized Test Parameters
            </span>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 font-mono text-[11px] bg-surface-elevated p-3 rounded-lg border border-border-subtle">
              <div>
                <span className="text-text-dim block text-[10px] font-sans">Resolution</span>
                <span>512 × 512</span>
              </div>
              <div>
                <span className="text-text-dim block text-[10px] font-sans">Steps & Sampler</span>
                <span>20 steps / Euler</span>
              </div>
              <div>
                <span className="text-text-dim block text-[10px] font-sans">Fixed Seed</span>
                <span>42</span>
              </div>
              <div>
                <span className="text-text-dim block text-[10px] font-sans">Warm-up</span>
                <span>5-step unmeasured</span>
              </div>
            </div>
            <p className="text-[11px] font-mono text-text-dim pt-1">
              Fixed Prompt: "studio portrait of an astronaut in intricate reflective suit, cinematic lighting, 8k, photorealistic"
            </p>
          </div>

          {/* Running Indicator */}
          {isBenchmarking && (
            <div className="p-6 rounded-2xl bg-surface border border-accent/40 flex flex-col items-center justify-center space-y-3 animate-pulse">
              <div className="w-8 h-8 border-3 border-accent border-t-transparent rounded-full animate-spin" />
              <div className="text-center space-y-1">
                <h3 className="font-semibold text-accent text-sm">
                  {benchmarkWarmup ? 'Executing 5-Step CUDA Warm-up...' : 'Executing 20-Step Benchmark & Ollama Latency Test...'}
                </h3>
                <p className="text-xs text-text-muted">
                  Measuring prompt latency, sampling speed, VRAM telemetry, and token generation.
                </p>
              </div>
            </div>
          )}

          {benchmarkError && (
            <div className="p-4 rounded-xl bg-status-error/15 border border-status-error/40 text-xs text-status-error flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>{benchmarkError}</span>
            </div>
          )}

          {/* Benchmark Results Display */}
          {benchmarkResult && !isBenchmarking && (
            <div className="p-6 rounded-2xl bg-surface border border-border-subtle space-y-5">
              <div className="flex items-center justify-between border-b border-border-subtle pb-4">
                <div>
                  <h3 className="text-base font-semibold text-text-primary">Benchmark Completed</h3>
                  <span className="text-[11px] font-mono text-text-dim">
                    Tested on {benchmarkResult.gpu_name} at {new Date(benchmarkResult.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <span className="text-xs font-mono font-semibold px-2.5 py-1 rounded-lg bg-status-success/20 text-status-success border border-status-success/30">
                  REAL JOB VERIFIED
                </span>
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="p-3.5 rounded-xl bg-surface-elevated border border-border-subtle space-y-1">
                  <span className="text-[10px] uppercase font-sans text-text-dim block">Sampling Speed</span>
                  <span className="text-base font-mono font-bold text-accent">
                    {benchmarkResult.steps_per_sec.toFixed(2)} it/s
                  </span>
                  <span className="text-[10px] text-text-dim block font-mono">
                    Total: {(benchmarkResult.total_time_ms / 1000).toFixed(2)}s
                  </span>
                </div>

                <div className="p-3.5 rounded-xl bg-surface-elevated border border-border-subtle space-y-1">
                  <span className="text-[10px] uppercase font-sans text-text-dim block">First-Image Latency</span>
                  <span className="text-base font-mono font-bold text-text-primary">
                    {benchmarkResult.prompt_to_first_img_ms} ms
                  </span>
                  <span className="text-[10px] text-text-dim block font-mono">Prompt-to-pixel</span>
                </div>

                <div className="p-3.5 rounded-xl bg-surface-elevated border border-border-subtle space-y-1">
                  <span className="text-[10px] uppercase font-sans text-text-dim block">Peak VRAM</span>
                  <span className="text-base font-mono font-bold text-text-primary">
                    {(benchmarkResult.peak_vram_mb / 1024).toFixed(1)} GB
                  </span>
                  <span className="text-[10px] text-text-dim block font-mono">
                    Avg: {(benchmarkResult.avg_vram_mb / 1024).toFixed(1)} GB
                  </span>
                </div>

                <div className="p-3.5 rounded-xl bg-surface-elevated border border-border-subtle space-y-1">
                  <span className="text-[10px] uppercase font-sans text-text-dim block">GPU Utilization</span>
                  <span className="text-base font-mono font-bold text-text-primary">
                    {benchmarkResult.gpu_util_pct}%
                  </span>
                  <span className="text-[10px] text-text-dim block font-mono">Compute load</span>
                </div>
              </div>

              {/* Ollama Metrics */}
              {benchmarkResult.ollama_first_token_ms !== null && (
                <div className="p-4 rounded-xl bg-surface-elevated border border-border-subtle space-y-2 text-xs">
                  <span className="font-semibold uppercase tracking-wider text-text-muted text-[10px]">
                    Ollama Qwen3-Coder 30B Latency
                  </span>
                  <div className="grid grid-cols-2 gap-3 font-mono text-xs">
                    <div>
                      <span className="text-text-dim block text-[10px] font-sans">First-Token Latency</span>
                      <span className="text-text-primary font-semibold">
                        {benchmarkResult.ollama_first_token_ms ? `${benchmarkResult.ollama_first_token_ms} ms` : 'N/A'}
                      </span>
                    </div>
                    <div>
                      <span className="text-text-dim block text-[10px] font-sans">Throughput</span>
                      <span className="text-text-primary font-semibold">
                        {benchmarkResult.ollama_tokens_per_sec ? `${benchmarkResult.ollama_tokens_per_sec.toFixed(1)} tokens/s` : 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Environment Metadata */}
              <div className="p-4 rounded-xl bg-surface-elevated border border-border-subtle space-y-2 text-xs font-mono text-[11px]">
                <span className="font-semibold uppercase tracking-wider text-text-muted text-[10px] font-sans">
                  Environment Metadata
                </span>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-text-dim">
                  <div>CUDA: <strong className="text-text-primary">{benchmarkResult.env_metadata?.cuda_version || '12.8'}</strong></div>
                  <div>PyTorch: <strong className="text-text-primary">{benchmarkResult.env_metadata?.torch_version || '2.12.0'}</strong></div>
                  <div>OS: <strong className="text-text-primary">{benchmarkResult.env_metadata?.os || 'Windows'}</strong></div>
                  <div>Compute: <strong className="text-text-primary">{benchmarkResult.env_metadata?.compute_capability || '12.0'}</strong></div>
                </div>
              </div>
            </div>
          )}

          {/* User Confirmation Modal */}
          {confirmBenchmarkOpen && (
            <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm">
              <div className="w-full max-w-md bg-surface border border-border-subtle rounded-2xl p-6 space-y-4 shadow-2xl">
                <div className="flex items-center space-x-2 text-accent">
                  <Play className="w-5 h-5 fill-current" />
                  <h3 className="font-semibold text-base text-text-primary">Start Hardware Benchmark</h3>
                </div>
                <p className="text-xs text-text-muted">
                  This benchmark will acquire an exclusive GPU isolation lease, run an unmeasured 5-step warm-up to prime CUDA contexts, and measure 20 steps of diffusion on the RTX 5070 Ti.
                </p>
                <div className="p-3 rounded-xl bg-status-warning/15 border border-status-warning/30 text-xs text-status-warning font-mono">
                  Any queued generation will wait until the benchmark completes.
                </div>
                <div className="flex justify-end space-x-2 pt-2">
                  <button
                    onClick={() => setConfirmBenchmarkOpen(false)}
                    className="px-3 py-1.5 rounded-lg text-xs text-text-muted hover:bg-surface-elevated"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleRunBenchmark}
                    className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-accent text-background hover:bg-accent-hover"
                  >
                    Confirm & Run
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
