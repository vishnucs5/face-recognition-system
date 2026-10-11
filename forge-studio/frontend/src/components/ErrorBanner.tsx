import React, { useState } from 'react';
import { AlertCircle, X, RefreshCw, Copy, Check, Terminal, ShieldAlert } from 'lucide-react';
import { useStudioStore } from '../store/useStudioStore';

export const ErrorBanner: React.FC = () => {
  const { errorMessage, setErrorMessage, isGenerating, isSubmitting } = useStudioStore();
  const [copied, setCopied] = useState(false);

  if (!errorMessage) return null;

  // Classify subsystem and remediation heuristically from sanitized message
  const lower = errorMessage.toLowerCase();
  let subsystem = 'Engine';
  let isTransient = false;
  let remediationCmd: string | null = null;

  if (lower.includes('out of memory') || lower.includes('vram') || lower.includes('cuda oom')) {
    subsystem = 'GPU';
    isTransient = true;
    remediationCmd = 'nvidia-smi --gpu-reset';
  } else if (lower.includes('comfyui') || lower.includes('8188')) {
    subsystem = 'ComfyUI';
    isTransient = true;
    remediationCmd = 'Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd c:\\vishnu\\ComfyUI; c:\\vishnu\\.venv\\Scripts\\python.exe main.py --listen 127.0.0.1 --port 8188"';
  } else if (lower.includes('ollama') || lower.includes('11434')) {
    subsystem = 'Ollama';
    isTransient = true;
    remediationCmd = 'ollama run qwen3-coder:30b';
  } else if (lower.includes('timeout') || lower.includes('timed out')) {
    subsystem = 'Network';
    isTransient = true;
  } else if (lower.includes('prompt') || lower.includes('dimension') || lower.includes('lora')) {
    subsystem = 'Validation';
    isTransient = false;
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(errorMessage);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="mx-6 mt-4 p-4 rounded-xl bg-status-error/10 border border-status-error/30 text-xs text-text-primary animate-in fade-in slide-in-from-top-2 duration-200 space-y-2.5">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start space-x-3">
          <div className="w-6 h-6 rounded-lg bg-status-error/20 flex items-center justify-center shrink-0 text-status-error mt-0.5">
            <AlertCircle className="w-4 h-4" />
          </div>
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <span className="font-semibold text-status-error">
                {subsystem} Notice
              </span>
              <span className="text-[10px] font-mono uppercase px-1.5 py-0.2 rounded bg-surface border border-status-error/30 text-text-muted">
                {isTransient ? 'Transient' : 'Configuration'}
              </span>
            </div>
            <p className="text-text-muted select-text">{errorMessage}</p>
          </div>
        </div>

        <div className="flex items-center space-x-1.5 shrink-0">
          <button
            onClick={handleCopy}
            className="px-2.5 py-1 rounded-lg bg-surface border border-border-subtle hover:bg-surface-elevated text-text-dim hover:text-text-primary transition-colors flex items-center gap-1 font-mono text-[10px]"
            title="Copy sanitized error details (no credentials or profile paths)"
          >
            {copied ? <Check className="w-3 h-3 text-status-success" /> : <Copy className="w-3 h-3" />}
            <span>{copied ? 'Copied' : 'Copy Error Details'}</span>
          </button>
          <button
            onClick={() => setErrorMessage(null)}
            className="text-text-dim hover:text-text-primary p-1 rounded-lg transition-colors"
            title="Dismiss notice"
            aria-label="Dismiss error notice"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Copyable Remediation Guidance */}
      {remediationCmd && (
        <div className="pl-9 pt-1 flex items-center justify-between gap-2 bg-surface p-2 rounded-lg font-mono text-[11px] border border-border-subtle">
          <div className="flex items-center space-x-2 truncate">
            <Terminal className="w-3.5 h-3.5 text-accent shrink-0" />
            <span className="text-text-muted truncate select-all">{remediationCmd}</span>
          </div>
          <button
            onClick={() => {
              navigator.clipboard.writeText(remediationCmd!);
              setCopied(true);
              setTimeout(() => setCopied(false), 2000);
            }}
            className="px-2 py-0.5 rounded bg-surface-elevated hover:bg-surface text-text-dim hover:text-accent border border-border-subtle transition-colors shrink-0 text-[10px]"
            title="Copy PowerShell remediation command"
          >
            Copy Command
          </button>
        </div>
      )}
    </div>
  );
};
