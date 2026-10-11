import React, { useEffect, useState } from 'react';
import {
  Cpu,
  FolderOpen,
  CheckCircle2,
  AlertTriangle,
  Play,
  RotateCcw,
  Sparkles,
  Info,
  Terminal,
  Layers,
  ArrowRight,
  ShieldCheck,
  FileText,
  Sliders,
  X,
  RefreshCw,
  Clock,
  Ban,
  Check,
  Zap,
  Copy,
} from 'lucide-react';
import { useStudioStore } from '../store/useStudioStore';
import {
  CaptionPreviewResponse,
  DatasetValidationResponse,
  TrainingHealthResponse,
  TrainingJobResponse,
} from '../types';

export const TrainingView: React.FC = () => {
  const { setActiveNav, setErrorMessage } = useStudioStore();

  const [health, setHealth] = useState<TrainingHealthResponse | null>(null);
  const [loadingHealth, setLoadingHealth] = useState(true);

  // Wizard Step: 1 = Goal, 2 = Dataset, 3 = Captions, 4 = Config, 5 = Monitor
  const [activeStep, setActiveStep] = useState<1 | 2 | 3 | 4 | 5>(1);

  // Goal
  const [goal, setGoal] = useState<'character' | 'product' | 'style' | 'object' | 'custom'>('character');

  // Dataset
  const [datasetPath, setDatasetPath] = useState('');
  const [validatingDataset, setValidatingDataset] = useState(false);
  const [datasetReport, setDatasetReport] = useState<DatasetValidationResponse | null>(null);

  // Captions
  const [captionPreview, setCaptionPreview] = useState<CaptionPreviewResponse | null>(null);
  const [previewingCaptions, setPreviewingCaptions] = useState(false);
  const [applyingCaptions, setApplyingCaptions] = useState(false);
  const [captionsAppliedSuccess, setCaptionsAppliedSuccess] = useState(false);

  // Training Config
  const [loraName, setLoraName] = useState('cyber_character_lora');
  const [triggerWord, setTriggerWord] = useState('cbr_char');
  const [baseModel, setBaseModel] = useState('v1-5-pruned-emaonly.safetensors');
  const [resolution, setResolution] = useState(1024);
  const [networkDim, setNetworkDim] = useState(16);
  const [networkAlpha, setNetworkAlpha] = useState(16);
  const [learningRate, setLearningRate] = useState(0.0001);
  const [textEncoderLr, setTextEncoderLr] = useState(0.00005);
  const [batchSize, setBatchSize] = useState(1);
  const [repeats, setRepeats] = useState(10);
  const [epochs, setEpochs] = useState(10);
  const [maxSteps, setMaxSteps] = useState(1000);
  const [saveInterval, setSaveInterval] = useState(2);
  const [seed, setSeed] = useState(42);
  const [mixedPrecision, setMixedPrecision] = useState<'bf16' | 'fp16' | 'no'>('bf16');
  const [gradientCheckpointing, setGradientCheckpointing] = useState(true);

  // Jobs
  const [jobs, setJobs] = useState<TrainingJobResponse[]>([]);
  const [isSubmittingTraining, setIsSubmittingTraining] = useState(false);
  const [selectedJob, setSelectedJob] = useState<TrainingJobResponse | null>(null);
  const [logsOpen, setLogsOpen] = useState(false);
  const [installTab, setInstallTab] = useState<'ai-toolkit' | 'kohya_ss'>('ai-toolkit');
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);

  const copyCommand = (cmd: string, key: string) => {
    navigator.clipboard.writeText(cmd);
    setCopiedCmd(key);
    setTimeout(() => setCopiedCmd(null), 2000);
  };

  useEffect(() => {
    fetchHealth();
    fetchJobs();
    const interval = setInterval(() => {
      fetchJobs();
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  const fetchHealth = async () => {
    setLoadingHealth(true);
    try {
      const res = await fetch('/api/training/health');
      if (res.ok) {
        const data: TrainingHealthResponse = await res.json();
        setHealth(data);
      }
    } catch (err) {
      console.warn('Failed to fetch training health:', err);
    } finally {
      setLoadingHealth(false);
    }
  };

  const fetchJobs = async () => {
    try {
      const res = await fetch('/api/training/jobs');
      if (res.ok) {
        const data: TrainingJobResponse[] = await res.json();
        setJobs(data);
        if (data.length > 0 && (!selectedJob || !data.some((j) => j.id === selectedJob.id))) {
          setSelectedJob(data[0]);
        }
      }
    } catch (err) {
      console.warn('Failed to fetch training jobs:', err);
    }
  };

  const handleValidateDataset = async () => {
    if (!datasetPath.trim()) {
      setErrorMessage('Please specify an absolute folder path containing your training images.');
      return;
    }
    setValidatingDataset(true);
    setDatasetReport(null);
    try {
      const res = await fetch('/api/training/validate-dataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_path: datasetPath.trim() }),
      });
      if (res.ok) {
        const report: DatasetValidationResponse = await res.json();
        setDatasetReport(report);
        if (report.suggested_resolution) {
          setResolution(report.suggested_resolution);
        }
      } else {
        const err = await res.json();
        setErrorMessage(err.detail || 'Failed to inspect dataset directory.');
      }
    } catch (err: any) {
      setErrorMessage(err?.message || 'Error communicating with backend dataset validator.');
    } finally {
      setValidatingDataset(false);
    }
  };

  const handlePreviewCaptions = async (action: 'inspect' | 'prepend_trigger') => {
    if (!datasetPath.trim()) return;
    setPreviewingCaptions(true);
    setCaptionsAppliedSuccess(false);
    try {
      const res = await fetch('/api/training/preview-captions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_path: datasetPath.trim(),
          trigger_word: triggerWord.trim() || undefined,
          action,
        }),
      });
      if (res.ok) {
        const data: CaptionPreviewResponse = await res.json();
        setCaptionPreview(data);
      }
    } catch (err) {
      console.error('Failed to preview captions:', err);
    } finally {
      setPreviewingCaptions(false);
    }
  };

  const handleApplyCaptions = async () => {
    if (!captionPreview || !datasetPath.trim()) return;
    setApplyingCaptions(true);
    try {
      const updates = captionPreview.items.map((it) => ({
        caption_file: it.caption_file,
        content: it.proposed_caption,
      }));
      const res = await fetch('/api/training/apply-captions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_path: datasetPath.trim(),
          updates,
          backup_existing: true,
        }),
      });
      if (res.ok) {
        setCaptionsAppliedSuccess(true);
        // Re-validate dataset to clear missing captions warning
        handleValidateDataset();
      } else {
        const err = await res.json();
        setErrorMessage(err.detail || 'Failed to apply captions.');
      }
    } catch (err: any) {
      setErrorMessage(err?.message || 'Error applying captions.');
    } finally {
      setApplyingCaptions(false);
    }
  };

  const handleStartTraining = async () => {
    if (!health?.engine_available) {
      setErrorMessage(
        'Training engine unavailable — install kohya_ss or AI-Toolkit locally to enable training.'
      );
      return;
    }
    if (!datasetPath.trim()) {
      setErrorMessage('Please validate a dataset directory before starting training.');
      return;
    }
    if (!loraName.trim()) {
      setErrorMessage('Please provide a valid output LoRA name.');
      return;
    }

    setIsSubmittingTraining(true);
    try {
      const payload = {
        goal,
        dataset_path: datasetPath.trim(),
        base_model: baseModel,
        lora_name: loraName.trim(),
        trigger_word: triggerWord.trim(),
        resolution,
        network_dim: networkDim,
        network_alpha: networkAlpha,
        learning_rate: learningRate,
        text_encoder_lr: textEncoderLr,
        batch_size: batchSize,
        repeats,
        epochs,
        max_train_steps: maxSteps,
        save_interval: saveInterval,
        seed,
        mixed_precision: mixedPrecision,
        gradient_checkpointing: gradientCheckpointing,
        captioning_mode: 'txt',
      };

      const res = await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        setErrorMessage(err.detail || 'Failed to initiate training job.');
        setIsSubmittingTraining(false);
        return;
      }

      const job: TrainingJobResponse = await res.json();
      setJobs([job, ...jobs]);
      setSelectedJob(job);
      setActiveStep(5); // Switch to monitor view
    } catch (err: any) {
      setErrorMessage(err?.message || 'Network error initiating training job.');
    } finally {
      setIsSubmittingTraining(false);
    }
  };

  const handleCancelJob = async (jobId: string) => {
    try {
      const res = await fetch(`/api/training/jobs/${jobId}/cancel`, { method: 'POST' });
      if (res.ok) {
        fetchJobs();
      }
    } catch (err) {
      console.error('Failed to cancel job:', err);
    }
  };

  const handleRetryJob = async (jobId: string) => {
    try {
      const res = await fetch(`/api/training/jobs/${jobId}/retry`, { method: 'POST' });
      if (res.ok) {
        const newJob: TrainingJobResponse = await res.json();
        setJobs([newJob, ...jobs]);
        setSelectedJob(newJob);
      }
    } catch (err) {
      console.error('Failed to retry job:', err);
    }
  };

  const handleImportArtifactToVault = async (jobId: string) => {
    try {
      const res = await fetch(`/api/training/jobs/${jobId}/import-vault`, { method: 'POST' });
      if (res.ok) {
        alert('Artifact successfully imported into LoRA Vault!');
        setActiveNav('loras');
      } else {
        const err = await res.json();
        setErrorMessage(err.detail || 'Failed to import artifact to vault.');
      }
    } catch (err: any) {
      setErrorMessage(err?.message || 'Error importing artifact.');
    }
  };

  const goals = [
    { id: 'character', label: 'Character', desc: 'Consistent faces, outfits, and distinctive personality' },
    { id: 'product', label: 'Product', desc: 'Precision industrial objects, packaging, and renders' },
    { id: 'style', label: 'Style / Art', desc: 'Consistent artistic aesthetics, lighting, and palette' },
    { id: 'object', label: 'Object', desc: 'Specific props, vehicles, or botanical entities' },
    { id: 'custom', label: 'Custom', desc: 'Advanced configuration with custom parameters' },
  ];

  return (
    <div className="flex-1 p-6 md:p-10 overflow-y-auto space-y-8 max-w-6xl select-none">
      {/* 1. Header & Engine Telemetry */}
      <div className="space-y-4 border-b border-border-subtle pb-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-text-primary flex items-center gap-2.5">
              <Cpu className="w-6 h-6 text-accent" />
              <span>LoRA Training Studio</span>
              <span className="text-xs font-mono font-normal px-2 py-0.5 rounded bg-surface-elevated text-accent border border-border-subtle">
                RTX 5070 Ti Profile
              </span>
            </h1>
            <p className="text-xs text-text-muted mt-1">
              Fine-tune custom LoRA adapters locally using hardware-accelerated mixed precision.
            </p>
          </div>

          <div className="flex items-center space-x-2.5">
            <button
              onClick={fetchHealth}
              className="p-2 rounded-xl bg-surface-elevated hover:bg-surface border border-border-subtle text-text-muted hover:text-text-primary transition-colors"
              title="Refresh engine status"
            >
              <RefreshCw className={`w-4 h-4 ${loadingHealth ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={() => setActiveNav('loras')}
              className="flex items-center space-x-1.5 px-3 py-2 rounded-xl text-xs font-medium bg-surface-elevated hover:bg-surface border border-border-subtle text-accent transition-colors"
            >
              <Layers className="w-3.5 h-3.5" />
              <span>LoRA Vault</span>
            </button>
          </div>
        </div>

        {/* Engine Status Callout */}
        <div
          className={`p-4 rounded-xl border flex items-center justify-between text-xs ${
            health?.engine_available
              ? 'bg-status-success/10 border-status-success/30 text-status-success'
              : 'bg-status-error/10 border-status-error/30 text-status-error'
          }`}
        >
          <div className="flex items-center space-x-2.5">
            {health?.engine_available ? (
              <CheckCircle2 className="w-4 h-4 shrink-0 text-status-success" />
            ) : (
              <AlertTriangle className="w-4 h-4 shrink-0 text-status-error" />
            )}
            <div>
              <span className="font-semibold block">
                {health?.engine_available
                  ? `Training Engine Ready (${health.engine_name})`
                  : 'Training Engine Unavailable'}
              </span>
              <p className="text-[11px] opacity-90 font-mono mt-0.5">
                {health?.message || 'Detecting engine configuration...'}
              </p>
            </div>
          </div>

          <div className="hidden sm:flex items-center space-x-3 text-[11px] font-mono shrink-0">
            <span className="px-2 py-0.5 rounded bg-background/50 border border-border-subtle">
              VRAM: {health?.gpu_vram_gb || 16} GB
            </span>
            <span className="px-2 py-0.5 rounded bg-background/50 border border-border-subtle">
              Mixed: bf16/fp16
            </span>
          </div>
        </div>

        {/* Engine Installation Guide (Visible when engine is unavailable) */}
        {!health?.engine_available && (
          <div className="p-5 rounded-2xl bg-surface border border-border-subtle space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="space-y-0.5">
                <div className="flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-accent" />
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-text-primary font-mono">
                    Recommended Engine Installation
                  </h3>
                </div>
                <p className="text-xs text-text-muted">
                  Forge Studio does not silently modify packages. Choose an engine below and run the commands in PowerShell:
                </p>
              </div>

              {/* Tab Selector */}
              <div className="flex items-center bg-surface-elevated p-1 rounded-xl border border-border-subtle text-xs shrink-0">
                <button
                  onClick={() => setInstallTab('ai-toolkit')}
                  className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                    installTab === 'ai-toolkit'
                      ? 'bg-surface text-accent font-semibold shadow-sm border border-border-subtle'
                      : 'text-text-muted hover:text-text-primary'
                  }`}
                >
                  AI-Toolkit (Recommended)
                </button>
                <button
                  onClick={() => setInstallTab('kohya_ss')}
                  className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                    installTab === 'kohya_ss'
                      ? 'bg-surface text-accent font-semibold shadow-sm border border-border-subtle'
                      : 'text-text-muted hover:text-text-primary'
                  }`}
                >
                  kohya_ss / sd-scripts
                </button>
              </div>
            </div>

            {installTab === 'ai-toolkit' ? (
              <div className="space-y-3">
                <div className="p-3.5 rounded-xl bg-surface-elevated border border-border-subtle text-xs space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-text-primary font-mono text-[11px]">
                      Option A: Install AI-Toolkit (Native SDXL / RTX 5070 Ti bf16)
                    </span>
                    <button
                      onClick={() =>
                        copyCommand(
                          `git clone https://github.com/ostris/ai-toolkit.git c:\\vishnu\\ai-toolkit\ncd c:\\vishnu\\ai-toolkit\ngit submodule update --init --recursive\npython -m venv venv\n.\\venv\\Scripts\\Activate.ps1\npip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\npip install -r requirements.txt`,
                          'ait'
                        )
                      }
                      className="flex items-center gap-1 px-2.5 py-1 rounded bg-surface hover:bg-surface-elevated border border-border-subtle text-accent text-[11px] font-mono transition-colors"
                    >
                      {copiedCmd === 'ait' ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                      <span>{copiedCmd === 'ait' ? 'Copied All' : 'Copy All'}</span>
                    </button>
                  </div>
                  <pre className="font-mono text-[11px] text-text-muted bg-background/80 p-3 rounded-lg overflow-x-auto whitespace-pre leading-relaxed border border-border-subtle">
{`git clone https://github.com/ostris/ai-toolkit.git c:\\vishnu\\ai-toolkit
cd c:\\vishnu\\ai-toolkit
git submodule update --init --recursive
python -m venv venv
.\\venv\\Scripts\\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt`}
                  </pre>
                  <p className="text-[11px] text-text-dim">
                    Once installed at <code className="text-accent font-mono">c:\vishnu\ai-toolkit</code>, click the refresh button above to detect it automatically.
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="p-3.5 rounded-xl bg-surface-elevated border border-border-subtle text-xs space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-text-primary font-mono text-[11px]">
                      Option B: Install kohya_ss (Standard LoRA Engine)
                    </span>
                    <button
                      onClick={() =>
                        copyCommand(
                          `git clone --recursive https://github.com/bmaltais/kohya_ss.git c:\\vishnu\\kohya_ss\ncd c:\\vishnu\\kohya_ss\n.\\setup.bat`,
                          'kohya'
                        )
                      }
                      className="flex items-center gap-1 px-2.5 py-1 rounded bg-surface hover:bg-surface-elevated border border-border-subtle text-accent text-[11px] font-mono transition-colors"
                    >
                      {copiedCmd === 'kohya' ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                      <span>{copiedCmd === 'kohya' ? 'Copied All' : 'Copy All'}</span>
                    </button>
                  </div>
                  <pre className="font-mono text-[11px] text-text-muted bg-background/80 p-3 rounded-lg overflow-x-auto whitespace-pre leading-relaxed border border-border-subtle">
{`git clone --recursive https://github.com/bmaltais/kohya_ss.git c:\\vishnu\\kohya_ss
cd c:\\vishnu\\kohya_ss
.\\setup.bat`}
                  </pre>
                  <p className="text-[11px] text-text-dim">
                    Alternatively, for standalone <code className="text-accent font-mono">sd-scripts</code>: install into <code className="text-accent font-mono">c:\vishnu\sd-scripts</code> with <code className="text-accent font-mono">pip install -r requirements.txt</code>.
                  </p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Step Progression Tabs */}
        <div className="flex items-center bg-surface-elevated p-1 rounded-xl border border-border-subtle text-xs overflow-x-auto">
          {[
            { num: 1, label: '1. Goal' },
            { num: 2, label: '2. Dataset' },
            { num: 3, label: '3. Captions' },
            { num: 4, label: '4. Parameters' },
            { num: 5, label: '5. Execution & Jobs' },
          ].map((s) => (
            <button
              key={s.num}
              onClick={() => setActiveStep(s.num as any)}
              className={`flex-1 py-1.5 px-3 rounded-lg font-medium transition-all whitespace-nowrap text-center ${
                activeStep === s.num
                  ? 'bg-surface text-accent font-semibold shadow-sm border border-border-subtle'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      {/* 2. Step 1: Goal Selector */}
      {activeStep === 1 && (
        <div className="space-y-6">
          <div className="space-y-1">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-text-muted flex items-center gap-1.5 font-mono">
              <Sparkles className="w-3.5 h-3.5 text-accent" />
              Select Adaptation Objective
            </h2>
            <p className="text-xs text-text-dim">
              Choosing a specific objective automatically tunes default repeat multipliers and learning rate balance.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {goals.map((g) => {
              const isSelected = goal === g.id;
              return (
                <div
                  key={g.id}
                  onClick={() => setGoal(g.id as any)}
                  className={`p-5 rounded-2xl border cursor-pointer transition-all space-y-2 bg-surface ${
                    isSelected
                      ? 'border-accent ring-1 ring-accent bg-surface-elevated/70 shadow-lg shadow-accent/5'
                      : 'border-border-subtle hover:border-border-strong'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold text-text-primary">{g.label}</span>
                    {isSelected && <Check className="w-4 h-4 text-accent" />}
                  </div>
                  <p className="text-xs text-text-muted leading-relaxed">{g.desc}</p>
                </div>
              );
            })}
          </div>

          <div className="flex justify-end pt-4">
            <button
              onClick={() => setActiveStep(2)}
              className="px-5 py-2.5 rounded-xl bg-accent text-background font-semibold text-xs flex items-center gap-2 hover:bg-accent-hover transition-colors shadow-sm"
            >
              <span>Next: Select Dataset Folder</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}

      {/* 3. Step 2: Dataset Selection & Validation */}
      {activeStep === 2 && (
        <div className="space-y-6">
          <div className="space-y-1">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-text-muted flex items-center gap-1.5 font-mono">
              <FolderOpen className="w-3.5 h-3.5 text-accent" />
              Dataset Folder Ingestion & Integrity Analysis
            </h2>
            <p className="text-xs text-text-dim">
              Enter an absolute local directory path containing high-resolution images (PNG, JPG, or WEBP).
            </p>
          </div>

          {/* Directory Input */}
          <div className="p-5 rounded-2xl bg-surface border border-border-subtle space-y-4">
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-text-muted">
                Local Dataset Directory Path
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  placeholder="C:\datasets\my_character"
                  value={datasetPath}
                  onChange={(e) => setDatasetPath(e.target.value)}
                  className="flex-1 bg-surface-elevated text-text-primary font-mono text-xs rounded-xl p-3 border border-border-subtle focus:border-accent focus:outline-none"
                />
                <button
                  onClick={handleValidateDataset}
                  disabled={validatingDataset || !datasetPath.trim()}
                  className="px-4 py-3 rounded-xl bg-accent text-background font-semibold text-xs flex items-center gap-1.5 hover:bg-accent-hover transition-colors shrink-0 disabled:opacity-60"
                >
                  {validatingDataset ? (
                    <>
                      <div className="w-3.5 h-3.5 border-2 border-background border-t-transparent rounded-full animate-spin" />
                      <span>Validating...</span>
                    </>
                  ) : (
                    <>
                      <ShieldCheck className="w-3.5 h-3.5" />
                      <span>Validate Dataset</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Validation Report Card */}
            {datasetReport && (
              <div className="space-y-4 pt-3 border-t border-border-subtle">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-xs font-semibold px-2.5 py-0.5 rounded-full border ${
                        datasetReport.is_valid
                          ? 'bg-status-success/15 text-status-success border-status-success/30'
                          : 'bg-status-error/15 text-status-error border-status-error/30'
                      }`}
                    >
                      {datasetReport.is_valid ? 'Dataset Validated' : 'Dataset Needs Attention'}
                    </span>
                    <span className="text-xs font-mono text-text-dim">
                      {datasetReport.total_images} images found
                    </span>
                  </div>
                  <span className="text-xs font-mono text-text-dim">
                    Formats: {datasetReport.supported_formats.join(', ') || 'None'}
                  </span>
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-xs font-mono">
                  <div className="p-3 rounded-xl bg-surface-elevated border border-border-subtle">
                    <span className="text-text-dim block text-[10px] uppercase font-sans">Resolution Range</span>
                    <span className="text-text-primary font-medium">{datasetReport.resolution_range}</span>
                  </div>
                  <div className="p-3 rounded-xl bg-surface-elevated border border-border-subtle">
                    <span className="text-text-dim block text-[10px] uppercase font-sans">Suggested Crop</span>
                    <span className="text-accent font-semibold">{datasetReport.suggested_resolution} × {datasetReport.suggested_resolution}</span>
                  </div>
                  <div className="p-3 rounded-xl bg-surface-elevated border border-border-subtle">
                    <span className="text-text-dim block text-[10px] uppercase font-sans">Missing Captions</span>
                    <span
                      className={`font-semibold ${
                        datasetReport.missing_captions_count > 0 ? 'text-accent' : 'text-status-success'
                      }`}
                    >
                      {datasetReport.missing_captions_count} files
                    </span>
                  </div>
                  <div className="p-3 rounded-xl bg-surface-elevated border border-border-subtle">
                    <span className="text-text-dim block text-[10px] uppercase font-sans">Duplicates / Corrupt</span>
                    <span className="text-text-muted">
                      {datasetReport.duplicates_count} dup / {datasetReport.corrupt_images_count} bad
                    </span>
                  </div>
                </div>

                {/* Issues List */}
                {datasetReport.issues.length > 0 && (
                  <div className="space-y-1.5 p-3 rounded-xl bg-surface-elevated border border-border-subtle text-xs">
                    <span className="text-text-dim uppercase font-semibold text-[10px] block">
                      Dataset Observations & Warnings ({datasetReport.issues.length})
                    </span>
                    <div className="max-h-40 overflow-y-auto space-y-1 font-mono text-[11px]">
                      {datasetReport.issues.map((iss, idx) => (
                        <div key={idx} className="flex items-start gap-2 text-text-muted">
                          <span
                            className={`w-1.5 h-1.5 rounded-full mt-1.5 shrink-0 ${
                              iss.type === 'corrupt' ? 'bg-status-error' : 'bg-accent'
                            }`}
                          />
                          <span className="text-text-primary font-medium">{iss.file}:</span>
                          <span>{iss.message}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="flex justify-between pt-4">
            <button
              onClick={() => setActiveStep(1)}
              className="px-4 py-2 rounded-xl bg-surface-elevated text-text-dim hover:text-text-primary text-xs font-medium"
            >
              Back: Goal
            </button>
            <button
              onClick={() => {
                setActiveStep(3);
                handlePreviewCaptions('inspect');
              }}
              disabled={!datasetReport || datasetReport.total_images === 0}
              className="px-5 py-2.5 rounded-xl bg-accent text-background font-semibold text-xs flex items-center gap-2 hover:bg-accent-hover transition-colors shadow-sm disabled:opacity-60"
            >
              <span>Next: Review & Format Captions</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}

      {/* 4. Step 3: Caption Workflow */}
      {activeStep === 3 && (
        <div className="space-y-6">
          <div className="space-y-1">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-text-muted flex items-center gap-1.5 font-mono">
              <FileText className="w-3.5 h-3.5 text-accent" />
              Caption Inspection & Trigger Word Injection
            </h2>
            <p className="text-xs text-text-dim">
              Preview existing .txt captions beside your training images and safely inject trigger words with automatic backups (.txt.bak).
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-surface border border-border-subtle space-y-4">
            {/* Trigger word input & batch tool */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border-subtle">
              <div className="space-y-1 flex-1">
                <label className="text-xs font-medium text-text-muted">Unique Concept Trigger Word</label>
                <input
                  type="text"
                  placeholder="e.g. cbr_char"
                  value={triggerWord}
                  onChange={(e) => setTriggerWord(e.target.value)}
                  className="w-full bg-surface-elevated text-text-primary font-mono text-xs rounded-xl p-2.5 border border-border-subtle focus:border-accent focus:outline-none"
                />
              </div>

              <div className="flex items-center gap-2 self-end">
                <button
                  onClick={() => handlePreviewCaptions('prepend_trigger')}
                  disabled={previewingCaptions || !triggerWord.trim()}
                  className="px-3.5 py-2.5 rounded-xl bg-surface-elevated hover:bg-surface border border-border-subtle text-accent text-xs font-semibold flex items-center gap-1.5 transition-colors disabled:opacity-60"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Prepend Trigger to All</span>
                </button>
                <button
                  onClick={handleApplyCaptions}
                  disabled={applyingCaptions || !captionPreview || captionPreview.changed_files === 0}
                  className="px-4 py-2.5 rounded-xl bg-accent text-background font-semibold text-xs flex items-center gap-1.5 hover:bg-accent-hover transition-colors shadow-sm disabled:opacity-60"
                >
                  {applyingCaptions ? (
                    'Applying...'
                  ) : (
                    <>
                      <Check className="w-3.5 h-3.5" />
                      <span>Apply & Backup (.bak)</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {captionsAppliedSuccess && (
              <div className="p-3 rounded-xl bg-status-success/15 border border-status-success/30 text-status-success text-xs flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 shrink-0" />
                <span>Captions successfully updated and backed up! You can proceed to training configuration.</span>
              </div>
            )}

            {/* Captions Table */}
            {captionPreview && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs font-mono text-text-dim">
                  <span>
                    Previewing {captionPreview.total_files} files ({captionPreview.changed_files} modified)
                  </span>
                  <span>Safety: Original .txt files copied to .txt.bak</span>
                </div>

                <div className="max-h-72 overflow-y-auto space-y-2 pr-1 scrollbar-thin">
                  {captionPreview.items.map((item, idx) => (
                    <div
                      key={idx}
                      className={`p-3 rounded-xl border text-xs font-mono transition-colors ${
                        item.changed
                          ? 'bg-surface-elevated/80 border-accent/40'
                          : 'bg-surface-elevated/30 border-border-subtle'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-text-primary font-semibold">{item.image_file}</span>
                        <span className={`text-[10px] uppercase font-semibold ${item.changed ? 'text-accent' : 'text-text-dim'}`}>
                          {item.changed ? 'Will Update' : 'Unchanged'}
                        </span>
                      </div>
                      <p className="text-text-muted leading-relaxed">
                        {item.proposed_caption || <span className="italic text-text-dim">(No caption present)</span>}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="flex justify-between pt-4">
            <button
              onClick={() => setActiveStep(2)}
              className="px-4 py-2 rounded-xl bg-surface-elevated text-text-dim hover:text-text-primary text-xs font-medium"
            >
              Back: Dataset
            </button>
            <button
              onClick={() => setActiveStep(4)}
              className="px-5 py-2.5 rounded-xl bg-accent text-background font-semibold text-xs flex items-center gap-2 hover:bg-accent-hover transition-colors shadow-sm"
            >
              <span>Next: Configure Hyperparameters</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}

      {/* 5. Step 4: Hyperparameters & RTX 5070 Ti Preset */}
      {activeStep === 4 && (
        <div className="space-y-6">
          <div className="space-y-1">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-text-muted flex items-center gap-1.5 font-mono">
              <Sliders className="w-3.5 h-3.5 text-accent" />
              Hyperparameters & Precision Tuning
            </h2>
            <p className="text-xs text-text-dim">
              Verified safe defaults for the 16GB NVIDIA GeForce RTX 5070 Ti Blackwell architecture.
            </p>
          </div>

          {/* Preset Badge */}
          <div className="p-4 rounded-xl bg-accent/10 border border-accent/30 text-xs flex items-start gap-3">
            <Zap className="w-4 h-4 text-accent shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-semibold text-accent block">
                RTX 5070 Ti Blackwell Safe Profile Loaded
              </span>
              <p className="text-text-muted text-[11px] leading-relaxed">
                Network Dimension 16 • Alpha 16 • Mixed Precision bf16 • Batch Size 1 • Gradient Checkpointing Enabled.
                Estimated peak VRAM: <strong>~12.5 GB</strong> (leaves ~3.5 GB headroom for OS desktop composition).
              </p>
            </div>
          </div>

          {/* Config Grid */}
          <div className="p-6 rounded-2xl bg-surface border border-border-subtle grid grid-cols-1 md:grid-cols-2 gap-5 text-xs">
            <div className="space-y-1.5">
              <label className="text-text-muted font-medium block">Output LoRA Filename</label>
              <input
                type="text"
                value={loraName}
                onChange={(e) => setLoraName(e.target.value.replace(/[^a-zA-Z0-9_\-]/g, ''))}
                className="w-full bg-surface-elevated text-text-primary font-mono text-xs rounded-xl p-3 border border-border-subtle focus:border-accent focus:outline-none"
              />
              <span className="text-[10px] text-text-dim">Letters, numbers, underscores only.</span>
            </div>

            <div className="space-y-1.5">
              <label className="text-text-muted font-medium block">Base Model Checkpoint</label>
              <input
                type="text"
                value={baseModel}
                onChange={(e) => setBaseModel(e.target.value)}
                className="w-full bg-surface-elevated text-text-primary font-mono text-xs rounded-xl p-3 border border-border-subtle focus:border-accent focus:outline-none"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-text-muted font-medium block">Training Resolution ({resolution}px)</label>
              <select
                value={resolution}
                onChange={(e) => setResolution(parseInt(e.target.value))}
                className="w-full bg-surface-elevated text-text-primary text-xs rounded-xl p-3 border border-border-subtle focus:border-accent focus:outline-none"
              >
                <option value={512}>512 × 512 (Fastest, low VRAM)</option>
                <option value={768}>768 × 768 (Balanced)</option>
                <option value={1024}>1024 × 1024 (SDXL Native High Detail)</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <label className="text-text-muted font-medium block">Network Dim</label>
                <input
                  type="number"
                  min="4"
                  max="128"
                  value={networkDim}
                  onChange={(e) => setNetworkDim(parseInt(e.target.value) || 16)}
                  className="w-full bg-surface-elevated text-text-primary font-mono text-xs rounded-xl p-3 border border-border-subtle focus:border-accent focus:outline-none"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-text-muted font-medium block">Network Alpha</label>
                <input
                  type="number"
                  min="1"
                  max="128"
                  value={networkAlpha}
                  onChange={(e) => setNetworkAlpha(parseInt(e.target.value) || 16)}
                  className="w-full bg-surface-elevated text-text-primary font-mono text-xs rounded-xl p-3 border border-border-subtle focus:border-accent focus:outline-none"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <label className="text-text-muted font-medium block">UNet Learning Rate</label>
                <input
                  type="number"
                  step="0.00005"
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.0001)}
                  className="w-full bg-surface-elevated text-text-primary font-mono text-xs rounded-xl p-3 border border-border-subtle focus:border-accent focus:outline-none"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-text-muted font-medium block">Text Encoder LR</label>
                <input
                  type="number"
                  step="0.00001"
                  value={textEncoderLr}
                  onChange={(e) => setTextEncoderLr(parseFloat(e.target.value) || 0.00005)}
                  className="w-full bg-surface-elevated text-text-primary font-mono text-xs rounded-xl p-3 border border-border-subtle focus:border-accent focus:outline-none"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <label className="text-text-muted font-medium block">Total Epochs</label>
                <input
                  type="number"
                  min="1"
                  max="50"
                  value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value) || 10)}
                  className="w-full bg-surface-elevated text-text-primary font-mono text-xs rounded-xl p-3 border border-border-subtle focus:border-accent focus:outline-none"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-text-muted font-medium block">Max Train Steps</label>
                <input
                  type="number"
                  min="50"
                  max="10000"
                  value={maxSteps}
                  onChange={(e) => setMaxSteps(parseInt(e.target.value) || 1000)}
                  className="w-full bg-surface-elevated text-text-primary font-mono text-xs rounded-xl p-3 border border-border-subtle focus:border-accent focus:outline-none"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <label className="text-text-muted font-medium block">Mixed Precision</label>
                <select
                  value={mixedPrecision}
                  onChange={(e) => setMixedPrecision(e.target.value as any)}
                  className="w-full bg-surface-elevated text-text-primary text-xs rounded-xl p-3 border border-border-subtle focus:border-accent focus:outline-none"
                >
                  <option value="bf16">bf16 (Blackwell / Ada native)</option>
                  <option value="fp16">fp16 (Standard CUDA)</option>
                  <option value="no">no (Full fp32)</option>
                </select>
              </div>

              <div className="space-y-1.5 flex flex-col justify-end">
                <label className="flex items-center gap-2 cursor-pointer p-3 rounded-xl bg-surface-elevated border border-border-subtle">
                  <input
                    type="checkbox"
                    checked={gradientCheckpointing}
                    onChange={(e) => setGradientCheckpointing(e.target.checked)}
                    className="accent-accent w-4 h-4 rounded"
                  />
                  <span className="text-text-primary font-medium">Gradient Checkpointing</span>
                </label>
              </div>
            </div>
          </div>

          <div className="flex justify-between pt-4">
            <button
              onClick={() => setActiveStep(3)}
              className="px-4 py-2 rounded-xl bg-surface-elevated text-text-dim hover:text-text-primary text-xs font-medium"
            >
              Back: Captions
            </button>

            <button
              onClick={handleStartTraining}
              disabled={isSubmittingTraining || !health?.engine_available}
              className={`px-6 py-3 rounded-xl font-semibold text-xs flex items-center gap-2 transition-all shadow-lg ${
                health?.engine_available
                  ? 'bg-accent text-background hover:bg-accent-hover shadow-accent/15'
                  : 'bg-surface-elevated text-text-dim border border-border-subtle cursor-not-allowed opacity-70'
              }`}
            >
              {isSubmittingTraining ? (
                <>
                  <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                  <span>Initiating...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>Start Training</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* 6. Step 5: Execution, Monitor & History */}
      {activeStep === 5 && (
        <div className="space-y-6">
          <div className="space-y-1">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-text-muted flex items-center gap-1.5 font-mono">
              <Terminal className="w-3.5 h-3.5 text-accent" />
              Training Execution & Lifecycle Monitor
            </h2>
            <p className="text-xs text-text-dim">
              Inspect real-time loss curves, epoch checkpoints, cancel running tasks, and import finished LoRA adapters.
            </p>
          </div>

          {jobs.length === 0 ? (
            <div className="p-16 text-center rounded-2xl bg-surface border border-border-subtle border-dashed space-y-4">
              <Cpu className="w-12 h-12 mx-auto text-text-dim stroke-[1.2]" />
              <div className="space-y-1">
                <h3 className="text-sm font-semibold text-text-primary">No training jobs recorded</h3>
                <p className="text-xs text-text-muted max-w-md mx-auto leading-relaxed">
                  Configure your dataset and hyperparameters in steps 1 through 4, then click Start Training.
                </p>
              </div>
              <button
                onClick={() => setActiveStep(4)}
                className="px-4 py-2 rounded-xl text-xs font-semibold bg-surface-elevated hover:bg-surface border border-border-subtle text-accent inline-flex items-center gap-1.5"
              >
                <span>Configure Training Job</span>
              </button>
            </div>
          ) : (
            <div className="space-y-5">
              {/* Selected Job Card */}
              {selectedJob && (
                <div className="p-6 rounded-2xl bg-surface border border-border-strong space-y-5 shadow-xl relative overflow-hidden">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <div className="flex items-center gap-2.5">
                        <h3 className="text-base font-semibold text-text-primary tracking-tight">
                          {selectedJob.lora_name}
                        </h3>
                        <span
                          className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded font-bold border ${
                            selectedJob.status === 'completed'
                              ? 'bg-status-success/20 text-status-success border-status-success/40'
                              : selectedJob.status === 'training'
                              ? 'bg-accent/20 text-accent border-accent/40 animate-pulse'
                              : selectedJob.status === 'failed'
                              ? 'bg-status-error/20 text-status-error border-status-error/40'
                              : 'bg-surface-elevated text-text-dim border-border-subtle'
                          }`}
                        >
                          {selectedJob.status}
                        </span>
                      </div>
                      <p className="text-xs text-text-dim font-mono mt-0.5">
                        Job ID: {selectedJob.id} • Target Engine: {selectedJob.training_engine}
                      </p>
                    </div>

                    <div className="flex items-center space-x-2">
                      {selectedJob.status === 'training' && (
                        <button
                          onClick={() => handleCancelJob(selectedJob.id)}
                          className="px-3 py-1.5 rounded-lg bg-status-error/15 text-status-error hover:bg-status-error/25 border border-status-error/30 text-xs font-medium flex items-center gap-1"
                        >
                          <Ban className="w-3.5 h-3.5" />
                          <span>Cancel</span>
                        </button>
                      )}

                      {selectedJob.status === 'failed' && (
                        <button
                          onClick={() => handleRetryJob(selectedJob.id)}
                          className="px-3 py-1.5 rounded-lg bg-surface-elevated text-accent hover:bg-surface border border-border-subtle text-xs font-medium flex items-center gap-1"
                        >
                          <RotateCcw className="w-3.5 h-3.5" />
                          <span>Retry</span>
                        </button>
                      )}

                      {selectedJob.status === 'completed' && selectedJob.artifact_path && (
                        <button
                          onClick={() => handleImportArtifactToVault(selectedJob.id)}
                          className="px-3 py-1.5 rounded-lg bg-accent text-background hover:bg-accent-hover text-xs font-semibold flex items-center gap-1 shadow-sm"
                        >
                          <Layers className="w-3.5 h-3.5" />
                          <span>Import into Vault</span>
                        </button>
                      )}

                      <button
                        onClick={() => setLogsOpen(!logsOpen)}
                        className="px-3 py-1.5 rounded-lg bg-surface-elevated text-text-muted hover:text-text-primary border border-border-subtle text-xs font-medium flex items-center gap-1"
                      >
                        <Terminal className="w-3.5 h-3.5" />
                        <span>Logs</span>
                      </button>
                    </div>
                  </div>

                  {/* Progress bar */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-mono">
                      <span className="text-text-muted">
                        Step {selectedJob.current_step} / {selectedJob.total_steps} (Epoch{' '}
                        {selectedJob.current_epoch} / {selectedJob.total_epochs})
                      </span>
                      <span className="text-accent font-semibold">
                        {Math.round((selectedJob.current_step / Math.max(1, selectedJob.total_steps)) * 100)}%
                      </span>
                    </div>
                    <div className="w-full bg-surface-elevated rounded-full h-2 overflow-hidden border border-border-subtle">
                      <div
                        className="bg-accent h-full transition-all duration-300 rounded-full"
                        style={{
                          width: `${Math.min(
                            100,
                            (selectedJob.current_step / Math.max(1, selectedJob.total_steps)) * 100
                          )}%`,
                        }}
                      />
                    </div>
                  </div>

                  {/* Failure Callout */}
                  {selectedJob.status === 'failed' && (
                    <div className="p-3.5 rounded-xl bg-status-error/15 border border-status-error/30 text-xs text-status-error space-y-1">
                      <span className="font-semibold block">Training Process Unsuccessful</span>
                      <p className="font-mono text-[11px] leading-relaxed">
                        {selectedJob.error_message || 'The training engine halted unexpectedly.'}
                      </p>
                    </div>
                  )}

                  {/* Collapsible Logs Drawer */}
                  {logsOpen && (
                    <div className="p-3.5 rounded-xl bg-background border border-border-subtle font-mono text-[11px] space-y-2">
                      <div className="flex items-center justify-between text-text-dim text-[10px] uppercase font-semibold">
                        <span>Terminal Output Streams</span>
                        <span>{selectedJob.logs.length} lines</span>
                      </div>
                      <div className="max-h-48 overflow-y-auto space-y-0.5 text-text-muted pr-1 scrollbar-thin">
                        {selectedJob.logs.map((line, idx) => (
                          <div key={idx} className="whitespace-pre-wrap">
                            {line}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Jobs History Table */}
              <div className="space-y-3">
                <span className="text-xs font-semibold uppercase tracking-wider text-text-muted font-mono block">
                  Training Job History ({jobs.length})
                </span>
                <div className="space-y-2">
                  {jobs.map((j) => (
                    <div
                      key={j.id}
                      onClick={() => setSelectedJob(j)}
                      className={`p-4 rounded-xl border cursor-pointer transition-all flex items-center justify-between text-xs font-mono ${
                        selectedJob?.id === j.id
                          ? 'bg-surface-elevated border-accent/60'
                          : 'bg-surface border-border-subtle hover:border-border-strong'
                      }`}
                    >
                      <div className="space-y-0.5">
                        <div className="flex items-center gap-2">
                          <span className="text-text-primary font-semibold font-sans">{j.lora_name}</span>
                          <span className="text-[10px] px-1.5 py-0.2 rounded bg-surface border border-border-subtle text-accent">
                            {j.goal}
                          </span>
                        </div>
                        <span className="text-[11px] text-text-dim">{j.id.slice(0, 8)} • {j.dataset_image_count} images</span>
                      </div>

                      <div className="flex items-center gap-3">
                        <span
                          className={`text-[10px] uppercase px-2 py-0.5 rounded font-bold ${
                            j.status === 'completed'
                              ? 'bg-status-success/20 text-status-success'
                              : j.status === 'training'
                              ? 'bg-accent/20 text-accent animate-pulse'
                              : j.status === 'failed'
                              ? 'bg-status-error/20 text-status-error'
                              : 'bg-surface-elevated text-text-dim'
                          }`}
                        >
                          {j.status}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
