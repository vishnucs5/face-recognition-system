import React, { useEffect, useState } from 'react';
import {
  Layers,
  Plus,
  Check,
  Sparkles,
  FolderOpen,
  ArrowRight,
  Search,
  Copy,
  Edit2,
  Trash2,
  AlertTriangle,
  HardDrive,
  FileCode,
  Tag,
  Clock,
  ExternalLink,
  X,
  RefreshCw,
  Info,
} from 'lucide-react';
import { useStudioStore } from '../store/useStudioStore';
import { LoraVaultItem } from '../types';

export const LorasView: React.FC = () => {
  const {
    loraVault,
    setLoraVault,
    loras,
    removeLora,
    sendLoraToComposer,
    setActiveNav,
    setErrorMessage,
  } = useStudioStore();

  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [modelFilter, setModelFilter] = useState<string>('all');
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Import Modal State
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [importPath, setImportPath] = useState('');
  const [importName, setImportName] = useState('');
  const [importTriggers, setImportTriggers] = useState('');
  const [importBaseModel, setImportBaseModel] = useState('SDXL 1.0');
  const [importWeight, setImportWeight] = useState(0.8);
  const [importNotes, setImportNotes] = useState('');
  const [isSubmittingImport, setIsSubmittingImport] = useState(false);

  // Edit Modal State
  const [editingLora, setEditingLora] = useState<LoraVaultItem | null>(null);
  const [editName, setEditName] = useState('');
  const [editTriggers, setEditTriggers] = useState('');
  const [editBaseModel, setEditBaseModel] = useState('');
  const [editWeight, setEditWeight] = useState(0.8);
  const [editNotes, setEditNotes] = useState('');
  const [isSubmittingEdit, setIsSubmittingEdit] = useState(false);

  // Local weight adjustments per card
  const [cardWeights, setCardWeights] = useState<Record<string, number>>({});

  useEffect(() => {
    fetchLoraVault();
  }, []);

  const fetchLoraVault = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/loras');
      if (res.ok) {
        const data: LoraVaultItem[] = await res.json();
        setLoraVault(data);
      }
    } catch (err) {
      console.error('Failed to fetch LoRA vault items:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCopyTriggers = (lora: LoraVaultItem) => {
    if (!lora.trigger_words || lora.trigger_words.length === 0) return;
    const text = lora.trigger_words.join(', ');
    navigator.clipboard.writeText(text);
    setCopiedId(lora.id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleImportSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!importPath.trim()) {
      setErrorMessage('Please specify an absolute path to a .safetensors file.');
      return;
    }
    setIsSubmittingImport(true);
    try {
      const payload = {
        file_path: importPath.trim(),
        name: importName.trim() || undefined,
        trigger_words: importTriggers
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean),
        base_model: importBaseModel,
        recommended_weight: importWeight,
        notes: importNotes.trim() || undefined,
      };

      const res = await fetch('/api/loras/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        setErrorMessage(err.detail || 'Failed to import LoRA.');
        setIsSubmittingImport(false);
        return;
      }

      const imported: LoraVaultItem = await res.json();
      setLoraVault([imported, ...loraVault.filter((l) => l.id !== imported.id)]);
      setImportModalOpen(false);
      setImportPath('');
      setImportName('');
      setImportTriggers('');
      setImportNotes('');
    } catch (err: any) {
      setErrorMessage(err?.message || 'Network error importing LoRA.');
    } finally {
      setIsSubmittingImport(false);
    }
  };

  const handleOpenEdit = (lora: LoraVaultItem) => {
    setEditingLora(lora);
    setEditName(lora.name);
    setEditTriggers(lora.trigger_words.join(', '));
    setEditBaseModel(lora.base_model);
    setEditWeight(lora.recommended_weight);
    setEditNotes(lora.notes || '');
  };

  const handleEditSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingLora) return;
    setIsSubmittingEdit(true);
    try {
      const payload = {
        name: editName.trim() || undefined,
        trigger_words: editTriggers
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean),
        base_model: editBaseModel,
        recommended_weight: editWeight,
        notes: editNotes.trim() || undefined,
      };

      const res = await fetch(`/api/loras/${editingLora.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        setErrorMessage(err.detail || 'Failed to update LoRA metadata.');
        setIsSubmittingEdit(false);
        return;
      }

      const updated: LoraVaultItem = await res.json();
      setLoraVault(loraVault.map((l) => (l.id === updated.id ? updated : l)));
      setEditingLora(null);
    } catch (err: any) {
      setErrorMessage(err?.message || 'Error updating LoRA metadata.');
    } finally {
      setIsSubmittingEdit(false);
    }
  };

  const handleRemoveLora = async (lora: LoraVaultItem) => {
    if (!confirm(`Remove "${lora.name}" from Forge Studio index?\n\n(The physical file on disk will NOT be deleted).`)) {
      return;
    }
    try {
      const res = await fetch(`/api/loras/${lora.id}`, { method: 'DELETE' });
      if (res.ok) {
        setLoraVault(loraVault.filter((l) => l.id !== lora.id));
        removeLora(lora.filename);
      }
    } catch (err) {
      console.error('Failed to remove LoRA:', err);
    }
  };

  const filteredLoras = loraVault.filter((item) => {
    const q = searchQuery.toLowerCase();
    const matchesQuery =
      item.name.toLowerCase().includes(q) ||
      item.filename.toLowerCase().includes(q) ||
      item.trigger_words.some((t) => t.toLowerCase().includes(q));
    const matchesModel = modelFilter === 'all' || item.base_model === modelFilter;
    return matchesQuery && matchesModel;
  });

  const modelFamilies = ['all', 'SDXL 1.0', 'SD 1.5', 'Flux', 'Pony'];

  return (
    <div className="flex-1 p-6 md:p-10 overflow-y-auto space-y-8 max-w-6xl select-none">
      {/* Header & Quick Action Bar */}
      <div className="space-y-4 border-b border-border-subtle pb-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-text-primary flex items-center gap-2.5">
              <Layers className="w-6 h-6 text-accent" />
              <span>LoRA Adaptation Vault</span>
              <span className="text-xs font-mono font-normal px-2 py-0.5 rounded bg-surface-elevated text-accent border border-border-subtle">
                {loraVault.length} Indexed
              </span>
            </h1>
            <p className="text-xs text-text-muted mt-1">
              Low-Rank Adaptation models with automatic ComfyUI pipeline injection and local metadata indexing.
            </p>
          </div>

          <div className="flex items-center space-x-2.5">
            <button
              onClick={fetchLoraVault}
              className="p-2 rounded-xl bg-surface-elevated hover:bg-surface border border-border-subtle text-text-muted hover:text-text-primary transition-colors"
              title="Refresh LoRA index"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={() => setImportModalOpen(true)}
              className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-accent text-background hover:bg-accent-hover shadow-sm transition-colors flex items-center gap-1.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            >
              <Plus className="w-4 h-4" />
              <span>Import .safetensors LoRA</span>
            </button>
            <button
              onClick={() => setActiveNav('generate')}
              className="flex items-center space-x-1.5 px-3 py-2 rounded-xl text-xs font-medium bg-surface-elevated hover:bg-surface border border-border-subtle text-accent transition-colors"
            >
              <span>Back to Studio</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Filter & Search Toolbar */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-2">
          {/* Search Input */}
          <div className="relative w-full sm:w-72">
            <Search className="w-3.5 h-3.5 absolute left-3 top-3 text-text-dim" />
            <input
              type="text"
              placeholder="Search by name, trigger word, filename..."
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

          {/* Model Family Filter Tabs */}
          <div className="flex items-center bg-surface-elevated p-1 rounded-xl border border-border-subtle text-xs overflow-x-auto">
            {modelFamilies.map((fam) => (
              <button
                key={fam}
                onClick={() => setModelFilter(fam)}
                className={`px-3 py-1 rounded-lg font-medium transition-colors whitespace-nowrap ${
                  modelFilter === fam
                    ? 'bg-surface text-accent font-semibold shadow-sm'
                    : 'text-text-muted hover:text-text-primary'
                }`}
              >
                {fam === 'all' ? 'All Families' : fam}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* LoRA Vault Cards Grid */}
      <div className="space-y-4">
        {filteredLoras.length === 0 ? (
          <div className="p-16 text-center rounded-2xl bg-surface border border-border-subtle border-dashed space-y-4">
            <FolderOpen className="w-12 h-12 mx-auto text-text-dim stroke-[1.2]" />
            <div className="space-y-1">
              <h3 className="text-sm font-semibold text-text-primary">No LoRA models found</h3>
              <p className="text-xs text-text-muted max-w-md mx-auto leading-relaxed">
                Import an existing <code className="font-mono text-accent">.safetensors</code> file using the button above, or place files in <code className="font-mono text-accent">models/loras/</code> to auto-index.
              </p>
            </div>
            <button
              onClick={() => setImportModalOpen(true)}
              className="px-4 py-2 rounded-xl text-xs font-semibold bg-surface-elevated hover:bg-surface border border-border-subtle text-accent inline-flex items-center gap-1.5"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Import First LoRA</span>
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {filteredLoras.map((item) => {
              const isAttached = loras.some((l) => l.name === item.filename);
              const activeLoraObj = loras.find((l) => l.name === item.filename);
              const currentWeight = cardWeights[item.id] ?? item.recommended_weight;

              return (
                <div
                  key={item.id}
                  className={`p-6 rounded-2xl border transition-all space-y-4 bg-surface ${
                    isAttached
                      ? 'border-accent/60 ring-1 ring-accent/30 shadow-md shadow-accent/5'
                      : 'border-border-subtle hover:border-border-strong'
                  }`}
                >
                  {/* Top Bar: Title, Model Badge, Status */}
                  <div className="flex items-start justify-between gap-3">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <h3 className="text-sm font-semibold text-text-primary tracking-tight">
                          {item.name}
                        </h3>
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-surface-elevated text-accent border border-border-subtle">
                          {item.base_model}
                        </span>
                      </div>
                      <p className="text-[11px] text-text-dim font-mono truncate max-w-sm" title={item.filename}>
                        {item.filename}
                      </p>
                    </div>

                    <div className="flex items-center gap-1.5 shrink-0">
                      <span
                        className={`text-[9px] font-mono uppercase px-2 py-0.5 rounded font-semibold border ${
                          item.is_available
                            ? 'bg-status-success/15 text-status-success border-status-success/30'
                            : 'bg-status-error/15 text-status-error border-status-error/30'
                        }`}
                      >
                        {item.is_available ? 'Available' : 'File Missing'}
                      </span>
                    </div>
                  </div>

                  {/* Technical Meta Strip */}
                  <div className="grid grid-cols-3 gap-2 text-[11px] font-mono">
                    <div className="p-2 rounded-lg bg-surface-elevated border border-border-subtle">
                      <span className="text-text-dim block text-[9px] font-sans uppercase">File Size</span>
                      <span className="text-text-muted font-medium">{item.file_size_mb} MB</span>
                    </div>
                    <div className="p-2 rounded-lg bg-surface-elevated border border-border-subtle">
                      <span className="text-text-dim block text-[9px] font-sans uppercase">Status</span>
                      <span className="text-text-muted capitalize font-medium">{item.training_status}</span>
                    </div>
                    <div className="p-2 rounded-lg bg-surface-elevated border border-border-subtle">
                      <span className="text-text-dim block text-[9px] font-sans uppercase">Rec. Weight</span>
                      <span className="text-accent font-semibold">{item.recommended_weight}x</span>
                    </div>
                  </div>

                  {/* Trigger Words Strip */}
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="text-text-dim uppercase font-semibold flex items-center gap-1 text-[10px]">
                        <Tag className="w-3 h-3 text-accent" />
                        Trigger Words
                      </span>
                      {item.trigger_words.length > 0 && (
                        <button
                          onClick={() => handleCopyTriggers(item)}
                          className="text-accent hover:underline flex items-center gap-1 text-[11px]"
                        >
                          {copiedId === item.id ? (
                            <>
                              <Check className="w-3 h-3 text-status-success" />
                              <span className="text-status-success">Copied!</span>
                            </>
                          ) : (
                            <>
                              <Copy className="w-3 h-3" />
                              <span>Copy All</span>
                            </>
                          )}
                        </button>
                      )}
                    </div>
                    {item.trigger_words.length > 0 ? (
                      <div className="flex flex-wrap gap-1.5">
                        {item.trigger_words.map((tw) => (
                          <span
                            key={tw}
                            className="px-2 py-0.5 rounded text-[10px] font-mono bg-surface-elevated text-text-primary border border-border-subtle"
                          >
                            {tw}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <span className="text-[11px] text-text-dim italic font-sans block">
                        No trigger words specified
                      </span>
                    )}
                  </div>

                  {/* Notes (if present) */}
                  {item.notes && (
                    <p className="text-xs text-text-muted bg-surface-elevated/40 p-2.5 rounded-lg border border-border-subtle/50 text-[11px] leading-relaxed">
                      {item.notes}
                    </p>
                  )}

                  {/* Weight Slider & Actions */}
                  <div className="pt-2 border-t border-border-subtle/60 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    {/* Weight Adjustment */}
                    <div className="flex items-center space-x-2">
                      <span className="text-[11px] text-text-dim font-mono">Weight:</span>
                      <input
                        type="range"
                        min="0.1"
                        max="1.5"
                        step="0.05"
                        value={currentWeight}
                        onChange={(e) => {
                          const val = parseFloat(e.target.value);
                          setCardWeights((prev) => ({ ...prev, [item.id]: val }));
                        }}
                        className="w-24 accent-accent cursor-pointer"
                      />
                      <span className="text-xs font-mono font-semibold text-accent w-8 text-right">
                        {currentWeight.toFixed(2)}
                      </span>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => handleOpenEdit(item)}
                        className="p-2 rounded-lg bg-surface-elevated hover:bg-surface border border-border-subtle text-text-dim hover:text-text-primary transition-colors"
                        title="Edit metadata"
                      >
                        <Edit2 className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => handleRemoveLora(item)}
                        className="p-2 rounded-lg bg-surface-elevated hover:bg-surface border border-border-subtle text-text-dim hover:text-status-error transition-colors"
                        title="Remove from Forge Studio index"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>

                      {isAttached ? (
                        <button
                          onClick={() => removeLora(item.filename)}
                          className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-status-error/15 text-status-error hover:bg-status-error/25 border border-status-error/30 transition-colors flex items-center gap-1.5"
                        >
                          <Check className="w-3.5 h-3.5" />
                          <span>Attached ({activeLoraObj?.strength}x)</span>
                        </button>
                      ) : (
                        <button
                          onClick={() => sendLoraToComposer(item, currentWeight)}
                          disabled={!item.is_available}
                          className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors ${
                            !item.is_available
                              ? 'bg-surface-elevated text-text-dim border border-border-subtle cursor-not-allowed opacity-60'
                              : 'bg-accent text-background hover:bg-accent-hover shadow-sm'
                          }`}
                        >
                          <Sparkles className="w-3.5 h-3.5" />
                          <span>Send to Studio</span>
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Import Modal */}
      {importModalOpen && (
        <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="w-full max-w-lg bg-surface border border-border-strong rounded-2xl shadow-2xl p-6 space-y-5 animate-in zoom-in-95 duration-150">
            <div className="flex items-center justify-between border-b border-border-subtle pb-3">
              <div className="flex items-center space-x-2 text-text-primary font-semibold text-sm">
                <Layers className="w-4 h-4 text-accent" />
                <span>Import Local LoRA (.safetensors)</span>
              </div>
              <button
                onClick={() => setImportModalOpen(false)}
                className="text-text-dim hover:text-text-primary"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleImportSubmit} className="space-y-4 text-xs">
              <div className="space-y-1">
                <label className="text-text-muted font-medium block">
                  File Path on Local Machine <span className="text-status-error">*</span>
                </label>
                <input
                  type="text"
                  placeholder="C:\path\to\adapter.safetensors"
                  value={importPath}
                  onChange={(e) => setImportPath(e.target.value)}
                  required
                  className="w-full bg-surface-elevated text-text-primary font-mono text-xs rounded-lg p-2.5 border border-border-subtle focus:border-accent focus:outline-none"
                />
                <span className="text-[10px] text-text-dim">
                  Must be an existing .safetensors model file.
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-text-muted font-medium block">Display Name</label>
                  <input
                    type="text"
                    placeholder="e.g. Cyberpunk Mecha"
                    value={importName}
                    onChange={(e) => setImportName(e.target.value)}
                    className="w-full bg-surface-elevated text-text-primary text-xs rounded-lg p-2.5 border border-border-subtle focus:border-accent focus:outline-none"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-text-muted font-medium block">Base Model Family</label>
                  <select
                    value={importBaseModel}
                    onChange={(e) => setImportBaseModel(e.target.value)}
                    className="w-full bg-surface-elevated text-text-primary text-xs rounded-lg p-2.5 border border-border-subtle focus:border-accent focus:outline-none"
                  >
                    <option value="SDXL 1.0">SDXL 1.0</option>
                    <option value="SD 1.5">SD 1.5</option>
                    <option value="Flux">Flux</option>
                    <option value="Pony">Pony</option>
                  </select>
                </div>
              </div>

              <div className="space-y-1">
                <label className="text-text-muted font-medium block">
                  Trigger Words (comma-separated)
                </label>
                <input
                  type="text"
                  placeholder="e.g. mecha_suit, cybernetic, neon"
                  value={importTriggers}
                  onChange={(e) => setImportTriggers(e.target.value)}
                  className="w-full bg-surface-elevated text-text-primary text-xs rounded-lg p-2.5 border border-border-subtle focus:border-accent focus:outline-none"
                />
              </div>

              <div className="space-y-1">
                <label className="text-text-muted font-medium block">
                  Recommended Weight ({importWeight.toFixed(2)})
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1.5"
                  step="0.05"
                  value={importWeight}
                  onChange={(e) => setImportWeight(parseFloat(e.target.value))}
                  className="w-full accent-accent cursor-pointer"
                />
              </div>

              <div className="space-y-1">
                <label className="text-text-muted font-medium block">Notes & Recommendations</label>
                <textarea
                  rows={2}
                  placeholder="Optimal CFG, recommended sampler, or style instructions..."
                  value={importNotes}
                  onChange={(e) => setImportNotes(e.target.value)}
                  className="w-full bg-surface-elevated text-text-primary text-xs rounded-lg p-2.5 border border-border-subtle focus:border-accent focus:outline-none resize-none"
                />
              </div>

              <div className="flex items-center justify-end space-x-2.5 pt-3 border-t border-border-subtle">
                <button
                  type="button"
                  onClick={() => setImportModalOpen(false)}
                  className="px-4 py-2 rounded-lg bg-surface-elevated text-text-dim hover:text-text-primary text-xs font-medium"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isSubmittingImport}
                  className="px-4 py-2 rounded-lg bg-accent text-background hover:bg-accent-hover text-xs font-semibold shadow-sm flex items-center gap-1.5"
                >
                  {isSubmittingImport ? (
                    <>
                      <div className="w-3.5 h-3.5 border-2 border-background border-t-transparent rounded-full animate-spin" />
                      <span>Indexing...</span>
                    </>
                  ) : (
                    <span>Import to Vault</span>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Edit Metadata Modal */}
      {editingLora && (
        <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="w-full max-w-lg bg-surface border border-border-strong rounded-2xl shadow-2xl p-6 space-y-5 animate-in zoom-in-95 duration-150">
            <div className="flex items-center justify-between border-b border-border-subtle pb-3">
              <div className="flex items-center space-x-2 text-text-primary font-semibold text-sm">
                <Edit2 className="w-4 h-4 text-accent" />
                <span>Edit LoRA Metadata</span>
              </div>
              <button onClick={() => setEditingLora(null)} className="text-text-dim hover:text-text-primary">
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleEditSubmit} className="space-y-4 text-xs">
              <div className="space-y-1">
                <label className="text-text-muted font-medium block">Display Name</label>
                <input
                  type="text"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  className="w-full bg-surface-elevated text-text-primary text-xs rounded-lg p-2.5 border border-border-subtle focus:border-accent focus:outline-none"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-text-muted font-medium block">Base Model</label>
                  <select
                    value={editBaseModel}
                    onChange={(e) => setEditBaseModel(e.target.value)}
                    className="w-full bg-surface-elevated text-text-primary text-xs rounded-lg p-2.5 border border-border-subtle focus:border-accent focus:outline-none"
                  >
                    <option value="SDXL 1.0">SDXL 1.0</option>
                    <option value="SD 1.5">SD 1.5</option>
                    <option value="Flux">Flux</option>
                    <option value="Pony">Pony</option>
                  </select>
                </div>
                <div className="space-y-1">
                  <label className="text-text-muted font-medium block">
                    Recommended Weight ({editWeight.toFixed(2)})
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="1.5"
                    step="0.05"
                    value={editWeight}
                    onChange={(e) => setEditWeight(parseFloat(e.target.value))}
                    className="w-full accent-accent cursor-pointer mt-2"
                  />
                </div>
              </div>

              <div className="space-y-1">
                <label className="text-text-muted font-medium block">Trigger Words (comma-separated)</label>
                <input
                  type="text"
                  value={editTriggers}
                  onChange={(e) => setEditTriggers(e.target.value)}
                  className="w-full bg-surface-elevated text-text-primary text-xs rounded-lg p-2.5 border border-border-subtle focus:border-accent focus:outline-none"
                />
              </div>

              <div className="space-y-1">
                <label className="text-text-muted font-medium block">Notes</label>
                <textarea
                  rows={2}
                  value={editNotes}
                  onChange={(e) => setEditNotes(e.target.value)}
                  className="w-full bg-surface-elevated text-text-primary text-xs rounded-lg p-2.5 border border-border-subtle focus:border-accent focus:outline-none resize-none"
                />
              </div>

              <div className="flex items-center justify-end space-x-2.5 pt-3 border-t border-border-subtle">
                <button
                  type="button"
                  onClick={() => setEditingLora(null)}
                  className="px-4 py-2 rounded-lg bg-surface-elevated text-text-dim hover:text-text-primary text-xs font-medium"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isSubmittingEdit}
                  className="px-4 py-2 rounded-lg bg-accent text-background hover:bg-accent-hover text-xs font-semibold shadow-sm"
                >
                  {isSubmittingEdit ? 'Saving...' : 'Save Changes'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
