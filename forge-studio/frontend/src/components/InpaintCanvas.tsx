import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Paintbrush,
  Eraser,
  RotateCcw,
  Upload,
  X,
  AlertCircle,
  CheckCircle2,
  Sliders,
  Undo2,
  Redo2,
  FlipHorizontal,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Sparkles,
  Layers,
} from 'lucide-react';
import { useStudioStore } from '../store/useStudioStore';

export const InpaintCanvas: React.FC = () => {
  const {
    source_image_id,
    source_image_preview,
    mask_image_id,
    brush_size,
    brush_mode,
    setSourceImage,
    setMaskImage,
    setBrushSize,
    setBrushMode,
    clearMask,
    setErrorMessage,
  } = useStudioStore();

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenMaskRef = useRef<HTMLCanvasElement | null>(null);
  const imageObjRef = useRef<HTMLImageElement | null>(null);

  // History stacks
  const undoStackRef = useRef<ImageData[]>([]);
  const redoStackRef = useRef<ImageData[]>([]);
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);

  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);
  const [isDrawing, setIsDrawing] = useState(false);
  const [hasMaskDrawn, setHasMaskDrawn] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [featherRadius, setFeatherRadius] = useState<number>(0); // 0 to 16px blur
  const [zoomScale, setZoomScale] = useState<number>(1.0); // 1.0, 1.5, 2.0
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number; visible: boolean; radius: number }>({
    x: 0,
    y: 0,
    visible: false,
    radius: 16,
  });

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Helper to check if mask has any white pixels painted
  const checkMaskContent = useCallback(() => {
    const offscreen = offscreenMaskRef.current;
    if (!offscreen) {
      setHasMaskDrawn(false);
      return;
    }
    const ctx = offscreen.getContext('2d');
    if (!ctx) return;
    try {
      const data = ctx.getImageData(0, 0, offscreen.width, offscreen.height).data;
      let found = false;
      // Stride sampling every 16th pixel for speed
      for (let i = 0; i < data.length; i += 64) {
        if (data[i] > 30) {
          found = true;
          break;
        }
      }
      setHasMaskDrawn(found);
    } catch (e) {
      // Cross-origin fallback
      setHasMaskDrawn(true);
    }
  }, []);

  // Save current canvas state to undo stack
  const saveHistoryState = useCallback(() => {
    const offscreen = offscreenMaskRef.current;
    if (!offscreen) return;
    const ctx = offscreen.getContext('2d');
    if (!ctx) return;
    try {
      const snapshot = ctx.getImageData(0, 0, offscreen.width, offscreen.height);
      undoStackRef.current.push(snapshot);
      if (undoStackRef.current.length > 15) {
        undoStackRef.current.shift();
      }
      redoStackRef.current = [];
      setCanUndo(true);
      setCanRedo(false);
    } catch (e) {
      console.warn('Could not capture canvas history snapshot:', e);
    }
  }, []);

  // Sync offscreen mask to on-screen canvas
  const redrawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Draw source image
    if (imageObjRef.current) {
      ctx.drawImage(imageObjRef.current, 0, 0, canvas.width, canvas.height);
    }

    // 2. Overlay the mask with semi-transparent red
    if (offscreenMaskRef.current) {
      ctx.save();
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = canvas.width;
      tempCanvas.height = canvas.height;
      const tempCtx = tempCanvas.getContext('2d');
      if (tempCtx) {
        if (featherRadius > 0) {
          tempCtx.filter = `blur(${featherRadius}px)`;
        }
        tempCtx.drawImage(offscreenMaskRef.current, 0, 0, canvas.width, canvas.height);
        tempCtx.filter = 'none';
        tempCtx.globalCompositeOperation = 'source-in';
        tempCtx.fillStyle = 'rgba(239, 68, 68, 0.65)'; // Semi-transparent red overlay
        tempCtx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.drawImage(tempCanvas, 0, 0);
      }
      ctx.restore();
    }
  }, [featherRadius]);

  // Upload mask to backend
  const uploadCurrentMask = useCallback(async () => {
    const offscreen = offscreenMaskRef.current;
    if (!offscreen) return;

    try {
      setIsUploading(true);

      let targetCanvas = offscreen;
      if (featherRadius > 0) {
        const feathered = document.createElement('canvas');
        feathered.width = offscreen.width;
        feathered.height = offscreen.height;
        const fCtx = feathered.getContext('2d');
        if (fCtx) {
          fCtx.fillStyle = '#000000';
          fCtx.fillRect(0, 0, feathered.width, feathered.height);
          fCtx.filter = `blur(${featherRadius}px)`;
          fCtx.drawImage(offscreen, 0, 0);
          targetCanvas = feathered;
        }
      }

      const blob = await new Promise<Blob | null>((resolve) => {
        targetCanvas.toBlob((b) => resolve(b), 'image/png');
      });

      if (!blob) return;

      const formData = new FormData();
      formData.append('file', blob, 'inpaint_mask.png');

      const res = await fetch('/api/generations/upload-mask', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error('Failed to upload painted mask');
      }

      const data = await res.json();
      setMaskImage(data.mask_image_id, data.url);
    } catch (err: any) {
      console.error('Error uploading inpaint mask:', err);
      setErrorMessage(err.message || 'Mask upload error');
    } finally {
      setIsUploading(false);
    }
  }, [featherRadius, setErrorMessage, setMaskImage]);

  // Re-upload when feathering changes on an existing mask
  useEffect(() => {
    if (hasMaskDrawn && source_image_preview) {
      redrawCanvas();
      const timer = setTimeout(() => {
        uploadCurrentMask();
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [featherRadius, hasMaskDrawn, redrawCanvas, source_image_preview, uploadCurrentMask]);

  // Initialize or re-render when source_image_preview changes
  useEffect(() => {
    if (!source_image_preview) {
      imageObjRef.current = null;
      offscreenMaskRef.current = null;
      undoStackRef.current = [];
      redoStackRef.current = [];
      setCanUndo(false);
      setCanRedo(false);
      setHasMaskDrawn(false);
      clearMask();
      return;
    }

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = source_image_preview;
    img.onload = () => {
      imageObjRef.current = img;

      const canvas = canvasRef.current;
      if (!canvas) return;

      const width = img.naturalWidth || 1024;
      const height = img.naturalHeight || 1024;

      canvas.width = width;
      canvas.height = height;

      // Offscreen pure black & white mask
      const offscreen = document.createElement('canvas');
      offscreen.width = width;
      offscreen.height = height;
      const offCtx = offscreen.getContext('2d');
      if (offCtx) {
        offCtx.fillStyle = '#000000';
        offCtx.fillRect(0, 0, width, height);
      }
      offscreenMaskRef.current = offscreen;
      undoStackRef.current = [];
      redoStackRef.current = [];
      setCanUndo(false);
      setCanRedo(false);
      setHasMaskDrawn(false);
      redrawCanvas();
    };
  }, [source_image_preview, clearMask, redrawCanvas]);

  // Coordinate mapping taking into account client scale and zoom
  const getCanvasCoords = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!source_image_preview) return;
    saveHistoryState();
    setIsDrawing(true);
    const { x, y } = getCanvasCoords(e);
    lastPointRef.current = { x, y };

    const offscreen = offscreenMaskRef.current;
    if (offscreen) {
      const ctx = offscreen.getContext('2d');
      if (ctx) {
        ctx.save();
        ctx.beginPath();
        ctx.arc(x, y, brush_size / 2, 0, Math.PI * 2);
        ctx.fillStyle = brush_mode === 'draw' ? '#ffffff' : '#000000';
        ctx.fill();
        ctx.restore();
      }
    }
    setHasMaskDrawn(true);
    redrawCanvas();
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (canvas) {
      const rect = canvas.getBoundingClientRect();
      const clientX = e.clientX - rect.left;
      const clientY = e.clientY - rect.top;
      const displayRadius = (brush_size / 2) * (rect.width / canvas.width);
      setCursorPos({
        x: clientX,
        y: clientY,
        visible: true,
        radius: displayRadius,
      });
    }

    if (!isDrawing || !lastPointRef.current) return;
    const curr = getCanvasCoords(e);

    const offscreen = offscreenMaskRef.current;
    if (offscreen) {
      const ctx = offscreen.getContext('2d');
      if (ctx) {
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(lastPointRef.current.x, lastPointRef.current.y);
        ctx.lineTo(curr.x, curr.y);
        ctx.lineWidth = brush_size;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = brush_mode === 'draw' ? '#ffffff' : '#000000';
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(curr.x, curr.y, brush_size / 2, 0, Math.PI * 2);
        ctx.fillStyle = brush_mode === 'draw' ? '#ffffff' : '#000000';
        ctx.fill();
        ctx.restore();
      }
    }

    lastPointRef.current = curr;
    setHasMaskDrawn(true);
    redrawCanvas();
  };

  const handleMouseUp = () => {
    if (!isDrawing) return;
    setIsDrawing(false);
    lastPointRef.current = null;
    checkMaskContent();
    uploadCurrentMask();
  };

  const handleMouseLeave = () => {
    setCursorPos((prev) => ({ ...prev, visible: false }));
    if (isDrawing) {
      handleMouseUp();
    }
  };

  const handleUndo = () => {
    const offscreen = offscreenMaskRef.current;
    if (!offscreen || undoStackRef.current.length === 0) return;
    const ctx = offscreen.getContext('2d');
    if (!ctx) return;

    try {
      const currentSnapshot = ctx.getImageData(0, 0, offscreen.width, offscreen.height);
      redoStackRef.current.push(currentSnapshot);

      const prevState = undoStackRef.current.pop()!;
      ctx.putImageData(prevState, 0, 0);

      setCanUndo(undoStackRef.current.length > 0);
      setCanRedo(true);

      checkMaskContent();
      redrawCanvas();
      uploadCurrentMask();
    } catch (e) {
      console.warn('Error during undo:', e);
    }
  };

  const handleRedo = () => {
    const offscreen = offscreenMaskRef.current;
    if (!offscreen || redoStackRef.current.length === 0) return;
    const ctx = offscreen.getContext('2d');
    if (!ctx) return;

    try {
      const currentSnapshot = ctx.getImageData(0, 0, offscreen.width, offscreen.height);
      undoStackRef.current.push(currentSnapshot);

      const nextState = redoStackRef.current.pop()!;
      ctx.putImageData(nextState, 0, 0);

      setCanUndo(true);
      setCanRedo(redoStackRef.current.length > 0);

      checkMaskContent();
      redrawCanvas();
      uploadCurrentMask();
    } catch (e) {
      console.warn('Error during redo:', e);
    }
  };

  const handleInvertMask = () => {
    const offscreen = offscreenMaskRef.current;
    if (!offscreen) return;
    const ctx = offscreen.getContext('2d');
    if (!ctx) return;

    try {
      saveHistoryState();
      const imgData = ctx.getImageData(0, 0, offscreen.width, offscreen.height);
      const d = imgData.data;
      for (let i = 0; i < d.length; i += 4) {
        d[i] = 255 - d[i];
        d[i + 1] = 255 - d[i + 1];
        d[i + 2] = 255 - d[i + 2];
        d[i + 3] = 255;
      }
      ctx.putImageData(imgData, 0, 0);

      checkMaskContent();
      redrawCanvas();
      uploadCurrentMask();
    } catch (e) {
      console.warn('Error during mask invert:', e);
    }
  };

  const handleClearMask = () => {
    const offscreen = offscreenMaskRef.current;
    if (offscreen) {
      saveHistoryState();
      const ctx = offscreen.getContext('2d');
      if (ctx) {
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, offscreen.width, offscreen.height);
      }
    }
    setHasMaskDrawn(false);
    clearMask();
    redrawCanvas();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setErrorMessage('Please select a valid image file (PNG, JPG, WEBP).');
      return;
    }

    try {
      setIsUploading(true);
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch('/api/generations/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error('Failed to upload image');
      }

      const data = await res.json();
      setSourceImage(data.source_image_id, data.url);
      setMaskImage(null, null);
      setHasMaskDrawn(false);
    } catch (err: any) {
      setErrorMessage(err.message || 'Image upload failed');
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // Keyboard shortcut listener on container
  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      if (e.shiftKey) {
        handleRedo();
      } else {
        handleUndo();
      }
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
      e.preventDefault();
      handleRedo();
    } else if (e.key === '[') {
      e.preventDefault();
      setBrushSize(Math.max(8, brush_size - 4));
    } else if (e.key === ']') {
      e.preventDefault();
      setBrushSize(Math.min(128, brush_size + 4));
    }
  };

  return (
    <div
      ref={containerRef}
      onKeyDown={handleKeyDown}
      tabIndex={0}
      className="flex flex-col gap-3 p-4 bg-surface rounded-xl border border-border-subtle focus:outline-none focus:border-accent/40"
    >
      {/* Header and Controls */}
      <div className="flex items-center justify-between flex-wrap gap-2 pb-2 border-b border-border-subtle">
        <div className="flex items-center gap-2">
          <Paintbrush className="w-4 h-4 text-accent" />
          <span className="text-xs font-semibold uppercase tracking-wider text-text-primary">
            Inpaint Mask Studio
          </span>
          {source_image_preview && (
            <span
              className={`text-[11px] px-2 py-0.5 rounded-full flex items-center gap-1 font-medium ${
                hasMaskDrawn && mask_image_id
                  ? 'bg-emerald-950/70 text-emerald-400 border border-emerald-800'
                  : 'bg-amber-950/60 text-accent border border-accent/40'
              }`}
            >
              {hasMaskDrawn && mask_image_id ? (
                <>
                  <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                  Mask Ready
                </>
              ) : (
                <>
                  <AlertCircle className="w-3 h-3 text-accent" />
                  Paint Mask Required
                </>
              )}
            </span>
          )}
        </div>

        {/* Toolbar */}
        {source_image_preview && (
          <div className="flex items-center gap-1.5 flex-wrap">
            {/* Draw mode */}
            <button
              type="button"
              onClick={() => setBrushMode('draw')}
              className={`px-2 py-1 rounded text-xs flex items-center gap-1 transition-colors ${
                brush_mode === 'draw'
                  ? 'bg-accent text-neutral-950 font-semibold'
                  : 'bg-surface-elevated hover:bg-neutral-800 text-text-muted hover:text-text-primary'
              }`}
              title="Paint Mask (White)"
            >
              <Paintbrush className="w-3.5 h-3.5" />
              <span>Brush</span>
            </button>

            {/* Erase mode */}
            <button
              type="button"
              onClick={() => setBrushMode('erase')}
              className={`px-2 py-1 rounded text-xs flex items-center gap-1 transition-colors ${
                brush_mode === 'erase'
                  ? 'bg-accent text-neutral-950 font-semibold'
                  : 'bg-surface-elevated hover:bg-neutral-800 text-text-muted hover:text-text-primary'
              }`}
              title="Erase Mask"
            >
              <Eraser className="w-3.5 h-3.5" />
              <span>Erase</span>
            </button>

            {/* Invert Mask */}
            <button
              type="button"
              onClick={handleInvertMask}
              className="p-1.5 rounded text-xs bg-surface-elevated hover:bg-neutral-800 text-text-muted hover:text-text-primary transition-colors"
              title="Invert Mask (Swap White and Black)"
            >
              <FlipHorizontal className="w-3.5 h-3.5" />
            </button>

            {/* Undo */}
            <button
              type="button"
              onClick={handleUndo}
              disabled={!canUndo}
              className={`p-1.5 rounded text-xs transition-colors ${
                canUndo
                  ? 'bg-surface-elevated hover:bg-neutral-800 text-text-muted hover:text-text-primary cursor-pointer'
                  : 'bg-surface-elevated/40 text-neutral-600 cursor-not-allowed'
              }`}
              title="Undo Stroke (Ctrl+Z)"
            >
              <Undo2 className="w-3.5 h-3.5" />
            </button>

            {/* Redo */}
            <button
              type="button"
              onClick={handleRedo}
              disabled={!canRedo}
              className={`p-1.5 rounded text-xs transition-colors ${
                canRedo
                  ? 'bg-surface-elevated hover:bg-neutral-800 text-text-muted hover:text-text-primary cursor-pointer'
                  : 'bg-surface-elevated/40 text-neutral-600 cursor-not-allowed'
              }`}
              title="Redo Stroke (Ctrl+Y)"
            >
              <Redo2 className="w-3.5 h-3.5" />
            </button>

            {/* Clear Mask */}
            <button
              type="button"
              onClick={handleClearMask}
              className="p-1.5 rounded text-xs bg-surface-elevated hover:bg-red-950/60 hover:text-red-400 text-text-muted flex items-center gap-1 transition-colors"
              title="Clear entire mask"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>

            {/* Replace image */}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="px-2 py-1 rounded text-xs bg-surface-elevated hover:bg-neutral-800 text-text-muted hover:text-text-primary flex items-center gap-1 transition-colors"
              title="Upload new source image"
            >
              <Upload className="w-3.5 h-3.5" />
              <span>Change</span>
            </button>
          </div>
        )}
      </div>

      {/* Sliders Bar (Brush Size, Feathering, Zoom) */}
      {source_image_preview && (
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 px-3 py-2 bg-surface-elevated/50 rounded-lg border border-border-subtle text-xs">
          {/* Brush Size */}
          <div className="flex items-center gap-2 flex-1">
            <Sliders className="w-3.5 h-3.5 text-text-muted shrink-0" />
            <span className="text-text-muted whitespace-nowrap">Size:</span>
            <input
              type="range"
              min="8"
              max="128"
              value={brush_size}
              onChange={(e) => setBrushSize(parseInt(e.target.value, 10))}
              className="w-full accent-accent cursor-pointer h-1.5 bg-neutral-800 rounded-lg"
            />
            <span className="text-accent font-mono w-10 text-right">{brush_size}px</span>
          </div>

          <div className="h-4 w-px bg-border-subtle hidden sm:block" />

          {/* Feathering Slider */}
          <div className="flex items-center gap-2 flex-1">
            <Sparkles className="w-3.5 h-3.5 text-text-muted shrink-0" />
            <span className="text-text-muted whitespace-nowrap">Feather:</span>
            <input
              type="range"
              min="0"
              max="16"
              value={featherRadius}
              onChange={(e) => setFeatherRadius(parseInt(e.target.value, 10))}
              className="w-full accent-accent cursor-pointer h-1.5 bg-neutral-800 rounded-lg"
            />
            <span className="text-accent font-mono w-10 text-right">{featherRadius}px</span>
          </div>

          <div className="h-4 w-px bg-border-subtle hidden sm:block" />

          {/* Zoom controls */}
          <div className="flex items-center gap-1 shrink-0 self-end sm:self-auto">
            <span className="text-text-muted mr-1">Zoom:</span>
            {[
              { label: 'Fit', scale: 1.0 },
              { label: '1.5×', scale: 1.5 },
              { label: '2.0×', scale: 2.0 },
            ].map((z) => (
              <button
                key={z.label}
                type="button"
                onClick={() => setZoomScale(z.scale)}
                className={`px-1.5 py-0.5 rounded text-[11px] font-mono transition-colors ${
                  zoomScale === z.scale
                    ? 'bg-accent/20 text-accent border border-accent/40 font-semibold'
                    : 'bg-surface-elevated text-text-muted hover:text-text-primary border border-border-subtle'
                }`}
              >
                {z.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Main Canvas Area */}
      <div
        className="relative min-h-[340px] max-h-[540px] flex items-center justify-center bg-background rounded-lg border border-border-subtle overflow-auto p-2"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleFileChange}
        />

        {!source_image_preview ? (
          <div
            onClick={() => fileInputRef.current?.click()}
            className="flex flex-col items-center justify-center p-8 text-center cursor-pointer hover:bg-surface-elevated/40 transition-colors w-full h-full min-h-[320px]"
          >
            <div className="w-14 h-14 rounded-full bg-surface-elevated border border-border-subtle flex items-center justify-center mb-3 text-accent shadow-inner">
              <Upload className="w-6 h-6" />
            </div>
            <p className="text-sm font-medium text-text-primary mb-1">
              Upload Source Image to Inpaint
            </p>
            <p className="text-xs text-text-muted max-w-xs mb-3">
              Drop an image here or click to browse. Paint over regions you want the AI to replace or repair.
            </p>
            <span className="text-xs font-semibold text-accent bg-accent/10 px-3 py-1 rounded-md border border-accent/20">
              Browse Image
            </span>
          </div>
        ) : (
          <div className="relative inline-block max-w-full max-h-full">
            <canvas
              ref={canvasRef}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseLeave}
              className={`max-w-full max-h-[480px] object-contain rounded shadow-lg border border-border-subtle ${
                brush_mode === 'draw' ? 'cursor-crosshair' : 'cursor-cell'
              }`}
              style={{
                touchAction: 'none',
                transform: zoomScale > 1.0 ? `scale(${zoomScale})` : undefined,
                transformOrigin: 'center center',
              }}
            />

            {/* Brush cursor ring preview */}
            {cursorPos.visible && (
              <div
                className="pointer-events-none absolute rounded-full border border-accent/80 bg-accent/20"
                style={{
                  left: `${cursorPos.x - cursorPos.radius}px`,
                  top: `${cursorPos.y - cursorPos.radius}px`,
                  width: `${cursorPos.radius * 2}px`,
                  height: `${cursorPos.radius * 2}px`,
                }}
              />
            )}
          </div>
        )}

        {isUploading && (
          <div className="absolute inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center text-accent text-xs font-medium gap-2">
            <div className="w-4 h-4 border-2 border-accent border-t-transparent rounded-full animate-spin" />
            Syncing inpaint mask...
          </div>
        )}
      </div>

      {source_image_preview && (
        <div className="flex items-center justify-between text-[11px] text-text-muted px-1 flex-wrap gap-2">
          <span>
            Shortcut keys: <kbd className="font-mono bg-surface-elevated px-1 py-0.5 rounded border border-border-subtle">Ctrl+Z</kbd> Undo, <kbd className="font-mono bg-surface-elevated px-1 py-0.5 rounded border border-border-subtle">Ctrl+Y</kbd> Redo, <kbd className="font-mono bg-surface-elevated px-1 py-0.5 rounded border border-border-subtle">[</kbd> / <kbd className="font-mono bg-surface-elevated px-1 py-0.5 rounded border border-border-subtle">]</kbd> resize brush.
          </span>
          <span>Denoise 0.70–0.90 recommended for repairs.</span>
        </div>
      )}
    </div>
  );
};
