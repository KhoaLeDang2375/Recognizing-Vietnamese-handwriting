import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ImageUp, RotateCcw, ScanLine, Upload } from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

const ACCEPT = "image/jpeg,image/png,image/bmp,image/tiff";

export type UploadedImage = {
  file: File;
  url: string;
  name: string;
  sizeBytes: number;
  width: number;
  height: number;
};

type Props = {
  value: UploadedImage | null;
  onChange: (img: UploadedImage | null) => void;
  onRun: () => void;
  isLoading: boolean;
  canRun: boolean;
};

export function UploadCard({ value, onChange, onRun, isLoading, canRun }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return;
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        onChange({
          file,
          url,
          name: file.name,
          sizeBytes: file.size,
          width: img.naturalWidth,
          height: img.naturalHeight,
        });
      };
      img.src = url;
    },
    [onChange],
  );

  useEffect(() => {
    return () => {
      if (value?.url) URL.revokeObjectURL(value.url);
    };
    // We only want this on unmount of the latest URL, not on every value change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value?.url]);

  return (
    <Card>
      <CardHeader>
        <div>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-4 w-4 text-lavender-500" />
            <span>Upload handwriting</span>
          </CardTitle>
          <CardDescription>One line per image · JPG / PNG / BMP / TIFF</CardDescription>
        </div>
        {value ? (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              if (value.url) URL.revokeObjectURL(value.url);
              onChange(null);
            }}
          >
            <RotateCcw className="h-3.5 w-3.5" /> Clear
          </Button>
        ) : null}
      </CardHeader>

      <AnimatePresence mode="wait" initial={false}>
        {value ? (
          <motion.div
            key="preview"
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.18 }}
            className="space-y-4"
          >
            <div className="overflow-hidden rounded-2xl border border-lavender-100 bg-white">
              <img
                src={value.url}
                alt={value.name}
                className="max-h-72 w-full object-contain"
              />
            </div>
            <div className="flex flex-wrap items-center gap-2 text-xs text-ink-muted">
              <Badge tone="slate">{value.width} × {value.height} px</Badge>
              <Badge tone="slate">{(value.sizeBytes / 1024).toFixed(1)} KB</Badge>
              <Badge tone="slate">AR {(value.width / value.height).toFixed(2)}</Badge>
              <span className="ml-auto truncate font-medium text-ink">{value.name}</span>
            </div>

            <Button
              onClick={onRun}
              disabled={!canRun || isLoading}
              size="lg"
              className="w-full"
            >
              {isLoading ? (
                <>
                  <motion.span
                    aria-hidden
                    className="h-4 w-4 rounded-full border-2 border-white/40 border-t-white"
                    animate={{ rotate: 360 }}
                    transition={{ repeat: Infinity, duration: 0.9, ease: "linear" }}
                  />
                  Running inference…
                </>
              ) : (
                <>
                  <ScanLine className="h-4 w-4" /> Recognize text
                </>
              )}
            </Button>
          </motion.div>
        ) : (
          <motion.button
            key="dropzone"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            type="button"
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              const file = e.dataTransfer.files?.[0];
              if (file) handleFile(file);
            }}
            className={
              "flex w-full flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed py-12 text-center transition-colors " +
              (dragOver
                ? "border-lavender-400 bg-lavender-50"
                : "border-lavender-200 bg-white/60 hover:border-lavender-300 hover:bg-lavender-50/50")
            }
          >
            <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-lavender-100 text-lavender-600">
              <ImageUp className="h-6 w-6" />
            </span>
            <span className="font-display text-base font-semibold text-ink">
              Drop an image, or click to browse
            </span>
            <span className="max-w-xs text-xs text-ink-muted">
              Tip: enable <em>Adaptive thresholding</em> for photos with shadows
              or uneven lighting.
            </span>
          </motion.button>
        )}
      </AnimatePresence>

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
          e.currentTarget.value = "";
        }}
      />
    </Card>
  );
}
