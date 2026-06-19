import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, ClipboardCopy, Cpu, FileText, Inbox, ScanText, Timer, Sparkles, Loader2, Check } from "lucide-react";
import { useCallback, useState } from "react";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn, formatPercent, formatSeconds } from "@/lib/utils";
import type { InferResponse } from "@/lib/gradio";

type Props = {
  status: "idle" | "loading" | "success" | "error";
  response: InferResponse | null;
  error: string | null;
  showRaw: boolean;
  spellCorrected: string | null;
  isSpellChecking: boolean;
  spellCheckError: string | null;
  onSpellCheck: () => void;
};

function confidenceTone(conf: number): { label: string; cls: string; tone: "green" | "amber" | "red" } {
  if (conf >= 0.85) return { label: "High", cls: "bg-emerald-500", tone: "green" };
  if (conf >= 0.6) return { label: "Medium", cls: "bg-amber-500", tone: "amber" };
  return { label: "Low", cls: "bg-rose-500", tone: "red" };
}

export function ResultsPanel({
  status,
  response,
  error,
  showRaw,
  spellCorrected,
  isSpellChecking,
  spellCheckError,
  onSpellCheck,
}: Props) {
  return (
    <Card>
      <CardHeader>
        <div>
          <CardTitle className="flex items-center gap-2">
            <ScanText className="h-4 w-4 text-lavender-500" />
            <span>Inference results</span>
          </CardTitle>
          <CardDescription>
            {response?.model
              ? <>Processed via <span className="font-medium text-ink">{response.model}</span> · {formatSeconds(response.elapsed)}</>
              : "Results appear here after recognition."}
          </CardDescription>
        </div>
        {response?.model ? (
          <Badge tone="lavender">
            <Cpu className="h-3.5 w-3.5" />
            {response.model}
          </Badge>
        ) : null}
      </CardHeader>

      <AnimatePresence mode="wait" initial={false}>
        {status === "loading" ? (
          <SkeletonResults key="skeleton" />
        ) : status === "error" ? (
          <ErrorState key="error" message={error ?? "Unknown error."} />
        ) : status === "success" && response ? (
          response.results.length > 0 ? (
            <SuccessResults
              key="ok"
              response={response}
              showRaw={showRaw}
              spellCorrected={spellCorrected}
              isSpellChecking={isSpellChecking}
              spellCheckError={spellCheckError}
              onSpellCheck={onSpellCheck}
            />
          ) : (
            <EmptyResults key="empty" message="No text detected. Try enabling Adaptive thresholding." raw={showRaw ? response.raw : undefined} />
          )
        ) : (
          <IdleState key="idle" />
        )}
      </AnimatePresence>
    </Card>
  );
}

function IdleState() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-lavender-200 bg-white/40 py-16 text-center text-ink-muted"
    >
      <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-lavender-50 text-lavender-500">
        <Inbox className="h-6 w-6" />
      </span>
      <p className="text-sm">Awaiting input</p>
      <p className="max-w-xs text-xs text-ink-soft">
        Upload an image and hit <span className="font-medium text-ink-muted">Recognize text</span> to see the model's predictions.
      </p>
    </motion.div>
  );
}

function SkeletonResults() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="space-y-3"
    >
      {[0, 1].map((i) => (
        <div
          key={i}
          className="animate-pulse rounded-2xl border border-lavender-100 bg-white/70 p-4"
        >
          <div className="mb-3 h-4 w-3/4 rounded bg-lavender-100" />
          <div className="h-3 w-1/3 rounded bg-lavender-50" />
        </div>
      ))}
    </motion.div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4 }}
      className="flex items-start gap-3 rounded-2xl border border-rose-200 bg-rose-50/60 p-4 text-rose-700"
    >
      <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" />
      <div className="text-sm">
        <p className="font-semibold">Inference failed</p>
        <p className="mt-1 break-words text-rose-700/90">{message}</p>
      </div>
    </motion.div>
  );
}

function EmptyResults({ message, raw }: { message: string; raw?: string | undefined }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4 }}
      className="space-y-3"
    >
      <div className="rounded-2xl border border-amber-200 bg-amber-50/60 p-4 text-sm text-amber-800">
        {message}
      </div>
      {raw ? <RawLogs raw={raw} /> : null}
    </motion.div>
  );
}

function SuccessResults({
  response,
  showRaw,
  spellCorrected,
  isSpellChecking,
  spellCheckError,
  onSpellCheck,
}: {
  response: InferResponse;
  showRaw: boolean;
  spellCorrected: string | null;
  isSpellChecking: boolean;
  spellCheckError: string | null;
  onSpellCheck: () => void;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    if (!spellCorrected) return;
    try {
      await navigator.clipboard.writeText(spellCorrected);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      /* ignore */
    }
  }, [spellCorrected]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4 }}
      className="space-y-4"
    >
      <div className="space-y-3">
        {response.results.map((r, idx) => (
          <ResultRow key={idx} text={r.text} conf={r.conf} elapsed={response.elapsed} />
        ))}
      </div>

      <div className="border-t border-lavender-100/60 pt-4 mt-2">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-ink-soft">
            Spell Correction
          </h4>
        </div>

        {isSpellChecking ? (
          <div className="flex items-center gap-2 text-sm text-ink-muted bg-lavender-50/30 border border-lavender-100/50 rounded-2xl p-4 animate-pulse">
            <Loader2 className="h-4 w-4 animate-spin text-lavender-500" />
            <span>Đang sửa chính tả...</span>
          </div>
        ) : spellCheckError ? (
          <div className="flex items-start gap-2.5 text-xs text-rose-700 bg-rose-50/60 border border-rose-100 rounded-2xl p-4">
            <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold">Lỗi sửa chính tả</p>
              <p className="mt-0.5 text-rose-700/80">{spellCheckError}</p>
              <Button
                variant="secondary"
                size="sm"
                className="mt-2 text-rose-700 border-rose-200/50 hover:bg-rose-50"
                onClick={onSpellCheck}
              >
                Thử lại
              </Button>
            </div>
          </div>
        ) : spellCorrected ? (
          <div className="group relative rounded-2xl border border-dashed border-lavender-300 bg-lavender-50/20 p-4 transition-colors hover:border-lavender-400">
            <div className="flex items-start justify-between gap-3">
              <p className="font-display text-lg font-medium leading-snug text-ink whitespace-pre-wrap">
                {spellCorrected}
              </p>
              <button
                type="button"
                onClick={handleCopy}
                className="opacity-0 transition-opacity group-hover:opacity-100 focus:opacity-100 shrink-0"
                aria-label="Copy corrected text"
              >
                {copied ? (
                  <Check className="h-4 w-4 text-emerald-500" />
                ) : (
                  <ClipboardCopy className="h-4 w-4 text-ink-soft hover:text-lavender-600" />
                )}
              </button>
            </div>
            <div className="mt-2 flex items-center gap-1.5 text-[10px] font-medium text-lavender-500">
              <Sparkles className="h-3 w-3" />
              <span>Đã sửa bởi bmd1905/vietnamese-correction-v2</span>
            </div>
          </div>
        ) : (
          <Button
            variant="secondary"
            className="w-full flex items-center justify-center gap-2"
            onClick={onSpellCheck}
          >
            <Sparkles className="h-4 w-4 text-lavender-500" />
            <span>Sửa chính tả</span>
          </Button>
        )}
      </div>

      {showRaw ? <RawLogs raw={response.raw} /> : null}
    </motion.div>
  );
}

function ResultRow({ text, conf, elapsed }: { text: string; conf: number; elapsed: number }) {
  const tone = confidenceTone(conf);
  const [copied, setCopied] = useState(false);

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      /* ignore */
    }
  }, [text]);

  return (
    <div className="group rounded-2xl border border-lavender-100 bg-white p-4 shadow-soft transition-colors hover:border-lavender-200">
      <div className="flex items-start justify-between gap-3">
        <p className="font-display text-lg font-medium leading-snug text-ink">{text}</p>
        <button
          type="button"
          onClick={copy}
          className="opacity-0 transition-opacity group-hover:opacity-100 focus:opacity-100"
          aria-label="Copy text"
        >
          <ClipboardCopy
            className={cn("h-4 w-4", copied ? "text-emerald-500" : "text-ink-soft hover:text-lavender-600")}
          />
        </button>
      </div>
      <div className="mt-3 flex items-center justify-between gap-3 text-xs text-ink-muted">
        <span className="flex items-center gap-2">
          <span className={cn("dot", tone.cls)} />
          <span>
            Confidence <span className="font-medium text-ink">{formatPercent(conf)}</span>
            <span className="ml-1 text-ink-soft">· {tone.label}</span>
          </span>
        </span>
        <span className="flex items-center gap-1 text-ink-soft">
          <Timer className="h-3.5 w-3.5" />
          {formatSeconds(elapsed)}
        </span>
      </div>
    </div>
  );
}

function RawLogs({ raw }: { raw: string }) {
  const tail = raw && raw.length > 3000 ? raw.slice(-3000) : raw || "(empty)";
  return (
    <details className="rounded-2xl border border-lavender-100 bg-white/70 p-3 text-xs text-ink-muted">
      <summary className="flex cursor-pointer items-center gap-2 font-medium text-ink">
        <FileText className="h-3.5 w-3.5 text-lavender-500" />
        System logs
        <Button
          asChild
          variant="ghost"
          size="sm"
          className="ml-auto text-xs"
          onClick={(e) => {
            e.preventDefault();
            navigator.clipboard?.writeText(raw).catch(() => undefined);
          }}
        >
          <span><ClipboardCopy className="h-3 w-3" /> Copy</span>
        </Button>
      </summary>
      <pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap rounded-xl bg-slate-950/95 p-3 font-mono text-[11px] leading-snug text-slate-100">
        {tail}
      </pre>
    </details>
  );
}
