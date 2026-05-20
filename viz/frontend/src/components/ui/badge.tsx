import * as React from "react";
import { cn } from "@/lib/utils";

type Tone = "lavender" | "green" | "amber" | "red" | "slate";

const toneClasses: Record<Tone, string> = {
  lavender: "bg-lavender-50 text-lavender-700 border-lavender-200/70",
  green: "bg-emerald-50 text-emerald-700 border-emerald-200/70",
  amber: "bg-amber-50 text-amber-700 border-amber-200/70",
  red: "bg-rose-50 text-rose-700 border-rose-200/70",
  slate: "bg-slate-50 text-slate-700 border-slate-200/70",
};

export function Badge({
  className,
  tone = "lavender",
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & { tone?: Tone }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium",
        toneClasses[tone],
        className,
      )}
      {...props}
    />
  );
}
