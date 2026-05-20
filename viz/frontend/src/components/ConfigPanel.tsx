import { Wand2, Gauge, Terminal } from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { RadioGroup, RadioCard } from "@/components/ui/radio-group";
import { Switch } from "@/components/ui/switch";

export type ConfigValues = {
  model: string;
  usePreprocess: boolean;
  showRaw: boolean;
};

type Props = {
  models: { key: string; label: string; description: string }[];
  value: ConfigValues;
  onChange: (next: ConfigValues) => void;
};

export function ConfigPanel({ models, value, onChange }: Props) {
  const setModel = (model: string) => onChange({ ...value, model });
  const setPrep = (usePreprocess: boolean) => onChange({ ...value, usePreprocess });
  const setRaw = (showRaw: boolean) => onChange({ ...value, showRaw });

  return (
    <Card>
      <CardHeader>
        <div>
          <CardTitle className="flex items-center gap-2">
            <Gauge className="h-4 w-4 text-lavender-500" /> Model & processing
          </CardTitle>
          <CardDescription>Pick an architecture and tune the input pipeline.</CardDescription>
        </div>
      </CardHeader>

      <RadioGroup value={value.model} onValueChange={setModel} className="gap-2">
        {models.map((m) => (
          <RadioCard
            key={m.key}
            value={m.key}
            label={m.label}
            description={m.description}
          />
        ))}
      </RadioGroup>

      <div className="mt-5 space-y-3">
        <ToggleRow
          icon={<Wand2 className="h-4 w-4 text-lavender-500" />}
          title="Adaptive thresholding"
          subtitle="Cleans uneven lighting and shadows before OCR."
          checked={value.usePreprocess}
          onChange={setPrep}
        />
        <ToggleRow
          icon={<Terminal className="h-4 w-4 text-lavender-500" />}
          title="Show raw logs"
          subtitle="Display the last 3 KB of infer_rec.py output."
          checked={value.showRaw}
          onChange={setRaw}
        />
      </div>
    </Card>
  );
}

function ToggleRow({
  icon,
  title,
  subtitle,
  checked,
  onChange,
}: {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  checked: boolean;
  onChange: (next: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-start justify-between gap-4 rounded-2xl border border-lavender-100 bg-white/60 p-3 transition-colors hover:bg-lavender-50/60">
      <span className="flex items-start gap-3">
        <span className="mt-0.5">{icon}</span>
        <span className="flex flex-col">
          <span className="text-sm font-medium text-ink">{title}</span>
          <span className="text-xs text-ink-muted">{subtitle}</span>
        </span>
      </span>
      <Switch checked={checked} onCheckedChange={onChange} />
    </label>
  );
}
