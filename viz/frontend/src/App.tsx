import { useMemo, useState } from "react";
import { MonitorPlay } from "lucide-react";
import { Header } from "@/components/Header";
import { Hero } from "@/components/Hero";
import { TeamSection } from "@/components/TeamSection";
import { SectionHeading } from "@/components/SectionHeading";
import { ConfigPanel, type ConfigValues } from "@/components/ConfigPanel";
import { UploadCard, type UploadedImage } from "@/components/UploadCard";
import { ResultsPanel } from "@/components/ResultsPanel";
import { ProcessedImageCard } from "@/components/ProcessedImageCard";
import { ArchitectureSection } from "@/components/ArchitectureSection";
import { TrainingSection } from "@/components/TrainingSection";
import { BenchmarkSection } from "@/components/BenchmarkSection";
import { useGradioInfer } from "@/hooks/useGradioInfer";
import { resolveGradioUrl, spellCheck } from "@/lib/gradio";


const MODELS = [
  {
    key: "SVTR (High Accuracy)",
    label: "SVTR · High accuracy",
    description: "Transformer-based, best characters/word, slower.",
  },
  {
    key: "CRNN (High Speed)",
    label: "CRNN · High speed",
    description: "CNN + RNN, faster inference, slightly lower accuracy.",
  },
];

export default function App() {
  const [config, setConfig] = useState<ConfigValues>({
    model: MODELS[0]!.key,
    usePreprocess: false,
    showRaw: false,
  });
  const [image, setImage] = useState<UploadedImage | null>(null);
  const { status, response, error, run } = useGradioInfer();

  const [spellCorrected, setSpellCorrected] = useState<string | null>(null);
  const [isSpellChecking, setIsSpellChecking] = useState<boolean>(false);
  const [spellCheckError, setSpellCheckError] = useState<string | null>(null);

  const gradioUrl = useMemo(() => resolveGradioUrl(), []);

  const onRun = async () => {
    if (!image) return;
    setSpellCorrected(null);
    setSpellCheckError(null);
    setIsSpellChecking(false);
    await run(image.file, config.model, config.usePreprocess);
  };

  const handleSpellCheck = async () => {
    if (!response || response.results.length === 0) return;
    setIsSpellChecking(true);
    setSpellCheckError(null);
    try {
      const rawText = response.results.map((r) => r.text).join("\n");
      const corrected = await spellCheck(rawText);
      setSpellCorrected(corrected);
    } catch (err) {
      setSpellCheckError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsSpellChecking(false);
    }
  };

  return (
    <div className="min-h-screen w-full">
      <Header />
      <Hero />

      <div className="mx-auto w-full max-w-6xl px-4 pb-20 sm:px-6">
        <TeamSection />

        <section id="demo" className="mt-16 scroll-mt-20 sm:mt-20">
          <SectionHeading
            eyebrow="Demo trực quan"
            eyebrowIcon={<MonitorPlay className="h-3.5 w-3.5" />}
            title="Dùng thử nhận dạng trực tiếp"
            description="Tải lên ảnh dòng chữ viết tay tiếng Việt, chọn mô hình CRNN hoặc SVTR và xem kết quả nhận dạng theo thời gian thực."
          />
          <div className="grid gap-6 lg:grid-cols-[1fr_1.05fr]">
            <div className="flex flex-col gap-6">
              <ConfigPanel models={MODELS} value={config} onChange={setConfig} />
              <UploadCard
                value={image}
                onChange={setImage}
                onRun={onRun}
                isLoading={status === "loading"}
                canRun={Boolean(image)}
              />
            </div>
            <div>
              <ResultsPanel
                status={status}
                response={response}
                error={error}
                showRaw={config.showRaw}
                spellCorrected={spellCorrected}
                isSpellChecking={isSpellChecking}
                spellCheckError={spellCheckError}
                onSpellCheck={handleSpellCheck}
              />
            </div>
          </div>

          {response?.processed_image ? (
            <div className="mt-6">
              <ProcessedImageCard
                processedImage={response.processed_image}
                preprocessed={response.preprocessed}
                originalUrl={image?.url}
              />
            </div>
          ) : null}
        </section>

        <ArchitectureSection />
        <TrainingSection />
        <BenchmarkSection />
      </div>

      <footer className="border-t border-lavender-100/60 bg-cream/60">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-1 px-4 py-10 text-center text-xs text-ink-soft sm:px-6">
          <p>DS107 · Vietnamese Handwriting OCR · 2026</p>
          <p>
            Backend: <code className="font-mono">{gradioUrl}</code>
          </p>
        </div>
      </footer>
    </div>
  );
}
