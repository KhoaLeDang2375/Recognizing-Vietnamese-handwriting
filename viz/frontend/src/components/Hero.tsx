import { ArrowRight, BookOpen, PenLine, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";

const STATS = [
  { value: "UIT-HWDB", label: "Bộ dữ liệu chữ viết tay" },
  { value: "7,273", label: "Mẫu dòng văn bản" },
  { value: "161", label: "Lớp ký tự (gồm blank)" },
  { value: "2 mô hình", label: "CRNN · SVTR baseline" },
];

export function Hero() {
  return (
    <section
      id="hero"
      className="relative overflow-hidden border-b border-lavender-100/60"
    >
      <div className="mx-auto w-full max-w-6xl px-4 py-16 text-center sm:px-6 sm:py-24">
        <span className="pill mx-auto">
          <Sparkles className="h-3.5 w-3.5" />
          Đồ án DS107 · Nhận dạng chữ viết tay tiếng Việt
        </span>

        <h1 className="mx-auto mt-6 max-w-3xl font-display text-4xl font-bold leading-tight tracking-tight text-ink sm:text-6xl">
          Vietnamese{" "}
          <span className="bg-gradient-to-r from-lavender-500 to-lavender-700 bg-clip-text text-transparent">
            Handwriting OCR
          </span>
        </h1>

        <p className="mx-auto mt-5 max-w-2xl text-base leading-relaxed text-ink-muted sm:text-lg">
          Hệ thống nhận dạng chữ viết tay tiếng Việt hiệu năng cao, xây dựng
          trên bộ dữ liệu UIT-HWDB và engine PaddleOCR, so sánh hai kiến trúc
          CRNN và SVTR với chiến lược 2-Stage Fine-Tuning.
        </p>

        <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
          <Button asChild size="lg">
            <a href="#demo">
              <PenLine className="h-4 w-4" />
              Dùng thử Demo
              <ArrowRight className="h-4 w-4" />
            </a>
          </Button>
          <Button asChild size="lg" variant="secondary">
            <a href="#architecture">
              <BookOpen className="h-4 w-4" />
              Đọc báo cáo
            </a>
          </Button>
        </div>

        <dl className="mx-auto mt-14 grid max-w-4xl grid-cols-2 gap-3 sm:grid-cols-4 sm:gap-4">
          {STATS.map((stat) => (
            <div
              key={stat.label}
              className="card-surface flex flex-col items-center gap-1 px-3 py-5"
            >
              <dt className="font-display text-xl font-bold text-lavender-600 sm:text-2xl">
                {stat.value}
              </dt>
              <dd className="text-xs text-ink-soft">{stat.label}</dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}
