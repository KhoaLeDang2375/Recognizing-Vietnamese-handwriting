import { Cpu, Flame, Settings2, Snowflake } from "lucide-react";
import { SectionHeading } from "@/components/SectionHeading";

const STAGES = [
  {
    badge: "Stage 1",
    title: "Frozen Backbone — Căn chỉnh phân phối",
    icon: <Snowflake className="h-5 w-5" strokeWidth={2} />,
    goal: "Alignment",
    points: [
      "Đóng băng toàn bộ Backbone (freeze_backbone = true), chỉ huấn luyện Neck và Head.",
      "Cho phép dùng learning rate tương đối lớn (5×10⁻⁴ với CRNN, 1×10⁻³ với SVTR) mà không phá hủy đặc trưng pretrained.",
      "Mục tiêu: uốn nắn phần dự đoán làm quen với bảng mã 161 ký tự tiếng Việt.",
    ],
  },
  {
    badge: "Stage 2",
    title: "Full Fine-tuning — Thích ứng sâu",
    icon: <Flame className="h-5 w-5" strokeWidth={2} />,
    goal: "Deep Adaptation",
    points: [
      "Mở khóa toàn bộ mạng (freeze_backbone = false), nạp best_accuracy của Stage 1 làm điểm khởi đầu.",
      "Dùng learning rate rất nhỏ 5×10⁻⁵ cùng warm-up dài và EMA (0.9999, riêng SVTR) để huấn luyện ổn định.",
      "Áp dụng Early Stopping với patience = 10 epoch nhằm chọn đúng mô hình tốt nhất, tránh overfitting.",
    ],
  },
];

const ENV_CHIPS = [
  "Kaggle Notebook",
  "NVIDIA H100",
  "PaddlePaddle 2.6.2",
  "PaddleOCR 2.7",
  "CUDA 12.x",
  "Python 3.10",
  "seed = 2026",
];

type HpRow = { label: string; values: [string, string, string, string]; highlight?: boolean };

const HP_ROWS: HpRow[] = [
  { label: "Optimizer", values: ["Adam", "Adam", "AdamW", "AdamW"] },
  { label: "β₁, β₂", values: ["0.9, 0.999", "0.9, 0.999", "0.9, 0.99", "0.9, 0.99"] },
  {
    label: "Learning rate",
    values: ["5×10⁻⁴", "5×10⁻⁵", "1×10⁻³", "5×10⁻⁵"],
    highlight: true,
  },
  { label: "LR Scheduler", values: ["Cosine", "Cosine", "Cosine", "Cosine"] },
  { label: "Warm-up epochs", values: ["2", "3", "0", "5"] },
  { label: "Số epoch", values: ["30", "50", "15", "35"] },
  {
    label: "Batch train / card",
    values: ["128 (×2 GPU)", "32 (×2 GPU)", "32", "16"],
  },
  {
    label: "Regularization",
    values: ["L2 (1e-4)", "L2 (1e-4)", "WD = 0.1", "WD = 0.1"],
  },
  {
    label: "freeze_backbone",
    values: ["true", "false", "true", "false"],
    highlight: true,
  },
  { label: "AMP / EMA", values: ["— / —", "— / —", "O2 / —", "O2 / 0.9999"] },
  { label: "Aug prob", values: ["0.8", "0.8", "0.5", "0.5"] },
  {
    label: "Image shape",
    values: ["3×32×640", "3×32×640", "3×48×800", "3×48×800"],
  },
];

export function TrainingSection() {
  return (
    <section id="training" className="mt-16 scroll-mt-20 sm:mt-20">
      <SectionHeading
        eyebrow="Chương 3-4 · Huấn luyện"
        eyebrowIcon={<Settings2 className="h-3.5 w-3.5" />}
        title="Chiến lược huấn luyện & Siêu tham số"
        description="Do dữ liệu chữ viết tay tiếng Việt hạn chế, nhóm áp dụng chiến lược 2-Stage Fine-Tuning để chuyển giao tri thức từ trọng số pretrained một cách êm ái, tránh hiện tượng phá hủy đặc trưng (knowledge destruction)."
      />

      {/* 2-stage cards */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {STAGES.map((stage) => (
          <article key={stage.badge} className="card-surface p-6">
            <div className="mb-4 flex items-center gap-3">
              <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-lavender-100 text-lavender-600">
                {stage.icon}
              </span>
              <div>
                <span className="pill">{stage.badge}</span>
                <p className="mt-1 font-mono text-[11px] uppercase tracking-wide text-lavender-500">
                  Mục tiêu: {stage.goal}
                </p>
              </div>
            </div>
            <h3 className="mb-3 font-display text-base font-semibold text-ink">
              {stage.title}
            </h3>
            <ul className="space-y-2">
              {stage.points.map((point) => (
                <li
                  key={point}
                  className="flex gap-2 text-sm leading-relaxed text-ink-muted"
                >
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-lavender-400" />
                  <span>{point}</span>
                </li>
              ))}
            </ul>
          </article>
        ))}
      </div>

      {/* Hyperparameter dashboard */}
      <div className="card-surface mt-6 p-6 sm:p-8">
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="font-display text-base font-semibold text-ink">
              Bảng cấu hình siêu tham số
            </h3>
            <p className="text-sm text-ink-muted">
              Bốn cấu hình huấn luyện trích từ Bảng 4.4 của báo cáo.
            </p>
          </div>
          <span className="pill">
            <Cpu className="h-3.5 w-3.5" />
            4 cấu hình
          </span>
        </div>

        <div className="overflow-x-auto rounded-xl border border-lavender-100">
          <table className="w-full min-w-[640px] border-collapse text-sm">
            <thead>
              <tr className="bg-lavender-50/80 text-ink">
                <th className="px-4 py-2.5 text-left font-display text-xs font-semibold">
                  Siêu tham số
                </th>
                <th
                  colSpan={2}
                  className="border-l border-lavender-100 px-4 py-2.5 text-center font-display text-xs font-semibold text-lavender-700"
                >
                  CRNN
                </th>
                <th
                  colSpan={2}
                  className="border-l border-lavender-100 px-4 py-2.5 text-center font-display text-xs font-semibold text-lavender-700"
                >
                  SVTR
                </th>
              </tr>
              <tr className="bg-lavender-50/50 text-ink-soft">
                <th className="px-4 py-1.5 text-left text-[11px] font-medium" />
                <th className="border-l border-lavender-100 px-4 py-1.5 text-center text-[11px] font-medium">
                  Stage 1
                </th>
                <th className="px-4 py-1.5 text-center text-[11px] font-medium">
                  Stage 2
                </th>
                <th className="border-l border-lavender-100 px-4 py-1.5 text-center text-[11px] font-medium">
                  Stage 1
                </th>
                <th className="px-4 py-1.5 text-center text-[11px] font-medium">
                  Stage 2
                </th>
              </tr>
            </thead>
            <tbody>
              {HP_ROWS.map((row) => (
                <tr
                  key={row.label}
                  className={
                    row.highlight
                      ? "bg-lavender-50/40"
                      : "odd:bg-white/40 even:bg-cream-50/40"
                  }
                >
                  <td className="px-4 py-2.5 font-medium text-ink">
                    {row.label}
                  </td>
                  {row.values.map((value, i) => (
                    <td
                      key={i}
                      className={`px-4 py-2.5 text-center font-mono text-xs text-ink-muted ${
                        i === 0 || i === 2
                          ? "border-l border-lavender-100"
                          : ""
                      }`}
                    >
                      {value}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-5 flex flex-wrap gap-2">
          {ENV_CHIPS.map((chip) => (
            <span
              key={chip}
              className="rounded-lg border border-lavender-100 bg-white/60 px-2.5 py-1 font-mono text-[11px] text-ink-soft"
            >
              {chip}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}
