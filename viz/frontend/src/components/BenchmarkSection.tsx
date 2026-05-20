import { BarChart3, Trophy } from "lucide-react";
import { SectionHeading } from "@/components/SectionHeading";

type Metric = {
  key: string;
  note: string;
  better: "high" | "low";
  crnn: number;
  svtr: number;
  crnnLabel?: string;
  svtrLabel?: string;
};

const METRICS: Metric[] = [
  {
    key: "Word Accuracy",
    note: "Tỷ lệ mẫu dự đoán đúng hoàn toàn (Exact Match).",
    better: "high",
    crnn: 0,
    svtr: 11.44,
    crnnLabel: "≈ 0.00%",
    svtrLabel: "11.44%",
  },
  {
    key: "CER",
    note: "Character Error Rate — tỷ lệ lỗi ở cấp ký tự.",
    better: "low",
    crnn: 37.4,
    svtr: 9.5,
    crnnLabel: "37.40%",
    svtrLabel: "9.50%",
  },
  {
    key: "WER",
    note: "Word Error Rate — tỷ lệ lỗi ở cấp từ.",
    better: "low",
    crnn: 86.8,
    svtr: 29.1,
    crnnLabel: "86.80%",
    svtrLabel: "29.10%",
  },
  {
    key: "Confidence",
    note: "Độ tin cậy trung bình của dự đoán.",
    better: "high",
    crnn: 61.3,
    svtr: 92.7,
    crnnLabel: "61.30%",
    svtrLabel: "92.70%",
  },
];

const TABLE_ROWS = [
  { metric: "Word Accuracy", crnn: "≈ 0.00%", svtr: "11.44%", better: "high" },
  { metric: "CER", crnn: "37.40%", svtr: "9.50%", better: "low" },
  { metric: "WER", crnn: "86.80%", svtr: "29.10%", better: "low" },
  { metric: "NED", crnn: "0.374", svtr: "0.095", better: "low" },
  { metric: "Confidence", crnn: "61.30%", svtr: "92.70%", better: "high" },
];

function Bar({
  label,
  value,
  display,
  color,
}: {
  label: string;
  value: number;
  display: string;
  color: string;
}) {
  return (
    <div>
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium text-ink-muted">{label}</span>
        <span className="font-mono font-semibold text-ink">{display}</span>
      </div>
      <div className="mt-1 h-2.5 overflow-hidden rounded-full bg-lavender-50">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${Math.max(value, 1.5)}%` }}
        />
      </div>
    </div>
  );
}

export function BenchmarkSection() {
  return (
    <section id="benchmarks" className="mt-16 scroll-mt-20 sm:mt-20">
      <SectionHeading
        eyebrow="Chương 6 · Kết quả & Thảo luận"
        eyebrowIcon={<BarChart3 className="h-3.5 w-3.5" />}
        title="Performance Benchmarks"
        description="Kết quả đánh giá trên tập kiểm thử UIT-HWDB-line cho thấy SVTR vượt trội trên toàn bộ các chỉ số so với baseline CRNN."
      />

      {/* Comparison table */}
      <div className="card-surface mb-6 overflow-x-auto p-0">
        <table className="w-full min-w-[560px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-lavender-100 bg-lavender-50/60 text-ink">
              <th className="px-5 py-3 text-left font-display text-xs font-semibold">
                Chỉ số
              </th>
              <th className="px-5 py-3 text-center font-display text-xs font-semibold">
                CRNN
              </th>
              <th className="px-5 py-3 text-center font-display text-xs font-semibold text-lavender-700">
                SVTR
              </th>
              <th className="px-5 py-3 text-center font-display text-xs font-semibold">
                Mô hình tốt hơn
              </th>
            </tr>
          </thead>
          <tbody>
            {TABLE_ROWS.map((row) => (
              <tr
                key={row.metric}
                className="border-b border-lavender-100/60 last:border-0"
              >
                <td className="px-5 py-3 font-medium text-ink">
                  {row.metric}
                  <span className="ml-2 font-mono text-[10px] uppercase text-ink-soft">
                    {row.better === "high" ? "↑ tốt" : "↓ tốt"}
                  </span>
                </td>
                <td className="px-5 py-3 text-center font-mono text-xs text-ink-muted">
                  {row.crnn}
                </td>
                <td className="px-5 py-3 text-center font-mono text-xs font-semibold text-lavender-700">
                  {row.svtr}
                </td>
                <td className="px-5 py-3 text-center">
                  <span className="inline-flex items-center gap-1 rounded-full bg-lavender-100 px-2.5 py-1 text-[11px] font-semibold text-lavender-700">
                    <Trophy className="h-3 w-3" />
                    SVTR
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Metric bar cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {METRICS.map((m) => (
          <div key={m.key} className="card-surface p-5">
            <div className="flex items-start justify-between gap-3">
              <h3 className="font-display text-sm font-semibold text-ink">
                {m.key}
              </h3>
              <span className="shrink-0 rounded-md bg-cream-200/70 px-2 py-0.5 text-[10px] font-medium text-ink-soft">
                {m.better === "high" ? "Cao hơn tốt hơn" : "Thấp hơn tốt hơn"}
              </span>
            </div>
            <p className="mt-1 text-xs text-ink-soft">{m.note}</p>
            <div className="mt-4 space-y-3">
              <Bar
                label="CRNN"
                value={m.crnn}
                display={m.crnnLabel ?? `${m.crnn}%`}
                color="bg-amber-400"
              />
              <Bar
                label="SVTR"
                value={m.svtr}
                display={m.svtrLabel ?? `${m.svtr}%`}
                color="bg-lavender-500"
              />
            </div>
          </div>
        ))}
      </div>

      {/* Verdict callout */}
      <div className="mt-6 rounded-2xl border border-lavender-200/70 bg-gradient-to-br from-lavender-50 to-cream-100 p-6 sm:p-8">
        <div className="flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-lavender-500 text-white shadow-glow">
            <Trophy className="h-5 w-5" />
          </span>
          <h3 className="font-display text-base font-semibold text-ink">
            Kết luận: SVTR là lựa chọn tối ưu
          </h3>
        </div>
        <p className="mt-3 text-sm leading-relaxed text-ink-muted">
          SVTR thể hiện ưu thế trên mọi chỉ số đánh giá: khoảng cách chỉnh sửa
          chuẩn hóa (NED 0.095) thấp hơn gần 4 lần so với CRNN (0.374), độ tin
          cậy đạt 92.7% so với 61.3%. Về quá trình huấn luyện, SVTR hội tụ Loss
          xuống mức 15–20 trong khi CRNN dừng ở khoảng 90. Việc kết hợp kiến
          trúc Transformer thuần thị giác với chiến lược 2-Stage Fine-Tuning
          giúp hệ thống đạt hiệu năng nhận diện chữ viết tay tiếng Việt mạnh
          mẽ nhất trên tập dữ liệu thử nghiệm.
        </p>
      </div>
    </section>
  );
}
