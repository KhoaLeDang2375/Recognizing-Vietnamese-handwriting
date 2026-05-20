import { motion } from "framer-motion";
import { ArrowRight, Images, Wand2 } from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type Props = {
  processedImage: string | null;
  preprocessed: boolean;
  originalUrl?: string;
};

function Figure({
  src,
  label,
  highlight,
}: {
  src: string;
  label: string;
  highlight?: boolean;
}) {
  return (
    <figure className="space-y-2">
      <div
        className={
          "overflow-hidden rounded-2xl border bg-white " +
          (highlight ? "border-lavender-200" : "border-lavender-100")
        }
      >
        <img
          src={src}
          alt={label}
          className="max-h-72 w-full object-contain"
        />
      </div>
      <figcaption className="text-center text-xs font-medium text-ink-soft">
        {label}
      </figcaption>
    </figure>
  );
}

export function ProcessedImageCard({
  processedImage,
  preprocessed,
  originalUrl,
}: Props) {
  if (!processedImage) return null;

  const showComparison = preprocessed && Boolean(originalUrl);

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
    >
      <Card>
        <CardHeader>
          <div>
            <CardTitle className="flex items-center gap-2">
              <Images className="h-4 w-4 text-lavender-500" />
              <span>Ảnh đưa vào mô hình</span>
            </CardTitle>
            <CardDescription>
              {preprocessed
                ? "Ảnh sau bước tiền xử lý — đây chính là ảnh mà mô hình OCR thực sự nhận được."
                : "Mô hình nhận trực tiếp ảnh gốc (chưa bật tiền xử lý)."}
            </CardDescription>
          </div>
          <Badge tone={preprocessed ? "lavender" : "slate"}>
            <Wand2 className="h-3.5 w-3.5" />
            {preprocessed ? "Đã tiền xử lý" : "Ảnh gốc"}
          </Badge>
        </CardHeader>

        {showComparison ? (
          <div className="grid grid-cols-1 items-center gap-3 sm:grid-cols-[1fr_auto_1fr]">
            <Figure src={originalUrl as string} label="Ảnh gốc tải lên" />
            <ArrowRight className="mx-auto hidden h-5 w-5 shrink-0 text-lavender-400 sm:block" />
            <Figure src={processedImage} label="Sau tiền xử lý" highlight />
          </div>
        ) : (
          <Figure
            src={processedImage}
            label={preprocessed ? "Sau tiền xử lý" : "Ảnh mô hình nhận"}
            highlight
          />
        )}
      </Card>
    </motion.div>
  );
}
