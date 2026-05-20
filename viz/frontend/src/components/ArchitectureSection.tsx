import { useEffect, useState, type ReactNode } from "react";
import {
  Boxes,
  Layers,
  Network,
  ScanLine,
  Type,
  Workflow,
  X,
  ZoomIn,
} from "lucide-react";
import { SectionHeading } from "@/components/SectionHeading";

type FigureData = {
  src: string;
  alt: string;
  caption: string;
};

const CRNN_FIGURE: FigureData = {
  src: "/assets/images/architecture/CRNN.png",
  alt: "Sơ đồ kiến trúc CRNN: ảnh đầu vào kích thước 32×640 đi qua Backbone ResNet-34 tạo Feature Sequence, qua Neck BiLSTM hidden 256 gồm luồng LSTM xuôi và ngược, tới Head CTC với 161 ký tự; phía trên minh họa hai stage huấn luyện Warm-up (Backbone đóng băng) và Fine-tuning (toàn bộ trainable).",
  caption:
    "Hình 3.1. Kiến trúc tổng quan và chiến lược 2-Stage Fine-Tuning của mô hình CRNN.",
};

const SVTR_FIGURE: FigureData = {
  src: "/assets/images/architecture/SVTR.png",
  alt: "Sơ đồ kiến trúc SVTRNet: ảnh đầu vào 48×800 qua Patch Embedding dạng Conv, Backbone SVTRNet ba stage (Dim 128/256/384, 4/8/12 heads) kết hợp Local Mixer và Global Mixer cùng các bước Patch Merging, Neck SequenceEncoder thực hiện Reshape & Flatten, Head CTC với lớp Fully Connected 161 lớp cộng blank token.",
  caption:
    "Hình 3.2. Kiến trúc phân cấp SVTRNet với 3 stage Patch Merging và cơ chế Local/Global Mixing.",
};

const PIPELINE_STEPS: {
  step: string;
  title: string;
  subtitle: string;
  desc: string;
  icon: ReactNode;
}[] = [
  {
    step: "01",
    title: "Tiền xử lý",
    subtitle: "Preprocessing",
    desc: "Ảnh dòng văn bản được chuẩn hóa về độ phân giải cố định (32×640 cho CRNN, 48×800 cho SVTR), giải mã sang định dạng BGR ba kênh và chuẩn hóa cường độ pixel.",
    icon: <ScanLine className="h-5 w-5" strokeWidth={2} />,
  },
  {
    step: "02",
    title: "Trích xuất đặc trưng",
    subtitle: "Backbone",
    desc: "CRNN dùng ResNet-34 với các khối tích chập có kết nối tắt; SVTR dùng SVTRNet với các khối Mixing kết hợp Local và Global Attention. Đầu ra là một feature sequence theo trục ngang.",
    icon: <Boxes className="h-5 w-5" strokeWidth={2} />,
  },
  {
    step: "03",
    title: "Mô hình hóa chuỗi",
    subtitle: "Neck",
    desc: "CRNN dùng cụm BiLSTM hai chiều (hidden 256) khai thác ngữ cảnh hai hướng; SVTR dùng SequenceEncoder dạng Reshape vì backbone đã tự mô hình hóa ngữ cảnh.",
    icon: <Workflow className="h-5 w-5" strokeWidth={2} />,
  },
  {
    step: "04",
    title: "Giải mã",
    subtitle: "Head — CTC",
    desc: "Cả hai mô hình dùng CTC Head, huấn luyện bằng CTCLoss và hậu xử lý CTCLabelDecode, ánh xạ mỗi bước thời gian sang phân phối xác suất ký tự cộng blank.",
    icon: <Type className="h-5 w-5" strokeWidth={2} />,
  },
];

function Lightbox({
  figure,
  onClose,
}: {
  figure: FigureData;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [onClose]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={figure.caption}
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/80 p-4 backdrop-blur-sm animate-fade-in"
    >
      <button
        type="button"
        onClick={onClose}
        aria-label="Đóng ảnh phóng to"
        className="absolute right-4 top-4 flex h-10 w-10 items-center justify-center rounded-full bg-white/90 text-ink shadow-soft transition hover:bg-white"
      >
        <X className="h-5 w-5" />
      </button>
      <div
        className="max-h-[92vh] max-w-[96vw] overflow-auto rounded-xl bg-white p-2 shadow-glow"
        onClick={(e) => e.stopPropagation()}
      >
        <img src={figure.src} alt={figure.alt} className="block max-w-none" />
      </div>
    </div>
  );
}

function ArchitectureFigure({
  figure,
  onZoom,
}: {
  figure: FigureData;
  onZoom: () => void;
}) {
  return (
    <figure className="mb-6">
      <div className="overflow-x-auto rounded-xl border border-lavender-100 bg-white p-3 shadow-soft">
        <button
          type="button"
          onClick={onZoom}
          aria-label={`Phóng to: ${figure.caption}`}
          className="group relative block w-full cursor-zoom-in"
        >
          <img
            src={figure.src}
            alt={figure.alt}
            loading="lazy"
            className="mx-auto block h-auto w-full min-w-[680px] rounded-lg sm:min-w-0"
          />
          <span className="pointer-events-none absolute right-3 top-3 flex items-center gap-1.5 rounded-full bg-ink/75 px-2.5 py-1 text-xs font-medium text-white opacity-80 transition sm:opacity-0 sm:group-hover:opacity-100">
            <ZoomIn className="h-3.5 w-3.5" />
            Phóng to
          </span>
        </button>
      </div>
      <figcaption className="mx-auto mt-3 max-w-3xl text-center text-xs italic leading-relaxed text-ink-soft">
        {figure.caption}
      </figcaption>
    </figure>
  );
}

function SubsectionHeading({
  index,
  title,
  icon,
}: {
  index: string;
  title: string;
  icon: ReactNode;
}) {
  return (
    <div className="mb-5 flex items-center gap-3">
      <span
        aria-hidden
        className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-lavender-100 text-lavender-600"
      >
        {icon}
      </span>
      <h3 className="font-display text-lg font-semibold tracking-tight text-ink">
        <span className="text-lavender-600">{index}</span> · {title}
      </h3>
    </div>
  );
}

export function ArchitectureSection() {
  const [zoom, setZoom] = useState<FigureData | null>(null);

  return (
    <section id="architecture" className="mt-16 scroll-mt-20 sm:mt-20">
      <SectionHeading
        eyebrow="Chương 3 · Phương pháp đề xuất"
        eyebrowIcon={<Layers className="h-3.5 w-3.5" />}
        title="Phương pháp & Kiến trúc mô hình"
        description="Hệ thống được xây dựng trên hai kiến trúc tiêu biểu — CRNN và SVTR — cùng dùng chung một pipeline xử lý thống nhất và hàm mất mát CTC để đảm bảo so sánh khách quan."
      />

      {/* 3.1 — Pipeline overview */}
      <div className="mb-12">
        <h3 className="font-display text-lg font-semibold tracking-tight text-ink">
          <span className="text-lavender-600">3.1</span> · Tổng quan pipeline
          hệ thống
        </h3>
        <p className="mb-5 mt-2 max-w-3xl text-sm leading-relaxed text-ink-muted">
          Luồng xử lý dữ liệu được thiết kế thống nhất cho cả hai kiến trúc, đi
          từ khâu tiếp nhận ảnh đầu vào đến khi xuất ra chuỗi văn bản cuối cùng,
          gồm bốn thành phần chính. Việc thống nhất CTC làm hàm mất mát giúp mọi
          khác biệt về hiệu năng phản ánh đúng năng lực của bản thân kiến trúc
          trích xuất đặc trưng và mô hình hóa chuỗi.
        </p>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {PIPELINE_STEPS.map((s) => (
            <div key={s.step} className="card-surface flex flex-col gap-3 p-5">
              <div className="flex items-center justify-between">
                <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-lavender-100 text-lavender-600">
                  {s.icon}
                </span>
                <span className="font-display text-sm font-bold text-lavender-200">
                  {s.step}
                </span>
              </div>
              <div>
                <h4 className="font-display text-sm font-semibold text-ink">
                  {s.title}
                </h4>
                <p className="font-mono text-[11px] uppercase tracking-wide text-lavender-500">
                  {s.subtitle}
                </p>
              </div>
              <p className="text-xs leading-relaxed text-ink-muted">{s.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 3.2 — Baselines */}
      <div className="mb-6">
        <h3 className="font-display text-lg font-semibold tracking-tight text-ink">
          <span className="text-lavender-600">3.2</span> · Cấu trúc mạng cơ sở
          (Baselines)
        </h3>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-ink-muted">
          Để có cơ sở đánh giá hiệu năng và lựa chọn kiến trúc tối ưu cho bài
          toán nhận dạng chữ viết tay tiếng Việt với các đặc thù riêng, nhóm
          tiến hành triển khai và so sánh hai cấu trúc mạng cơ sở (baselines)
          mang tính đại diện cho hai trường phái: mạng nơ-ron tích chập kết hợp
          hồi quy tuần tự (CRNN) và mạng Transformer thuần thị giác (SVTR).
        </p>
      </div>

      {/* 3.2.1 — CRNN */}
      <article className="card-surface mb-6 p-6 sm:p-8">
        <SubsectionHeading
          index="3.2.1"
          title="CRNN Pipeline"
          icon={<Network className="h-5 w-5" strokeWidth={2} />}
        />
        <p className="mb-6 max-w-3xl text-sm leading-relaxed text-ink-muted">
          Kiến trúc CRNN kết hợp sức mạnh của mạng nơ-ron tích chập (CNN) trong
          trích xuất đặc trưng thị giác với mạng nơ-ron hồi quy (RNN) trong mô
          hình hóa chuỗi. Nhóm sử dụng Backbone ResNet-34 thay cho MobileNet
          trong cấu hình gốc và nâng hidden size của BiLSTM lên 256 để khớp với
          feature map của ResNet-34.
        </p>

        <ArchitectureFigure figure={CRNN_FIGURE} onZoom={() => setZoom(CRNN_FIGURE)} />

        <div className="space-y-4 text-sm leading-relaxed text-ink-muted">
          <p>
            Như minh họa trong Hình 3.1, kiến trúc CRNN mà nhóm chúng tôi triển
            khai được tổ chức thành ba thành phần kế tiếp nhau, mỗi thành phần
            đảm nhận một vai trò chuyên biệt trong quá trình chuyển một ảnh dòng
            chữ viết tay thành chuỗi văn bản. Backbone là mạng ResNet-34 giữ vai
            trò trích xuất đặc trưng thị giác, biến đổi các điểm ảnh thô thành
            biểu diễn giàu ngữ nghĩa về hình dạng nét chữ. Neck được xây dựng từ
            các lớp BiLSTM, đảm nhận việc mô hình hóa ngữ cảnh tuần tự giữa các
            vùng đặc trưng dọc theo trục thời gian. Head dựa trên cơ chế CTC
            thực hiện giải mã chuỗi, ánh xạ biểu diễn ẩn thành phân phối xác
            suất trên tập ký tự. Cách phân tách rạch ròi này giúp nhóm thuận
            tiện trong việc tinh chỉnh từng khối cũng như truy vết nguồn gốc sai
            số khi thực nghiệm.
          </p>
          <p>
            Về luồng dữ liệu, ảnh đầu vào trước hết được chuẩn hóa về kích thước
            cố định 32×640, trong đó chiều cao được ép về 32 pixel còn chiều rộng
            giữ ở 640 pixel nhằm bảo toàn thông tin trải dài theo phương ngang
            của dòng chữ. Ảnh sau chuẩn hóa được đưa qua Backbone ResNet-34 để
            sinh ra các feature map cô đọng đặc trưng không gian của nét bút.
            Nhờ cơ chế residual connection, ResNet-34 có thể huấn luyện ở độ sâu
            lớn mà tránh được hiện tượng suy biến đạo hàm, qua đó học được đặc
            trưng từ mức thấp như cạnh và nét cho tới mức cao là cấu trúc ký tự.
            Feature map thu được sau đó được reshape thành một feature sequence —
            chuỗi các vector đặc trưng sắp theo trục thời gian, mỗi vector tương
            ứng với một lát cắt dọc hẹp của ảnh theo hướng từ trái sang phải.
          </p>
          <p>
            Chuỗi đặc trưng này tiếp tục được xử lý bởi khối BiLSTM với hidden
            size 256 cho mỗi chiều. Điểm cốt lõi của BiLSTM nằm ở hai luồng hồi
            quy chạy song song: một luồng LSTM xuôi đọc chuỗi từ trái sang phải
            và một luồng LSTM ngược đọc theo hướng đối nghịch. Việc kết hợp hai
            chiều cho phép mô hình khai thác ngữ cảnh ở cả hai phía của mỗi vị
            trí thay vì chỉ dựa vào thông tin một chiều. Đặc điểm này đặc biệt
            quan trọng với tiếng Việt, bởi dấu thanh và dấu phụ thường xuất hiện
            ở những vị trí dễ gây nhập nhằng (ambiguity) nếu mô hình chỉ quan sát
            theo một hướng. Nhờ ngữ cảnh hai chiều, mô hình có thêm cơ sở để
            phân biệt các ký tự có hình dạng gần giống nhau nhưng khác nhau ở
            dấu.
          </p>
          <p>
            Ở giai đoạn cuối, Head thực hiện một phép chiếu tuyến tính đưa mỗi
            vector ẩn của BiLSTM lên không gian 161 lớp, gồm 160 ký tự thuộc bảng
            chữ tiếng Việt và một blank token. Cơ chế CTC alignment cho phép mô
            hình học ánh xạ trực tiếp từ ảnh sang chuỗi văn bản mà không cần phân
            đoạn ký tự rõ ràng ở khâu gán nhãn. Đây là lợi thế thiết yếu với chữ
            viết tay, nơi hiện tượng các ký tự dính liền nhau (character-touching)
            khiến việc xác định ranh giới ký tự trở nên rất khó khăn. Trong quá
            trình suy luận, thuật toán giải mã greedy gộp các ký tự lặp liên tiếp
            và loại bỏ blank token để thu được chuỗi văn bản cuối cùng.
          </p>
          <p>
            Phần phía trên của Hình 3.1 mô tả chiến lược huấn luyện 2-Stage
            Fine-Tuning mà nhóm áp dụng cho CRNN. Ở Stage 1 (Warm-up), toàn bộ
            Backbone được đóng băng và chỉ Neck cùng Head được cập nhật, nhằm để
            hai khối này căn chỉnh với phân phối ký tự tiếng Việt trước. Cách làm
            này tránh cho gradient lớn ở những bước đầu phá hỏng các trọng số
            pretrained vốn đã mã hóa nhiều đặc trưng thị giác hữu ích. Sang
            Stage 2 (Fine-tuning), toàn bộ mạng được mở khóa và huấn luyện đồng
            thời với một learning rate nhỏ, cho phép Backbone thích ứng sâu với
            đặc thù nét chữ viết tay tiếng Việt mà vẫn giữ được sự ổn định. Nhờ
            tách bạch hai giai đoạn, nhóm cân bằng được giữa tốc độ hội tụ và
            chất lượng đặc trưng cuối cùng.
          </p>
        </div>
      </article>

      {/* 3.2.2 — SVTR */}
      <article className="card-surface p-6 sm:p-8">
        <SubsectionHeading
          index="3.2.2"
          title="SVTR Pipeline"
          icon={<Layers className="h-5 w-5" strokeWidth={2} />}
        />
        <p className="mb-6 max-w-3xl text-sm leading-relaxed text-ink-muted">
          SVTR (Scene Text Recognition with a Single Visual Model) là kiến trúc
          hiện đại loại bỏ hoàn toàn lớp RNN, thay vào đó dùng cơ chế
          self-attention trên các patch của ảnh. Nhóm cấu trúc pipeline SVTR
          theo dạng phân cấp với độ phân giải đặc trưng giảm dần qua từng stage.
        </p>

        <ArchitectureFigure figure={SVTR_FIGURE} onZoom={() => setZoom(SVTR_FIGURE)} />

        <div className="space-y-4 text-sm leading-relaxed text-ink-muted">
          <p>
            Khác với CRNN, kiến trúc SVTR mà nhóm chúng tôi triển khai loại bỏ
            hoàn toàn các lớp RNN và thay thế bằng những khối trộn đặc trưng dựa
            trên Transformer (Transformer-based mixing blocks). Sự thay đổi này
            xuất phát từ hai hạn chế cố hữu của RNN: khả năng mô hình hóa phụ
            thuộc xa (long-term dependency) suy giảm trên chuỗi dài, và bản chất
            tính toán tuần tự khiến quá trình huấn luyện lẫn suy luận khó song
            song hóa. Bằng cơ chế self-attention, SVTR cho phép mọi vị trí trong
            chuỗi tương tác trực tiếp với nhau, đồng thời khai thác tốt khả năng
            tính toán song song của phần cứng hiện đại. Đây chính là lý do nhóm
            lựa chọn SVTR làm baseline đại diện cho trường phái Transformer
            thuần thị giác.
          </p>
          <p>
            Luồng xử lý của SVTR bắt đầu từ ảnh đầu vào kích thước 48×800, được
            đưa qua khối Patch Embedding dạng Conv-based. Tại đây, ảnh được chia
            thành nhiều patch nhỏ, mỗi patch chỉ chứa một phần rất nhỏ của nét
            chữ và được ánh xạ thành một vector đặc trưng đóng vai trò token đầu
            vào, tương ứng với feature map 48×128 ở giai đoạn này. Ý tưởng token
            hóa ảnh này tương tự mô hình Vision Transformer (ViT), song cách cài
            đặt bằng các lớp tích chập được tinh chỉnh riêng cho bài toán nhận
            dạng văn bản nhằm bảo toàn cấu trúc liền mạch của dòng chữ. Cách
            biểu diễn đó tạo tiền đề để các khối Mixer phía sau khai thác quan
            hệ giữa các patch.
          </p>
          <p>
            Backbone SVTRNet được tổ chức phân cấp qua ba stage với năng lực
            biểu diễn tăng dần. Stage 1 sử dụng dimension 128 với 4 head và 3
            layer; Stage 2 nâng lên dimension 256 với 8 head và 6 layer; Stage 3
            đạt dimension 384 với 12 head và 9 layer. Xen giữa các stage là
            những khối Patch Merging dạng Conv-based, có nhiệm vụ giảm độ phân
            giải không gian của feature map đồng thời gia tăng chiều sâu kênh đặc
            trưng. Thiết kế này mô phỏng cấu trúc phân cấp (hierarchical
            structure) của Swin Transformer, giúp mô hình vừa nắm bắt chi tiết
            cục bộ ở các stage đầu, vừa tổng hợp ngữ nghĩa trừu tượng ở các stage
            sau.
          </p>
          <p>
            Sức mạnh của SVTR nằm ở sự phối hợp giữa hai loại khối trộn đặc trưng
            là Local Mixer và Global Mixer. Local Mixer hoạt động trên một
            sliding window kích thước 7×11, tập trung trích xuất đặc trưng cục bộ
            của nét chữ, dấu phụ và dấu thanh — yếu tố mang tính quyết định với
            tiếng Việt khi những dấu nhỏ cần được nắm bắt thật chính xác. Trong
            khi đó, Global Mixer dựa trên Multi-head Attention, mô hình hóa quan
            hệ ngữ cảnh toàn cục giữa toàn bộ ký tự trong một dòng. Như thể hiện
            trong Hình 3.2, Stage 1 chỉ dùng Local Mixer để học các đặc trưng
            cấp thấp, còn Stage 2 và Stage 3 kết hợp cả Local Mixer lẫn Global
            Mixer nhằm khai thác đồng thời quan hệ không gian chi tiết và ngữ
            nghĩa cấp cao.
          </p>
          <p>
            Sau khi đi qua Backbone, feature map cuối cùng được khối Neck
            SequenceEncoder xử lý bằng thao tác Reshape & Flatten, chuyển thành
            một visual ID sequence — chuỗi đặc trưng một chiều theo trục thời
            gian sẵn sàng cho khâu giải mã. Head của SVTR là một lớp Fully
            Connected ánh xạ mỗi bước thời gian lên 161 lớp gồm các ký tự tiếng
            Việt và blank token, sau đó áp dụng CTC decoding để sinh ra chuỗi
            văn bản. Nhóm chủ ý giữ nguyên cấu hình 161 classes cộng blank giống
            hệt CRNN, nhằm bảo đảm phép so sánh giữa hai mô hình là công bằng và
            chỉ phản ánh khác biệt do kiến trúc Backbone và Neck mang lại.
          </p>
          <p>
            Tổng hòa các đặc điểm trên, SVTR thể hiện nhiều lợi thế cho bài toán
            nhận dạng chữ viết tay (HTR) tiếng Việt. Khả năng trích xuất đặc
            trưng đa tỉ lệ (multi-scale) kết hợp với cơ chế attention giúp mô
            hình nắm bắt tốt các trường hợp dấu phụ chồng lớp như â, ơ, ư đi kèm
            dấu thanh — vốn là thách thức kinh điển của tiếng Việt. Đồng thời, do
            không tồn tại phụ thuộc tuần tự như trong LSTM, SVTR có thể song song
            hóa quá trình tính toán và mang lại tốc độ suy luận nhanh hơn. Những
            ưu điểm này khiến SVTR trở thành đối trọng giàu tiềm năng so với
            baseline CRNN trong các thực nghiệm của nhóm.
          </p>
        </div>
      </article>

      {zoom && <Lightbox figure={zoom} onClose={() => setZoom(null)} />}
    </section>
  );
}
