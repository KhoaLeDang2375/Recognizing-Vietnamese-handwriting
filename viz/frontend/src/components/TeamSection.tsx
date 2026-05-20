import { Users } from "lucide-react";
import { SectionHeading } from "@/components/SectionHeading";

type Member = {
  name: string;
  studentId: string;
  role: string;
  initials: string;
};

const MEMBERS: Member[] = [
  {
    name: "Lê Đăng Khoa",
    studentId: "23520740",
    role: "Kiến trúc & Huấn luyện SVTR",
    initials: "LK",
  },
  {
    name: "Lại Thị Thu Hương",
    studentId: "23520585",
    role: "Dữ liệu UIT-HWDB & Tiền xử lý",
    initials: "LH",
  },
  {
    name: "Phan Trần Văn Khang",
    studentId: "23520708",
    role: "Demo trực quan & Tích hợp hệ thống",
    initials: "PK",
  },
  {
    name: "Trần Thị Kim Anh",
    studentId: "23520079",
    role: "Huấn luyện CRNN & Đánh giá kết quả",
    initials: "TA",
  },
];

export function TeamSection() {
  return (
    <section id="team" className="mt-16 scroll-mt-20 sm:mt-20">
      <SectionHeading
        eyebrow="Thành viên nhóm"
        eyebrowIcon={<Users className="h-3.5 w-3.5" />}
        title="Nhóm thực hiện đồ án"
        description="Đồ án môn DS107 do nhóm bốn thành viên thực hiện, phân chia theo các mảng công việc chính của dự án nhận dạng chữ viết tay tiếng Việt."
      />

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {MEMBERS.map((member) => (
          <article
            key={member.studentId}
            className="card-surface flex flex-col items-center gap-3 p-6 text-center transition-transform hover:-translate-y-1"
          >
            <span
              aria-hidden
              className="flex h-16 w-16 items-center justify-center rounded-full bg-gradient-to-br from-lavender-200 to-lavender-400 font-display text-xl font-bold text-white shadow-soft"
            >
              {member.initials}
            </span>
            <div>
              <h3 className="font-display text-base font-semibold tracking-tight text-ink">
                {member.name}
              </h3>
              <p className="mt-0.5 font-mono text-xs text-ink-soft">
                {member.studentId}
              </p>
            </div>
            <p className="mt-auto rounded-lg bg-lavender-50/80 px-3 py-2 text-xs font-medium leading-relaxed text-lavender-700">
              {member.role}
            </p>
          </article>
        ))}
      </div>
    </section>
  );
}
