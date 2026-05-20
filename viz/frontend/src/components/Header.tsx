import { PenLine } from "lucide-react";

const NAV_LINKS = [
  { href: "#team", label: "Nhóm" },
  { href: "#demo", label: "Demo" },
  { href: "#architecture", label: "Phương pháp" },
  { href: "#training", label: "Huấn luyện" },
  { href: "#benchmarks", label: "Kết quả" },
];

export function Header() {
  return (
    <header className="sticky top-0 z-40 border-b border-lavender-100/60 bg-cream/80 backdrop-blur-md">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <a href="#hero" className="flex items-center gap-2.5">
          <span
            aria-hidden
            className="flex h-9 w-9 items-center justify-center rounded-xl bg-lavender-100 text-lavender-600 shadow-soft"
          >
            <PenLine className="h-4 w-4" strokeWidth={2} />
          </span>
          <span className="font-display text-sm font-bold tracking-tight text-ink sm:text-base">
            Vietnamese Handwriting OCR
          </span>
        </a>

        <nav className="hidden items-center gap-1 md:flex">
          {NAV_LINKS.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="rounded-lg px-3 py-1.5 text-sm font-medium text-ink-muted transition-colors hover:bg-lavender-50/80 hover:text-ink"
            >
              {link.label}
            </a>
          ))}
        </nav>
      </div>
    </header>
  );
}
