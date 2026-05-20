import type { ReactNode } from "react";

export function SectionHeading({
  eyebrow,
  eyebrowIcon,
  title,
  description,
}: {
  eyebrow: string;
  eyebrowIcon?: ReactNode;
  title: string;
  description?: string;
}) {
  return (
    <div className="mb-8">
      <span className="pill">
        {eyebrowIcon}
        {eyebrow}
      </span>
      <h2 className="mt-3 font-display text-2xl font-bold tracking-tight text-ink sm:text-3xl">
        {title}
      </h2>
      {description && (
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-ink-muted">
          {description}
        </p>
      )}
    </div>
  );
}
