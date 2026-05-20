import * as React from "react";
import * as RadioGroupPrimitive from "@radix-ui/react-radio-group";
import { Check } from "lucide-react";
import { cn } from "@/lib/utils";

export const RadioGroup = React.forwardRef<
  React.ElementRef<typeof RadioGroupPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof RadioGroupPrimitive.Root>
>(({ className, ...props }, ref) => (
  <RadioGroupPrimitive.Root ref={ref} className={cn("grid gap-2", className)} {...props} />
));
RadioGroup.displayName = "RadioGroup";

type ItemProps = React.ComponentPropsWithoutRef<typeof RadioGroupPrimitive.Item> & {
  label: string;
  description?: string;
};

export const RadioCard = React.forwardRef<
  React.ElementRef<typeof RadioGroupPrimitive.Item>,
  ItemProps
>(({ className, label, description, ...props }, ref) => (
  <RadioGroupPrimitive.Item
    ref={ref}
    className={cn(
      "group flex w-full cursor-pointer items-start gap-3 rounded-2xl border border-lavender-100 bg-white/70 p-3 text-left transition-all hover:border-lavender-200 hover:bg-lavender-50/60 data-[state=checked]:border-lavender-400 data-[state=checked]:bg-lavender-50 data-[state=checked]:shadow-soft focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-lavender-400",
      className,
    )}
    {...props}
  >
    <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-lavender-300 bg-white group-data-[state=checked]:border-lavender-500 group-data-[state=checked]:bg-lavender-500">
      <RadioGroupPrimitive.Indicator className="flex items-center justify-center">
        <Check className="h-3 w-3 text-white" strokeWidth={3} />
      </RadioGroupPrimitive.Indicator>
    </span>
    <span className="flex flex-col">
      <span className="text-sm font-medium text-ink">{label}</span>
      {description ? (
        <span className="mt-0.5 text-xs text-ink-muted">{description}</span>
      ) : null}
    </span>
  </RadioGroupPrimitive.Item>
));
RadioCard.displayName = "RadioCard";
