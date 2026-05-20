/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    container: { center: true, padding: "1.5rem" },
    extend: {
      fontFamily: {
        display: ['"Plus Jakarta Sans"', "ui-sans-serif", "system-ui", "sans-serif"],
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      colors: {
        cream: {
          DEFAULT: "#FBF7F0",
          50: "#FFFCF7",
          100: "#FBF7F0",
          200: "#F4ECDD",
        },
        lavender: {
          50: "#F5F1FF",
          100: "#ECE4FF",
          200: "#D9CBFF",
          300: "#BFA8FF",
          400: "#A285F6",
          500: "#8B6BEA",
          600: "#7553D6",
          700: "#5F40B5",
        },
        ink: {
          DEFAULT: "#1F1B2E",
          muted: "#5A5468",
          soft: "#8A8398",
        },
      },
      borderRadius: {
        xl: "1rem",
        "2xl": "1.25rem",
        "3xl": "1.75rem",
      },
      boxShadow: {
        soft: "0 1px 2px rgba(31, 27, 46, 0.04), 0 8px 24px -12px rgba(123, 95, 220, 0.18)",
        glow: "0 0 0 1px rgba(139, 107, 234, 0.18), 0 18px 40px -20px rgba(139, 107, 234, 0.45)",
      },
      keyframes: {
        "fade-in": { from: { opacity: "0", transform: "translateY(4px)" }, to: { opacity: "1", transform: "translateY(0)" } },
      },
      animation: {
        "fade-in": "fade-in 200ms ease-out both",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
