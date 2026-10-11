/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#11110F",
        surface: {
          DEFAULT: "#1A1916",
          elevated: "#23211D",
          hover: "#2A2722",
        },
        border: {
          subtle: "#2A2722",
          strong: "#3B3730",
        },
        text: {
          primary: "#F1EDE4",
          muted: "#A59F93",
          dim: "#6B665D",
        },
        accent: {
          DEFAULT: "#D7A85C",
          hover: "#C49547",
          subtle: "rgba(215, 168, 92, 0.15)",
        },
        status: {
          success: "#5FA076",
          warning: "#D7A85C",
          error: "#D45A5A",
        }
      },
      fontFamily: {
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
    },
  },
  plugins: [],
}
