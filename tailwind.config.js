// tailwind.config.js
const colors = require('tailwindcss/colors')

module.exports = {
  darkMode: false, // or 'media' or 'class'
  theme: {
    extend: {},
    colors: {
      gray: colors.trueGray,
    }
  },
  variants: {
    extend: {
      backgroundColor: ['active', 'checked'],
    },
  },
  plugins: [],
}
