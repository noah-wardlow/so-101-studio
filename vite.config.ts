import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { mujocoReact } from 'mujoco-react/vite';
import path from 'path';

export default defineConfig({
  server: {
    watch: {
      ignored: [
        '**/.venv/**',
        '**/artifacts/**',
        '**/dist/**',
        '**/__pycache__/**',
      ],
    },
  },
  plugins: [
    tailwindcss(),
    mujocoReact({
      models: {
        so101: 'public/models/so101/SO101.xml',
      },
    }),
    react(),
  ],
  resolve: {
    dedupe: ['react', 'react-dom', 'three', '@react-three/fiber', '@react-three/drei'],
    alias: {
      '@': path.resolve('./src'),
      react: path.resolve('./node_modules/react'),
      'react-dom': path.resolve('./node_modules/react-dom'),
      'three/addons': path.resolve('./node_modules/three/examples/jsm'),
      three: path.resolve('./node_modules/three'),
      '@react-three/fiber': path.resolve('./node_modules/@react-three/fiber'),
      '@react-three/drei': path.resolve('./node_modules/@react-three/drei'),
    },
  },
});
