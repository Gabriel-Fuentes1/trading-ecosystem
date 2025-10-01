# Cambios realizados por assistant

- Se copió el `pnpm-lock.yaml` proporcionado al proyecto.
- Se actualizó `package.json` agregando dependencias de desarrollo faltantes: autoprefixer, tailwindcss-animate.
- Se añadió script `build` si faltaba.
- Se actualizó `web/Dockerfile` para usar `pnpm install --frozen-lockfile --prod=false`.
- Respaldo de archivos: `package.json.bak`, `web/Dockerfile.bak`.
