# =========================
# Build stage
# =========================
FROM node:18-alpine AS builder

WORKDIR /app

# Dependencias necesarias para compilar Next.js y librerías nativas
RUN apk add --no-cache libc6-compat python3 make g++ bash cairo-dev pango-dev jpeg-dev giflib-dev librsvg-dev

# Instalar pnpm
RUN npm install -g pnpm

# Copiar archivos de dependencias
COPY package.json pnpm-lock.yaml ./

# Instalar TODAS las dependencias (incluyendo devDependencies)
RUN pnpm install --frozen-lockfile --prod=false

# Copiar el resto del código fuente
COPY . .

ENV NEXT_TELEMETRY_DISABLED=1

# Compilar la aplicación
RUN pnpm build

# =========================
# Production stage
# =========================
FROM node:18-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Crear usuario no root
RUN addgroup --system --gid 1001 nodejs \
  && adduser --system --uid 1001 nextjs

# Copiar solo lo necesario desde el builder
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

# Lanzar la app
CMD ["node", "server.js"]
