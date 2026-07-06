FROM node:22-bookworm-slim AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY index.html tsconfig.json vite.config.ts ./
COPY public ./public
COPY src ./src
RUN npm run build

FROM nginx:1.27-alpine

ENV SIGNALFORGE_API_BASE_URL=http://localhost:8000

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY docker-entrypoint.d /docker-entrypoint.d
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
