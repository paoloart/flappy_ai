FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY index.js ./

# Create data directory
RUN mkdir -p /app/data

ENV PORT=3001
ENV DATA_DIR=/app/data

EXPOSE 3001

CMD ["node", "index.js"]

