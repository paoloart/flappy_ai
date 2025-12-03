<template>
  <div class="leaderboard-panel panel">
    <div class="panel-header">
      <span>🏆 Leaderboard</span>
    </div>
    
    <div class="panel-content">
      <!-- Champion Highlight -->
      <div class="champion-highlight" v-if="champion">
        <div class="champion-badge">👑</div>
        <div class="champion-details">
          <span class="champion-name">{{ champion.name }}</span>
          <span class="champion-score">{{ champion.score }} pipes</span>
        </div>
      </div>

      <!-- Leaderboard List -->
      <div class="leaderboard-list" v-if="entries.length > 0">
        <div 
          v-for="(entry, index) in entries" 
          :key="entry.id"
          class="leaderboard-entry"
          :class="{ 'is-champion': index === 0 }"
        >
          <span class="entry-rank" :class="getRankClass(index)">{{ index + 1 }}</span>
          <span class="entry-name">{{ entry.name }}</span>
          <span class="entry-score">{{ entry.score }}</span>
        </div>
      </div>

      <!-- Empty State -->
      <div class="empty-state" v-else-if="!isLoading">
        <p>No scores yet!</p>
        <p class="hint">Train an AI to get on the board</p>
      </div>

      <!-- Loading -->
      <div class="loading-state" v-if="isLoading">
        <div class="spinner"></div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue'
import { apiClient, type LeaderboardEntry } from '@/services/apiClient'

export default defineComponent({
  name: 'LeaderboardPanel',
  data() {
    return {
      entries: [] as LeaderboardEntry[],
      champion: null as LeaderboardEntry | null,
      isLoading: false,
    }
  },
  mounted() {
    this.loadLeaderboard()
  },
  methods: {
    async loadLeaderboard() {
      this.isLoading = true
      try {
        const response = await apiClient.getLeaderboard(5)
        this.entries = response.entries
        this.champion = response.champion || null
      } catch (error) {
        console.error('Failed to load leaderboard:', error)
      } finally {
        this.isLoading = false
      }
    },
    getRankClass(index: number): string {
      if (index === 0) return 'rank-gold'
      if (index === 1) return 'rank-silver'
      if (index === 2) return 'rank-bronze'
      return ''
    },
  },
})
</script>

<style scoped>
.leaderboard-panel {
  background: var(--color-bg-mid);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
}

.panel-header {
  padding: var(--spacing-sm) var(--spacing-md);
  border-bottom: 1px solid var(--color-border);
  font-family: var(--font-display);
  font-size: 0.85rem;
  color: var(--color-text);
}

.panel-content {
  padding: var(--spacing-md);
}

.champion-highlight {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm) var(--spacing-md);
  background: linear-gradient(135deg, rgba(255, 215, 0, 0.15), rgba(255, 193, 7, 0.05));
  border-radius: var(--radius-sm);
  margin-bottom: var(--spacing-md);
}

.champion-badge {
  font-size: 1.2rem;
  animation: float 2s ease-in-out infinite;
}

@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-2px); }
}

.champion-details {
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.champion-name {
  font-family: var(--font-display);
  font-size: 0.9rem;
  color: var(--color-accent);
}

.champion-score {
  font-size: 0.75rem;
  color: var(--color-text-muted);
}

.leaderboard-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.leaderboard-entry {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-sm);
  transition: background 0.2s ease;
}

.leaderboard-entry:hover {
  background: rgba(255, 255, 255, 0.03);
}

.leaderboard-entry.is-champion {
  background: rgba(255, 215, 0, 0.08);
}

.entry-rank {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--color-bg-light);
  font-family: var(--font-display);
  font-size: 0.65rem;
  color: var(--color-text-muted);
}

.entry-rank.rank-gold {
  background: linear-gradient(135deg, #ffd700, #ffb700);
  color: #000;
}

.entry-rank.rank-silver {
  background: linear-gradient(135deg, #c0c0c0, #a0a0a0);
  color: #000;
}

.entry-rank.rank-bronze {
  background: linear-gradient(135deg, #cd7f32, #b87333);
  color: #fff;
}

.entry-name {
  flex: 1;
  font-family: var(--font-display);
  font-size: 0.8rem;
  color: var(--color-text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.entry-score {
  font-family: var(--font-display);
  font-size: 0.8rem;
  color: var(--color-primary);
}

.empty-state {
  text-align: center;
  padding: var(--spacing-md);
}

.empty-state p {
  margin: 0;
  color: var(--color-text-muted);
  font-size: 0.85rem;
}

.empty-state .hint {
  font-size: 0.75rem;
  margin-top: var(--spacing-xs);
  opacity: 0.7;
}

.loading-state {
  display: flex;
  justify-content: center;
  padding: var(--spacing-md);
}

.spinner {
  width: 20px;
  height: 20px;
  border: 2px solid var(--color-border);
  border-top-color: var(--color-primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>

