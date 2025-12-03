<template>
  <div class="leaderboard-modal" v-if="isOpen" @click.self="close">
    <div class="leaderboard-content">
      <header class="leaderboard-header">
        <h2 class="leaderboard-title">
          🏆 Leaderboard
        </h2>
        <button class="close-btn" @click="close">×</button>
      </header>

      <!-- Champion Showcase -->
      <div class="champion-showcase" v-if="champion">
        <div class="champion-crown">👑</div>
        <div class="champion-info">
          <span class="champion-label">Current Champion</span>
          <span class="champion-name">{{ champion.name }}</span>
          <span class="champion-score">{{ champion.score }} pipes</span>
        </div>
        <button 
          class="btn btn-accent btn-small"
          @click="$emit('challenge')"
        >
          Challenge it!
        </button>
      </div>

      <!-- Leaderboard Table -->
      <div class="leaderboard-table-wrapper">
        <table class="leaderboard-table" v-if="entries.length > 0">
          <thead>
            <tr>
              <th class="rank-col">#</th>
              <th class="name-col">Name</th>
              <th class="pipes-col">Pipes</th>
              <th class="score-col" title="Score adjusted for network efficiency">Score</th>
              <th class="params-col">Network</th>
            </tr>
          </thead>
          <tbody>
            <tr 
              v-for="(entry, index) in entries" 
              :key="entry.id"
              :class="{ 'is-champion': entry.isChampion, 'is-you': entry.isYou }"
            >
              <td class="rank-col">
                <span class="rank-badge" :class="getRankClass(index)">
                  {{ index + 1 }}
                </span>
              </td>
              <td class="name-col">
                <span class="entry-name">{{ entry.name }}</span>
                <span v-if="entry.isChampion" class="crown-icon">👑</span>
                <span v-if="entry.isYou" class="you-badge">You</span>
              </td>
              <td class="pipes-col">
                <span class="entry-pipes">{{ entry.pipes || entry.score }}</span>
              </td>
              <td class="score-col">
                <span class="entry-score">{{ entry.score }}</span>
              </td>
              <td class="params-col">
                <span class="entry-params" :title="entry.architecture || 'Unknown'">
                  {{ formatParams(entry.params) }}
                </span>
              </td>
            </tr>
          </tbody>
        </table>

        <div class="empty-state" v-else-if="!isLoading">
          <p>No scores yet. Be the first to train an AI!</p>
        </div>

        <div class="loading-state" v-if="isLoading">
          <div class="spinner"></div>
          <p>Loading leaderboard...</p>
        </div>
      </div>

      <!-- Submit Score Section -->
      <div class="submit-section" v-if="canSubmit">
        <div class="submit-header">
          <h3>Submit Your Score</h3>
          <div class="your-stats">
            <span class="your-pipes">Pipes: <strong>{{ pendingScore }}</strong></span>
            <span class="your-score">Adjusted Score: <strong>{{ adjustedScore }}</strong></span>
            <span class="your-network">{{ pendingArchitecture }} ({{ efficiencyText }})</span>
          </div>
        </div>
        
        <div class="submit-form">
          <input
            type="text"
            v-model="networkName"
            placeholder="Enter network name..."
            maxlength="20"
            class="name-input"
            @keyup.enter="submitScore"
          />
          <button 
            class="btn btn-primary"
            :disabled="!networkName.trim() || isSubmitting"
            @click="submitScore"
          >
            {{ isSubmitting ? 'Submitting...' : 'Submit' }}
          </button>
        </div>

        <p class="submit-hint" v-if="willBeChampion">
          🎉 This will make you the new champion!
        </p>
      </div>

      <!-- Footer -->
      <footer class="leaderboard-footer">
        <p class="footer-note">
          Score = max pipes cleared in a single greedy evaluation run
        </p>
      </footer>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue'
import { apiClient, type LeaderboardEntry, calculateAdjustedScore, getEfficiencyMultiplier } from '@/services/apiClient'

export default defineComponent({
  name: 'Leaderboard',
  props: {
    isOpen: {
      type: Boolean,
      default: false,
    },
    canSubmit: {
      type: Boolean,
      default: false,
    },
    pendingScore: {
      type: Number,
      default: 0,
    },
    pendingParams: {
      type: Number,
      default: 0,
    },
    pendingArchitecture: {
      type: String,
      default: '6→64→64→2',
    },
  },
  emits: ['close', 'challenge', 'submit'],
  data() {
    return {
      entries: [] as LeaderboardEntry[],
      champion: null as LeaderboardEntry | null,
      networkName: '',
      isLoading: false,
      isSubmitting: false,
      lastSubmittedScore: null as number | null,
    }
  },
  computed: {
    adjustedScore(): number {
      return calculateAdjustedScore(this.pendingScore, this.pendingParams)
    },
    efficiencyMultiplier(): number {
      return getEfficiencyMultiplier(this.pendingParams)
    },
    efficiencyText(): string {
      const mult = this.efficiencyMultiplier
      if (mult > 1.05) return `×${mult.toFixed(2)} bonus`
      if (mult < 0.95) return `×${mult.toFixed(2)} penalty`
      return '×1.00'
    },
    willBeChampion(): boolean {
      if (!this.champion) return this.adjustedScore > 0
      return this.adjustedScore > this.champion.score
    },
  },
  watch: {
    isOpen(newVal) {
      if (newVal) {
        this.loadLeaderboard()
      }
    },
  },
  methods: {
    async loadLeaderboard() {
      this.isLoading = true
      try {
        const response = await apiClient.getLeaderboard(10)
        this.entries = response.entries
        this.champion = response.champion || null
      } catch (error) {
        console.error('Failed to load leaderboard:', error)
      } finally {
        this.isLoading = false
      }
    },
    async submitScore() {
      if (!this.networkName.trim() || this.isSubmitting) return

      this.isSubmitting = true
      try {
        const response = await apiClient.submitScore({
          name: this.networkName.trim(),
          pipes: this.pendingScore,
          params: this.pendingParams,
          architecture: this.pendingArchitecture,
        })

        if (response.success) {
          this.lastSubmittedScore = this.pendingScore
          this.$emit('submit', {
            entry: response.entry,
            isNewChampion: response.isNewChampion,
          })
          
          // Reload leaderboard to show new entry
          await this.loadLeaderboard()
          
          // Mark the new entry as "you"
          const yourEntry = this.entries.find(e => e.id === response.entry.id)
          if (yourEntry) {
            yourEntry.isYou = true
          }
        }
      } catch (error) {
        console.error('Failed to submit score:', error)
      } finally {
        this.isSubmitting = false
      }
    },
    close() {
      this.$emit('close')
    },
    formatDate(dateStr: string): string {
      const date = new Date(dateStr)
      const now = new Date()
      const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24))
      
      if (diffDays === 0) return 'Today'
      if (diffDays === 1) return 'Yesterday'
      if (diffDays < 7) return `${diffDays}d ago`
      
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
    },
    getRankClass(index: number): string {
      if (index === 0) return 'rank-gold'
      if (index === 1) return 'rank-silver'
      if (index === 2) return 'rank-bronze'
      return ''
    },
    formatParams(params: number | undefined): string {
      if (!params) return '—'
      if (params >= 1000) {
        return `${(params / 1000).toFixed(1)}K`
      }
      return params.toString()
    },
  },
})
</script>

<style scoped>
.leaderboard-modal {
  position: fixed;
  inset: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  background: rgba(0, 0, 0, 0.8);
  backdrop-filter: blur(4px);
  z-index: 100;
  padding: var(--spacing-lg);
}

.leaderboard-content {
  background: var(--color-bg-mid);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  width: 100%;
  max-width: 500px;
  max-height: 90vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
}

.leaderboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-lg);
  border-bottom: 1px solid var(--color-border);
}

.leaderboard-title {
  font-size: 1.25rem;
  font-family: var(--font-display);
  color: var(--color-text);
  margin: 0;
}

.close-btn {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--color-bg-light);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  color: var(--color-text-muted);
  font-size: 1.5rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.close-btn:hover {
  background: var(--color-bg-dark);
  color: var(--color-text);
}

.champion-showcase {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-lg);
  background: linear-gradient(135deg, rgba(255, 215, 0, 0.1), rgba(255, 193, 7, 0.05));
  border-bottom: 1px solid var(--color-border);
}

.champion-crown {
  font-size: 2rem;
  animation: float 2s ease-in-out infinite;
}

@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-4px); }
}

.champion-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.champion-label {
  font-size: 0.7rem;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.champion-name {
  font-family: var(--font-display);
  font-size: 1rem;
  color: var(--color-accent);
}

.champion-score {
  font-size: 0.85rem;
  color: var(--color-text-muted);
}

.leaderboard-table-wrapper {
  flex: 1;
  overflow-y: auto;
  padding: var(--spacing-md);
}

.leaderboard-table {
  width: 100%;
  border-collapse: collapse;
}

.leaderboard-table th {
  font-size: 0.7rem;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  text-align: left;
  padding: var(--spacing-sm);
  border-bottom: 1px solid var(--color-border);
}

.leaderboard-table td {
  padding: var(--spacing-sm);
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.leaderboard-table tr:last-child td {
  border-bottom: none;
}

.leaderboard-table tr.is-champion {
  background: rgba(255, 215, 0, 0.08);
}

.leaderboard-table tr.is-you {
  background: rgba(0, 217, 255, 0.1);
}

.rank-col { width: 35px; }
.name-col { }
.pipes-col { width: 50px; text-align: right; }
.score-col { width: 55px; text-align: right; }
.params-col { width: 60px; text-align: right; }

.rank-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: var(--color-bg-light);
  font-family: var(--font-display);
  font-size: 0.75rem;
  color: var(--color-text-muted);
}

.rank-badge.rank-gold {
  background: linear-gradient(135deg, #ffd700, #ffb700);
  color: #000;
}

.rank-badge.rank-silver {
  background: linear-gradient(135deg, #c0c0c0, #a0a0a0);
  color: #000;
}

.rank-badge.rank-bronze {
  background: linear-gradient(135deg, #cd7f32, #b87333);
  color: #fff;
}

.entry-name {
  font-family: var(--font-display);
  color: var(--color-text);
}

.crown-icon {
  margin-left: 4px;
  font-size: 0.8rem;
}

.you-badge {
  margin-left: 6px;
  font-size: 0.65rem;
  background: var(--color-primary);
  color: var(--color-bg-dark);
  padding: 2px 6px;
  border-radius: 4px;
  text-transform: uppercase;
}

.entry-pipes {
  font-family: var(--font-display);
  font-size: 0.85rem;
  color: var(--color-text-muted);
}

.entry-score {
  font-family: var(--font-display);
  color: var(--color-primary);
}

.entry-params {
  font-size: 0.75rem;
  color: var(--color-text-muted);
  cursor: help;
}

.entry-date {
  font-size: 0.75rem;
  color: var(--color-text-muted);
}

.empty-state,
.loading-state {
  text-align: center;
  padding: var(--spacing-xl);
  color: var(--color-text-muted);
}

.spinner {
  width: 24px;
  height: 24px;
  border: 2px solid var(--color-border);
  border-top-color: var(--color-primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin: 0 auto var(--spacing-sm);
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.submit-section {
  padding: var(--spacing-lg);
  border-top: 1px solid var(--color-border);
  background: var(--color-bg-light);
}

.submit-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: var(--spacing-md);
}

.submit-header h3 {
  font-size: 0.9rem;
  color: var(--color-text);
  margin: 0;
}

.your-stats {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 2px;
}

.your-pipes,
.your-score,
.your-network {
  font-size: 0.8rem;
  color: var(--color-text-muted);
}

.your-pipes strong {
  color: var(--color-text);
}

.your-score strong {
  color: var(--color-accent);
}

.your-network {
  font-size: 0.7rem;
  opacity: 0.8;
}

.submit-form {
  display: flex;
  gap: var(--spacing-sm);
}

.name-input {
  flex: 1;
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--color-bg-dark);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  color: var(--color-text);
  font-size: 0.9rem;
}

.name-input:focus {
  outline: none;
  border-color: var(--color-primary);
}

.name-input::placeholder {
  color: var(--color-text-muted);
}

.submit-hint {
  margin-top: var(--spacing-sm);
  font-size: 0.8rem;
  color: var(--color-accent);
  text-align: center;
}

.leaderboard-footer {
  padding: var(--spacing-sm) var(--spacing-lg);
  border-top: 1px solid var(--color-border);
  background: var(--color-bg-dark);
}

.footer-note {
  font-size: 0.7rem;
  color: var(--color-text-muted);
  text-align: center;
  margin: 0;
}

.btn-accent {
  background: var(--color-accent);
  color: var(--color-bg-dark);
  border: none;
}

.btn-accent:hover {
  background: #ffcc00;
}

.btn-small {
  padding: var(--spacing-xs) var(--spacing-sm);
  font-size: 0.8rem;
}
</style>




