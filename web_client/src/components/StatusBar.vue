<template>
  <div class="status-bar panel">
    <div class="status-grid">
      <div class="status-item">
        <span class="status-label">Mode</span>
        <span class="status-value" :class="modeClass">{{ modeText }}</span>
      </div>
      <div class="status-item">
        <span class="status-label">Episode</span>
        <span class="status-value text-primary">{{ episode }}</span>
      </div>
      <div class="status-item">
        <span class="status-label">Last Score</span>
        <span class="status-value text-accent">{{ lastScore }}</span>
      </div>
      <div 
        ref="bestScoreItem"
        class="status-item best-score-item" 
        @mouseenter="onMouseEnter" 
        @mouseleave="showTooltip = false"
      >
        <span class="status-label">Best Score</span>
        <div class="best-score-row">
          <span class="status-value text-success">{{ bestScore }}</span>
          <button 
            v-if="canSubmitToLeaderboard" 
            class="submit-btn"
            @click.stop="$emit('submit-score')"
          >
            Submit
          </button>
        </div>
        
        <!-- Auto-eval tooltip -->
        <div 
          v-if="showTooltip && autoEvalHistory.length > 0" 
          class="eval-tooltip"
          :style="tooltipStyle"
        >
          <div class="tooltip-header">Auto-Eval History (last {{ autoEvalHistory.length }})</div>
          <canvas ref="tooltipCanvas" class="tooltip-chart" width="200" height="80"></canvas>
          <div class="tooltip-stats">
            <span>Avg: {{ latestAvg.toFixed(1) }}</span>
            <span>Max: {{ latestMax }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, type PropType } from 'vue'

interface AutoEvalResult {
  avgScore: number
  maxScore: number
  minScore: number
  scores: number[]
  episode: number
}

export default defineComponent({
  name: 'StatusBar',
  props: {
    mode: {
      type: String as () => 'idle' | 'configuring' | 'training' | 'eval' | 'manual',
      default: 'idle',
    },
    episode: {
      type: Number,
      default: 0,
    },
    lastScore: {
      type: Number,
      default: 0,
    },
    bestScore: {
      type: Number,
      default: 0,
    },
    stepsPerSecond: {
      type: Number,
      default: 0,
    },
    autoEvalHistory: {
      type: Array as PropType<AutoEvalResult[]>,
      default: () => [],
    },
    canSubmitToLeaderboard: {
      type: Boolean,
      default: false,
    },
  },
  emits: ['submit-score'],
  data() {
    return {
      showTooltip: false,
      tooltipTop: 0,
      tooltipLeft: 0,
    }
  },
  computed: {
    modeText(): string {
      const modes: Record<string, string> = {
        idle: 'Ready',
        configuring: 'Configuring',
        training: 'Training',
        eval: 'Evaluating',
        manual: 'Manual',
      }
      return modes[this.mode] || 'Unknown'
    },
    modeClass(): string {
      return `mode-${this.mode}`
    },
    latestAvg(): number {
      if (this.autoEvalHistory.length === 0) return 0
      return this.autoEvalHistory[this.autoEvalHistory.length - 1].avgScore
    },
    latestMax(): number {
      if (this.autoEvalHistory.length === 0) return 0
      return this.autoEvalHistory[this.autoEvalHistory.length - 1].maxScore
    },
    tooltipStyle(): Record<string, string> {
      return {
        top: `${this.tooltipTop}px`,
        left: `${this.tooltipLeft}px`,
      }
    },
  },
  watch: {
    showTooltip(val: boolean) {
      if (val && this.autoEvalHistory.length > 0) {
        this.$nextTick(() => this.drawTooltipChart())
      }
    },
  },
  methods: {
    onMouseEnter() {
      const el = this.$refs.bestScoreItem as HTMLElement
      if (el) {
        const rect = el.getBoundingClientRect()
        const tooltipWidth = 220
        // Position below the element, but ensure it doesn't go off screen to the right
        this.tooltipTop = rect.bottom + 8
        this.tooltipLeft = Math.min(rect.left + rect.width / 2 - tooltipWidth / 2, window.innerWidth - tooltipWidth - 16)
        // Also ensure it doesn't go off screen to the left
        this.tooltipLeft = Math.max(16, this.tooltipLeft)
      }
      this.showTooltip = true
    },
    drawTooltipChart() {
      const canvas = this.$refs.tooltipCanvas as HTMLCanvasElement
      if (!canvas) return

      const ctx = canvas.getContext('2d')
      if (!ctx) return

      const width = canvas.width
      const height = canvas.height
      const padding = { top: 10, right: 10, bottom: 20, left: 30 }
      const chartWidth = width - padding.left - padding.right
      const chartHeight = height - padding.top - padding.bottom

      // Clear canvas
      ctx.fillStyle = '#1a1a2e'
      ctx.fillRect(0, 0, width, height)

      const history = this.autoEvalHistory
      if (history.length === 0) return

      // Get all scores for range calculation
      const allMaxScores = history.map(h => h.maxScore)
      const maxVal = Math.max(...allMaxScores, 1)

      // Draw grid lines
      ctx.strokeStyle = '#333'
      ctx.lineWidth = 0.5
      for (let i = 0; i <= 4; i++) {
        const y = padding.top + (chartHeight * i) / 4
        ctx.beginPath()
        ctx.moveTo(padding.left, y)
        ctx.lineTo(width - padding.right, y)
        ctx.stroke()
      }

      // Draw max scores (bars)
      const barWidth = Math.max(4, chartWidth / history.length - 2)
      ctx.fillStyle = 'rgba(0, 255, 136, 0.3)'
      history.forEach((h, i) => {
        const x = padding.left + (i / Math.max(1, history.length - 1)) * chartWidth - barWidth / 2
        const barHeight = (h.maxScore / maxVal) * chartHeight
        ctx.fillRect(x, padding.top + chartHeight - barHeight, barWidth, barHeight)
      })

      // Draw avg scores (line)
      ctx.strokeStyle = '#00ff88'
      ctx.lineWidth = 2
      ctx.beginPath()
      history.forEach((h, i) => {
        const x = padding.left + (i / Math.max(1, history.length - 1)) * chartWidth
        const y = padding.top + chartHeight - (h.avgScore / maxVal) * chartHeight
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()

      // Draw dots
      ctx.fillStyle = '#00ff88'
      history.forEach((h, i) => {
        const x = padding.left + (i / Math.max(1, history.length - 1)) * chartWidth
        const y = padding.top + chartHeight - (h.avgScore / maxVal) * chartHeight
        ctx.beginPath()
        ctx.arc(x, y, 3, 0, Math.PI * 2)
        ctx.fill()
      })

      // Draw Y axis labels
      ctx.fillStyle = '#888'
      ctx.font = '9px monospace'
      ctx.textAlign = 'right'
      ctx.fillText(maxVal.toString(), padding.left - 4, padding.top + 4)
      ctx.fillText('0', padding.left - 4, padding.top + chartHeight + 4)

      // Draw X axis labels (episode numbers)
      ctx.textAlign = 'center'
      if (history.length > 0) {
        ctx.fillText(`Ep ${history[0].episode}`, padding.left, height - 4)
        if (history.length > 1) {
          ctx.fillText(`Ep ${history[history.length - 1].episode}`, width - padding.right, height - 4)
        }
      }
    },
  },
})
</script>

<style scoped>
.status-bar {
  padding: var(--spacing-md);
  overflow: visible;
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
  gap: var(--spacing-sm);
}

.status-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}

.best-score-item {
  position: relative;
  cursor: help;
}

.best-score-row {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
}

.submit-btn {
  padding: 2px 6px;
  font-size: 0.65rem;
  font-weight: 600;
  text-transform: uppercase;
  background: linear-gradient(135deg, var(--color-accent), #ffaa00);
  color: var(--color-bg-dark);
  border: none;
  border-radius: var(--radius-sm);
  cursor: pointer;
  animation: pulse-glow 1.5s ease-in-out infinite;
  transition: transform 0.2s ease;
}

.submit-btn:hover {
  transform: scale(1.05);
}

@keyframes pulse-glow {
  0%, 100% { 
    box-shadow: 0 0 4px rgba(255, 215, 0, 0.4); 
  }
  50% { 
    box-shadow: 0 0 12px rgba(255, 215, 0, 0.8); 
  }
}

.status-label {
  font-size: 0.7rem;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 2px;
}

.status-value {
  font-family: var(--font-display);
  font-size: 0.85rem;
}

.mode-idle { color: var(--color-text-muted); }
.mode-training { color: var(--color-primary); }
.mode-eval { color: var(--color-success); }
.mode-manual { color: var(--color-accent); }

/* Auto-eval tooltip */
.eval-tooltip {
  position: fixed;
  z-index: 9999;
  background: var(--color-bg-dark);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: var(--spacing-sm);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
  min-width: 220px;
  pointer-events: none;
}

.eval-tooltip::before {
  display: none;
}

.tooltip-header {
  font-size: 0.7rem;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: var(--spacing-xs);
  text-align: center;
}

.tooltip-chart {
  display: block;
  border-radius: var(--radius-sm);
  margin-bottom: var(--spacing-xs);
}

.tooltip-stats {
  display: flex;
  justify-content: space-between;
  font-size: 0.75rem;
  color: var(--color-text-muted);
}

.tooltip-stats span:last-child {
  color: var(--color-success);
}
</style>









