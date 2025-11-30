<template>
  <div class="network-viewer panel" @click="openDetailView" title="Click to open full visualization">
    <div class="panel-header">
      <span>Neural Network</span>
      <span class="header-info">{{ networkInfo }}</span>
      <span class="expand-hint">🔍</span>
    </div>

    <!-- Static Network Diagram -->
    <svg class="network-svg" viewBox="0 0 320 200" preserveAspectRatio="xMidYMid meet">
      <!-- Layer labels -->
      <text x="40" y="15" class="layer-label" text-anchor="middle">Input</text>
      <text x="120" y="15" class="layer-label" text-anchor="middle">H1</text>
      <text x="200" y="15" class="layer-label" text-anchor="middle">H2</text>
      <text x="280" y="15" class="layer-label" text-anchor="middle">Output</text>

      <!-- Static edges (simplified) -->
      <g class="edges" opacity="0.3">
        <!-- Input to H1 -->
        <line v-for="i in 6" :key="'ih1-'+i" :x1="50" :y1="20 + i * 25" x2="110" y2="65" stroke="#4a90a4" stroke-width="0.5"/>
        <line v-for="i in 6" :key="'ih1b-'+i" :x1="50" :y1="20 + i * 25" x2="110" y2="95" stroke="#4a90a4" stroke-width="0.5"/>
        <line v-for="i in 6" :key="'ih1c-'+i" :x1="50" :y1="20 + i * 25" x2="110" y2="125" stroke="#4a90a4" stroke-width="0.5"/>
        
        <!-- H1 to H2 -->
        <line x1="130" y1="65" x2="190" y2="65" stroke="#4a90a4" stroke-width="0.5"/>
        <line x1="130" y1="65" x2="190" y2="95" stroke="#4a90a4" stroke-width="0.5"/>
        <line x1="130" y1="65" x2="190" y2="125" stroke="#4a90a4" stroke-width="0.5"/>
        <line x1="130" y1="95" x2="190" y2="65" stroke="#4a90a4" stroke-width="0.5"/>
        <line x1="130" y1="95" x2="190" y2="95" stroke="#4a90a4" stroke-width="0.5"/>
        <line x1="130" y1="95" x2="190" y2="125" stroke="#4a90a4" stroke-width="0.5"/>
        <line x1="130" y1="125" x2="190" y2="65" stroke="#4a90a4" stroke-width="0.5"/>
        <line x1="130" y1="125" x2="190" y2="95" stroke="#4a90a4" stroke-width="0.5"/>
        <line x1="130" y1="125" x2="190" y2="125" stroke="#4a90a4" stroke-width="0.5"/>
        
        <!-- H2 to Output -->
        <line x1="210" y1="65" x2="270" y2="75" stroke="#4a90a4" stroke-width="0.5"/>
        <line x1="210" y1="65" x2="270" y2="115" stroke="#4a90a4" stroke-width="0.5"/>
        <line x1="210" y1="95" x2="270" y2="75" stroke="#4a90a4" stroke-width="0.5"/>
        <line x1="210" y1="95" x2="270" y2="115" stroke="#4a90a4" stroke-width="0.5"/>
        <line x1="210" y1="125" x2="270" y2="75" stroke="#4a90a4" stroke-width="0.5"/>
        <line x1="210" y1="125" x2="270" y2="115" stroke="#4a90a4" stroke-width="0.5"/>
      </g>

      <!-- Input nodes with live values -->
      <g class="input-layer">
        <g v-for="(label, i) in inputLabels" :key="'in-'+i" :transform="`translate(40, ${30 + i * 25})`">
          <circle r="8" :fill="getInputColor(i)" stroke="#3d3d5a" stroke-width="1"/>
          <text x="-12" y="4" class="input-label" text-anchor="end">{{ label }}</text>
        </g>
      </g>

      <!-- Hidden layer nodes (static) -->
      <g class="hidden-layers">
        <!-- H1 -->
        <circle cx="120" cy="65" r="6" fill="#2a4a5a" stroke="#3d3d5a"/>
        <circle cx="120" cy="95" r="6" fill="#2a4a5a" stroke="#3d3d5a"/>
        <circle cx="120" cy="125" r="6" fill="#2a4a5a" stroke="#3d3d5a"/>
        <text x="120" y="155" class="layer-size" text-anchor="middle">(64)</text>
        
        <!-- H2 -->
        <circle cx="200" cy="65" r="6" fill="#2a4a5a" stroke="#3d3d5a"/>
        <circle cx="200" cy="95" r="6" fill="#2a4a5a" stroke="#3d3d5a"/>
        <circle cx="200" cy="125" r="6" fill="#2a4a5a" stroke="#3d3d5a"/>
        <text x="200" y="155" class="layer-size" text-anchor="middle">(64)</text>
      </g>

      <!-- Output nodes with live values -->
      <g class="output-layer">
        <!-- Idle -->
        <g transform="translate(280, 75)">
          <circle 
            r="12" 
            :fill="selectedAction === 0 ? '#00d9ff' : '#2a4a5a'" 
            :stroke="selectedAction === 0 ? '#00d9ff' : '#3d3d5a'" 
            stroke-width="2"
          />
          <text x="18" y="4" class="output-label">idle</text>
        </g>
        <!-- Flap -->
        <g transform="translate(280, 115)">
          <circle 
            r="12" 
            :fill="selectedAction === 1 ? '#ff6b9d' : '#2a4a5a'" 
            :stroke="selectedAction === 1 ? '#ff6b9d' : '#3d3d5a'" 
            stroke-width="2"
          />
          <text x="18" y="4" class="output-label">flap</text>
        </g>
      </g>
    </svg>

    <!-- Action Decision -->
    <div class="action-decision">
      <div class="decision-arrow" :class="{ 'flap': selectedAction === 1 }">
        {{ selectedAction === 0 ? '→ Idle' : '↑ Flap!' }}
      </div>
    </div>

    <!-- Q-values display -->
    <div class="q-values">
      <div class="q-value" :class="{ selected: selectedAction === 0 }">
        <span class="q-label">Idle (no action)</span>
        <span class="q-number" :class="{ positive: qValues[0] > 0, negative: qValues[0] < 0 }">
          {{ formatQValue(qValues[0]) }}
        </span>
      </div>
      <div class="q-value" :class="{ selected: selectedAction === 1 }">
        <span class="q-label">Flap (jump)</span>
        <span class="q-number" :class="{ positive: qValues[1] > 0, negative: qValues[1] < 0 }">
          {{ formatQValue(qValues[1]) }}
        </span>
      </div>
    </div>

    <!-- Input Values Table -->
    <div class="input-values">
      <div class="input-row" v-for="(label, i) in inputLabels" :key="i">
        <span class="input-name">{{ label }}</span>
        <span class="input-val">{{ formatInputValue(i) }}</span>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, type PropType } from 'vue'

const INPUT_LABELS = ['y', 'vel', 'dx₁', 'dy₁', 'dx₂', 'dy₂']

export default defineComponent({
  name: 'NetworkViewer',
  props: {
    activations: {
      type: Array as PropType<number[][]>,
      default: () => [],
    },
    qValues: {
      type: Array as unknown as PropType<[number, number]>,
      default: () => [0, 0] as [number, number],
    },
    selectedAction: {
      type: Number,
      default: 0,
    },
  },
  data() {
    return {
      inputLabels: INPUT_LABELS,
    }
  },
  computed: {
    networkInfo(): string {
      return '6 → 64 → 64 → 2'
    },
    inputValues(): number[] {
      return this.activations[0] || [0, 0, 0, 0, 0, 0]
    },
  },
  methods: {
    getInputColor(index: number): string {
      const value = this.inputValues[index] || 0
      const normalized = Math.max(-1, Math.min(1, value))
      
      if (normalized > 0) {
        const intensity = Math.floor(normalized * 200)
        return `rgb(${100 + intensity}, ${200}, ${255})`
      } else {
        const intensity = Math.floor(-normalized * 200)
        return `rgb(${255}, ${150 - intensity}, ${100})`
      }
    },
    formatQValue(value: number): string {
      if (value === 0) return '0'
      if (Math.abs(value) < 0.001) {
        return value.toExponential(2)
      }
      if (Math.abs(value) >= 1000) {
        return value.toExponential(2)
      }
      return value.toFixed(3)
    },
    formatInputValue(index: number): string {
      const value = this.inputValues[index] || 0
      return value.toFixed(2)
    },
    openDetailView() {
      // Open a new browser window with the network visualization
      const width = 1400
      const height = 800
      const left = (window.screen.width - width) / 2
      const top = (window.screen.height - height) / 2
      
      window.open(
        '/network-detail.html',
        'NetworkVisualization',
        `width=${width},height=${height},left=${left},top=${top},resizable=yes,scrollbars=no,toolbar=no,menubar=no,location=no,status=no`
      )
    },
  },
})
</script>

<style scoped>
.network-viewer {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  cursor: pointer;
  transition: all 0.2s ease;
}

.network-viewer:hover {
  border-color: var(--color-primary);
  box-shadow: 0 0 20px rgba(0, 217, 255, 0.15);
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-family: var(--font-display);
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--color-primary);
  padding-bottom: var(--spacing-xs);
  border-bottom: 1px solid var(--color-border);
}

.header-info {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--color-text-muted);
}

.expand-hint {
  font-size: 0.9rem;
  opacity: 0.5;
  transition: all 0.2s ease;
}

.network-viewer:hover .expand-hint {
  opacity: 1;
  transform: scale(1.2);
}

.network-svg {
  width: 100%;
  height: 180px;
  background: var(--color-bg-light);
  border-radius: var(--radius-md);
}

.layer-label {
  font-family: var(--font-display);
  font-size: 9px;
  fill: var(--color-text-muted);
}

.layer-size {
  font-family: var(--font-mono);
  font-size: 8px;
  fill: var(--color-text-muted);
}

.input-label, .output-label {
  font-family: var(--font-mono);
  font-size: 8px;
  fill: var(--color-text);
}

.action-decision {
  display: flex;
  justify-content: center;
  padding: var(--spacing-xs);
}

.decision-arrow {
  font-family: var(--font-display);
  font-size: 1rem;
  color: var(--color-primary);
  padding: var(--spacing-xs) var(--spacing-md);
  background: var(--color-bg-light);
  border-radius: var(--radius-sm);
  border: 1px solid var(--color-border);
}

.decision-arrow.flap {
  color: var(--color-accent);
  border-color: var(--color-accent);
}

.q-values {
  display: flex;
  gap: var(--spacing-sm);
}

.q-value {
  flex: 1;
  padding: var(--spacing-sm);
  background: var(--color-bg-light);
  border-radius: var(--radius-sm);
  border: 1px solid var(--color-border);
  transition: all 0.15s ease;
}

.q-value.selected {
  border-color: var(--color-primary);
  box-shadow: 0 0 10px rgba(0, 217, 255, 0.2);
}

.q-label {
  display: block;
  font-family: var(--font-display);
  font-size: 0.65rem;
  text-transform: uppercase;
  color: var(--color-text-muted);
  margin-bottom: 2px;
}

.q-number {
  font-family: var(--font-mono);
  font-size: 1rem;
  color: var(--color-text);
}

.q-number.positive {
  color: var(--color-success);
}

.q-number.negative {
  color: var(--color-danger);
}

.input-values {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--spacing-xs);
  padding: var(--spacing-sm);
  background: var(--color-bg-light);
  border-radius: var(--radius-sm);
}

.input-row {
  display: flex;
  justify-content: space-between;
  font-size: 0.7rem;
}

.input-name {
  font-family: var(--font-mono);
  color: var(--color-text-muted);
}

.input-val {
  font-family: var(--font-mono);
  color: var(--color-text);
}
</style>
