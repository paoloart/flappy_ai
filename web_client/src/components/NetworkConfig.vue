<template>
  <div class="network-config">
    <div class="config-header">
      <h2 class="config-title">Configure Neural Network</h2>
      <p class="config-subtitle">Set up the architecture before training</p>
    </div>

    <div class="config-body">
      <!-- Layer Count Selector -->
      <div class="config-section">
        <label class="section-label">Hidden Layers</label>
        <div class="layer-count-buttons">
          <button
            v-for="n in 4"
            :key="n"
            class="layer-btn"
            :class="{ active: layerCount === n }"
            @click="setLayerCount(n)"
          >
            {{ n }}
          </button>
        </div>
      </div>

      <!-- Per-Layer Node Count -->
      <div class="config-section">
        <label class="section-label">Nodes per Layer</label>
        <div class="layer-configs">
          <div
            v-for="(nodes, i) in layers"
            :key="i"
            class="layer-config"
          >
            <span class="layer-label">H{{ i + 1 }}</span>
            <div class="node-buttons">
              <button
                v-for="size in nodeSizes"
                :key="size"
                class="node-btn"
                :class="{ active: nodes === size }"
                @click="setLayerNodes(i, size)"
              >
                {{ size }}
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Network Preview -->
      <div class="config-section">
        <label class="section-label">Architecture Preview</label>
        <div class="network-preview">
          <svg class="preview-svg" :viewBox="svgViewBox" preserveAspectRatio="xMidYMid meet">
            <!-- Connections -->
            <g class="connections">
              <template v-for="(conn, ci) in connections" :key="'conn-'+ci">
                <line
                  v-for="(line, li) in conn"
                  :key="'line-'+ci+'-'+li"
                  :x1="line.x1"
                  :y1="line.y1"
                  :x2="line.x2"
                  :y2="line.y2"
                  stroke="#4a90a4"
                  stroke-width="0.5"
                  opacity="0.3"
                />
              </template>
            </g>

            <!-- Layer labels -->
            <text
              v-for="(label, i) in layerLabels"
              :key="'label-'+i"
              :x="layerX(i)"
              y="12"
              class="layer-label-text"
              text-anchor="middle"
            >
              {{ label }}
            </text>

            <!-- Nodes -->
            <g v-for="(layer, li) in allLayers" :key="'layer-'+li">
              <circle
                v-for="(_, ni) in layer.displayNodes"
                :key="'node-'+li+'-'+ni"
                :cx="layerX(li)"
                :cy="nodeY(ni, layer.displayNodes)"
                :r="nodeRadius"
                :fill="layerColor(li)"
                stroke="#3d3d5a"
                stroke-width="1"
              />
              <!-- Show "..." if truncated -->
              <text
                v-if="layer.truncated"
                :x="layerX(li)"
                :y="nodeY(layer.displayNodes, layer.displayNodes) + 15"
                class="truncation-text"
                text-anchor="middle"
              >
                ...{{ layer.total }}
              </text>
            </g>
          </svg>
          <div class="architecture-text">{{ architectureText }}</div>
        </div>
      </div>
    </div>

    <div class="config-footer">
      <button class="btn btn-primary btn-large" @click="startTraining">
        Start Training
      </button>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, type PropType } from 'vue'

export default defineComponent({
  name: 'NetworkConfig',
  props: {
    initialConfig: {
      type: Array as PropType<number[]>,
      default: () => [64, 64],
    },
  },
  emits: ['start'],
  data() {
    return {
      layers: [...this.initialConfig] as number[],
      nodeSizes: [16, 32, 64, 128],
      maxDisplayNodes: 6,
    }
  },
  computed: {
    layerCount(): number {
      return this.layers.length
    },
    architectureText(): string {
      return [6, ...this.layers, 2].join(' → ')
    },
    svgViewBox(): string {
      const width = 60 + (this.layers.length + 1) * 70
      return `0 0 ${width} 160`
    },
    layerLabels(): string[] {
      const labels = ['Input']
      for (let i = 0; i < this.layers.length; i++) {
        labels.push(`H${i + 1}`)
      }
      labels.push('Output')
      return labels
    },
    allLayers(): { total: number; displayNodes: number; truncated: boolean }[] {
      const result = []
      // Input layer (6 nodes)
      result.push({ total: 6, displayNodes: Math.min(6, this.maxDisplayNodes), truncated: 6 > this.maxDisplayNodes })
      // Hidden layers
      for (const size of this.layers) {
        result.push({
          total: size,
          displayNodes: Math.min(size, this.maxDisplayNodes),
          truncated: size > this.maxDisplayNodes,
        })
      }
      // Output layer (2 nodes)
      result.push({ total: 2, displayNodes: 2, truncated: false })
      return result
    },
    connections(): { x1: number; y1: number; x2: number; y2: number }[][] {
      const conns: { x1: number; y1: number; x2: number; y2: number }[][] = []
      for (let li = 0; li < this.allLayers.length - 1; li++) {
        const layerConns: { x1: number; y1: number; x2: number; y2: number }[] = []
        const fromLayer = this.allLayers[li]
        const toLayer = this.allLayers[li + 1]
        const fromX = this.layerX(li)
        const toX = this.layerX(li + 1)
        
        // Only draw a subset of connections to avoid clutter
        const fromNodes = Math.min(fromLayer.displayNodes, 4)
        const toNodes = Math.min(toLayer.displayNodes, 4)
        
        for (let fi = 0; fi < fromNodes; fi++) {
          for (let ti = 0; ti < toNodes; ti++) {
            layerConns.push({
              x1: fromX,
              y1: this.nodeY(fi, fromLayer.displayNodes),
              x2: toX,
              y2: this.nodeY(ti, toLayer.displayNodes),
            })
          }
        }
        conns.push(layerConns)
      }
      return conns
    },
    nodeRadius(): number {
      return 8
    },
  },
  watch: {
    initialConfig: {
      handler(newConfig: number[]) {
        this.layers = [...newConfig]
      },
      immediate: true,
    },
  },
  methods: {
    setLayerCount(count: number) {
      if (count > this.layers.length) {
        // Add layers with default 64 nodes
        while (this.layers.length < count) {
          this.layers.push(64)
        }
      } else if (count < this.layers.length) {
        // Remove layers
        this.layers = this.layers.slice(0, count)
      }
    },
    setLayerNodes(index: number, size: number) {
      this.layers[index] = size
    },
    layerX(index: number): number {
      return 40 + index * 70
    },
    nodeY(nodeIndex: number, totalNodes: number): number {
      const spacing = 120 / Math.max(totalNodes, 1)
      const startY = 30 + (120 - (totalNodes - 1) * spacing) / 2
      return startY + nodeIndex * spacing
    },
    layerColor(layerIndex: number): string {
      if (layerIndex === 0) return '#4a90a4' // Input - blue
      if (layerIndex === this.allLayers.length - 1) return '#00ff88' // Output - green
      return '#ff9f43' // Hidden - orange
    },
    startTraining() {
      this.$emit('start', [...this.layers])
    },
  },
})
</script>

<style scoped>
.network-config {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
  padding: var(--spacing-lg);
  background: var(--color-bg-dark);
  border-radius: var(--radius-lg);
  border: 1px solid var(--color-border);
  max-width: 500px;
  margin: 0 auto;
}

.config-header {
  text-align: center;
}

.config-title {
  font-family: var(--font-display);
  font-size: 1.5rem;
  color: var(--color-primary);
  margin: 0 0 var(--spacing-xs) 0;
}

.config-subtitle {
  color: var(--color-text-muted);
  font-size: 0.9rem;
  margin: 0;
}

.config-body {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.config-section {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.section-label {
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--color-text-muted);
}

.layer-count-buttons {
  display: flex;
  gap: var(--spacing-xs);
}

.layer-btn {
  flex: 1;
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--color-bg-light);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  color: var(--color-text);
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}

.layer-btn:hover {
  border-color: var(--color-primary);
}

.layer-btn.active {
  background: var(--color-primary);
  border-color: var(--color-primary);
  color: var(--color-bg-dark);
}

.layer-configs {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.layer-config {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.layer-label {
  width: 30px;
  font-size: 0.85rem;
  font-weight: 600;
  color: #ff9f43;
}

.node-buttons {
  display: flex;
  gap: var(--spacing-xs);
  flex: 1;
}

.node-btn {
  flex: 1;
  padding: var(--spacing-xs) var(--spacing-sm);
  background: var(--color-bg-light);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  color: var(--color-text);
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.node-btn:hover {
  border-color: var(--color-primary);
}

.node-btn.active {
  background: #ff9f43;
  border-color: #ff9f43;
  color: var(--color-bg-dark);
}

.network-preview {
  background: var(--color-bg-mid);
  border-radius: var(--radius-md);
  padding: var(--spacing-sm);
}

.preview-svg {
  width: 100%;
  height: 140px;
}

.layer-label-text {
  font-size: 10px;
  fill: var(--color-text-muted);
}

.truncation-text {
  font-size: 8px;
  fill: var(--color-text-muted);
}

.architecture-text {
  text-align: center;
  font-family: var(--font-mono);
  font-size: 0.9rem;
  color: var(--color-text);
  padding-top: var(--spacing-xs);
}

.config-footer {
  display: flex;
  justify-content: center;
}

.btn-large {
  padding: var(--spacing-md) var(--spacing-xl);
  font-size: 1.1rem;
}
</style>

