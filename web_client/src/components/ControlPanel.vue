<template>
  <div class="control-panel panel">
    <div class="panel-header">
      <span>Training Controls</span>
      <div class="header-actions">
        <button 
          class="btn-icon" 
          :class="{ active: isPaused }"
          @click="togglePause"
          :title="isPaused ? 'Resume' : 'Pause'"
        >
          {{ isPaused ? '▶' : '⏸' }}
        </button>
      </div>
    </div>

    <!-- Mode Switcher -->
    <div class="mode-switcher">
      <button 
        class="mode-btn" 
        :class="{ active: currentMode === 'training' }"
        @click="setMode('training')"
        :disabled="isPaused && currentMode !== 'training'"
      >
        🎓 Train
      </button>
      <button 
        class="mode-btn" 
        :class="{ active: currentMode === 'eval' }"
        @click="setMode('eval')"
      >
        🎯 Eval
      </button>
    </div>

    <div class="controls-body">
      <!-- Epsilon Control -->
      <div class="control-group">
        <label class="control-label" :title="epsilonTooltip">
          <span>Exploration (ε)</span>
          <span class="control-value">{{ epsilon.toFixed(3) }}</span>
        </label>
        <input
          type="range"
          class="form-range"
          :value="epsilon"
          min="0"
          max="1"
          step="0.01"
          @input="updateEpsilon"
          :disabled="currentMode !== 'training'"
        />
        <div class="control-hint">
          <label class="toggle-label small">
            <input
              type="checkbox"
              :checked="autoDecay"
              @change="toggleAutoDecay"
              :disabled="currentMode !== 'training'"
            />
            <span>Auto-decay</span>
          </label>
          <span class="hint-text" v-if="!autoDecay">Manual</span>
        </div>
      </div>

      <!-- Epsilon Decay Rate -->
      <div class="control-group" v-if="autoDecay">
        <label class="control-label" :title="decayRateTooltip">
          <span>Decay Rate</span>
          <span class="control-value">{{ formatDecaySteps(epsilonDecaySteps) }}</span>
        </label>
        <input
          type="range"
          class="form-range"
          :value="epsilonDecaySteps"
          min="100000"
          max="2000000"
          step="50000"
          @input="updateEpsilonDecaySteps"
          :disabled="currentMode !== 'training'"
        />
        <span class="hint-text">Steps to reach minimum ε</span>
      </div>

      <!-- Learning Rate -->
      <div class="control-group">
        <label class="control-label" :title="learningRateTooltip">
          <span>Learning Rate</span>
          <span class="control-value" :class="{ 'lr-auto': lrScheduler }">{{ formatLearningRate(learningRate) }}</span>
        </label>
        <input
          type="range"
          class="form-range"
          :value="learningRate"
          min="0.00001"
          max="0.002"
          step="0.00001"
          @input="updateLearningRate"
          :disabled="currentMode !== 'training' || lrScheduler"
        />
        <div class="control-hint">
          <label class="toggle-label small">
            <input
              type="checkbox"
              :checked="lrScheduler"
              @change="toggleLRScheduler"
              :disabled="currentMode !== 'training'"
            />
            <span>Auto-schedule</span>
          </label>
          <span class="hint-text" v-if="lrScheduler">Reduces on plateau</span>
          <span class="hint-text" v-else>Manual</span>
        </div>
      </div>

      <!-- Train Frequency -->
      <div class="control-group">
        <label class="control-label" :title="trainFreqTooltip">
          <span>Train Frequency</span>
          <span class="control-value">{{ trainFreq }}</span>
        </label>
        <input
          type="range"
          class="form-range"
          :value="trainFreq"
          min="1"
          max="32"
          step="1"
          @input="updateTrainFreq"
          :disabled="currentMode !== 'training'"
        />
        <span class="hint-text">{{ trainFreqHint }}</span>
      </div>

      <!-- Rewards Section -->
      <div class="control-section">
        <div class="section-header">Rewards</div>
        
        <!-- Pass Pipe Reward -->
        <div class="control-group compact">
          <label class="control-label" :title="passPipeTooltip">
            <span>Pass Pipe</span>
            <span class="control-value positive">+{{ passPipeReward.toFixed(1) }}</span>
          </label>
          <input
            type="range"
            class="form-range"
            :value="passPipeReward"
            min="0.1"
            max="5"
            step="0.1"
            @input="updatePassPipeReward"
          />
        </div>

        <!-- Death Penalty -->
        <div class="control-group compact">
          <label class="control-label" :title="deathPenaltyTooltip">
            <span>Death Penalty</span>
            <span class="control-value negative">{{ deathPenalty.toFixed(1) }}</span>
          </label>
          <input
            type="range"
            class="form-range"
            :value="Math.abs(deathPenalty)"
            min="0.1"
            max="5"
            step="0.1"
            @input="updateDeathPenalty"
          />
        </div>

        <!-- Step Penalty -->
        <div class="control-group compact">
          <label class="control-label" :title="stepPenaltyTooltip">
            <span>Step Cost</span>
            <span class="control-value negative">{{ stepPenalty.toFixed(3) }}</span>
          </label>
          <input
            type="range"
            class="form-range"
            :value="Math.abs(stepPenalty)"
            min="0"
            max="0.1"
            step="0.005"
            @input="updateStepPenalty"
          />
        </div>

        <!-- Center Reward (Dense Shaping) -->
        <div class="control-group compact">
          <label class="control-label" :title="centerBonusTooltip">
            <span>Center Bonus</span>
            <span class="control-value" :class="{ positive: centerReward > 0 }">
              {{ centerReward > 0 ? '+' : '' }}{{ centerReward.toFixed(2) }}
            </span>
          </label>
          <input
            type="range"
            class="form-range"
            :value="centerReward"
            min="0"
            max="1"
            step="0.05"
            @input="updateCenterReward"
          />
          <span class="control-hint-text">Reward for moving toward pipe gap</span>
        </div>
      </div>

      <!-- Fast Mode Toggle -->
      <div class="control-group">
        <label class="toggle-control" :title="fastModeTooltip">
          <input 
            type="checkbox" 
            :checked="fastMode"
            @change="toggleFastMode"
          />
          <span class="toggle-label">⚡ Fast Training</span>
          <span class="toggle-hint">(uses max CPU)</span>
        </label>
      </div>
    </div>

    <div class="control-actions">
      <button class="btn btn-danger btn-small" @click="resetTraining">
        🔄 Reset Training
      </button>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue'

export default defineComponent({
  name: 'ControlPanel',
  props: {
    epsilon: {
      type: Number,
      default: 1.0,
    },
    learningRate: {
      type: Number,
      default: 0.001,
    },
    passPipeReward: {
      type: Number,
      default: 1.0,
    },
    deathPenalty: {
      type: Number,
      default: -1.0,
    },
    stepPenalty: {
      type: Number,
      default: -0.01,
    },
    centerReward: {
      type: Number,
      default: 0.1,
    },
    fastMode: {
      type: Boolean,
      default: false,
    },
    autoDecay: {
      type: Boolean,
      default: true,
    },
    lrScheduler: {
      type: Boolean,
      default: false,
    },
    trainFreq: {
      type: Number,
      default: 8,
    },
    epsilonDecaySteps: {
      type: Number,
      default: 200000,
    },
    isPaused: {
      type: Boolean,
      default: false,
    },
    currentMode: {
      type: String as () => 'training' | 'eval' | 'manual',
      default: 'training',
    },
  },
  emits: [
    'update:epsilon',
    'update:learningRate',
    'update:epsilonDecaySteps',
    'update:passPipeReward',
    'update:deathPenalty',
    'update:stepPenalty',
    'update:centerReward',
    'update:fastMode',
    'update:autoDecay',
    'update:lrScheduler',
    'update:trainFreq',
    'update:isPaused',
    'update:mode',
    'reset',
  ],
  computed: {
    epsilonTooltip(): string {
      return `Exploration Rate (ε): Probability of taking a random action instead of using the learned policy.

• Higher values (0.5-1.0): More exploration, tries new strategies
• Lower values (0.05-0.2): More exploitation, uses learned policy
• Auto-decay gradually reduces exploration over time`
    },
    decayRateTooltip(): string {
      return `Epsilon Decay Rate: Number of training steps to reduce exploration from start to minimum.

Lower values = faster transition to exploitation. Higher values = more exploration time.`
    },
    learningRateTooltip(): string {
      return `Learning Rate: How much the neural network adjusts its weights on each training step.

• Higher values: Faster learning but may be unstable
• Lower values: Slower but more stable learning
• Auto-schedule reduces LR when training plateaus`
    },
    trainFreqTooltip(): string {
      return `Train Frequency: How often the neural network learns from experiences.

• Lower values (1-4): More frequent training, slower but better learning quality
• Medium values (8-16): Good balance between speed and quality
• Higher values (16-32): Faster training speed, may need more episodes to learn

The network updates its weights every N game steps, where N is this value.`
    },
    passPipeTooltip(): string {
      return `Pass Pipe Reward: Positive reward given when the bird successfully passes through a pipe gap.

Higher values encourage the agent to prioritize passing pipes.`
    },
    deathPenaltyTooltip(): string {
      return `Death Penalty: Negative reward given when the bird collides with a pipe or the ground.

More negative values make the agent more cautious about avoiding collisions.`
    },
    stepPenaltyTooltip(): string {
      return `Step Cost: Small negative reward given every game step to encourage efficiency.

Encourages the agent to complete episodes quickly rather than stalling.`
    },
    centerBonusTooltip(): string {
      return `Center Bonus: Dense reward shaping that gives small rewards for moving toward the pipe gap center.

Helps guide the agent during early training by rewarding progress toward the goal.`
    },
    fastModeTooltip(): string {
      return `Fast Training: Disables game visualization and runs training at maximum speed using all available CPU.

Useful for long training sessions. The game won't be visible but training will be much faster.`
    },
    trainFreqHint(): string {
      if (this.trainFreq <= 4) return 'High quality, slower'
      if (this.trainFreq <= 12) return 'Balanced'
      return 'Faster, may need more episodes'
    },
  },
  methods: {
    updateEpsilon(event: Event) {
      const value = parseFloat((event.target as HTMLInputElement).value)
      this.$emit('update:epsilon', value)
    },
    updateLearningRate(event: Event) {
      const raw = parseFloat((event.target as HTMLInputElement).value)
      const lr = Math.max(0.00001, Math.min(0.002, raw))
      this.$emit('update:learningRate', lr)
    },
    updatePassPipeReward(event: Event) {
      const value = parseFloat((event.target as HTMLInputElement).value)
      this.$emit('update:passPipeReward', value)
    },
    updateDeathPenalty(event: Event) {
      const value = -parseFloat((event.target as HTMLInputElement).value)
      this.$emit('update:deathPenalty', value)
    },
    updateStepPenalty(event: Event) {
      const value = -parseFloat((event.target as HTMLInputElement).value)
      this.$emit('update:stepPenalty', value)
    },
    updateCenterReward(event: Event) {
      const value = parseFloat((event.target as HTMLInputElement).value)
      this.$emit('update:centerReward', value)
    },
    toggleFastMode(event: Event) {
      const checked = (event.target as HTMLInputElement).checked
      this.$emit('update:fastMode', checked)
    },
    toggleAutoDecay(event: Event) {
      const checked = (event.target as HTMLInputElement).checked
      this.$emit('update:autoDecay', checked)
    },
    toggleLRScheduler(event: Event) {
      const checked = (event.target as HTMLInputElement).checked
      this.$emit('update:lrScheduler', checked)
    },
    updateTrainFreq(event: Event) {
      const value = parseInt((event.target as HTMLInputElement).value)
      this.$emit('update:trainFreq', value)
    },
    updateEpsilonDecaySteps(event: Event) {
      const value = parseInt((event.target as HTMLInputElement).value)
      this.$emit('update:epsilonDecaySteps', value)
    },
    formatDecaySteps(steps: number): string {
      if (steps >= 1000000) {
        return `${(steps / 1000000).toFixed(1)}M`
      }
      return `${(steps / 1000).toFixed(0)}K`
    },
    formatLearningRate(lr: number): string {
      return lr.toExponential(1)
    },
    togglePause() {
      this.$emit('update:isPaused', !this.isPaused)
    },
    setMode(mode: 'training' | 'eval' | 'manual') {
      this.$emit('update:mode', mode)
    },
    resetTraining() {
      if (confirm('Reset all training progress? This cannot be undone.')) {
        this.$emit('reset')
      }
    },
  },
})
</script>

<style scoped>
.control-panel {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-actions {
  display: flex;
  gap: var(--spacing-xs);
}

.btn-icon {
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--color-bg-light);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  color: var(--color-text-muted);
  cursor: pointer;
  font-size: 0.9rem;
  transition: all 0.2s ease;
}

.btn-icon:hover {
  background: var(--color-bg-mid);
  color: var(--color-text);
}

.btn-icon.active {
  background: var(--color-primary);
  border-color: var(--color-primary);
  color: var(--color-bg-dark);
}

.mode-switcher {
  display: flex;
  gap: 4px;
  background: var(--color-bg-light);
  padding: 4px;
  border-radius: var(--radius-md);
}

.mode-btn {
  flex: 1;
  padding: var(--spacing-xs) var(--spacing-sm);
  background: transparent;
  border: none;
  border-radius: var(--radius-sm);
  color: var(--color-text-muted);
  font-size: 0.75rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.mode-btn:hover:not(:disabled) {
  color: var(--color-text);
  background: var(--color-bg-mid);
}

.mode-btn.active {
  background: var(--color-primary);
  color: var(--color-bg-dark);
}

.mode-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.controls-body {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.control-section {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.section-header {
  font-family: var(--font-display);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--color-text-muted);
  padding-bottom: 4px;
  border-bottom: 1px solid var(--color-border);
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.control-group.compact {
  gap: 4px;
}

.control-group.compact .control-label {
  font-size: 0.8rem;
}

.control-group.compact .form-range {
  height: 4px;
}

.toggle-control {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  cursor: pointer;
  padding: var(--spacing-sm);
  background: var(--color-bg-light);
  border-radius: var(--radius-sm);
  border: 1px solid var(--color-border);
}

.toggle-control input[type="checkbox"] {
  width: 18px;
  height: 18px;
  accent-color: var(--color-primary);
}

.toggle-label {
  font-family: var(--font-display);
  font-size: 0.85rem;
  color: var(--color-text);
}

.toggle-hint {
  font-size: 0.7rem;
  color: var(--color-text-muted);
}

.control-label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.85rem;
  color: var(--color-text-muted);
}

.control-value {
  font-family: var(--font-display);
  font-size: 0.8rem;
  color: var(--color-primary);
  background: rgba(0, 217, 255, 0.1);
  padding: 2px 6px;
  border-radius: var(--radius-sm);
}

.control-value.speed-fast {
  color: var(--color-accent);
  background: rgba(255, 215, 0, 0.1);
}

.control-value.lr-auto {
  color: var(--color-accent);
  background: rgba(255, 215, 0, 0.15);
  animation: lr-pulse 2s ease-in-out infinite;
}

@keyframes lr-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

.control-value.positive {
  color: var(--color-success);
  background: rgba(0, 255, 136, 0.1);
}

.control-value.negative {
  color: var(--color-danger);
  background: rgba(255, 107, 157, 0.1);
}

.control-hint-text {
  font-size: 0.65rem;
  color: var(--color-text-muted);
  font-style: italic;
}

.control-hint {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.hint-text {
  font-size: 0.7rem;
  color: var(--color-accent);
  text-transform: uppercase;
}

.toggle-label.small {
  font-size: 0.75rem;
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  cursor: pointer;
}

.toggle-label.small input {
  width: 14px;
  height: 14px;
  accent-color: var(--color-primary);
}

.control-presets {
  display: flex;
  gap: 4px;
  margin-top: 4px;
}

.preset-btn {
  flex: 1;
  padding: 4px 8px;
  background: var(--color-bg-light);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  color: var(--color-text-muted);
  font-size: 0.7rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.preset-btn:hover {
  background: var(--color-bg-mid);
  color: var(--color-text);
  border-color: var(--color-primary);
}

.control-actions {
  margin-top: var(--spacing-sm);
  padding-top: var(--spacing-md);
  border-top: 1px solid var(--color-border);
}

.btn-danger {
  background: rgba(255, 82, 82, 0.2);
  border-color: #ff5252;
  color: #ff5252;
}

.btn-danger:hover {
  background: #ff5252;
  color: var(--color-bg-dark);
}

.btn-small {
  padding: var(--spacing-xs) var(--spacing-sm);
  font-size: 0.8rem;
  width: 100%;
}

.form-range:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
