<template>
  <div class="app-container">
    <header class="app-header">
      <h1 class="app-title text-display">
        <span class="text-primary glow">FLAPPY</span>
        <span class="text-accent">AI</span>
      </h1>
      <div class="header-right">
        <div class="mode-indicator">
          <span v-if="mode === 'idle'" class="badge badge-idle">Ready</span>
          <span v-else-if="mode === 'configuring'" class="badge badge-configuring">Configuring</span>
          <span
            v-else-if="mode === 'training'"
            class="badge badge-training"
            :class="{ 'animate-pulse': !isPaused && !isAutoEval }"
          >
            {{ isAutoEval ? `Auto-eval ${autoEvalTrial}/${autoEvalTrials}` : (isPaused ? 'Paused' : 'Training') }}
          </span>
          <span v-else-if="mode === 'eval'" class="badge badge-eval">Evaluating</span>
          <span v-else-if="mode === 'manual'" class="badge badge-manual">Manual Control</span>
        </div>

        <!-- Checkpoint controls (top of page, training only) -->
        <div v-if="mode === 'training'" class="checkpoint-controls">
          <button class="btn btn-secondary btn-small" @click="saveCheckpoint">
            💾 Save Checkpoint
          </button>
          <button class="btn btn-secondary btn-small" @click="triggerCheckpointLoad">
            📂 Load Checkpoint
          </button>
          <input
            ref="checkpointInput"
            type="file"
            accept="application/json"
            class="checkpoint-input"
            @change="onCheckpointFileSelected"
          />
        </div>
      </div>
    </header>

    <main class="app-main">
      <!-- Left Panel: Controls + Neural Network -->
      <aside class="left-panel" v-if="mode === 'configuring' || mode === 'training' || mode === 'eval'">
        <!-- Training controls shown during configuring and training -->
        <ControlPanel
          v-if="mode === 'configuring' || mode === 'training'"
          :epsilon="epsilon"
          :learningRate="learningRate"
          :epsilonDecaySteps="epsilonDecaySteps"
          :passPipeReward="passPipeReward"
          :deathPenalty="deathPenalty"
          :stepPenalty="stepPenalty"
          :centerReward="centerReward"
          :fastMode="fastMode"
          :autoDecay="autoDecay"
          :lrScheduler="lrScheduler"
          :trainFreq="trainFreq"
          :isPaused="isPaused"
          :currentMode="mode"
          @update:epsilon="updateEpsilon"
          @update:learningRate="updateLearningRate"
          @update:epsilonDecaySteps="updateEpsilonDecaySteps"
          @update:passPipeReward="updatePassPipeReward"
          @update:deathPenalty="updateDeathPenalty"
          @update:stepPenalty="updateStepPenalty"
          @update:centerReward="updateCenterReward"
          @update:fastMode="updateFastMode"
          @update:autoDecay="updateAutoDecay"
          @update:lrScheduler="updateLRScheduler"
          @update:trainFreq="updateTrainFreq"
          @update:isPaused="updatePaused"
          @update:mode="changeMode"
          @reset="resetTraining"
        />

        <!-- Eval mode info -->
        <div v-if="mode === 'eval'" class="eval-info panel">
          <div class="panel-header">
            <span>Evaluation Mode</span>
            <button 
              class="btn-icon-eval" 
              :class="{ active: isPaused }"
              @click="togglePause"
              :title="isPaused ? 'Resume' : 'Pause'"
            >
              {{ isPaused ? '▶' : '⏸' }}
            </button>
          </div>
          <p class="eval-text">Running fully greedy (ε=0)</p>
          <p class="eval-text">No exploration, no training</p>
          <p class="eval-status" v-if="isPaused">⏸ Paused</p>
          <button class="btn btn-secondary" @click="backToTraining">
            ← Back to Training
          </button>
        </div>

        <!-- Neural network visualization (hidden during fast training) -->
        <NetworkViewer
          v-if="!fastMode"
          :activations="networkActivations"
          :qValues="qValues"
          :selectedAction="selectedAction"
          :hiddenLayers="hiddenLayersConfig"
        />
        <div v-else class="fast-mode-nn-placeholder panel">
          <div class="panel-header">Neural Network</div>
          <div class="placeholder-content">
            <span class="placeholder-icon">⚡</span>
            <span class="placeholder-text">Visualization paused during fast training</span>
          </div>
        </div>
      </aside>

      <div class="game-area">
        <GameCanvas
          ref="gameCanvas"
          :mode="mode"
          :fastMode="fastMode"
          :isPaused="isPaused"
          :hiddenLayersConfig="hiddenLayersConfig"
          @score-update="handleScoreUpdate"
          @episode-end="handleEpisodeEnd"
          @metrics-update="handleMetricsUpdate"
          @network-update="handleNetworkUpdate"
          @auto-eval-result="handleAutoEvalResult"
          @architecture-loaded="handleArchitectureLoaded"
        />
        <div v-if="mode === 'idle'" class="game-overlay">
          <div class="overlay-content idle-content">
            <button class="btn btn-primary btn-hero" @click="openTrainingConfig">
              🧠 Train AI
            </button>
            <button class="btn btn-text btn-small" @click="startManualPlay">
              🎮 Play manually
            </button>
          </div>
        </div>
        
        <!-- Network Config Overlay -->
        <div v-if="mode === 'configuring'" class="game-overlay config-overlay">
          <NetworkConfig
            :initialConfig="hiddenLayersConfig"
            @start="startTrainingWithConfig"
          />
        </div>
        
        <!-- Game Over Overlay -->
        <div v-if="showGameOver" class="game-overlay game-over-overlay">
          <div class="overlay-content">
            <h2 class="game-over-title">Game Over</h2>
            <p class="game-over-score">Score: <span class="text-accent">{{ lastGameScore }}</span></p>
            <p class="game-over-best" v-if="lastGameScore === bestScore && bestScore > 0">🏆 New Best!</p>
            <div class="start-buttons">
              <button class="btn btn-primary" @click="restartGame">
                {{ mode === 'manual' ? '🎮 Play Again' : '🔄 Continue Training' }}
              </button>
              <button class="btn btn-secondary" @click="backToMenu">
                🏠 Menu
              </button>
            </div>
          </div>
        </div>
      </div>

      <aside class="sidebar">
        <StatusBar
          :mode="mode"
          :episode="episode"
          :lastScore="lastGameScore"
          :bestScore="bestScore"
          :stepsPerSecond="stepsPerSecond"
          :autoEvalHistory="autoEvalHistory"
          :canSubmitToLeaderboard="canSubmitToLeaderboard"
          @submit-score="showLeaderboard = true"
        />

        <div class="sidebar-panels">
          <!-- Show leaderboard panel when idle -->
          <LeaderboardPanel v-if="mode === 'idle'" />
          
          <!-- Show metrics panel when active -->
          <MetricsPanel
            v-else
            :epsilon="effectiveEpsilon"
            :avgReward="avgReward"
            :episodeReward="episodeReward"
            :loss="loss"
            :episode="episode"
            :bestScore="bestScore"
            :stepsPerSecond="stepsPerSecond"
            :bufferSize="bufferSize"
            :totalSteps="totalSteps"
            :avgLength="avgLength"
            :isTraining="mode === 'training' && !isPaused"
            :isWarmup="isWarmup"
          />
        </div>
      </aside>
    </main>

    <footer class="app-footer">
      <p class="disclaimer text-muted">
        ⚠️ Settings reset when you close this tab. 
        <a href="#" class="text-primary" @click.prevent="showLeaderboard = true">View Leaderboard</a>
      </p>
    </footer>

    <!-- Leaderboard Modal -->
    <Leaderboard
      :isOpen="showLeaderboard"
      :canSubmit="canSubmitScore"
      :pendingScore="bestScore"
      :pendingParams="networkParams"
      :pendingArchitecture="networkArchitecture"
      @close="showLeaderboard = false"
      @challenge="handleChallengeChampion"
      @submit="handleScoreSubmitted"
    />
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue'
import GameCanvas from './components/GameCanvas.vue'
import StatusBar from './components/StatusBar.vue'
import MetricsPanel from './components/MetricsPanel.vue'
import ControlPanel from './components/ControlPanel.vue'
import NetworkViewer from './components/NetworkViewer.vue'
import NetworkConfig from './components/NetworkConfig.vue'
import Leaderboard from './components/Leaderboard.vue'
import LeaderboardPanel from './components/LeaderboardPanel.vue'
import { apiClient, calculateAdjustedScore } from './services/apiClient'

export type GameMode = 'idle' | 'configuring' | 'training' | 'eval' | 'manual'

export default defineComponent({
  name: 'App',
  components: {
    GameCanvas,
    StatusBar,
    MetricsPanel,
    ControlPanel,
    NetworkViewer,
    NetworkConfig,
    Leaderboard,
    LeaderboardPanel,
  },
  data() {
    return {
      mode: 'idle' as GameMode,
      episode: 0,
      score: 0,
      bestScore: 0,
      stepsPerSecond: 0,
      epsilon: 1.0,
      learningRate: 0.001,
      passPipeReward: 1.0,
      deathPenalty: -1.0,
      stepPenalty: -0.01,
      centerReward: 0.1,
      fastMode: false,
      autoDecay: true,
      lrScheduler: false,
      trainFreq: 8,
      epsilonDecaySteps: 200000,
      avgReward: 0,
      episodeReward: 0,
      loss: 0,
      bufferSize: 0,
      totalSteps: 0,
      avgLength: 0,
      isWarmup: true,
      isAutoEval: false,
      autoEvalTrial: 0,
      autoEvalTrials: 100,
      lastAutoEvalResult: null as { avgScore: number; maxScore: number; minScore: number; scores: number[]; episode: number } | null,
      autoEvalHistory: [] as { avgScore: number; maxScore: number; minScore: number; scores: number[]; episode: number }[],
      networkActivations: [] as number[][],
      qValues: [0, 0] as [number, number],
      selectedAction: 0,
      isPaused: false,
      showGameOver: false,
      lastGameScore: 0,
      showLeaderboard: false,
      hiddenLayersConfig: [64, 64] as number[],
      lowestLeaderboardScore: 0,
    }
  },
  computed: {
    canSubmitScore(): boolean {
      // Can submit if we have a best score and just finished an eval
      return this.bestScore > 0 && (this.mode === 'eval' || this.showGameOver)
    },
    // Epsilon actually used by the policy: training uses slider epsilon, eval is fully greedy (ε=0)
    effectiveEpsilon(): number {
      return this.mode === 'eval' ? 0 : this.epsilon
    },
    // Calculate total network parameters for leaderboard submission
    networkParams(): number {
      const inputDim = 6
      const outputDim = 2
      const layers = [inputDim, ...this.hiddenLayersConfig, outputDim]
      let totalParams = 0
      for (let i = 0; i < layers.length - 1; i++) {
        // weights + biases for each layer
        totalParams += layers[i] * layers[i + 1] + layers[i + 1]
      }
      return totalParams
    },
    // Network architecture string for leaderboard display
    networkArchitecture(): string {
      const layers = [6, ...this.hiddenLayersConfig, 2]
      return layers.join('→')
    },
    // Check if current best score qualifies for leaderboard
    canSubmitToLeaderboard(): boolean {
      if (this.bestScore <= 0) return false
      const adjustedScore = calculateAdjustedScore(this.bestScore, this.networkParams)
      return adjustedScore > this.lowestLeaderboardScore
    },
  },
  async mounted() {
    // Load leaderboard threshold on startup
    await this.refreshLeaderboardThreshold()
  },
  methods: {
    async refreshLeaderboardThreshold() {
      this.lowestLeaderboardScore = await apiClient.getLowestScore()
    },
    startManualPlay() {
      this.mode = 'manual'
      this.showGameOver = false
      const gameCanvas = this.$refs.gameCanvas as InstanceType<typeof GameCanvas>
      if (gameCanvas) {
        gameCanvas.startGame()
      }
    },
    openTrainingConfig() {
      // Show config overlay - user must click "Start Training" to begin
      this.mode = 'configuring'
      this.showGameOver = false
      this.isPaused = false
    },
    startTrainingWithConfig(hiddenLayers: number[]) {
      this.hiddenLayersConfig = hiddenLayers
      this.mode = 'training'
      const gameCanvas = this.$refs.gameCanvas as InstanceType<typeof GameCanvas>
      if (gameCanvas) {
        gameCanvas.startTraining(hiddenLayers)
      }
    },
    handleScoreUpdate(newScore: number) {
      this.score = newScore
      if (newScore > this.bestScore) {
        this.bestScore = newScore
      }
    },
    handleEpisodeEnd(stats: { score: number; reward: number }) {
      this.lastGameScore = stats.score
      // Store the final episode reward (for charting)
      this.episodeReward = stats.reward
      
      // In manual mode, show game over screen
      if (this.mode === 'manual') {
        this.showGameOver = true
      }
      // For training, auto-continue (no game over screen)
    },
    handleMetricsUpdate(metrics: {
      episode: number
      avgReward: number
      episodeReward: number
      epsilon: number
      loss: number
      stepsPerSecond: number
      bufferSize: number
      totalSteps: number
      avgLength: number
      isWarmup?: boolean
      learningRate?: number
      isAutoEval?: boolean
      autoEvalTrial?: number
      autoEvalTrials?: number
    }) {
      this.episode = metrics.episode
      this.avgReward = metrics.avgReward
      // In fast mode, episodeReward comes from metrics (worker reports last completed episode)
      // In normal mode, it's set by handleEpisodeEnd
      if (this.fastMode) {
        this.episodeReward = metrics.episodeReward
      }
      this.epsilon = metrics.epsilon
      this.loss = metrics.loss
      this.stepsPerSecond = metrics.stepsPerSecond
      this.bufferSize = metrics.bufferSize
      this.totalSteps = metrics.totalSteps
      this.avgLength = metrics.avgLength
      this.isWarmup = metrics.isWarmup ?? false
      // Update learning rate from metrics (may change with scheduler)
      if (metrics.learningRate !== undefined) {
        this.learningRate = metrics.learningRate
      }
      // Update auto-eval state
      this.isAutoEval = metrics.isAutoEval ?? false
      this.autoEvalTrial = metrics.autoEvalTrial ?? 0
      this.autoEvalTrials = metrics.autoEvalTrials ?? 100
    },
    handleAutoEvalResult(result: { avgScore: number; maxScore: number; minScore: number; scores: number[]; episode: number }) {
      console.log('[App] Auto-eval result:', result)
      this.lastAutoEvalResult = result
      // Add to history (keep last 10)
      this.autoEvalHistory.push(result)
      if (this.autoEvalHistory.length > 10) {
        this.autoEvalHistory.shift()
      }
      // Update best score if auto-eval found a better one
      if (result.maxScore > this.bestScore) {
        this.bestScore = result.maxScore
      }
    },
    handleArchitectureLoaded(hiddenLayers: number[]) {
      // Called when a checkpoint with architecture info is loaded
      this.hiddenLayersConfig = hiddenLayers
      // Go to configuring mode so user can review and start training
      this.mode = 'configuring'
    },
    handleNetworkUpdate(viz: { activations: number[][]; qValues: number[]; selectedAction: number }) {
      this.networkActivations = viz.activations
      if (viz.qValues && viz.qValues.length === 2) {
        this.qValues = [viz.qValues[0], viz.qValues[1]] as [number, number]
        this.selectedAction = viz.selectedAction
      }
      
      // Store network data in localStorage for the detail window
      try {
        localStorage.setItem('flappy-ai-network-data', JSON.stringify({
          activations: this.networkActivations,
          qValues: this.qValues,
          selectedAction: this.selectedAction,
        }))
      } catch (e) {
        // Ignore storage errors
      }
    },
    updateEpsilon(value: number) {
      this.epsilon = value
      this.autoDecay = false
      const gameCanvas = this.$refs.gameCanvas as InstanceType<typeof GameCanvas>
      if (gameCanvas) {
        gameCanvas.setEpsilon(value)
        gameCanvas.setAutoDecay(false)
      }
    },
    updateLearningRate(value: number) {
      this.learningRate = value
      const gameCanvas = this.$refs.gameCanvas as InstanceType<typeof GameCanvas>
      if (gameCanvas && (gameCanvas as any).setLearningRate) {
        ;(gameCanvas as any).setLearningRate(value)
      }
    },
    updatePassPipeReward(value: number) {
      this.passPipeReward = value
      const gameCanvas = this.$refs.gameCanvas as InstanceType<typeof GameCanvas>
      if (gameCanvas) {
        gameCanvas.setRewardConfig({ passPipe: value })
      }
    },
    updateDeathPenalty(value: number) {
      this.deathPenalty = value
      const gameCanvas = this.$refs.gameCanvas as InstanceType<typeof GameCanvas>
      if (gameCanvas) {
        gameCanvas.setRewardConfig({ deathPenalty: value })
      }
    },
    updateStepPenalty(value: number) {
      this.stepPenalty = value
      const gameCanvas = this.$refs.gameCanvas as InstanceType<typeof GameCanvas>
      if (gameCanvas) {
        gameCanvas.setRewardConfig({ stepPenalty: value })
      }
    },
    updateCenterReward(value: number) {
      this.centerReward = value
      const gameCanvas = this.$refs.gameCanvas as InstanceType<typeof GameCanvas>
      if (gameCanvas) {
        gameCanvas.setRewardConfig({ centerReward: value })
      }
    },
    updateFastMode(value: boolean) {
      this.fastMode = value
    },
    updateAutoDecay(value: boolean) {
      this.autoDecay = value
      const gameCanvas = this.$refs.gameCanvas as InstanceType<typeof GameCanvas>
      if (gameCanvas) {
        gameCanvas.setAutoDecay(value)
      }
    },
    updateLRScheduler(value: boolean) {
      this.lrScheduler = value
      const gameCanvas = this.$refs.gameCanvas as InstanceType<typeof GameCanvas>
      if (gameCanvas && (gameCanvas as any).setLRScheduler) {
        ;(gameCanvas as any).setLRScheduler(value)
      }
    },
    updateEpsilonDecaySteps(value: number) {
      this.epsilonDecaySteps = value
      const gameCanvas = this.$refs.gameCanvas as InstanceType<typeof GameCanvas>
      if (gameCanvas) {
        gameCanvas.setEpsilonDecaySteps(value)
      }
    },
    updateTrainFreq(value: number) {
      this.trainFreq = value
      const gameCanvas = this.$refs.gameCanvas as InstanceType<typeof GameCanvas>
      if (gameCanvas && (gameCanvas as any).setTrainFreq) {
        ;(gameCanvas as any).setTrainFreq(value)
      }
    },
    saveCheckpoint() {
      const gameCanvas = this.$refs.gameCanvas as InstanceType<typeof GameCanvas>
      if (gameCanvas && (gameCanvas as any).saveCheckpointToFile) {
        ;(gameCanvas as any).saveCheckpointToFile()
      }
    },
    triggerCheckpointLoad() {
      const input = this.$refs.checkpointInput as HTMLInputElement | undefined
      if (input) {
        input.value = ''
        input.click()
      }
    },
    onCheckpointFileSelected(event: Event) {
      const input = event.target as HTMLInputElement
      const file = input.files?.[0]
      if (!file) return

      const reader = new FileReader()
      reader.onload = () => {
        const text = typeof reader.result === 'string' ? reader.result : ''
        const gameCanvas = this.$refs.gameCanvas as InstanceType<typeof GameCanvas>
        if (text && gameCanvas && (gameCanvas as any).loadCheckpointFromJSON) {
          ;(gameCanvas as any).loadCheckpointFromJSON(text)
        }
      }
      reader.readAsText(file)
    },
    updatePaused(value: boolean) {
      this.isPaused = value
      const gameCanvas = this.$refs.gameCanvas as InstanceType<typeof GameCanvas>
      if (gameCanvas) {
        gameCanvas.setPaused(value)
      }
    },
    togglePause() {
      this.updatePaused(!this.isPaused)
    },
    changeMode(newMode: GameMode) {
      if (newMode === this.mode) return
      
      const gameCanvas = this.$refs.gameCanvas as InstanceType<typeof GameCanvas>
      
      if (newMode === 'eval') {
        // Switch to eval mode - fully greedy policy (ε=0), normal speed
        this.mode = 'eval'
        this.epsilon = 0  // Force epsilon to 0 for evaluation (purely greedy)
        this.fastMode = false
        this.isPaused = false
        this.showGameOver = false
        // Set epsilon on agent as well
        if (gameCanvas) {
          gameCanvas.setEpsilon(0)
          gameCanvas.startEval()
        }
      } else if (newMode === 'training') {
        // Resume training
        this.mode = 'training'
        this.isPaused = false
        this.showGameOver = false
        if (gameCanvas) {
          gameCanvas.startTraining()
        }
      } else if (newMode === 'manual') {
        // Switch to manual mode
        this.mode = 'manual'
        this.showGameOver = false
        if (gameCanvas) {
          gameCanvas.startGame()
        }
      }
    },
    resetTraining() {
      const gameCanvas = this.$refs.gameCanvas as InstanceType<typeof GameCanvas>
      if (gameCanvas) {
        gameCanvas.resetTraining()
      }
      // Reset all state
      this.episode = 0
      this.score = 0
      this.bestScore = 0
      this.avgReward = 0
      this.episodeReward = 0
      this.loss = 0
      this.bufferSize = 0
      this.totalSteps = 0
      this.avgLength = 0
      this.epsilon = 1.0
      this.autoDecay = true
      this.isWarmup = true
      this.isAutoEval = false
      this.autoEvalTrial = 0
      this.lastAutoEvalResult = null
      this.autoEvalHistory = []
      this.isPaused = false
      this.showGameOver = false
      // Go to configuring mode so user can adjust network config and click "Start Training"
      this.mode = 'configuring'
    },
    restartGame() {
      this.showGameOver = false
      this.score = 0
      
      const gameCanvas = this.$refs.gameCanvas as InstanceType<typeof GameCanvas>
      if (this.mode === 'manual') {
        if (gameCanvas) gameCanvas.startGame()
      } else {
        if (gameCanvas) gameCanvas.startTraining()
      }
    },
    backToMenu() {
      this.showGameOver = false
      this.mode = 'idle'
      this.score = 0
      
      const gameCanvas = this.$refs.gameCanvas as InstanceType<typeof GameCanvas>
      if (gameCanvas) {
        gameCanvas.stopGame()
      }
    },
    backToTraining() {
      this.showGameOver = false
      this.mode = 'training'
      this.isPaused = false
      const gameCanvas = this.$refs.gameCanvas as InstanceType<typeof GameCanvas>
      if (gameCanvas) {
        gameCanvas.startTraining()
      }
    },
    handleChallengeChampion() {
      this.showLeaderboard = false
      this.openTrainingConfig()
    },
    async handleScoreSubmitted(result: { entry: { name: string; score: number }; isNewChampion: boolean }) {
      if (result.isNewChampion) {
        // Show celebration message
        console.log('🎉 New champion:', result.entry.name, 'with score', result.entry.score)
      }
      // Refresh the leaderboard threshold
      await this.refreshLeaderboardThreshold()
    },
  },
})
</script>

<style scoped>
.app-container {
  height: 100%;
  display: flex;
  flex-direction: column;
  background: linear-gradient(180deg, var(--color-bg-dark) 0%, var(--color-bg-mid) 100%);
}

.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-md) var(--spacing-xl);
  border-bottom: 1px solid var(--color-border);
}

.app-title {
  font-size: 1.25rem;
  display: flex;
  gap: var(--spacing-sm);
}

.header-right {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
}

.mode-indicator {
  display: flex;
  align-items: center;
}

.checkpoint-controls {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
}

.checkpoint-input {
  display: none;
}

.badge {
  padding: var(--spacing-xs) var(--spacing-md);
  border-radius: var(--radius-xl);
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.badge-idle {
  background: var(--color-bg-light);
  color: var(--color-text-muted);
}

.badge-training {
  background: linear-gradient(135deg, var(--color-primary), #0099cc);
  color: var(--color-bg-dark);
}

.badge-configuring {
  background: linear-gradient(135deg, #ff9f43, #ff6b35);
  color: var(--color-bg-dark);
}

.badge-eval {
  background: var(--color-success);
  color: var(--color-bg-dark);
}

.badge-manual {
  background: var(--color-accent);
  color: var(--color-bg-dark);
}

.app-main {
  flex: 1;
  display: flex;
  gap: var(--spacing-lg);
  padding: var(--spacing-lg);
  overflow: hidden;
}

.left-panel {
  width: 400px;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  overflow-y: auto;
  max-height: 100%;
}

.fast-mode-nn-placeholder {
  min-height: 200px;
  display: flex;
  flex-direction: column;
}

.fast-mode-nn-placeholder .placeholder-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-sm);
  color: var(--color-text-muted);
}

.fast-mode-nn-placeholder .placeholder-icon {
  font-size: 2rem;
  animation: pulse 1.5s ease-in-out infinite;
}

.fast-mode-nn-placeholder .placeholder-text {
  font-size: 0.8rem;
  text-align: center;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.game-area {
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: center;
  position: relative;
  min-width: 0;
}

.game-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  background: rgba(15, 15, 35, 0.85);
  backdrop-filter: blur(4px);
  z-index: 10;
}

.config-overlay {
  overflow-y: auto;
  padding: var(--spacing-md);
}

.game-over-overlay {
  background: rgba(15, 15, 35, 0.9);
}

.overlay-content {
  text-align: center;
}

.idle-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-lg);
}

.btn-hero {
  padding: var(--spacing-lg) var(--spacing-xl);
  font-size: 1.4rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  border-radius: var(--radius-lg);
  box-shadow: 0 4px 20px rgba(0, 200, 255, 0.3);
  transition: all 0.3s ease;
}

.btn-hero:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 30px rgba(0, 200, 255, 0.5);
}

.btn-text {
  background: transparent;
  border: none;
  color: var(--color-text-muted);
  font-size: 0.9rem;
  padding: var(--spacing-sm) var(--spacing-md);
  cursor: pointer;
  transition: color 0.2s ease;
}

.btn-text:hover {
  color: var(--color-text);
}

.game-over-title {
  font-size: 2rem;
  color: var(--color-text);
  margin-bottom: var(--spacing-md);
}

.game-over-score {
  font-size: 1.2rem;
  color: var(--color-text-muted);
  margin-bottom: var(--spacing-sm);
}

.game-over-best {
  font-size: 1rem;
  color: var(--color-accent);
  margin-bottom: var(--spacing-lg);
  animation: bounce 0.5s ease-in-out;
}

@keyframes bounce {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.2); }
}

.instructions {
  margin-bottom: var(--spacing-lg);
  font-size: 0.9rem;
}

.instructions kbd {
  background: var(--color-bg-light);
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-sm);
  border: 1px solid var(--color-border);
  font-family: var(--font-body);
}

.start-buttons {
  display: flex;
  gap: var(--spacing-md);
  justify-content: center;
}

.sidebar {
  width: 450px;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  overflow-y: auto;
}

.sidebar-panels {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  overflow-y: auto;
}

.sidebar-footer {
  display: flex;
  gap: var(--spacing-lg);
  padding-top: var(--spacing-md);
  border-top: 1px solid var(--color-border);
}

.toggle-label {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  font-size: 0.85rem;
  color: var(--color-text-muted);
  cursor: pointer;
}

.toggle-label input {
  accent-color: var(--color-primary);
}

.app-footer {
  padding: var(--spacing-sm) var(--spacing-xl);
  border-top: 1px solid var(--color-border);
  text-align: center;
}

.disclaimer {
  font-size: 0.75rem;
}

.disclaimer a {
  text-decoration: none;
}

.disclaimer a:hover {
  text-decoration: underline;
}

/* Tablet responsive */
@media (max-width: 1200px) {
  .left-panel {
    width: 350px;
  }
  
  .sidebar {
    width: 320px;
  }
}

@media (max-width: 1024px) {
  .left-panel {
    width: 320px;
  }
  
  .sidebar {
    width: 280px;
  }
}

/* Mobile responsive */
@media (max-width: 768px) {
  .app-main {
    flex-direction: column;
  }

  .left-panel {
    display: none;
  }

  .sidebar {
    width: 100%;
    max-height: 45vh;
  }

  .app-title {
    font-size: 1rem;
  }

  .start-buttons {
    flex-direction: column;
  }
}

/* Eval mode styles */
.eval-info .panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.eval-text {
  font-size: 0.85rem;
  color: var(--color-text-muted);
  margin: var(--spacing-xs) 0;
}

.eval-status {
  font-size: 0.9rem;
  color: var(--color-accent);
  font-weight: 600;
  margin: var(--spacing-sm) 0;
  padding: var(--spacing-xs) var(--spacing-sm);
  background: rgba(255, 107, 157, 0.1);
  border-radius: var(--radius-sm);
  text-align: center;
}

.btn-icon-eval {
  background: var(--color-bg-light);
  border: 1px solid var(--color-border);
  color: var(--color-text);
  width: 32px;
  height: 32px;
  border-radius: var(--radius-sm);
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}

.btn-icon-eval:hover {
  background: var(--color-bg-mid);
  border-color: var(--color-primary);
}

.btn-icon-eval.active {
  background: var(--color-primary);
  border-color: var(--color-primary);
  color: var(--color-bg-dark);
}
</style>
