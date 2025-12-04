<template>
  <div class="game-canvas-container" ref="container" @touchstart.passive="onTouchStart">
    <canvas ref="canvas" class="game-canvas"></canvas>
    
    <!-- Touch indicator for mobile -->
    <div v-if="showTouchHint && isMobile" class="touch-hint">
      <span class="touch-icon">👆</span>
      <span class="touch-text">Tap to flap!</span>
    </div>
    
    <!-- Fast mode overlay - game not rendered -->
    <div v-if="mode === 'training' && fastMode" class="fast-mode-overlay">
      <div class="fast-mode-content">
        <div class="fast-header">
          <div class="fast-icon">⚡</div>
          <div class="fast-title">FAST TRAINING</div>
          <div class="fast-subtitle">Game visualization disabled</div>
        </div>
        <img src="/dqn-workflow.jpg" alt="DQN Workflow" class="dqn-workflow-img" />
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue'
import {
  GameEngine,
  Renderer,
  InputController,
  GameConfig,
  type GameAction,
  type RawGameState,
} from '@/game'
import { TrainingLoop } from '@/rl'

export default defineComponent({
  name: 'GameCanvas',
  props: {
    mode: {
      type: String as () => 'idle' | 'configuring' | 'training' | 'eval' | 'manual',
      default: 'idle',
    },
    fastMode: {
      type: Boolean,
      default: false,
    },
    isPaused: {
      type: Boolean,
      default: false,
    },
    hiddenLayersConfig: {
      type: Array as () => number[],
      default: () => [64, 64],
    },
  },
  emits: ['score-update', 'episode-end', 'state-update', 'metrics-update', 'network-update', 'auto-eval-result', 'architecture-loaded'],
  data() {
    return {
      engine: null as GameEngine | null,
      renderer: null as Renderer | null,
      inputController: null as InputController | null,
      trainingLoop: null as TrainingLoop | null,
      animationId: null as number | null,
      trainingTimeoutId: null as number | null,
      isRunning: false,
      internalPaused: false,
      gameOver: false,
      pendingAction: 0 as GameAction,
      lastFrameTime: 0,
      lastScore: 0,
      lastMetricsTime: 0,
      showTouchHint: true,
      isMobile: false,
      // For eval-mode visualization: last observation/Q-values/action actually used
      lastEvalObservation: [] as number[],
      lastEvalQValues: [0, 0] as [number, number],
      lastEvalAction: 0 as 0 | 1,
      // For reward display during training
      lastReward: 0 as number,
    }
  },
  computed: {
    // computed properties
  },
  watch: {
    isPaused(newVal: boolean) {
      this.setPaused(newVal)
    },
  },
  async mounted() {
    // Detect mobile device
    this.isMobile = 'ontouchstart' in window || navigator.maxTouchPoints > 0
    await this.initGame()
  },
  beforeUnmount() {
    this.stopGame()
    if (this.inputController) {
      this.inputController.disable()
    }
    if (this.trainingLoop) {
      this.trainingLoop.dispose()
    }
  },
  methods: {
    async initGame() {
      const canvas = this.$refs.canvas as HTMLCanvasElement
      if (!canvas) return

      // Initialize renderer
      this.renderer = new Renderer(canvas)
      await this.renderer.loadSprites()

      // Initialize game engine
      this.engine = new GameEngine()

      // Initialize input controller
      this.inputController = new InputController()

      // Reset to show initial state
      this.engine.reset()
      this.renderFrame()
    },

    startGame() {
      if (!this.engine || !this.renderer) return

      // Stop fast training if it was active (worker runs independently)
      if (this.trainingLoop) {
        this.trainingLoop.setFastMode(false)
      }

      this.engine.reset()
      this.renderer.resetFloor()
      this.pendingAction = 0
      this.isRunning = true
      this.internalPaused = false
      this.gameOver = false
      this.lastScore = 0

      // Enable input for manual/training play
      const container = this.$refs.container as HTMLElement
      this.inputController?.enable(container, this.handleInput)

      this.lastFrameTime = performance.now()
      // Use $nextTick to ensure mode prop has updated before starting the loop
      this.$nextTick(() => this.gameLoop())
    },

    startTraining(hiddenLayers: number[] = [64, 64]) {
      if (!this.engine || !this.renderer) return

      // Initialize training loop if not already (or reinitialize if architecture changed)
      if (!this.trainingLoop && this.engine) {
        this.trainingLoop = new TrainingLoop(this.engine as GameEngine, {
          inputDim: 6,
          hiddenLayers,
          actionDim: 2,
        }, {
          onStep: (metrics) => {
            this.$emit('metrics-update', metrics)
          },
          onAutoEvalResult: (result) => {
            this.$emit('auto-eval-result', result)
          },
        })
      }

      this.renderer.resetFloor()
      this.isRunning = true
      this.internalPaused = false
      this.gameOver = false
      this.lastScore = 0

      // Start training (synchronous - pure JS neural network)
      if (this.trainingLoop) {
        this.trainingLoop.start()
      }

      this.lastFrameTime = performance.now()
      this.lastMetricsTime = performance.now()
      // Use $nextTick to ensure mode prop has updated before starting the loop
      this.$nextTick(() => this.runTrainingLoop())
    },

    runTrainingLoop() {
      // Guard: only run if we're in training mode (prevents race with other loops)
      if (this.mode !== 'training') return
      if (!this.isRunning || !this.trainingLoop || !this.renderer || !this.engine) return
      
      if (this.internalPaused) {
        this.animationId = requestAnimationFrame(() => this.runTrainingLoop())
        return
      }

      const now = performance.now()
      const elapsed = now - this.lastFrameTime

      if (this.fastMode) {
        this.trainingLoop.setFastMode(true)

        // Periodically refresh metrics from the worker-driven loop
        if (now - this.lastMetricsTime >= 500) {
          const metrics = this.trainingLoop.getMetrics()
          this.$emit('metrics-update', metrics)
          
          // Emit network visualization with REAL activations
          const viz = this.trainingLoop.getNetworkVisualization()
          this.$emit('network-update', viz)
          
          this.lastMetricsTime = now
        }

        this.animationId = requestAnimationFrame(() => this.runTrainingLoop())
        return
      }

      // Ensure worker fast mode is disabled when returning to normal speed
      this.trainingLoop.setFastMode(false)

      // Normal mode: run at game speed (30fps)
      if (elapsed >= GameConfig.FRAME_TIME) {
        this.lastFrameTime = now - (elapsed % GameConfig.FRAME_TIME)

        const stepResult = this.trainingLoop.step()
        
        // Capture the reward for display
        if (stepResult?.result) {
          this.lastReward = stepResult.result.reward
        }
        
        if (stepResult?.episodeEnded) {
          this.$emit('episode-end', {
            score: stepResult.result.info.score,
            reward: stepResult.finalReward,  // Use the captured final reward
            length: stepResult.finalLength,  // Include episode length for charting
          })
          this.lastScore = 0
        }

        // Render every frame in normal mode (with reward indicator)
        const gameState = this.engine.getState()
        const cumulativeReward = this.trainingLoop.getMetrics().episodeReward
        this.renderer.render(gameState as RawGameState, false, this.lastReward, cumulativeReward)

        // Emit network visualization immediately after step (in sync with action)
        const viz = this.trainingLoop.getNetworkVisualization()
        this.$emit('network-update', viz)
      }

      // Update metrics at game FPS
      if (now - this.lastMetricsTime >= GameConfig.FRAME_TIME) {
        const metrics = this.trainingLoop.getMetrics()
        this.$emit('metrics-update', metrics)
        this.lastMetricsTime = now
      }

      this.animationId = requestAnimationFrame(() => this.runTrainingLoop())
    },

    stopGame() {
      this.isRunning = false
      this.internalPaused = false
      this.gameOver = false
      if (this.animationId !== null) {
        cancelAnimationFrame(this.animationId)
        this.animationId = null
      }
      if (this.trainingTimeoutId !== null) {
        clearTimeout(this.trainingTimeoutId)
        this.trainingTimeoutId = null
      }
      this.inputController?.disable()
    },

    handleInput(action: GameAction) {
      this.pendingAction = action
    },

    onTouchStart() {
      // Hide touch hint after first touch
      this.showTouchHint = false
    },

    gameLoop() {
      // Guard: only run if we're in manual mode (prevents race with other loops)
      if (this.mode !== 'manual') return
      if (!this.isRunning || !this.engine || !this.renderer) return

      const now = performance.now()
      const elapsed = now - this.lastFrameTime

      // Run at game FPS
      if (elapsed >= GameConfig.FRAME_TIME) {
        this.lastFrameTime = now - (elapsed % GameConfig.FRAME_TIME)

        // Don't step if paused or game already over
        if (!this.internalPaused && !this.gameOver) {
          // Step the game
          const action = this.pendingAction
          this.pendingAction = 0 // Reset pending action

          const result = this.engine.step(action)

          // Emit score updates
          if (result.info.score > this.lastScore) {
            this.lastScore = result.info.score
            this.$emit('score-update', result.info.score)
          }

          // Handle game over (only emit once)
          if (result.done && !this.gameOver) {
            this.gameOver = true
            this.$emit('episode-end', {
              score: result.info.score,
              reward: result.reward,
            })
            // Manual mode: game stays over until user restarts
          }

          // Emit state for NN visualization
          this.$emit('state-update', {
            observation: result.observation,
            state: this.engine.getState(),
          })
        }
      }

      // Render
      this.renderFrame()

      // Continue loop
      this.animationId = requestAnimationFrame(() => this.gameLoop())
    },

    renderFrame() {
      if (!this.engine || !this.renderer) return

      const state = this.engine.getState()
      const showMessage = this.mode === 'idle' && !this.isRunning

      this.renderer.render(state as RawGameState, showMessage)
    },

    /**
     * External method to set agent action (for RL)
     */
    setAction(action: GameAction) {
      this.pendingAction = action
    },

    /**
     * Get current observation (for RL)
     */
    getObservation(): number[] {
      return this.engine?.getObservation() || []
    },

    // ===== Training hyperparameter controls =====

    setEpsilon(value: number) {
      this.trainingLoop?.setEpsilon(value)
    },

    setAutoDecay(enabled: boolean) {
      this.trainingLoop?.setAutoDecay(enabled)
    },

    setEpsilonDecaySteps(steps: number) {
      this.trainingLoop?.setEpsilonDecaySteps(steps)
    },

    /**
     * Download current network weights as a JSON checkpoint
     */
    saveCheckpointToFile() {
      if (!this.trainingLoop) return
      const agent = this.trainingLoop.getAgent()
      if (!agent) return

      const payload = {
        type: 'flappy-ai-checkpoint-v2',
        createdAt: new Date().toISOString(),
        architecture: {
          hiddenLayers: [...this.hiddenLayersConfig],
        },
        info: {
          epsilon: agent.getEpsilon ? agent.getEpsilon() : undefined,
          epsilonDecaySteps: this.trainingLoop.getEpsilonDecaySteps
            ? this.trainingLoop.getEpsilonDecaySteps()
            : undefined,
        },
        network: agent.save(),
      }

      const json = JSON.stringify(payload, null, 2)
      const blob = new Blob([json], { type: 'application/json' })
      const url = URL.createObjectURL(blob)

      const a = document.createElement('a')
      a.href = url
      a.download = `flappy-ai-checkpoint-${new Date()
        .toISOString()
        .replace(/[:.]/g, '-')}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    },

    /**
     * Load network weights from a JSON string (uploaded file)
     */
    loadCheckpointFromJSON(json: string) {
      let payload: any
      try {
        payload = JSON.parse(json)
      } catch (e) {
        console.warn('[GameCanvas] Invalid checkpoint JSON', e)
        return
      }

      // Support both v1 and v2 checkpoint formats
      const isV1 = payload?.type === 'flappy-ai-checkpoint-v1'
      const isV2 = payload?.type === 'flappy-ai-checkpoint-v2'
      if (!payload || (!isV1 && !isV2) || !payload.network) {
        console.warn('[GameCanvas] Invalid checkpoint format')
        return
      }

      // Extract architecture (default [64, 64] for v1 checkpoints)
      const hiddenLayers = payload.architecture?.hiddenLayers || [64, 64]
      
      // Store the network data and info temporarily
      const networkData = payload.network
      const checkpointInfo = payload.info

      // Dispose of existing training loop
      if (this.trainingLoop) {
        this.trainingLoop.dispose()
        this.trainingLoop = null
      }
      if (this.animationId) {
        cancelAnimationFrame(this.animationId)
        this.animationId = null
      }
      this.isRunning = false

      // Emit architecture-loaded so App.vue updates hiddenLayersConfig and enters configuring mode
      this.$emit('architecture-loaded', hiddenLayers)

      // Create a new training loop with the loaded architecture
      if (this.engine) {
        this.trainingLoop = new TrainingLoop(this.engine as GameEngine, {
          inputDim: 6,
          hiddenLayers,
          actionDim: 2,
        }, {
          onStep: (metrics) => {
            this.$emit('metrics-update', metrics)
          },
          onAutoEvalResult: (result) => {
            this.$emit('auto-eval-result', result)
          },
        })

        // Load the network weights
        const agent = this.trainingLoop.getAgent()
        if (agent) {
          try {
            agent.load(networkData)
            
            // Restore epsilon if present
            if (checkpointInfo && typeof checkpointInfo.epsilon === 'number') {
              this.trainingLoop.setEpsilon(checkpointInfo.epsilon)
            }
            
            console.log('[GameCanvas] Checkpoint loaded with architecture:', hiddenLayers)
          } catch (e) {
            console.warn('[GameCanvas] Failed to load network weights', e)
          }
        }
      }
    },

    setLearningRate(lr: number) {
      this.trainingLoop?.setLearningRate(lr)
    },

    setLRScheduler(enabled: boolean) {
      this.trainingLoop?.setLRScheduler(enabled)
    },

    setTrainFreq(value: number) {
      this.trainingLoop?.setTrainFreq(value)
    },

    setRewardConfig(config: Partial<{ passPipe: number; deathPenalty: number; stepPenalty: number; centerReward: number; flapCost: number }>) {
      this.trainingLoop?.setRewardConfig(config)
    },

    resetTraining() {
      // Stop and dispose of the training loop
      // User will need to click "Start Training" to begin again (with potentially new architecture)
      if (this.trainingLoop) {
        // Stop fast mode if active
        if (this.fastMode) {
          this.trainingLoop.setFastMode(false)
        }
        // Dispose of the training loop - it will be recreated with new config
        this.trainingLoop.dispose()
        this.trainingLoop = null
      }
      // Cancel animation loop
      if (this.animationId) {
        cancelAnimationFrame(this.animationId)
        this.animationId = null
      }
      this.isRunning = false
    },

    setPaused(paused: boolean) {
      if (paused) {
        this.internalPaused = true
        if (this.animationId !== null) {
          cancelAnimationFrame(this.animationId)
          this.animationId = null
        }
        if (this.trainingTimeoutId !== null) {
          clearTimeout(this.trainingTimeoutId)
          this.trainingTimeoutId = null
        }
        // Stop fast training in worker when pausing
        if (this.fastMode && this.trainingLoop) {
          this.trainingLoop.setFastMode(false)
        }
      } else {
        this.internalPaused = false
        if (this.isRunning) {
          this.lastFrameTime = performance.now()
          this.lastMetricsTime = performance.now()
          // Resume fast training if it was active
          if (this.fastMode && this.trainingLoop) {
            this.trainingLoop.setFastMode(true)
          }
          if (this.trainingLoop) {
            this.runTrainingLoop()
          } else {
            this.gameLoop()
          }
        }
      }
    },

    startEval() {
      if (!this.engine || !this.renderer) return

      // Stop fast training if it was active (worker runs independently)
      if (this.trainingLoop) {
        this.trainingLoop.setFastMode(false)
      }

      // Initialize training loop if not already (we need the agent for eval)
      if (!this.trainingLoop && this.engine) {
        this.trainingLoop = new TrainingLoop(this.engine as GameEngine, {
          inputDim: 6,
          hiddenLayers: [64, 64],
          actionDim: 2,
        }, {
          onStep: (metrics) => {
            this.$emit('metrics-update', metrics)
          },
        })
        this.trainingLoop.start()
      }

      this.engine.reset()
      this.renderer.resetFloor()
      this.isRunning = true
      this.internalPaused = false
      this.gameOver = false
      this.lastScore = 0

      this.lastFrameTime = performance.now()
      this.lastMetricsTime = performance.now()
      // Use $nextTick to ensure mode prop has updated before starting the loop
      this.$nextTick(() => this.runEvalLoop())
    },

    runEvalLoop() {
      // Guard: only run if we're in eval mode (prevents race with other loops)
      if (this.mode !== 'eval') return
      if (!this.isRunning || !this.trainingLoop || !this.renderer || !this.engine) return
      if (this.internalPaused) {
        this.animationId = requestAnimationFrame(() => this.runEvalLoop())
        return
      }

      const now = performance.now()

      // Throttle to ~30fps for eval (game speed)
      if (now - this.lastFrameTime < 33) {
        this.animationId = requestAnimationFrame(() => this.runEvalLoop())
        return
      }
      this.lastFrameTime = now

      // Run one game step
      const agent = this.trainingLoop.getAgent()
      if (agent) {
        const observation = this.engine.getObservation()
        const action = agent.act(observation, false) as 0 | 1

        // Store for visualization (exact inputs/outputs used this step)
        this.lastEvalObservation = observation
        const qVals = agent.getLastQValues()
        this.lastEvalQValues = [qVals[0] ?? 0, qVals[1] ?? 0]
        this.lastEvalAction = action

        const result = this.engine.step(action)

        if (result.info.score > this.lastScore) {
          this.lastScore = result.info.score
          this.$emit('score-update', result.info.score)
        }

        // Emit network visualization for this step (before potential reset)
        const viz = agent.getNetworkVisualization(observation)
        this.$emit('network-update', viz)

        if (result.done) {
          this.$emit('episode-end', {
            score: result.info.score,
            reward: 0,
          })
          // Auto-restart eval
          this.engine.reset()
          this.renderer?.resetFloor()
          this.lastScore = 0
        }
      }

      // Render
      const gameState = this.engine.getState()
      this.renderer.render(gameState as RawGameState, false)

      this.animationId = requestAnimationFrame(() => this.runEvalLoop())
    },
  },
})
</script>

<style scoped>
.game-canvas-container {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
  position: relative;
  touch-action: none;
}

.game-canvas {
  /* Scale canvas to be larger while keeping aspect ratio */
  width: 432px;  /* 1.5x original 288px */
  height: 768px; /* 1.5x original 512px */
  max-width: 100%;
  max-height: 100%;
  border-radius: var(--radius-lg);
  box-shadow: 0 0 40px rgba(0, 217, 255, 0.15);
  image-rendering: pixelated;
  image-rendering: crisp-edges;
  cursor: pointer;
  /* Ensure proper scaling */
  object-fit: contain;
}

.touch-hint {
  position: absolute;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  animation: bounce 1s ease-in-out infinite;
  pointer-events: none;
}

.touch-icon {
  font-size: 2rem;
}

.touch-text {
  font-family: var(--font-display);
  font-size: 0.6rem;
  color: var(--color-text);
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
  background: rgba(0, 0, 0, 0.5);
  padding: 4px 8px;
  border-radius: var(--radius-sm);
}

@keyframes bounce {
  0%, 100% { transform: translateX(-50%) translateY(0); }
  50% { transform: translateX(-50%) translateY(-10px); }
}

.fast-mode-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(135deg, #0a1628 0%, #1a2a4a 100%);
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding-top: 2rem;
  pointer-events: none;
  overflow-y: auto;
}

.fast-mode-content {
  text-align: center;
  max-width: 90%;
}

.fast-header {
  margin-bottom: 1.5rem;
}

.fast-icon {
  font-size: 3rem;
  animation: pulse 1s ease-in-out infinite;
}

.fast-title {
  font-family: var(--font-display);
  font-size: 1.5rem;
  color: var(--color-accent);
  letter-spacing: 0.2em;
  margin-top: 0.5rem;
}

.fast-subtitle {
  font-size: 0.8rem;
  color: var(--color-text-muted);
  margin-top: 0.25rem;
}

.dqn-workflow-img {
  max-width: 100%;
  max-height: 60vh;
  width: auto;
  height: auto;
  border-radius: 8px;
  opacity: 0.95;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.7; transform: scale(1.1); }
}
</style>

