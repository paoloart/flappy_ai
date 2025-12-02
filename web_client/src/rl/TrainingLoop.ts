/**
 * Training Loop - orchestrates RL training with the game engine
 * Uses WorkerDQNAgent for off-main-thread training
 */

import { GameEngine, type RewardConfig, type StepResult } from '@/game'
import { WorkerDQNAgent, type DQNConfig } from './WorkerDQNAgent'
import type { TrainingMetrics } from './types'

export interface TrainingCallbacks {
  onStep?: (metrics: TrainingMetrics, qValues: number[]) => void
  onEpisodeEnd?: (metrics: TrainingMetrics) => void
  onTrainingStart?: () => void
  onTrainingStop?: () => void
}

// Must match WARMUP_SIZE in training.worker.ts
const WARMUP_SIZE = 50000

export class TrainingLoop {
  private engine: GameEngine
  private agent: WorkerDQNAgent | null = null
  private callbacks: TrainingCallbacks
  private createAgent: () => WorkerDQNAgent
  private initialized: boolean = false

  // Worker fast-mode state
  private fastModeActive: boolean = false
  private lastWorkerMetrics: TrainingMetrics | null = null

  // Training state
  private isRunning: boolean = false
  private speedFactor: number = 1.0

  // Current episode state
  private currentState: number[] = []
  private episodeReward: number = 0
  private episodeLength: number = 0
  private episode: number = 0

  // Metrics tracking
  private recentRewards: number[] = []
  private recentLengths: number[] = []
  private readonly metricsWindow = 50
  private lastStepTime: number = 0
  private stepsSinceLastMetric: number = 0
  private stepsPerSecond: number = 0

  constructor(
    engine: GameEngine,
    agentConfig: Partial<DQNConfig> = {},
    callbacks: TrainingCallbacks = {},
    agentFactory: () => WorkerDQNAgent = () => new WorkerDQNAgent(agentConfig)
  ) {
    this.engine = engine
    this.callbacks = callbacks
    this.createAgent = agentFactory
  }

  /**
   * Initialize the agent
   */
  init(): void {
    if (this.initialized) return

    console.log('[TrainingLoop] Initializing with Web Worker...')

    // Create the worker-based agent
    this.agent = this.createAgent()

    // Listen for worker-driven metrics (fast mode)
    this.agent.onFastMetrics((metrics) => {
      this.lastWorkerMetrics = metrics
      this.callbacks.onStep?.(metrics, this.agent?.getLastQValues() ?? [0, 0])
    })

    this.initialized = true
    console.log('[TrainingLoop] Initialized!')
  }

  /**
   * Start training
   */
  start(): void {
    if (this.isRunning) return

    // Ensure agent is initialized
    this.init()

    this.isRunning = true
    this.currentState = this.engine.reset()
    this.episodeReward = 0
    this.episodeLength = 0
    this.lastStepTime = performance.now()

    this.callbacks.onTrainingStart?.()
  }

  /**
   * Stop training
   */
  stop(): void {
    this.isRunning = false
    if (this.fastModeActive) {
      this.setFastMode(false)
    }
    this.callbacks.onTrainingStop?.()
  }

  /**
   * Execute one training step
   * Returns true if the episode ended
   */
  step(): { result: StepResult; episodeEnded: boolean; finalReward?: number; finalLength?: number } | null {
    if (!this.isRunning || !this.agent) return null

    if (this.fastModeActive) {
      return null
    }

    // Agent selects action (fast - runs on main thread)
    const action = this.agent.act(this.currentState, true) as 0 | 1

    // Step the environment
    const result = this.engine.step(action)

    // Store transition (sent to worker for training - non-blocking)
    this.agent.remember({
      state: this.currentState,
      action,
      reward: result.reward,
      nextState: result.observation,
      done: result.done,
    })

    // Update episode stats
    this.episodeReward += result.reward
    this.episodeLength++
    this.stepsSinceLastMetric++

    // Train the agent only in fallback mode (worker handles its own training)
    // This is expensive so only call when necessary
    if (!this.agent.isUsingWorker() && this.agent.getSteps() % 32 === 0) {
      this.agent.replay()
    }

    // Calculate steps per second
    const now = performance.now()
    const elapsed = now - this.lastStepTime
    if (elapsed >= 1000) {
      this.stepsPerSecond = (this.stepsSinceLastMetric / elapsed) * 1000
      this.stepsSinceLastMetric = 0
      this.lastStepTime = now
    }

    // Get current Q-values for visualization
    const qValues = this.agent.getLastQValues()

    // Emit step callback
    this.callbacks.onStep?.(this.getMetrics(), qValues)

    // Handle episode end
    if (result.done) {
      this.episode++

      // Capture final episode reward BEFORE resetting
      const finalEpisodeReward = this.episodeReward
      const finalEpisodeLength = this.episodeLength

      // Update metrics
      this.recentRewards.push(finalEpisodeReward)
      this.recentLengths.push(finalEpisodeLength)
      if (this.recentRewards.length > this.metricsWindow) {
        this.recentRewards.shift()
        this.recentLengths.shift()
      }

      this.callbacks.onEpisodeEnd?.(this.getMetrics())

      // Reset for next episode
      this.currentState = this.engine.reset()
      this.episodeReward = 0
      this.episodeLength = 0

      // Return with the FINAL episode reward (not the reset value)
      return { 
        result, 
        episodeEnded: true, 
        finalReward: finalEpisodeReward,
        finalLength: finalEpisodeLength 
      }
    }

    // Continue episode
    this.currentState = result.observation
    return { result, episodeEnded: false }
  }

  /**
   * Run multiple steps (for faster training)
   */
  runSteps(count: number): void {
    for (let i = 0; i < count && this.isRunning; i++) {
      this.step()
    }
  }

  /**
   * Get current training metrics
   */
  getMetrics(): TrainingMetrics {
    if (this.fastModeActive && this.lastWorkerMetrics) {
      // Worker already has the correct episode count (started with main thread's count)
      return this.lastWorkerMetrics
    }

    // In fast mode but no worker metrics yet - return placeholder with warmup=true
    // until we get actual metrics from the worker
    if (this.fastModeActive) {
      return {
        episode: this.episode,
        episodeReward: 0,
        episodeLength: 0,
        avgReward: 0,
        avgLength: 0,
        epsilon: this.agent?.getEpsilon() ?? 1.0,
        loss: 0,
        bufferSize: 0,
        stepsPerSecond: 0,
        totalSteps: this.agent?.getSteps() ?? 0,
        isWarmup: true, // Assume warmup until worker tells us otherwise
        learningRate: this.agent?.getLearningRate() ?? 0.0005,
      }
    }

    const avgReward =
      this.recentRewards.length > 0
        ? this.recentRewards.reduce((a, b) => a + b, 0) /
          this.recentRewards.length
        : 0

    const avgLength =
      this.recentLengths.length > 0
        ? this.recentLengths.reduce((a, b) => a + b, 0) /
          this.recentLengths.length
        : 0

    const bufferSize = this.agent?.getBufferSize() ?? 0
    return {
      episode: this.episode,
      episodeReward: this.episodeReward,
      episodeLength: this.episodeLength,
      avgReward,
      avgLength,
      epsilon: this.agent?.getEpsilon() ?? 1.0,
      loss: this.agent?.getLastLoss() ?? 0,
      bufferSize,
      stepsPerSecond: this.stepsPerSecond,
      totalSteps: this.agent?.getSteps() ?? 0,
      isWarmup: bufferSize < WARMUP_SIZE,
      learningRate: this.agent?.getLearningRate() ?? 0.0005,
    }
  }

  setFastMode(enabled: boolean): void {
    if (!this.agent) return

    if (enabled && !this.fastModeActive) {
      if (!this.agent.isUsingWorker()) {
        console.warn('[TrainingLoop] Fast mode requires worker support; falling back to main thread')
        return
      }
      this.agent.syncWeightsToWorker()
      this.fastModeActive = true
      // Pass current episode count and total steps to worker so it continues from where we left off
      this.agent.startFastTraining(this.episode, this.agent.getSteps())
    } else if (!enabled && this.fastModeActive) {
      this.fastModeActive = false
      this.agent.stopFastTraining()
      // Pull the freshest weights trained in fast mode back to the main thread so
      // subsequent on-thread inference uses the same parameters the worker just
      // optimized.
      if (this.agent.isUsingWorker()) {
        this.agent.requestWeights()
      }
      
      // Preserve metrics from fast mode before clearing
      if (this.lastWorkerMetrics) {
        // Worker already started with our episode count, so just use its final count
        this.episode = this.lastWorkerMetrics.episode
        // Sync epsilon from worker so it continues from where fast mode left off
        this.agent.syncEpsilonFromWorker(this.lastWorkerMetrics.epsilon)
      }
      
      // Only reset current episode stats, not historical data
      this.lastWorkerMetrics = null
      this.episodeReward = 0
      this.episodeLength = 0
      this.stepsPerSecond = 0
      this.currentState = this.engine.reset()
    }
  }

  /**
   * Get current game state for rendering
   */
  getCurrentState(): number[] {
    return this.currentState
  }

  /**
   * Get network visualization data (simplified - no weights)
   */
  getNetworkVisualization(): {
    activations: number[][]
    qValues: number[]
    selectedAction: number
  } {
    // Skip network visualization entirely during worker-driven fast mode to avoid
    // pulling inference resources on the main thread.
    if (this.fastModeActive) {
      return {
        activations: [],
        qValues: [0, 0],
        selectedAction: 0,
      }
    }

    // Ensure we have a valid state
    const state = this.currentState.length > 0
      ? this.currentState
      : [0, 0, 0, 0, 0, 0] // Default 6-dim state
    
    if (!this.agent) {
      return { 
        activations: [state, [], [], [0, 0]], 
        qValues: [0, 0], 
        selectedAction: 0 
      }
    }
    return this.agent.getNetworkVisualization(state)
  }

  /**
   * Check if training is running
   */
  getIsRunning(): boolean {
    return this.isRunning
  }

  // ===== Hyperparameter setters =====

  setSpeedFactor(factor: number): void {
    this.speedFactor = Math.max(0.25, Math.min(10, factor))
  }

  getSpeedFactor(): number {
    return this.speedFactor
  }

  setEpsilon(value: number): void {
    this.agent?.setEpsilon(value)
  }

  setAutoDecay(enabled: boolean): void {
    this.agent?.setAutoDecay(enabled)
  }

  getEpsilonDecaySteps(): number {
    return this.agent?.getEpsilonDecaySteps() ?? 500000
  }

  setEpsilonDecaySteps(steps: number): void {
    this.agent?.setEpsilonDecaySteps(steps)
  }

  setLearningRate(lr: number): void {
    this.agent?.setLearningRate(lr)
  }

  setLRScheduler(enabled: boolean): void {
    this.agent?.setLRScheduler(enabled)
  }

  setGamma(value: number): void {
    this.agent?.setGamma(value)
  }

  setRewardConfig(config: Partial<RewardConfig>): void {
    this.engine.setRewardConfig(config)
    this.agent?.setRewardConfig(config)
  }

  /**
   * Reset training completely
   */
  reset(): void {
    this.stop()
    this.agent?.reset()
    this.episode = 0
    this.episodeReward = 0
    this.episodeLength = 0
    this.recentRewards = []
    this.recentLengths = []
    this.stepsPerSecond = 0
    this.currentState = this.engine.reset()
  }

  /**
   * Get the agent for saving/loading
   */
  getAgent(): WorkerDQNAgent | null {
    return this.agent
  }

  /**
   * Dispose resources
   */
  dispose(): void {
    this.stop()
    // Terminate the worker
    this.agent?.terminate()
    this.agent = null
    this.initialized = false
  }
}
